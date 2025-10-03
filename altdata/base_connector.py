"""
Base Alternative Data Connector
Abstract base class for all alternative data source connectors
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "EXCELLENT"  # >99% complete, <1% errors
    GOOD = "GOOD"           # >95% complete, <5% errors
    FAIR = "FAIR"           # >90% complete, <10% errors
    POOR = "POOR"           # <90% complete or >10% errors


@dataclass
class DataQualityMetrics:
    """Data quality metrics for monitoring"""
    completeness: float  # % of expected data points received
    accuracy: float      # % of data points within expected ranges
    timeliness: float    # % of data received within SLA
    consistency: float   # % of data consistent with historical patterns
    overall_quality: DataQuality
    timestamp: datetime


@dataclass
class AltDataPoint:
    """Single alternative data point"""
    source: str
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    quality_score: Optional[float] = None


# Prometheus metrics
ALTDATA_REQUESTS = Counter(
    'altdata_requests_total',
    'Total alternative data requests',
    ['source', 'status']
)

ALTDATA_LATENCY = Histogram(
    'altdata_request_latency_seconds',
    'Alternative data request latency',
    ['source'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

ALTDATA_QUALITY = Gauge(
    'altdata_quality_score',
    'Alternative data quality score',
    ['source', 'metric']
)

ALTDATA_ERRORS = Counter(
    'altdata_errors_total',
    'Total alternative data errors',
    ['source', 'error_type']
)


class BaseAltDataConnector(ABC):
    """
    Abstract base class for alternative data connectors

    All alternative data sources must implement this interface
    to ensure consistent integration with the platform.
    """

    def __init__(
        self,
        source_name: str,
        api_key: Optional[str] = None,
        rate_limit_per_minute: int = 60,
        timeout_seconds: int = 30
    ):
        self.source_name = source_name
        self.api_key = api_key
        self.rate_limit_per_minute = rate_limit_per_minute
        self.timeout_seconds = timeout_seconds

        # Rate limiting
        self._request_timestamps: List[float] = []

        # Quality tracking
        self._quality_metrics: List[DataQualityMetrics] = []

        logger.info(f"Initialized {source_name} connector")

    @abstractmethod
    def fetch_latest(self, symbol: str) -> Optional[AltDataPoint]:
        """
        Fetch latest data point for a symbol

        Args:
            symbol: Stock symbol or data series ID

        Returns:
            Latest data point or None if unavailable
        """
        pass

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[AltDataPoint]:
        """
        Fetch historical data for a symbol

        Args:
            symbol: Stock symbol or data series ID
            start_date: Start date
            end_date: End date

        Returns:
            List of historical data points
        """
        pass

    @abstractmethod
    def validate_data(self, data_point: AltDataPoint) -> bool:
        """
        Validate a data point for quality and correctness

        Args:
            data_point: Data point to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols/series available from this data source

        Returns:
            List of available symbols
        """
        pass

    def check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits

        Returns:
            True if we can make a request, False otherwise
        """
        now = time.time()
        cutoff = now - 60  # Last minute

        # Remove old timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > cutoff
        ]

        # Check limit
        if len(self._request_timestamps) >= self.rate_limit_per_minute:
            logger.warning(
                f"Rate limit exceeded for {self.source_name}: "
                f"{len(self._request_timestamps)}/{self.rate_limit_per_minute}"
            )
            return False

        return True

    def record_request(self):
        """Record a request for rate limiting"""
        self._request_timestamps.append(time.time())

    def calculate_quality_metrics(
        self,
        data_points: List[AltDataPoint],
        expected_count: Optional[int] = None
    ) -> DataQualityMetrics:
        """
        Calculate quality metrics for a batch of data points

        Args:
            data_points: List of data points
            expected_count: Expected number of data points (for completeness)

        Returns:
            DataQualityMetrics
        """
        if not data_points:
            return DataQualityMetrics(
                completeness=0.0,
                accuracy=0.0,
                timeliness=0.0,
                consistency=0.0,
                overall_quality=DataQuality.POOR,
                timestamp=datetime.utcnow()
            )

        # Completeness: actual / expected
        if expected_count:
            completeness = len(data_points) / expected_count
        else:
            completeness = 1.0  # Unknown expected count

        # Accuracy: % of valid data points
        valid_count = sum(1 for dp in data_points if self.validate_data(dp))
        accuracy = valid_count / len(data_points)

        # Timeliness: % of data points received within 24 hours
        now = datetime.utcnow()
        timely_count = sum(
            1 for dp in data_points
            if (now - dp.timestamp).total_seconds() < 86400
        )
        timeliness = timely_count / len(data_points)

        # Consistency: % of data points with quality scores > 0.7
        scored_points = [dp for dp in data_points if dp.quality_score is not None]
        if scored_points:
            consistent_count = sum(
                1 for dp in scored_points if dp.quality_score > 0.7
            )
            consistency = consistent_count / len(scored_points)
        else:
            consistency = 1.0  # No quality scores available

        # Overall quality
        avg_score = (completeness + accuracy + timeliness + consistency) / 4

        if avg_score >= 0.99:
            overall = DataQuality.EXCELLENT
        elif avg_score >= 0.95:
            overall = DataQuality.GOOD
        elif avg_score >= 0.90:
            overall = DataQuality.FAIR
        else:
            overall = DataQuality.POOR

        metrics = DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            timeliness=timeliness,
            consistency=consistency,
            overall_quality=overall,
            timestamp=datetime.utcnow()
        )

        # Record metrics
        self._quality_metrics.append(metrics)

        # Update Prometheus metrics
        ALTDATA_QUALITY.labels(source=self.source_name, metric='completeness').set(completeness)
        ALTDATA_QUALITY.labels(source=self.source_name, metric='accuracy').set(accuracy)
        ALTDATA_QUALITY.labels(source=self.source_name, metric='timeliness').set(timeliness)
        ALTDATA_QUALITY.labels(source=self.source_name, metric='consistency').set(consistency)

        return metrics

    def get_recent_quality_metrics(self, hours: int = 24) -> List[DataQualityMetrics]:
        """Get quality metrics from the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            m for m in self._quality_metrics
            if m.timestamp > cutoff
        ]

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the data source

        Returns:
            Health status dictionary
        """
        try:
            # Try to get available symbols (lightweight check)
            symbols = self.get_available_symbols()

            # Get recent quality metrics
            recent_metrics = self.get_recent_quality_metrics(hours=1)

            if recent_metrics:
                latest_quality = recent_metrics[-1]
                quality_status = latest_quality.overall_quality.value
            else:
                quality_status = "UNKNOWN"

            return {
                "source": self.source_name,
                "status": "healthy",
                "symbols_available": len(symbols),
                "quality_status": quality_status,
                "rate_limit_usage": len(self._request_timestamps),
                "rate_limit_max": self.rate_limit_per_minute,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Health check failed for {self.source_name}: {e}")
            ALTDATA_ERRORS.labels(source=self.source_name, error_type='health_check').inc()

            return {
                "source": self.source_name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def __repr__(self):
        return f"<{self.__class__.__name__} source={self.source_name}>"
