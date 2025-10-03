"""
Alternative Data Monitoring
Quality tracking, alerting, and cost monitoring for alternative data sources
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AltDataAlert:
    """Alternative data alert"""
    source: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]


class AltDataMonitor:
    """
    Alternative Data Monitoring

    Tracks:
    - Data freshness (staleness alerts)
    - Quality degradation
    - Source availability
    - Cost vs value
    """

    def __init__(
        self,
        freshness_threshold_hours: int = 24,
        quality_threshold: float = 0.90,
        cost_tracking_enabled: bool = True
    ):
        self.freshness_threshold_hours = freshness_threshold_hours
        self.quality_threshold = quality_threshold
        self.cost_tracking_enabled = cost_tracking_enabled

        # Tracking
        self._last_fetch_times: Dict[str, datetime] = {}
        self._quality_history: Dict[str, List[float]] = {}
        self._cost_tracking: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[AltDataAlert] = []

        logger.info("Alternative Data Monitor initialized")

    def record_fetch(
        self,
        source: str,
        quality_score: float,
        cost: float = 0.0
    ):
        """
        Record a data fetch

        Args:
            source: Data source name
            quality_score: Quality score (0.0 to 1.0)
            cost: Cost of this fetch (optional)
        """
        now = datetime.utcnow()

        # Update last fetch time
        self._last_fetch_times[source] = now

        # Update quality history
        if source not in self._quality_history:
            self._quality_history[source] = []

        self._quality_history[source].append(quality_score)

        # Keep last 1000 quality scores
        if len(self._quality_history[source]) > 1000:
            self._quality_history[source] = self._quality_history[source][-1000:]

        # Update cost tracking
        if self.cost_tracking_enabled and cost > 0:
            if source not in self._cost_tracking:
                self._cost_tracking[source] = {
                    "total_cost": 0.0,
                    "fetch_count": 0,
                    "first_fetch": now,
                    "last_fetch": now
                }

            self._cost_tracking[source]["total_cost"] += cost
            self._cost_tracking[source]["fetch_count"] += 1
            self._cost_tracking[source]["last_fetch"] = now

    def check_freshness(self, source: str) -> Optional[AltDataAlert]:
        """
        Check if data source is stale

        Args:
            source: Data source name

        Returns:
            Alert if stale, None otherwise
        """
        if source not in self._last_fetch_times:
            return AltDataAlert(
                source=source,
                alert_type="no_data",
                severity=AlertSeverity.WARNING,
                message=f"No data ever fetched from {source}",
                timestamp=datetime.utcnow(),
                metadata={}
            )

        last_fetch = self._last_fetch_times[source]
        hours_since_fetch = (datetime.utcnow() - last_fetch).total_seconds() / 3600

        if hours_since_fetch > self.freshness_threshold_hours:
            return AltDataAlert(
                source=source,
                alert_type="stale_data",
                severity=AlertSeverity.WARNING if hours_since_fetch < 48 else AlertSeverity.CRITICAL,
                message=f"Data from {source} is {hours_since_fetch:.1f} hours old (threshold: {self.freshness_threshold_hours}h)",
                timestamp=datetime.utcnow(),
                metadata={
                    "hours_since_fetch": hours_since_fetch,
                    "last_fetch": last_fetch.isoformat()
                }
            )

        return None

    def check_quality_degradation(self, source: str) -> Optional[AltDataAlert]:
        """
        Check if data quality has degraded

        Args:
            source: Data source name

        Returns:
            Alert if quality degraded, None otherwise
        """
        if source not in self._quality_history:
            return None

        history = self._quality_history[source]

        if len(history) < 10:
            return None  # Not enough history

        # Get recent quality (last 10 fetches)
        recent_quality = sum(history[-10:]) / 10

        if recent_quality < self.quality_threshold:
            return AltDataAlert(
                source=source,
                alert_type="quality_degradation",
                severity=AlertSeverity.WARNING if recent_quality > 0.80 else AlertSeverity.CRITICAL,
                message=f"Quality for {source} degraded to {recent_quality:.1%} (threshold: {self.quality_threshold:.1%})",
                timestamp=datetime.utcnow(),
                metadata={
                    "recent_quality": recent_quality,
                    "threshold": self.quality_threshold,
                    "sample_size": len(history[-10:])
                }
            )

        return None

    def check_all_sources(self, sources: List[str]) -> List[AltDataAlert]:
        """
        Check all sources for issues

        Args:
            sources: List of source names to check

        Returns:
            List of alerts
        """
        alerts = []

        for source in sources:
            # Check freshness
            freshness_alert = self.check_freshness(source)
            if freshness_alert:
                alerts.append(freshness_alert)
                self._alerts.append(freshness_alert)

            # Check quality
            quality_alert = self.check_quality_degradation(source)
            if quality_alert:
                alerts.append(quality_alert)
                self._alerts.append(quality_alert)

        return alerts

    def get_recent_alerts(self, hours: int = 24) -> List[AltDataAlert]:
        """Get alerts from last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self._alerts
            if alert.timestamp > cutoff
        ]

    def get_cost_report(self, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cost tracking report

        Args:
            source: Specific source (or None for all sources)

        Returns:
            Cost report
        """
        if not self.cost_tracking_enabled:
            return {"error": "Cost tracking not enabled"}

        if source:
            if source not in self._cost_tracking:
                return {"error": f"No cost data for {source}"}

            data = self._cost_tracking[source]
            avg_cost = data["total_cost"] / data["fetch_count"] if data["fetch_count"] > 0 else 0

            return {
                "source": source,
                "total_cost": data["total_cost"],
                "fetch_count": data["fetch_count"],
                "avg_cost_per_fetch": avg_cost,
                "first_fetch": data["first_fetch"].isoformat(),
                "last_fetch": data["last_fetch"].isoformat()
            }

        # All sources
        total_cost = sum(data["total_cost"] for data in self._cost_tracking.values())
        total_fetches = sum(data["fetch_count"] for data in self._cost_tracking.values())

        sources_report = []
        for src, data in self._cost_tracking.items():
            avg_cost = data["total_cost"] / data["fetch_count"] if data["fetch_count"] > 0 else 0
            sources_report.append({
                "source": src,
                "total_cost": data["total_cost"],
                "fetch_count": data["fetch_count"],
                "avg_cost_per_fetch": avg_cost,
                "cost_percentage": (data["total_cost"] / total_cost * 100) if total_cost > 0 else 0
            })

        # Sort by cost
        sources_report.sort(key=lambda x: x["total_cost"], reverse=True)

        return {
            "total_cost": total_cost,
            "total_fetches": total_fetches,
            "avg_cost_per_fetch": total_cost / total_fetches if total_fetches > 0 else 0,
            "sources": sources_report
        }

    def get_quality_report(self, source: str) -> Dict[str, Any]:
        """
        Get quality report for a source

        Args:
            source: Data source name

        Returns:
            Quality statistics
        """
        if source not in self._quality_history:
            return {"error": f"No quality data for {source}"}

        history = self._quality_history[source]

        if not history:
            return {"error": "No quality samples"}

        return {
            "source": source,
            "sample_count": len(history),
            "current_quality": history[-1],
            "avg_quality_overall": sum(history) / len(history),
            "avg_quality_recent_10": sum(history[-10:]) / min(10, len(history)),
            "min_quality": min(history),
            "max_quality": max(history),
            "quality_trend": "improving" if history[-1] > sum(history[-10:]) / 10 else "degrading"
        }

    def get_dashboard_data(self, sources: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data

        Args:
            sources: List of sources to include

        Returns:
            Dashboard data
        """
        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "sources": []
        }

        for source in sources:
            source_data = {
                "name": source,
                "status": "unknown"
            }

            # Freshness
            if source in self._last_fetch_times:
                last_fetch = self._last_fetch_times[source]
                hours_since = (datetime.utcnow() - last_fetch).total_seconds() / 3600
                source_data["hours_since_fetch"] = hours_since
                source_data["last_fetch"] = last_fetch.isoformat()

                if hours_since < self.freshness_threshold_hours:
                    source_data["status"] = "healthy"
                elif hours_since < 48:
                    source_data["status"] = "warning"
                else:
                    source_data["status"] = "critical"

            # Quality
            if source in self._quality_history and self._quality_history[source]:
                history = self._quality_history[source]
                recent_quality = sum(history[-10:]) / min(10, len(history))
                source_data["quality_score"] = recent_quality

                if recent_quality < 0.80:
                    source_data["status"] = "critical"
                elif recent_quality < self.quality_threshold:
                    if source_data["status"] == "healthy":
                        source_data["status"] = "warning"

            # Cost
            if source in self._cost_tracking:
                cost_data = self._cost_tracking[source]
                source_data["total_cost"] = cost_data["total_cost"]
                source_data["fetch_count"] = cost_data["fetch_count"]

            dashboard["sources"].append(source_data)

        # Summary
        healthy_count = sum(1 for s in dashboard["sources"] if s["status"] == "healthy")
        warning_count = sum(1 for s in dashboard["sources"] if s["status"] == "warning")
        critical_count = sum(1 for s in dashboard["sources"] if s["status"] == "critical")

        dashboard["summary"] = {
            "total_sources": len(sources),
            "healthy": healthy_count,
            "warning": warning_count,
            "critical": critical_count,
            "overall_status": "critical" if critical_count > 0 else ("warning" if warning_count > 0 else "healthy")
        }

        return dashboard


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    monitor = AltDataMonitor(
        freshness_threshold_hours=24,
        quality_threshold=0.90,
        cost_tracking_enabled=True
    )

    # Simulate data fetches
    print("=== Simulating Data Fetches ===\n")

    sources = ["twitter_sentiment", "reddit_sentiment", "fred_economic", "earnings_surprise"]

    for i in range(15):
        for source in sources:
            quality = 0.95 - (i * 0.02) if source == "twitter_sentiment" else 0.98
            cost = 0.001 if source == "fred_economic" else 0.0

            monitor.record_fetch(source, quality, cost)

    # Check for alerts
    print("=== Checking for Alerts ===\n")
    alerts = monitor.check_all_sources(sources)

    for alert in alerts:
        print(f"[{alert.severity.value.upper()}] {alert.source}: {alert.message}")

    # Quality report
    print("\n=== Quality Report: twitter_sentiment ===\n")
    quality_report = monitor.get_quality_report("twitter_sentiment")
    for key, value in quality_report.items():
        print(f"{key}: {value}")

    # Cost report
    print("\n=== Cost Report ===\n")
    cost_report = monitor.get_cost_report()
    print(f"Total Cost: ${cost_report['total_cost']:.4f}")
    print(f"Total Fetches: {cost_report['total_fetches']}")
    print(f"\nBy Source:")
    for src_report in cost_report['sources']:
        print(f"  {src_report['source']}: ${src_report['total_cost']:.4f} ({src_report['cost_percentage']:.1f}%)")

    # Dashboard
    print("\n=== Dashboard ===\n")
    dashboard = monitor.get_dashboard_data(sources)
    print(f"Overall Status: {dashboard['summary']['overall_status']}")
    print(f"Healthy: {dashboard['summary']['healthy']}")
    print(f"Warning: {dashboard['summary']['warning']}")
    print(f"Critical: {dashboard['summary']['critical']}")
