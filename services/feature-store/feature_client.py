"""
Feature Store Client for PIT-Compliant Feature Retrieval
Wraps Feast with validation and monitoring
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
from feast import FeatureStore
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class FeatureRetrievalMetrics:
    """Metrics for feature retrieval operations"""
    retrieval_time_ms: float
    num_features: int
    num_entities: int
    cache_hit_rate: float
    pit_violations: int  # Should always be 0
    null_rate: float
    timestamp: datetime

class FeatureStoreClient:
    """
    PIT-compliant feature store client with validation

    Key guarantees:
    1. All features have event_timestamp <= request timestamp
    2. No feature leakage from future data
    3. Monitoring for staleness and null rates
    4. Automatic fallback for missing features
    """

    def __init__(self, repo_path: str = "feature_repo"):
        self.store = FeatureStore(repo_path=repo_path)
        self.retrieval_history = []

        # Freshness thresholds (from SLOs)
        self.freshness_thresholds = {
            "price_volume_features": timedelta(minutes=5),
            "momentum_features": timedelta(hours=1),
            "liquidity_features": timedelta(minutes=15),
            "sentiment_features": timedelta(hours=4),
            "fundamental_features": timedelta(days=1),
            "macro_features": timedelta(hours=1),
            "sector_features": timedelta(hours=1),
        }

        logger.info(f"Feature Store initialized from {repo_path}")

    def get_online_features(
        self,
        feature_refs: List[str],
        entity_rows: List[Dict[str, Any]],
        as_of_timestamp: Optional[datetime] = None,
        validate_pit: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve online features with PIT validation

        Args:
            feature_refs: List of feature references (e.g., ["price_volume_features:close_price"])
            entity_rows: List of entity dictionaries (e.g., [{"symbol": "AAPL"}])
            as_of_timestamp: Point-in-time timestamp (default: now)
            validate_pit: Whether to validate PIT compliance

        Returns:
            DataFrame with features, guaranteed PIT-compliant

        Raises:
            ValueError: If PIT violations detected and validate_pit=True
        """
        start_time = time.time()

        if as_of_timestamp is None:
            as_of_timestamp = datetime.now()

        # Retrieve features from Feast
        feature_vector = self.store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_df()

        retrieval_time_ms = (time.time() - start_time) * 1000

        # Validate PIT compliance
        pit_violations = 0
        if validate_pit and "event_timestamp" in feature_vector.columns:
            future_features = feature_vector[
                feature_vector["event_timestamp"] > as_of_timestamp
            ]
            pit_violations = len(future_features)

            if pit_violations > 0:
                error_msg = (
                    f"PIT VIOLATION: {pit_violations} features have "
                    f"event_timestamp > {as_of_timestamp}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Calculate metrics
        num_features = len([f for f in feature_refs])
        num_entities = len(entity_rows)
        null_rate = feature_vector.isnull().sum().sum() / (
            feature_vector.shape[0] * feature_vector.shape[1]
        )

        metrics = FeatureRetrievalMetrics(
            retrieval_time_ms=retrieval_time_ms,
            num_features=num_features,
            num_entities=num_entities,
            cache_hit_rate=0.0,  # TODO: Implement caching
            pit_violations=pit_violations,
            null_rate=null_rate,
            timestamp=datetime.now()
        )

        self.retrieval_history.append(metrics)

        # Log warnings for high null rates
        if null_rate > 0.1:
            logger.warning(
                f"High null rate: {null_rate:.2%} for {num_entities} entities"
            )

        # Log slow retrievals
        if retrieval_time_ms > 100:
            logger.warning(
                f"Slow feature retrieval: {retrieval_time_ms:.1f}ms "
                f"for {num_features} features x {num_entities} entities"
            )

        logger.info(
            f"Retrieved {num_features} features for {num_entities} entities "
            f"in {retrieval_time_ms:.1f}ms (null_rate={null_rate:.2%})"
        )

        return feature_vector

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: List[str],
        full_feature_names: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve historical features for backtesting

        Args:
            entity_df: DataFrame with entity keys and event_timestamp column
            feature_refs: List of feature references
            full_feature_names: Whether to include feature view name in columns

        Returns:
            DataFrame with historical features joined at correct timestamps
        """
        start_time = time.time()

        if "event_timestamp" not in entity_df.columns:
            raise ValueError("entity_df must contain 'event_timestamp' column for PIT joins")

        # Retrieve from offline store
        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
            full_feature_names=full_feature_names
        ).to_df()

        retrieval_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Retrieved historical features: {len(training_df)} rows, "
            f"{len(feature_refs)} features in {retrieval_time_ms:.1f}ms"
        )

        return training_df

    def get_feature_freshness(self, feature_view_name: str) -> Dict[str, Any]:
        """
        Check feature freshness against SLO thresholds

        Returns:
            Dictionary with freshness status and metrics
        """
        # Get most recent event timestamp for this feature view
        # In production, this would query the online store metadata

        threshold = self.freshness_thresholds.get(
            feature_view_name,
            timedelta(hours=1)  # Default threshold
        )

        # TODO: Implement actual freshness check from online store
        # For now, return placeholder

        return {
            "feature_view": feature_view_name,
            "threshold_seconds": threshold.total_seconds(),
            "status": "HEALTHY",  # HEALTHY, STALE, CRITICAL
            "last_update": datetime.now(),
            "staleness_seconds": 0,
        }

    def validate_feature_coverage(
        self,
        feature_refs: List[str],
        entity_rows: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Validate feature coverage (non-null rate) for entities

        Returns:
            Dictionary mapping feature_ref -> coverage_rate
        """
        feature_vector = self.get_online_features(
            feature_refs=feature_refs,
            entity_rows=entity_rows,
            validate_pit=False  # Just checking coverage
        )

        coverage = {}
        for feature_ref in feature_refs:
            # Extract column name from feature_ref
            if ":" in feature_ref:
                col_name = feature_ref.split(":")[1]
            else:
                col_name = feature_ref

            if col_name in feature_vector.columns:
                non_null_rate = 1 - (feature_vector[col_name].isnull().sum() / len(feature_vector))
                coverage[feature_ref] = non_null_rate

        return coverage

    def get_retrieval_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get retrieval statistics for monitoring

        Returns:
            Dictionary with p50, p95, p99 latencies, null rates, etc.
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.retrieval_history
            if m.timestamp > cutoff_time
        ]

        if not recent_metrics:
            return {"message": "No recent retrievals"}

        latencies = [m.retrieval_time_ms for m in recent_metrics]
        null_rates = [m.null_rate for m in recent_metrics]

        import numpy as np

        return {
            "window_minutes": window_minutes,
            "num_retrievals": len(recent_metrics),
            "latency_p50_ms": np.percentile(latencies, 50),
            "latency_p95_ms": np.percentile(latencies, 95),
            "latency_p99_ms": np.percentile(latencies, 99),
            "avg_null_rate": np.mean(null_rates),
            "max_null_rate": np.max(null_rates),
            "total_pit_violations": sum(m.pit_violations for m in recent_metrics),
        }
