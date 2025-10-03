"""
SLO Tracker
Monitors error budgets and availability targets
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import time
import structlog
from redis import Redis

logger = structlog.get_logger(__name__)


class SLOTracker:
    """
    Track SLO metrics and error budgets

    Targets:
    - Availability: 99.9% (43 minutes downtime/month)
    - Latency: p95 < 150ms, p99 < 300ms
    - Error rate: < 0.1%
    """

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.availability_target = 0.999  # 99.9%
        self.p95_target_ms = 150
        self.p99_target_ms = 300
        self.error_rate_target = 0.001  # 0.1%

        logger.info("slo_tracker_initialized")

    def record_request(
        self,
        endpoint: str,
        latency_ms: float,
        success: bool,
        timestamp: Optional[float] = None
    ):
        """Record request metrics for SLO tracking"""
        timestamp = timestamp or time.time()
        day_key = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

        # Increment counters
        total_key = f"slo:requests:{endpoint}:{day_key}"
        success_key = f"slo:success:{endpoint}:{day_key}"
        latency_key = f"slo:latency:{endpoint}:{day_key}"

        self.redis.incr(total_key)
        self.redis.expire(total_key, 86400 * 31)  # 31 days

        if success:
            self.redis.incr(success_key)
            self.redis.expire(success_key, 86400 * 31)

        # Store latency for percentile calculation (simplified)
        self.redis.rpush(latency_key, latency_ms)
        self.redis.expire(latency_key, 86400 * 31)

        logger.debug(
            "slo_request_recorded",
            endpoint=endpoint,
            latency_ms=latency_ms,
            success=success
        )

    def get_error_budget(self, window_days: int = 30) -> Dict[str, Any]:
        """
        Calculate remaining error budget

        Returns:
            {
                "availability_target": 0.999,
                "current_availability": 0.9995,
                "error_budget_remaining_pct": 50.0,
                "error_budget_minutes_remaining": 21.5,
                "status": "healthy"
            }
        """
        # Get total requests and successes over window
        total_requests = 0
        total_successes = 0

        for days_ago in range(window_days):
            date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

            # Aggregate across all endpoints
            for key in self.redis.scan_iter(match=f"slo:requests:*:{date}"):
                total_requests += int(self.redis.get(key) or 0)

            for key in self.redis.scan_iter(match=f"slo:success:*:{date}"):
                total_successes += int(self.redis.get(key) or 0)

        if total_requests == 0:
            return {
                "availability_target": self.availability_target,
                "current_availability": 1.0,
                "error_budget_remaining_pct": 100.0,
                "error_budget_minutes_remaining": 43.0,  # Full month budget
                "status": "healthy",
                "total_requests": 0
            }

        current_availability = total_successes / total_requests

        # Calculate error budget
        # Target: 99.9% = 43 minutes downtime/month
        allowed_failures = total_requests * (1 - self.availability_target)
        actual_failures = total_requests - total_successes
        remaining_failures = allowed_failures - actual_failures

        error_budget_remaining_pct = (remaining_failures / allowed_failures * 100) if allowed_failures > 0 else 100

        # Convert to minutes (approximate)
        total_minutes_in_window = window_days * 24 * 60
        error_budget_minutes_total = total_minutes_in_window * (1 - self.availability_target)
        error_budget_minutes_remaining = error_budget_minutes_total * (error_budget_remaining_pct / 100)

        # Determine status
        if error_budget_remaining_pct < 10:
            status = "critical"
        elif error_budget_remaining_pct < 25:
            status = "warning"
        else:
            status = "healthy"

        return {
            "availability_target": self.availability_target,
            "current_availability": round(current_availability, 5),
            "error_budget_remaining_pct": round(error_budget_remaining_pct, 2),
            "error_budget_minutes_remaining": round(error_budget_minutes_remaining, 2),
            "status": status,
            "total_requests": total_requests,
            "total_successes": total_successes,
            "window_days": window_days
        }

    def get_latency_percentiles(self, endpoint: str, days: int = 1) -> Dict[str, float]:
        """
        Calculate latency percentiles for endpoint

        Returns:
            {
                "p50": 45.2,
                "p95": 120.5,
                "p99": 250.0,
                "p95_target": 150,
                "p99_target": 300,
                "p95_met": True,
                "p99_met": True
            }
        """
        latencies = []

        for days_ago in range(days):
            date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            latency_key = f"slo:latency:{endpoint}:{date}"

            values = self.redis.lrange(latency_key, 0, -1)
            latencies.extend([float(v) for v in values])

        if not latencies:
            return {
                "p50": 0,
                "p95": 0,
                "p99": 0,
                "p95_target": self.p95_target_ms,
                "p99_target": self.p99_target_ms,
                "p95_met": True,
                "p99_met": True,
                "sample_size": 0
            }

        latencies.sort()
        n = len(latencies)

        p50 = latencies[int(n * 0.50)] if n > 0 else 0
        p95 = latencies[int(n * 0.95)] if n > 0 else 0
        p99 = latencies[int(n * 0.99)] if n > 0 else 0

        return {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "p95_target": self.p95_target_ms,
            "p99_target": self.p99_target_ms,
            "p95_met": p95 <= self.p95_target_ms,
            "p99_met": p99 <= self.p99_target_ms,
            "sample_size": n
        }

    def get_slo_status(self) -> Dict[str, Any]:
        """Get overall SLO status"""
        error_budget = self.get_error_budget(window_days=30)

        # Get latency for main endpoints
        plan_latency = self.get_latency_percentiles("/api/v1/plan", days=1)
        alerts_latency = self.get_latency_percentiles("/api/v1/alerts", days=1)

        overall_status = "healthy"
        if error_budget["status"] == "critical":
            overall_status = "critical"
        elif error_budget["status"] == "warning" or not plan_latency["p95_met"]:
            overall_status = "warning"

        return {
            "overall_status": overall_status,
            "error_budget": error_budget,
            "latency": {
                "plan": plan_latency,
                "alerts": alerts_latency
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global SLO tracker instance
_slo_tracker: Optional[SLOTracker] = None


def init_slo_tracker(redis_client: Redis) -> SLOTracker:
    """Initialize global SLO tracker"""
    global _slo_tracker
    _slo_tracker = SLOTracker(redis_client)
    logger.info("slo_tracker_initialized")
    return _slo_tracker


def get_slo_tracker() -> SLOTracker:
    """Get global SLO tracker instance"""
    if _slo_tracker is None:
        raise RuntimeError("SLOTracker not initialized. Call init_slo_tracker() first.")
    return _slo_tracker
