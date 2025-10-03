"""
Observability Layer
Structured logging, Prometheus metrics, and request tracing
"""
import structlog
import time
from contextvars import ContextVar
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge
from ulid import ULID

# Context var for request tracing
request_id_ctx: ContextVar[str] = ContextVar('request_id', default='')
user_id_ctx: ContextVar[str] = ContextVar('user_id', default='')
mode_ctx: ContextVar[str] = ContextVar('mode', default='')

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

# Plan metrics
plan_requests_total = Counter(
    'signal_plan_requests_total',
    'Total plan requests',
    ['mode', 'status']
)
plan_latency_seconds = Histogram(
    'signal_plan_latency_seconds',
    'Plan generation latency',
    ['mode'],
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
)
picks_generated_total = Counter(
    'signal_picks_total',
    'Picks generated',
    ['action', 'confidence']
)

# Degradation metrics
degraded_responses_total = Counter(
    'signal_degraded_total',
    'Degraded responses',
    ['reason']
)

# Upstream metrics
upstream_latency_seconds = Histogram(
    'signal_upstream_latency_seconds',
    'Upstream service latency',
    ['service'],
    buckets=[0.01, 0.05, 0.08, 0.1, 0.15, 0.2]
)
upstream_errors_total = Counter(
    'signal_upstream_errors_total',
    'Upstream errors',
    ['service', 'error_type']
)

# Alert metrics
alert_throttled_total = Counter(
    'signal_alerts_throttled_total',
    'Alerts suppressed by throttle',
    ['reason']
)

# SLO metrics
slo_availability = Gauge(
    'signal_slo_availability',
    'Service availability (success rate)'
)
slo_latency_p95 = Histogram(
    'signal_slo_latency_p95',
    'p95 latency',
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
)
slo_error_budget_pct = Gauge(
    'signal_slo_error_budget_pct',
    'Error budget remaining %'
)
slo_budget_burn_rate = Gauge(
    'signal_slo_budget_burn_rate',
    'Error budget burn rate (1h)'
)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO"):
    """Setup structured logging with structlog"""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, log_level.upper(), structlog.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def log_startup_info(settings):
    """Log startup information"""
    logger = structlog.get_logger(__name__)
    logger.info(
        "service_starting",
        service=settings.SERVICE_NAME,
        version=settings.VERSION,
        redis_host=settings.REDIS_HOST,
        redis_port=settings.REDIS_PORT,
        debug=settings.DEBUG
    )


# =============================================================================
# REQUEST TRACING
# =============================================================================

def generate_request_id() -> str:
    """Generate ULID for request tracking"""
    return str(ULID())


def set_request_context(request_id: str, user_id: str = "", mode: str = ""):
    """Set request context for logging"""
    request_id_ctx.set(request_id)
    user_id_ctx.set(user_id)
    mode_ctx.set(mode)

    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        user_id=user_id,
        mode=mode
    )


def get_request_id() -> str:
    """Get current request ID from context"""
    return request_id_ctx.get()


# =============================================================================
# LOGGING HELPERS
# =============================================================================

def log_plan_decision(pick: dict):
    """Log structured plan decision"""
    logger = structlog.get_logger(__name__)
    logger.info(
        "plan_pick_generated",
        symbol=pick.get('symbol'),
        action=pick.get('action'),
        shares=pick.get('shares'),
        risk_usd=pick.get('max_risk_usd'),
        confidence=pick.get('confidence'),
        reason_codes=pick.get('reason_codes'),
        decision_path=pick.get('decision_path'),
        limits_applied=pick.get('limits_applied')
    )

    # Update metrics
    picks_generated_total.labels(
        action=pick.get('action', 'UNKNOWN'),
        confidence=pick.get('confidence', 'unknown')
    ).inc()


def log_degradation(service: str, reason: str):
    """Log graceful degradation"""
    logger = structlog.get_logger(__name__)
    logger.warning(
        "upstream_degraded",
        service=service,
        reason=reason
    )
    degraded_responses_total.labels(reason=service).inc()


# =============================================================================
# UPSTREAM TIMER
# =============================================================================

class UpstreamTimer:
    """Context manager for timing upstream calls"""

    def __init__(self, service: str):
        self.service = service
        self.start = None
        self.logger = structlog.get_logger(__name__)

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start
        upstream_latency_seconds.labels(service=self.service).observe(latency)

        if exc_type:
            upstream_errors_total.labels(
                service=self.service,
                error_type=exc_type.__name__
            ).inc()

        self.logger.info(
            "upstream_call",
            service=self.service,
            latency_ms=latency * 1000,
            success=exc_type is None
        )

        return False  # Don't suppress exceptions
