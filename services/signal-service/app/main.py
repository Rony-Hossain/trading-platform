"""
Signal Service Main Application
FastAPI service for beginner-friendly trading recommendations
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import structlog
from contextlib import asynccontextmanager

from .config import settings
from .core.observability import setup_logging, log_startup_info
from .core.policy_manager import init_policy_manager
from .core.decision_store import init_decision_store
from .core.idempotency import init_idempotency_manager
from .core.swr_cache import init_swr_cache_manager
from .core.guardrails import init_guardrail_engine
from .core.fitness_checker import init_fitness_checker
from .core.slo_tracker import init_slo_tracker
from pathlib import Path
from redis import Redis

# Setup logging
setup_logging(settings.LOG_LEVEL)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    log_startup_info(settings)
    logger.info(
        "signal_service_starting",
        service=settings.SERVICE_NAME,
        version=settings.VERSION,
        port=settings.PORT
    )

    # Initialize Redis connection
    redis_client = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        db=settings.REDIS_DB,
        decode_responses=False
    )
    logger.info(
        "redis_connected",
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT
    )

    # Initialize policy manager
    policy_path = Path(settings.POLICY_FILE)
    if policy_path.exists():
        policy_manager = init_policy_manager(policy_path)
        logger.info(
            "policy_manager_initialized",
            config_path=str(policy_path),
            version=policy_manager.get('version')
        )
    else:
        logger.warning(
            "policy_file_not_found",
            config_path=str(policy_path),
            message="Using default policies"
        )

    # Initialize decision store
    decision_store = init_decision_store(
        redis_client,
        ttl_days=settings.DECISION_SNAPSHOT_TTL_DAYS
    )
    logger.info(
        "decision_store_initialized",
        ttl_days=settings.DECISION_SNAPSHOT_TTL_DAYS
    )

    # Initialize idempotency manager
    idempotency_manager = init_idempotency_manager(
        redis_client,
        ttl_seconds=settings.IDEMPOTENCY_TTL_SECONDS
    )
    logger.info(
        "idempotency_manager_initialized",
        ttl_seconds=settings.IDEMPOTENCY_TTL_SECONDS
    )

    # Initialize SWR cache manager
    swr_cache_manager = init_swr_cache_manager(
        redis_client,
        default_ttl=settings.PLAN_CACHE_TTL,
        default_stale_ttl=settings.PLAN_CACHE_TTL * 4  # 4x fresh TTL for stale window
    )
    logger.info(
        "swr_cache_manager_initialized",
        default_ttl=settings.PLAN_CACHE_TTL,
        default_stale_ttl=settings.PLAN_CACHE_TTL * 4
    )

    # Initialize guardrail engine
    guardrail_engine = init_guardrail_engine()
    logger.info("guardrail_engine_initialized")

    # Initialize fitness checker
    fitness_checker = init_fitness_checker()
    logger.info("fitness_checker_initialized")

    # Initialize SLO tracker
    slo_tracker = init_slo_tracker(redis_client)
    logger.info("slo_tracker_initialized")

    yield

    # Shutdown
    logger.info("signal_service_shutting_down")
    redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="Signal Service",
    description="Beginner-friendly trading orchestration layer",
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
if settings.ENABLE_METRICS:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


# Health check
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.VERSION
    }


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with service info"""
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs"
    }


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred"
            }
        }
    )


# API routers
from .api.v1 import plan, alerts, actions, positions, explain, internal
app.include_router(plan.router, prefix="/api/v1", tags=["Plan"])
app.include_router(alerts.router, prefix="/api/v1", tags=["Alerts"])
app.include_router(actions.router, prefix="/api/v1", tags=["Actions"])
app.include_router(positions.router, prefix="/api/v1", tags=["Positions"])
app.include_router(explain.router, prefix="/api/v1", tags=["Explain"])
app.include_router(internal.router, prefix="/internal", tags=["Internal"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.DEBUG
    )
