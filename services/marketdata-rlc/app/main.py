from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from starlette.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .core.config import settings
from .core.logging import setup_logging
from .jobs.feedback import FeedbackJob
from .loop.policy_loop import PolicyLoop
from .models.onnx_runtime import RlcPredictor
from .policy.publisher import PolicyPublisher
from .policy.synthesizer import PolicySynthesizer
from .store import timescale
from .store.timescale import TsPool

setup_logging()
log = logging.getLogger("rlc.main")

predictor: RlcPredictor | None = None
publisher: PolicyPublisher | None = None
synthesizer: PolicySynthesizer | None = None
loop: PolicyLoop | None = None
feedback: FeedbackJob | None = None
timescale_pool: TsPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, publisher, synthesizer, loop, feedback, timescale_pool

    timescale_pool = TsPool(settings.pg_dsn)
    timescale.ts_pool = timescale_pool
    await timescale_pool.start()

    predictor = RlcPredictor(settings.model_path, settings.error_model_path)
    publisher = PolicyPublisher(settings.kafka_bootstrap)
    await publisher.start()

    synthesizer = PolicySynthesizer()
    loop = PolicyLoop(
        predictor=predictor,
        synthesizer=synthesizer,
        publisher=publisher,
        providers=settings.providers,
    )
    await loop.start()

    feedback = FeedbackJob(settings.providers)
    await feedback.start()

    log.info("marketdata-rlc starting (mode=%s env=%s)", settings.mode, settings.env)
    try:
        yield
    finally:
        log.info("marketdata-rlc shutting down")
        if feedback:
            await feedback.stop()
        if loop:
            await loop.stop()
        if publisher:
            await publisher.stop()
        if timescale_pool:
            await timescale_pool.stop()


tags_metadata = [
    {"name": "policy", "description": "Publish and inspect rate-limit policies."},
    {"name": "ops", "description": "Operational endpoints for health and diagnostics."},
]

app = FastAPI(
    title="MarketData RLC",
    description="Predictive rate-limit controller service for market data workers.",
    version="0.1.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
