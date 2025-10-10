from fastapi import APIRouter
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

router = APIRouter()

# ✅ Added Prometheus metrics
registry = CollectorRegistry()

# Prefilter metrics
prefilter_hits = Counter('news_prefilter_hits_total', 'Total prefilter dedupe hits', registry=registry)
prefilter_misses = Counter('news_prefilter_misses_total', 'Total prefilter misses', registry=registry)

# Ingest metrics
ingest_items_total = Counter('news_ingest_items_total', 'Total items ingested', ['source'], registry=registry)
ingest_errors_total = Counter('news_ingest_errors_total', 'Total ingest errors', ['source'], registry=registry)

# Lag metrics (seconds)
poll_lag_seconds = Gauge('news_poll_lag_seconds', 'Adapter polling lag', ['source'], registry=registry)
ingest_lag_seconds = Histogram('news_ingest_lag_seconds', 'Time from published to committed', registry=registry)
event_lag_seconds = Histogram('news_event_lag_seconds', 'Time from commit to event delivery', registry=registry)

# Circuit breaker metrics
circuit_open_total = Counter('news_circuit_open_total', 'Total circuit breaker opens', ['source'], registry=registry)

# Outbox metrics
outbox_sent_total = Counter('news_outbox_sent_total', 'Total outbox events sent', registry=registry)
outbox_dlq_total = Counter('news_outbox_dlq_total', 'Total outbox events sent to DLQ', registry=registry)

# Upsert metrics
upsert_created_total = Counter('news_upsert_created_total', 'Total new content created', registry=registry)
upsert_corrected_total = Counter('news_upsert_corrected_total', 'Total content corrections', registry=registry)
upsert_noop_total = Counter('news_upsert_noop_total', 'Total no-op upserts', registry=registry)

@router.get("/health")
def health_root():
    return {"status": "ok"}

@router.get("/metrics")
def metrics():
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
