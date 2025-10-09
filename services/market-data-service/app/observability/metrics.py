from prometheus_client import Counter, Gauge, Histogram

# --- Providers & routing ---
PROVIDER_SELECTED = Counter(
    "provider_selected_total",
    "Provider selections by capability",
    ["capability", "provider"],
)

PROVIDER_ERRORS = Counter(
    "provider_errors_total",
    "Provider errors by code/type",
    ["provider", "endpoint", "code"],
)

PROVIDER_LATENCY = Histogram(
    "provider_latency_ms_bucket",
    "Latency of provider calls (ms)",
    ["provider", "endpoint"],
    buckets=(25, 50, 100, 200, 400, 800, 1600, 3200),
)

CIRCUIT_STATE = Gauge(
    "circuit_state",
    "Circuit breaker state (0=open, 0.5=half_open, 1=closed)",
    ["provider"],
)

# --- Ingestion / backfill ---
JOBS_PROCESSED = Counter(
    "jobs_processed_total",
    "Jobs processed by type and provider",
    ["type", "provider"],
)

GAPS_FOUND = Counter(
    "gaps_found_total",
    "Gaps detected by interval",
    ["interval"],
)

BACKFILL_ENQUEUED = Counter(
    "backfill_jobs_enqueued_total",
    "Backfill jobs enqueued by tier",
    ["tier"],
)

BACKFILL_COMPLETED = Counter(
    "backfill_jobs_completed_total",
    "Backfill jobs completed by tier and status",
    ["tier", "status"],
)

BACKFILL_QUEUE_DEPTH = Gauge(
    "backfill_queue_depth",
    "Current depth of backfill queue",
)

BACKFILL_OLDEST_AGE = Gauge(
    "backfill_oldest_age_seconds",
    "Age (seconds) of the oldest queued backfill job",
)

WRITE_BATCH_LATENCY = Histogram(
    "write_batch_ms_bucket",
    "DB write batch latency (ms)",
    buckets=(5, 10, 20, 40, 80, 160, 320, 640, 1280),
)

INGESTION_LAG = Gauge(
    "ingestion_lag_seconds",
    "Lag (seconds) between now and newest bar written",
    ["interval"],
)

# --- WebSocket / streaming ---
WS_CLIENTS = Gauge(
    "ws_clients_gauge",
    "Active WebSocket clients",
)

WS_PUBLISH_LATENCY = Histogram(
    "ws_publish_latency_ms_bucket",
    "WS publish latency (ms)",
    buckets=(1, 2, 5, 10, 20, 40, 80, 160),
)

WS_DROPPED = Counter(
    "ws_dropped_messages_total",
    "Messages dropped due to client backpressure",
)

# --- SLO monitors ---
SLO_GAP_VIOLATION_RATE = Gauge(
    "slo_gap_violation_rate",
    "Fraction of sampled symbols violating gap<=2x interval",
    ["tier", "interval"],
)
