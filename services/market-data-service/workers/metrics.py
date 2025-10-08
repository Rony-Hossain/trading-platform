from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Policy lifecycle
worker_policy_applied_total = Counter(
    "worker_policy_applied_total",
    "Count of policy bundles applied by this worker",
    ["provider"],
)

worker_policy_ttl_expired_total = Counter(
    "worker_policy_ttl_expired_total",
    "Number of times the worker fell back due to TTL expiry",
    ["provider"],
)

# Token buckets & pacing
worker_token_bucket_tokens = Gauge(
    "worker_token_bucket_tokens",
    "Current tokens remaining in the worker token bucket",
    ["provider"],
)

worker_current_batch_size = Gauge(
    "worker_current_batch_size",
    "Current batch size hint received from RLC",
    ["provider"],
)

worker_current_inter_batch_delay_ms = Gauge(
    "worker_current_inter_batch_delay_ms",
    "Current inter-batch delay hint in milliseconds",
    ["provider"],
)

# Fetch telemetry
worker_batch_fetch_total = Counter(
    "worker_batch_fetch_total",
    "Number of provider batches fetched",
    ["provider"],
)

worker_fetch_errors_total = Counter(
    "worker_fetch_errors_total",
    "Number of provider batch fetch errors",
    ["provider", "reason"],
)

worker_fetch_latency_ms = Histogram(
    "worker_fetch_latency_ms",
    "Latency of provider batch fetches",
    ["provider"],
)


def start_metrics_server(port: int = 9101) -> None:
    """Start a Prometheus metrics endpoint."""
    start_http_server(port)
