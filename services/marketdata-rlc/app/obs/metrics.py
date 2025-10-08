from prometheus_client import Counter, Gauge, Histogram

# Published policies
rlc_policy_updates = Counter(
    "rlc_policy_updates_total",
    "Number of policy bundles published",
    ["mode", "provider"],
)

# Shadow vs baseline deltas
rlc_shadow_delta_ms = Gauge(
    "rlc_shadow_delta_ms",
    "Shadow policy p95 delta against baseline in milliseconds",
    ["provider"],
)

# Model inference timings
model_inference_ms = Histogram(
    "rlc_model_inference_ms",
    "ONNX model inference latency in milliseconds",
)

# Bandit telemetry
bandit_arm_selection_total = Counter(
    "bandit_arm_selection_total",
    "Bandit arm selections by provider and arm label",
    ["provider", "arm"],
)

# Observed error rates from feedback loop
rlc_observed_error_rate = Gauge(
    "rlc_observed_error_rate",
    "Observed worker error rate (0-1) for last minute",
    ["provider"],
)

rlc_error_constraint_violations_total = Counter(
    "rlc_error_constraint_violations_total",
    "Count of feedback windows exceeding error constraint",
    ["provider"],
)
