import os
import time
from contextlib import contextmanager


@contextmanager
def maybe_trace(label: str):
    """Conditionally capture VizTracer traces with contextual labels."""
    enabled = os.getenv("VIZTRACER_ENABLED", "false").lower() in {"1", "true", "yes"}
    if not enabled:
        yield
        return

    try:
        from viztracer import VizTracer  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        yield
        return

    ts = int(time.time() * 1000)
    out_dir = os.getenv("VIZ_OUT_DIR", "/tmp/traces")
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"trace_{label}_{ts}.json")

    tracer = VizTracer(output_file=output_file, ignore_c_function=True)
    tracer.start()
    try:
        yield
    finally:
        tracer.stop()
        tracer.save()
