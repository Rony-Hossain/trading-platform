import asyncio
import logging
import os
import time
from typing import Any, Dict, List

import aiohttp

from workers.metrics import (
    start_metrics_server,
    worker_batch_fetch_total,
    worker_fetch_errors_total,
    worker_fetch_latency_ms,
)
from workers.policy_client import PolicyClient

log = logging.getLogger("collector.providerA")

PROVIDER = os.getenv("PROVIDER_ID", "providerA")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
POLICY_TOPIC = os.getenv("POLICY_TOPIC", "policy.updates")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9101"))

API_BASE = os.getenv("PROVIDER_API_BASE", "https://api.example.com")
API_KEY = os.getenv("PROVIDER_API_KEY", "changeme")


async def fetch_batch(session: aiohttp.ClientSession, batch_size: int) -> int:
    params = {"limit": batch_size}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with session.get(f"{API_BASE}/v1/ticks", params=params, headers=headers, timeout=15) as resp:
        resp.raise_for_status()
        payload: List[Dict[str, Any]] = await resp.json()
        # TODO: publish to internal Kafka / storage system
        return len(payload)


async def run() -> None:
    start_metrics_server(METRICS_PORT)
    policy_client = PolicyClient(kafka_bootstrap=KAFKA_BOOTSTRAP, topic=POLICY_TOPIC, provider=PROVIDER)
    await policy_client.start()

    async with aiohttp.ClientSession() as session:
        while True:
            snapshot = policy_client.get_snapshot()
            if not snapshot.bucket.allow(1.0):
                await asyncio.sleep(0.01)
                continue

            batch_size = snapshot.hints.batch_size
            start = time.perf_counter()
            try:
                count = await fetch_batch(session, batch_size)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                worker_batch_fetch_total.labels(PROVIDER).inc()
                worker_fetch_latency_ms.labels(PROVIDER).observe(elapsed_ms)
                log.info(
                    "Fetched %d records in %.1f ms (batch=%d delay=%dms)",
                    count,
                    elapsed_ms,
                    batch_size,
                    snapshot.hints.delay_ms,
                )
            except Exception as exc:  # noqa: BLE001
                worker_fetch_errors_total.labels(PROVIDER, type(exc).__name__).inc()
                log.warning("Fetch error: %s", exc)
                await asyncio.sleep(0.5)

            await asyncio.sleep(snapshot.hints.delay_ms / 1000.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
