from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from aiokafka import AIOKafkaConsumer

from .metrics import (
    start_metrics_server,  # re-export for convenience
    worker_current_batch_size,
    worker_current_inter_batch_delay_ms,
    worker_policy_applied_total,
    worker_policy_ttl_expired_total,
    worker_token_bucket_tokens,
)

log = logging.getLogger("policy.client")


@dataclass
class TokenBucket:
    provider: str
    refill_rate: float
    burst: int
    tokens: float
    last_ts: float

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_ts
        self.tokens = min(self.burst, self.tokens + elapsed * self.refill_rate)
        self.last_ts = now
        if self.tokens >= cost:
            self.tokens -= cost
            worker_token_bucket_tokens.labels(self.provider).set(self.tokens)
            return True
        worker_token_bucket_tokens.labels(self.provider).set(self.tokens)
        return False


@dataclass
class BatchHints:
    batch_size: int = 100
    delay_ms: int = 100


@dataclass
class PolicySnapshot:
    ttl_deadline: float
    bucket: TokenBucket
    hints: BatchHints


class PolicyClient:
    """Kafka consumer that keeps the latest policy bundle for a provider."""

    def __init__(self, kafka_bootstrap: str, topic: str, provider: str) -> None:
        self.bootstrap = kafka_bootstrap
        self.topic = topic
        self.provider = provider
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.snap: Optional[PolicySnapshot] = None

    async def start(self) -> None:
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap,
            group_id=f"worker-{self.provider}",
            enable_auto_commit=True,
            auto_offset_reset="latest",
            key_deserializer=lambda k: k,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await self.consumer.start()
        asyncio.create_task(self._consume_loop(), name=f"policy-consumer-{self.provider}")
        log.info("PolicyClient started (provider=%s topic=%s)", self.provider, self.topic)

    async def stop(self) -> None:
        if self.consumer:
            await self.consumer.stop()
            self.consumer = None

    async def _consume_loop(self) -> None:
        assert self.consumer
        async for msg in self.consumer:
            if msg.key != self.provider.encode():
                continue
            payload = msg.value
            policy = payload.get("token_policy", {})
            hints_payload = payload.get("batch_hints") or {}

            ttl_s = int(policy.get("ttl_s", 60))
            bucket = TokenBucket(
                provider=self.provider,
                refill_rate=float(policy.get("refill_rate", 50.0)),
                burst=int(policy.get("burst", 100)),
                tokens=float(policy.get("burst", 100)),
                last_ts=time.monotonic(),
            )
            hints = BatchHints(
                batch_size=int(hints_payload.get("batch_size", 100)),
                delay_ms=int(hints_payload.get("inter_batch_delay_ms", 100)),
            )

            self.snap = PolicySnapshot(
                ttl_deadline=time.monotonic() + ttl_s,
                bucket=bucket,
                hints=hints,
            )
            worker_policy_applied_total.labels(self.provider).inc()
            worker_current_batch_size.labels(self.provider).set(hints.batch_size)
            worker_current_inter_batch_delay_ms.labels(self.provider).set(hints.delay_ms)
            worker_token_bucket_tokens.labels(self.provider).set(bucket.tokens)
            log.info(
                "Applied policy for %s (refill=%.1f burst=%d hints=%s)",
                self.provider,
                bucket.refill_rate,
                bucket.burst,
                hints,
            )

    def get_snapshot(self) -> PolicySnapshot:
        now = time.monotonic()
        if not self.snap or now > self.snap.ttl_deadline:
            worker_policy_ttl_expired_total.labels(self.provider).inc()
            fallback_bucket = TokenBucket(
                provider=self.provider,
                refill_rate=50.0,
                burst=100,
                tokens=100,
                last_ts=now,
            )
            fallback_hints = BatchHints(batch_size=50, delay_ms=200)
            self.snap = PolicySnapshot(
                ttl_deadline=now + 30,
                bucket=fallback_bucket,
                hints=fallback_hints,
            )
            worker_current_batch_size.labels(self.provider).set(fallback_hints.batch_size)
            worker_current_inter_batch_delay_ms.labels(self.provider).set(fallback_hints.delay_ms)
            worker_token_bucket_tokens.labels(self.provider).set(fallback_bucket.tokens)
            log.warning("Policy TTL expired for %s; using fallback settings", self.provider)
        return self.snap
