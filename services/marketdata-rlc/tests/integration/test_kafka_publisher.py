import asyncio
import json

import pytest
from aiokafka import AIOKafkaConsumer
from testcontainers.kafka import KafkaContainer

from app.policy.publisher import PolicyPublisher
from app.schemas.policy import BatchHints, PolicyBundle, TierQuota, TokenPolicy


pytestmark = pytest.mark.asyncio


async def _consume_one(bootstrap: str, topic: str, key: bytes, timeout: float = 10.0):
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id="itest",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        key_deserializer=lambda x: x,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    await consumer.start()
    try:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            records = await consumer.getmany(timeout_ms=500)
            for batch in records.values():
                for record in batch:
                    if record.key == key:
                        return record.value
        raise TimeoutError("Message not received in time")
    finally:
        await consumer.stop()


async def test_policy_publisher_shadow() -> None:
    with KafkaContainer() as kafka:
        bootstrap = kafka.get_bootstrap_server()

        publisher = PolicyPublisher(bootstrap)
        await publisher.start()
        bundle = PolicyBundle(
            provider="providerA",
            token_policy=TokenPolicy(refill_rate=100.0, burst=200, jitter_ms=5, ttl_s=60),
            tier_quota=TierQuota(t0_max=100, t1_max=900, t2_mode="60s"),
            mode="shadow",
            batch_hints=BatchHints(batch_size=150, inter_batch_delay_ms=120),
        )
        await publisher.publish_policy(bundle)
        await publisher.stop()

        payload = await _consume_one(bootstrap, "policy.shadow", b"providerA")
        assert payload["provider"] == "providerA"
        assert payload["mode"] == "shadow"
        assert payload["token_policy"]["refill_rate"] == 100.0
