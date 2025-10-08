import asyncio
import json
import logging
from typing import Optional

from aiokafka import AIOKafkaProducer

from ..core.config import settings
from ..schemas.policy import PolicyBundle

log = logging.getLogger("rlc.publisher")


class PolicyPublisher:
    """Kafka publisher for policy bundles."""

    def __init__(self, bootstrap_servers: str, client_id: str = "marketdata-rlc") -> None:
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self._producer: Optional[AIOKafkaProducer] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._producer:
            return
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            client_id=self.client_id,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            linger_ms=50,
            acks="all",
        )
        await producer.start()
        self._producer = producer
        log.info("Kafka producer connected to %s", self.bootstrap_servers)

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            self._producer = None
            log.info("Kafka producer stopped")

    async def publish_policy(self, bundle: PolicyBundle) -> None:
        assert self._producer, "Producer not started"
        topic = settings.kafka_topic_shadow if bundle.mode == "shadow" else settings.kafka_topic_updates
        key = bundle.provider.encode("utf-8")
        payload = bundle.model_dump()
        async with self._lock:
            await self._producer.send_and_wait(topic=topic, key=key, value=payload)
        log.debug("Published policy bundle to %s (provider=%s)", topic, bundle.provider)
