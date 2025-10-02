"""
Streaming Integration Layer
Routes events between Kafka (durable) and Redis Streams (low-latency)
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from redis_streams_client import RedisStreamsClient, StreamPriority

logger = logging.getLogger(__name__)

class EventRoute(Enum):
    """Event routing strategy"""
    KAFKA_ONLY = "KAFKA_ONLY"          # Durable, asynchronous
    REDIS_ONLY = "REDIS_ONLY"          # Fast, ephemeral
    BOTH = "BOTH"                       # Write to both (critical events)
    REDIS_THEN_KAFKA = "REDIS_THEN_KAFKA"  # Fast path first, then durable

@dataclass
class EventMetadata:
    """Metadata for event routing"""
    event_type: str
    priority: StreamPriority
    route: EventRoute
    source: str
    timestamp: datetime

class StreamingIntegration:
    """
    Unified streaming integration

    Routing logic:
    - CRITICAL path (market data → prediction → order): Redis Streams (<5ms)
    - DURABLE storage (orders, fills, positions): Kafka
    - MONITORING (metrics, alerts): Both (Redis for real-time, Kafka for storage)
    """

    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        redis_host: str = "localhost",
        redis_port: int = 6380
    ):
        # Kafka producer (durable writes)
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            compression_type="lz4",
            acks="all",  # Wait for all in-sync replicas
            retries=3,
            max_in_flight_requests_per_connection=5,
            linger_ms=0,  # No batching for low latency
        )

        # Redis Streams client (low-latency)
        self.redis_client = RedisStreamsClient(
            host=redis_host,
            port=redis_port
        )

        # Metrics
        self.kafka_publish_count = 0
        self.redis_publish_count = 0
        self.route_times = {"kafka": [], "redis": [], "both": []}

        logger.info("Streaming Integration initialized")

    # ==========================================================================
    # UNIFIED PUBLISH API
    # ==========================================================================

    def publish_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        route: EventRoute = EventRoute.BOTH,
        priority: StreamPriority = StreamPriority.NORMAL,
        kafka_topic: Optional[str] = None,
        redis_stream: Optional[str] = None
    ):
        """
        Publish event with automatic routing

        Args:
            event_type: Type of event (e.g., "market_data.quote")
            data: Event payload
            route: Routing strategy
            priority: Priority level
            kafka_topic: Kafka topic (defaults to event_type)
            redis_stream: Redis stream (defaults to event_type)
        """
        start_time = time.time()

        # Add metadata
        metadata = EventMetadata(
            event_type=event_type,
            priority=priority,
            route=route,
            source="streaming_integration",
            timestamp=datetime.now()
        )

        enriched_data = {
            **data,
            "_metadata": {
                "event_type": event_type,
                "priority": priority.value,
                "timestamp": metadata.timestamp.isoformat(),
                "source": metadata.source
            }
        }

        # Route based on strategy
        if route == EventRoute.KAFKA_ONLY:
            self._publish_to_kafka(kafka_topic or event_type, enriched_data)

        elif route == EventRoute.REDIS_ONLY:
            self._publish_to_redis(redis_stream or event_type, enriched_data, priority)

        elif route == EventRoute.BOTH:
            # Publish to both in parallel (don't wait for Kafka ack)
            self._publish_to_redis(redis_stream or event_type, enriched_data, priority)
            self._publish_to_kafka(kafka_topic or event_type, enriched_data)

        elif route == EventRoute.REDIS_THEN_KAFKA:
            # Fast path first
            self._publish_to_redis(redis_stream or event_type, enriched_data, priority)

            # Asynchronous Kafka write (fire and forget)
            self.kafka_producer.send(
                kafka_topic or event_type,
                value=enriched_data
            )

        route_time_ms = (time.time() - start_time) * 1000
        self.route_times[route.value.lower().split("_")[0]].append(route_time_ms)

        # Log slow routes
        if route_time_ms > 10:
            logger.warning(
                f"Slow routing: {route_time_ms:.2f}ms for {event_type} ({route.value})"
            )

    def _publish_to_kafka(self, topic: str, data: Dict[str, Any]):
        """Publish to Kafka"""
        future = self.kafka_producer.send(topic, value=data)

        try:
            # Wait for send to complete (blocking)
            record_metadata = future.get(timeout=1.0)
            self.kafka_publish_count += 1

        except KafkaError as e:
            logger.error(f"Kafka publish failed: {e}")
            # TODO: Send to DLQ

    def _publish_to_redis(
        self,
        stream: str,
        data: Dict[str, Any],
        priority: StreamPriority
    ):
        """Publish to Redis Streams"""
        try:
            self.redis_client.publish(
                stream_name=stream,
                data=data,
                priority=priority
            )
            self.redis_publish_count += 1

        except Exception as e:
            logger.error(f"Redis publish failed: {e}")
            # Fallback to Kafka
            self._publish_to_kafka(stream, data)

    # ==========================================================================
    # PREDEFINED EVENT PUBLISHERS
    # ==========================================================================

    def publish_market_data(self, symbol: str, quote: Dict[str, Any]):
        """Publish market data quote (CRITICAL path)"""
        self.publish_event(
            event_type="market_data.level1_quotes",
            data={"symbol": symbol, **quote},
            route=EventRoute.REDIS_THEN_KAFKA,  # Fast path + durable
            priority=StreamPriority.CRITICAL
        )

    def publish_alpha_signal(self, symbol: str, prediction: Dict[str, Any]):
        """Publish alpha prediction (CRITICAL path)"""
        self.publish_event(
            event_type="signals.alpha_predictions",
            data={"symbol": symbol, **prediction},
            route=EventRoute.REDIS_THEN_KAFKA,
            priority=StreamPriority.CRITICAL
        )

    def publish_order(self, order: Dict[str, Any]):
        """Publish new order (CRITICAL + DURABLE)"""
        self.publish_event(
            event_type="orders.new",
            data=order,
            route=EventRoute.BOTH,  # Must be durable
            priority=StreamPriority.CRITICAL
        )

    def publish_fill(self, fill: Dict[str, Any]):
        """Publish order fill (CRITICAL + DURABLE)"""
        self.publish_event(
            event_type="orders.fills",
            data=fill,
            route=EventRoute.BOTH,
            priority=StreamPriority.CRITICAL
        )

    def publish_position_update(self, position: Dict[str, Any]):
        """Publish position update (DURABLE)"""
        self.publish_event(
            event_type="portfolio.positions",
            data=position,
            route=EventRoute.KAFKA_ONLY,  # Compacted topic
            priority=StreamPriority.NORMAL
        )

    def publish_risk_alert(self, alert: Dict[str, Any]):
        """Publish risk alert (HIGH priority, BOTH)"""
        self.publish_event(
            event_type="risk.var_breaches",
            data=alert,
            route=EventRoute.BOTH,
            priority=StreamPriority.HIGH
        )

    # ==========================================================================
    # MONITORING
    # ==========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics"""
        redis_metrics = self.redis_client.get_metrics()

        import numpy as np

        kafka_times = self.route_times.get("kafka", [])
        redis_times = self.route_times.get("redis", [])

        metrics = {
            "kafka_publish_count": self.kafka_publish_count,
            "redis_publish_count": self.redis_publish_count,
            **redis_metrics
        }

        if kafka_times:
            metrics["kafka_route_p95_ms"] = np.percentile(kafka_times[-1000:], 95)

        if redis_times:
            metrics["redis_route_p95_ms"] = np.percentile(redis_times[-1000:], 95)

        return metrics

    def flush(self):
        """Flush pending messages"""
        self.kafka_producer.flush(timeout=5.0)

    def close(self):
        """Close connections"""
        self.kafka_producer.close()
        self.redis_client.close()

if __name__ == "__main__":
    # Example usage
    integration = StreamingIntegration()

    # Publish market data (critical path)
    integration.publish_market_data(
        symbol="AAPL",
        quote={"bid": 150.0, "ask": 150.05, "last": 150.02, "volume": 1000000}
    )

    # Publish alpha signal
    integration.publish_alpha_signal(
        symbol="AAPL",
        prediction={"alpha": 0.0012, "confidence": 0.85, "horizon_minutes": 30}
    )

    # Publish order
    integration.publish_order({
        "order_id": "ORDER123",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "order_type": "limit",
        "limit_price": 150.0
    })

    # Get metrics
    metrics = integration.get_metrics()
    print("Metrics:", json.dumps(metrics, indent=2))

    integration.close()
