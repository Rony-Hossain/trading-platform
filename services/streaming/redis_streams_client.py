"""
Redis Streams Client for Ultra-Low Latency Event Processing
Used for critical path: market data → prediction → order
Target: <5ms end-to-end latency
"""
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import redis
import json
from enum import Enum

logger = logging.getLogger(__name__)

class StreamPriority(Enum):
    """Stream priority levels"""
    CRITICAL = "CRITICAL"   # <5ms target
    HIGH = "HIGH"           # <10ms target
    NORMAL = "NORMAL"       # <50ms target

@dataclass
class StreamMessage:
    """Message in Redis Stream"""
    stream_name: str
    message_id: str
    data: Dict[str, Any]
    timestamp: datetime
    consumer_id: Optional[str] = None

class RedisStreamsClient:
    """
    Redis Streams client for ultra-low latency event processing

    Design principles:
    1. Minimal serialization overhead (JSON for speed)
    2. Consumer groups for load balancing
    3. Acknowledgment for reliability
    4. TTL for memory management
    5. Metrics for latency monitoring
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6380,  # redis-streams instance
        db: int = 0,
        max_stream_length: int = 10000,
        default_ttl_ms: int = 60000  # 1 minute TTL
    ):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False  # Keep bytes for speed
        )
        self.max_stream_length = max_stream_length
        self.default_ttl_ms = default_ttl_ms

        # Metrics
        self.publish_count = 0
        self.consume_count = 0
        self.latencies = []

        logger.info(f"Redis Streams client connected to {host}:{port}")

    # ==========================================================================
    # PRODUCER METHODS
    # ==========================================================================

    def publish(
        self,
        stream_name: str,
        data: Dict[str, Any],
        max_length: Optional[int] = None,
        priority: StreamPriority = StreamPriority.NORMAL
    ) -> str:
        """
        Publish message to stream

        Args:
            stream_name: Name of stream
            data: Message data (will be JSON-serialized)
            max_length: Max stream length (FIFO eviction)
            priority: Priority level (affects routing)

        Returns:
            Message ID
        """
        start_time = time.time()

        # Add metadata
        data["_timestamp"] = datetime.now().isoformat()
        data["_priority"] = priority.value

        # Serialize to JSON (faster than pickle for small messages)
        serialized = {
            k: json.dumps(v) if not isinstance(v, (str, bytes)) else v
            for k, v in data.items()
        }

        # Publish to stream with max length (capped list)
        message_id = self.client.xadd(
            name=stream_name,
            fields=serialized,
            maxlen=max_length or self.max_stream_length,
            approximate=True  # Faster, allows slightly longer stream
        )

        # Track metrics
        publish_latency_us = (time.time() - start_time) * 1_000_000
        self.publish_count += 1
        self.latencies.append(publish_latency_us)

        if publish_latency_us > 100:  # >100us is slow
            logger.warning(
                f"Slow publish: {publish_latency_us:.1f}µs to {stream_name}"
            )

        return message_id.decode() if isinstance(message_id, bytes) else message_id

    def publish_batch(
        self,
        stream_name: str,
        messages: List[Dict[str, Any]],
        max_length: Optional[int] = None
    ) -> List[str]:
        """
        Publish multiple messages efficiently using pipeline

        Args:
            stream_name: Name of stream
            messages: List of message data
            max_length: Max stream length

        Returns:
            List of message IDs
        """
        start_time = time.time()

        pipeline = self.client.pipeline()

        for data in messages:
            data["_timestamp"] = datetime.now().isoformat()

            serialized = {
                k: json.dumps(v) if not isinstance(v, (str, bytes)) else v
                for k, v in data.items()
            }

            pipeline.xadd(
                name=stream_name,
                fields=serialized,
                maxlen=max_length or self.max_stream_length,
                approximate=True
            )

        message_ids = pipeline.execute()

        batch_latency_ms = (time.time() - start_time) * 1000
        self.publish_count += len(messages)

        logger.debug(
            f"Published {len(messages)} messages to {stream_name} "
            f"in {batch_latency_ms:.2f}ms "
            f"({len(messages) / (batch_latency_ms / 1000):.0f} msg/sec)"
        )

        return [
            mid.decode() if isinstance(mid, bytes) else mid
            for mid in message_ids
        ]

    # ==========================================================================
    # CONSUMER METHODS
    # ==========================================================================

    def create_consumer_group(
        self,
        stream_name: str,
        group_name: str,
        start_id: str = "0"
    ):
        """
        Create consumer group for load balancing

        Args:
            stream_name: Name of stream
            group_name: Name of consumer group
            start_id: Starting message ID ("0" = beginning, "$" = end)
        """
        try:
            self.client.xgroup_create(
                name=stream_name,
                groupname=group_name,
                id=start_id,
                mkstream=True
            )
            logger.info(f"Created consumer group: {group_name} on {stream_name}")

        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group already exists: {group_name}")
            else:
                raise

    def consume(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 1000
    ) -> List[StreamMessage]:
        """
        Consume messages from stream using consumer group

        Args:
            stream_name: Name of stream
            group_name: Consumer group name
            consumer_name: This consumer's name
            count: Max messages to consume
            block_ms: Block for up to this many ms

        Returns:
            List of StreamMessage objects
        """
        start_time = time.time()

        # Read from stream
        results = self.client.xreadgroup(
            groupname=group_name,
            consumername=consumer_name,
            streams={stream_name: ">"},
            count=count,
            block=block_ms
        )

        messages = []

        if results:
            for stream, stream_messages in results:
                stream_name_str = stream.decode() if isinstance(stream, bytes) else stream

                for message_id, fields in stream_messages:
                    message_id_str = message_id.decode() if isinstance(message_id, bytes) else message_id

                    # Deserialize
                    data = {}
                    for k, v in fields.items():
                        key = k.decode() if isinstance(k, bytes) else k
                        value = v.decode() if isinstance(v, bytes) else v

                        # Try to parse JSON
                        try:
                            data[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            data[key] = value

                    # Extract timestamp
                    timestamp_str = data.pop("_timestamp", None)
                    timestamp = (
                        datetime.fromisoformat(timestamp_str)
                        if timestamp_str
                        else datetime.now()
                    )

                    # Calculate end-to-end latency
                    latency_ms = (datetime.now() - timestamp).total_seconds() * 1000

                    messages.append(
                        StreamMessage(
                            stream_name=stream_name_str,
                            message_id=message_id_str,
                            data=data,
                            timestamp=timestamp,
                            consumer_id=consumer_name
                        )
                    )

                    self.consume_count += 1

                    # Log slow messages
                    if latency_ms > 10:
                        logger.warning(
                            f"High latency: {latency_ms:.1f}ms for message {message_id_str}"
                        )

        consume_time_ms = (time.time() - start_time) * 1000

        if messages and consume_time_ms > 5:
            logger.debug(
                f"Consumed {len(messages)} messages in {consume_time_ms:.2f}ms"
            )

        return messages

    def ack(self, stream_name: str, group_name: str, message_ids: List[str]):
        """
        Acknowledge processed messages

        Args:
            stream_name: Name of stream
            group_name: Consumer group name
            message_ids: List of message IDs to acknowledge
        """
        self.client.xack(stream_name, group_name, *message_ids)

    def consume_loop(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[[StreamMessage], None],
        count: int = 10,
        block_ms: int = 100
    ):
        """
        Continuous consumption loop

        Args:
            stream_name: Name of stream
            group_name: Consumer group name
            consumer_name: This consumer's name
            handler: Callback function to process each message
            count: Max messages per batch
            block_ms: Block duration
        """
        logger.info(
            f"Starting consume loop: {stream_name} / {group_name} / {consumer_name}"
        )

        while True:
            try:
                messages = self.consume(
                    stream_name=stream_name,
                    group_name=group_name,
                    consumer_name=consumer_name,
                    count=count,
                    block_ms=block_ms
                )

                for message in messages:
                    try:
                        # Process message
                        handler(message)

                        # Acknowledge success
                        self.ack(stream_name, group_name, [message.message_id])

                    except Exception as e:
                        logger.error(
                            f"Error processing message {message.message_id}: {e}"
                        )
                        # Don't ack - message will be redelivered

            except KeyboardInterrupt:
                logger.info("Stopping consume loop")
                break
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                time.sleep(1)  # Back off on error

    # ==========================================================================
    # MONITORING
    # ==========================================================================

    def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get stream information"""
        info = self.client.xinfo_stream(stream_name)

        return {
            "length": info[b"length"],
            "first_entry": info[b"first-entry"],
            "last_entry": info[b"last-entry"],
            "consumer_groups": info[b"groups"]
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        import numpy as np

        if self.latencies:
            latencies_array = np.array(self.latencies[-1000:])  # Last 1000

            return {
                "publish_count": self.publish_count,
                "consume_count": self.consume_count,
                "publish_latency_p50_us": np.percentile(latencies_array, 50),
                "publish_latency_p95_us": np.percentile(latencies_array, 95),
                "publish_latency_p99_us": np.percentile(latencies_array, 99),
                "publish_latency_max_us": np.max(latencies_array)
            }
        else:
            return {
                "publish_count": self.publish_count,
                "consume_count": self.consume_count
            }

    def trim_stream(self, stream_name: str, max_length: int):
        """Trim stream to max length"""
        self.client.xtrim(stream_name, maxlen=max_length, approximate=True)

    def close(self):
        """Close client"""
        self.client.close()

if __name__ == "__main__":
    # Example usage
    client = RedisStreamsClient()

    # Publish test messages
    for i in range(100):
        client.publish(
            stream_name="test_stream",
            data={"value": i, "timestamp": time.time()},
            priority=StreamPriority.CRITICAL
        )

    # Print metrics
    metrics = client.get_metrics()
    print("Metrics:", metrics)

    client.close()
