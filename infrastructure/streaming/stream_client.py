"""
Redis Streams Client for Trading Platform
Provides low-latency streaming infrastructure for features, signals, and market data
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


# Prometheus metrics
MESSAGES_PRODUCED = Counter(
    'stream_messages_produced_total',
    'Total messages produced to streams',
    ['stream_name']
)

MESSAGES_CONSUMED = Counter(
    'stream_messages_consumed_total',
    'Total messages consumed from streams',
    ['stream_name', 'consumer_group']
)

STREAM_LAG = Gauge(
    'stream_lag_ms',
    'Stream lag in milliseconds',
    ['stream_name', 'consumer_group']
)

PROCESSING_TIME = Histogram(
    'stream_processing_time_ms',
    'Message processing time in milliseconds',
    ['stream_name'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

STREAM_ERRORS = Counter(
    'stream_errors_total',
    'Total stream errors',
    ['stream_name', 'error_type']
)


@dataclass
class StreamMessage:
    """Represents a message in a stream"""
    stream_name: str
    message_id: str
    data: Dict[str, Any]
    timestamp: float


class RedisStreamClient:
    """
    Redis Streams client with producer and consumer capabilities
    Provides low-latency streaming with exactly-once semantics
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self._running = False

    async def connect(self):
        """Establish Redis connection"""
        try:
            self.redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50
            )
            await self.redis.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")

    async def create_consumer_group(
        self,
        stream_name: str,
        group_name: str,
        start_id: str = "0"
    ):
        """Create a consumer group for a stream"""
        try:
            await self.redis.xgroup_create(
                stream_name,
                group_name,
                id=start_id,
                mkstream=True
            )
            logger.info(f"Created consumer group '{group_name}' for stream '{stream_name}'")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group '{group_name}' already exists for '{stream_name}'")
            else:
                raise

    async def produce(
        self,
        stream_name: str,
        data: Dict[str, Any],
        maxlen: Optional[int] = None,
        approximate: bool = True
    ) -> str:
        """
        Produce a message to a stream

        Args:
            stream_name: Name of the stream
            data: Message data (will be JSON serialized)
            maxlen: Maximum stream length (for trimming)
            approximate: Use approximate trimming for better performance

        Returns:
            Message ID
        """
        try:
            # Add timestamp
            message = {
                **data,
                "_timestamp": time.time(),
                "_produced_at": datetime.utcnow().isoformat()
            }

            # Serialize complex objects
            serialized = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in message.items()
            }

            # Produce to stream
            message_id = await self.redis.xadd(
                stream_name,
                serialized,
                maxlen=maxlen,
                approximate=approximate
            )

            MESSAGES_PRODUCED.labels(stream_name=stream_name).inc()

            return message_id

        except Exception as e:
            STREAM_ERRORS.labels(stream_name=stream_name, error_type="produce").inc()
            logger.error(f"Error producing to {stream_name}: {e}")
            raise

    async def consume(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 1000
    ) -> List[StreamMessage]:
        """
        Consume messages from a stream using consumer group

        Args:
            stream_name: Name of the stream
            group_name: Consumer group name
            consumer_name: Unique consumer name
            count: Number of messages to read
            block_ms: Milliseconds to block waiting for messages

        Returns:
            List of StreamMessage objects
        """
        try:
            # Read new messages
            messages = await self.redis.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={stream_name: ">"},
                count=count,
                block=block_ms
            )

            result = []
            if messages:
                for stream, msg_list in messages:
                    for msg_id, data in msg_list:
                        # Parse message
                        parsed_data = {}
                        for key, value in data.items():
                            try:
                                # Try to parse JSON
                                parsed_data[key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                parsed_data[key] = value

                        timestamp = float(parsed_data.get("_timestamp", time.time()))

                        result.append(StreamMessage(
                            stream_name=stream,
                            message_id=msg_id,
                            data=parsed_data,
                            timestamp=timestamp
                        ))

                        MESSAGES_CONSUMED.labels(
                            stream_name=stream,
                            consumer_group=group_name
                        ).inc()

            return result

        except Exception as e:
            STREAM_ERRORS.labels(stream_name=stream_name, error_type="consume").inc()
            logger.error(f"Error consuming from {stream_name}: {e}")
            raise

    async def ack(self, stream_name: str, group_name: str, *message_ids: str):
        """Acknowledge message processing"""
        try:
            await self.redis.xack(stream_name, group_name, *message_ids)
        except Exception as e:
            logger.error(f"Error acknowledging messages: {e}")
            raise

    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get information about a stream"""
        try:
            info = await self.redis.xinfo_stream(stream_name)
            return info
        except Exception as e:
            logger.error(f"Error getting stream info for {stream_name}: {e}")
            return {}

    async def get_consumer_group_info(
        self,
        stream_name: str,
        group_name: str
    ) -> Dict[str, Any]:
        """Get consumer group information including lag"""
        try:
            groups = await self.redis.xinfo_groups(stream_name)
            for group in groups:
                if group['name'] == group_name:
                    return group
            return {}
        except Exception as e:
            logger.error(f"Error getting group info: {e}")
            return {}

    async def monitor_lag(self, stream_name: str, group_name: str):
        """Monitor and report stream lag"""
        try:
            # Get stream length
            stream_info = await self.get_stream_info(stream_name)
            stream_length = stream_info.get('length', 0)

            # Get consumer group pending messages
            group_info = await self.get_consumer_group_info(stream_name, group_name)
            pending = group_info.get('pending', 0)

            # Calculate lag (simplified - in production use last-delivered-id)
            lag_estimate = pending

            STREAM_LAG.labels(
                stream_name=stream_name,
                consumer_group=group_name
            ).set(lag_estimate)

            return lag_estimate

        except Exception as e:
            logger.error(f"Error monitoring lag: {e}")
            return 0

    async def start_consumer_loop(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[[StreamMessage], Any],
        count: int = 10,
        block_ms: int = 1000
    ):
        """
        Start a consumer loop that processes messages with a handler

        Args:
            stream_name: Stream to consume from
            group_name: Consumer group
            consumer_name: Unique consumer identifier
            handler: Async function to process messages
            count: Messages per batch
            block_ms: Block time
        """
        self._running = True
        logger.info(f"Starting consumer loop for {stream_name} (group: {group_name})")

        # Create consumer group if needed
        await self.create_consumer_group(stream_name, group_name)

        while self._running:
            try:
                # Consume messages
                messages = await self.consume(
                    stream_name,
                    group_name,
                    consumer_name,
                    count=count,
                    block_ms=block_ms
                )

                # Process messages
                for msg in messages:
                    start_time = time.time()

                    try:
                        await handler(msg)

                        # Acknowledge successful processing
                        await self.ack(stream_name, group_name, msg.message_id)

                        # Record processing time
                        processing_time = (time.time() - start_time) * 1000
                        PROCESSING_TIME.labels(stream_name=stream_name).observe(processing_time)

                    except Exception as e:
                        logger.error(f"Error processing message {msg.message_id}: {e}")
                        STREAM_ERRORS.labels(
                            stream_name=stream_name,
                            error_type="handler"
                        ).inc()

                # Monitor lag periodically
                await self.monitor_lag(stream_name, group_name)

            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)

        logger.info(f"Consumer loop stopped for {stream_name}")

    async def stop_consumer_loop(self):
        """Stop the consumer loop"""
        self._running = False
