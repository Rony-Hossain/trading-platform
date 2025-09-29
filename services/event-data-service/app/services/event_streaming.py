"""Event Streaming Architecture for Real-Time Processing

High-performance event streaming system that provides real-time event distribution,
processing, and analytics capabilities with multiple streaming backends.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
from collections import defaultdict, deque

import aioredis
from aioredis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EventORM, EventHeadlineORM

logger = logging.getLogger(__name__)


class StreamBackend(str, Enum):
    """Supported streaming backends."""
    REDIS_STREAMS = "redis_streams"
    REDIS_PUBSUB = "redis_pubsub"
    WEBHOOK = "webhook"
    SERVER_SENT_EVENTS = "sse"
    WEBSOCKET = "websocket"


class EventType(str, Enum):
    """Real-time event types."""
    EVENT_CREATED = "event.created"
    EVENT_UPDATED = "event.updated"
    EVENT_DELETED = "event.deleted"
    EVENT_SCHEDULED = "event.scheduled"
    EVENT_OCCURRED = "event.occurred"
    EVENT_CANCELLED = "event.cancelled"
    HEADLINE_CREATED = "headline.created"
    MARKET_ALERT = "market.alert"
    SENTIMENT_UPDATE = "sentiment.update"
    PRICE_ALERT = "price.alert"
    SYSTEM_STATUS = "system.status"


class StreamingMode(str, Enum):
    """Streaming delivery modes."""
    AT_LEAST_ONCE = "at_least_once"  # May deliver duplicates
    AT_MOST_ONCE = "at_most_once"    # May lose messages
    EXACTLY_ONCE = "exactly_once"     # Guaranteed delivery (expensive)


@dataclass
class StreamEvent:
    """Structured event for streaming."""
    id: str
    type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    ttl_seconds: Optional[int] = None
    priority: int = 1  # 1=high, 2=normal, 3=low


@dataclass
class StreamConsumer:
    """Stream consumer configuration."""
    id: str
    name: str
    topics: List[str]
    filters: Optional[Dict[str, Any]] = None
    batch_size: int = 1
    max_wait_ms: int = 1000
    auto_ack: bool = True
    dead_letter_queue: bool = True
    retry_count: int = 3
    callback: Optional[Callable] = None


@dataclass
class StreamMetrics:
    """Streaming performance metrics."""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    bytes_transferred: int = 0
    active_consumers: int = 0
    active_streams: int = 0
    average_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    last_reset: datetime = None


class EventStreamingService:
    """High-performance event streaming service for real-time processing."""
    
    def __init__(self):
        """Initialize the event streaming service."""
        # Configuration
        self.enabled = os.getenv("EVENT_STREAMING_ENABLED", "true").lower() == "true"
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_db = int(os.getenv("EVENT_STREAMING_REDIS_DB", "2"))
        self.stream_prefix = os.getenv("EVENT_STREAMING_PREFIX", "events")
        
        # Performance settings
        self.max_stream_length = int(os.getenv("EVENT_STREAMING_MAX_LENGTH", "10000"))
        self.batch_size = int(os.getenv("EVENT_STREAMING_BATCH_SIZE", "100"))
        self.consumer_timeout = int(os.getenv("EVENT_STREAMING_TIMEOUT_MS", "5000"))
        self.retention_hours = int(os.getenv("EVENT_STREAMING_RETENTION_HOURS", "24"))
        
        # WebSocket settings
        self.websocket_max_connections = int(os.getenv("WS_MAX_CONNECTIONS", "1000"))
        self.sse_max_connections = int(os.getenv("SSE_MAX_CONNECTIONS", "500"))
        
        # Backends
        self.enabled_backends = self._parse_backends()
        
        # State
        self.redis: Optional[Redis] = None
        self.consumers: Dict[str, StreamConsumer] = {}
        self.websocket_connections: Set[Any] = set()
        self.sse_connections: Set[Any] = set()
        self.metrics = StreamMetrics()
        self._consumer_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Message deduplication
        self.message_cache: Dict[str, datetime] = {}
        self.cache_cleanup_interval = 300  # 5 minutes
        
        logger.info(f"EventStreamingService initialized (enabled={self.enabled})")
    
    def _parse_backends(self) -> Set[StreamBackend]:
        """Parse enabled streaming backends from configuration."""
        backends_str = os.getenv("EVENT_STREAMING_BACKENDS", "redis_streams,websocket")
        backends = set()
        
        for backend in backends_str.split(","):
            backend = backend.strip().lower()
            try:
                backends.add(StreamBackend(backend))
            except ValueError:
                logger.warning(f"Unknown streaming backend: {backend}")
        
        return backends
    
    async def start(self):
        """Start the event streaming service."""
        if not self.enabled:
            logger.info("Event streaming service disabled")
            return
        
        try:
            # Initialize Redis connection
            if StreamBackend.REDIS_STREAMS in self.enabled_backends or StreamBackend.REDIS_PUBSUB in self.enabled_backends:
                self.redis = await aioredis.from_url(
                    self.redis_url,
                    db=self.redis_db,
                    decode_responses=True
                )
                await self.redis.ping()
                logger.info("Connected to Redis for event streaming")
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())
            self._metrics_task = asyncio.create_task(self._metrics_worker())
            
            # Initialize metrics
            self.metrics.last_reset = datetime.utcnow()
            
            logger.info(f"Event streaming service started with backends: {', '.join(b.value for b in self.enabled_backends)}")
            
        except Exception as e:
            logger.error(f"Failed to start event streaming service: {e}")
            self.enabled = False
    
    async def stop(self):
        """Stop the event streaming service."""
        logger.info("Stopping event streaming service")
        
        # Stop consumer tasks
        for task in self._consumer_tasks.values():
            task.cancel()
        
        # Stop background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        
        # Close connections
        if self.redis:
            await self.redis.close()
        
        logger.info("Event streaming service stopped")
    
    async def publish_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str = "event_service",
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        priority: int = 1
    ) -> str:
        """Publish an event to all configured streaming backends."""
        
        if not self.enabled:
            return ""
        
        # Create stream event
        event_id = str(uuid.uuid4())
        stream_event = StreamEvent(
            id=event_id,
            type=event_type,
            timestamp=datetime.utcnow(),
            source=source,
            data=data,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds,
            priority=priority
        )
        
        # Check for duplicates
        content_hash = self._hash_event_content(stream_event)
        if self._is_duplicate(content_hash):
            logger.debug(f"Duplicate event detected, skipping: {event_id}")
            return event_id
        
        # Publish to all enabled backends
        publish_tasks = []
        
        if StreamBackend.REDIS_STREAMS in self.enabled_backends:
            publish_tasks.append(self._publish_to_redis_stream(stream_event))
        
        if StreamBackend.REDIS_PUBSUB in self.enabled_backends:
            publish_tasks.append(self._publish_to_redis_pubsub(stream_event))
        
        if StreamBackend.WEBSOCKET in self.enabled_backends:
            publish_tasks.append(self._publish_to_websockets(stream_event))
        
        if StreamBackend.SERVER_SENT_EVENTS in self.enabled_backends:
            publish_tasks.append(self._publish_to_sse(stream_event))
        
        # Execute all publishes concurrently
        if publish_tasks:
            results = await asyncio.gather(*publish_tasks, return_exceptions=True)
            
            # Count successful publishes
            success_count = sum(1 for result in results if not isinstance(result, Exception))
            if success_count > 0:
                self.metrics.messages_sent += 1
                self.metrics.bytes_transferred += len(json.dumps(asdict(stream_event)))
            
            # Log any failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    backend = list(self.enabled_backends)[i]
                    logger.error(f"Failed to publish to {backend}: {result}")
                    self.metrics.messages_failed += 1
        
        return event_id
    
    def _hash_event_content(self, event: StreamEvent) -> str:
        """Generate hash for event content to detect duplicates."""
        content = f"{event.type}:{event.source}:{json.dumps(event.data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_duplicate(self, content_hash: str) -> bool:
        """Check if event is a duplicate within time window."""
        now = datetime.utcnow()
        
        # Clean old entries
        cutoff = now - timedelta(seconds=60)  # 1 minute window
        self.message_cache = {
            h: ts for h, ts in self.message_cache.items() 
            if ts > cutoff
        }
        
        # Check for duplicate
        if content_hash in self.message_cache:
            return True
        
        # Add to cache
        self.message_cache[content_hash] = now
        return False
    
    async def _publish_to_redis_stream(self, event: StreamEvent):
        """Publish event to Redis Streams."""
        if not self.redis:
            return
        
        try:
            stream_name = f"{self.stream_prefix}:{event.type.value}"
            
            # Prepare message data
            message_data = {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "data": json.dumps(event.data),
                "metadata": json.dumps(event.metadata or {}),
                "priority": str(event.priority)
            }
            
            if event.ttl_seconds:
                message_data["ttl"] = str(event.ttl_seconds)
            
            # Add to stream with automatic trimming
            await self.redis.xadd(
                stream_name,
                message_data,
                maxlen=self.max_stream_length,
                approximate=True
            )
            
            logger.debug(f"Published event {event.id} to Redis stream {stream_name}")
            
        except Exception as e:
            logger.error(f"Failed to publish to Redis stream: {e}")
            raise
    
    async def _publish_to_redis_pubsub(self, event: StreamEvent):
        """Publish event to Redis Pub/Sub."""
        if not self.redis:
            return
        
        try:
            channel = f"{self.stream_prefix}:pubsub:{event.type.value}"
            
            message = {
                "id": event.id,
                "type": event.type.value,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "data": event.data,
                "metadata": event.metadata or {},
                "priority": event.priority
            }
            
            await self.redis.publish(channel, json.dumps(message, default=str))
            
            logger.debug(f"Published event {event.id} to Redis pubsub {channel}")
            
        except Exception as e:
            logger.error(f"Failed to publish to Redis pubsub: {e}")
            raise
    
    async def _publish_to_websockets(self, event: StreamEvent):
        """Publish event to active WebSocket connections."""
        if not self.websocket_connections:
            return
        
        try:
            message = {
                "id": event.id,
                "type": event.type.value,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "data": event.data,
                "metadata": event.metadata or {}
            }
            
            message_text = json.dumps(message, default=str)
            
            # Send to all active WebSocket connections
            disconnected = set()
            for ws in self.websocket_connections.copy():
                try:
                    await ws.send_text(message_text)
                except Exception as e:
                    logger.debug(f"WebSocket connection failed: {e}")
                    disconnected.add(ws)
            
            # Remove disconnected connections
            self.websocket_connections -= disconnected
            
            logger.debug(f"Published event {event.id} to {len(self.websocket_connections)} WebSocket connections")
            
        except Exception as e:
            logger.error(f"Failed to publish to WebSockets: {e}")
            raise
    
    async def _publish_to_sse(self, event: StreamEvent):
        """Publish event to Server-Sent Events connections."""
        if not self.sse_connections:
            return
        
        try:
            message = {
                "id": event.id,
                "type": event.type.value,
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "data": event.data,
                "metadata": event.metadata or {}
            }
            
            # Format as SSE
            sse_data = f"id: {event.id}\n"
            sse_data += f"event: {event.type.value}\n"
            sse_data += f"data: {json.dumps(message, default=str)}\n\n"
            
            # Send to all active SSE connections
            disconnected = set()
            for sse in self.sse_connections.copy():
                try:
                    await sse.send(sse_data)
                except Exception as e:
                    logger.debug(f"SSE connection failed: {e}")
                    disconnected.add(sse)
            
            # Remove disconnected connections
            self.sse_connections -= disconnected
            
            logger.debug(f"Published event {event.id} to {len(self.sse_connections)} SSE connections")
            
        except Exception as e:
            logger.error(f"Failed to publish to SSE: {e}")
            raise
    
    async def create_consumer(
        self,
        consumer_id: str,
        name: str,
        topics: List[str],
        callback: Callable,
        filters: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        auto_ack: bool = True
    ) -> bool:
        """Create a new stream consumer."""
        
        if consumer_id in self.consumers:
            logger.warning(f"Consumer {consumer_id} already exists")
            return False
        
        try:
            consumer = StreamConsumer(
                id=consumer_id,
                name=name,
                topics=topics,
                filters=filters,
                batch_size=batch_size,
                auto_ack=auto_ack,
                callback=callback
            )
            
            self.consumers[consumer_id] = consumer
            
            # Start consumer task for Redis Streams
            if StreamBackend.REDIS_STREAMS in self.enabled_backends:
                task = asyncio.create_task(self._consumer_worker(consumer))
                self._consumer_tasks[consumer_id] = task
            
            logger.info(f"Created consumer {consumer_id} for topics: {topics}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create consumer {consumer_id}: {e}")
            return False
    
    async def _consumer_worker(self, consumer: StreamConsumer):
        """Background worker for consuming Redis Streams."""
        if not self.redis:
            return
        
        try:
            # Create consumer group for each topic
            for topic in consumer.topics:
                stream_name = f"{self.stream_prefix}:{topic}"
                group_name = f"group_{consumer.id}"
                
                try:
                    await self.redis.xgroup_create(
                        stream_name, 
                        group_name, 
                        id="0", 
                        mkstream=True
                    )
                except Exception:
                    # Group already exists
                    pass
            
            # Consumer loop
            while True:
                try:
                    # Prepare streams for reading
                    streams = {}
                    for topic in consumer.topics:
                        stream_name = f"{self.stream_prefix}:{topic}"
                        streams[stream_name] = ">"
                    
                    # Read messages
                    messages = await self.redis.xreadgroup(
                        f"group_{consumer.id}",
                        consumer.name,
                        streams,
                        count=consumer.batch_size,
                        block=consumer.max_wait_ms
                    )
                    
                    if messages:
                        await self._process_consumer_messages(consumer, messages)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Consumer {consumer.id} error: {e}")
                    await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Consumer worker {consumer.id} failed: {e}")
    
    async def _process_consumer_messages(self, consumer: StreamConsumer, messages):
        """Process messages for a consumer."""
        processed_messages = []
        
        for stream_name, stream_messages in messages:
            topic = stream_name.split(":")[-1]
            
            for message_id, fields in stream_messages:
                try:
                    # Parse message
                    event_data = {
                        "id": fields.get("id"),
                        "timestamp": fields.get("timestamp"),
                        "source": fields.get("source"),
                        "data": json.loads(fields.get("data", "{}")),
                        "metadata": json.loads(fields.get("metadata", "{}")),
                        "priority": int(fields.get("priority", "1"))
                    }
                    
                    # Apply filters
                    if self._message_matches_filters(event_data, consumer.filters):
                        # Call consumer callback
                        if consumer.callback:
                            await consumer.callback(topic, event_data)
                        
                        processed_messages.append((stream_name, message_id))
                        self.metrics.messages_received += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process message {message_id}: {e}")
                    self.metrics.messages_failed += 1
        
        # Acknowledge processed messages
        if consumer.auto_ack and processed_messages:
            for stream_name, message_id in processed_messages:
                try:
                    await self.redis.xack(
                        stream_name,
                        f"group_{consumer.id}",
                        message_id
                    )
                except Exception as e:
                    logger.error(f"Failed to ack message {message_id}: {e}")
    
    def _message_matches_filters(self, message: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Check if message matches consumer filters."""
        if not filters:
            return True
        
        for key, expected_value in filters.items():
            if key == "source":
                if message.get("source") != expected_value:
                    return False
            elif key == "priority":
                if message.get("priority", 1) > expected_value:
                    return False
            elif key == "data_contains":
                data = message.get("data", {})
                for data_key, data_value in expected_value.items():
                    if data.get(data_key) != data_value:
                        return False
        
        return True
    
    async def add_websocket_connection(self, websocket):
        """Add a WebSocket connection for real-time updates."""
        if len(self.websocket_connections) < self.websocket_max_connections:
            self.websocket_connections.add(websocket)
            logger.debug(f"Added WebSocket connection, total: {len(self.websocket_connections)}")
            return True
        else:
            logger.warning("WebSocket connection limit reached")
            return False
    
    def remove_websocket_connection(self, websocket):
        """Remove a WebSocket connection."""
        self.websocket_connections.discard(websocket)
        logger.debug(f"Removed WebSocket connection, total: {len(self.websocket_connections)}")
    
    async def add_sse_connection(self, sse_response):
        """Add a Server-Sent Events connection for real-time updates."""
        if len(self.sse_connections) < self.sse_max_connections:
            self.sse_connections.add(sse_response)
            logger.debug(f"Added SSE connection, total: {len(self.sse_connections)}")
            return True
        else:
            logger.warning("SSE connection limit reached")
            return False
    
    def remove_sse_connection(self, sse_response):
        """Remove a Server-Sent Events connection."""
        self.sse_connections.discard(sse_response)
        logger.debug(f"Removed SSE connection, total: {len(self.sse_connections)}")
    
    async def get_stream_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics."""
        stats = {
            "enabled": self.enabled,
            "backends": [b.value for b in self.enabled_backends],
            "metrics": asdict(self.metrics),
            "connections": {
                "websockets": len(self.websocket_connections),
                "sse": len(self.sse_connections),
                "consumers": len(self.consumers)
            },
            "configuration": {
                "max_stream_length": self.max_stream_length,
                "batch_size": self.batch_size,
                "retention_hours": self.retention_hours,
                "consumer_timeout": self.consumer_timeout
            }
        }
        
        # Add Redis-specific stats
        if self.redis:
            try:
                redis_info = await self.redis.info("memory")
                stats["redis"] = {
                    "memory_usage": redis_info.get("used_memory", 0),
                    "connected": True
                }
            except Exception:
                stats["redis"] = {"connected": False}
        
        return stats
    
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks."""
        while True:
            try:
                await asyncio.sleep(self.cache_cleanup_interval)
                
                # Clean up old streams
                if self.redis and StreamBackend.REDIS_STREAMS in self.enabled_backends:
                    await self._cleanup_old_streams()
                
                # Update metrics
                self.metrics.active_consumers = len(self.consumers)
                self.metrics.active_streams = len(self.enabled_backends)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    async def _metrics_worker(self):
        """Background worker for metrics calculation."""
        last_sent = 0
        last_time = time.time()
        
        while True:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                current_time = time.time()
                current_sent = self.metrics.messages_sent
                
                # Calculate throughput
                time_diff = current_time - last_time
                if time_diff > 0:
                    self.metrics.throughput_per_second = (current_sent - last_sent) / time_diff
                
                last_sent = current_sent
                last_time = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics worker error: {e}")
    
    async def _cleanup_old_streams(self):
        """Clean up old Redis streams based on retention policy."""
        try:
            # Get all stream keys
            pattern = f"{self.stream_prefix}:*"
            stream_keys = await self.redis.keys(pattern)
            
            cutoff_time = int((datetime.utcnow() - timedelta(hours=self.retention_hours)).timestamp() * 1000)
            
            for stream_key in stream_keys:
                try:
                    # Trim old messages
                    await self.redis.xtrim(
                        stream_key,
                        minid=cutoff_time,
                        approximate=True
                    )
                except Exception as e:
                    logger.debug(f"Failed to trim stream {stream_key}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old streams: {e}")


def build_event_streaming_service() -> EventStreamingService:
    """Factory function to create event streaming service instance."""
    return EventStreamingService()