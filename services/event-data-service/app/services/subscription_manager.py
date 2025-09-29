"""Event subscription system for real-time notifications to strategy services."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import httpx
from pydantic import BaseModel, ConfigDict


logger = logging.getLogger(__name__)


class SubscriptionStatus(str, Enum):
    """Subscription status enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    """Event types for subscriptions."""
    EVENT_CREATED = "event.created"
    EVENT_UPDATED = "event.updated" 
    EVENT_IMPACT_CHANGED = "event.impact_changed"
    EVENT_STATUS_CHANGED = "event.status_changed"
    HEADLINE_LINKED = "headline.linked"
    CLUSTER_FORMED = "cluster.formed"
    ALL = "*"


@dataclass
class SubscriptionFilter:
    """Filter criteria for event subscriptions."""
    symbols: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    min_impact_score: Optional[int] = None
    max_impact_score: Optional[int] = None
    event_types: Optional[List[EventType]] = None
    statuses: Optional[List[str]] = None
    
    def matches_event(self, event_data: Dict[str, Any], event_type: EventType) -> bool:
        """Check if event matches subscription filter criteria."""
        # Check event type filter
        if self.event_types and EventType.ALL not in self.event_types:
            if event_type not in self.event_types:
                return False
        
        # Check symbol filter
        if self.symbols:
            event_symbol = event_data.get("symbol", "").upper()
            if event_symbol not in [s.upper() for s in self.symbols]:
                return False
        
        # Check category filter
        if self.categories:
            event_category = event_data.get("category", "")
            if event_category not in self.categories:
                return False
        
        # Check impact score filter
        impact_score = event_data.get("impact_score")
        if impact_score is not None:
            if self.min_impact_score is not None and impact_score < self.min_impact_score:
                return False
            if self.max_impact_score is not None and impact_score > self.max_impact_score:
                return False
        
        # Check status filter
        if self.statuses:
            event_status = event_data.get("status", "")
            if event_status not in self.statuses:
                return False
        
        return True


@dataclass
class Subscription:
    """Event subscription configuration."""
    id: str
    service_name: str
    webhook_url: str
    filters: SubscriptionFilter
    headers: Optional[Dict[str, str]] = None
    timeout: float = 5.0
    retry_count: int = 3
    retry_delay: float = 1.0
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_notification: Optional[datetime] = None
    failure_count: int = 0
    max_failures: int = 5
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


class SubscriptionRequest(BaseModel):
    """Pydantic model for subscription creation requests."""
    model_config = ConfigDict(use_enum_values=True)
    
    service_name: str
    webhook_url: str
    symbols: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    min_impact_score: Optional[int] = None
    max_impact_score: Optional[int] = None
    event_types: Optional[List[EventType]] = None
    statuses: Optional[List[str]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: float = 5.0
    retry_count: int = 3
    retry_delay: float = 1.0


class SubscriptionResponse(BaseModel):
    """Pydantic model for subscription responses."""
    model_config = ConfigDict(use_enum_values=True)
    
    id: str
    service_name: str
    webhook_url: str
    filters: Dict[str, Any]
    status: SubscriptionStatus
    created_at: datetime
    updated_at: datetime
    last_notification: Optional[datetime]
    failure_count: int


@dataclass
class NotificationPayload:
    """Notification payload sent to subscribers."""
    subscription_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    service_name: str = "event-data-service"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "subscription_id": self.subscription_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "service_name": self.service_name
        }


class EventSubscriptionManager:
    """Manages event subscriptions and notifications for strategy services."""
    
    def __init__(self, service_name: str = "event-data-service"):
        self.service_name = service_name
        self.subscriptions: Dict[str, Subscription] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._notification_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the subscription manager and notification worker."""
        if self._running:
            return
            
        self._running = True
        self._http_client = httpx.AsyncClient()
        self._worker_task = asyncio.create_task(self._notification_worker())
        logger.info("Event subscription manager started")
        
    async def stop(self):
        """Stop the subscription manager and cleanup resources."""
        if not self._running:
            return
            
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
                
        if self._http_client:
            await self._http_client.aclose()
            
        logger.info("Event subscription manager stopped")
        
    def create_subscription(self, request: SubscriptionRequest) -> Subscription:
        """Create a new event subscription."""
        subscription_id = str(uuid.uuid4())
        
        filters = SubscriptionFilter(
            symbols=request.symbols,
            categories=request.categories,
            min_impact_score=request.min_impact_score,
            max_impact_score=request.max_impact_score,
            event_types=request.event_types or [EventType.ALL],
            statuses=request.statuses
        )
        
        subscription = Subscription(
            id=subscription_id,
            service_name=request.service_name,
            webhook_url=request.webhook_url,
            filters=filters,
            headers=request.headers or {},
            timeout=request.timeout,
            retry_count=request.retry_count,
            retry_delay=request.retry_delay
        )
        
        self.subscriptions[subscription_id] = subscription
        logger.info(f"Created subscription {subscription_id} for service {request.service_name}")
        return subscription
        
    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID."""
        return self.subscriptions.get(subscription_id)
        
    def list_subscriptions(self, service_name: Optional[str] = None) -> List[Subscription]:
        """List subscriptions, optionally filtered by service name."""
        subscriptions = list(self.subscriptions.values())
        if service_name:
            subscriptions = [s for s in subscriptions if s.service_name == service_name]
        return subscriptions
        
    def update_subscription(self, subscription_id: str, **updates) -> Optional[Subscription]:
        """Update subscription configuration."""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return None
            
        for key, value in updates.items():
            if hasattr(subscription, key):
                setattr(subscription, key, value)
        
        subscription.updated_at = datetime.now(timezone.utc)
        logger.info(f"Updated subscription {subscription_id}")
        return subscription
        
    def delete_subscription(self, subscription_id: str) -> bool:
        """Delete a subscription."""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            logger.info(f"Deleted subscription {subscription_id}")
            return True
        return False
        
    async def notify_event(self, event_data: Dict[str, Any], event_type: EventType):
        """Queue event notification for all matching subscriptions."""
        if not self._running:
            return
            
        matching_subscriptions = []
        for subscription in self.subscriptions.values():
            if (subscription.status == SubscriptionStatus.ACTIVE and 
                subscription.filters.matches_event(event_data, event_type)):
                matching_subscriptions.append(subscription)
        
        if matching_subscriptions:
            logger.info(f"Found {len(matching_subscriptions)} matching subscriptions for {event_type}")
            
        for subscription in matching_subscriptions:
            payload = NotificationPayload(
                subscription_id=subscription.id,
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                data=event_data
            )
            await self._notification_queue.put((subscription, payload))
            
    async def _notification_worker(self):
        """Background worker to process notification queue."""
        logger.info("Notification worker started")
        
        while self._running:
            try:
                # Wait for notification with timeout to allow graceful shutdown
                try:
                    subscription, payload = await asyncio.wait_for(
                        self._notification_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                    
                await self._send_notification(subscription, payload)
                
            except Exception as e:
                logger.error(f"Error in notification worker: {e}")
                await asyncio.sleep(1)
                
        logger.info("Notification worker stopped")
        
    async def _send_notification(self, subscription: Subscription, payload: NotificationPayload):
        """Send notification to a specific subscription."""
        if not self._http_client:
            logger.error("HTTP client not initialized")
            return
            
        headers = {
            "Content-Type": "application/json",
            "X-Event-Service": self.service_name,
            "X-Subscription-ID": subscription.id,
            **subscription.headers
        }
        
        for attempt in range(subscription.retry_count + 1):
            try:
                response = await self._http_client.post(
                    subscription.webhook_url,
                    json=payload.to_dict(),
                    headers=headers,
                    timeout=subscription.timeout
                )
                
                if response.status_code == 200:
                    subscription.last_notification = datetime.now(timezone.utc)
                    subscription.failure_count = 0
                    logger.debug(f"Notification sent successfully to subscription {subscription.id}")
                    return
                else:
                    logger.warning(f"Webhook returned status {response.status_code} for subscription {subscription.id}")
                    
            except Exception as e:
                logger.warning(f"Notification attempt {attempt + 1} failed for subscription {subscription.id}: {e}")
                
                if attempt < subscription.retry_count:
                    await asyncio.sleep(subscription.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # All attempts failed
        subscription.failure_count += 1
        if subscription.failure_count >= subscription.max_failures:
            subscription.status = SubscriptionStatus.FAILED
            logger.error(f"Subscription {subscription.id} marked as failed after {subscription.failure_count} failures")
        else:
            logger.warning(f"Notification failed for subscription {subscription.id}, failure count: {subscription.failure_count}")


def build_subscription_manager(service_name: str = "event-data-service") -> EventSubscriptionManager:
    """Build and configure the event subscription manager."""
    return EventSubscriptionManager(service_name)