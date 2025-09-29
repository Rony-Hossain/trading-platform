"""Event lifecycle tracking system for monitoring event progression and impact analysis."""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy import func, select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EventORM, EventHeadlineORM

logger = logging.getLogger(__name__)


class EventStatus(str, Enum):
    """Event lifecycle status enumeration."""
    SCHEDULED = "scheduled"
    OCCURRED = "occurred"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"
    IMPACT_ANALYZED = "impact_analyzed"


class LifecycleStage(str, Enum):
    """Event lifecycle stages."""
    PRE_EVENT = "pre_event"
    EVENT_WINDOW = "event_window"
    POST_EVENT = "post_event"
    ANALYSIS_COMPLETE = "analysis_complete"


@dataclass
class ImpactMetrics:
    """Impact analysis metrics for an event."""
    event_id: str
    symbol: str
    pre_event_price: Optional[float] = None
    event_time_price: Optional[float] = None
    post_event_price_1h: Optional[float] = None
    post_event_price_1d: Optional[float] = None
    post_event_price_3d: Optional[float] = None
    max_move_pct: Optional[float] = None
    min_move_pct: Optional[float] = None
    volume_change_pct: Optional[float] = None
    volatility_spike: Optional[float] = None
    headline_sentiment: Optional[float] = None
    headline_count: int = 0
    predicted_impact: Optional[float] = None
    actual_impact: Optional[float] = None
    accuracy_score: Optional[float] = None
    analysis_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.analysis_timestamp:
            data["analysis_timestamp"] = self.analysis_timestamp.isoformat()
        return data


@dataclass
class LifecycleEvent:
    """Event lifecycle tracking record."""
    event_id: str
    symbol: str
    title: str
    category: str
    scheduled_at: datetime
    current_status: EventStatus
    current_stage: LifecycleStage
    status_history: List[Dict[str, Any]]
    impact_metrics: Optional[ImpactMetrics] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["scheduled_at"] = self.scheduled_at.isoformat()
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()
        if self.impact_metrics:
            data["impact_metrics"] = self.impact_metrics.to_dict()
        return data


class EventLifecycleTracker:
    """Tracks event lifecycle progression and analyzes impact."""
    
    def __init__(self, session_factory, config: Optional[Dict[str, Any]] = None):
        self.session_factory = session_factory
        self.config = config or {}
        self._lifecycle_cache: Dict[str, LifecycleEvent] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Configuration
        self.monitor_interval_minutes = int(os.getenv("LIFECYCLE_MONITOR_INTERVAL_MINUTES", "15"))
        self.analysis_delay_hours = int(os.getenv("LIFECYCLE_ANALYSIS_DELAY_HOURS", "24"))
        self.pre_event_window_hours = int(os.getenv("LIFECYCLE_PRE_EVENT_WINDOW_HOURS", "4"))
        self.post_event_window_hours = int(os.getenv("LIFECYCLE_POST_EVENT_WINDOW_HOURS", "72"))
        
    async def start(self):
        """Start the lifecycle tracking system."""
        if self._running:
            return
            
        self._running = True
        await self._initialize_lifecycle_tracking()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Event lifecycle tracker started")
        
    async def stop(self):
        """Stop the lifecycle tracking system."""
        if not self._running:
            return
            
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Event lifecycle tracker stopped")
        
    async def _initialize_lifecycle_tracking(self):
        """Initialize lifecycle tracking for existing events."""
        async with self.session_factory() as session:
            # Get all scheduled events that need lifecycle tracking
            stmt = select(EventORM).where(
                and_(
                    EventORM.status == EventStatus.SCHEDULED.value,
                    EventORM.scheduled_at > datetime.now(timezone.utc) - timedelta(days=7),
                    EventORM.scheduled_at < datetime.now(timezone.utc) + timedelta(days=30)
                )
            )
            
            result = await session.execute(stmt)
            events = result.scalars().all()
            
            for event in events:
                await self._create_lifecycle_event(event)
                
            logger.info(f"Initialized lifecycle tracking for {len(events)} events")
            
    async def _create_lifecycle_event(self, event: EventORM) -> LifecycleEvent:
        """Create lifecycle tracking for an event."""
        now = datetime.now(timezone.utc)
        
        # Determine current stage
        if event.scheduled_at > now + timedelta(hours=self.pre_event_window_hours):
            stage = LifecycleStage.PRE_EVENT
        elif event.scheduled_at > now - timedelta(hours=1):
            stage = LifecycleStage.EVENT_WINDOW
        elif event.scheduled_at > now - timedelta(hours=self.post_event_window_hours):
            stage = LifecycleStage.POST_EVENT
        else:
            stage = LifecycleStage.ANALYSIS_COMPLETE
            
        lifecycle_event = LifecycleEvent(
            event_id=event.id,
            symbol=event.symbol,
            title=event.title,
            category=event.category,
            scheduled_at=event.scheduled_at,
            current_status=EventStatus(event.status),
            current_stage=stage,
            status_history=[{
                "status": event.status,
                "timestamp": now.isoformat(),
                "reason": "lifecycle_tracking_initialized"
            }],
            created_at=now,
            updated_at=now
        )
        
        self._lifecycle_cache[event.id] = lifecycle_event
        return lifecycle_event
        
    async def _monitoring_loop(self):
        """Main monitoring loop for lifecycle tracking."""
        logger.info("Lifecycle monitoring loop started")
        
        while self._running:
            try:
                await self._update_lifecycle_stages()
                await self._detect_status_changes()
                await self._analyze_completed_events()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitor_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in lifecycle monitoring loop: {e}")
                await asyncio.sleep(60)  # Short delay on error
                
        logger.info("Lifecycle monitoring loop stopped")
        
    async def _update_lifecycle_stages(self):
        """Update lifecycle stages based on current time."""
        now = datetime.now(timezone.utc)
        updates = 0
        
        for event_id, lifecycle_event in list(self._lifecycle_cache.items()):
            old_stage = lifecycle_event.current_stage
            new_stage = self._calculate_lifecycle_stage(lifecycle_event.scheduled_at, now)
            
            if new_stage != old_stage:
                lifecycle_event.current_stage = new_stage
                lifecycle_event.updated_at = now
                lifecycle_event.status_history.append({
                    "stage": new_stage.value,
                    "timestamp": now.isoformat(),
                    "reason": "stage_progression"
                })
                updates += 1
                
                logger.info(f"Event {event_id} ({lifecycle_event.symbol}) progressed from {old_stage.value} to {new_stage.value}")
                
        if updates > 0:
            logger.info(f"Updated lifecycle stages for {updates} events")
            
    def _calculate_lifecycle_stage(self, scheduled_at: datetime, current_time: datetime) -> LifecycleStage:
        """Calculate the current lifecycle stage for an event."""
        time_to_event = (scheduled_at - current_time).total_seconds() / 3600  # Hours
        time_since_event = (current_time - scheduled_at).total_seconds() / 3600  # Hours
        
        if time_to_event > self.pre_event_window_hours:
            return LifecycleStage.PRE_EVENT
        elif time_to_event > -1:  # Event window: 4 hours before to 1 hour after
            return LifecycleStage.EVENT_WINDOW
        elif time_since_event < self.post_event_window_hours:
            return LifecycleStage.POST_EVENT
        else:
            return LifecycleStage.ANALYSIS_COMPLETE
            
    async def _detect_status_changes(self):
        """Detect status changes in the database and update lifecycle tracking."""
        event_ids = list(self._lifecycle_cache.keys())
        if not event_ids:
            return
            
        async with self.session_factory() as session:
            stmt = select(EventORM).where(EventORM.id.in_(event_ids))
            result = await session.execute(stmt)
            current_events = {event.id: event for event in result.scalars().all()}
            
            changes = 0
            for event_id, lifecycle_event in self._lifecycle_cache.items():
                current_event = current_events.get(event_id)
                if not current_event:
                    continue
                    
                db_status = EventStatus(current_event.status)
                if db_status != lifecycle_event.current_status:
                    old_status = lifecycle_event.current_status
                    lifecycle_event.current_status = db_status
                    lifecycle_event.updated_at = datetime.now(timezone.utc)
                    lifecycle_event.status_history.append({
                        "status": db_status.value,
                        "timestamp": lifecycle_event.updated_at.isoformat(),
                        "reason": "status_change_detected",
                        "previous_status": old_status.value
                    })
                    changes += 1
                    
                    logger.info(f"Event {event_id} status changed from {old_status.value} to {db_status.value}")
                    
                    # Trigger impact analysis if event occurred
                    if db_status == EventStatus.OCCURRED:
                        await self._schedule_impact_analysis(lifecycle_event)
                        
            if changes > 0:
                logger.info(f"Detected status changes for {changes} events")
                
    async def _schedule_impact_analysis(self, lifecycle_event: LifecycleEvent):
        """Schedule impact analysis for an occurred event."""
        analysis_time = lifecycle_event.scheduled_at + timedelta(hours=self.analysis_delay_hours)
        now = datetime.now(timezone.utc)
        
        if now >= analysis_time:
            # Analyze immediately if delay has passed
            await self._analyze_event_impact(lifecycle_event)
        else:
            # Schedule for later analysis
            logger.info(f"Scheduled impact analysis for event {lifecycle_event.event_id} at {analysis_time}")
            
    async def _analyze_completed_events(self):
        """Analyze events that are ready for impact analysis."""
        now = datetime.now(timezone.utc)
        
        for lifecycle_event in list(self._lifecycle_cache.values()):
            if (lifecycle_event.current_status == EventStatus.OCCURRED and 
                lifecycle_event.current_stage == LifecycleStage.ANALYSIS_COMPLETE and
                lifecycle_event.impact_metrics is None):
                
                # Check if enough time has passed for analysis
                analysis_time = lifecycle_event.scheduled_at + timedelta(hours=self.analysis_delay_hours)
                if now >= analysis_time:
                    await self._analyze_event_impact(lifecycle_event)
                    
    async def _analyze_event_impact(self, lifecycle_event: LifecycleEvent):
        """Perform comprehensive impact analysis for an event."""
        logger.info(f"Starting impact analysis for event {lifecycle_event.event_id} ({lifecycle_event.symbol})")
        
        try:
            # Initialize impact metrics
            impact_metrics = ImpactMetrics(
                event_id=lifecycle_event.event_id,
                symbol=lifecycle_event.symbol,
                analysis_timestamp=datetime.now(timezone.utc)
            )
            
            # Get original predicted impact
            async with self.session_factory() as session:
                event = await session.get(EventORM, lifecycle_event.event_id)
                if event:
                    impact_metrics.predicted_impact = event.impact_score
                    
            # Analyze price movements (placeholder - would integrate with market data service)
            await self._analyze_price_movements(impact_metrics, lifecycle_event)
            
            # Analyze headline sentiment and volume
            await self._analyze_headline_impact(impact_metrics, lifecycle_event)
            
            # Calculate overall actual impact
            impact_metrics.actual_impact = self._calculate_actual_impact(impact_metrics)
            
            # Calculate accuracy score
            if impact_metrics.predicted_impact and impact_metrics.actual_impact:
                impact_metrics.accuracy_score = self._calculate_accuracy_score(
                    impact_metrics.predicted_impact, 
                    impact_metrics.actual_impact
                )
                
            # Store impact metrics
            lifecycle_event.impact_metrics = impact_metrics
            lifecycle_event.current_status = EventStatus.IMPACT_ANALYZED
            lifecycle_event.updated_at = datetime.now(timezone.utc)
            lifecycle_event.status_history.append({
                "status": EventStatus.IMPACT_ANALYZED.value,
                "timestamp": lifecycle_event.updated_at.isoformat(),
                "reason": "impact_analysis_completed",
                "accuracy_score": impact_metrics.accuracy_score
            })
            
            # Update database
            async with self.session_factory() as session:
                event = await session.get(EventORM, lifecycle_event.event_id)
                if event:
                    event.status = EventStatus.IMPACT_ANALYZED.value
                    
                    # Add impact analysis to metadata
                    metadata = event.metadata_json or {}
                    metadata["lifecycle"] = lifecycle_event.to_dict()
                    event.metadata_json = metadata
                    
                    await session.commit()
                    
            logger.info(f"Impact analysis completed for event {lifecycle_event.event_id}. "
                       f"Predicted: {impact_metrics.predicted_impact}, "
                       f"Actual: {impact_metrics.actual_impact:.2f}, "
                       f"Accuracy: {impact_metrics.accuracy_score:.2f}")
                       
        except Exception as e:
            logger.error(f"Failed to analyze impact for event {lifecycle_event.event_id}: {e}")
            
    async def _analyze_price_movements(self, impact_metrics: ImpactMetrics, lifecycle_event: LifecycleEvent):
        """Analyze price movements around the event (placeholder implementation)."""
        # This would integrate with the market data service to get actual price data
        # For now, we'll simulate some analysis
        
        # Placeholder: simulate price movement analysis
        # In real implementation, this would fetch:
        # - Pre-event price (4 hours before)
        # - Event time price
        # - Post-event prices (1h, 1d, 3d after)
        # - Volume changes
        # - Volatility spikes
        
        logger.info(f"Analyzing price movements for {impact_metrics.symbol} around {lifecycle_event.scheduled_at}")
        
        # Simulate some impact metrics (would be real data in production)
        impact_metrics.max_move_pct = 3.5  # Placeholder
        impact_metrics.min_move_pct = -1.2  # Placeholder
        impact_metrics.volume_change_pct = 150.0  # Placeholder
        impact_metrics.volatility_spike = 2.3  # Placeholder
        
    async def _analyze_headline_impact(self, impact_metrics: ImpactMetrics, lifecycle_event: LifecycleEvent):
        """Analyze headline sentiment and volume around the event."""
        async with self.session_factory() as session:
            # Get headlines linked to this event
            headline_stmt = select(EventHeadlineORM).where(
                EventHeadlineORM.event_id == lifecycle_event.event_id
            )
            
            headline_result = await session.execute(headline_stmt)
            headlines = headline_result.scalars().all()
            
            impact_metrics.headline_count = len(headlines)
            
            if headlines:
                # Analyze headline sentiment (placeholder - would use actual sentiment analysis)
                # This would integrate with the sentiment service
                impact_metrics.headline_sentiment = 0.3  # Placeholder (positive sentiment)
                
                logger.info(f"Analyzed {len(headlines)} headlines for event {lifecycle_event.event_id}")
            
    def _calculate_actual_impact(self, impact_metrics: ImpactMetrics) -> float:
        """Calculate overall actual impact score based on multiple factors."""
        # Weighted combination of different impact factors
        factors = []
        
        # Price movement factor (40% weight)
        if impact_metrics.max_move_pct is not None:
            price_factor = min(10, abs(impact_metrics.max_move_pct) * 2)  # Scale to 0-10
            factors.append((price_factor, 0.4))
            
        # Volume factor (20% weight)
        if impact_metrics.volume_change_pct is not None:
            volume_factor = min(10, impact_metrics.volume_change_pct / 50)  # Scale to 0-10
            factors.append((volume_factor, 0.2))
            
        # Volatility factor (20% weight)
        if impact_metrics.volatility_spike is not None:
            vol_factor = min(10, impact_metrics.volatility_spike * 2)  # Scale to 0-10
            factors.append((vol_factor, 0.2))
            
        # Headline factor (20% weight)
        if impact_metrics.headline_count > 0:
            headline_factor = min(10, impact_metrics.headline_count * 2)  # Scale to 0-10
            factors.append((headline_factor, 0.2))
            
        if not factors:
            return 5.0  # Default neutral impact
            
        # Calculate weighted average
        total_weight = sum(weight for _, weight in factors)
        weighted_sum = sum(value * weight for value, weight in factors)
        
        return weighted_sum / total_weight
        
    def _calculate_accuracy_score(self, predicted: float, actual: float) -> float:
        """Calculate accuracy score between predicted and actual impact."""
        # Accuracy score from 0-1 based on how close prediction was
        error = abs(predicted - actual)
        max_error = 10.0  # Maximum possible error on 1-10 scale
        
        # Convert to 0-1 score (1 = perfect prediction, 0 = maximum error)
        accuracy = max(0, 1 - (error / max_error))
        return accuracy
        
    async def track_event(self, event: EventORM):
        """Start tracking lifecycle for a new event."""
        if event.id not in self._lifecycle_cache:
            lifecycle_event = await self._create_lifecycle_event(event)
            logger.info(f"Started lifecycle tracking for event {event.id} ({event.symbol})")
            return lifecycle_event
        return self._lifecycle_cache[event.id]
        
    async def update_event_status(self, event_id: str, new_status: EventStatus, reason: str = "manual_update"):
        """Manually update event status."""
        if event_id in self._lifecycle_cache:
            lifecycle_event = self._lifecycle_cache[event_id]
            old_status = lifecycle_event.current_status
            lifecycle_event.current_status = new_status
            lifecycle_event.updated_at = datetime.now(timezone.utc)
            lifecycle_event.status_history.append({
                "status": new_status.value,
                "timestamp": lifecycle_event.updated_at.isoformat(),
                "reason": reason,
                "previous_status": old_status.value
            })
            
            logger.info(f"Updated event {event_id} status from {old_status.value} to {new_status.value}")
            
            # Update database
            async with self.session_factory() as session:
                event = await session.get(EventORM, event_id)
                if event:
                    event.status = new_status.value
                    await session.commit()
                    
            return lifecycle_event
        return None
        
    def get_lifecycle_event(self, event_id: str) -> Optional[LifecycleEvent]:
        """Get lifecycle tracking data for an event."""
        return self._lifecycle_cache.get(event_id)
        
    def get_events_by_stage(self, stage: LifecycleStage) -> List[LifecycleEvent]:
        """Get all events in a specific lifecycle stage."""
        return [event for event in self._lifecycle_cache.values() if event.current_stage == stage]
        
    def get_events_by_status(self, status: EventStatus) -> List[LifecycleEvent]:
        """Get all events with a specific status."""
        return [event for event in self._lifecycle_cache.values() if event.current_status == status]
        
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get lifecycle tracking statistics."""
        events = list(self._lifecycle_cache.values())
        
        # Count by status
        status_counts = {}
        for status in EventStatus:
            status_counts[status.value] = len([e for e in events if e.current_status == status])
            
        # Count by stage
        stage_counts = {}
        for stage in LifecycleStage:
            stage_counts[stage.value] = len([e for e in events if e.current_stage == stage])
            
        # Accuracy statistics
        analyzed_events = [e for e in events if e.impact_metrics and e.impact_metrics.accuracy_score is not None]
        avg_accuracy = sum(e.impact_metrics.accuracy_score for e in analyzed_events) / len(analyzed_events) if analyzed_events else 0
        
        return {
            "total_tracked_events": len(events),
            "status_distribution": status_counts,
            "stage_distribution": stage_counts,
            "analyzed_events": len(analyzed_events),
            "average_accuracy": avg_accuracy,
            "config": {
                "monitor_interval_minutes": self.monitor_interval_minutes,
                "analysis_delay_hours": self.analysis_delay_hours,
                "pre_event_window_hours": self.pre_event_window_hours,
                "post_event_window_hours": self.post_event_window_hours
            }
        }


def build_lifecycle_tracker(session_factory, config: Optional[Dict[str, Any]] = None) -> EventLifecycleTracker:
    """Build and configure the event lifecycle tracker."""
    return EventLifecycleTracker(session_factory, config)