"""GraphQL resolvers for event data service."""

from __future__ import annotations

import json
import strawberry
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy import and_, or_, select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_session
from ..models import EventORM, EventHeadlineORM
from .types import (
    Event, EventHeadline, EventCluster, CategoryInfo, FeedStatus, ClusterAnalysis,
    EventFilter, ClusterFilter, EventRelationshipInput, EventRelationship, EventGraph,
    EventCreateInput, EventUpdateInput, MutationResult
)


class EventResolver:
    """Resolvers for Event type."""
    
    def __init__(self, session_factory):
        self._session_factory = session_factory
    
    async def _get_event_headlines(self, event_id: str) -> List[EventHeadline]:
        """Get headlines for an event."""
        async with self._session_factory() as session:
            stmt = select(EventHeadlineORM).where(EventHeadlineORM.event_id == event_id)
            result = await session.execute(stmt)
            headlines = result.scalars().all()
            
            return [
                EventHeadline(
                    id=h.id,
                    event_id=h.event_id,
                    symbol=h.symbol,
                    headline=h.headline,
                    published_at=h.published_at,
                    summary=h.summary,
                    url=h.url,
                    source=h.source,
                    external_id=h.external_id,
                    created_at=h.created_at,
                )
                for h in headlines
            ]
    
    async def _get_event_clusters(self, event_id: str, clustering_engine) -> List[EventCluster]:
        """Get clusters containing an event."""
        if not clustering_engine:
            return []
            
        clusters = []
        for cluster in clustering_engine._clusters.values():
            if event_id in cluster.event_ids:
                clusters.append(EventCluster(
                    cluster_id=cluster.cluster_id,
                    cluster_type=cluster.cluster_type,
                    primary_symbol=cluster.primary_symbol,
                    related_symbols=cluster.related_symbols,
                    event_ids=cluster.event_ids,
                    cluster_score=cluster.cluster_score,
                    created_at=cluster.created_at,
                ))
        
        return clusters


@strawberry.type
class Query:
    """GraphQL Query root."""
    
    @strawberry.field
    async def events(
        self,
        filter: Optional[EventFilter] = None,
        limit: int = 100,
        offset: int = 0,
        info: strawberry.Info = None,
    ) -> List[Event]:
        """Query events with filtering and pagination."""
        session_factory = info.context.get("session_factory")
        
        async with session_factory() as session:
            stmt = select(EventORM)
            
            # Apply filters
            if filter:
                conditions = []
                
                if filter.symbols:
                    conditions.append(EventORM.symbol.in_(filter.symbols))
                
                if filter.categories:
                    conditions.append(EventORM.category.in_(filter.categories))
                
                if filter.statuses:
                    conditions.append(EventORM.status.in_(filter.statuses))
                
                if filter.impact_score_min is not None:
                    conditions.append(EventORM.impact_score >= filter.impact_score_min)
                
                if filter.impact_score_max is not None:
                    conditions.append(EventORM.impact_score <= filter.impact_score_max)
                
                if filter.start_time:
                    conditions.append(EventORM.scheduled_at >= filter.start_time)
                
                if filter.end_time:
                    conditions.append(EventORM.scheduled_at <= filter.end_time)
                
                if filter.sources:
                    conditions.append(EventORM.source.in_(filter.sources))
                
                if filter.search_text:
                    search_term = f"%{filter.search_text}%"
                    conditions.append(
                        or_(
                            EventORM.title.ilike(search_term),
                            EventORM.description.ilike(search_term),
                            EventORM.symbol.ilike(search_term)
                        )
                    )
                
                if filter.has_headlines is not None:
                    if filter.has_headlines:
                        stmt = stmt.join(EventHeadlineORM, EventORM.id == EventHeadlineORM.event_id)
                    else:
                        stmt = stmt.outerjoin(EventHeadlineORM, EventORM.id == EventHeadlineORM.event_id)
                        conditions.append(EventHeadlineORM.id.is_(None))
                
                if conditions:
                    stmt = stmt.where(and_(*conditions))
            
            # Apply pagination
            stmt = stmt.offset(offset).limit(limit).order_by(EventORM.scheduled_at.desc())
            
            result = await session.execute(stmt)
            events = result.scalars().all()
            
            return [
                Event(
                    id=e.id,
                    symbol=e.symbol,
                    title=e.title,
                    category=e.category,
                    scheduled_at=e.scheduled_at,
                    timezone=e.timezone,
                    description=e.description,
                    status=e.status,
                    source=e.source,
                    external_id=e.external_id,
                    impact_score=e.impact_score,
                    created_at=e.created_at,
                    updated_at=e.updated_at,
                )
                for e in events
            ]
    
    @strawberry.field
    async def event(self, id: str, info: strawberry.Info = None) -> Optional[Event]:
        """Get a specific event by ID."""
        session_factory = info.context.get("session_factory")
        
        async with session_factory() as session:
            event = await session.get(EventORM, id)
            if not event:
                return None
                
            return Event(
                id=event.id,
                symbol=event.symbol,
                title=event.title,
                category=event.category,
                scheduled_at=event.scheduled_at,
                timezone=event.timezone,
                description=event.description,
                status=event.status,
                source=event.source,
                external_id=event.external_id,
                impact_score=event.impact_score,
                created_at=event.created_at,
                updated_at=event.updated_at,
            )
    
    @strawberry.field
    async def headlines(
        self,
        symbol: Optional[str] = None,
        event_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        info: strawberry.Info = None,
    ) -> List[EventHeadline]:
        """Query headlines with filtering."""
        session_factory = info.context.get("session_factory")
        
        async with session_factory() as session:
            stmt = select(EventHeadlineORM)
            
            conditions = []
            if symbol:
                conditions.append(EventHeadlineORM.symbol == symbol)
            if event_id:
                conditions.append(EventHeadlineORM.event_id == event_id)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            stmt = stmt.offset(offset).limit(limit).order_by(EventHeadlineORM.published_at.desc())
            
            result = await session.execute(stmt)
            headlines = result.scalars().all()
            
            return [
                EventHeadline(
                    id=h.id,
                    event_id=h.event_id,
                    symbol=h.symbol,
                    headline=h.headline,
                    published_at=h.published_at,
                    summary=h.summary,
                    url=h.url,
                    source=h.source,
                    external_id=h.external_id,
                    created_at=h.created_at,
                )
                for h in headlines
            ]
    
    @strawberry.field
    async def clusters(
        self,
        filter: Optional[ClusterFilter] = None,
        info: strawberry.Info = None,
    ) -> List[EventCluster]:
        """Query event clusters."""
        clustering_engine = info.context.get("clustering_engine")
        if not clustering_engine:
            return []
        
        # Get all clusters
        all_clusters = list(clustering_engine._clusters.values())
        
        # Apply filters
        if filter:
            filtered_clusters = []
            
            for cluster in all_clusters:
                # Filter by cluster types
                if filter.cluster_types and cluster.cluster_type not in filter.cluster_types:
                    continue
                
                # Filter by symbols
                if filter.symbols:
                    cluster_symbols = {cluster.primary_symbol} | set(cluster.related_symbols)
                    if not any(symbol in cluster_symbols for symbol in filter.symbols):
                        continue
                
                # Filter by score
                if filter.min_score is not None and cluster.cluster_score < filter.min_score:
                    continue
                if filter.max_score is not None and cluster.cluster_score > filter.max_score:
                    continue
                
                # Filter by event count
                if filter.min_events is not None and len(cluster.event_ids) < filter.min_events:
                    continue
                
                # Filter by time (using cluster creation time as proxy)
                if filter.start_time and cluster.created_at < filter.start_time:
                    continue
                if filter.end_time and cluster.created_at > filter.end_time:
                    continue
                
                filtered_clusters.append(cluster)
            
            all_clusters = filtered_clusters
        
        return [
            EventCluster(
                cluster_id=c.cluster_id,
                cluster_type=c.cluster_type,
                primary_symbol=c.primary_symbol,
                related_symbols=c.related_symbols,
                event_ids=c.event_ids,
                cluster_score=c.cluster_score,
                created_at=c.created_at,
            )
            for c in all_clusters
        ]
    
    @strawberry.field
    async def event_relationships(
        self,
        input: EventRelationshipInput,
        info: strawberry.Info = None,
    ) -> EventGraph:
        """Find relationships for an event and build a graph."""
        session_factory = info.context.get("session_factory")
        clustering_engine = info.context.get("clustering_engine")
        categorizer = info.context.get("categorizer")
        
        async with session_factory() as session:
            # Get the central event
            central_event_orm = await session.get(EventORM, input.event_id)
            if not central_event_orm:
                return EventGraph(
                    central_event=None,
                    related_events=[],
                    relationships=[],
                    clusters=[],
                )
            
            central_event = Event(
                id=central_event_orm.id,
                symbol=central_event_orm.symbol,
                title=central_event_orm.title,
                category=central_event_orm.category,
                scheduled_at=central_event_orm.scheduled_at,
                timezone=central_event_orm.timezone,
                description=central_event_orm.description,
                status=central_event_orm.status,
                source=central_event_orm.source,
                external_id=central_event_orm.external_id,
                impact_score=central_event_orm.impact_score,
                created_at=central_event_orm.created_at,
                updated_at=central_event_orm.updated_at,
            )
            
            # Find related events
            related_events = []
            relationships = []
            
            # Time-based relationships
            if input.include_temporal:
                time_window = timedelta(hours=input.time_window_hours or 168)
                start_time = central_event_orm.scheduled_at - time_window
                end_time = central_event_orm.scheduled_at + time_window
                
                stmt = select(EventORM).where(
                    and_(
                        EventORM.id != input.event_id,
                        EventORM.scheduled_at >= start_time,
                        EventORM.scheduled_at <= end_time
                    )
                )
                
                result = await session.execute(stmt)
                temporal_events = result.scalars().all()
                
                for event_orm in temporal_events:
                    event = Event(
                        id=event_orm.id,
                        symbol=event_orm.symbol,
                        title=event_orm.title,
                        category=event_orm.category,
                        scheduled_at=event_orm.scheduled_at,
                        timezone=event_orm.timezone,
                        description=event_orm.description,
                        status=event_orm.status,
                        source=event_orm.source,
                        external_id=event_orm.external_id,
                        impact_score=event_orm.impact_score,
                        created_at=event_orm.created_at,
                        updated_at=event_orm.updated_at,
                    )
                    
                    # Determine relationship type and strength
                    rel_type = "temporal"
                    strength = 0.5
                    
                    if event_orm.symbol == central_event_orm.symbol:
                        rel_type = "same_company"
                        strength = 0.9
                    elif event_orm.category == central_event_orm.category:
                        rel_type = "same_category"
                        strength = 0.7
                    
                    related_events.append(event)
                    relationships.append(EventRelationship(
                        from_event=central_event,
                        to_event=event,
                        relationship_type=rel_type,
                        strength=strength,
                        distance=1,
                    ))
            
            # Get clusters containing this event
            clusters = []
            if clustering_engine:
                for cluster in clustering_engine._clusters.values():
                    if input.event_id in cluster.event_ids:
                        clusters.append(EventCluster(
                            cluster_id=cluster.cluster_id,
                            cluster_type=cluster.cluster_type,
                            primary_symbol=cluster.primary_symbol,
                            related_symbols=cluster.related_symbols,
                            event_ids=cluster.event_ids,
                            cluster_score=cluster.cluster_score,
                            created_at=cluster.created_at,
                        ))
            
            return EventGraph(
                central_event=central_event,
                related_events=related_events,
                relationships=relationships,
                clusters=clusters,
            )
    
    @strawberry.field
    async def categories(self, info: strawberry.Info = None) -> List[CategoryInfo]:
        """Get event categories with counts."""
        session_factory = info.context.get("session_factory")
        categorizer = info.context.get("categorizer")
        
        # Get available categories from categorizer
        categories = categorizer.categories() if categorizer else []
        
        # Get counts from database
        async with session_factory() as session:
            stmt = select(EventORM.category, func.count(EventORM.id)).group_by(EventORM.category)
            result = await session.execute(stmt)
            category_counts = dict(result.all())
        
        return [
            CategoryInfo(
                name=category,
                count=category_counts.get(category, 0),
                description=f"Events categorized as {category}",
            )
            for category in categories
        ]
    
    @strawberry.field
    async def feed_status(self, info: strawberry.Info = None) -> List[FeedStatus]:
        """Get feed health status."""
        feed_monitor = info.context.get("feed_monitor")
        if not feed_monitor:
            return []
        
        status_data = await feed_monitor.snapshot()
        
        return [
            FeedStatus(
                name=name,
                status=data.get("status", "unknown"),
                consecutive_failures=data.get("consecutive_failures", 0),
                last_success=datetime.fromisoformat(data["last_success"]) if data.get("last_success") else None,
                last_failure=datetime.fromisoformat(data["last_failure"]) if data.get("last_failure") else None,
                last_event_count=data.get("last_event_count"),
                message=data.get("message"),
                alert_active=data.get("alert_active", False),
            )
            for name, data in status_data.items()
        ]


@strawberry.type
class Mutation:
    """GraphQL Mutation root."""
    
    @strawberry.field
    async def create_event(
        self,
        input: EventCreateInput,
        info: strawberry.Info = None,
    ) -> MutationResult:
        """Create a new event."""
        session_factory = info.context.get("session_factory")
        categorizer = info.context.get("categorizer")
        
        try:
            async with session_factory() as session:
                # Parse metadata if provided
                metadata = {}
                if input.metadata:
                    try:
                        metadata = json.loads(input.metadata)
                    except json.JSONDecodeError:
                        return MutationResult(
                            success=False,
                            message="Invalid JSON in metadata field",
                            errors=["Metadata must be valid JSON"]
                        )
                
                # Apply categorization
                canonical_category = input.category
                if categorizer:
                    result = categorizer.categorize(
                        raw_category=input.category,
                        title=input.title,
                        description=input.description,
                        metadata=metadata,
                    )
                    canonical_category = result.category
                    metadata.setdefault("classification", {}).update({
                        "raw_category": input.category,
                        "canonical_category": result.category,
                        "confidence": result.confidence,
                        "matched_keywords": result.matched_keywords,
                    })
                
                # Create event
                now = datetime.utcnow()
                event_orm = EventORM(
                    symbol=input.symbol,
                    title=input.title,
                    category=canonical_category,
                    scheduled_at=input.scheduled_at,
                    timezone=input.timezone,
                    description=input.description,
                    status=input.status,
                    source=input.source,
                    external_id=input.external_id,
                    impact_score=input.impact_score,
                    metadata_json=metadata,
                    created_at=now,
                    updated_at=now,
                )
                
                session.add(event_orm)
                await session.commit()
                await session.refresh(event_orm)
                
                event = Event(
                    id=event_orm.id,
                    symbol=event_orm.symbol,
                    title=event_orm.title,
                    category=event_orm.category,
                    scheduled_at=event_orm.scheduled_at,
                    timezone=event_orm.timezone,
                    description=event_orm.description,
                    status=event_orm.status,
                    source=event_orm.source,
                    external_id=event_orm.external_id,
                    impact_score=event_orm.impact_score,
                    created_at=event_orm.created_at,
                    updated_at=event_orm.updated_at,
                )
                
                return MutationResult(
                    success=True,
                    message="Event created successfully",
                    event=event,
                )
                
        except Exception as e:
            return MutationResult(
                success=False,
                message=f"Failed to create event: {str(e)}",
                errors=[str(e)],
            )
    
    @strawberry.field
    async def update_event(
        self,
        id: str,
        input: EventUpdateInput,
        info: strawberry.Info = None,
    ) -> MutationResult:
        """Update an existing event."""
        session_factory = info.context.get("session_factory")
        categorizer = info.context.get("categorizer")
        
        try:
            async with session_factory() as session:
                event_orm = await session.get(EventORM, id)
                if not event_orm:
                    return MutationResult(
                        success=False,
                        message="Event not found",
                        errors=["Event with specified ID does not exist"],
                    )
                
                # Update fields
                if input.symbol is not None:
                    event_orm.symbol = input.symbol
                if input.title is not None:
                    event_orm.title = input.title
                if input.scheduled_at is not None:
                    event_orm.scheduled_at = input.scheduled_at
                if input.timezone is not None:
                    event_orm.timezone = input.timezone
                if input.description is not None:
                    event_orm.description = input.description
                if input.status is not None:
                    event_orm.status = input.status
                if input.impact_score is not None:
                    event_orm.impact_score = input.impact_score
                
                # Handle category and metadata updates
                if input.category is not None or input.metadata is not None:
                    metadata = dict(event_orm.metadata_json or {})
                    
                    if input.metadata:
                        try:
                            new_metadata = json.loads(input.metadata)
                            metadata.update(new_metadata)
                        except json.JSONDecodeError:
                            return MutationResult(
                                success=False,
                                message="Invalid JSON in metadata field",
                                errors=["Metadata must be valid JSON"]
                            )
                    
                    # Apply categorization if category changed
                    if input.category is not None:
                        canonical_category = input.category
                        if categorizer:
                            result = categorizer.categorize(
                                raw_category=input.category,
                                title=event_orm.title,
                                description=event_orm.description,
                                metadata=metadata,
                            )
                            canonical_category = result.category
                            metadata.setdefault("classification", {}).update({
                                "raw_category": input.category,
                                "canonical_category": result.category,
                                "confidence": result.confidence,
                                "matched_keywords": result.matched_keywords,
                            })
                        
                        event_orm.category = canonical_category
                    
                    event_orm.metadata_json = metadata
                
                event_orm.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(event_orm)
                
                event = Event(
                    id=event_orm.id,
                    symbol=event_orm.symbol,
                    title=event_orm.title,
                    category=event_orm.category,
                    scheduled_at=event_orm.scheduled_at,
                    timezone=event_orm.timezone,
                    description=event_orm.description,
                    status=event_orm.status,
                    source=event_orm.source,
                    external_id=event_orm.external_id,
                    impact_score=event_orm.impact_score,
                    created_at=event_orm.created_at,
                    updated_at=event_orm.updated_at,
                )
                
                return MutationResult(
                    success=True,
                    message="Event updated successfully",
                    event=event,
                )
                
        except Exception as e:
            return MutationResult(
                success=False,
                message=f"Failed to update event: {str(e)}",
                errors=[str(e)],
            )
    
    @strawberry.field
    async def delete_event(
        self,
        id: str,
        info: strawberry.Info = None,
    ) -> MutationResult:
        """Delete an event."""
        session_factory = info.context.get("session_factory")
        
        try:
            async with session_factory() as session:
                event_orm = await session.get(EventORM, id)
                if not event_orm:
                    return MutationResult(
                        success=False,
                        message="Event not found",
                        errors=["Event with specified ID does not exist"],
                    )
                
                await session.delete(event_orm)
                await session.commit()
                
                return MutationResult(
                    success=True,
                    message="Event deleted successfully",
                )
                
        except Exception as e:
            return MutationResult(
                success=False,
                message=f"Failed to delete event: {str(e)}",
                errors=[str(e)],
            )