"""GraphQL types for event data service."""

from __future__ import annotations

import strawberry
from datetime import datetime
from typing import List, Optional, Dict, Any


@strawberry.type
class Event:
    """Event GraphQL type."""
    
    id: str
    symbol: str
    title: str
    category: str
    scheduled_at: datetime
    timezone: Optional[str] = None
    description: Optional[str] = None
    status: str = "scheduled"
    source: Optional[str] = None
    external_id: Optional[str] = None
    impact_score: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
    @strawberry.field
    def metadata(self) -> Optional[str]:
        """JSON metadata as string."""
        return None  # Will be populated by resolver
    
    @strawberry.field
    async def headlines(self) -> List[EventHeadline]:
        """Related headlines for this event."""
        return []  # Will be populated by resolver
    
    @strawberry.field
    async def clusters(self) -> List[EventCluster]:
        """Clusters containing this event."""
        return []  # Will be populated by resolver


@strawberry.type
class EventHeadline:
    """Event headline GraphQL type."""
    
    id: str
    event_id: Optional[str] = None
    symbol: str
    headline: str
    published_at: datetime
    summary: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    external_id: Optional[str] = None
    created_at: datetime
    
    @strawberry.field
    async def event(self) -> Optional[Event]:
        """Associated event."""
        return None  # Will be populated by resolver


@strawberry.type
class EventCluster:
    """Event cluster GraphQL type."""
    
    cluster_id: str
    cluster_type: str
    primary_symbol: str
    related_symbols: List[str]
    event_ids: List[str]
    cluster_score: float
    created_at: datetime
    
    @strawberry.field
    def metadata(self) -> Optional[str]:
        """JSON metadata as string."""
        return None  # Will be populated by resolver
    
    @strawberry.field
    async def events(self) -> List[Event]:
        """Events in this cluster."""
        return []  # Will be populated by resolver


@strawberry.type
class CategoryInfo:
    """Event category information."""
    
    name: str
    count: int
    description: Optional[str] = None


@strawberry.type
class FeedStatus:
    """Feed health status."""
    
    name: str
    status: str
    consecutive_failures: int
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_event_count: Optional[int] = None
    message: Optional[str] = None
    alert_active: bool = False


@strawberry.type
class ClusterAnalysis:
    """Cluster analysis summary."""
    
    total_clusters: int
    total_events: int
    cluster_types: Dict[str, int]
    high_impact_clusters: List[EventCluster]


@strawberry.input
class EventFilter:
    """Event filtering input."""
    
    symbols: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    impact_score_min: Optional[int] = None
    impact_score_max: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    sources: Optional[List[str]] = None
    has_headlines: Optional[bool] = None
    search_text: Optional[str] = None


@strawberry.input
class ClusterFilter:
    """Cluster filtering input."""
    
    cluster_types: Optional[List[str]] = None
    symbols: Optional[List[str]] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_events: Optional[int] = None


@strawberry.input
class EventRelationshipInput:
    """Input for finding event relationships."""
    
    event_id: str
    relationship_types: Optional[List[str]] = None
    max_distance: Optional[int] = 3
    include_supply_chain: bool = True
    include_sector: bool = True
    include_temporal: bool = True
    time_window_hours: Optional[int] = 168  # 1 week default


@strawberry.type
class EventRelationship:
    """Relationship between two events."""
    
    from_event: Event
    to_event: Event
    relationship_type: str
    strength: float
    distance: int
    metadata: Optional[str] = None


@strawberry.type
class EventGraph:
    """Graph of event relationships."""
    
    central_event: Event
    related_events: List[Event]
    relationships: List[EventRelationship]
    clusters: List[EventCluster]
    
    @strawberry.field
    def total_events(self) -> int:
        """Total number of events in the graph."""
        return len(self.related_events) + 1
    
    @strawberry.field
    def relationship_types(self) -> List[str]:
        """Unique relationship types in the graph."""
        return list(set(rel.relationship_type for rel in self.relationships))


@strawberry.input
class EventCreateInput:
    """Input for creating events via GraphQL."""
    
    symbol: str
    title: str
    category: str
    scheduled_at: datetime
    timezone: Optional[str] = None
    description: Optional[str] = None
    status: str = "scheduled"
    source: Optional[str] = None
    external_id: Optional[str] = None
    impact_score: Optional[int] = None
    metadata: Optional[str] = None  # JSON string


@strawberry.input
class EventUpdateInput:
    """Input for updating events via GraphQL."""
    
    symbol: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    timezone: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    impact_score: Optional[int] = None
    metadata: Optional[str] = None  # JSON string


@strawberry.type
class MutationResult:
    """Generic mutation result."""
    
    success: bool
    message: str
    event: Optional[Event] = None
    errors: Optional[List[str]] = None