"""Event clustering and relationship detection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EventORM


@dataclass
class EventCluster:
    """Represents a cluster of related events."""
    
    cluster_id: str
    cluster_type: str  # company, sector, supply_chain, theme
    primary_symbol: str
    related_symbols: List[str]
    event_ids: List[str]
    cluster_score: float
    metadata: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cluster_type": self.cluster_type,
            "primary_symbol": self.primary_symbol,
            "related_symbols": self.related_symbols,
            "event_ids": self.event_ids,
            "cluster_score": self.cluster_score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ClusteringRule:
    """Rule for detecting event relationships."""
    
    rule_type: str
    symbol_patterns: List[str]
    category_patterns: List[str]
    time_window_hours: int
    min_confidence: float
    weight: float


class EventClusteringEngine:
    """Detect and cluster related events across companies and sectors."""
    
    # Default clustering rules
    DEFAULT_RULES = [
        # Same company events within 24 hours
        ClusteringRule(
            rule_type="company_same_symbol",
            symbol_patterns=["exact_match"],
            category_patterns=["*"],
            time_window_hours=24,
            min_confidence=0.9,
            weight=1.0,
        ),
        # Earnings season clustering
        ClusteringRule(
            rule_type="sector_earnings",
            symbol_patterns=["sector_match"],
            category_patterns=["earnings", "earnings_call", "guidance"],
            time_window_hours=168,  # 1 week
            min_confidence=0.6,
            weight=0.8,
        ),
        # Regulatory/FDA events affecting sector
        ClusteringRule(
            rule_type="regulatory_sector",
            symbol_patterns=["sector_match"],
            category_patterns=["regulatory", "fda_approval"],
            time_window_hours=72,
            min_confidence=0.7,
            weight=0.9,
        ),
        # M&A wave detection
        ClusteringRule(
            rule_type="mna_wave",
            symbol_patterns=["sector_match"],
            category_patterns=["mna", "merger", "acquisition"],
            time_window_hours=720,  # 30 days
            min_confidence=0.5,
            weight=0.7,
        ),
        # Supply chain events
        ClusteringRule(
            rule_type="supply_chain",
            symbol_patterns=["supply_chain_match"],
            category_patterns=["product_launch", "regulatory", "guidance"],
            time_window_hours=48,
            min_confidence=0.6,
            weight=0.8,
        ),
    ]
    
    # Sector mapping for clustering
    SECTOR_MAPPING = {
        "AAPL": "technology",
        "MSFT": "technology", 
        "GOOGL": "technology",
        "AMZN": "technology",
        "TSLA": "automotive",
        "F": "automotive",
        "GM": "automotive",
        "JPM": "financials",
        "BAC": "financials",
        "WFC": "financials",
        "JNJ": "healthcare",
        "PFE": "healthcare",
        "MRNA": "healthcare",
        "XOM": "energy",
        "CVX": "energy",
        "COP": "energy",
        # Add more as needed
    }
    
    # Supply chain relationships
    SUPPLY_CHAIN_RELATIONSHIPS = {
        "AAPL": ["TSM", "QCOM", "AVGO"],  # Apple suppliers
        "TSLA": ["PANW", "LIT", "ALB"],   # Tesla battery/tech suppliers
        "F": ["GM", "TSLA"],             # Auto competitors
        "XOM": ["CVX", "COP", "SLB"],    # Oil & gas related
        # Add more relationships
    }
    
    def __init__(self, session_factory, config: Optional[Dict[str, Any]] = None) -> None:
        self._session_factory = session_factory
        self._config = config or {}
        self._rules = self._load_rules()
        self._sector_mapping = self._load_sector_mapping()
        self._supply_chain = self._load_supply_chain_relationships()
        self._clusters: Dict[str, EventCluster] = {}
        
    def _load_rules(self) -> List[ClusteringRule]:
        """Load clustering rules from config or environment."""
        rules = self._config.get("clustering_rules", [])
        if not rules:
            env_rules = os.getenv("EVENT_CLUSTERING_RULES")
            if env_rules:
                try:
                    data = json.loads(env_rules)
                    if isinstance(data, list):
                        rules = [
                            ClusteringRule(**rule) for rule in data
                            if all(k in rule for k in ["rule_type", "symbol_patterns", "category_patterns", "time_window_hours", "min_confidence", "weight"])
                        ]
                except (json.JSONDecodeError, TypeError):
                    pass
        return rules if rules else self.DEFAULT_RULES
    
    def _load_sector_mapping(self) -> Dict[str, str]:
        """Load sector mapping from config or environment."""
        mapping = self._config.get("sector_mapping", {})
        if not mapping:
            env_mapping = os.getenv("EVENT_SECTOR_MAPPING")
            if env_mapping:
                try:
                    data = json.loads(env_mapping)
                    if isinstance(data, dict):
                        mapping = data
                except json.JSONDecodeError:
                    pass
        return {**self.SECTOR_MAPPING, **mapping}
    
    def _load_supply_chain_relationships(self) -> Dict[str, List[str]]:
        """Load supply chain relationships from config or environment."""
        relationships = self._config.get("supply_chain_relationships", {})
        if not relationships:
            env_relationships = os.getenv("EVENT_SUPPLY_CHAIN_RELATIONSHIPS")
            if env_relationships:
                try:
                    data = json.loads(env_relationships)
                    if isinstance(data, dict):
                        relationships = data
                except json.JSONDecodeError:
                    pass
        return {**self.SUPPLY_CHAIN_RELATIONSHIPS, **relationships}
    
    async def cluster_events(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[EventCluster]:
        """Find and cluster related events in the specified time range."""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=7)
        if not end_time:
            end_time = datetime.utcnow() + timedelta(days=30)
            
        async with self._session_factory() as session:
            # Get events in time range
            stmt = select(EventORM).where(
                and_(
                    EventORM.scheduled_at >= start_time,
                    EventORM.scheduled_at <= end_time
                )
            ).order_by(EventORM.scheduled_at)
            
            result = await session.execute(stmt)
            events = result.scalars().all()
            
        if not events:
            return []
            
        # Apply clustering rules
        clusters = []
        for rule in self._rules:
            rule_clusters = await self._apply_clustering_rule(events, rule)
            clusters.extend(rule_clusters)
            
        # Merge overlapping clusters
        merged_clusters = self._merge_overlapping_clusters(clusters)
        
        # Store clusters for retrieval
        for cluster in merged_clusters:
            self._clusters[cluster.cluster_id] = cluster
            
        return merged_clusters
    
    async def _apply_clustering_rule(self, events: List[EventORM], rule: ClusteringRule) -> List[EventCluster]:
        """Apply a specific clustering rule to events."""
        clusters = []
        processed_events = set()
        
        for i, event in enumerate(events):
            if event.id in processed_events:
                continue
                
            # Check if event matches rule criteria
            if not self._event_matches_rule(event, rule):
                continue
                
            # Find related events within time window
            related_events = []
            time_window_start = event.scheduled_at - timedelta(hours=rule.time_window_hours // 2)
            time_window_end = event.scheduled_at + timedelta(hours=rule.time_window_hours // 2)
            
            for j, other_event in enumerate(events):
                if (i != j and 
                    other_event.id not in processed_events and
                    time_window_start <= other_event.scheduled_at <= time_window_end and
                    self._events_are_related(event, other_event, rule)):
                    related_events.append(other_event)
            
            # Create cluster if we have related events
            if related_events:
                cluster = self._create_cluster(event, related_events, rule)
                clusters.append(cluster)
                
                # Mark events as processed
                processed_events.add(event.id)
                for rel_event in related_events:
                    processed_events.add(rel_event.id)
                    
        return clusters
    
    def _event_matches_rule(self, event: EventORM, rule: ClusteringRule) -> bool:
        """Check if an event matches a clustering rule."""
        # Check category patterns
        if rule.category_patterns != ["*"]:
            if not any(pattern in event.category.lower() for pattern in rule.category_patterns):
                return False
                
        return True
    
    def _events_are_related(self, event1: EventORM, event2: EventORM, rule: ClusteringRule) -> bool:
        """Check if two events are related according to a rule."""
        for pattern in rule.symbol_patterns:
            if pattern == "exact_match":
                if event1.symbol == event2.symbol:
                    return True
            elif pattern == "sector_match":
                sector1 = self._sector_mapping.get(event1.symbol)
                sector2 = self._sector_mapping.get(event2.symbol)
                if sector1 and sector2 and sector1 == sector2:
                    return True
            elif pattern == "supply_chain_match":
                if self._are_supply_chain_related(event1.symbol, event2.symbol):
                    return True
                    
        return False
    
    def _are_supply_chain_related(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are related through supply chain."""
        # Check direct relationships
        if symbol2 in self._supply_chain.get(symbol1, []):
            return True
        if symbol1 in self._supply_chain.get(symbol2, []):
            return True
            
        # Check mutual relationships (both suppliers of same company)
        for company, suppliers in self._supply_chain.items():
            if symbol1 in suppliers and symbol2 in suppliers:
                return True
                
        return False
    
    def _create_cluster(self, primary_event: EventORM, related_events: List[EventORM], rule: ClusteringRule) -> EventCluster:
        """Create an event cluster from primary and related events."""
        all_events = [primary_event] + related_events
        all_symbols = list(set(event.symbol for event in all_events))
        event_ids = [event.id for event in all_events]
        
        # Generate cluster ID
        cluster_id = f"{rule.rule_type}_{primary_event.symbol}_{primary_event.scheduled_at.strftime('%Y%m%d_%H%M')}"
        
        # Calculate cluster score based on rule weight and event confidence
        cluster_score = rule.weight
        if hasattr(primary_event, 'impact_score') and primary_event.impact_score:
            cluster_score *= (primary_event.impact_score / 10.0)
        
        # Build metadata
        metadata = {
            "rule_type": rule.rule_type,
            "time_window_hours": rule.time_window_hours,
            "primary_category": primary_event.category,
            "categories": list(set(event.category for event in all_events)),
            "event_count": len(all_events),
            "time_span_hours": (max(event.scheduled_at for event in all_events) - 
                               min(event.scheduled_at for event in all_events)).total_seconds() / 3600,
        }
        
        # Add sector information if available
        sectors = set()
        for symbol in all_symbols:
            sector = self._sector_mapping.get(symbol)
            if sector:
                sectors.add(sector)
        if sectors:
            metadata["sectors"] = list(sectors)
            
        return EventCluster(
            cluster_id=cluster_id,
            cluster_type=rule.rule_type,
            primary_symbol=primary_event.symbol,
            related_symbols=[s for s in all_symbols if s != primary_event.symbol],
            event_ids=event_ids,
            cluster_score=cluster_score,
            metadata=metadata,
            created_at=datetime.utcnow(),
        )
    
    def _merge_overlapping_clusters(self, clusters: List[EventCluster]) -> List[EventCluster]:
        """Merge clusters that share events."""
        if not clusters:
            return []
            
        merged = []
        processed = set()
        
        for i, cluster in enumerate(clusters):
            if i in processed:
                continue
                
            # Find overlapping clusters
            overlapping = [cluster]
            overlapping_indices = {i}
            
            for j, other_cluster in enumerate(clusters[i+1:], i+1):
                if j in processed:
                    continue
                    
                # Check for event overlap
                if set(cluster.event_ids) & set(other_cluster.event_ids):
                    overlapping.append(other_cluster)
                    overlapping_indices.add(j)
                    
            # Merge overlapping clusters
            if len(overlapping) > 1:
                merged_cluster = self._merge_clusters(overlapping)
                merged.append(merged_cluster)
            else:
                merged.append(cluster)
                
            processed.update(overlapping_indices)
            
        return merged
    
    def _merge_clusters(self, clusters: List[EventCluster]) -> EventCluster:
        """Merge multiple overlapping clusters into one."""
        if len(clusters) == 1:
            return clusters[0]
            
        # Use the cluster with highest score as primary
        primary = max(clusters, key=lambda c: c.cluster_score)
        
        # Combine all unique events and symbols
        all_event_ids = set()
        all_symbols = set()
        
        for cluster in clusters:
            all_event_ids.update(cluster.event_ids)
            all_symbols.add(cluster.primary_symbol)
            all_symbols.update(cluster.related_symbols)
            
        # Create merged metadata
        merged_metadata = dict(primary.metadata)
        merged_metadata.update({
            "merged_clusters": [c.cluster_id for c in clusters],
            "merged_types": list(set(c.cluster_type for c in clusters)),
            "event_count": len(all_event_ids),
        })
        
        return EventCluster(
            cluster_id=f"merged_{primary.cluster_id}",
            cluster_type="merged",
            primary_symbol=primary.primary_symbol,
            related_symbols=[s for s in all_symbols if s != primary.primary_symbol],
            event_ids=list(all_event_ids),
            cluster_score=max(c.cluster_score for c in clusters),
            metadata=merged_metadata,
            created_at=datetime.utcnow(),
        )
    
    async def get_cluster(self, cluster_id: str) -> Optional[EventCluster]:
        """Get a specific cluster by ID."""
        return self._clusters.get(cluster_id)
    
    async def get_clusters_for_symbol(self, symbol: str) -> List[EventCluster]:
        """Get all clusters involving a specific symbol."""
        return [
            cluster for cluster in self._clusters.values()
            if symbol == cluster.primary_symbol or symbol in cluster.related_symbols
        ]
    
    async def get_clusters_by_type(self, cluster_type: str) -> List[EventCluster]:
        """Get all clusters of a specific type."""
        return [
            cluster for cluster in self._clusters.values()
            if cluster.cluster_type == cluster_type
        ]


def build_clustering_engine(session_factory) -> EventClusteringEngine:
    """Build default clustering engine."""
    return EventClusteringEngine(session_factory)