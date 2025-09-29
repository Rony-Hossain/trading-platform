"""Event Data Service - scheduled events REST API."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query, status, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from strawberry.fastapi import GraphQLRouter
import uuid
import json

from .database import SessionFactory, get_session, init_db
from .models import EventORM, EventHeadlineORM
from .schemas import Event, EventCreate, EventUpdate, EventHeadline, ImpactScoreUpdate, ImpactScoreResponse
from .services.calendar_ingestor import build_calendar_ingestor
from .services.headline_ingestor import build_headline_ingestor
from .services.feed_health import build_default_monitor
from .services.webhook_dispatcher import build_webhook_dispatcher
from .services.event_categorizer import build_default_categorizer
from .services.event_clustering import build_clustering_engine
from .services.subscription_manager import build_subscription_manager, EventType, SubscriptionRequest, SubscriptionResponse
from .services.event_enrichment import build_enrichment_service
from .services.event_lifecycle import build_lifecycle_tracker, EventStatus as LifecycleEventStatus
from .services.event_alerts import build_alert_system, AlertRule, AlertSeverity, AlertChannel, AlertReason
from .services.event_sentiment import build_sentiment_service, EventSentimentAnalysis, SentimentScore
from .services.historical_backfill import build_backfill_service, BackfillRequest, BackfillResult, BackfillStatus
from .services.data_retention import build_retention_service, RetentionStats, ArchivalResult
from .services.event_cache import build_cache_service, CacheKeyType
from .services.cache_decorators import cached_event, cached_event_list, cached_search_result, cache_invalidate_on_write
from .services.bulk_ingestion import build_bulk_ingestion_service, IngestionFormat, IngestionConfig, IngestionMode, ValidationLevel
from .services.event_streaming import build_event_streaming_service, EventType as StreamEventType
from .services.realtime_endpoints import build_websocket_manager, build_sse_manager
from .services.analytics_service import analytics_service
from .graphql import schema

SERVICE_NAME = "event-data-service"
logger = logging.getLogger(__name__)


def _to_schema(event: EventORM) -> Event:
    return Event(
        id=event.id,
        symbol=event.symbol,
        title=event.title,
        category=event.category,
        scheduled_at=event.scheduled_at,
        timezone=event.timezone,
        description=event.description,
        metadata=event.metadata_json,
        status=event.status,
        source=event.source,
        external_id=event.external_id,
        created_at=event.created_at,
        updated_at=event.updated_at,
        impact_score=event.impact_score,
    )


def create_app() -> FastAPI:
    feed_monitor = build_default_monitor(SERVICE_NAME)
    webhook_dispatcher = build_webhook_dispatcher(SERVICE_NAME)
    categorizer = build_default_categorizer()
    clustering_engine = build_clustering_engine(SessionFactory)
    subscription_manager = build_subscription_manager(SERVICE_NAME)
    enrichment_service = build_enrichment_service()
    lifecycle_tracker = build_lifecycle_tracker(SessionFactory)
    alert_system = build_alert_system()
    sentiment_service = build_sentiment_service()
    backfill_service = build_backfill_service(SessionFactory, categorizer, None, enrichment_service, feed_monitor)
    retention_service = build_retention_service(SessionFactory)
    cache_service = build_cache_service()
    bulk_ingestion_service = build_bulk_ingestion_service(SessionFactory, categorizer, None, enrichment_service, cache_service)
    streaming_service = build_event_streaming_service()
    websocket_manager = build_websocket_manager(streaming_service)
    sse_manager = build_sse_manager(streaming_service)
    calendar_ingestor = build_calendar_ingestor(SessionFactory, feed_monitor, webhook_dispatcher, categorizer, subscription_manager, enrichment_service)
    
    headline_ingestor = build_headline_ingestor(SessionFactory, feed_monitor, webhook_dispatcher, subscription_manager)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.start_time = datetime.utcnow()
        try:
            await init_db()
        except SQLAlchemyError as exc:
            print(f"Failed to initialize database: {exc}")
        await calendar_ingestor.start()
        await headline_ingestor.start()
        await subscription_manager.start()
        await enrichment_service.start()
        await lifecycle_tracker.start()
        await alert_system.start()
        await backfill_service.start()
        await retention_service.start()
        await cache_service.start()
        await bulk_ingestion_service.start()
        await streaming_service.start()
        try:
            yield
        finally:
            await calendar_ingestor.stop()
            await headline_ingestor.stop()
            await subscription_manager.stop()
            await enrichment_service.stop()
            await lifecycle_tracker.stop()
            await alert_system.stop()
            await backfill_service.stop()
            await retention_service.stop()
            await cache_service.stop()
            await bulk_ingestion_service.stop()
            await streaming_service.stop()

    app = FastAPI(
        title="Event Data Service",
        description="REST API for managing scheduled and real-time market events",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.feed_monitor = feed_monitor
    app.state.webhook_dispatcher = webhook_dispatcher
    app.state.categorizer = categorizer
    app.state.clustering_engine = clustering_engine
    app.state.subscription_manager = subscription_manager
    app.state.enrichment_service = enrichment_service
    app.state.lifecycle_tracker = lifecycle_tracker
    app.state.alert_system = alert_system
    app.state.sentiment_service = sentiment_service
    app.state.backfill_service = backfill_service
    app.state.retention_service = retention_service
    app.state.cache_service = cache_service
    app.state.bulk_ingestion_service = bulk_ingestion_service
    app.state.streaming_service = streaming_service
    app.state.websocket_manager = websocket_manager
    app.state.sse_manager = sse_manager
    app.state.analytics_service = analytics_service
    
    # Configure templates
    templates = Jinja2Templates(directory="app/templates")
    
    # GraphQL context provider
    async def get_graphql_context():
        return {
            "session_factory": SessionFactory,
            "feed_monitor": feed_monitor,
            "webhook_dispatcher": webhook_dispatcher,
            "categorizer": categorizer,
            "clustering_engine": clustering_engine,
            "subscription_manager": subscription_manager,
            "enrichment_service": enrichment_service,
            "lifecycle_tracker": lifecycle_tracker,
            "alert_system": alert_system,
            "sentiment_service": sentiment_service,
            "retention_service": retention_service,
            "backfill_service": backfill_service,
            "cache_service": cache_service,
            "bulk_ingestion_service": bulk_ingestion_service,
            "streaming_service": streaming_service,
        }
    
    # Add GraphQL router
    graphql_app = GraphQLRouter(schema, context_getter=get_graphql_context)
    app.include_router(graphql_app, prefix="/graphql")

    def _event_payload(event: EventORM) -> Dict[str, Any]:
        schema = _to_schema(event)
        data = schema.model_dump()
        return data

    async def _dispatch_event(event: EventORM, event_type: str) -> None:
        dispatcher = getattr(app.state, "webhook_dispatcher", None)
        if dispatcher and dispatcher.has_targets():
            await dispatcher.dispatch(event_type, _event_payload(event))

    def _apply_categorization(raw_category: Optional[str], title: Optional[str], description: Optional[str], metadata: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        meta = dict(metadata or {})
        categorizer = getattr(app.state, "categorizer", None)
        if categorizer:
            result = categorizer.categorize(
                raw_category=raw_category,
                title=title,
                description=description,
                metadata=meta,
            )
            meta.setdefault("raw_category", raw_category)
            classification = meta.setdefault("classification", {})
            classification.update(
                {
                    "raw_category": result.raw_category or raw_category,
                    "canonical_category": result.category,
                    "confidence": result.confidence,
                    "matched_keywords": result.matched_keywords,
                    "source": "heuristic",
                }
            )
            if result.tags:
                existing_tags = meta.get("tags")
                if not isinstance(existing_tags, list):
                    existing_tags = []
                meta["tags"] = sorted({*existing_tags, *result.tags})
            canonical = result.category or (raw_category or "other")
        else:
            canonical = raw_category or "other"
            meta.setdefault("raw_category", raw_category)
            classification = meta.setdefault("classification", {})
            classification.update(
                {
                    "raw_category": raw_category,
                    "canonical_category": canonical,
                    "confidence": 0.0,
                    "matched_keywords": [],
                    "source": "fallback",
                }
            )
        return canonical, meta

    @app.get("/", response_model=Dict[str, Any])
    async def root() -> Dict[str, Any]:
        return {
            "service": SERVICE_NAME,
            "status": "running",
            "version": app.version,
            "endpoints": {
                "health": "/health",
                "list_events": "/events",
                "create_event": "/events",
                "get_event": "/events/{event_id}",
                "update_event": "/events/{event_id}",
                "delete_event": "/events/{event_id}",
                "sync_events": "/events/sync",
                "sync_headlines": "/headlines/sync",
                "list_headlines": "/headlines",
                "feed_health": "/health/feeds",
                "categories": "/events/categories",
                "clusters": "/events/clusters",
                "cluster_detail": "/events/clusters/{cluster_id}",
                "symbol_clusters": "/events/clusters/symbol/{symbol}",
                "search_events": "/events/search",
                "graphql": "/graphql",
                "graphql_playground": "/graphql",
                "subscriptions": "/subscriptions",
                "create_subscription": "/subscriptions",
                "get_subscription": "/subscriptions/{subscription_id}",
                "update_subscription": "/subscriptions/{subscription_id}",
                "delete_subscription": "/subscriptions/{subscription_id}",
                "subscription_health": "/subscriptions/{subscription_id}/health",
                "market_context": "/enrichment/market-context/{symbol}",
                "enrich_event": "/enrichment/enrich-event",
                "batch_enrich": "/enrichment/batch-enrich",
                "enrichment_stats": "/enrichment/stats",
                "event_lifecycle": "/lifecycle/event/{event_id}",
                "update_lifecycle_status": "/lifecycle/event/{event_id}/status",
                "events_by_stage": "/lifecycle/events/by-stage/{stage}",
                "events_by_status": "/lifecycle/events/by-status/{status}",
                "lifecycle_stats": "/lifecycle/stats",
                "impact_analysis": "/lifecycle/impact-analysis",
                "alert_rules": "/alerts/rules",
                "create_alert_rule": "/alerts/rules",
                "get_alert_rule": "/alerts/rules/{rule_id}",
                "update_alert_rule": "/alerts/rules/{rule_id}",
                "delete_alert_rule": "/alerts/rules/{rule_id}",
                "alert_history": "/alerts/history",
                "alert_stats": "/alerts/stats",
                "event_sentiment": "/sentiment/events/{event_id}",
                "outcome_sentiment": "/sentiment/events/{event_id}/outcome",
                "sentiment_trends": "/sentiment/trends/{symbol}",
                "sentiment_stats": "/sentiment/stats",
                "backfill_symbol": "/backfill/symbols/{symbol}",
                "backfill_status": "/backfill/status/{symbol}",
                "backfill_active": "/backfill/active",
                "backfill_stats": "/backfill/stats",
            },
        }

    @app.get("/health", response_model=Dict[str, Any])
    async def health(session: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
        event_count = await session.scalar(select(func.count(EventORM.id)))
        headline_count = await session.scalar(select(func.count(EventHeadlineORM.id)))
        feed_health = {}
        if getattr(app.state, "feed_monitor", None):
            feed_health = await app.state.feed_monitor.snapshot()
        
        # Get cache health
        cache_health = {"enabled": False}
        if getattr(app.state, "cache_service", None):
            try:
                cache_stats = await app.state.cache_service.get_stats()
                cache_health = {
                    "enabled": cache_stats.get("enabled", False),
                    "status": "connected" if cache_stats.get("connection_status") == "connected" else "disconnected",
                    "total_keys": cache_stats.get("total_keys", 0),
                    "hit_rate": cache_stats.get("daily_stats", {}).get("hit_rate", 0),
                    "memory_usage_mb": round(cache_stats.get("memory_usage_bytes", 0) / 1024 / 1024, 2)
                }
            except Exception as e:
                cache_health = {"enabled": True, "status": "error", "error": str(e)}
        
        return {
            "status": "healthy",
            "service": SERVICE_NAME,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - app.state.start_time).total_seconds(),
            "event_count": event_count or 0,
            "headline_count": headline_count or 0,
            "feeds": feed_health,
            "cache": cache_health,
        }

    @app.get("/health/feeds", response_model=Dict[str, Any])
    async def feed_health() -> Dict[str, Any]:
        monitor = getattr(app.state, "feed_monitor", None)
        return await monitor.snapshot() if monitor else {}

    @app.get("/events/categories", response_model=Dict[str, Any])
    async def list_categories() -> Dict[str, Any]:
        categorizer = getattr(app.state, "categorizer", None)
        categories = categorizer.categories() if categorizer else []
        return {"categories": categories}

    @app.post("/events/sync", response_model=Dict[str, Any])
    async def sync_events() -> Dict[str, Any]:
        await calendar_ingestor.trigger_once()
        return {"status": "triggered"}

    @app.post("/headlines/sync", response_model=Dict[str, Any])
    async def sync_headlines() -> Dict[str, Any]:
        await headline_ingestor.trigger_once()
        return {"status": "triggered"}

    @app.get("/events/clusters", response_model=Dict[str, Any])
    async def list_clusters(
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        cluster_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        clustering_engine = getattr(app.state, "clustering_engine", None)
        if not clustering_engine:
            return {"clusters": []}
            
        # Parse datetime parameters
        start_dt = None
        end_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except ValueError:
                pass
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            except ValueError:
                pass
                
        clusters = await clustering_engine.cluster_events(start_dt, end_dt)
        
        # Filter by cluster type if specified
        if cluster_type:
            clusters = [c for c in clusters if c.cluster_type == cluster_type]
            
        return {
            "clusters": [cluster.to_dict() for cluster in clusters],
            "count": len(clusters),
            "parameters": {
                "start_time": start_time,
                "end_time": end_time,
                "cluster_type": cluster_type,
            }
        }

    @app.get("/events/clusters/{cluster_id}", response_model=Dict[str, Any])
    async def get_cluster(cluster_id: str) -> Dict[str, Any]:
        clustering_engine = getattr(app.state, "clustering_engine", None)
        if not clustering_engine:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Clustering engine not available")
            
        cluster = await clustering_engine.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cluster not found")
            
        return cluster.to_dict()

    @app.get("/events/clusters/symbol/{symbol}", response_model=Dict[str, Any])
    async def get_symbol_clusters(symbol: str) -> Dict[str, Any]:
        clustering_engine = getattr(app.state, "clustering_engine", None)
        if not clustering_engine:
            return {"clusters": [], "symbol": symbol}
            
        clusters = await clustering_engine.get_clusters_for_symbol(symbol.upper())
        return {
            "clusters": [cluster.to_dict() for cluster in clusters],
            "symbol": symbol.upper(),
            "count": len(clusters),
        }

    @app.post("/events/clusters/analyze", response_model=Dict[str, Any])
    async def analyze_clusters(
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        clustering_engine = getattr(app.state, "clustering_engine", None)
        if not clustering_engine:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Clustering engine not available")
            
        # Parse datetime parameters
        start_dt = None
        end_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except ValueError:
                pass
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            except ValueError:
                pass
                
        clusters = await clustering_engine.cluster_events(start_dt, end_dt)
        
        # Generate analysis summary
        cluster_types = {}
        total_events = 0
        high_impact_clusters = []
        
        for cluster in clusters:
            cluster_types[cluster.cluster_type] = cluster_types.get(cluster.cluster_type, 0) + 1
            total_events += len(cluster.event_ids)
            if cluster.cluster_score > 0.7:
                high_impact_clusters.append(cluster.to_dict())
                
        return {
            "analysis": {
                "total_clusters": len(clusters),
                "total_events": total_events,
                "cluster_types": cluster_types,
                "high_impact_clusters": high_impact_clusters,
                "analysis_period": {
                    "start_time": start_time,
                    "end_time": end_time,
                }
            },
            "clusters": [cluster.to_dict() for cluster in clusters]
        }

    @app.get("/events/search", response_model=Dict[str, Any])
    async def search_events(
        # Multi-value filters
        symbols: Optional[str] = Query(default=None, description="Comma-separated list of symbols (e.g. AAPL,MSFT,GOOGL)"),
        categories: Optional[str] = Query(default=None, description="Comma-separated list of categories (e.g. earnings,regulatory,product_launch)"),
        statuses: Optional[str] = Query(default=None, description="Comma-separated list of statuses (e.g. scheduled,completed,cancelled)"),
        sources: Optional[str] = Query(default=None, description="Comma-separated list of sources"),
        
        # Impact score filters
        impact_min: Optional[int] = Query(default=None, ge=1, le=10, description="Minimum impact score (1-10)"),
        impact_max: Optional[int] = Query(default=None, ge=1, le=10, description="Maximum impact score (1-10)"),
        impact_exact: Optional[int] = Query(default=None, ge=1, le=10, description="Exact impact score"),
        
        # Date range filters
        scheduled_after: Optional[datetime] = Query(default=None, description="Events scheduled on/after this timestamp"),
        scheduled_before: Optional[datetime] = Query(default=None, description="Events scheduled on/before this timestamp"),
        created_after: Optional[datetime] = Query(default=None, description="Events created on/after this timestamp"),
        created_before: Optional[datetime] = Query(default=None, description="Events created on/before this timestamp"),
        updated_after: Optional[datetime] = Query(default=None, description="Events updated on/after this timestamp"),
        
        # Text search
        search_text: Optional[str] = Query(default=None, description="Search in title, description, and metadata"),
        title_contains: Optional[str] = Query(default=None, description="Search in title only"),
        description_contains: Optional[str] = Query(default=None, description="Search in description only"),
        
        # Relationship filters
        has_headlines: Optional[bool] = Query(default=None, description="Filter events that have/don't have headlines"),
        has_metadata: Optional[bool] = Query(default=None, description="Filter events that have/don't have metadata"),
        has_external_id: Optional[bool] = Query(default=None, description="Filter events that have/don't have external ID"),
        
        # Clustering filters
        in_clusters: Optional[bool] = Query(default=None, description="Filter events that are/aren't in clusters"),
        cluster_types: Optional[str] = Query(default=None, description="Comma-separated cluster types to include"),
        
        # Pagination and sorting
        limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of results to return"),
        offset: int = Query(default=0, ge=0, description="Number of results to skip"),
        sort_by: str = Query(default="scheduled_at", description="Field to sort by: scheduled_at, created_at, updated_at, impact_score, symbol"),
        sort_order: str = Query(default="desc", description="Sort order: asc or desc"),
        
        # Response options
        include_headlines: bool = Query(default=False, description="Include related headlines in response"),
        include_clusters: bool = Query(default=False, description="Include cluster information in response"),
        include_metadata: bool = Query(default=False, description="Include full metadata in response"),
        
        session: AsyncSession = Depends(get_session),
    ) -> Dict[str, Any]:
        """Advanced event search with comprehensive filtering and pagination."""
        
        # Build base query
        stmt = select(EventORM)
        conditions = []
        
        # Multi-value filters
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            if symbol_list:
                conditions.append(EventORM.symbol.in_(symbol_list))
        
        if categories:
            category_list = [c.strip().lower() for c in categories.split(",") if c.strip()]
            if category_list:
                conditions.append(EventORM.category.in_(category_list))
        
        if statuses:
            status_list = [s.strip().lower() for s in statuses.split(",") if s.strip()]
            if status_list:
                conditions.append(EventORM.status.in_(status_list))
        
        if sources:
            source_list = [s.strip() for s in sources.split(",") if s.strip()]
            if source_list:
                conditions.append(EventORM.source.in_(source_list))
        
        # Impact score filters
        if impact_exact is not None:
            conditions.append(EventORM.impact_score == impact_exact)
        else:
            if impact_min is not None:
                conditions.append(EventORM.impact_score >= impact_min)
            if impact_max is not None:
                conditions.append(EventORM.impact_score <= impact_max)
        
        # Date range filters
        if scheduled_after:
            conditions.append(EventORM.scheduled_at >= scheduled_after)
        if scheduled_before:
            conditions.append(EventORM.scheduled_at <= scheduled_before)
        if created_after:
            conditions.append(EventORM.created_at >= created_after)
        if created_before:
            conditions.append(EventORM.created_at <= created_before)
        if updated_after:
            conditions.append(EventORM.updated_at >= updated_after)
        
        # Text search filters
        if search_text:
            search_term = f"%{search_text}%"
            conditions.append(
                or_(
                    EventORM.title.ilike(search_term),
                    EventORM.description.ilike(search_term),
                    EventORM.symbol.ilike(search_term),
                    EventORM.metadata_json.astext.ilike(search_term)
                )
            )
        
        if title_contains:
            conditions.append(EventORM.title.ilike(f"%{title_contains}%"))
        
        if description_contains:
            conditions.append(EventORM.description.ilike(f"%{description_contains}%"))
        
        # Relationship filters
        if has_headlines is not None:
            if has_headlines:
                stmt = stmt.join(EventHeadlineORM, EventORM.id == EventHeadlineORM.event_id)
            else:
                stmt = stmt.outerjoin(EventHeadlineORM, EventORM.id == EventHeadlineORM.event_id)
                conditions.append(EventHeadlineORM.id.is_(None))
        
        if has_metadata is not None:
            if has_metadata:
                conditions.append(EventORM.metadata_json.is_not(None))
            else:
                conditions.append(EventORM.metadata_json.is_(None))
        
        if has_external_id is not None:
            if has_external_id:
                conditions.append(EventORM.external_id.is_not(None))
            else:
                conditions.append(EventORM.external_id.is_(None))
        
        # Apply all conditions
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        # Get total count before pagination
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await session.execute(count_stmt)
        total_count = total_result.scalar()
        
        # Apply sorting
        if sort_by == "scheduled_at":
            order_field = EventORM.scheduled_at
        elif sort_by == "created_at":
            order_field = EventORM.created_at
        elif sort_by == "updated_at":
            order_field = EventORM.updated_at
        elif sort_by == "impact_score":
            order_field = EventORM.impact_score
        elif sort_by == "symbol":
            order_field = EventORM.symbol
        else:
            order_field = EventORM.scheduled_at
        
        if sort_order.lower() == "asc":
            stmt = stmt.order_by(order_field.asc())
        else:
            stmt = stmt.order_by(order_field.desc())
        
        # Apply pagination
        stmt = stmt.offset(offset).limit(limit)
        
        # Execute query
        result = await session.execute(stmt)
        events = result.scalars().all()
        
        # Build response
        response_events = []
        clustering_engine = getattr(app.state, "clustering_engine", None)
        
        for event in events:
            event_data = _to_schema(event).model_dump()
            
            # Include metadata if requested
            if include_metadata and event.metadata_json:
                event_data["metadata"] = event.metadata_json
            
            # Include headlines if requested
            if include_headlines:
                headline_stmt = select(EventHeadlineORM).where(EventHeadlineORM.event_id == event.id)
                headline_result = await session.execute(headline_stmt)
                headlines = headline_result.scalars().all()
                event_data["headlines"] = [
                    {
                        "id": h.id,
                        "headline": h.headline,
                        "published_at": h.published_at.isoformat(),
                        "url": h.url,
                        "source": h.source,
                    }
                    for h in headlines
                ]
            
            # Include cluster information if requested
            if include_clusters and clustering_engine:
                event_clusters = []
                for cluster in clustering_engine._clusters.values():
                    if event.id in cluster.event_ids:
                        event_clusters.append({
                            "cluster_id": cluster.cluster_id,
                            "cluster_type": cluster.cluster_type,
                            "cluster_score": cluster.cluster_score,
                            "related_symbols": cluster.related_symbols,
                        })
                event_data["clusters"] = event_clusters
            
            response_events.append(event_data)
        
        # Apply cluster filtering if requested
        if in_clusters is not None or cluster_types:
            if clustering_engine:
                cluster_event_ids = set()
                for cluster in clustering_engine._clusters.values():
                    # Filter by cluster types if specified
                    if cluster_types:
                        requested_types = [t.strip() for t in cluster_types.split(",") if t.strip()]
                        if cluster.cluster_type not in requested_types:
                            continue
                    cluster_event_ids.update(cluster.event_ids)
                
                # Filter events based on cluster membership
                if in_clusters is not None:
                    if in_clusters:
                        response_events = [e for e in response_events if e["id"] in cluster_event_ids]
                    else:
                        response_events = [e for e in response_events if e["id"] not in cluster_event_ids]
        
        # Build pagination info
        has_next = offset + limit < total_count
        has_previous = offset > 0
        total_pages = (total_count + limit - 1) // limit if total_count > 0 else 0
        current_page = (offset // limit) + 1 if total_count > 0 else 0
        
        return {
            "events": response_events,
            "pagination": {
                "total_count": total_count,
                "returned_count": len(response_events),
                "limit": limit,
                "offset": offset,
                "has_next": has_next,
                "has_previous": has_previous,
                "total_pages": total_pages,
                "current_page": current_page,
            },
            "filters_applied": {
                "symbols": symbols.split(",") if symbols else None,
                "categories": categories.split(",") if categories else None,
                "statuses": statuses.split(",") if statuses else None,
                "impact_range": [impact_min, impact_max] if impact_min or impact_max else None,
                "date_range": {
                    "scheduled_after": scheduled_after.isoformat() if scheduled_after else None,
                    "scheduled_before": scheduled_before.isoformat() if scheduled_before else None,
                },
                "search_text": search_text,
                "sorting": {"field": sort_by, "order": sort_order},
            },
            "includes": {
                "headlines": include_headlines,
                "clusters": include_clusters,
                "metadata": include_metadata,
            }
        }

    @app.get("/events", response_model=List[Event])
    async def list_events(
        symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
        category: Optional[str] = Query(default=None, description="Filter by category"),
        status_filter: Optional[str] = Query(default=None, alias="status", description="Filter by status"),
        start_after: Optional[datetime] = Query(default=None, description="Events scheduled on/after this timestamp"),
        end_before: Optional[datetime] = Query(default=None, description="Events scheduled on/before this timestamp"),
        session: AsyncSession = Depends(get_session),
    ) -> List[Event]:
        # Build cache key from query parameters
        cache_params = {
            'symbol': symbol,
            'category': category,
            'status': status_filter,
            'start_after': start_after.isoformat() if start_after else None,
            'end_before': end_before.isoformat() if end_before else None
        }
        # Remove None values
        cache_params = {k: v for k, v in cache_params.items() if v is not None}
        
        # Try cache first
        cached_result = await cache_service.get(CacheKeyType.EVENT_LIST, "list", **cache_params)
        if cached_result:
            return [Event(**event) for event in cached_result]
        
        # Fetch from database
        stmt = select(EventORM)
        if symbol:
            stmt = stmt.where(EventORM.symbol.ilike(symbol))
        if category:
            stmt = stmt.where(EventORM.category.ilike(category))
        if status_filter:
            stmt = stmt.where(EventORM.status.ilike(status_filter))
        if start_after:
            stmt = stmt.where(EventORM.scheduled_at >= start_after)
        if end_before:
            stmt = stmt.where(EventORM.scheduled_at <= end_before)
        stmt = stmt.order_by(EventORM.scheduled_at)
        result = await session.execute(stmt)
        events = result.scalars().all()
        
        # Convert to schema
        result_list = [_to_schema(event) for event in events]
        
        # Cache the result (shorter TTL for lists)
        await cache_service.set(
            CacheKeyType.EVENT_LIST, 
            "list", 
            [event.dict() for event in result_list],
            ttl=300,  # 5 minutes for list queries
            **cache_params
        )
        
        return result_list

    @app.get("/headlines", response_model=List[EventHeadline])
    async def list_headlines(
        symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
        since: Optional[datetime] = Query(default=None, description="Headlines published after timestamp"),
        session: AsyncSession = Depends(get_session),
    ) -> List[EventHeadline]:
        stmt = select(EventHeadlineORM)
        if symbol:
            stmt = stmt.where(EventHeadlineORM.symbol.ilike(symbol))
        if since:
            stmt = stmt.where(EventHeadlineORM.published_at >= since)
        stmt = stmt.order_by(EventHeadlineORM.published_at.desc())
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return [EventHeadline.model_validate(row) for row in rows]

    @app.get("/events/{event_id}/headlines", response_model=List[EventHeadline])
    async def get_event_headlines(event_id: str, session: AsyncSession = Depends(get_session)) -> List[EventHeadline]:
        stmt = (
            select(EventHeadlineORM)
            .where(EventHeadlineORM.event_id == event_id)
            .order_by(EventHeadlineORM.published_at.desc())
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()
        return [EventHeadline.model_validate(row) for row in rows]

    @app.post("/events", response_model=Event, status_code=status.HTTP_201_CREATED)
    async def create_event(payload: EventCreate, session: AsyncSession = Depends(get_session)) -> Event:
        now = datetime.utcnow()
        metadata = dict(payload.metadata or {})
        canonical_category, metadata = _apply_categorization(
            payload.category, payload.title, payload.description, metadata
        )
        event = EventORM(
            symbol=payload.symbol,
            title=payload.title,
            category=canonical_category,
            scheduled_at=payload.scheduled_at,
            timezone=payload.timezone,
            description=payload.description,
            metadata_json=metadata,
            status=payload.status,
            source=payload.source,
            external_id=payload.external_id,
            created_at=now,
            updated_at=now,
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        
        # Enrich event with market context
        enrichment_service = getattr(app.state, "enrichment_service", None)
        if enrichment_service:
            try:
                enriched_payload = await enrichment_service.enrich_event(_event_payload(event))
                if enriched_payload.get("metadata") != event.metadata_json:
                    event.metadata_json = enriched_payload.get("metadata", {})
                    await session.commit()
                    await session.refresh(event)
            except Exception as e:
                logger.warning(f"Failed to enrich event {event.id}: {e}")
        
        await _dispatch_event(event, "event.created")
        # Start lifecycle tracking
        lifecycle_tracker = getattr(app.state, "lifecycle_tracker", None)
        if lifecycle_tracker:
            await lifecycle_tracker.track_event(event)
        
        # Evaluate for alerts
        alert_system = getattr(app.state, "alert_system", None)
        if alert_system:
            await alert_system.evaluate_event(_event_payload(event), "event.created")
        
        # Perform sentiment analysis
        sentiment_service = getattr(app.state, "sentiment_service", None)
        if sentiment_service:
            try:
                await sentiment_service.analyze_event_sentiment(event, session)
            except Exception as e:
                logger.warning(f"Failed to analyze sentiment for event {event.id}: {e}")
        
        # Check if this is a new symbol and trigger backfill if needed
        backfill_service = getattr(app.state, "backfill_service", None)
        if backfill_service:
            try:
                # Check if this symbol has any historical events
                existing_count = await session.scalar(
                    select(func.count(EventORM.id)).where(
                        and_(
                            EventORM.symbol == event.symbol,
                            EventORM.id != event.id  # Exclude current event
                        )
                    )
                )
                
                # If no existing events, trigger backfill
                if existing_count == 0:
                    logger.info(f"New symbol detected: {event.symbol}, triggering historical backfill")
                    await backfill_service.request_backfill(
                        symbol=event.symbol,
                        priority=2  # Medium priority for automatic backfill
                    )
            except Exception as e:
                logger.warning(f"Failed to trigger automatic backfill for {event.symbol}: {e}")
        
        # Notify subscriptions
        subscription_manager = getattr(app.state, "subscription_manager", None)
        if subscription_manager:
            await subscription_manager.notify_event(_event_payload(event), EventType.EVENT_CREATED)
        
        # Publish to streaming service
        streaming_service = getattr(app.state, "streaming_service", None)
        if streaming_service:
            await streaming_service.publish_event(
                StreamEventType.EVENT_CREATED,
                _event_payload(event),
                source="event_api",
                metadata={"api_endpoint": "create_event"}
            )
        
        return _to_schema(event)

    @app.get("/events/{event_id}", response_model=Event)
    async def get_event(event_id: str, session: AsyncSession = Depends(get_session)) -> Event:
        # Try cache first
        cached_event = await cache_service.get(CacheKeyType.EVENT, event_id)
        if cached_event:
            return Event(**cached_event)
        
        # Fetch from database
        event = await session.get(EventORM, event_id)
        if not event:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
        
        result = _to_schema(event)
        
        # Cache the result
        await cache_service.set(CacheKeyType.EVENT, event_id, result.dict())
        
        return result

    @app.put("/events/{event_id}", response_model=Event)
    async def replace_event(
        event_id: str, payload: EventCreate, session: AsyncSession = Depends(get_session)
    ) -> Event:
        event = await session.get(EventORM, event_id)
        if not event:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
        metadata = dict(payload.metadata or {})
        canonical_category, metadata = _apply_categorization(
            payload.category, payload.title, payload.description, metadata
        )
        event.symbol = payload.symbol
        event.title = payload.title
        event.category = canonical_category
        event.scheduled_at = payload.scheduled_at
        event.timezone = payload.timezone
        event.description = payload.description
        event.metadata_json = metadata
        event.status = payload.status
        event.source = payload.source
        event.external_id = payload.external_id
        event.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(event)
        await _dispatch_event(event, "event.replaced")
        
        # Evaluate for alerts
        alert_system = getattr(app.state, "alert_system", None)
        if alert_system:
            await alert_system.evaluate_event(_event_payload(event), "event.replaced")
        
        # Notify subscriptions
        subscription_manager = getattr(app.state, "subscription_manager", None)
        if subscription_manager:
            await subscription_manager.notify_event(_event_payload(event), EventType.EVENT_UPDATED)
        
        # Invalidate cache
        await cache_service.invalidate_event(event_id, event.symbol)
        
        # Publish to streaming service
        streaming_service = getattr(app.state, "streaming_service", None)
        if streaming_service:
            await streaming_service.publish_event(
                StreamEventType.EVENT_UPDATED,
                _event_payload(event),
                source="event_api",
                metadata={"api_endpoint": "update_event"}
            )
        
        return _to_schema(event)

    @app.patch("/events/{event_id}", response_model=Event)
    async def update_event(
        event_id: str, payload: EventUpdate, session: AsyncSession = Depends(get_session)
    ) -> Event:
        event = await session.get(EventORM, event_id)
        if not event:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
        data = payload.model_dump(exclude_unset=True)
        for key, value in data.items():
            if key == "metadata":
                setattr(event, "metadata_json", value)
            else:
                setattr(event, key, value)
        metadata = event.metadata_json or {}
        if not isinstance(metadata, dict):
            metadata = {}
        raw_category = data.get("category") or metadata.get("classification", {}).get("raw_category") or event.category
        title = data.get("title") or event.title
        description = data.get("description") or event.description
        canonical_category, metadata = _apply_categorization(raw_category, title, description, metadata)
        event.category = canonical_category
        event.metadata_json = metadata
        event.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(event)
        await _dispatch_event(event, "event.updated")
        
        # Evaluate for alerts
        alert_system = getattr(app.state, "alert_system", None)
        if alert_system:
            await alert_system.evaluate_event(_event_payload(event), "event.updated")
        
        # Re-analyze sentiment after updates
        sentiment_service = getattr(app.state, "sentiment_service", None)
        if sentiment_service:
            try:
                await sentiment_service.analyze_event_sentiment(event, session, force_refresh=True)
            except Exception as e:
                logger.warning(f"Failed to re-analyze sentiment for updated event {event.id}: {e}")
        
        # Notify subscriptions
        subscription_manager = getattr(app.state, "subscription_manager", None)
        if subscription_manager:
            await subscription_manager.notify_event(_event_payload(event), EventType.EVENT_UPDATED)
        
        # Invalidate cache
        await cache_service.invalidate_event(event_id, event.symbol)
        
        # Publish to streaming service
        streaming_service = getattr(app.state, "streaming_service", None)
        if streaming_service:
            await streaming_service.publish_event(
                StreamEventType.EVENT_UPDATED,
                _event_payload(event),
                source="event_api",
                metadata={"api_endpoint": "update_event"}
            )
        
        return _to_schema(event)

    @app.delete("/events/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_event(event_id: str, session: AsyncSession = Depends(get_session)) -> None:
        event = await session.get(EventORM, event_id)
        if not event:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
        payload = _event_payload(event)
        await session.delete(event)
        await session.commit()
        dispatcher = getattr(app.state, "webhook_dispatcher", None)
        if dispatcher and dispatcher.has_targets():
            await dispatcher.dispatch("event.deleted", payload)
        # Notify subscriptions
        subscription_manager = getattr(app.state, "subscription_manager", None)
        if subscription_manager:
            await subscription_manager.notify_event(payload, EventType.EVENT_UPDATED)
        
        # Invalidate cache
        await cache_service.invalidate_event(event_id, payload.get('symbol'))
        
        # Publish to streaming service
        streaming_service = getattr(app.state, "streaming_service", None)
        if streaming_service:
            await streaming_service.publish_event(
                StreamEventType.EVENT_DELETED,
                payload,
                source="event_api",
                metadata={"api_endpoint": "delete_event"}
            )
        
        return None

    @app.patch("/events/{event_id}/impact", response_model=ImpactScoreResponse)
    async def update_event_impact(event_id: str, payload: ImpactScoreUpdate, session: AsyncSession = Depends(get_session)) -> ImpactScoreResponse:
        event = await session.get(EventORM, event_id)
        if not event:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
        event.impact_score = payload.impact_score
        await session.commit()
        await session.refresh(event)
        
        # Evaluate for alerts
        alert_system = getattr(app.state, "alert_system", None)
        if alert_system:
            await alert_system.evaluate_event(_event_payload(event), "event.impact_updated")
        
        dispatcher = getattr(app.state, "webhook_dispatcher", None)
        if dispatcher and dispatcher.has_targets():
            await dispatcher.dispatch("event.impact_updated", _event_payload(event))
        # Notify subscriptions
        subscription_manager = getattr(app.state, "subscription_manager", None)
        if subscription_manager:
            await subscription_manager.notify_event(_event_payload(event), EventType.EVENT_IMPACT_CHANGED)
        return ImpactScoreResponse(event_id=event.id, impact_score=event.impact_score)

    # ===== Alert Management Endpoints =====
    
    @app.get("/alerts/rules", response_model=Dict[str, Any])
    async def list_alert_rules() -> Dict[str, Any]:
        """List all alert rules."""
        alert_system = getattr(app.state, "alert_system", None)
        if not alert_system:
            raise HTTPException(status_code=503, detail="Alert system not available")
        
        rules = alert_system.list_rules()
        return {
            "rules": [rule.to_dict() for rule in rules],
            "count": len(rules)
        }
    
    @app.post("/alerts/rules", response_model=Dict[str, Any])
    async def create_alert_rule(rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new alert rule."""
        alert_system = getattr(app.state, "alert_system", None)
        if not alert_system:
            raise HTTPException(status_code=503, detail="Alert system not available")
        
        try:
            rule_id = await alert_system.add_rule(rule_data)
            return {"rule_id": rule_id, "message": "Alert rule created successfully"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to create alert rule: {str(e)}")
    
    @app.get("/alerts/rules/{rule_id}", response_model=Dict[str, Any])
    async def get_alert_rule(rule_id: str) -> Dict[str, Any]:
        """Get a specific alert rule."""
        alert_system = getattr(app.state, "alert_system", None)
        if not alert_system:
            raise HTTPException(status_code=503, detail="Alert system not available")
        
        rule = alert_system.get_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail="Alert rule not found")
        
        return rule.to_dict()
    
    @app.put("/alerts/rules/{rule_id}", response_model=Dict[str, Any])
    async def update_alert_rule(rule_id: str, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing alert rule."""
        alert_system = getattr(app.state, "alert_system", None)
        if not alert_system:
            raise HTTPException(status_code=503, detail="Alert system not available")
        
        try:
            success = await alert_system.update_rule(rule_id, rule_data)
            if not success:
                raise HTTPException(status_code=404, detail="Alert rule not found")
            return {"message": "Alert rule updated successfully"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to update alert rule: {str(e)}")
    
    @app.delete("/alerts/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_alert_rule(rule_id: str) -> None:
        """Delete an alert rule."""
        alert_system = getattr(app.state, "alert_system", None)
        if not alert_system:
            raise HTTPException(status_code=503, detail="Alert system not available")
        
        success = await alert_system.remove_rule(rule_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert rule not found")
    
    @app.get("/alerts/history", response_model=Dict[str, Any])
    async def get_alert_history(
        limit: int = 100,
        offset: int = 0,
        severity: Optional[str] = None,
        channel: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get alert history with optional filtering."""
        alert_system = getattr(app.state, "alert_system", None)
        if not alert_system:
            raise HTTPException(status_code=503, detail="Alert system not available")
        
        history = await alert_system.get_alert_history(limit, offset, severity, channel)
        return {
            "alerts": [alert.to_dict() for alert in history],
            "count": len(history),
            "limit": limit,
            "offset": offset
        }
    
    @app.get("/alerts/stats", response_model=Dict[str, Any])
    async def get_alert_stats() -> Dict[str, Any]:
        """Get alert system statistics."""
        alert_system = getattr(app.state, "alert_system", None)
        if not alert_system:
            raise HTTPException(status_code=503, detail="Alert system not available")
        
        stats = await alert_system.get_stats()
        return stats

    # ===== Sentiment Analysis Endpoints =====
    
    @app.get("/sentiment/events/{event_id}", response_model=Dict[str, Any])
    async def analyze_event_sentiment(
        event_id: str, 
        force_refresh: bool = Query(False, description="Force refresh of cached analysis"),
        session: AsyncSession = Depends(get_session)
    ) -> Dict[str, Any]:
        """Analyze sentiment for a specific event across all timeframes."""
        sentiment_service = getattr(app.state, "sentiment_service", None)
        if not sentiment_service:
            raise HTTPException(status_code=503, detail="Sentiment service not available")
        
        event = await session.get(EventORM, event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        analysis = await sentiment_service.analyze_event_sentiment(event, session, force_refresh)
        if not analysis:
            raise HTTPException(status_code=503, detail="Sentiment analysis unavailable")
        
        return {
            "event_id": analysis.event_id,
            "symbol": analysis.symbol,
            "category": analysis.category,
            "analyzed_at": analysis.analyzed_at.isoformat(),
            "overall_sentiment": {
                "compound": analysis.overall_sentiment.compound,
                "label": analysis.overall_sentiment.label,
                "confidence": analysis.overall_sentiment.confidence,
                "volume": analysis.overall_sentiment.volume
            },
            "sentiment_momentum": analysis.sentiment_momentum,
            "sentiment_divergence": analysis.sentiment_divergence,
            "outcome_prediction": analysis.outcome_prediction,
            "prediction_confidence": analysis.prediction_confidence,
            "timeframes": {
                k: {
                    "compound": v.compound,
                    "label": v.label,
                    "confidence": v.confidence,
                    "volume": v.volume
                } for k, v in analysis.timeframes.items()
            },
            "sources": {
                k: {
                    "compound": v.compound,
                    "label": v.label,
                    "confidence": v.confidence,
                    "volume": v.volume
                } for k, v in analysis.sources.items()
            }
        }
    
    @app.get("/sentiment/events/{event_id}/outcome", response_model=Dict[str, Any])
    async def analyze_event_outcome_sentiment(
        event_id: str,
        session: AsyncSession = Depends(get_session)
    ) -> Dict[str, Any]:
        """Analyze sentiment specifically around event outcomes."""
        sentiment_service = getattr(app.state, "sentiment_service", None)
        if not sentiment_service:
            raise HTTPException(status_code=503, detail="Sentiment service not available")
        
        event = await session.get(EventORM, event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        outcome_sentiment = await sentiment_service.analyze_event_outcome_sentiment(event, session)
        if not outcome_sentiment:
            return {"message": "No outcome sentiment data available", "event_id": event_id}
        
        return {
            "event_id": event_id,
            "outcome_sentiment": {
                "compound": outcome_sentiment.compound,
                "positive": outcome_sentiment.positive,
                "negative": outcome_sentiment.negative,
                "neutral": outcome_sentiment.neutral,
                "label": outcome_sentiment.label,
                "confidence": outcome_sentiment.confidence,
                "volume": outcome_sentiment.volume,
                "source": outcome_sentiment.source,
                "metadata": outcome_sentiment.metadata
            }
        }
    
    @app.get("/sentiment/trends/{symbol}", response_model=Dict[str, Any])
    async def get_sentiment_trends(
        symbol: str,
        days: int = Query(7, ge=1, le=30, description="Number of days to analyze")
    ) -> Dict[str, Any]:
        """Get sentiment trends for a symbol over time."""
        sentiment_service = getattr(app.state, "sentiment_service", None)
        if not sentiment_service:
            raise HTTPException(status_code=503, detail="Sentiment service not available")
        
        trends = await sentiment_service.get_sentiment_trends(symbol.upper(), days)
        return {
            "symbol": symbol.upper(),
            "timeframe_days": days,
            "trends": trends
        }
    
    @app.get("/sentiment/stats", response_model=Dict[str, Any])
    async def get_sentiment_stats() -> Dict[str, Any]:
        """Get sentiment analysis statistics."""
        sentiment_service = getattr(app.state, "sentiment_service", None)
        if not sentiment_service:
            raise HTTPException(status_code=503, detail="Sentiment service not available")
        
        # Get stats from sentiment service configuration
        return {
            "service": "event-sentiment-integration",
            "enabled": sentiment_service.enabled,
            "sentiment_service_url": sentiment_service.sentiment_service_url,
            "configuration": {
                "pre_event_hours": sentiment_service.pre_event_hours,
                "post_event_hours": sentiment_service.post_event_hours,
                "event_window_hours": sentiment_service.event_window_hours,
                "timeout": sentiment_service.timeout
            },
            "cache_stats": {
                "cached_analyses": len(sentiment_service._analysis_cache),
                "cache_ttl_seconds": sentiment_service._cache_ttl
            }
        }

    # ===== Historical Data Backfill Endpoints =====
    
    @app.post("/backfill/symbols/{symbol}", response_model=Dict[str, Any])
    async def request_symbol_backfill(
        symbol: str,
        start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
        end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
        categories: Optional[str] = Query(None, description="Comma-separated list of categories"),
        sources: Optional[str] = Query(None, description="Comma-separated list of sources"),
        priority: int = Query(1, ge=1, le=3, description="Priority (1=high, 2=medium, 3=low)")
    ) -> Dict[str, Any]:
        """Request historical data backfill for a specific symbol."""
        backfill_service = getattr(app.state, "backfill_service", None)
        if not backfill_service:
            raise HTTPException(status_code=503, detail="Backfill service not available")
        
        # Parse dates
        start_dt = None
        end_dt = None
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format (use YYYY-MM-DD)")
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format (use YYYY-MM-DD)")
        
        # Parse categories and sources
        category_list = None
        if categories:
            category_list = [c.strip() for c in categories.split(",") if c.strip()]
        
        source_list = None
        if sources:
            source_list = [s.strip() for s in sources.split(",") if s.strip()]
        
        try:
            request_id = await backfill_service.request_backfill(
                symbol=symbol.upper(),
                start_date=start_dt,
                end_date=end_dt,
                categories=category_list,
                sources=source_list,
                priority=priority
            )
            
            return {
                "message": f"Backfill requested for {symbol.upper()}",
                "request_id": request_id,
                "symbol": symbol.upper(),
                "parameters": {
                    "start_date": start_dt.isoformat() if start_dt else None,
                    "end_date": end_dt.isoformat() if end_dt else None,
                    "categories": category_list,
                    "sources": source_list,
                    "priority": priority
                }
            }
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backfill request failed: {str(e)}")
    
    @app.get("/backfill/status/{symbol}", response_model=Dict[str, Any])
    async def get_backfill_status(symbol: str) -> Dict[str, Any]:
        """Get current backfill status for a symbol."""
        backfill_service = getattr(app.state, "backfill_service", None)
        if not backfill_service:
            raise HTTPException(status_code=503, detail="Backfill service not available")
        
        status = await backfill_service.get_backfill_status(symbol.upper())
        
        if not status:
            return {
                "symbol": symbol.upper(),
                "status": "not_found",
                "message": "No active backfill found for this symbol"
            }
        
        return {
            "symbol": status.symbol,
            "status": "in_progress",
            "progress": {
                "total_requests": status.total_requests,
                "completed_requests": status.completed_requests,
                "completion_percentage": (status.completed_requests / status.total_requests * 100) if status.total_requests > 0 else 0,
                "current_source": status.current_source,
                "current_date_range": status.current_date_range,
                "events_processed": status.events_processed,
                "started_at": status.started_at.isoformat(),
                "estimated_completion": status.estimated_completion.isoformat() if status.estimated_completion else None
            }
        }
    
    @app.get("/backfill/active", response_model=Dict[str, Any])
    async def list_active_backfills() -> Dict[str, Any]:
        """List all active backfill operations."""
        backfill_service = getattr(app.state, "backfill_service", None)
        if not backfill_service:
            raise HTTPException(status_code=503, detail="Backfill service not available")
        
        active_backfills = await backfill_service.list_active_backfills()
        
        return {
            "active_backfills": [
                {
                    "symbol": backfill.symbol,
                    "total_requests": backfill.total_requests,
                    "completed_requests": backfill.completed_requests,
                    "completion_percentage": (backfill.completed_requests / backfill.total_requests * 100) if backfill.total_requests > 0 else 0,
                    "current_source": backfill.current_source,
                    "events_processed": backfill.events_processed,
                    "started_at": backfill.started_at.isoformat(),
                    "estimated_completion": backfill.estimated_completion.isoformat() if backfill.estimated_completion else None
                }
                for backfill in active_backfills
            ],
            "count": len(active_backfills)
        }
    
    @app.get("/backfill/stats", response_model=Dict[str, Any])
    async def get_backfill_stats() -> Dict[str, Any]:
        """Get backfill service statistics."""
        backfill_service = getattr(app.state, "backfill_service", None)
        if not backfill_service:
            raise HTTPException(status_code=503, detail="Backfill service not available")
        
        stats = await backfill_service.get_backfill_statistics()
        return stats

    # ===== Event Subscription Endpoints =====
    
    @app.post("/subscriptions", response_model=SubscriptionResponse)
    async def create_subscription(request: SubscriptionRequest) -> SubscriptionResponse:
        """Create a new event subscription for real-time notifications."""
        subscription = app.state.subscription_manager.create_subscription(request)
        
        return SubscriptionResponse(
            id=subscription.id,
            service_name=subscription.service_name,
            webhook_url=subscription.webhook_url,
            filters={
                "symbols": subscription.filters.symbols,
                "categories": subscription.filters.categories,
                "min_impact_score": subscription.filters.min_impact_score,
                "max_impact_score": subscription.filters.max_impact_score,
                "event_types": [et.value for et in subscription.filters.event_types] if subscription.filters.event_types else None,
                "statuses": subscription.filters.statuses,
            },
            status=subscription.status,
            created_at=subscription.created_at,
            updated_at=subscription.updated_at,
            last_notification=subscription.last_notification,
            failure_count=subscription.failure_count,
        )
    
    @app.get("/subscriptions", response_model=List[SubscriptionResponse])
    async def list_subscriptions(service_name: Optional[str] = Query(None, description="Filter by service name")) -> List[SubscriptionResponse]:
        """List all subscriptions, optionally filtered by service name."""
        subscriptions = app.state.subscription_manager.list_subscriptions(service_name)
        
        return [
            SubscriptionResponse(
                id=sub.id,
                service_name=sub.service_name,
                webhook_url=sub.webhook_url,
                filters={
                    "symbols": sub.filters.symbols,
                    "categories": sub.filters.categories,
                    "min_impact_score": sub.filters.min_impact_score,
                    "max_impact_score": sub.filters.max_impact_score,
                    "event_types": [et.value for et in sub.filters.event_types] if sub.filters.event_types else None,
                    "statuses": sub.filters.statuses,
                },
                status=sub.status,
                created_at=sub.created_at,
                updated_at=sub.updated_at,
                last_notification=sub.last_notification,
                failure_count=sub.failure_count,
            )
            for sub in subscriptions
        ]
    
    @app.get("/subscriptions/{subscription_id}", response_model=SubscriptionResponse)
    async def get_subscription(subscription_id: str) -> SubscriptionResponse:
        """Get subscription details by ID."""
        subscription = app.state.subscription_manager.get_subscription(subscription_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        return SubscriptionResponse(
            id=subscription.id,
            service_name=subscription.service_name,
            webhook_url=subscription.webhook_url,
            filters={
                "symbols": subscription.filters.symbols,
                "categories": subscription.filters.categories,
                "min_impact_score": subscription.filters.min_impact_score,
                "max_impact_score": subscription.filters.max_impact_score,
                "event_types": [et.value for et in subscription.filters.event_types] if subscription.filters.event_types else None,
                "statuses": subscription.filters.statuses,
            },
            status=subscription.status,
            created_at=subscription.created_at,
            updated_at=subscription.updated_at,
            last_notification=subscription.last_notification,
            failure_count=subscription.failure_count,
        )
    
    @app.patch("/subscriptions/{subscription_id}")
    async def update_subscription(
        subscription_id: str,
        status_update: Optional[str] = Query(None, description="Update subscription status"),
    ):
        """Update subscription configuration."""
        updates = {}
        if status_update:
            try:
                from .services.subscription_manager import SubscriptionStatus
                updates["status"] = SubscriptionStatus(status_update)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status value")
        
        subscription = app.state.subscription_manager.update_subscription(subscription_id, **updates)
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        return {"message": "Subscription updated successfully", "subscription_id": subscription_id}
    
    @app.delete("/subscriptions/{subscription_id}")
    async def delete_subscription(subscription_id: str):
        """Delete a subscription."""
        success = app.state.subscription_manager.delete_subscription(subscription_id)
        if not success:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        return {"message": "Subscription deleted successfully", "subscription_id": subscription_id}
    
    @app.get("/subscriptions/{subscription_id}/health")
    async def get_subscription_health(subscription_id: str):
        """Get subscription health and stats."""
        subscription = app.state.subscription_manager.get_subscription(subscription_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        return {
            "subscription_id": subscription_id,
            "status": subscription.status.value,
            "failure_count": subscription.failure_count,
            "max_failures": subscription.max_failures,
            "last_notification": subscription.last_notification,
            "webhook_url": subscription.webhook_url,
            "service_name": subscription.service_name,
        }

    # ===== Event Enrichment Endpoints =====
    
    @app.get("/enrichment/market-context/{symbol}")
    async def get_market_context(symbol: str):
        """Get comprehensive market context for a symbol."""
        enrichment_service = getattr(app.state, "enrichment_service", None)
        if not enrichment_service:
            raise HTTPException(status_code=503, detail="Enrichment service not available")
        
        try:
            context = await enrichment_service.get_market_context(symbol.upper())
            return context.to_dict()
        except Exception as e:
            logger.error(f"Failed to get market context for {symbol}: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch market context")
    
    @app.post("/enrichment/enrich-event")
    async def enrich_single_event(event_data: Dict[str, Any]):
        """Enrich a single event with market context."""
        enrichment_service = getattr(app.state, "enrichment_service", None)
        if not enrichment_service:
            raise HTTPException(status_code=503, detail="Enrichment service not available")
        
        try:
            enriched_event = await enrichment_service.enrich_event(event_data)
            return enriched_event
        except Exception as e:
            logger.error(f"Failed to enrich event: {e}")
            raise HTTPException(status_code=500, detail="Failed to enrich event")
    
    @app.post("/enrichment/batch-enrich")
    async def batch_enrich_events(events: List[Dict[str, Any]]):
        """Enrich multiple events in batch."""
        enrichment_service = getattr(app.state, "enrichment_service", None)
        if not enrichment_service:
            raise HTTPException(status_code=503, detail="Enrichment service not available")
        
        try:
            enriched_events = await enrichment_service.batch_enrich_events(events)
            return {"events": enriched_events, "count": len(enriched_events)}
        except Exception as e:
            logger.error(f"Failed to batch enrich events: {e}")
            raise HTTPException(status_code=500, detail="Failed to batch enrich events")
    
    @app.get("/enrichment/stats")
    async def get_enrichment_stats():
        """Get enrichment service statistics."""
        enrichment_service = getattr(app.state, "enrichment_service", None)
        if not enrichment_service:
            raise HTTPException(status_code=503, detail="Enrichment service not available")
        
        return enrichment_service.get_enrichment_stats()

    # ===== Event Lifecycle Tracking Endpoints =====
    
    @app.get("/lifecycle/event/{event_id}")
    async def get_event_lifecycle(event_id: str):
        """Get lifecycle tracking data for a specific event."""
        lifecycle_tracker = getattr(app.state, "lifecycle_tracker", None)
        if not lifecycle_tracker:
            raise HTTPException(status_code=503, detail="Lifecycle tracker not available")
        
        lifecycle_event = lifecycle_tracker.get_lifecycle_event(event_id)
        if not lifecycle_event:
            raise HTTPException(status_code=404, detail="Event lifecycle data not found")
        
        return lifecycle_event.to_dict()
    
    @app.patch("/lifecycle/event/{event_id}/status")
    async def update_event_lifecycle_status(
        event_id: str, 
        new_status: str,
        reason: str = "manual_update"
    ):
        """Manually update event lifecycle status."""
        lifecycle_tracker = getattr(app.state, "lifecycle_tracker", None)
        if not lifecycle_tracker:
            raise HTTPException(status_code=503, detail="Lifecycle tracker not available")
        
        try:
            status_enum = LifecycleEventStatus(new_status)
        except ValueError:
            valid_statuses = [status.value for status in LifecycleEventStatus]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid status. Valid options: {valid_statuses}"
            )
        
        lifecycle_event = await lifecycle_tracker.update_event_status(event_id, status_enum, reason)
        if not lifecycle_event:
            raise HTTPException(status_code=404, detail="Event not found in lifecycle tracking")
        
        # Evaluate for alerts on lifecycle status changes
        alert_system = getattr(app.state, "alert_system", None)
        if alert_system and lifecycle_event:
            await alert_system.evaluate_event(
                {
                    "event_id": event_id,
                    "lifecycle_status": new_status,
                    "reason": reason,
                    "lifecycle_data": lifecycle_event.to_dict() if hasattr(lifecycle_event, 'to_dict') else str(lifecycle_event)
                },
                "event.lifecycle_updated"
            )
        
        return {
            "message": "Event status updated successfully",
            "event_id": event_id,
            "new_status": new_status,
            "reason": reason
        }
    
    @app.get("/lifecycle/events/by-stage/{stage}")
    async def get_events_by_stage(stage: str):
        """Get all events in a specific lifecycle stage."""
        lifecycle_tracker = getattr(app.state, "lifecycle_tracker", None)
        if not lifecycle_tracker:
            raise HTTPException(status_code=503, detail="Lifecycle tracker not available")
        
        try:
            from .services.event_lifecycle import LifecycleStage
            stage_enum = LifecycleStage(stage)
        except ValueError:
            from .services.event_lifecycle import LifecycleStage
            valid_stages = [stage.value for stage in LifecycleStage]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage. Valid options: {valid_stages}"
            )
        
        events = lifecycle_tracker.get_events_by_stage(stage_enum)
        return {
            "stage": stage,
            "count": len(events),
            "events": [event.to_dict() for event in events]
        }
    
    @app.get("/lifecycle/events/by-status/{status}")
    async def get_events_by_status(status: str):
        """Get all events with a specific status."""
        lifecycle_tracker = getattr(app.state, "lifecycle_tracker", None)
        if not lifecycle_tracker:
            raise HTTPException(status_code=503, detail="Lifecycle tracker not available")
        
        try:
            status_enum = LifecycleEventStatus(status)
        except ValueError:
            valid_statuses = [status.value for status in LifecycleEventStatus]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Valid options: {valid_statuses}"
            )
        
        events = lifecycle_tracker.get_events_by_status(status_enum)
        return {
            "status": status,
            "count": len(events),
            "events": [event.to_dict() for event in events]
        }
    
    @app.get("/lifecycle/stats")
    async def get_lifecycle_stats():
        """Get lifecycle tracking statistics."""
        lifecycle_tracker = getattr(app.state, "lifecycle_tracker", None)
        if not lifecycle_tracker:
            raise HTTPException(status_code=503, detail="Lifecycle tracker not available")
        
        return lifecycle_tracker.get_lifecycle_stats()
    
    @app.get("/lifecycle/impact-analysis")
    async def get_impact_analysis(
        min_accuracy: Optional[float] = Query(None, description="Minimum accuracy score"),
        category: Optional[str] = Query(None, description="Filter by event category"),
        limit: int = Query(50, description="Maximum number of results")
    ):
        """Get impact analysis results with optional filtering."""
        lifecycle_tracker = getattr(app.state, "lifecycle_tracker", None)
        if not lifecycle_tracker:
            raise HTTPException(status_code=503, detail="Lifecycle tracker not available")
        
        # Get all analyzed events
        analyzed_events = []
        for lifecycle_event in lifecycle_tracker._lifecycle_cache.values():
            if (lifecycle_event.impact_metrics and 
                lifecycle_event.impact_metrics.accuracy_score is not None):
                
                # Apply filters
                if min_accuracy and lifecycle_event.impact_metrics.accuracy_score < min_accuracy:
                    continue
                if category and lifecycle_event.category.lower() != category.lower():
                    continue
                    
                analyzed_events.append(lifecycle_event)
        
        # Sort by accuracy score (best first)
        analyzed_events.sort(
            key=lambda e: e.impact_metrics.accuracy_score, 
            reverse=True
        )
        
        # Limit results
        limited_events = analyzed_events[:limit]
        
        # Calculate summary statistics
        if analyzed_events:
            accuracy_scores = [e.impact_metrics.accuracy_score for e in analyzed_events]
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            best_accuracy = max(accuracy_scores)
            worst_accuracy = min(accuracy_scores)
        else:
            avg_accuracy = best_accuracy = worst_accuracy = 0
        
        return {
            "summary": {
                "total_analyzed": len(analyzed_events),
                "returned_count": len(limited_events),
                "average_accuracy": avg_accuracy,
                "best_accuracy": best_accuracy,
                "worst_accuracy": worst_accuracy
            },
            "events": [event.to_dict() for event in limited_events]
        }

    # ===== Data Retention Management Endpoints =====
    
    @app.get("/retention/stats", response_model=RetentionStats, tags=["Data Retention"])
    async def get_retention_stats() -> RetentionStats:
        """Get current data retention and storage statistics."""
        return await retention_service.get_retention_stats()
    
    @app.get("/retention/rules", tags=["Data Retention"])
    async def get_retention_rules():
        """Get current retention rules configuration."""
        return await retention_service.get_retention_rules()
    
    @app.post("/retention/cleanup", tags=["Data Retention"])
    async def run_retention_cleanup():
        """Manually trigger retention cleanup process."""
        results = await retention_service.run_retention_cleanup()
        return {
            "message": "Retention cleanup completed",
            "results": [
                {
                    "rule_name": result.rule_name,
                    "category": result.category.value,
                    "records_processed": result.records_archived,
                    "archive_size_bytes": result.archive_size_bytes,
                    "archive_location": result.archive_location,
                    "duration_seconds": result.duration_seconds,
                    "success": result.success,
                    "error_message": result.error_message
                }
                for result in results
            ]
        }
    
    @app.post("/retention/validate-rule", tags=["Data Retention"])
    async def validate_retention_rule(
        rule_name: str = Query(..., description="Name of the retention rule"),
        category: str = Query(..., description="Data category (events, headlines)"),
        policy: str = Query(..., description="Retention policy (active, warm, cold, compliance, delete)"),
        age_days: int = Query(..., description="Age threshold in days"),
        conditions: Optional[str] = Query(None, description="JSON conditions for filtering")
    ):
        """Validate a retention rule and estimate its impact."""
        from .services.data_retention import RetentionRule, RetentionPolicy, DataCategory
        import json
        
        try:
            # Parse conditions if provided
            parsed_conditions = {}
            if conditions:
                parsed_conditions = json.loads(conditions)
            
            # Create rule object
            rule = RetentionRule(
                name=rule_name,
                category=DataCategory(category),
                policy=RetentionPolicy(policy),
                age_days=age_days,
                conditions=parsed_conditions
            )
            
            # Validate rule
            validation_result = await retention_service.validate_retention_policy(rule)
            
            return {
                "message": "Retention rule validation completed",
                "validation": validation_result
            }
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid rule configuration: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Validation failed: {str(e)}"
            )

    # ===== Cache Management Endpoints =====
    
    @app.get("/cache/stats", tags=["Cache Management"])
    async def get_cache_stats():
        """Get current cache statistics and performance metrics."""
        return await cache_service.get_stats()
    
    @app.post("/cache/invalidate/event/{event_id}", tags=["Cache Management"])
    async def invalidate_event_cache(event_id: str, symbol: Optional[str] = Query(None)):
        """Invalidate cache entries for a specific event."""
        await cache_service.invalidate_event(event_id, symbol)
        return {"message": f"Cache invalidated for event {event_id}"}
    
    @app.post("/cache/invalidate/symbol/{symbol}", tags=["Cache Management"])
    async def invalidate_symbol_cache(symbol: str):
        """Invalidate cache entries for a specific symbol."""
        await cache_service.invalidate_symbol(symbol)
        return {"message": f"Cache invalidated for symbol {symbol}"}
    
    @app.post("/cache/invalidate/pattern", tags=["Cache Management"])
    async def invalidate_pattern_cache(pattern: str = Query(..., description="Cache key pattern to invalidate")):
        """Invalidate cache entries matching a pattern."""
        deleted_count = await cache_service.delete_pattern(pattern)
        return {
            "message": f"Cache invalidated for pattern: {pattern}",
            "deleted_keys": deleted_count
        }
    
    @app.delete("/cache/clear", tags=["Cache Management"])
    async def clear_all_cache():
        """Clear all cache data (use with caution)."""
        success = await cache_service.clear_all()
        if success:
            return {"message": "All cache data cleared"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear cache"
            )
    
    @app.get("/cache/health", tags=["Cache Management"])
    async def cache_health():
        """Get cache service health status."""
        if not cache_service.enabled:
            return {"status": "disabled", "message": "Cache service is disabled"}
        
        try:
            stats = await cache_service.get_stats()
            return {
                "status": "healthy" if stats.get("connection_status") == "connected" else "unhealthy",
                "enabled": stats.get("enabled", False),
                "connection_status": stats.get("connection_status"),
                "total_keys": stats.get("total_keys", 0),
                "hit_rate": stats.get("daily_stats", {}).get("hit_rate", 0)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # ===== Bulk Ingestion Endpoints =====
    
    @app.post("/bulk/ingest", tags=["Bulk Ingestion"])
    async def ingest_bulk_data(
        file_path: str = Query(..., description="Path to the file to ingest"),
        format_type: str = Query(..., description="File format: csv, json, jsonl"),
        batch_size: int = Query(1000, description="Batch size for processing"),
        mode: str = Query("upsert", description="Ingestion mode: insert_only, upsert, replace, append"),
        validation_level: str = Query("permissive", description="Validation level: strict, permissive, none"),
        auto_categorize: bool = Query(True, description="Enable automatic categorization"),
        auto_enrich: bool = Query(False, description="Enable automatic enrichment (expensive)"),
        skip_cache_invalidation: bool = Query(False, description="Skip cache invalidation for performance")
    ):
        """Ingest bulk event data from a file."""
        
        try:
            # Validate parameters
            try:
                format_enum = IngestionFormat(format_type.lower())
                mode_enum = IngestionMode(mode.lower())
                validation_enum = ValidationLevel(validation_level.lower())
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid parameter: {e}"
                )
            
            # Create ingestion configuration
            config = IngestionConfig(
                batch_size=batch_size,
                mode=mode_enum,
                validation_level=validation_enum,
                auto_categorize=auto_categorize,
                auto_enrich=auto_enrich,
                skip_cache_invalidation=skip_cache_invalidation
            )
            
            # Start ingestion
            result = await bulk_ingestion_service.ingest_file(
                file_path=file_path,
                format_type=format_enum,
                config=config
            )
            
            return {
                "message": "Bulk ingestion completed",
                "operation_id": result.operation_id,
                "status": result.status,
                "statistics": {
                    "total_records": result.stats.total_records,
                    "processed_records": result.stats.processed_records,
                    "inserted_records": result.stats.inserted_records,
                    "updated_records": result.stats.updated_records,
                    "skipped_records": result.stats.skipped_records,
                    "failed_records": result.stats.failed_records,
                    "duplicate_records": result.stats.duplicate_records,
                    "validation_errors": result.stats.validation_errors,
                    "processing_time_seconds": result.stats.processing_time_seconds,
                    "throughput_records_per_second": result.stats.throughput_records_per_second,
                    "batch_count": result.stats.batch_count
                },
                "file_info": result.file_info,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings)
            }
            
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {file_path}"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Bulk ingestion failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ingestion failed: {str(e)}"
            )
    
    @app.get("/bulk/operations", tags=["Bulk Ingestion"])
    async def list_bulk_operations():
        """List active bulk ingestion operations."""
        operations = await bulk_ingestion_service.list_active_operations()
        return {
            "active_operations": operations,
            "count": len(operations)
        }
    
    @app.get("/bulk/operations/{operation_id}", tags=["Bulk Ingestion"])
    async def get_bulk_operation_status(operation_id: str):
        """Get status of a specific bulk ingestion operation."""
        operation = await bulk_ingestion_service.get_operation_status(operation_id)
        
        if not operation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Operation not found: {operation_id}"
            )
        
        return operation
    
    @app.get("/bulk/stats", tags=["Bulk Ingestion"])
    async def get_bulk_ingestion_stats():
        """Get bulk ingestion service statistics and configuration."""
        return await bulk_ingestion_service.get_ingestion_stats()
    
    @app.post("/bulk/validate", tags=["Bulk Ingestion"])
    async def validate_bulk_file(
        file_path: str = Query(..., description="Path to the file to validate"),
        format_type: str = Query(..., description="File format: csv, json, jsonl"),
        sample_size: int = Query(100, description="Number of records to validate")
    ):
        """Validate a bulk data file without ingesting it."""
        
        try:
            # Import validation logic
            from .services.bulk_ingestion import BulkIngestionService
            
            # Validate parameters
            try:
                format_enum = IngestionFormat(format_type.lower())
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid format: {e}"
                )
            
            # Validate file exists
            from pathlib import Path
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"File not found: {file_path}"
                )
            
            # Get file info
            file_size_mb = file_path_obj.stat().st_size / 1024 / 1024
            
            # Create temporary service instance for validation
            temp_service = BulkIngestionService(SessionFactory)
            
            # Read sample records
            if format_enum == IngestionFormat.CSV:
                record_generator = temp_service._read_csv_file(file_path)
            elif format_enum == IngestionFormat.JSON:
                record_generator = temp_service._read_json_file(file_path)
            elif format_enum == IngestionFormat.JSONL:
                record_generator = temp_service._read_jsonl_file(file_path)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported format for validation: {format_type}"
                )
            
            # Validate sample records
            valid_records = 0
            invalid_records = 0
            errors = []
            sample_records = []
            
            count = 0
            async for record in record_generator:
                if count >= sample_size:
                    break
                
                count += 1
                sample_records.append(record)
                
                # Validate record
                is_valid, validation_error = await temp_service._validate_record(
                    record, ValidationLevel.STRICT
                )
                
                if is_valid:
                    valid_records += 1
                else:
                    invalid_records += 1
                    errors.append({
                        "record_number": count,
                        "error": validation_error,
                        "record_preview": {k: str(v)[:100] for k, v in record.items()}
                    })
            
            return {
                "file_info": {
                    "path": file_path,
                    "size_mb": round(file_size_mb, 2),
                    "format": format_type
                },
                "validation_summary": {
                    "sample_size": count,
                    "valid_records": valid_records,
                    "invalid_records": invalid_records,
                    "validation_rate": round(valid_records / count, 3) if count > 0 else 0,
                    "estimated_total_errors": round(invalid_records * (file_size_mb / 10)) if invalid_records > 0 else 0
                },
                "errors": errors[:10],  # Show first 10 errors
                "recommendations": [
                    "Use 'permissive' validation level to skip invalid records" if invalid_records > 0 else "File appears valid for ingestion",
                    f"Consider batch size of {min(1000, max(100, count // 10))}" if count > 100 else "Small file - use default batch size",
                    "Enable auto_categorization for better data quality" if valid_records > 0 else None
                ]
            }
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Validation failed: {str(e)}"
            )

    # ===== Real-Time Streaming Endpoints =====
    
    @app.websocket("/stream/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time event streaming."""
        connection_id = str(uuid.uuid4())
        
        client_info = {
            "client_ip": websocket.client.host if websocket.client else "unknown",
            "user_agent": websocket.headers.get("user-agent", "unknown")
        }
        
        # Establish connection
        if not await websocket_manager.connect(websocket, connection_id, client_info):
            return
        
        try:
            while True:
                # Receive messages from client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    await websocket_manager.handle_client_message(connection_id, message)
                except json.JSONDecodeError:
                    await websocket_manager.send_to_connection(connection_id, {
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                except Exception as e:
                    logger.error(f"WebSocket message handling error: {e}")
                    await websocket_manager.send_to_connection(connection_id, {
                        "type": "error",
                        "message": f"Message processing failed: {str(e)}"
                    })
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            websocket_manager.disconnect(connection_id)
    
    @app.get("/stream/sse")
    async def sse_endpoint(
        request: Request,
        topics: Optional[str] = Query(None, description="Comma-separated list of event types to subscribe to"),
        symbols: Optional[str] = Query(None, description="Comma-separated list of symbols to filter"),
        min_priority: Optional[int] = Query(None, description="Minimum priority level (1=high, 2=normal, 3=low)"),
        sources: Optional[str] = Query(None, description="Comma-separated list of sources to filter")
    ):
        """Server-Sent Events endpoint for real-time event streaming."""
        
        connection_id = str(uuid.uuid4())
        
        # Parse topics
        topic_list = None
        if topics:
            topic_list = [t.strip() for t in topics.split(",")]
        
        # Parse filters
        filters = {}
        if symbols:
            filters["symbols"] = [s.strip().upper() for s in symbols.split(",")]
        if min_priority is not None:
            filters["min_priority"] = min_priority
        if sources:
            filters["sources"] = [s.strip() for s in sources.split(",")]
        
        try:
            return await sse_manager.create_sse_response(
                request, 
                connection_id, 
                topic_list, 
                filters if filters else None
            )
        except Exception as e:
            logger.error(f"Failed to create SSE connection: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to establish SSE connection: {str(e)}"
            )
    
    @app.get("/stream/stats", tags=["Real-Time Streaming"])
    async def get_streaming_stats():
        """Get real-time streaming statistics and connection information."""
        
        streaming_stats = await streaming_service.get_stream_stats()
        websocket_stats = websocket_manager.get_connection_stats()
        sse_stats = sse_manager.get_connection_stats()
        
        return {
            "streaming_service": streaming_stats,
            "websocket_connections": websocket_stats,
            "sse_connections": sse_stats,
            "total_real_time_connections": (
                websocket_stats["total_connections"] + 
                sse_stats["total_connections"]
            )
        }
    
    @app.post("/stream/test", tags=["Real-Time Streaming"])
    async def test_streaming(
        event_type: str = Query(..., description="Event type to test"),
        symbol: str = Query("TEST", description="Symbol for test event"),
        message: str = Query("Test streaming message", description="Test message content")
    ):
        """Send a test event through the streaming system."""
        
        try:
            # Validate event type
            try:
                stream_event_type = StreamEventType(event_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid event type. Valid types: {[e.value for e in StreamEventType]}"
                )
            
            # Create test event data
            test_data = {
                "symbol": symbol.upper(),
                "title": f"Test Event - {message}",
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "test": True
            }
            
            # Publish test event
            event_id = await streaming_service.publish_event(
                stream_event_type,
                test_data,
                source="streaming_test_api",
                metadata={
                    "test_event": True,
                    "api_endpoint": "test_streaming"
                }
            )
            
            return {
                "message": "Test event published successfully",
                "event_id": event_id,
                "event_type": event_type,
                "data": test_data
            }
            
        except Exception as e:
            logger.error(f"Failed to send test streaming event: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to publish test event: {str(e)}"
            )
    
    @app.get("/stream/health", tags=["Real-Time Streaming"])
    async def stream_health():
        """Get streaming service health status."""
        
        if not streaming_service.enabled:
            return {"status": "disabled", "message": "Streaming service is disabled"}
        
        try:
            stats = await streaming_service.get_stream_stats()
            
            # Determine health status
            redis_connected = stats.get("redis", {}).get("connected", False)
            has_active_backends = len(stats.get("backends", [])) > 0
            
            if redis_connected and has_active_backends:
                status_value = "healthy"
            elif has_active_backends:
                status_value = "degraded"  # Some backends working
            else:
                status_value = "unhealthy"
            
            return {
                "status": status_value,
                "enabled": stats.get("enabled", False),
                "backends": stats.get("backends", []),
                "active_connections": stats.get("connections", {}),
                "throughput": stats.get("metrics", {}).get("throughput_per_second", 0),
                "message_stats": {
                    "sent": stats.get("metrics", {}).get("messages_sent", 0),
                    "received": stats.get("metrics", {}).get("messages_received", 0),
                    "failed": stats.get("metrics", {}).get("messages_failed", 0)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # ===== Analytics and Dashboard Endpoints =====
    
    @app.get("/dashboard", response_class=HTMLResponse, tags=["Analytics Dashboard"])
    async def analytics_dashboard(request: Request):
        """Serve the analytics dashboard HTML page."""
        return templates.TemplateResponse("dashboard.html", {"request": request})
    
    @app.get("/analytics/dashboard", tags=["Analytics"])
    async def get_dashboard_data():
        """Get comprehensive dashboard data for analytics visualization."""
        try:
            dashboard_data = await analytics_service.get_dashboard_data()
            return dashboard_data
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get dashboard data: {str(e)}"
            )
    
    @app.get("/analytics/metrics", tags=["Analytics"])
    async def get_event_metrics(
        start_date: Optional[datetime] = Query(None, description="Start date for metrics (ISO format)"),
        end_date: Optional[datetime] = Query(None, description="End date for metrics (ISO format)"),
        symbols: Optional[str] = Query(None, description="Comma-separated list of symbols to filter"),
        categories: Optional[str] = Query(None, description="Comma-separated list of categories to filter"),
        use_cache: bool = Query(True, description="Whether to use cached results")
    ):
        """Get comprehensive event metrics and statistics."""
        try:
            symbol_list = symbols.split(",") if symbols else None
            category_list = categories.split(",") if categories else None
            
            metrics = await analytics_service.get_event_metrics(
                start_date=start_date,
                end_date=end_date,
                symbols=symbol_list,
                categories=category_list,
                use_cache=use_cache
            )
            
            return {
                "metrics": {
                    "total_events": metrics.total_events,
                    "events_by_category": metrics.events_by_category,
                    "events_by_status": metrics.events_by_status,
                    "events_by_source": metrics.events_by_source,
                    "events_by_symbol": metrics.events_by_symbol,
                    "average_impact_score": metrics.average_impact_score,
                    "high_impact_events": metrics.high_impact_events,
                    "events_with_headlines": metrics.events_with_headlines,
                    "total_headlines": metrics.total_headlines
                },
                "parameters": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "symbols": symbol_list,
                    "categories": category_list
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get event metrics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get event metrics: {str(e)}"
            )
    
    @app.get("/analytics/timeseries", tags=["Analytics"])
    async def get_time_series_data(
        metric: str = Query(..., description="Metric to analyze (event_count, impact_score)"),
        start_date: datetime = Query(..., description="Start date (ISO format)"),
        end_date: datetime = Query(..., description="End date (ISO format)"),
        interval: str = Query("1h", description="Time interval (5m, 15m, 1h, 1d, 1w)"),
        symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
        categories: Optional[str] = Query(None, description="Comma-separated list of categories")
    ):
        """Get time series data for various metrics."""
        try:
            symbol_list = symbols.split(",") if symbols else None
            category_list = categories.split(",") if categories else None
            
            data_points = await analytics_service.get_time_series_data(
                metric=metric,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                symbols=symbol_list,
                categories=category_list
            )
            
            return {
                "metric": metric,
                "interval": interval,
                "data_points": [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "value": point.value,
                        "category": point.category,
                        "metadata": point.metadata
                    }
                    for point in data_points
                ],
                "parameters": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "symbols": symbol_list,
                    "categories": category_list
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get time series data: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get time series data: {str(e)}"
            )
    
    @app.get("/analytics/trends", tags=["Analytics"])
    async def get_trend_analysis(
        metric: str = Query(..., description="Metric to analyze (event_count, impact_score)"),
        period_days: int = Query(30, description="Analysis period in days"),
        symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
        categories: Optional[str] = Query(None, description="Comma-separated list of categories")
    ):
        """Get trend analysis for a specific metric."""
        try:
            symbol_list = symbols.split(",") if symbols else None
            category_list = categories.split(",") if categories else None
            
            trend_analysis = await analytics_service.analyze_trends(
                metric=metric,
                period_days=period_days,
                symbols=symbol_list,
                categories=category_list
            )
            
            return {
                "metric": metric,
                "analysis": {
                    "period": trend_analysis.period,
                    "growth_rate": trend_analysis.growth_rate,
                    "trend_direction": trend_analysis.trend_direction,
                    "volatility": trend_analysis.volatility,
                    "peak_timestamp": trend_analysis.peak_timestamp.isoformat(),
                    "peak_value": trend_analysis.peak_value
                },
                "data_points": [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "value": point.value,
                        "category": point.category
                    }
                    for point in trend_analysis.data_points
                ],
                "parameters": {
                    "period_days": period_days,
                    "symbols": symbol_list,
                    "categories": category_list
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get trend analysis: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get trend analysis: {str(e)}"
            )
    
    @app.get("/analytics/performance", tags=["Analytics"])
    async def get_performance_report(
        period_days: int = Query(7, description="Report period in days"),
        limit: int = Query(10, description="Limit for top results")
    ):
        """Get comprehensive performance report."""
        try:
            performance_report = await analytics_service.generate_performance_report(
                period_days=period_days,
                limit=limit
            )
            
            return {
                "report": {
                    "period": performance_report.report_period,
                    "most_active_symbols": performance_report.most_active_symbols,
                    "trending_categories": [
                        {
                            "category": category,
                            "event_count": count,
                            "growth_rate": growth_rate
                        }
                        for category, count, growth_rate in performance_report.trending_categories
                    ],
                    "impact_distribution": performance_report.impact_distribution,
                    "source_reliability": performance_report.source_reliability,
                    "headline_coverage": performance_report.headline_coverage
                },
                "parameters": {
                    "period_days": period_days,
                    "limit": limit
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance report: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get performance report: {str(e)}"
            )
    
    @app.post("/analytics/cache/clear", tags=["Analytics"])
    async def clear_analytics_cache():
        """Clear analytics cache to force fresh data retrieval."""
        try:
            analytics_service.clear_cache()
            return {
                "status": "success", 
                "message": "Analytics cache cleared successfully",
                "cleared_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to clear analytics cache: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear analytics cache: {str(e)}"
            )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
