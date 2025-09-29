"""Historical Data Backfill Service

Provides capabilities to automatically backfill historical event data
for new symbols added to the trading platform.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

import httpx
from sqlalchemy import select, update, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from ..models import EventORM, EventHeadlineORM
from .event_categorizer import EventCategorizer
from .event_impact import EventImpactScorer
from .event_enrichment import EventEnrichmentService
from .feed_health import FeedHealthMonitor

logger = logging.getLogger(__name__)


class BackfillStatus(str, Enum):
    """Status of backfill operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class BackfillSource(str, Enum):
    """Available data sources for backfill."""
    FINANCIAL_MODELING_PREP = "financial_modeling_prep"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    YAHOO_FINANCE = "yahoo_finance"
    QUANDL = "quandl"
    SEC_EDGAR = "sec_edgar"


@dataclass
class BackfillRequest:
    """Request for historical data backfill."""
    symbol: str
    start_date: datetime
    end_date: datetime
    categories: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    max_events: Optional[int] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BackfillResult:
    """Result of backfill operation."""
    symbol: str
    status: BackfillStatus
    events_created: int
    events_updated: int
    events_skipped: int
    start_date: datetime
    end_date: datetime
    duration_seconds: float
    sources_used: List[str]
    categories_found: List[str]
    errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class BackfillProgress:
    """Progress tracking for backfill operations."""
    symbol: str
    total_requests: int
    completed_requests: int
    current_source: str
    current_date_range: str
    events_processed: int
    started_at: datetime
    estimated_completion: Optional[datetime]


class HistoricalBackfillService:
    """Service for backfilling historical event data for new symbols."""
    
    def __init__(
        self,
        session_factory,
        categorizer: Optional[EventCategorizer] = None,
        impact_scorer: Optional[EventImpactScorer] = None,
        enrichment_service: Optional[EventEnrichmentService] = None,
        health_monitor: Optional[FeedHealthMonitor] = None
    ):
        self.session_factory = session_factory
        self.categorizer = categorizer
        self.impact_scorer = impact_scorer
        self.enrichment_service = enrichment_service
        self.health_monitor = health_monitor
        
        # Configuration
        self.enabled = os.getenv("BACKFILL_ENABLED", "true").lower() == "true"
        self.max_concurrent_symbols = int(os.getenv("BACKFILL_MAX_CONCURRENT_SYMBOLS", "3"))
        self.max_days_per_request = int(os.getenv("BACKFILL_MAX_DAYS_PER_REQUEST", "90"))
        self.default_lookback_days = int(os.getenv("BACKFILL_DEFAULT_LOOKBACK_DAYS", "365"))
        self.rate_limit_delay = float(os.getenv("BACKFILL_RATE_LIMIT_DELAY", "1.0"))
        self.timeout = float(os.getenv("BACKFILL_TIMEOUT", "30.0"))
        self.retry_attempts = int(os.getenv("BACKFILL_RETRY_ATTEMPTS", "3"))
        self.batch_size = int(os.getenv("BACKFILL_BATCH_SIZE", "100"))
        
        # Data source configuration
        self.sources = self._configure_sources()
        
        # Active backfill tracking
        self._active_backfills: Dict[str, BackfillProgress] = {}
        self._backfill_queue: asyncio.Queue = asyncio.Queue()
        self._worker_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"HistoricalBackfillService initialized (enabled={self.enabled})")
    
    def _configure_sources(self) -> Dict[str, Dict[str, Any]]:
        """Configure available data sources for backfill."""
        sources = {}
        
        # Financial Modeling Prep
        if os.getenv("FMP_API_KEY"):
            sources[BackfillSource.FINANCIAL_MODELING_PREP] = {
                "api_key": os.getenv("FMP_API_KEY"),
                "base_url": "https://financialmodelingprep.com/api/v3",
                "endpoints": {
                    "earnings": "/earning_calendar",
                    "splits": "/stock_split_calendar",
                    "dividends": "/stock_dividend_calendar",
                    "ipo": "/ipo_calendar"
                },
                "rate_limit": 250,  # requests per minute
                "priority": 1
            }
        
        # Alpha Vantage
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            sources[BackfillSource.ALPHA_VANTAGE] = {
                "api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
                "base_url": "https://www.alphavantage.co/query",
                "endpoints": {
                    "earnings": "function=EARNINGS_CALENDAR",
                    "news": "function=NEWS_SENTIMENT"
                },
                "rate_limit": 5,  # requests per minute
                "priority": 2
            }
        
        # Polygon.io
        if os.getenv("POLYGON_API_KEY"):
            sources[BackfillSource.POLYGON] = {
                "api_key": os.getenv("POLYGON_API_KEY"),
                "base_url": "https://api.polygon.io",
                "endpoints": {
                    "splits": "/v3/reference/splits",
                    "dividends": "/v3/reference/dividends",
                    "news": "/v2/reference/news"
                },
                "rate_limit": 5,  # requests per minute
                "priority": 3
            }
        
        # Finnhub
        if os.getenv("FINNHUB_API_KEY"):
            sources[BackfillSource.FINNHUB] = {
                "api_key": os.getenv("FINNHUB_API_KEY"),
                "base_url": "https://finnhub.io/api/v1",
                "endpoints": {
                    "earnings": "/calendar/earnings",
                    "ipo": "/calendar/ipo",
                    "economic": "/calendar/economic"
                },
                "rate_limit": 60,  # requests per minute
                "priority": 4
            }
        
        return sources
    
    async def start(self):
        """Start the backfill service."""
        if not self.enabled:
            logger.info("Historical backfill service disabled")
            return
        
        logger.info("Starting historical backfill service")
        
        # Start worker tasks
        for i in range(self.max_concurrent_symbols):
            task = asyncio.create_task(self._backfill_worker(f"worker-{i}"))
            self._worker_tasks.append(task)
    
    async def stop(self):
        """Stop the backfill service."""
        logger.info("Stopping historical backfill service")
        
        self._shutdown_event.set()
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        self._worker_tasks.clear()
    
    async def request_backfill(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        categories: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        priority: int = 1
    ) -> str:
        """Request historical data backfill for a symbol."""
        if not self.enabled:
            raise ValueError("Backfill service is disabled")
        
        # Set default dates
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=self.default_lookback_days)
        
        # Check if symbol already has recent data
        async with self.session_factory() as session:
            existing_events = await session.execute(
                select(EventORM)
                .where(
                    and_(
                        EventORM.symbol == symbol.upper(),
                        EventORM.scheduled_at >= start_date,
                        EventORM.scheduled_at <= end_date
                    )
                )
                .limit(10)
            )
            
            if existing_events.scalars().first():
                logger.info(f"Symbol {symbol} already has recent data, skipping backfill")
                return f"skipped-{symbol}-{datetime.utcnow().isoformat()}"
        
        # Create backfill request
        request = BackfillRequest(
            symbol=symbol.upper(),
            start_date=start_date,
            end_date=end_date,
            categories=categories,
            sources=sources or list(self.sources.keys()),
            priority=priority,
            metadata={
                "requested_at": datetime.utcnow().isoformat(),
                "auto_requested": True
            }
        )
        
        # Add to queue
        await self._backfill_queue.put(request)
        
        request_id = f"backfill-{symbol}-{datetime.utcnow().isoformat()}"
        logger.info(f"Queued backfill request for {symbol}: {request_id}")
        
        return request_id
    
    async def get_backfill_status(self, symbol: str) -> Optional[BackfillProgress]:
        """Get current backfill status for a symbol."""
        return self._active_backfills.get(symbol.upper())
    
    async def list_active_backfills(self) -> List[BackfillProgress]:
        """List all active backfill operations."""
        return list(self._active_backfills.values())
    
    async def _backfill_worker(self, worker_id: str):
        """Background worker for processing backfill requests."""
        logger.info(f"Backfill worker {worker_id} started")
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get next request with timeout
                    request = await asyncio.wait_for(
                        self._backfill_queue.get(),
                        timeout=5.0
                    )
                    
                    logger.info(f"Worker {worker_id} processing backfill for {request.symbol}")
                    
                    # Process the backfill request
                    result = await self._process_backfill_request(request)
                    
                    logger.info(
                        f"Backfill completed for {request.symbol}: "
                        f"{result.events_created} created, {result.events_updated} updated, "
                        f"{result.events_skipped} skipped"
                    )
                    
                except asyncio.TimeoutError:
                    # No requests in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    await asyncio.sleep(5)  # Brief pause before retrying
                
        except asyncio.CancelledError:
            logger.info(f"Backfill worker {worker_id} cancelled")
        except Exception as e:
            logger.error(f"Backfill worker {worker_id} fatal error: {e}")
    
    async def _process_backfill_request(self, request: BackfillRequest) -> BackfillResult:
        """Process a single backfill request."""
        symbol = request.symbol
        start_time = datetime.utcnow()
        
        # Initialize progress tracking
        progress = BackfillProgress(
            symbol=symbol,
            total_requests=len(request.sources or []),
            completed_requests=0,
            current_source="",
            current_date_range=f"{request.start_date.date()} to {request.end_date.date()}",
            events_processed=0,
            started_at=start_time,
            estimated_completion=None
        )
        self._active_backfills[symbol] = progress
        
        result = BackfillResult(
            symbol=symbol,
            status=BackfillStatus.IN_PROGRESS,
            events_created=0,
            events_updated=0,
            events_skipped=0,
            start_date=request.start_date,
            end_date=request.end_date,
            duration_seconds=0.0,
            sources_used=[],
            categories_found=[],
            errors=[],
            metadata=request.metadata or {}
        )
        
        try:
            # Process each data source
            for source in request.sources or list(self.sources.keys()):
                if source not in self.sources:
                    result.errors.append(f"Source {source} not configured")
                    continue
                
                progress.current_source = source
                
                try:
                    source_result = await self._backfill_from_source(
                        request, source, self.sources[source]
                    )
                    
                    # Aggregate results
                    result.events_created += source_result["events_created"]
                    result.events_updated += source_result["events_updated"]
                    result.events_skipped += source_result["events_skipped"]
                    result.sources_used.append(source)
                    
                    # Track categories found
                    for category in source_result.get("categories", []):
                        if category not in result.categories_found:
                            result.categories_found.append(category)
                    
                    progress.events_processed += source_result["events_created"]
                    progress.completed_requests += 1
                    
                except Exception as e:
                    error_msg = f"Source {source} failed: {str(e)}"
                    result.errors.append(error_msg)
                    logger.error(f"Backfill error for {symbol} from {source}: {e}")
                
                # Rate limiting between sources
                await asyncio.sleep(self.rate_limit_delay)
            
            # Determine final status
            if result.events_created > 0 or result.events_updated > 0:
                result.status = BackfillStatus.COMPLETED if not result.errors else BackfillStatus.PARTIAL
            elif result.errors:
                result.status = BackfillStatus.FAILED
            else:
                result.status = BackfillStatus.COMPLETED
            
        except Exception as e:
            result.status = BackfillStatus.FAILED
            result.errors.append(f"Backfill failed: {str(e)}")
            logger.error(f"Backfill failed for {symbol}: {e}")
        
        finally:
            # Clean up progress tracking
            if symbol in self._active_backfills:
                del self._active_backfills[symbol]
            
            # Calculate duration
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()
        
        return result
    
    async def _backfill_from_source(
        self, 
        request: BackfillRequest, 
        source: str, 
        source_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Backfill data from a specific source."""
        events_created = 0
        events_updated = 0
        events_skipped = 0
        categories_found = set()
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Split date range into chunks if necessary
            current_date = request.start_date
            
            while current_date < request.end_date:
                chunk_end = min(
                    current_date + timedelta(days=self.max_days_per_request),
                    request.end_date
                )
                
                # Fetch data for this date chunk
                events_data = await self._fetch_events_from_source(
                    client, request.symbol, current_date, chunk_end, 
                    source, source_config, request.categories
                )
                
                # Process and store events
                async with self.session_factory() as session:
                    for event_data in events_data:
                        try:
                            result = await self._process_event_data(
                                session, event_data, request.symbol
                            )
                            
                            if result["action"] == "created":
                                events_created += 1
                            elif result["action"] == "updated":
                                events_updated += 1
                            else:
                                events_skipped += 1
                            
                            if result.get("category"):
                                categories_found.add(result["category"])
                        
                        except Exception as e:
                            logger.warning(f"Failed to process event data: {e}")
                            events_skipped += 1
                
                current_date = chunk_end
                
                # Rate limiting between chunks
                await asyncio.sleep(self.rate_limit_delay)
        
        return {
            "events_created": events_created,
            "events_updated": events_updated,
            "events_skipped": events_skipped,
            "categories": list(categories_found)
        }
    
    async def _fetch_events_from_source(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        source: str,
        source_config: Dict[str, Any],
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch events from a specific data source."""
        events = []
        
        try:
            if source == BackfillSource.FINANCIAL_MODELING_PREP:
                events = await self._fetch_from_fmp(
                    client, symbol, start_date, end_date, source_config, categories
                )
            elif source == BackfillSource.ALPHA_VANTAGE:
                events = await self._fetch_from_alpha_vantage(
                    client, symbol, start_date, end_date, source_config, categories
                )
            elif source == BackfillSource.POLYGON:
                events = await self._fetch_from_polygon(
                    client, symbol, start_date, end_date, source_config, categories
                )
            elif source == BackfillSource.FINNHUB:
                events = await self._fetch_from_finnhub(
                    client, symbol, start_date, end_date, source_config, categories
                )
            else:
                logger.warning(f"Unsupported source: {source}")
        
        except Exception as e:
            logger.error(f"Failed to fetch from {source}: {e}")
        
        return events
    
    async def _fetch_from_fmp(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any],
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch events from Financial Modeling Prep."""
        events = []
        base_url = config["base_url"]
        api_key = config["api_key"]
        
        # Fetch earnings calendar
        if not categories or "earnings" in categories:
            url = f"{base_url}/earning_calendar"
            params = {
                "apikey": api_key,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d")
            }
            
            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    if item.get("symbol") == symbol:
                        events.append({
                            "symbol": symbol,
                            "title": f"{item.get('symbol')} Earnings Call",
                            "category": "earnings",
                            "scheduled_at": datetime.fromisoformat(item["date"]),
                            "description": f"Q{item.get('quarter', '')} {item.get('year', '')} earnings call",
                            "source": "financial_modeling_prep",
                            "external_id": f"fmp-earnings-{symbol}-{item['date']}",
                            "metadata": {
                                "eps_estimate": item.get("epsEstimate"),
                                "eps_actual": item.get("eps"),
                                "revenue_estimate": item.get("revenueEstimate"),
                                "revenue_actual": item.get("revenue"),
                                "quarter": item.get("quarter"),
                                "year": item.get("year")
                            }
                        })
        
        # Fetch stock splits
        if not categories or "split" in categories:
            url = f"{base_url}/stock_split_calendar"
            params = {
                "apikey": api_key,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d")
            }
            
            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data:
                    if item.get("symbol") == symbol:
                        events.append({
                            "symbol": symbol,
                            "title": f"{symbol} Stock Split",
                            "category": "split",
                            "scheduled_at": datetime.fromisoformat(item["date"]),
                            "description": f"Stock split ratio {item.get('numerator', '')}:{item.get('denominator', '')}",
                            "source": "financial_modeling_prep",
                            "external_id": f"fmp-split-{symbol}-{item['date']}",
                            "metadata": {
                                "numerator": item.get("numerator"),
                                "denominator": item.get("denominator")
                            }
                        })
        
        return events
    
    async def _fetch_from_alpha_vantage(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any],
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch events from Alpha Vantage."""
        events = []
        base_url = config["base_url"]
        api_key = config["api_key"]
        
        # Fetch earnings calendar
        if not categories or "earnings" in categories:
            params = {
                "function": "EARNINGS_CALENDAR",
                "symbol": symbol,
                "apikey": api_key
            }
            
            response = await client.get(base_url, params=params)
            if response.status_code == 200:
                # Alpha Vantage returns CSV for earnings calendar
                text = response.text
                lines = text.strip().split('\n')
                
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            event_date = datetime.strptime(parts[2], "%Y-%m-%d")
                            if start_date <= event_date <= end_date:
                                events.append({
                                    "symbol": symbol,
                                    "title": f"{symbol} Earnings Release",
                                    "category": "earnings",
                                    "scheduled_at": event_date,
                                    "description": f"Earnings release for {symbol}",
                                    "source": "alpha_vantage",
                                    "external_id": f"av-earnings-{symbol}-{parts[2]}",
                                    "metadata": {
                                        "fiscal_date": parts[1] if len(parts) > 1 else None,
                                        "estimate": parts[3] if len(parts) > 3 else None
                                    }
                                })
        
        return events
    
    async def _fetch_from_polygon(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any],
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch events from Polygon.io."""
        events = []
        base_url = config["base_url"]
        api_key = config["api_key"]
        
        # Fetch stock splits
        if not categories or "split" in categories:
            url = f"{base_url}/v3/reference/splits"
            params = {
                "ticker": symbol,
                "execution_date.gte": start_date.strftime("%Y-%m-%d"),
                "execution_date.lte": end_date.strftime("%Y-%m-%d"),
                "apikey": api_key
            }
            
            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    events.append({
                        "symbol": symbol,
                        "title": f"{symbol} Stock Split",
                        "category": "split",
                        "scheduled_at": datetime.fromisoformat(item["execution_date"]),
                        "description": f"Stock split ratio {item.get('split_from', '')}:{item.get('split_to', '')}",
                        "source": "polygon",
                        "external_id": f"polygon-split-{symbol}-{item['execution_date']}",
                        "metadata": {
                            "split_from": item.get("split_from"),
                            "split_to": item.get("split_to")
                        }
                    })
        
        # Fetch dividends
        if not categories or "dividend" in categories:
            url = f"{base_url}/v3/reference/dividends"
            params = {
                "ticker": symbol,
                "ex_dividend_date.gte": start_date.strftime("%Y-%m-%d"),
                "ex_dividend_date.lte": end_date.strftime("%Y-%m-%d"),
                "apikey": api_key
            }
            
            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("results", []):
                    events.append({
                        "symbol": symbol,
                        "title": f"{symbol} Dividend Payment",
                        "category": "dividend",
                        "scheduled_at": datetime.fromisoformat(item["ex_dividend_date"]),
                        "description": f"Dividend payment of ${item.get('cash_amount', 0):.4f} per share",
                        "source": "polygon",
                        "external_id": f"polygon-dividend-{symbol}-{item['ex_dividend_date']}",
                        "metadata": {
                            "cash_amount": item.get("cash_amount"),
                            "currency": item.get("currency"),
                            "dividend_type": item.get("dividend_type"),
                            "frequency": item.get("frequency")
                        }
                    })
        
        return events
    
    async def _fetch_from_finnhub(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        config: Dict[str, Any],
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch events from Finnhub."""
        events = []
        base_url = config["base_url"]
        api_key = config["api_key"]
        
        # Fetch earnings calendar
        if not categories or "earnings" in categories:
            url = f"{base_url}/calendar/earnings"
            params = {
                "symbol": symbol,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "token": api_key
            }
            
            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("earningsCalendar", []):
                    events.append({
                        "symbol": symbol,
                        "title": f"{symbol} Earnings Call",
                        "category": "earnings",
                        "scheduled_at": datetime.fromisoformat(item["date"]),
                        "description": f"Earnings call for {symbol}",
                        "source": "finnhub",
                        "external_id": f"finnhub-earnings-{symbol}-{item['date']}",
                        "metadata": {
                            "eps_estimate": item.get("epsEstimate"),
                            "eps_actual": item.get("epsActual"),
                            "revenue_estimate": item.get("revenueEstimate"),
                            "revenue_actual": item.get("revenueActual"),
                            "quarter": item.get("quarter"),
                            "year": item.get("year")
                        }
                    })
        
        return events
    
    async def _process_event_data(
        self, 
        session: AsyncSession, 
        event_data: Dict[str, Any], 
        symbol: str
    ) -> Dict[str, Any]:
        """Process and store event data in the database."""
        # Check if event already exists
        existing_event = await session.execute(
            select(EventORM).where(
                and_(
                    EventORM.symbol == symbol,
                    EventORM.source == event_data["source"],
                    EventORM.external_id == event_data["external_id"]
                )
            )
        )
        existing = existing_event.scalar_one_or_none()
        
        # Apply categorization if categorizer is available
        category = event_data["category"]
        metadata = event_data.get("metadata", {})
        
        if self.categorizer:
            canonical_category, metadata = self.categorizer.categorize_event(
                category, 
                event_data["title"], 
                event_data.get("description", ""),
                metadata
            )
            category = canonical_category
        
        # Apply impact scoring if impact scorer is available
        impact_score = None
        if self.impact_scorer:
            impact_score = self.impact_scorer.calculate_impact_score({
                "symbol": symbol,
                "category": category,
                "title": event_data["title"],
                "description": event_data.get("description", ""),
                "metadata": metadata
            })
        
        if existing:
            # Update existing event
            existing.title = event_data["title"]
            existing.category = category
            existing.description = event_data.get("description", "")
            existing.metadata_json = metadata
            existing.updated_at = datetime.utcnow()
            if impact_score:
                existing.impact_score = impact_score
            
            await session.commit()
            return {"action": "updated", "category": category}
        else:
            # Create new event
            event = EventORM(
                symbol=symbol,
                title=event_data["title"],
                category=category,
                scheduled_at=event_data["scheduled_at"],
                description=event_data.get("description", ""),
                metadata_json=metadata,
                status="scheduled",
                source=event_data["source"],
                external_id=event_data["external_id"],
                impact_score=impact_score,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            session.add(event)
            await session.commit()
            
            # Enrich event if enrichment service is available
            if self.enrichment_service:
                try:
                    await session.refresh(event)
                    enriched_data = await self.enrichment_service.enrich_event({
                        "id": event.id,
                        "symbol": symbol,
                        "category": category,
                        "metadata": metadata
                    })
                    
                    if enriched_data.get("metadata"):
                        event.metadata_json = enriched_data["metadata"]
                        await session.commit()
                
                except Exception as e:
                    logger.warning(f"Failed to enrich backfilled event {event.id}: {e}")
            
            return {"action": "created", "category": category}
    
    async def get_backfill_statistics(self) -> Dict[str, Any]:
        """Get backfill service statistics."""
        async with self.session_factory() as session:
            # Count events by source and recency
            total_events = await session.scalar(select(func.count(EventORM.id)))
            
            backfilled_events = await session.scalar(
                select(func.count(EventORM.id)).where(
                    EventORM.source.in_([
                        "financial_modeling_prep", "alpha_vantage", 
                        "polygon", "finnhub"
                    ])
                )
            )
            
            recent_backfills = await session.scalar(
                select(func.count(EventORM.id)).where(
                    and_(
                        EventORM.source.in_([
                            "financial_modeling_prep", "alpha_vantage", 
                            "polygon", "finnhub"
                        ]),
                        EventORM.created_at >= datetime.utcnow() - timedelta(days=7)
                    )
                )
            )
        
        return {
            "service": "historical-backfill",
            "enabled": self.enabled,
            "configured_sources": list(self.sources.keys()),
            "statistics": {
                "total_events": total_events or 0,
                "backfilled_events": backfilled_events or 0,
                "recent_backfills_7d": recent_backfills or 0
            },
            "active_backfills": len(self._active_backfills),
            "queue_size": self._backfill_queue.qsize(),
            "configuration": {
                "max_concurrent_symbols": self.max_concurrent_symbols,
                "default_lookback_days": self.default_lookback_days,
                "max_days_per_request": self.max_days_per_request,
                "rate_limit_delay": self.rate_limit_delay,
                "timeout": self.timeout,
                "retry_attempts": self.retry_attempts,
                "batch_size": self.batch_size
            }
        }


def build_backfill_service(
    session_factory,
    categorizer: Optional[EventCategorizer] = None,
    impact_scorer: Optional[EventImpactScorer] = None,
    enrichment_service: Optional[EventEnrichmentService] = None,
    health_monitor: Optional[FeedHealthMonitor] = None
) -> HistoricalBackfillService:
    """Factory function to create backfill service instance."""
    return HistoricalBackfillService(
        session_factory=session_factory,
        categorizer=categorizer,
        impact_scorer=impact_scorer,
        enrichment_service=enrichment_service,
        health_monitor=health_monitor
    )