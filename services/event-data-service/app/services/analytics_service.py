"""Event Analytics Service

Provides comprehensive analytics and reporting capabilities for event data,
including metrics, trends, performance analysis, and dashboard data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

from sqlalchemy import func, text, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import EventORM, EventHeadlineORM
from ..database import SessionFactory

logger = logging.getLogger(__name__)


@dataclass
class EventMetrics:
    """Container for event metrics."""
    total_events: int
    events_by_category: Dict[str, int]
    events_by_status: Dict[str, int]
    events_by_source: Dict[str, int]
    events_by_symbol: Dict[str, int]
    average_impact_score: float
    high_impact_events: int
    events_with_headlines: int
    total_headlines: int


@dataclass
class TimeSeriesPoint:
    """Data point for time series analytics."""
    timestamp: datetime
    value: int
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrendAnalysis:
    """Container for trend analysis results."""
    period: str
    growth_rate: float
    trend_direction: str  # "up", "down", "stable"
    volatility: float
    peak_timestamp: datetime
    peak_value: int
    data_points: List[TimeSeriesPoint]


@dataclass
class PerformanceReport:
    """Container for performance analysis."""
    report_period: str
    most_active_symbols: List[Tuple[str, int]]
    trending_categories: List[Tuple[str, int, float]]  # category, count, growth_rate
    impact_distribution: Dict[str, int]
    source_reliability: Dict[str, Dict[str, Any]]
    headline_coverage: Dict[str, float]  # symbol -> coverage percentage


class EventAnalyticsService:
    """Service for event analytics and reporting."""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes default cache
        self._metrics_cache: Dict[str, Tuple[datetime, Any]] = {}
    
    async def get_event_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> EventMetrics:
        """Get comprehensive event metrics."""
        
        cache_key = f"metrics_{start_date}_{end_date}_{symbols}_{categories}"
        
        if use_cache and cache_key in self._metrics_cache:
            cached_time, cached_data = self._metrics_cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl:
                return cached_data
        
        async with SessionFactory() as session:
            # Build base query
            query = session.query(EventORM)
            
            if start_date:
                query = query.filter(EventORM.scheduled_at >= start_date)
            if end_date:
                query = query.filter(EventORM.scheduled_at <= end_date)
            if symbols:
                query = query.filter(EventORM.symbol.in_(symbols))
            if categories:
                query = query.filter(EventORM.category.in_(categories))
            
            # Execute queries for different metrics
            total_events = await session.scalar(
                query.with_only_columns(func.count(EventORM.id))
            )
            
            # Events by category
            category_result = await session.execute(
                query.with_only_columns(
                    EventORM.category,
                    func.count(EventORM.id).label('count')
                ).group_by(EventORM.category)
            )
            events_by_category = dict(category_result.fetchall())
            
            # Events by status
            status_result = await session.execute(
                query.with_only_columns(
                    EventORM.status,
                    func.count(EventORM.id).label('count')
                ).group_by(EventORM.status)
            )
            events_by_status = dict(status_result.fetchall())
            
            # Events by source
            source_result = await session.execute(
                query.with_only_columns(
                    EventORM.source,
                    func.count(EventORM.id).label('count')
                ).group_by(EventORM.source)
            )
            events_by_source = dict(source_result.fetchall())
            
            # Top symbols
            symbol_result = await session.execute(
                query.with_only_columns(
                    EventORM.symbol,
                    func.count(EventORM.id).label('count')
                ).group_by(EventORM.symbol).order_by(func.count(EventORM.id).desc()).limit(20)
            )
            events_by_symbol = dict(symbol_result.fetchall())
            
            # Impact score statistics
            impact_stats = await session.execute(
                query.filter(EventORM.impact_score.is_not(None))
                .with_only_columns(
                    func.avg(EventORM.impact_score).label('avg_impact'),
                    func.count(func.case((EventORM.impact_score >= 7, 1))).label('high_impact')
                )
            )
            avg_impact, high_impact = impact_stats.fetchone()
            
            # Headlines statistics
            headline_stats = await session.execute(
                text("""
                    SELECT 
                        COUNT(DISTINCT e.id) as events_with_headlines,
                        COUNT(h.id) as total_headlines
                    FROM events e
                    LEFT JOIN event_headlines h ON e.id = h.event_id
                    WHERE (:start_date IS NULL OR e.scheduled_at >= :start_date)
                    AND (:end_date IS NULL OR e.scheduled_at <= :end_date)
                    AND (:symbols IS NULL OR e.symbol = ANY(:symbols))
                    AND (:categories IS NULL OR e.category = ANY(:categories))
                """),
                {
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbols': symbols,
                    'categories': categories
                }
            )
            events_with_headlines, total_headlines = headline_stats.fetchone()
            
            metrics = EventMetrics(
                total_events=total_events or 0,
                events_by_category=events_by_category,
                events_by_status=events_by_status,
                events_by_source=events_by_source,
                events_by_symbol=events_by_symbol,
                average_impact_score=float(avg_impact or 0),
                high_impact_events=high_impact or 0,
                events_with_headlines=events_with_headlines or 0,
                total_headlines=total_headlines or 0
            )
            
            # Cache the result
            if use_cache:
                self._metrics_cache[cache_key] = (datetime.utcnow(), metrics)
            
            return metrics
    
    async def get_time_series_data(
        self,
        metric: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
        symbols: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> List[TimeSeriesPoint]:
        """Get time series data for various metrics."""
        
        async with SessionFactory() as session:
            # Determine the appropriate date_trunc interval
            pg_interval = {
                "5m": "5 minutes",
                "15m": "15 minutes", 
                "1h": "hour",
                "1d": "day",
                "1w": "week"
            }.get(interval, "hour")
            
            base_filters = [
                EventORM.scheduled_at >= start_date,
                EventORM.scheduled_at <= end_date
            ]
            
            if symbols:
                base_filters.append(EventORM.symbol.in_(symbols))
            if categories:
                base_filters.append(EventORM.category.in_(categories))
            
            if metric == "event_count":
                result = await session.execute(
                    text("""
                        SELECT 
                            date_trunc(:interval, scheduled_at) as time_bucket,
                            COUNT(*) as value,
                            category
                        FROM events 
                        WHERE scheduled_at >= :start_date 
                        AND scheduled_at <= :end_date
                        AND (:symbols IS NULL OR symbol = ANY(:symbols))
                        AND (:categories IS NULL OR category = ANY(:categories))
                        GROUP BY time_bucket, category
                        ORDER BY time_bucket
                    """),
                    {
                        'interval': pg_interval,
                        'start_date': start_date,
                        'end_date': end_date,
                        'symbols': symbols,
                        'categories': categories
                    }
                )
                
                return [
                    TimeSeriesPoint(
                        timestamp=row.time_bucket,
                        value=row.value,
                        category=row.category
                    )
                    for row in result.fetchall()
                ]
            
            elif metric == "impact_score":
                result = await session.execute(
                    text("""
                        SELECT 
                            date_trunc(:interval, scheduled_at) as time_bucket,
                            AVG(impact_score)::int as value
                        FROM events 
                        WHERE scheduled_at >= :start_date 
                        AND scheduled_at <= :end_date
                        AND impact_score IS NOT NULL
                        AND (:symbols IS NULL OR symbol = ANY(:symbols))
                        AND (:categories IS NULL OR category = ANY(:categories))
                        GROUP BY time_bucket
                        ORDER BY time_bucket
                    """),
                    {
                        'interval': pg_interval,
                        'start_date': start_date,
                        'end_date': end_date,
                        'symbols': symbols,
                        'categories': categories
                    }
                )
                
                return [
                    TimeSeriesPoint(timestamp=row.time_bucket, value=row.value)
                    for row in result.fetchall()
                ]
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
    
    async def analyze_trends(
        self,
        metric: str,
        period_days: int = 30,
        symbols: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> TrendAnalysis:
        """Analyze trends for a specific metric over a period."""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Get time series data
        data_points = await self.get_time_series_data(
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            symbols=symbols,
            categories=categories
        )
        
        if len(data_points) < 2:
            return TrendAnalysis(
                period=f"{period_days}d",
                growth_rate=0.0,
                trend_direction="stable",
                volatility=0.0,
                peak_timestamp=end_date,
                peak_value=0,
                data_points=data_points
            )
        
        # Calculate trend metrics
        values = [point.value for point in data_points]
        first_value = values[0] if values[0] > 0 else 1
        last_value = values[-1]
        
        growth_rate = ((last_value - first_value) / first_value) * 100
        
        # Determine trend direction
        if growth_rate > 5:
            trend_direction = "up"
        elif growth_rate < -5:
            trend_direction = "down"
        else:
            trend_direction = "stable"
        
        # Calculate volatility (coefficient of variation)
        if len(values) > 1:
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            volatility = (std_dev / mean_val) * 100 if mean_val > 0 else 0
        else:
            volatility = 0
        
        # Find peak
        peak_value = max(values)
        peak_index = values.index(peak_value)
        peak_timestamp = data_points[peak_index].timestamp
        
        return TrendAnalysis(
            period=f"{period_days}d",
            growth_rate=growth_rate,
            trend_direction=trend_direction,
            volatility=volatility,
            peak_timestamp=peak_timestamp,
            peak_value=peak_value,
            data_points=data_points
        )
    
    async def generate_performance_report(
        self,
        period_days: int = 7,
        limit: int = 10
    ) -> PerformanceReport:
        """Generate comprehensive performance report."""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        previous_start = start_date - timedelta(days=period_days)
        
        async with SessionFactory() as session:
            # Most active symbols
            symbol_result = await session.execute(
                text("""
                    SELECT symbol, COUNT(*) as event_count
                    FROM events 
                    WHERE scheduled_at >= :start_date AND scheduled_at <= :end_date
                    GROUP BY symbol
                    ORDER BY event_count DESC
                    LIMIT :limit
                """),
                {'start_date': start_date, 'end_date': end_date, 'limit': limit}
            )
            most_active_symbols = list(symbol_result.fetchall())
            
            # Trending categories with growth rates
            current_categories = await session.execute(
                text("""
                    SELECT category, COUNT(*) as current_count
                    FROM events 
                    WHERE scheduled_at >= :start_date AND scheduled_at <= :end_date
                    GROUP BY category
                """),
                {'start_date': start_date, 'end_date': end_date}
            )
            current_counts = dict(current_categories.fetchall())
            
            previous_categories = await session.execute(
                text("""
                    SELECT category, COUNT(*) as previous_count
                    FROM events 
                    WHERE scheduled_at >= :previous_start AND scheduled_at < :start_date
                    GROUP BY category
                """),
                {'previous_start': previous_start, 'start_date': start_date}
            )
            previous_counts = dict(previous_categories.fetchall())
            
            trending_categories = []
            for category, current_count in current_counts.items():
                previous_count = previous_counts.get(category, 1)
                growth_rate = ((current_count - previous_count) / previous_count) * 100
                trending_categories.append((category, current_count, growth_rate))
            
            trending_categories.sort(key=lambda x: x[2], reverse=True)
            trending_categories = trending_categories[:limit]
            
            # Impact distribution
            impact_result = await session.execute(
                text("""
                    SELECT 
                        CASE 
                            WHEN impact_score >= 8 THEN 'High (8-10)'
                            WHEN impact_score >= 5 THEN 'Medium (5-7)'
                            WHEN impact_score >= 1 THEN 'Low (1-4)'
                            ELSE 'Unrated'
                        END as impact_range,
                        COUNT(*) as count
                    FROM events 
                    WHERE scheduled_at >= :start_date AND scheduled_at <= :end_date
                    GROUP BY impact_range
                """),
                {'start_date': start_date, 'end_date': end_date}
            )
            impact_distribution = dict(impact_result.fetchall())
            
            # Source reliability (events vs headlines ratio)
            source_result = await session.execute(
                text("""
                    SELECT 
                        e.source,
                        COUNT(e.id) as event_count,
                        COUNT(h.id) as headline_count,
                        AVG(e.impact_score) as avg_impact
                    FROM events e
                    LEFT JOIN event_headlines h ON e.id = h.event_id
                    WHERE e.scheduled_at >= :start_date AND e.scheduled_at <= :end_date
                    GROUP BY e.source
                """),
                {'start_date': start_date, 'end_date': end_date}
            )
            
            source_reliability = {}
            for row in source_result.fetchall():
                source, event_count, headline_count, avg_impact = row
                source_reliability[source] = {
                    'event_count': event_count,
                    'headline_count': headline_count or 0,
                    'headlines_per_event': (headline_count or 0) / event_count if event_count > 0 else 0,
                    'avg_impact_score': float(avg_impact or 0)
                }
            
            # Headline coverage by symbol
            coverage_result = await session.execute(
                text("""
                    SELECT 
                        e.symbol,
                        COUNT(e.id) as total_events,
                        COUNT(DISTINCT CASE WHEN h.id IS NOT NULL THEN e.id END) as events_with_headlines
                    FROM events e
                    LEFT JOIN event_headlines h ON e.id = h.event_id
                    WHERE e.scheduled_at >= :start_date AND e.scheduled_at <= :end_date
                    GROUP BY e.symbol
                    HAVING COUNT(e.id) >= 2
                """),
                {'start_date': start_date, 'end_date': end_date}
            )
            
            headline_coverage = {}
            for symbol, total_events, events_with_headlines in coverage_result.fetchall():
                coverage_pct = (events_with_headlines / total_events) * 100 if total_events > 0 else 0
                headline_coverage[symbol] = coverage_pct
            
            return PerformanceReport(
                report_period=f"{period_days}d",
                most_active_symbols=most_active_symbols,
                trending_categories=trending_categories,
                impact_distribution=impact_distribution,
                source_reliability=source_reliability,
                headline_coverage=headline_coverage
            )
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        
        # Get metrics for different time periods
        current_metrics = await self.get_event_metrics()
        
        # Last 7 days metrics
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly_metrics = await self.get_event_metrics(start_date=week_ago)
        
        # Last 24 hours metrics  
        day_ago = datetime.utcnow() - timedelta(days=1)
        daily_metrics = await self.get_event_metrics(start_date=day_ago)
        
        # Trend analysis
        event_trends = await self.analyze_trends("event_count", period_days=30)
        impact_trends = await self.analyze_trends("impact_score", period_days=30)
        
        # Performance report
        performance = await self.generate_performance_report(period_days=7)
        
        # Time series data for charts
        now = datetime.utcnow()
        chart_start = now - timedelta(days=7)
        
        event_chart_data = await self.get_time_series_data(
            metric="event_count",
            start_date=chart_start,
            end_date=now,
            interval="1d"
        )
        
        impact_chart_data = await self.get_time_series_data(
            metric="impact_score", 
            start_date=chart_start,
            end_date=now,
            interval="1d"
        )
        
        return {
            "summary": {
                "total_events": current_metrics.total_events,
                "weekly_events": weekly_metrics.total_events,
                "daily_events": daily_metrics.total_events,
                "avg_impact_score": current_metrics.average_impact_score,
                "high_impact_events": current_metrics.high_impact_events,
                "headline_coverage": (current_metrics.events_with_headlines / current_metrics.total_events * 100) 
                                   if current_metrics.total_events > 0 else 0
            },
            "trends": {
                "event_count": asdict(event_trends),
                "impact_score": asdict(impact_trends)
            },
            "distributions": {
                "by_category": current_metrics.events_by_category,
                "by_status": current_metrics.events_by_status,
                "by_source": current_metrics.events_by_source,
                "by_impact": performance.impact_distribution
            },
            "top_performers": {
                "active_symbols": performance.most_active_symbols,
                "trending_categories": performance.trending_categories,
                "source_reliability": performance.source_reliability
            },
            "charts": {
                "event_timeline": [asdict(point) for point in event_chart_data],
                "impact_timeline": [asdict(point) for point in impact_chart_data]
            },
            "metadata": {
                "last_updated": datetime.utcnow().isoformat(),
                "report_period": "7d",
                "cache_status": "live"
            }
        }
    
    def clear_cache(self):
        """Clear analytics cache."""
        self._metrics_cache.clear()
        logger.info("Analytics cache cleared")


# Global analytics service instance
analytics_service = EventAnalyticsService()