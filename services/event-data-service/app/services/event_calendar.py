"""
Event Calendar Service - Manages scheduled catalysts and events
Integrates with external calendar APIs and maintains event database
"""

import asyncio
import logging
import aiohttp
import feedparser
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from ..core.database import (
    ScheduledEvent, EventOutcome, EventData,
    EventType, ImpactLevel, EventStatus
)

logger = logging.getLogger(__name__)


class EventCalendarService:
    """Service for managing event calendar data"""
    
    def __init__(self):
        self.running = False
        self.refresh_interval = 3600  # 1 hour
        self.last_refresh = {}
        self.data_sources = {
            "earnings_calendar": "https://api.nasdaq.com/api/calendar/earnings",
            "fda_calendar": "https://www.fda.gov/news-events/fda-newsroom/rss",
            "sec_filings": "https://www.sec.gov/cgi-bin/browse-edgar",
        }
    
    async def start_background_tasks(self):
        """Start background refresh tasks"""
        self.running = True
        asyncio.create_task(self._background_refresh_loop())
        logger.info("Event calendar background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks"""
        self.running = False
        logger.info("Event calendar background tasks stopped")
    
    async def _background_refresh_loop(self):
        """Background task to refresh event data"""
        while self.running:
            try:
                await self._refresh_all_calendars()
                await asyncio.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Error in background refresh: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _refresh_all_calendars(self):
        """Refresh all event calendar sources"""
        try:
            # Refresh earnings calendar
            await self._refresh_earnings_calendar()
            
            # Refresh regulatory events
            await self._refresh_regulatory_events()
            
            # Refresh product launches (synthetic for now)
            await self._refresh_product_launches()
            
            self.last_refresh["all"] = datetime.now()
            logger.info("Completed calendar refresh cycle")
            
        except Exception as e:
            logger.error(f"Error refreshing calendars: {e}")
    
    async def _refresh_earnings_calendar(self):
        """Refresh earnings calendar data"""
        try:
            # For demo purposes, create synthetic earnings events
            # In production, this would call real APIs like Finnhub, Alpha Vantage, etc.
            
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD"]
            
            for symbol in symbols:
                # Create upcoming earnings event
                event_date = date.today() + timedelta(days=7 + (hash(symbol) % 30))
                
                # Check if event already exists
                # In a real implementation, we'd query the database first
                
                # Create synthetic event data
                event_data = {
                    "symbol": symbol,
                    "company_name": f"{symbol} Inc.",
                    "event_type": EventType.EARNINGS,
                    "event_date": event_date,
                    "title": f"{symbol} Q4 2024 Earnings Call",
                    "description": f"Quarterly earnings call and financial results for {symbol}",
                    "impact_level": ImpactLevel.HIGH if symbol in ["AAPL", "MSFT", "GOOGL"] else ImpactLevel.MEDIUM,
                    "status": EventStatus.SCHEDULED,
                    "source": "synthetic_calendar",
                    "metadata": {
                        "quarter": "Q4",
                        "fiscal_year": 2024,
                        "estimated_eps": 2.50 + (hash(symbol) % 100) / 100,
                        "estimated_revenue": 50000 + (hash(symbol) % 20000),
                        "announcement_time": "AMC"  # After Market Close
                    }
                }
                
                # This would store to database in real implementation
                logger.debug(f"Would store earnings event for {symbol}")
            
            self.last_refresh["earnings"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error refreshing earnings calendar: {e}")
    
    async def _refresh_regulatory_events(self):
        """Refresh regulatory and FDA events"""
        try:
            # Synthetic regulatory events
            regulatory_events = [
                {
                    "symbol": "PFE",
                    "event_type": EventType.FDA_APPROVAL,
                    "title": "FDA Decision on New Drug Application",
                    "event_date": date.today() + timedelta(days=14),
                    "impact_level": ImpactLevel.HIGH,
                },
                {
                    "symbol": "MRNA",
                    "event_type": EventType.CLINICAL_TRIAL,
                    "title": "Phase 3 Clinical Trial Results",
                    "event_date": date.today() + timedelta(days=21),
                    "impact_level": ImpactLevel.MEDIUM,
                },
            ]
            
            for event in regulatory_events:
                event.update({
                    "company_name": f"{event['symbol']} Inc.",
                    "description": f"Regulatory event for {event['symbol']}",
                    "status": EventStatus.SCHEDULED,
                    "source": "synthetic_regulatory",
                    "metadata": {"regulatory_body": "FDA"}
                })
                
                logger.debug(f"Would store regulatory event for {event['symbol']}")
            
            self.last_refresh["regulatory"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error refreshing regulatory events: {e}")
    
    async def _refresh_product_launches(self):
        """Refresh product launch events"""
        try:
            # Synthetic product launch events
            product_events = [
                {
                    "symbol": "AAPL",
                    "event_type": EventType.PRODUCT_LAUNCH,
                    "title": "iPhone 16 Launch Event",
                    "event_date": date.today() + timedelta(days=45),
                    "impact_level": ImpactLevel.HIGH,
                },
                {
                    "symbol": "TSLA",
                    "event_type": EventType.PRODUCT_LAUNCH,
                    "title": "Cybertruck Production Update",
                    "event_date": date.today() + timedelta(days=30),
                    "impact_level": ImpactLevel.MEDIUM,
                },
            ]
            
            for event in product_events:
                event.update({
                    "company_name": f"{event['symbol']} Inc.",
                    "description": f"Product launch event for {event['symbol']}",
                    "status": EventStatus.SCHEDULED,
                    "source": "synthetic_products",
                    "metadata": {"product_category": "consumer_tech"}
                })
                
                logger.debug(f"Would store product event for {event['symbol']}")
            
            self.last_refresh["products"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error refreshing product events: {e}")
    
    async def get_upcoming_events(
        self,
        end_date: date,
        event_types: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        min_impact: Optional[str] = None,
        db: Session = None,
    ) -> List[EventData]:
        """Get upcoming events from database or generate synthetic data"""
        
        # For now, return synthetic data
        # In production, this would query the database
        
        upcoming_events = []
        today = date.today()
        
        # Synthetic upcoming events
        symbols_list = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]
        
        for i, symbol in enumerate(symbols_list):
            if len(upcoming_events) >= 20:  # Limit results
                break
                
            event_date = today + timedelta(days=7 + i * 3)
            if event_date > end_date:
                continue
            
            event = EventData(
                id=i + 1,
                symbol=symbol,
                company_name=f"{symbol} Inc.",
                event_type="earnings",
                event_date=event_date,
                event_time=None,
                title=f"{symbol} Q4 2024 Earnings Call",
                description=f"Quarterly earnings call and financial results for {symbol}",
                impact_level="high" if symbol in ["AAPL", "MSFT", "GOOGL"] else "medium",
                status="scheduled",
                source="synthetic_calendar",
                source_url=None,
                metadata={
                    "quarter": "Q4",
                    "fiscal_year": 2024,
                    "estimated_eps": 2.50 + (hash(symbol) % 100) / 100,
                    "estimated_revenue": 50000 + (hash(symbol) % 20000),
                    "announcement_time": "AMC"
                },
                actual_outcome=None,
                outcome_timestamp=None,
                surprise_score=None,
            )
            
            # Apply filters
            if event_types and event.event_type not in event_types:
                continue
            if min_impact and event.impact_level != min_impact:
                continue
                
            upcoming_events.append(event)
        
        return upcoming_events
    
    async def get_recent_events(
        self,
        start_date: date,
        event_types: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        include_outcomes: bool = True,
        db: Session = None,
    ) -> List[EventData]:
        """Get recent events with outcomes"""
        
        recent_events = []
        today = date.today()
        
        # Synthetic recent events with outcomes
        symbols_list = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN"]
        
        for i, symbol in enumerate(symbols_list):
            event_date = today - timedelta(days=3 + i * 2)
            if event_date < start_date:
                continue
            
            # Create event with synthetic outcome
            surprise_score = 0.15 + (hash(symbol) % 50) / 100  # Random surprise
            actual_eps = 2.75 + (hash(symbol) % 100) / 100
            estimated_eps = 2.50 + (hash(symbol) % 100) / 100
            
            event = EventData(
                id=100 + i,
                symbol=symbol,
                company_name=f"{symbol} Inc.",
                event_type="earnings",
                event_date=event_date,
                event_time=datetime.combine(event_date, datetime.min.time().replace(hour=16)),
                title=f"{symbol} Q3 2024 Earnings Results",
                description=f"Q3 2024 earnings results for {symbol}",
                impact_level="high" if symbol in ["AAPL", "MSFT"] else "medium",
                status="completed",
                source="synthetic_calendar",
                source_url=None,
                metadata={
                    "quarter": "Q3",
                    "fiscal_year": 2024,
                    "estimated_eps": estimated_eps,
                    "actual_eps": actual_eps,
                    "surprise_percent": ((actual_eps - estimated_eps) / estimated_eps) * 100,
                },
                actual_outcome=f"Reported EPS of ${actual_eps:.2f}, beating estimates of ${estimated_eps:.2f}",
                outcome_timestamp=datetime.combine(event_date, datetime.min.time().replace(hour=16, minute=30)),
                surprise_score=surprise_score,
            )
            
            # Apply filters
            if event_types and event.event_type not in event_types:
                continue
                
            recent_events.append(event)
        
        return recent_events
    
    async def get_symbol_events(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        event_types: Optional[List[str]] = None,
        db: Session = None,
    ) -> List[EventData]:
        """Get all events for a specific symbol"""
        
        events = []
        today = date.today()
        
        # Get upcoming events for symbol
        if end_date and end_date >= today:
            upcoming = await self.get_upcoming_events(
                end_date=end_date,
                symbols=[symbol],
                event_types=event_types,
                db=db
            )
            events.extend(upcoming)
        
        # Get recent events for symbol
        if start_date and start_date <= today:
            recent = await self.get_recent_events(
                start_date=start_date,
                symbols=[symbol],
                event_types=event_types,
                db=db
            )
            events.extend(recent)
        
        # Sort by event date
        events.sort(key=lambda x: x.event_date)
        
        return events
    
    async def get_calendar_view(
        self,
        start_date: date,
        end_date: date,
        event_types: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        db: Session = None,
    ) -> Dict[str, List[EventData]]:
        """Get calendar view organized by date"""
        
        calendar = {}
        current_date = start_date
        
        while current_date <= end_date:
            calendar[current_date.isoformat()] = []
            current_date += timedelta(days=1)
        
        # Get all events in date range
        all_events = []
        
        # Get recent events
        if start_date <= date.today():
            recent = await self.get_recent_events(
                start_date=start_date,
                event_types=event_types,
                symbols=symbols,
                db=db
            )
            all_events.extend(recent)
        
        # Get upcoming events
        if end_date >= date.today():
            upcoming = await self.get_upcoming_events(
                end_date=end_date,
                event_types=event_types,
                symbols=symbols,
                db=db
            )
            all_events.extend(upcoming)
        
        # Organize by date
        for event in all_events:
            date_key = event.event_date.isoformat()
            if date_key in calendar:
                calendar[date_key].append(event)
        
        return calendar
    
    async def get_event_types(self) -> Dict[str, Any]:
        """Get available event types and their descriptions"""
        return {
            "event_types": [
                {
                    "type": "earnings",
                    "name": "Earnings Announcements",
                    "description": "Quarterly and annual earnings reports",
                    "typical_impact": "high",
                },
                {
                    "type": "product_launch",
                    "name": "Product Launches",
                    "description": "New product announcements and launches",
                    "typical_impact": "medium",
                },
                {
                    "type": "analyst_day",
                    "name": "Analyst Days",
                    "description": "Investor and analyst presentation events",
                    "typical_impact": "medium",
                },
                {
                    "type": "regulatory_decision",
                    "name": "Regulatory Decisions",
                    "description": "Government and regulatory agency decisions",
                    "typical_impact": "high",
                },
                {
                    "type": "fda_approval",
                    "name": "FDA Approvals",
                    "description": "FDA drug and device approvals",
                    "typical_impact": "high",
                },
                {
                    "type": "clinical_trial",
                    "name": "Clinical Trial Results",
                    "description": "Clinical trial outcomes and data releases",
                    "typical_impact": "medium",
                },
            ]
        }
    
    async def manual_refresh(
        self,
        symbol: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Manually trigger refresh for specific symbol or event type"""
        
        try:
            if symbol:
                # Refresh events for specific symbol
                logger.info(f"Manual refresh triggered for symbol: {symbol}")
                # In production, this would refresh specific symbol data
                
            elif event_type:
                # Refresh specific event type
                logger.info(f"Manual refresh triggered for event type: {event_type}")
                if event_type == "earnings":
                    await self._refresh_earnings_calendar()
                elif event_type == "regulatory":
                    await self._refresh_regulatory_events()
                elif event_type == "product_launch":
                    await self._refresh_product_launches()
            else:
                # Refresh all
                await self._refresh_all_calendars()
            
            return {
                "status": "success",
                "refreshed_at": datetime.now().isoformat(),
                "symbol": symbol,
                "event_type": event_type,
            }
            
        except Exception as e:
            logger.error(f"Error in manual refresh: {e}")
            return {
                "status": "error",
                "error": str(e),
                "refreshed_at": datetime.now().isoformat(),
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the event calendar service"""
        
        last_refresh_time = self.last_refresh.get("all")
        is_healthy = True
        
        if last_refresh_time:
            time_since_refresh = (datetime.now() - last_refresh_time).total_seconds()
            is_healthy = time_since_refresh < (self.refresh_interval * 2)  # Allow 2x interval
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "last_refresh": last_refresh_time.isoformat() if last_refresh_time else None,
            "refresh_interval_seconds": self.refresh_interval,
            "running": self.running,
        }
    
    async def get_stats(self, db: Session = None) -> Dict[str, Any]:
        """Get service statistics"""
        
        # In production, this would query actual database stats
        return {
            "total_events": 150,  # Synthetic
            "upcoming_events": 75,
            "completed_events": 75,
            "event_types": {
                "earnings": 80,
                "product_launch": 30,
                "regulatory": 25,
                "other": 15,
            },
            "symbols_tracked": 50,
            "last_refresh": self.last_refresh.get("all", datetime.now()).isoformat(),
        }


# Global service instance
event_calendar_service = EventCalendarService()