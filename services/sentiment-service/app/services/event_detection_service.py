"""
Event detection and management service for identifying scheduled events.
Integrates with earnings calendars, FDA approvals, and other event sources.
"""

import logging
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from .sentiment_momentum import (
    SentimentMomentumAnalyzer, EventWindow, EventType, 
    PreEventAnalysis, EventOutcome
)

logger = logging.getLogger(__name__)

@dataclass
class EventSchedule:
    symbol: str
    event_type: EventType
    event_date: datetime
    event_title: str
    confirmed: bool = False
    confidence_score: float = 0.8
    source: str = "automatic"
    metadata: Dict[str, Any] = None

class EventDetectionService:
    """Service for detecting and managing scheduled events"""
    
    def __init__(self):
        self.momentum_analyzer = SentimentMomentumAnalyzer()
        self.known_events = {}  # symbol -> List[EventSchedule]
        self.analysis_cache = {}  # event_id -> PreEventAnalysis
        
    async def detect_upcoming_events(self, symbols: List[str], 
                                   days_ahead: int = 30) -> List[EventSchedule]:
        """
        Detect upcoming events for given symbols.
        
        Args:
            symbols: List of stock symbols to check
            days_ahead: How many days ahead to look for events
            
        Returns:
            List of detected events
        """
        try:
            logger.info(f"Detecting events for {len(symbols)} symbols, {days_ahead} days ahead")
            
            all_events = []
            
            # Check each symbol for events
            for symbol in symbols:
                try:
                    # Get earnings calendar
                    earnings_events = await self._get_earnings_events(symbol, days_ahead)
                    all_events.extend(earnings_events)
                    
                    # Get FDA events (for biotech stocks)
                    if await self._is_biotech_stock(symbol):
                        fda_events = await self._get_fda_events(symbol, days_ahead)
                        all_events.extend(fda_events)
                    
                    # Get analyst events
                    analyst_events = await self._get_analyst_events(symbol, days_ahead)
                    all_events.extend(analyst_events)
                    
                except Exception as e:
                    logger.warning(f"Error detecting events for {symbol}: {e}")
                    continue
            
            # Sort by event date
            all_events.sort(key=lambda x: x.event_date)
            
            # Update known events cache
            for event in all_events:
                if event.symbol not in self.known_events:
                    self.known_events[event.symbol] = []
                self.known_events[event.symbol].append(event)
            
            logger.info(f"Detected {len(all_events)} upcoming events")
            return all_events
            
        except Exception as e:
            logger.error(f"Error detecting events: {e}")
            return []
    
    async def _get_earnings_events(self, symbol: str, days_ahead: int) -> List[EventSchedule]:
        """Get earnings events from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is None or calendar.empty:
                return []
            
            events = []
            today = datetime.now()
            cutoff = today + timedelta(days=days_ahead)
            
            # Yahoo Finance calendar has earnings dates
            for date_col in calendar.columns:
                try:
                    event_date = pd.to_datetime(date_col)
                    if today <= event_date <= cutoff:
                        event = EventSchedule(
                            symbol=symbol,
                            event_type=EventType.EARNINGS,
                            event_date=event_date,
                            event_title=f"{symbol} Earnings Call",
                            confirmed=True,
                            confidence_score=0.9,
                            source="yahoo_finance",
                            metadata={'quarter': self._get_quarter(event_date)}
                        )
                        events.append(event)
                except Exception as e:
                    logger.debug(f"Error parsing calendar date for {symbol}: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.warning(f"Error getting earnings events for {symbol}: {e}")
            return []
    
    async def _get_fda_events(self, symbol: str, days_ahead: int) -> List[EventSchedule]:
        """Get FDA approval events (simplified - would integrate with FDA calendar)"""
        try:
            # This is a placeholder - in production, would integrate with FDA databases
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.warning(f"Error getting FDA events for {symbol}: {e}")
            return []
    
    async def _get_analyst_events(self, symbol: str, days_ahead: int) -> List[EventSchedule]:
        """Get analyst day and guidance events"""
        try:
            # This is a placeholder - would integrate with financial calendars
            # For demonstration, create some mock events
            events = []
            
            # Mock analyst day (quarterly)
            today = datetime.now()
            next_quarter = today + timedelta(days=90)
            
            if days_ahead >= 90:
                event = EventSchedule(
                    symbol=symbol,
                    event_type=EventType.ANALYST_DAY,
                    event_date=next_quarter,
                    event_title=f"{symbol} Analyst Day",
                    confirmed=False,
                    confidence_score=0.5,
                    source="predicted",
                    metadata={'type': 'quarterly_update'}
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.warning(f"Error getting analyst events for {symbol}: {e}")
            return []
    
    async def _is_biotech_stock(self, symbol: str) -> bool:
        """Check if stock is in biotech sector"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            sector = info.get('sector', '').lower()
            industry = info.get('industry', '').lower()
            
            biotech_keywords = ['biotech', 'pharmaceutical', 'drug', 'medical', 'healthcare']
            
            return any(keyword in sector or keyword in industry for keyword in biotech_keywords)
            
        except Exception as e:
            logger.debug(f"Error checking biotech status for {symbol}: {e}")
            return False
    
    def _get_quarter(self, date: datetime) -> str:
        """Get quarter string for earnings date"""
        quarter = (date.month - 1) // 3 + 1
        return f"Q{quarter} {date.year}"
    
    async def analyze_event_momentum(self, event: EventSchedule, 
                                   db: Session) -> PreEventAnalysis:
        """
        Analyze sentiment momentum for a specific event.
        
        Args:
            event: Event to analyze
            db: Database session
            
        Returns:
            Pre-event momentum analysis
        """
        try:
            logger.info(f"Analyzing momentum for {event.symbol} {event.event_type.value}")
            
            # Create event window
            event_window = EventWindow(
                event_type=event.event_type,
                event_date=event.event_date,
                symbol=event.symbol,
                pre_event_hours=72,  # 3 days before
                post_event_hours=24, # 1 day after
                metadata=event.metadata or {}
            )
            
            # Check cache first
            event_id = f"{event.symbol}_{event.event_type.value}_{event.event_date.isoformat()}"
            if event_id in self.analysis_cache:
                return self.analysis_cache[event_id]
            
            # Perform momentum analysis
            analysis = await self.momentum_analyzer.analyze_pre_event_momentum(event_window, db)
            
            # Cache the result
            self.analysis_cache[event_id] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing event momentum: {e}")
            return self.momentum_analyzer._create_empty_analysis(
                EventWindow(event.event_type, event.event_date, event.symbol)
            )
    
    async def validate_event_predictions(self, event: EventSchedule, 
                                       analysis: PreEventAnalysis,
                                       db: Session) -> EventOutcome:
        """
        Validate event predictions against actual outcomes.
        
        Args:
            event: Original event
            analysis: Pre-event analysis
            db: Database session
            
        Returns:
            Event outcome with validation results
        """
        try:
            logger.info(f"Validating predictions for {event.symbol} {event.event_type.value}")
            
            # Only validate if event has passed
            if event.event_date > datetime.now():
                logger.info(f"Event {event.symbol} has not occurred yet")
                return self.momentum_analyzer._create_empty_outcome(analysis)
            
            # Perform validation
            outcome = await self.momentum_analyzer.validate_momentum_signals(analysis, db)
            
            return outcome
            
        except Exception as e:
            logger.error(f"Error validating event predictions: {e}")
            return self.momentum_analyzer._create_empty_outcome(analysis)
    
    async def get_active_monitoring_events(self, hours_ahead: int = 72) -> List[EventSchedule]:
        """
        Get events that should be actively monitored (within monitoring window).
        
        Args:
            hours_ahead: Hours ahead to consider for active monitoring
            
        Returns:
            List of events to monitor
        """
        try:
            active_events = []
            now = datetime.now()
            cutoff = now + timedelta(hours=hours_ahead)
            
            for symbol, events in self.known_events.items():
                for event in events:
                    # Event is within monitoring window
                    if now <= event.event_date <= cutoff:
                        active_events.append(event)
                    # Event is currently happening (within 24 hours)
                    elif abs((event.event_date - now).total_seconds()) <= 24 * 3600:
                        active_events.append(event)
            
            return sorted(active_events, key=lambda x: x.event_date)
            
        except Exception as e:
            logger.error(f"Error getting active monitoring events: {e}")
            return []
    
    async def run_event_monitoring_cycle(self, db: Session) -> Dict[str, Any]:
        """
        Run a complete event monitoring cycle.
        
        Returns:
            Summary of monitoring results
        """
        try:
            logger.info("Starting event monitoring cycle")
            
            # Get active events to monitor
            active_events = await self.get_active_monitoring_events(hours_ahead=72)
            
            monitoring_results = {
                'timestamp': datetime.now().isoformat(),
                'active_events': len(active_events),
                'analyses': [],
                'high_confidence_signals': [],
                'validation_results': []
            }
            
            # Analyze momentum for each active event
            for event in active_events:
                try:
                    analysis = await self.analyze_event_momentum(event, db)
                    
                    analysis_summary = {
                        'symbol': event.symbol,
                        'event_type': event.event_type.value,
                        'event_date': event.event_date.isoformat(),
                        'predicted_direction': analysis.predicted_direction.value,
                        'signal_strength': analysis.signal_strength,
                        'confidence_score': analysis.confidence_score,
                        'momentum_buildup_score': analysis.momentum_buildup_score
                    }
                    
                    monitoring_results['analyses'].append(analysis_summary)
                    
                    # Identify high-confidence signals
                    if analysis.confidence_score > 0.7 and analysis.signal_strength > 0.6:
                        monitoring_results['high_confidence_signals'].append(analysis_summary)
                    
                    # Validate if event has passed
                    if event.event_date < datetime.now():
                        outcome = await self.validate_event_predictions(event, analysis, db)
                        
                        validation_summary = {
                            'symbol': event.symbol,
                            'event_type': event.event_type.value,
                            'momentum_accuracy': outcome.momentum_prediction_accuracy,
                            'direction_accuracy': outcome.direction_prediction_accuracy,
                            'price_move_24h': outcome.price_move_24h,
                            'signal_correlation': outcome.signal_strength_correlation
                        }
                        
                        monitoring_results['validation_results'].append(validation_summary)
                
                except Exception as e:
                    logger.warning(f"Error processing event {event.symbol}: {e}")
                    continue
            
            logger.info(f"Event monitoring cycle completed: {len(active_events)} events processed")
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Error in event monitoring cycle: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'active_events': 0,
                'analyses': [],
                'high_confidence_signals': [],
                'validation_results': []
            }
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked events"""
        try:
            total_events = sum(len(events) for events in self.known_events.values())
            
            # Count by event type
            event_type_counts = {}
            for events in self.known_events.values():
                for event in events:
                    event_type = event.event_type.value
                    event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Count by timeframe
            now = datetime.now()
            upcoming_7d = 0
            upcoming_30d = 0
            
            for events in self.known_events.values():
                for event in events:
                    days_until = (event.event_date - now).days
                    if 0 <= days_until <= 7:
                        upcoming_7d += 1
                    elif 0 <= days_until <= 30:
                        upcoming_30d += 1
            
            return {
                'total_tracked_symbols': len(self.known_events),
                'total_events': total_events,
                'event_type_distribution': event_type_counts,
                'upcoming_7_days': upcoming_7d,
                'upcoming_30_days': upcoming_30d,
                'cached_analyses': len(self.analysis_cache)
            }
            
        except Exception as e:
            logger.error(f"Error getting event statistics: {e}")
            return {'error': str(e)}
    
    def add_manual_event(self, symbol: str, event_type: str, event_date: datetime,
                        event_title: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a manually identified event"""
        try:
            # Validate event type
            try:
                event_type_enum = EventType(event_type.lower())
            except ValueError:
                logger.error(f"Invalid event type: {event_type}")
                return False
            
            event = EventSchedule(
                symbol=symbol.upper(),
                event_type=event_type_enum,
                event_date=event_date,
                event_title=event_title,
                confirmed=True,
                confidence_score=1.0,
                source="manual",
                metadata=metadata or {}
            )
            
            if symbol not in self.known_events:
                self.known_events[symbol] = []
            
            self.known_events[symbol].append(event)
            logger.info(f"Added manual event: {symbol} {event_type} on {event_date}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding manual event: {e}")
            return False
    
    def remove_event(self, symbol: str, event_date: datetime, event_type: str) -> bool:
        """Remove a tracked event"""
        try:
            if symbol not in self.known_events:
                return False
            
            events = self.known_events[symbol]
            original_count = len(events)
            
            # Remove matching events
            self.known_events[symbol] = [
                event for event in events
                if not (event.event_date == event_date and event.event_type.value == event_type)
            ]
            
            removed_count = original_count - len(self.known_events[symbol])
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} events for {symbol}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing event: {e}")
            return False