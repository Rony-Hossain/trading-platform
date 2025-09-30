"""
FastAPI Endpoints for Fundamentals Signals

This module provides REST API endpoints for accessing Form 4 and Form 13F
derived signals with comprehensive filtering and subscription capabilities.

Key Features:
1. Signal Retrieval: Get signals with various filters
2. Performance Analytics: Track signal performance over time
3. Subscriptions: Manage signal subscriptions and notifications
4. Real-time Updates: WebSocket support for live signal feeds
5. Historical Analysis: Access historical signal data

Endpoints:
- GET /signals/form4: Retrieve Form 4 insider signals
- GET /signals/form13f: Retrieve Form 13F institutional signals
- GET /signals/performance: Get signal performance statistics
- POST /subscriptions: Create signal subscription
- WebSocket /ws/signals: Real-time signal feed
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis

from app.db.signals_persistence import SignalsPersistence
from app.core.form4_clustering import Form4Clusterer, InsiderSignal
from app.core.form13f_aggregation import Form13FAggregator, AggregatedSignal

logger = logging.getLogger(__name__)


# Pydantic models for API
class SignalFilter(BaseModel):
    """Filter parameters for signal queries."""
    ticker: Optional[str] = None
    min_strength: float = Field(default=0.0, ge=0.0, le=100.0)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)


class SubscriptionRequest(BaseModel):
    """Signal subscription request."""
    user_id: str
    signal_types: List[str] = Field(..., description="List of signal types: 'form4', 'form13f'")
    tickers: Optional[List[str]] = Field(default=None, description="List of tickers to subscribe to")
    min_strength: float = Field(default=50.0, ge=0.0, le=100.0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    webhook_url: Optional[str] = None


class SignalResponse(BaseModel):
    """Signal response model."""
    signal_id: str
    ticker: Optional[str]
    signal_type: str
    signal_strength: float
    confidence: float
    generated_date: datetime
    metadata: Dict[str, Any]


class PerformanceStats(BaseModel):
    """Signal performance statistics."""
    total_signals: int
    avg_return_1d: float
    avg_return_5d: float
    avg_return_21d: float
    avg_return_63d: float
    win_rate_1d: float
    win_rate_5d: float
    win_rate_21d: float
    win_rate_63d: float
    avg_volatility: float


class SignalsAPI:
    """FastAPI application for fundamentals signals."""
    
    def __init__(self, 
                 database_url: str,
                 redis_url: Optional[str] = None):
        """Initialize signals API."""
        self.app = FastAPI(
            title="Fundamentals Signals API",
            description="API for accessing Form 4 and Form 13F derived trading signals",
            version="1.0.0"
        )
        
        # Initialize persistence layer
        self.persistence = SignalsPersistence(database_url, redis_url)
        
        # WebSocket connection manager
        self.connection_manager = ConnectionManager()
        
        # Initialize routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize database on startup."""
            await self.persistence.initialize_database()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.utcnow()}
        
        @self.app.get("/signals/form4", response_model=List[Dict[str, Any]])
        async def get_form4_signals(
            ticker: Optional[str] = Query(None, description="Stock ticker filter"),
            min_strength: float = Query(0.0, ge=0.0, le=100.0, description="Minimum signal strength"),
            min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence level"),
            signal_type: Optional[str] = Query(None, description="Signal type filter: bullish, bearish, neutral"),
            start_date: Optional[datetime] = Query(None, description="Start date filter"),
            end_date: Optional[datetime] = Query(None, description="End date filter"),
            limit: int = Query(100, ge=1, le=1000, description="Maximum number of results")
        ):
            """
            Retrieve Form 4 insider trading signals.
            
            Returns signals based on insider trading activity with clustering analysis.
            """
            try:
                signals = await self.persistence.get_active_signals(
                    signal_type='form4',
                    ticker=ticker,
                    min_strength=min_strength,
                    min_confidence=min_confidence,
                    limit=limit
                )
                
                # Apply additional filters
                if signal_type:
                    signals = [s for s in signals if s.get('signal_type') == signal_type]
                
                if start_date:
                    signals = [s for s in signals if s.get('generated_date') >= start_date]
                
                if end_date:
                    signals = [s for s in signals if s.get('generated_date') <= end_date]
                
                return signals
                
            except Exception as e:
                logger.error(f"Failed to retrieve Form 4 signals: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/signals/form13f", response_model=List[Dict[str, Any]])
        async def get_form13f_signals(
            ticker: Optional[str] = Query(None, description="Stock ticker filter"),
            cusip: Optional[str] = Query(None, description="CUSIP filter"),
            min_strength: float = Query(0.0, ge=0.0, le=100.0, description="Minimum signal strength"),
            min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence level"),
            signal_direction: Optional[str] = Query(None, description="Signal direction: bullish, bearish, neutral"),
            min_institutions: int = Query(1, ge=1, description="Minimum number of institutions"),
            start_date: Optional[datetime] = Query(None, description="Start date filter"),
            end_date: Optional[datetime] = Query(None, description="End date filter"),
            limit: int = Query(100, ge=1, le=1000, description="Maximum number of results")
        ):
            """
            Retrieve Form 13F institutional holding change signals.
            
            Returns aggregated signals based on institutional buying/selling activity.
            """
            try:
                signals = await self.persistence.get_active_signals(
                    signal_type='form13f',
                    ticker=ticker,
                    min_strength=min_strength,
                    min_confidence=min_confidence,
                    limit=limit
                )
                
                # Apply additional filters
                if cusip:
                    signals = [s for s in signals if s.get('cusip') == cusip]
                
                if signal_direction:
                    signals = [s for s in signals if s.get('signal_direction') == signal_direction]
                
                if min_institutions:
                    signals = [s for s in signals if s.get('total_institutions', 0) >= min_institutions]
                
                if start_date:
                    signals = [s for s in signals if s.get('signal_date') >= start_date]
                
                if end_date:
                    signals = [s for s in signals if s.get('signal_date') <= end_date]
                
                return signals
                
            except Exception as e:
                logger.error(f"Failed to retrieve Form 13F signals: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/signals/performance", response_model=Dict[str, Any])
        async def get_signal_performance(
            signal_type: Optional[str] = Query(None, description="Signal type filter"),
            ticker: Optional[str] = Query(None, description="Ticker filter"),
            days_back: int = Query(90, ge=1, le=365, description="Days to look back")
        ):
            """
            Get signal performance statistics.
            
            Returns aggregated performance metrics for signals over specified period.
            """
            try:
                stats = await self.persistence.get_signal_performance_stats(
                    signal_type=signal_type,
                    ticker=ticker,
                    days_back=days_back
                )
                
                return {
                    "performance_stats": stats,
                    "period_days": days_back,
                    "signal_type": signal_type,
                    "ticker": ticker,
                    "generated_at": datetime.utcnow()
                }
                
            except Exception as e:
                logger.error(f"Failed to get performance stats: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/signals/top", response_model=List[Dict[str, Any]])
        async def get_top_signals(
            signal_type: str = Query(..., description="Signal type: form4 or form13f"),
            direction: Optional[str] = Query(None, description="Signal direction filter"),
            n: int = Query(10, ge=1, le=50, description="Number of top signals"),
            min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence")
        ):
            """
            Get top signals by strength and confidence.
            
            Returns the highest-rated signals based on combined strength and confidence score.
            """
            try:
                signals = await self.persistence.get_active_signals(
                    signal_type=signal_type,
                    min_confidence=min_confidence,
                    limit=n * 2  # Get more to filter and rank
                )
                
                # Filter by direction if specified
                if direction:
                    if signal_type == 'form4':
                        signals = [s for s in signals if s.get('signal_type') == direction]
                    elif signal_type == 'form13f':
                        signals = [s for s in signals if s.get('signal_direction') == direction]
                
                # Calculate combined score and sort
                for signal in signals:
                    strength = signal.get('signal_strength', 0)
                    confidence = signal.get('confidence', signal.get('confidence_level', 0))
                    signal['combined_score'] = strength * confidence
                
                # Sort by combined score and return top N
                top_signals = sorted(signals, key=lambda x: x.get('combined_score', 0), reverse=True)[:n]
                
                return top_signals
                
            except Exception as e:
                logger.error(f"Failed to get top signals: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/subscriptions", response_model=Dict[str, str])
        async def create_subscription(subscription: SubscriptionRequest):
            """
            Create a new signal subscription.
            
            Allows users to subscribe to specific signal types and tickers with custom filters.
            """
            try:
                # Validate signal types
                valid_types = ['form4', 'form13f']
                invalid_types = [t for t in subscription.signal_types if t not in valid_types]
                if invalid_types:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid signal types: {invalid_types}. Valid types: {valid_types}"
                    )
                
                subscription_id = await self.persistence.create_signal_subscription(
                    user_id=subscription.user_id,
                    signal_types=subscription.signal_types,
                    tickers=subscription.tickers or [],
                    min_strength=subscription.min_strength,
                    min_confidence=subscription.min_confidence,
                    webhook_url=subscription.webhook_url
                )
                
                return {
                    "subscription_id": subscription_id,
                    "status": "created",
                    "message": f"Subscription created for user {subscription.user_id}"
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to create subscription: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/subscriptions/{user_id}", response_model=List[Dict[str, Any]])
        async def get_user_subscriptions(user_id: str):
            """
            Get signal subscriptions for a user.
            
            Returns all active subscriptions for the specified user.
            """
            try:
                subscriptions = await self.persistence.get_signal_subscriptions(user_id)
                return subscriptions
                
            except Exception as e:
                logger.error(f"Failed to get subscriptions: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.websocket("/ws/signals")
        async def websocket_signals(websocket: WebSocket):
            """
            WebSocket endpoint for real-time signal feeds.
            
            Provides live updates of new signals as they are generated.
            """
            await self.connection_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    
                    # Parse subscription preferences
                    try:
                        preferences = json.loads(data)
                        await self.connection_manager.update_preferences(websocket, preferences)
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            "error": "Invalid JSON format"
                        }))
                        
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(websocket)
        
        @self.app.post("/signals/cleanup")
        async def cleanup_signals():
            """
            Clean up expired signals.
            
            Administrative endpoint to remove old and expired signals.
            """
            try:
                cleaned_count = await self.persistence.cleanup_expired_signals()
                return {
                    "status": "completed",
                    "signals_cleaned": cleaned_count,
                    "timestamp": datetime.utcnow()
                }
                
            except Exception as e:
                logger.error(f"Signal cleanup failed: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")


class ConnectionManager:
    """WebSocket connection manager for real-time signal feeds."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_preferences: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "message": "Signal feed connected"
        }))
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if websocket in self.connection_preferences:
            del self.connection_preferences[websocket]
    
    async def update_preferences(self, websocket: WebSocket, preferences: Dict):
        """Update subscription preferences for connection."""
        self.connection_preferences[websocket] = preferences
        
        await websocket.send_text(json.dumps({
            "type": "preferences",
            "status": "updated",
            "preferences": preferences
        }))
    
    async def broadcast_signal(self, signal: Dict[str, Any]):
        """Broadcast signal to all connected clients."""
        if not self.active_connections:
            return
        
        message = {
            "type": "signal",
            "data": signal,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all connections that match preferences
        disconnected = []
        
        for connection in self.active_connections:
            try:
                # Check if signal matches connection preferences
                if self._signal_matches_preferences(signal, connection):
                    await connection.send_text(json.dumps(message, default=str))
                    
            except Exception as e:
                logger.error(f"Failed to send signal to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    def _signal_matches_preferences(self, signal: Dict[str, Any], connection: WebSocket) -> bool:
        """Check if signal matches connection preferences."""
        preferences = self.connection_preferences.get(connection, {})
        
        # If no preferences set, send all signals
        if not preferences:
            return True
        
        # Check ticker filter
        if 'tickers' in preferences and preferences['tickers']:
            signal_ticker = signal.get('ticker')
            if signal_ticker not in preferences['tickers']:
                return False
        
        # Check signal type filter
        if 'signal_types' in preferences and preferences['signal_types']:
            # Determine signal type from data
            if 'signal_type' in signal:  # Form 4
                signal_type = 'form4'
            elif 'signal_direction' in signal:  # Form 13F
                signal_type = 'form13f'
            else:
                signal_type = 'unknown'
            
            if signal_type not in preferences['signal_types']:
                return False
        
        # Check minimum strength
        min_strength = preferences.get('min_strength', 0)
        signal_strength = signal.get('signal_strength', 0)
        if signal_strength < min_strength:
            return False
        
        # Check minimum confidence
        min_confidence = preferences.get('min_confidence', 0)
        confidence = signal.get('confidence', signal.get('confidence_level', 0))
        if confidence < min_confidence:
            return False
        
        return True


# Factory function to create the FastAPI app
def create_signals_app(database_url: str, redis_url: Optional[str] = None) -> FastAPI:
    """
    Create and configure the signals FastAPI application.
    
    Parameters:
    - database_url: PostgreSQL/TimescaleDB connection URL
    - redis_url: Optional Redis connection URL
    
    Returns:
    - Configured FastAPI application
    """
    signals_api = SignalsAPI(database_url, redis_url)
    return signals_api.app