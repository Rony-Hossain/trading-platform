"""
Sentiment Service - Working version without problematic dependencies
Includes core sentiment analysis and momentum endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel

from .core.database import get_db, create_tables
from .services.sentiment_analyzer import SentimentAnalyzer
from .services.data_collectors import TwitterCollector, RedditCollector, NewsCollector
from .services.additional_collectors import enhanced_collectors
from .services.sentiment_aggregator import SentimentAggregator
from .services.data_quality_validator import DataQualityValidator
from .services.event_detection_service import EventDetectionService
from .services.sentiment_momentum import EventType, MomentumDirection
from .services.topic_modeling_service import FinancialTopicModeler
from .services.novelty_scoring import WeightedSentimentAggregator
from .models.schemas import (
    SentimentData, SentimentCreate, SentimentAnalysis,
    SocialPost, NewsArticle, SentimentSummary,
    CollectionStatus, ErrorResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services (only working ones)
sentiment_analyzer = SentimentAnalyzer()
twitter_collector = TwitterCollector()
reddit_collector = RedditCollector() 
news_collector = NewsCollector()
sentiment_aggregator = SentimentAggregator()
data_quality_validator = DataQualityValidator()
event_detection_service = EventDetectionService()
topic_modeler = FinancialTopicModeler()
weighted_aggregator = WeightedSentimentAggregator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    logger.info("Sentiment Service started")
    yield
    # Shutdown
    logger.info("Sentiment Service stopped")

app = FastAPI(
    title="Sentiment Service",
    description="Social Media and News Sentiment Analysis for Financial Markets",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CollectionRequest(BaseModel):
    symbols: List[str]
    platforms: List[str] = ["twitter", "reddit", "news"]
    duration_hours: int = 24

class EventDetectionRequest(BaseModel):
    symbols: List[str]
    days_ahead: int = 30

class MomentumAnalysisRequest(BaseModel):
    event_type: str
    event_date: str  # ISO format
    pre_event_hours: int = 72
    post_event_hours: int = 24

class ManualEventRequest(BaseModel):
    symbol: str
    event_type: str
    event_date: str  # ISO format
    event_title: str
    metadata: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Sentiment Analysis Service",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Social media sentiment analysis",
            "News sentiment analysis", 
            "Quality-weighted sentiment analysis",
            "Event-driven momentum analysis",
            "Financial topic modeling with BERTopic",
            "Real-time data collection"
        ],
        "endpoints": {
            "sentiment": "/sentiment/{symbol}",
            "analysis": "/analysis/{symbol}",
            "weighted_sentiment": "/weighted-sentiment/{symbol}",
            "topics": "/topics/{symbol}",
            "topic_history": "/topics/{symbol}/history",
            "momentum": "/momentum/",
            "events": "/momentum/events/",
            "quality": "/quality/",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "sentiment-service",
        "database": "connected"
    }

@app.get("/sentiment/{symbol}", response_model=List[SentimentData])
async def get_sentiment_data(
    symbol: str,
    hours: int = Query(default=24, description="Hours to look back"),
    limit: int = Query(default=100, description="Maximum results"),
    db: Session = Depends(get_db)
):
    """Get sentiment data for a symbol"""
    try:
        data = sentiment_aggregator.get_sentiment_data(
            db, symbol.upper(), hours=hours, limit=limit
        )
        return data
    except Exception as e:
        logger.error(f"Error getting sentiment data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{symbol}", response_model=SentimentAnalysis)
async def get_sentiment_analysis(
    symbol: str,
    hours: int = Query(default=24, description="Hours to look back"),
    db: Session = Depends(get_db)
):
    """Get detailed sentiment analysis for a symbol"""
    try:
        analysis = sentiment_aggregator.get_sentiment_analysis(
            db, symbol.upper(), hours=hours
        )
        return analysis
    except Exception as e:
        logger.error(f"Error getting sentiment analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/{symbol}", response_model=SentimentSummary)
async def get_sentiment_summary(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Get sentiment summary for a symbol"""
    summary = sentiment_aggregator.get_sentiment_summary(db, symbol.upper())
    return summary

@app.post("/collect", response_model=CollectionStatus)
async def start_collection(
    request: CollectionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start data collection for symbols"""
    try:
        collection_id = f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start collection tasks
        for platform in request.platforms:
            if platform == "twitter":
                background_tasks.add_task(
                    twitter_collector.collect_tweets_for_symbols,
                    request.symbols, request.duration_hours, db
                )
            elif platform == "reddit":
                background_tasks.add_task(
                    reddit_collector.collect_posts_for_symbols,
                    request.symbols, request.duration_hours, db
                )
            elif platform == "news":
                background_tasks.add_task(
                    news_collector.collect_news_for_symbols,
                    request.symbols, request.duration_hours, db
                )
        
        return CollectionStatus(
            collection_id=collection_id,
            symbols=request.symbols,
            platforms=request.platforms,
            status="started",
            estimated_completion=datetime.now() + timedelta(hours=request.duration_hours)
        )
    
    except Exception as e:
        logger.error(f"Error starting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/posts/{symbol}", response_model=List[SocialPost])
async def get_social_posts(
    symbol: str,
    hours: int = Query(default=24, description="Hours to look back"),
    platform: Optional[str] = Query(default=None, description="Filter by platform"),
    limit: int = Query(default=50, description="Maximum results"),
    db: Session = Depends(get_db)
):
    """Get social media posts for a symbol"""
    posts = sentiment_aggregator.get_social_posts(
        db, symbol.upper(), hours=hours, platform=platform, limit=limit
    )
    return posts

@app.get("/news/{symbol}", response_model=List[NewsArticle])
async def get_news_articles(
    symbol: str,
    hours: int = Query(default=24, description="Hours to look back"),
    limit: int = Query(default=20, description="Maximum results"),
    db: Session = Depends(get_db)
):
    """Get news articles for a symbol"""
    articles = news_collector.get_recent_articles(db, symbol.upper(), hours=hours, limit=limit)
    return articles

@app.post("/analyze", response_model=SentimentData)
async def analyze_text(
    text: str,
    symbol: str = Query(..., description="Stock symbol for context"),
    db: Session = Depends(get_db)
):
    """Analyze sentiment for arbitrary text"""
    try:
        result = await sentiment_analyzer.analyze_sentiment(text, symbol.upper())
        
        # Store the analysis
        sentiment_data = SentimentCreate(
            symbol=symbol.upper(),
            content=text,
            sentiment_score=result['sentiment_score'],
            sentiment_label=result['sentiment_label'],
            confidence=result['confidence'],
            source="manual_analysis",
            metadata=result.get('metadata', {})
        )
        
        stored_data = sentiment_aggregator.store_sentiment_data(db, sentiment_data)
        return stored_data
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trends")
async def get_sentiment_trends(
    symbols: str = Query(..., description="Comma-separated symbols"),
    hours: int = Query(default=168, description="Hours to look back (default 7 days)"),
    db: Session = Depends(get_db)
):
    """Get sentiment trends for multiple symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        trends = {}
        
        for symbol in symbol_list:
            trend_data = sentiment_aggregator.get_sentiment_trend(db, symbol, hours=hours)
            trends[symbol] = trend_data
        
        return {
            'symbols': symbol_list,
            'timeframe_hours': hours,
            'trends': trends,
            'generated_at': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting sentiment trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_service_stats(db: Session = Depends(get_db)):
    """Get service statistics and metrics"""
    stats = sentiment_aggregator.get_service_stats(db)
    return {
        'service': 'sentiment-analysis',
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    }

# Weighted Sentiment Analysis Endpoints
@app.get("/weighted-sentiment/{symbol}")
async def get_weighted_sentiment_metrics(
    symbol: str,
    hours: int = Query(default=24, description="Hours to analyze"),
    db: Session = Depends(get_db)
):
    """Get quality-weighted sentiment metrics that account for novelty and credibility"""
    try:
        result = weighted_aggregator.calculate_weighted_sentiment_metrics(
            db, symbol.upper(), timeframe_hours=hours
        )
        return result
    except Exception as e:
        logger.error(f"Error getting weighted sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quality/duplicates")
async def get_duplicate_analysis(
    symbol: str = Query(None, description="Filter by symbol"),
    hours: int = Query(default=48, description="Hours to look back"),
    db: Session = Depends(get_db)
):
    """Analyze duplicate content patterns"""
    try:
        from .models.database_models import ContentDeduplication
        from sqlalchemy import and_
        
        query = db.query(ContentDeduplication)
        
        if symbol:
            query = query.filter(ContentDeduplication.symbol == symbol.upper())
            
        if hours:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            query = query.filter(ContentDeduplication.last_seen_at >= cutoff_time)
        
        duplicates = query.filter(
            ContentDeduplication.occurrence_count > 1
        ).order_by(ContentDeduplication.occurrence_count.desc()).limit(50).all()
        
        duplicate_data = []
        for dup in duplicates:
            duplicate_data.append({
                'symbol': dup.symbol,
                'content_hash': dup.content_hash,
                'occurrence_count': dup.occurrence_count,
                'first_seen': dup.first_seen_at.isoformat(),
                'last_seen': dup.last_seen_at.isoformat(),
                'platforms': dup.platforms,
                'sources': dup.sources,
                'representative_content': dup.representative_content[:200] + "..." if len(dup.representative_content or "") > 200 else dup.representative_content
            })
        
        return {
            'symbol_filter': symbol,
            'hours_analyzed': hours,
            'total_duplicates_found': len(duplicate_data),
            'duplicates': duplicate_data,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing duplicates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quality/source-credibility")
async def get_source_credibility_stats(
    db: Session = Depends(get_db)
):
    """Get source credibility statistics"""
    try:
        from .models.database_models import SourceCredibility
        
        sources = db.query(SourceCredibility).order_by(
            SourceCredibility.base_credibility_weight.desc()
        ).all()
        
        source_stats = []
        for source in sources:
            accuracy_rate = 0.0
            if source.total_posts > 0:
                accuracy_rate = source.accurate_predictions / source.total_posts
            
            source_stats.append({
                'source_name': source.source_name,
                'platform': source.platform,
                'credibility_weight': source.base_credibility_weight,
                'total_posts': source.total_posts,
                'accuracy_rate': round(accuracy_rate, 4),
                'spam_flags': source.spam_flags,
                'last_adjustment': source.last_adjustment_at.isoformat() if source.last_adjustment_at else None
            })
        
        return {
            'total_sources': len(source_stats),
            'sources': source_stats,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting source credibility stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Topic Modeling Endpoints
@app.get("/topics/{symbol}")
async def get_current_topics(
    symbol: str,
    hours: int = Query(default=24, description="Hours to analyze for current topics"),
    top_n: int = Query(default=10, description="Number of top topics to return"),
    db: Session = Depends(get_db)
):
    """Get current topic analysis for a symbol"""
    try:
        result = topic_modeler.get_current_topics(db, symbol.upper(), hours=hours, top_n=top_n)
        return result
    except Exception as e:
        logger.error(f"Error getting current topics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics/{symbol}/history")
async def get_topic_history(
    symbol: str,
    days: int = Query(default=7, description="Number of days to analyze"),
    window_hours: int = Query(default=24, description="Hours per analysis window"),
    db: Session = Depends(get_db)
):
    """Get historical topic trends for a symbol"""
    try:
        result = topic_modeler.get_topic_history(db, symbol.upper(), days=days, window_hours=window_hours)
        return result
    except Exception as e:
        logger.error(f"Error getting topic history for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/topics/{symbol}/sentiment-correlation")
async def get_topic_sentiment_correlation(
    symbol: str,
    hours: int = Query(default=72, description="Hours to analyze"),
    db: Session = Depends(get_db)
):
    """Analyze correlation between topics and sentiment patterns"""
    try:
        result = topic_modeler.get_topic_sentiment_correlation(db, symbol.upper(), hours=hours)
        return result
    except Exception as e:
        logger.error(f"Error getting topic-sentiment correlation for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/topics/{symbol}/fit")
async def fit_topic_model(
    symbol: str,
    hours: int = Query(default=72, description="Hours of data for training"),
    force_refit: bool = Query(default=False, description="Force model refitting"),
    db: Session = Depends(get_db)
):
    """Fit or refit topic model for a symbol"""
    try:
        result = topic_modeler.fit_topics(db, symbol.upper(), hours=hours, force_refit=force_refit)
        return result
    except Exception as e:
        logger.error(f"Error fitting topic model for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Momentum Analysis Endpoints (working ones from minimal service)
@app.post("/momentum/events/detect")
async def detect_upcoming_events(
    request: EventDetectionRequest,
    db = Depends(get_db)
):
    """Detect upcoming events for given symbols"""
    try:
        events = await event_detection_service.detect_upcoming_events(
            symbols=request.symbols,
            days_ahead=request.days_ahead
        )
        
        return {
            "symbols": request.symbols,
            "days_ahead": request.days_ahead,
            "events_found": len(events),
            "events": [
                {
                    "symbol": event.symbol,
                    "event_type": event.event_type.value,
                    "event_date": event.event_date.isoformat(),
                    "event_title": event.event_title,
                    "confirmed": event.confirmed,
                    "confidence_score": event.confidence_score,
                    "source": event.source,
                    "metadata": event.metadata
                }
                for event in events
            ]
        }
    except Exception as e:
        logger.error(f"Error detecting events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/momentum/analyze/{symbol}")
async def analyze_momentum(
    symbol: str,
    request: MomentumAnalysisRequest,
    db = Depends(get_db)
):
    """Analyze sentiment momentum for a specific event"""
    try:
        from .services.sentiment_momentum import SentimentMomentumAnalyzer, EventWindow, EventType
        
        # Parse event type
        try:
            event_type = EventType(request.event_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {request.event_type}")
        
        # Parse event date
        try:
            event_date = datetime.fromisoformat(request.event_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {request.event_date}")
        
        # Create event window
        event_window = EventWindow(
            event_type=event_type,
            event_date=event_date,
            symbol=symbol.upper(),
            pre_event_hours=request.pre_event_hours,
            post_event_hours=request.post_event_hours
        )
        
        # Analyze momentum
        momentum_analyzer = SentimentMomentumAnalyzer()
        analysis = await momentum_analyzer.analyze_pre_event_momentum(event_window, db)
        
        return {
            "symbol": symbol.upper(),
            "event_type": event_type.value,
            "event_date": event_date.isoformat(),
            "analysis": {
                "predicted_direction": analysis.predicted_direction.value,
                "signal_strength": analysis.signal_strength,
                "confidence_score": analysis.confidence_score,
                "momentum_buildup_score": analysis.momentum_buildup_score,
                "peak_bullish_momentum": analysis.peak_bullish_momentum,
                "peak_bearish_momentum": analysis.peak_bearish_momentum,
                "direction_consistency": analysis.direction_consistency,
                "timeline_length": len(analysis.momentum_timeline),
                "has_momentum_data": len(analysis.momentum_timeline) > 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing momentum for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/momentum/monitor")
async def get_active_monitoring(
    hours_ahead: int = 72,
    db = Depends(get_db)
):
    """Get events currently being monitored"""
    try:
        active_events = await event_detection_service.get_active_monitoring_events(hours_ahead)
        
        return {
            "monitoring_window_hours": hours_ahead,
            "active_events": len(active_events),
            "events": [
                {
                    "symbol": event.symbol,
                    "event_type": event.event_type.value,
                    "event_date": event.event_date.isoformat(),
                    "event_title": event.event_title,
                    "confirmed": event.confirmed,
                    "hours_until_event": (event.event_date - datetime.now()).total_seconds() / 3600
                }
                for event in active_events
            ]
        }
    except Exception as e:
        logger.error(f"Error getting monitoring events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/momentum/events/add")
async def add_manual_event(request: ManualEventRequest):
    """Add a manually identified event"""
    try:
        event_date = datetime.fromisoformat(request.event_date.replace('Z', '+00:00'))
        
        success = event_detection_service.add_manual_event(
            symbol=request.symbol.upper(),
            event_type=request.event_type,
            event_date=event_date,
            event_title=request.event_title,
            metadata=request.metadata
        )
        
        if success:
            return {
                "success": True,
                "message": f"Event added for {request.symbol}",
                "event": {
                    "symbol": request.symbol.upper(),
                    "event_type": request.event_type,
                    "event_date": event_date.isoformat(),
                    "event_title": request.event_title
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to add event")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding manual event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/momentum/events/status")
async def get_event_statistics():
    """Get statistics about tracked events"""
    try:
        stats = event_detection_service.get_event_statistics()
        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting event statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)