"""
Sentiment Service - Social Media and News Sentiment Analysis
Ingests data from Twitter/X, Reddit, and news sources for financial sentiment
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
# Temporarily commented out due to missing dependencies (torch, transformers, etc.)
# from .services.distribution_monitor import DistributionMonitor, MonitoringJob, AlertConfig, MonitoringInterval
# from .services.transformer_training_service import TransformerTrainingService, TrainingJobConfig
# from .services.financial_transformer import ModelArchitecture, FinancialTarget
from .services.event_detection_service import EventDetectionService
from .services.sentiment_momentum import EventType, MomentumDirection
from .models.schemas import (
    SentimentData, SentimentCreate, SentimentAnalysis,
    SocialPost, NewsArticle, SentimentSummary,
    CollectionStatus, ErrorResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
sentiment_analyzer = SentimentAnalyzer()
twitter_collector = TwitterCollector()
reddit_collector = RedditCollector() 
news_collector = NewsCollector()
sentiment_aggregator = SentimentAggregator()
data_quality_validator = DataQualityValidator()
# Temporarily commented out due to missing dependencies
# distribution_monitor = DistributionMonitor()
# transformer_training_service = TransformerTrainingService()
event_detection_service = EventDetectionService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    # await distribution_monitor.start_monitoring()
    logger.info("Sentiment Service started")
    yield
    # Shutdown
    # await distribution_monitor.stop_monitoring()
    logger.info("Sentiment Service stopped")

app = FastAPI(
    title="Sentiment Service",
    description="Social media and news sentiment analysis for financial markets",
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

# Models for API requests
class SentimentRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"  # 1h, 1d, 1w
    sources: List[str] = ["twitter", "reddit", "news"]

class CollectionRequest(BaseModel):
    symbols: List[str]
    sources: List[str] = ["twitter", "reddit", "news", "threads", "truthsocial"]
    keywords: Optional[List[str]] = None

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Sentiment Service",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "sentiment": "/sentiment/{symbol}",
            "analysis": "/analysis/{symbol}", 
            "collect": "/collect",
            "summary": "/summary/{symbol}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "service": "sentiment-service",
        "timestamp": datetime.now().isoformat(),
        "collectors": {
            "twitter": twitter_collector.is_healthy(),
            "reddit": reddit_collector.is_healthy(),
            "news": news_collector.is_healthy(),
            "threads": enhanced_collectors.get_collector("threads").is_healthy(),
            "truthsocial": enhanced_collectors.get_collector("truthsocial").is_healthy(),
            "discord": enhanced_collectors.get_collector("discord").is_healthy(),
            "telegram": enhanced_collectors.get_collector("telegram").is_healthy(),
            "bluesky": enhanced_collectors.get_collector("bluesky").is_healthy()
        },
        "analyzer": sentiment_analyzer.is_healthy()
    }
    return status

@app.get("/sentiment/{symbol}", response_model=List[SentimentData])
async def get_sentiment_data(
    symbol: str,
    timeframe: str = Query("1d", description="Time range: 1h, 1d, 1w, 1m"),
    sources: Optional[str] = Query(None, description="Comma-separated sources: twitter,reddit,news"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get raw sentiment data for a symbol"""
    try:
        source_list = sources.split(",") if sources else ["twitter", "reddit", "news"]
        
        # Calculate time range
        time_delta = {
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1), 
            "1w": timedelta(weeks=1),
            "1m": timedelta(days=30)
        }.get(timeframe, timedelta(days=1))
        
        start_time = datetime.now() - time_delta
        
        return sentiment_aggregator.get_sentiment_data(
            db, symbol, source_list, start_time, limit
        )
    except Exception as e:
        logger.error(f"Error getting sentiment data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{symbol}", response_model=SentimentAnalysis)
async def get_sentiment_analysis(
    symbol: str,
    timeframe: str = Query("1d", description="Time range for analysis"),
    db: Session = Depends(get_db)
):
    """Get aggregated sentiment analysis for a symbol"""
    try:
        time_delta = {
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1), 
            "1m": timedelta(days=30)
        }.get(timeframe, timedelta(days=1))
        
        start_time = datetime.now() - time_delta
        
        return await sentiment_aggregator.analyze_sentiment(
            db, symbol, start_time
        )
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/{symbol}", response_model=SentimentSummary)
async def get_sentiment_summary(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Get current sentiment summary for a symbol"""
    try:
        return await sentiment_aggregator.get_sentiment_summary(db, symbol)
    except Exception as e:
        logger.error(f"Error getting sentiment summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collect", response_model=CollectionStatus)
async def start_collection(
    request: CollectionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start background collection for specified symbols and sources"""
    try:
        # Add collection tasks to background
        if "twitter" in request.sources:
            background_tasks.add_task(
                twitter_collector.collect_for_symbols,
                request.symbols, request.keywords, db
            )
        
        if "reddit" in request.sources:
            background_tasks.add_task(
                reddit_collector.collect_for_symbols, 
                request.symbols, request.keywords, db
            )
        
        if "news" in request.sources:
            background_tasks.add_task(
                news_collector.collect_for_symbols,
                request.symbols, request.keywords, db
            )
        
        # Enhanced platforms
        enhanced_platforms = ["threads", "truthsocial", "discord", "telegram", "bluesky"]
        for platform in enhanced_platforms:
            if platform in request.sources:
                collector = enhanced_collectors.get_collector(platform)
                if collector and collector.is_healthy():
                    background_tasks.add_task(
                        collector.collect_for_symbols,
                        request.symbols, request.keywords, db
                    )
        
        return CollectionStatus(
            status="started",
            symbols=request.symbols,
            sources=request.sources,
            started_at=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error starting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/posts/{symbol}", response_model=List[SocialPost])
async def get_social_posts(
    symbol: str,
    source: str = Query(..., description="Source: twitter, reddit"),
    limit: int = Query(50, ge=1, le=500),
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """Get recent social media posts for a symbol"""
    try:
        start_time = datetime.now() - timedelta(hours=hours)
        
        if source == "twitter":
            return twitter_collector.get_recent_posts(db, symbol, start_time, limit)
        elif source == "reddit":
            return reddit_collector.get_recent_posts(db, symbol, start_time, limit)
        else:
            raise HTTPException(status_code=400, detail="Invalid source")
    except Exception as e:
        logger.error(f"Error getting social posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news/{symbol}", response_model=List[NewsArticle])
async def get_news_articles(
    symbol: str,
    limit: int = Query(50, ge=1, le=200),
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """Get recent news articles for a symbol"""
    try:
        start_time = datetime.now() - timedelta(hours=hours)
        return news_collector.get_recent_articles(db, symbol, start_time, limit)
    except Exception as e:
        logger.error(f"Error getting news articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=SentimentData)
async def analyze_text(
    text: str,
    symbol: Optional[str] = None,
    source: str = "manual",
    db: Session = Depends(get_db)
):
    """Analyze sentiment of provided text"""
    try:
        analysis = await sentiment_analyzer.analyze_text(text)
        
        # Store if symbol provided
        if symbol:
            sentiment_data = SentimentCreate(
                symbol=symbol,
                source=source,
                content=text,
                sentiment_score=analysis.compound,
                sentiment_label=analysis.label,
                confidence=analysis.confidence,
                metadata=analysis.metadata
            )
            return sentiment_aggregator.store_sentiment(db, sentiment_data)
        
        return {
            "content": text,
            "sentiment_score": analysis.compound,
            "sentiment_label": analysis.label,
            "confidence": analysis.confidence,
            "metadata": analysis.metadata,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trends")
async def get_sentiment_trends(
    symbols: str = Query(..., description="Comma-separated symbols"),
    timeframe: str = Query("1d", description="Time range"),
    db: Session = Depends(get_db)
):
    """Get sentiment trends for multiple symbols"""
    try:
        symbol_list = symbols.split(",")
        time_delta = {
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1m": timedelta(days=30)
        }.get(timeframe, timedelta(days=1))
        
        start_time = datetime.now() - time_delta
        
        trends = {}
        for symbol in symbol_list:
            trends[symbol] = await sentiment_aggregator.get_sentiment_trend(
                db, symbol, start_time
            )
        
        return {
            "trends": trends,
            "timeframe": timeframe,
            "generated_at": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting sentiment trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_service_stats(db: Session = Depends(get_db)):
    """Get service statistics and metrics"""
    try:
        return sentiment_aggregator.get_service_stats(db)
    except Exception as e:
        logger.error(f"Error getting service stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data Quality and Monitoring Endpoints

@app.get("/data-quality/check/{symbol}")
async def run_data_quality_check(
    symbol: str,
    db: Session = Depends(get_db)
):
    """
    Run a comprehensive data quality check for a specific symbol.
    
    This endpoint validates:
    - Data schema compliance and completeness
    - Population Stability Index (PSI) for distribution shifts
    - Statistical tests for data drift detection
    - Quality metrics and scoring
    - Actionable recommendations for issues found
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Running data quality check for {symbol}")
        
        report = await distribution_monitor.run_quality_check(symbol, db)
        
        return {
            'symbol': report.symbol,
            'validation_timestamp': report.validation_timestamp.isoformat(),
            'total_records': report.total_records,
            'overall_status': report.overall_status.value,
            'quality_score': report.quality_score,
            'validation_results': [
                {
                    'rule_name': r.rule_name,
                    'status': r.status.value,
                    'observed_value': r.observed_value,
                    'threshold': r.threshold,
                    'message': r.message,
                    'severity': r.severity.value
                }
                for r in report.validation_results
            ],
            'psi_analysis': [
                {
                    'feature_name': r.feature_name,
                    'psi_score': r.psi_score,
                    'interpretation': r.interpretation,
                    'bucket_count': len(r.bucket_details)
                }
                for r in report.psi_results
            ],
            'drift_detection': [
                {
                    'feature_name': r.feature_name,
                    'shift_detected': r.shift_detected,
                    'drift_type': r.drift_type.value,
                    'drift_score': r.drift_score,
                    'p_value': r.p_value,
                    'statistical_test': r.statistical_test,
                    'severity': r.severity.value
                }
                for r in report.drift_results
            ],
            'recommendations': report.recommendations,
            'data_characteristics': report.data_characteristics
        }
        
    except Exception as e:
        logger.error(f"Data quality check failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-quality/monitor/status")
async def get_monitoring_status():
    """
    Get the status of the continuous data quality monitoring system.
    
    Returns information about:
    - Running monitoring jobs and their schedules
    - Alert configurations and recent alerts
    - System health and performance metrics
    """
    try:
        status = distribution_monitor.get_monitoring_status()
        recent_alerts = distribution_monitor.get_recent_alerts(hours=24)
        
        return {
            'monitoring_system': status,
            'recent_alerts': recent_alerts,
            'system_health': {
                'monitoring_active': status['running'],
                'jobs_enabled': status['enabled_jobs'],
                'alerts_enabled': status['enabled_alerts'],
                'last_24h_alerts': len(recent_alerts)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data-quality/monitor/job")
async def add_monitoring_job(
    symbol: str = Query(..., description="Stock symbol to monitor"),
    interval: str = Query(default="daily", description="Monitoring interval: hourly, daily, weekly"),
    baseline_window_days: int = Query(default=30, description="Days of baseline data"),
    current_window_hours: int = Query(default=24, description="Hours of current data to validate")
):
    """
    Add a new continuous monitoring job for a symbol.
    
    The job will automatically run data quality checks at the specified interval,
    comparing current data against the baseline period to detect:
    - Distribution shifts and data drift
    - Quality degradation over time
    - Anomalies in sentiment patterns
    """
    try:
        symbol = symbol.upper()
        
        # Validate interval
        try:
            monitoring_interval = MonitoringInterval(interval.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid interval. Must be one of: {[e.value for e in MonitoringInterval]}"
            )
        
        job = MonitoringJob(
            job_id=f"monitor_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            interval=monitoring_interval,
            baseline_window_days=baseline_window_days,
            current_window_hours=current_window_hours,
            enabled=True
        )
        
        distribution_monitor.add_monitoring_job(job)
        
        return {
            'job_id': job.job_id,
            'symbol': job.symbol,
            'interval': job.interval.value,
            'baseline_window_days': job.baseline_window_days,
            'current_window_hours': job.current_window_hours,
            'status': 'created',
            'message': f'Monitoring job created for {symbol} with {interval} interval'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding monitoring job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/data-quality/monitor/job/{job_id}")
async def remove_monitoring_job(job_id: str):
    """Remove a monitoring job"""
    try:
        distribution_monitor.remove_monitoring_job(job_id)
        return {
            'job_id': job_id,
            'status': 'removed',
            'message': f'Monitoring job {job_id} has been removed'
        }
    except Exception as e:
        logger.error(f"Error removing monitoring job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data-quality/alerts/config")
async def add_alert_configuration(
    symbol: str = Query(..., description="Stock symbol"),
    alert_type: str = Query(..., description="Alert type: validation_failure, psi_drift, distribution_shift"),
    severity_threshold: str = Query(default="medium", description="Minimum severity: low, medium, high, critical"),
    notification_channels: str = Query(default="database", description="Comma-separated channels: database,email,webhook")
):
    """
    Configure alerts for data quality issues.
    
    Alert types:
    - validation_failure: Data fails schema validation rules
    - psi_drift: Population Stability Index indicates distribution shift
    - distribution_shift: Statistical tests detect significant data drift
    """
    try:
        symbol = symbol.upper()
        
        # Validate inputs
        valid_alert_types = ["validation_failure", "psi_drift", "distribution_shift"]
        if alert_type not in valid_alert_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid alert_type. Must be one of: {valid_alert_types}"
            )
        
        from .services.data_quality_validator import AlertSeverity
        try:
            severity = AlertSeverity(severity_threshold.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid severity. Must be one of: {[s.value for s in AlertSeverity]}"
            )
        
        channels = [ch.strip() for ch in notification_channels.split(",")]
        valid_channels = ["database", "email", "webhook"]
        invalid_channels = [ch for ch in channels if ch not in valid_channels]
        if invalid_channels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid channels: {invalid_channels}. Valid channels: {valid_channels}"
            )
        
        config = AlertConfig(
            alert_id=f"alert_{symbol}_{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            alert_type=alert_type,
            severity_threshold=severity,
            notification_channels=channels,
            enabled=True
        )
        
        distribution_monitor.add_alert_config(config)
        
        return {
            'alert_id': config.alert_id,
            'symbol': config.symbol,
            'alert_type': config.alert_type,
            'severity_threshold': config.severity_threshold.value,
            'notification_channels': config.notification_channels,
            'status': 'created',
            'message': f'Alert configuration created for {symbol}'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding alert configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-quality/alerts/recent")
async def get_recent_alerts(
    hours: int = Query(default=24, description="Hours to look back for alerts"),
    symbol: Optional[str] = Query(default=None, description="Filter by symbol")
):
    """Get recent data quality alerts"""
    try:
        alerts = distribution_monitor.get_recent_alerts(hours)
        
        if symbol:
            symbol = symbol.upper()
            alerts = [a for a in alerts if a['symbol'] == symbol]
        
        return {
            'alerts': alerts,
            'total_count': len(alerts),
            'timeframe_hours': hours,
            'symbol_filter': symbol,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data-quality/alerts/resolve/{event_id}")
async def resolve_alert(event_id: str):
    """Mark an alert as resolved"""
    try:
        success = await distribution_monitor.resolve_alert(event_id)
        
        if success:
            return {
                'event_id': event_id,
                'status': 'resolved',
                'message': f'Alert {event_id} marked as resolved'
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f'Alert {event_id} not found or already resolved'
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Transformer Training Endpoints

@app.post("/transformer/training/start")
async def start_transformer_training(
    symbols: str = Query(..., description="Comma-separated list of stock symbols"),
    training_approach: str = Query(default="multi_target", description="Training approach: multi_target, single_target, comparison"),
    model_architecture: str = Query(default="finbert", description="Model architecture: finbert, distilbert_financial, roberta_financial"),
    num_epochs: int = Query(default=3, description="Number of training epochs"),
    batch_size: int = Query(default=16, description="Training batch size"),
    learning_rate: float = Query(default=2e-5, description="Learning rate"),
    db: Session = Depends(get_db)
):
    """
    Start training a financial transformer model with multiple targets.
    
    This endpoint initiates training of FinBERT/DistilBERT models that can predict:
    - Traditional sentiment (positive/negative/neutral)
    - Price direction (up/down/flat) 
    - Volatility level (low/medium/high)
    - Price magnitude (regression)
    
    Multi-target training learns shared representations across all objectives,
    often outperforming single-target models through transfer learning.
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        # Validate inputs
        valid_approaches = ["multi_target", "single_target", "comparison"]
        if training_approach not in valid_approaches:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid training approach. Must be one of: {valid_approaches}"
            )
        
        try:
            model_arch = ModelArchitecture(model_architecture.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model architecture. Must be one of: {[e.value for e in ModelArchitecture]}"
            )
        
        # Create training job config
        job_config = transformer_training_service.create_default_job_config(
            symbols=symbol_list,
            approach=training_approach
        )
        
        # Override with user parameters
        job_config.model_architecture = model_arch
        job_config.training_config.num_epochs = num_epochs
        job_config.training_config.batch_size = batch_size
        job_config.training_config.learning_rate = learning_rate
        
        # Start training
        job_id = await transformer_training_service.start_training_job(job_config, db)
        
        return {
            'job_id': job_id,
            'status': 'started',
            'training_approach': training_approach,
            'model_architecture': model_architecture,
            'symbols': symbol_list,
            'training_config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            },
            'message': f'Training job started for {len(symbol_list)} symbols',
            'estimated_duration': '2-4 hours'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting transformer training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transformer/training/status/{job_id}")
async def get_training_status(job_id: str):
    """
    Get the status and progress of a transformer training job.
    
    Returns detailed information about:
    - Training progress and metrics
    - Dataset statistics and sample counts
    - Model evaluation results (when completed)
    - Error information (if failed)
    """
    try:
        status = transformer_training_service.get_job_status(job_id)
        
        if status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Training job {job_id} not found"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transformer/training/jobs")
async def list_training_jobs():
    """
    List all transformer training jobs with their current status.
    
    Returns a summary of all training jobs including:
    - Job ID and status
    - Start and end times
    - Dataset sample counts
    - Training approach used
    """
    try:
        jobs = transformer_training_service.list_jobs()
        return {
            'jobs': jobs,
            'total_jobs': len(jobs),
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/transformer/training/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a running training job"""
    try:
        success = transformer_training_service.cancel_job(job_id)
        
        if success:
            return {
                'job_id': job_id,
                'status': 'cancelled',
                'message': f'Training job {job_id} has been cancelled'
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f'Training job {job_id} not found or not running'
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling training job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transformer/training/recommendations")
async def get_training_recommendations(
    symbols: str = Query(..., description="Comma-separated list of stock symbols")
):
    """
    Get training recommendations based on available data and best practices.
    
    Analyzes the requested symbols and provides recommendations for:
    - Optimal training approach (multi-target vs single-target)
    - Recommended model architecture
    - Estimated training time and resource requirements
    - Data quality and sample size guidance
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        recommendations = transformer_training_service.get_training_recommendations(symbol_list)
        
        return {
            'symbols': symbol_list,
            'symbol_count': len(symbol_list),
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting training recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transformer/models/architectures")
async def list_available_architectures():
    """
    List available transformer model architectures for financial sentiment analysis.
    
    Returns information about supported models including:
    - FinBERT: Pre-trained on financial texts
    - DistilBERT: Lighter, faster variant
    - RoBERTa: Robust variant with improved training
    """
    try:
        architectures = []
        for arch in ModelArchitecture:
            arch_info = {
                'name': arch.value,
                'display_name': arch.name,
                'description': self._get_architecture_description(arch),
                'recommended_use': self._get_architecture_use_case(arch)
            }
            architectures.append(arch_info)
        
        return {
            'available_architectures': architectures,
            'default_architecture': ModelArchitecture.FINBERT.value,
            'total_count': len(architectures)
        }
        
    except Exception as e:
        logger.error(f"Error listing architectures: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transformer/targets")
async def list_financial_targets():
    """
    List all financial prediction targets supported by the multi-target training.
    
    Returns information about each target including:
    - Target type (classification vs regression)
    - Number of classes (for classification)
    - Expected value ranges
    - Business interpretation
    """
    try:
        targets = []
        for target in FinancialTarget:
            target_info = {
                'name': target.value,
                'display_name': target.name.replace('_', ' ').title(),
                'type': self._get_target_type(target),
                'description': self._get_target_description(target),
                'classes': self._get_target_classes(target)
            }
            targets.append(target_info)
        
        return {
            'financial_targets': targets,
            'total_targets': len(targets),
            'multi_target_benefits': [
                'Shared feature representations across objectives',
                'Better generalization through transfer learning',
                'Reduced training time vs separate models',
                'Improved performance on related tasks'
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing financial targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_architecture_description(self, arch: ModelArchitecture) -> str:
    """Get description for model architecture"""
    descriptions = {
        ModelArchitecture.FINBERT: "Pre-trained BERT model fine-tuned on financial texts",
        ModelArchitecture.DISTILBERT_FINANCIAL: "Lighter DistilBERT variant for faster inference",
        ModelArchitecture.ROBERTA_FINANCIAL: "RoBERTa model optimized for sentiment analysis"
    }
    return descriptions.get(arch, "Transformer model for financial analysis")

def _get_architecture_use_case(self, arch: ModelArchitecture) -> str:
    """Get recommended use case for architecture"""
    use_cases = {
        ModelArchitecture.FINBERT: "Best for financial domain accuracy",
        ModelArchitecture.DISTILBERT_FINANCIAL: "Best for production inference speed",
        ModelArchitecture.ROBERTA_FINANCIAL: "Best for general sentiment robustness"
    }
    return use_cases.get(arch, "General financial sentiment analysis")

def _get_target_type(self, target: FinancialTarget) -> str:
    """Get target type (classification/regression)"""
    if target == FinancialTarget.PRICE_MAGNITUDE:
        return "regression"
    return "classification"

def _get_target_description(self, target: FinancialTarget) -> str:
    """Get target description"""
    descriptions = {
        FinancialTarget.SENTIMENT: "Traditional sentiment classification (positive/negative/neutral)",
        FinancialTarget.PRICE_DIRECTION: "Next-day price movement direction (up/down/flat)",
        FinancialTarget.VOLATILITY: "Expected volatility level (low/medium/high)",
        FinancialTarget.PRICE_MAGNITUDE: "Magnitude of price change (continuous value)"
    }
    return descriptions.get(target, "Financial prediction target")

def _get_target_classes(self, target: FinancialTarget) -> Optional[List[str]]:
    """Get class labels for classification targets"""
    if target == FinancialTarget.SENTIMENT:
        return ["negative", "neutral", "positive"]
    elif target == FinancialTarget.PRICE_DIRECTION:
        return ["down", "flat", "up"]
    elif target == FinancialTarget.VOLATILITY:
        return ["low", "medium", "high"]
    elif target == FinancialTarget.PRICE_MAGNITUDE:
        return None  # Regression target
    return None

# Event-Window Sentiment Momentum Endpoints

@app.get("/momentum/events/detect")
async def detect_upcoming_events(
    symbols: str = Query(..., description="Comma-separated list of stock symbols"),
    days_ahead: int = Query(default=30, description="Days ahead to look for events")
):
    """
    Detect upcoming events for sentiment momentum analysis.
    
    This endpoint identifies scheduled events like:
    - Earnings announcements
    - FDA approvals (for biotech stocks)
    - Analyst days and guidance updates
    - Product launches and regulatory decisions
    
    Returns events within the specified time window for momentum monitoring.
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        events = await event_detection_service.detect_upcoming_events(
            symbols=symbol_list,
            days_ahead=days_ahead
        )
        
        # Format events for response
        formatted_events = []
        for event in events:
            formatted_events.append({
                'symbol': event.symbol,
                'event_type': event.event_type.value,
                'event_date': event.event_date.isoformat(),
                'event_title': event.event_title,
                'confirmed': event.confirmed,
                'confidence_score': event.confidence_score,
                'source': event.source,
                'days_until_event': (event.event_date - datetime.now()).days,
                'metadata': event.metadata
            })
        
        return {
            'symbols': symbol_list,
            'days_ahead': days_ahead,
            'total_events': len(formatted_events),
            'events': formatted_events,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error detecting events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/momentum/analyze/{symbol}")
async def analyze_event_momentum(
    symbol: str,
    event_type: str = Query(..., description="Event type: earnings, fda_approval, analyst_day, etc."),
    event_date: str = Query(..., description="Event date in ISO format (YYYY-MM-DD)"),
    pre_event_hours: int = Query(default=72, description="Hours before event to analyze"),
    db: Session = Depends(get_db)
):
    """
    Analyze sentiment momentum for a specific event window.
    
    This endpoint performs comprehensive momentum analysis including:
    - Short-term EMA calculations (4h, 12h, 24h)
    - Momentum acceleration metrics
    - Volume-weighted sentiment analysis
    - Positioning buildup detection
    - Pre-event prediction signals
    
    Focuses on detecting sentiment positioning before scheduled events.
    """
    try:
        symbol = symbol.upper()
        
        # Validate event type
        try:
            event_type_enum = EventType(event_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event type. Must be one of: {[e.value for e in EventType]}"
            )
        
        # Parse event date
        try:
            event_datetime = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid event date format. Use ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
            )
        
        # Create event schedule
        from .services.event_detection_service import EventSchedule
        event = EventSchedule(
            symbol=symbol,
            event_type=event_type_enum,
            event_date=event_datetime,
            event_title=f"{symbol} {event_type_enum.value}",
            confirmed=True,
            source="manual"
        )
        
        # Analyze momentum
        analysis = await event_detection_service.analyze_event_momentum(event, db)
        
        # Format momentum timeline
        momentum_timeline = []
        for metric in analysis.momentum_timeline:
            momentum_timeline.append({
                'timestamp': metric.timestamp.isoformat(),
                'sentiment_ema_4h': metric.sentiment_ema_4h,
                'sentiment_ema_12h': metric.sentiment_ema_12h,
                'sentiment_ema_24h': metric.sentiment_ema_24h,
                'ema_acceleration_4h': metric.ema_acceleration_4h,
                'ema_acceleration_12h': metric.ema_acceleration_12h,
                'volume_weighted_sentiment': metric.volume_weighted_sentiment,
                'bullish_momentum_score': metric.bullish_momentum_score,
                'bearish_momentum_score': metric.bearish_momentum_score,
                'momentum_direction': metric.momentum_direction.value,
                'momentum_strength': metric.momentum_strength,
                'signal_confidence': metric.signal_confidence
            })
        
        return {
            'symbol': symbol,
            'event_type': event_type,
            'event_date': event_date,
            'pre_event_hours': pre_event_hours,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': {
                'peak_bullish_momentum': analysis.peak_bullish_momentum,
                'peak_bearish_momentum': analysis.peak_bearish_momentum,
                'momentum_buildup_score': analysis.momentum_buildup_score,
                'direction_consistency': analysis.direction_consistency,
                'predicted_direction': analysis.predicted_direction.value,
                'signal_strength': analysis.signal_strength,
                'confidence_score': analysis.confidence_score
            },
            'momentum_timeline': momentum_timeline,
            'total_data_points': len(momentum_timeline)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing event momentum: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/momentum/monitor")
async def run_momentum_monitoring(
    hours_ahead: int = Query(default=72, description="Hours ahead to monitor events"),
    db: Session = Depends(get_db)
):
    """
    Run comprehensive momentum monitoring for all active events.
    
    This endpoint performs the complete monitoring cycle:
    - Identifies events within the monitoring window (default 72 hours)
    - Analyzes sentiment momentum for each event
    - Generates high-confidence trading signals
    - Validates predictions against actual outcomes
    
    Returns a comprehensive monitoring report with actionable signals.
    """
    try:
        logger.info("Running momentum monitoring cycle")
        
        # Run monitoring cycle
        monitoring_results = await event_detection_service.run_event_monitoring_cycle(db)
        
        return {
            'monitoring_cycle': monitoring_results,
            'system_status': {
                'monitoring_window_hours': hours_ahead,
                'high_confidence_threshold': 0.7,
                'signal_strength_threshold': 0.6
            }
        }
        
    except Exception as e:
        logger.error(f"Error running momentum monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/momentum/events/add")
async def add_manual_event(
    symbol: str = Query(..., description="Stock symbol"),
    event_type: str = Query(..., description="Event type"),
    event_date: str = Query(..., description="Event date in ISO format"),
    event_title: str = Query(..., description="Event title/description"),
    metadata: Optional[str] = Query(default=None, description="JSON metadata")
):
    """
    Add a manually identified event for momentum monitoring.
    
    Use this endpoint to add events that weren't automatically detected:
    - Custom corporate events
    - Regulatory announcements
    - Product launches
    - Conference presentations
    """
    try:
        symbol = symbol.upper()
        
        # Validate event type
        try:
            event_type_enum = EventType(event_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event type. Must be one of: {[e.value for e in EventType]}"
            )
        
        # Parse event date
        try:
            event_datetime = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid event date format. Use ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
            )
        
        # Parse metadata
        event_metadata = {}
        if metadata:
            try:
                import json
                event_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid metadata JSON format"
                )
        
        # Add event
        success = event_detection_service.add_manual_event(
            symbol=symbol,
            event_type=event_type,
            event_date=event_datetime,
            event_title=event_title,
            metadata=event_metadata
        )
        
        if success:
            return {
                'symbol': symbol,
                'event_type': event_type,
                'event_date': event_date,
                'event_title': event_title,
                'status': 'added',
                'message': f'Event added successfully for {symbol}'
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to add event"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding manual event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/momentum/validate/{symbol}")
async def validate_event_outcome(
    symbol: str,
    event_type: str = Query(..., description="Event type"),
    event_date: str = Query(..., description="Event date in ISO format"),
    db: Session = Depends(get_db)
):
    """
    Validate momentum predictions against actual market outcomes.
    
    This endpoint compares pre-event momentum analysis with actual:
    - Price movements (24h and 1 week post-event)
    - Volume spikes during the event
    - Volatility changes
    - Prediction accuracy metrics
    
    Use this to evaluate the effectiveness of momentum signals.
    """
    try:
        symbol = symbol.upper()
        
        # Validate inputs (similar to analyze_event_momentum)
        try:
            event_type_enum = EventType(event_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event type. Must be one of: {[e.value for e in EventType]}"
            )
        
        try:
            event_datetime = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid event date format"
            )
        
        # Check if event has passed
        if event_datetime > datetime.now():
            raise HTTPException(
                status_code=400,
                detail="Cannot validate future events. Event must have occurred."
            )
        
        # Create event and get analysis (from cache if available)
        from .services.event_detection_service import EventSchedule
        event = EventSchedule(
            symbol=symbol,
            event_type=event_type_enum,
            event_date=event_datetime,
            event_title=f"{symbol} {event_type_enum.value}",
            confirmed=True,
            source="validation"
        )
        
        # Get pre-event analysis
        analysis = await event_detection_service.analyze_event_momentum(event, db)
        
        # Validate against outcomes
        outcome = await event_detection_service.validate_event_predictions(event, analysis, db)
        
        return {
            'symbol': symbol,
            'event_type': event_type,
            'event_date': event_date,
            'validation_timestamp': datetime.now().isoformat(),
            'pre_event_prediction': {
                'predicted_direction': analysis.predicted_direction.value,
                'signal_strength': analysis.signal_strength,
                'confidence_score': analysis.confidence_score,
                'momentum_buildup_score': analysis.momentum_buildup_score
            },
            'actual_outcome': {
                'price_move_24h': outcome.price_move_24h,
                'price_move_1w': outcome.price_move_1w,
                'volume_spike': outcome.volume_spike,
                'volatility_spike': outcome.volatility_spike
            },
            'validation_results': {
                'momentum_prediction_accuracy': outcome.momentum_prediction_accuracy,
                'direction_prediction_accuracy': outcome.direction_prediction_accuracy,
                'signal_strength_correlation': outcome.signal_strength_correlation
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating event outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/momentum/events/status")
async def get_events_status():
    """
    Get status and statistics of tracked events and momentum monitoring.
    
    Returns comprehensive information about:
    - Total tracked events by type and timeframe
    - Active monitoring status
    - Cached analyses
    - System performance metrics
    """
    try:
        # Get event statistics
        event_stats = event_detection_service.get_event_statistics()
        
        # Get active monitoring events
        active_events = await event_detection_service.get_active_monitoring_events(hours_ahead=72)
        
        return {
            'event_statistics': event_stats,
            'active_monitoring': {
                'events_in_72h_window': len(active_events),
                'events_by_type': {
                    event_type.value: len([e for e in active_events if e.event_type == event_type])
                    for event_type in EventType
                },
                'next_events': [
                    {
                        'symbol': event.symbol,
                        'event_type': event.event_type.value,
                        'event_date': event.event_date.isoformat(),
                        'hours_until': int((event.event_date - datetime.now()).total_seconds() / 3600),
                        'confirmed': event.confirmed
                    }
                    for event in active_events[:10]  # Next 10 events
                ]
            },
            'monitoring_capabilities': {
                'supported_event_types': [event_type.value for event_type in EventType],
                'momentum_directions': [direction.value for direction in MomentumDirection],
                'default_pre_event_window_hours': 72,
                'ema_timeframes': ['4h', '12h', '24h'],
                'acceleration_metrics': ['4h_acceleration', '12h_acceleration']
            },
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting events status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")