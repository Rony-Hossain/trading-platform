from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from .config import settings
from .database import get_db_session, init_db
from .cache import get_redis, init_redis
from .models import (
    TechnicalAnalysisResponse, 
    ChartPatternsResponse, 
    AdvancedIndicatorsResponse,
    ForecastResponse,
    ComprehensiveAnalysisResponse,
    BatchAnalysisRequest
)
from .services.technical_analysis_service import TechnicalAnalysisService
from .services.pattern_recognition_service import PatternRecognitionService
from .services.forecast_service import ForecastService
from .middleware import setup_middleware

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Analysis API")
    await init_redis()
    await init_db()
    logger.info("Analysis API started successfully")
    yield
    # Shutdown
    logger.info("Shutting down Analysis API")

app = FastAPI(
    title="Analysis API", 
    description="Technical analysis, advanced indicators, patterns, and ML forecasts",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Check database connection
        async with get_db_session() as db:
            await db.execute("SELECT 1")
        
        # Check Redis connection
        redis = await get_redis()
        await redis.ping()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "analysis-api"
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/analyze/{symbol}", response_model=TechnicalAnalysisResponse)
async def get_technical_analysis(
    symbol: str,
    period: str = Query(default="6mo", description="Time period for analysis"),
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Get comprehensive technical analysis"""
    try:
        symbol = symbol.upper()
        logger.info("Getting technical analysis", symbol=symbol, period=period)
        
        analysis_service = TechnicalAnalysisService(db, redis)
        result = await analysis_service.get_technical_analysis(symbol, period)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Analysis for {symbol} not available")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting technical analysis", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analyze/{symbol}/patterns", response_model=ChartPatternsResponse)
async def get_chart_patterns(
    symbol: str,
    period: str = Query(default="3mo", description="Time period for pattern recognition"),
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Get chart patterns"""
    try:
        symbol = symbol.upper()
        logger.info("Getting chart patterns", symbol=symbol, period=period)
        
        pattern_service = PatternRecognitionService(db, redis)
        result = await pattern_service.get_chart_patterns(symbol, period)
        
        return result
    except Exception as e:
        logger.error("Error getting chart patterns", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analyze/{symbol}/advanced", response_model=AdvancedIndicatorsResponse)
async def get_advanced_indicators(
    symbol: str,
    period: str = Query(default="6mo", description="Time period for indicators"),
    indicators: Optional[str] = Query(default=None, description="Comma-separated list of indicators"),
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Get advanced indicators"""
    try:
        symbol = symbol.upper()
        indicator_list = indicators.split(",") if indicators else None
        logger.info("Getting advanced indicators", symbol=symbol, indicators=indicator_list)
        
        analysis_service = TechnicalAnalysisService(db, redis)
        result = await analysis_service.get_advanced_indicators(symbol, period, indicator_list)
        
        return result
    except Exception as e:
        logger.error("Error getting advanced indicators", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/forecast/{symbol}", response_model=ForecastResponse)
async def get_forecast(
    symbol: str,
    model_type: str = Query(default="ensemble", description="Model type: ensemble, lstm, xgboost"),
    horizon: int = Query(default=5, description="Forecast horizon in days"),
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Get ML forecast"""
    try:
        symbol = symbol.upper()
        logger.info("Getting forecast", symbol=symbol, model_type=model_type, horizon=horizon)
        
        forecast_service = ForecastService(db, redis)
        result = await forecast_service.get_forecast(symbol, model_type, horizon)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Forecast for {symbol} not available")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting forecast", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analyze/{symbol}/comprehensive", response_model=ComprehensiveAnalysisResponse)
async def get_comprehensive_analysis(
    symbol: str,
    period: str = Query(default="6mo", description="Time period for analysis"),
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Get comprehensive analysis payload"""
    try:
        symbol = symbol.upper()
        logger.info("Getting comprehensive analysis", symbol=symbol, period=period)
        
        # Initialize services
        analysis_service = TechnicalAnalysisService(db, redis)
        pattern_service = PatternRecognitionService(db, redis)
        forecast_service = ForecastService(db, redis)
        
        # Run all analyses concurrently
        import asyncio
        technical_task = analysis_service.get_technical_analysis(symbol, period)
        patterns_task = pattern_service.get_chart_patterns(symbol, period)
        forecast_task = forecast_service.get_forecast(symbol, "ensemble", 5)
        advanced_task = analysis_service.get_advanced_indicators(symbol, period, None)
        
        technical, patterns, forecast, advanced = await asyncio.gather(
            technical_task,
            patterns_task, 
            forecast_task,
            advanced_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(technical, Exception):
            technical = None
        if isinstance(patterns, Exception):
            patterns = None
        if isinstance(forecast, Exception):
            forecast = None
        if isinstance(advanced, Exception):
            advanced = None
        
        # Generate overall assessment
        overall_assessment = await _generate_overall_assessment(
            symbol, technical, patterns, forecast, advanced
        )
        
        return ComprehensiveAnalysisResponse(
            symbol=symbol,
            technical=technical,
            patterns=patterns,
            forecast=forecast,
            advanced_indicators=advanced,
            overall_assessment=overall_assessment
        )
    except Exception as e:
        logger.error("Error getting comprehensive analysis", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze/batch")
async def get_batch_analysis(
    request: BatchAnalysisRequest,
    db=Depends(get_db_session),
    redis=Depends(get_redis)
):
    """Batch analysis for multiple symbols"""
    try:
        if len(request.symbols) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed")
        
        logger.info("Getting batch analysis", symbols=request.symbols, analysis_type=request.analysis_type)
        
        results = {}
        
        # Process symbols concurrently
        import asyncio
        
        async def analyze_symbol(symbol: str):
            try:
                if request.analysis_type == "comprehensive":
                    return await get_comprehensive_analysis(symbol, "6mo", db, redis)
                elif request.analysis_type == "technical":
                    analysis_service = TechnicalAnalysisService(db, redis)
                    return await analysis_service.get_technical_analysis(symbol, "6mo")
                elif request.analysis_type == "forecast":
                    forecast_service = ForecastService(db, redis)
                    return await forecast_service.get_forecast(symbol, "ensemble", 5)
                else:
                    return None
            except Exception as e:
                logger.error("Batch analysis failed for symbol", symbol=symbol, error=str(e))
                return None
        
        # Run analyses concurrently
        tasks = [analyze_symbol(symbol.upper()) for symbol in request.symbols]
        analyses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        for symbol, analysis in zip(request.symbols, analyses):
            if not isinstance(analysis, Exception) and analysis:
                results[symbol.upper()] = analysis
        
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in batch analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

async def _generate_overall_assessment(
    symbol: str,
    technical: Optional[TechnicalAnalysisResponse],
    patterns: Optional[ChartPatternsResponse], 
    forecast: Optional[ForecastResponse],
    advanced: Optional[AdvancedIndicatorsResponse]
) -> Dict[str, Any]:
    """Generate overall investment assessment"""
    try:
        signals = []
        bullish_signals = 0
        bearish_signals = 0
        
        # Analyze technical signals
        if technical and technical.signals:
            overall_signal = technical.signals.overall_signal
            if overall_signal in ["BUY", "STRONG_BUY"]:
                bullish_signals += 2 if overall_signal == "STRONG_BUY" else 1
                signals.append(f"Technical analysis shows {overall_signal} signal")
            elif overall_signal in ["SELL", "STRONG_SELL"]:
                bearish_signals += 2 if overall_signal == "STRONG_SELL" else 1
                signals.append(f"Technical analysis shows {overall_signal} signal")
        
        # Analyze patterns
        if patterns and patterns.patterns:
            bullish_patterns = [p for p in patterns.patterns if p.direction == "bullish"]
            bearish_patterns = [p for p in patterns.patterns if p.direction == "bearish"]
            
            if bullish_patterns:
                bullish_signals += len(bullish_patterns)
                signals.append(f"Found {len(bullish_patterns)} bullish pattern(s)")
            if bearish_patterns:
                bearish_signals += len(bearish_patterns)
                signals.append(f"Found {len(bearish_patterns)} bearish pattern(s)")
        
        # Analyze forecast
        if forecast and forecast.trend_forecast:
            direction = forecast.trend_forecast.direction
            confidence = forecast.trend_forecast.confidence
            
            if direction == "UP" and confidence in ["MEDIUM", "HIGH"]:
                bullish_signals += 2 if confidence == "HIGH" else 1
                signals.append(f"ML forecast predicts upward trend with {confidence.lower()} confidence")
            elif direction == "DOWN" and confidence in ["MEDIUM", "HIGH"]:
                bearish_signals += 2 if confidence == "HIGH" else 1
                signals.append(f"ML forecast predicts downward trend with {confidence.lower()} confidence")
        
        # Determine overall recommendation
        net_signal = bullish_signals - bearish_signals
        
        if net_signal >= 3:
            recommendation = "STRONG_BUY"
            score = min(1.0, net_signal / 5)
        elif net_signal >= 1:
            recommendation = "BUY"
            score = min(0.6, net_signal / 3)
        elif net_signal <= -3:
            recommendation = "STRONG_SELL"
            score = max(-1.0, net_signal / 5)
        elif net_signal <= -1:
            recommendation = "SELL"
            score = max(-0.6, net_signal / 3)
        else:
            recommendation = "HOLD"
            score = 0.0
        
        # Generate key factors and risks
        key_factors = signals[:3]  # Top 3 signals
        
        risks = []
        opportunities = []
        
        if technical:
            rsi = technical.oscillators.rsi_14 if technical.oscillators else None
            if rsi:
                if rsi > 70:
                    risks.append("RSI indicates overbought conditions")
                elif rsi < 30:
                    opportunities.append("RSI indicates oversold conditions")
        
        if forecast and forecast.price_analysis:
            max_loss = forecast.price_analysis.max_expected_loss
            max_gain = forecast.price_analysis.max_expected_gain
            
            if max_loss and abs(max_loss) > 10:
                risks.append(f"Potential downside risk of {abs(max_loss):.1f}%")
            if max_gain and max_gain > 10:
                opportunities.append(f"Potential upside of {max_gain:.1f}%")
        
        return {
            "recommendation": recommendation,
            "score": score,
            "key_factors": key_factors,
            "risks": risks[:3],  # Top 3 risks
            "opportunities": opportunities[:3]  # Top 3 opportunities
        }
        
    except Exception as e:
        logger.error("Error generating overall assessment", symbol=symbol, error=str(e))
        return {
            "recommendation": "HOLD",
            "score": 0.0,
            "key_factors": ["Analysis unavailable"],
            "risks": ["Insufficient data for assessment"],
            "opportunities": []
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )