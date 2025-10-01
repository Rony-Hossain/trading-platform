"""
Analysis Service API
Provides comprehensive technical and statistical analysis for trading
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager

from .services.analysis_service import ComprehensiveAnalysisService
from .services.forecasting_service import ForecastingService
from .services.multi_factor_analysis import MultiFactorAnalysis
from typing import Optional
import os
from .services.composite_algo import CompositeAlgo
from .api.event_analysis import router as event_analysis_router
from .api.validation import router as validation_router
from .api.volatility_thresholds import router as volatility_thresholds_router
from .api.catalyst_triggers import router as catalyst_triggers_router
from .api.gap_trading import router as gap_trading_router
from .api.market_regime import router as market_regime_router
from .api.event_backtest import router as event_backtest_router
from .api.execution_modeling import router as execution_modeling_router
from .api.error_attribution import router as error_attribution_router
from .api.temporal_features import router as temporal_features_router
from .api.labeling import router as labeling_router
from .api.significance import router as significance_router
from .api.portfolio import router as portfolio_router
from .api.execution import router as execution_router
from .api.monitoring import router as monitoring_router
from .api.compliance import router as compliance_router
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
# Allow overriding market-data base URL when running in containers
MARKET_DATA_URL = os.getenv("MARKET_DATA_API", os.getenv("MARKET_DATA_URL", "http://market-data-api:8002"))
SENTIMENT_API_URL = os.getenv("SENTIMENT_API", os.getenv("SENTIMENT_API_URL", "http://sentiment-service:8005"))
FUNDAMENTALS_API_URL = os.getenv("FUNDAMENTALS_API", os.getenv("FUNDAMENTALS_API_URL", "http://fundamentals-service:8006"))
EVENT_DATA_URL = os.getenv("EVENT_DATA_API", os.getenv("EVENT_DATA_URL", "http://event-data-service:8007"))
analysis_service = ComprehensiveAnalysisService(market_data_url=MARKET_DATA_URL)
forecasting_service = ForecastingService(market_data_url=MARKET_DATA_URL)
composite_algo = CompositeAlgo(MARKET_DATA_URL, SENTIMENT_API_URL, FUNDAMENTALS_API_URL)
multi_factor_analysis = MultiFactorAnalysis(MARKET_DATA_URL, SENTIMENT_API_URL, FUNDAMENTALS_API_URL, EVENT_DATA_URL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Analysis Service started")
    yield
    # Shutdown
    logger.info("Analysis Service stopped")

app = FastAPI(
    title="Analysis Service",
    description="Comprehensive technical and statistical analysis for trading decisions",
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

# Include routers
app.include_router(event_analysis_router)
app.include_router(validation_router)
app.include_router(volatility_thresholds_router)
app.include_router(catalyst_triggers_router)
app.include_router(gap_trading_router)
app.include_router(market_regime_router)
app.include_router(event_backtest_router)
app.include_router(execution_modeling_router)
app.include_router(error_attribution_router)
app.include_router(temporal_features_router)
app.include_router(labeling_router, prefix="/labeling")
app.include_router(significance_router, prefix="/significance")
app.include_router(portfolio_router, prefix="/portfolio")
app.include_router(execution_router, prefix="/execution")
app.include_router(monitoring_router, prefix="/monitoring")
app.include_router(compliance_router, prefix="/compliance")

# Pydantic models
class AnalysisRequest(BaseModel):
    symbol: str
    period: Optional[str] = "1y"

class BatchAnalysisRequest(BaseModel):
    symbols: List[str]
    period: Optional[str] = "1y"

@app.get("/")
async def root():
    return {
        "service": "Analysis Service",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "endpoints": {
            "comprehensive_analysis": "/analyze/{symbol}",
            "quick_analysis": "/analyze/{symbol}/quick",
            "batch_analysis": "/analyze/batch",
            "technical_only": "/analyze/{symbol}/technical",
            "composite_score": "/score/{symbol}",
            "ml_forecast": "/forecast/{symbol}",
            "batch_forecast": "/forecast/batch",
            "model_evaluation": "/forecast/{symbol}/evaluate",
            "health": "/health"
        },
        "supported_periods": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
        "analysis_features": [
            "Technical Indicators (RSI, MACD, Bollinger Bands, Moving Averages)",
            "Statistical Analysis (Volatility, Risk Metrics, Correlation)",
            "Market Regime Detection",
            "Trading Signal Generation",
            "Feature Engineering",
            "Performance Analytics"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "analysis-service",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "market_data_service": MARKET_DATA_URL,
            "sentiment_service": SENTIMENT_API_URL,
            "fundamentals_service": FUNDAMENTALS_API_URL
        }
    }

@app.get("/analyze/{symbol}")
async def comprehensive_analysis(
    symbol: str,
    period: str = Query(default="1y", description="Analysis period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)")
):
    """
    Comprehensive analysis including technical indicators, statistical analysis,
    and market regime detection
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Starting comprehensive analysis for {symbol} with period {period}")
        
        result = await analysis_service.comprehensive_analysis(symbol, period)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{symbol}/quick")
async def quick_analysis(symbol: str):
    """
    Quick analysis with real-time data and basic technical indicators
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Starting quick analysis for {symbol}")
        
        result = await analysis_service.quick_analysis(symbol)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in quick analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{symbol}/technical")
async def technical_analysis_only(
    symbol: str,
    period: str = Query(default="6mo", description="Analysis period")
):
    """
    Technical analysis only (faster than comprehensive analysis)
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Starting technical analysis for {symbol}")
        
        # Get data and run only technical analysis
        df = await analysis_service.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=False)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        technical_result = analysis_service.technical_analysis.analyze(df, symbol)
        
        return {
            "symbol": symbol,
            "analysis_timestamp": datetime.now().isoformat(),
            "data_period": period,
            "data_points": len(df),
            "analysis_type": "technical_only",
            "technical_analysis": technical_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in technical analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def batch_analysis(request: BatchAnalysisRequest):
    """
    Batch analysis for multiple symbols (quick analysis only for performance)
    """
    try:
        if len(request.symbols) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed for batch analysis")
        
        logger.info(f"Starting batch analysis for {len(request.symbols)} symbols")
        
        results = {}
        errors = {}
        
        # Process symbols concurrently
        import asyncio
        
        async def analyze_symbol(symbol: str):
            try:
                symbol = symbol.upper()
                return await analysis_service.quick_analysis(symbol)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                return {'error': str(e)}
        
        # Run all analyses concurrently
        tasks = [analyze_symbol(symbol) for symbol in request.symbols]
        symbol_results = await asyncio.gather(*tasks)
        
        # Organize results
        for i, symbol in enumerate(request.symbols):
            symbol = symbol.upper()
            result = symbol_results[i]
            
            if 'error' in result:
                errors[symbol] = result['error']
            else:
                results[symbol] = result
        
        return {
            "batch_analysis_timestamp": datetime.now().isoformat(),
            "requested_symbols": [s.upper() for s in request.symbols],
            "successful_analyses": len(results),
            "failed_analyses": len(errors),
            "results": results,
            "errors": errors if errors else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{symbol}/indicators")
async def get_indicators(
    symbol: str,
    period: str = Query(default="3mo", description="Analysis period"),
    indicators: List[str] = Query(default=["sma", "ema", "rsi", "macd"], description="Indicators to calculate")
):
    """
    Get specific technical indicators for a symbol
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Getting indicators {indicators} for {symbol}")
        
        # Get data
        df = await analysis_service.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=False)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Calculate requested indicators
        from .core.indicators import TechnicalIndicators
        tech_indicators = TechnicalIndicators()
        
        results = {
            "symbol": symbol,
            "period": period,
            "data_points": len(df),
            "timestamp": datetime.now().isoformat(),
            "indicators": {}
        }
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if "sma" in indicators:
            results["indicators"]["sma"] = {
                "sma_20": tech_indicators.sma(close, 20).dropna().tail(10).to_dict(),
                "sma_50": tech_indicators.sma(close, 50).dropna().tail(10).to_dict()
            }
        
        if "ema" in indicators:
            results["indicators"]["ema"] = {
                "ema_12": tech_indicators.ema(close, 12).dropna().tail(10).to_dict(),
                "ema_26": tech_indicators.ema(close, 26).dropna().tail(10).to_dict()
            }
        
        if "rsi" in indicators:
            rsi_values = tech_indicators.rsi(close, 14).dropna().tail(10)
            results["indicators"]["rsi"] = {
                "values": rsi_values.to_dict(),
                "current": float(rsi_values.iloc[-1]) if not rsi_values.empty else None,
                "signal": "OVERBOUGHT" if rsi_values.iloc[-1] > 70 else "OVERSOLD" if rsi_values.iloc[-1] < 30 else "NEUTRAL"
            }
        
        if "macd" in indicators:
            macd_data = tech_indicators.macd(close)
            results["indicators"]["macd"] = {
                "macd": macd_data['macd'].dropna().tail(10).to_dict(),
                "signal": macd_data['signal'].dropna().tail(10).to_dict(),
                "histogram": macd_data['histogram'].dropna().tail(10).to_dict()
            }
        
        if "bollinger" in indicators:
            bb_data = tech_indicators.bollinger_bands(close)
            results["indicators"]["bollinger_bands"] = {
                "upper": bb_data['upper'].dropna().tail(10).to_dict(),
                "middle": bb_data['middle'].dropna().tail(10).to_dict(),
                "lower": bb_data['lower'].dropna().tail(10).to_dict(),
                "width": bb_data['width'].dropna().tail(10).to_dict()
            }
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{symbol}/performance")
async def get_performance_metrics(
    symbol: str,
    period: str = Query(default="1y", description="Analysis period")
):
    """
    Get detailed performance and risk metrics
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Getting performance metrics for {symbol}")
        
        # Get data
        df = await analysis_service.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=False)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Calculate performance metrics
        returns = df['close'].pct_change().dropna()
        
        volatility_metrics = analysis_service.statistical_analysis.calculate_volatility_metrics(returns)
        risk_metrics = analysis_service.statistical_analysis.calculate_risk_metrics(returns)
        trend_metrics = analysis_service.statistical_analysis.trend_analysis(df['close'])
        
        return {
            "symbol": symbol,
            "period": period,
            "data_points": len(df),
            "timestamp": datetime.now().isoformat(),
            "price_performance": {
                "total_return": float((df['close'].iloc[-1] / df['close'].iloc[0]) - 1),
                "current_price": float(df['close'].iloc[-1]),
                "period_high": float(df['high'].max()),
                "period_low": float(df['low'].min()),
                "average_volume": float(df['volume'].mean())
            },
            "volatility_metrics": volatility_metrics,
            "risk_metrics": risk_metrics,
            "trend_metrics": trend_metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/{symbol}")
async def ml_forecast(
    symbol: str,
    model_type: str = Query(default="ensemble", description="Model type: ensemble, lstm, random_forest, arima"),
    horizon: int = Query(default=5, description="Prediction horizon in days"),
    period: str = Query(default="2y", description="Training data period"),
    retrain: bool = Query(default=False, description="Force model retraining")
):
    """
    Generate ML-based price forecast for a symbol
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Generating {model_type} forecast for {symbol}")
        
        # Try ML forecast first
        try:
            result = await forecasting_service.generate_forecast(
                symbol=symbol,
                model_type=model_type,
                prediction_horizon=horizon,
                period=period,
                retrain=retrain
            )
            
            if 'error' not in result:
                return result
        except Exception as e:
            logger.warning(f"ML forecast failed for {symbol}: {e}")
        
        # Fallback to mock forecast with realistic data
        logger.info(f"Using mock forecast for {symbol}")
        return await _generate_mock_forecast(symbol, horizon)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ML forecast for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_mock_forecast(symbol: str, horizon: int = 5) -> Dict[str, Any]:
    """Generate realistic mock forecast data"""
    import random
    from datetime import datetime, timedelta
    
    # Get current price
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{MARKET_DATA_URL}/stocks/{symbol}/price") as resp:
                price_data = await resp.json()
                current_price = price_data.get('price', 100.0)
    except:
        current_price = 100.0  # Default fallback
    
    # Generate realistic predictions with slight trend and volatility
    predictions = []
    base_return = random.uniform(-0.02, 0.02)  # Base daily return -2% to +2%
    volatility = random.uniform(0.01, 0.03)    # Daily volatility 1% to 3%
    
    price = current_price
    for i in range(horizon):
        # Random walk with slight momentum
        daily_return = base_return + random.normalvariate(0, volatility)
        price = price * (1 + daily_return)
        predictions.append(round(price, 2))
    
    # Generate confidence intervals
    prediction_intervals = []
    for pred in predictions:
        lower = round(pred * (1 - volatility * 2), 2)
        upper = round(pred * (1 + volatility * 2), 2)
        prediction_intervals.append({
            "lower": lower,
            "upper": upper,
            "confidence": 0.68  # ~1 std dev
        })
    
    # Determine trend
    if predictions[-1] > current_price * 1.01:
        trend = "BULLISH"
        trend_confidence = "MEDIUM"
    elif predictions[-1] < current_price * 0.99:
        trend = "BEARISH"
        trend_confidence = "MEDIUM"
    else:
        trend = "SIDEWAYS"
        trend_confidence = "LOW"
    
    return {
        "symbol": symbol,
        "model_type": "Mock Ensemble",
        "forecast_generated": datetime.now().isoformat(),
        "prediction_horizon": horizon,
        "current_price": current_price,
        "predictions": predictions,
        "prediction_intervals": prediction_intervals,
        "forecast_dates": [
            (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d")
            for i in range(horizon)
        ],
        "trend_forecast": {
            "direction": trend,
            "confidence": trend_confidence
        },
        "price_analysis": {
            "expected_return_1d": round((predictions[0] - current_price) / current_price * 100, 2),
            "expected_return_5d": round((predictions[-1] - current_price) / current_price * 100, 2),
            "max_expected_gain": round(max((p - current_price) / current_price * 100 for p in predictions), 2),
            "max_expected_loss": round(min((p - current_price) / current_price * 100 for p in predictions), 2),
            "price_volatility": round(volatility * 100, 2)
        },
        "risk_analysis": {
            "max_downside_risk": round(max(0, (current_price - min(p['lower'] for p in prediction_intervals)) / current_price * 100), 2),
            "max_upside_potential": round(max(0, (max(p['upper'] for p in prediction_intervals) - current_price) / current_price * 100), 2)
        },
        "market_context": {
            "forecast_reliability": "MEDIUM",
            "data_quality_score": 0.85
        },
        "forecast_metadata": {
            "generated_at": datetime.now().isoformat(),
            "symbol": symbol,
            "model_version": "Mock 1.0",
            "note": "This is a mock forecast for demonstration purposes"
        }
    }

@app.post("/forecast/batch")
async def batch_forecast(request: BatchAnalysisRequest):
    """
    Generate ML forecasts for multiple symbols
    """
    try:
        if len(request.symbols) > 5:  # Limit for ML forecasting
            raise HTTPException(status_code=400, detail="Maximum 5 symbols allowed for batch forecasting")
        
        logger.info(f"Starting batch forecasting for {len(request.symbols)} symbols")
        
        result = await forecasting_service.batch_forecast(
            symbols=request.symbols,
            model_type="ensemble",  # Use ensemble for batch
            prediction_horizon=5
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch forecasting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/models/status")
async def model_cache_status():
    """
    Get current model cache status and statistics
    """
    try:
        return forecasting_service.get_cache_status()
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ensemble Stacking Endpoints

@app.get("/forecast/stacking/{symbol}")
async def generate_stacking_forecast(
    symbol: str,
    prediction_horizon: int = Query(default=5, description="Number of days to forecast"),
    period: str = Query(default="2y", description="Training data period"),
    retrain: bool = Query(default=False, description="Force model retraining"),
    blender_type: str = Query(default="ridge", description="Blending model type: ridge, linear, lasso, elastic_net, random_forest")
):
    """
    Generate forecast using ensemble stacking with formal blending.
    
    This endpoint uses advanced ensemble stacking with:
    - LSTM/GRU + RandomForest base models
    - Out-of-fold prediction methodology to prevent data leakage
    - Formal blending with Ridge/Linear/Lasso regression
    - Time-series cross-validation
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Generating stacking forecast for {symbol}")
        
        result = await forecasting_service.generate_stacking_forecast(
            symbol=symbol,
            prediction_horizon=prediction_horizon,
            period=period,
            retrain=retrain,
            blender_type=blender_type
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating stacking forecast for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/stacking/compare/{symbol}")
async def compare_ensemble_models(
    symbol: str,
    prediction_horizon: int = Query(default=5, description="Number of days to forecast"),
    period: str = Query(default="2y", description="Training data period")
):
    """
    Compare traditional ensemble vs stacking ensemble performance.
    
    Provides side-by-side comparison of:
    - Traditional ensemble (simple averaging)
    - Stacking ensemble (formal blending)
    
    Includes performance metrics and model recommendations.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Comparing ensemble models for {symbol}")
        
        result = await forecasting_service.compare_ensemble_models(
            symbol=symbol,
            prediction_horizon=prediction_horizon,
            period=period
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing ensemble models for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/stacking/status")
async def stacking_cache_status():
    """
    Get ensemble stacking model cache status and configuration.
    """
    try:
        return forecasting_service.get_stacking_cache_status()
    except Exception as e:
        logger.error(f"Error getting stacking cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast/stacking/cache/clear")
async def clear_stacking_cache(
    symbol: str = Query(default=None, description="Symbol to clear (leave empty to clear all)")
):
    """
    Clear ensemble stacking model cache.
    """
    try:
        forecasting_service.clear_stacking_cache(symbol)
        
        if symbol:
            return {"message": f"Cleared stacking cache for {symbol}"}
        else:
            return {"message": "Cleared all stacking models from cache"}
            
    except Exception as e:
        logger.error(f"Error clearing stacking cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Regime Analysis Endpoints

@app.get("/regime/analysis/{symbol}")
async def analyze_market_regime(
    symbol: str,
    period: str = Query(default="2y", description="Historical data period for regime analysis")
):
    """
    Analyze current market regime for a symbol.
    
    Provides comprehensive regime analysis including:
    - ATR bands and position
    - Realized volatility term structure
    - HMM state detection
    - Regime transitions
    - Recommended model weights
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Analyzing market regime for {symbol}")
        
        result = await forecasting_service.analyze_market_regime(
            symbol=symbol,
            period=period
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing market regime for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/regime-aware/{symbol}")
async def generate_regime_aware_forecast(
    symbol: str,
    prediction_horizon: int = Query(default=5, description="Number of days to forecast"),
    period: str = Query(default="2y", description="Training data period"),
    retrain: bool = Query(default=False, description="Force model retraining"),
    weighting_mode: str = Query(default="hybrid", description="Regime weighting mode: fixed, adaptive, hybrid, confidence_weighted")
):
    """
    Generate forecast using regime-aware ensemble stacking.
    
    This endpoint uses advanced regime-aware ensemble stacking with:
    - Dynamic model weighting based on current market regime
    - ATR bands and volatility term structure analysis
    - HMM-based regime detection
    - Adaptive performance tracking across regimes
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Generating regime-aware forecast for {symbol}")
        
        result = await forecasting_service.generate_regime_aware_forecast(
            symbol=symbol,
            prediction_horizon=prediction_horizon,
            period=period,
            retrain=retrain,
            weighting_mode=weighting_mode
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating regime-aware forecast for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regime/status")
async def regime_aware_cache_status():
    """
    Get regime-aware model cache status and configuration.
    """
    try:
        return forecasting_service.get_regime_aware_cache_status()
    except Exception as e:
        logger.error(f"Error getting regime-aware cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/regime/cache/clear")
async def clear_regime_aware_cache(
    symbol: str = Query(default=None, description="Symbol to clear (leave empty to clear all)")
):
    """
    Clear regime-aware model cache.
    """
    try:
        forecasting_service.clear_regime_aware_cache(symbol)
        
        if symbol:
            return {"message": f"Cleared regime-aware cache for {symbol}"}
        else:
            return {"message": "Cleared all regime-aware models from cache"}
            
    except Exception as e:
        logger.error(f"Error clearing regime-aware cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regime/validate/{symbol}")
async def validate_regime_detection(
    symbol: str,
    period: str = Query(default="2y", description="Historical data period for validation"),
    validation_period: int = Query(default=60, description="Number of days per validation window")
):
    """
    Validate regime detection accuracy across different market conditions.
    
    This endpoint tests the regime detection system by:
    - Splitting historical data into validation windows
    - Analyzing regime stability and confidence
    - Validating against actual market characteristics
    - Providing performance metrics across different conditions
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Validating regime detection for {symbol}")
        
        result = await forecasting_service.validate_regime_detection(
            symbol=symbol,
            period=period,
            validation_period=validation_period
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating regime detection for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/evaluate/{symbol}")
async def evaluate_advanced_models(
    symbol: str,
    period: str = Query(default="2y", description="Training data period"),
    cv_folds: int = Query(default=5, description="Cross-validation folds"),
    include_financial_metrics: bool = Query(default=True, description="Include financial performance metrics"),
    cv_method: str = Query(default="walk_forward", description="CV method: walk_forward, expanding_window, rolling_window, regime_aware"),
    include_significance_testing: bool = Query(default=False, description="Include DSR/PBO significance testing")
):
    """
    Enhanced model evaluation comparing RandomForest, LightGBM, and XGBoost
    with proper time-series cross-validation and financial metrics.
    
    This endpoint implements the enhanced Phase 1 requirement for model sophistication.
    
    Set include_significance_testing=true to add DSR/PBO analysis to the evaluation.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Starting advanced model evaluation for {symbol}")
        
        result = await forecasting_service.evaluate_advanced_models(
            symbol=symbol,
            period=period,
            cv_folds=cv_folds,
            include_financial_metrics=include_financial_metrics,
            cv_method=cv_method
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result)
        
        # Optionally add significance testing results
        if include_significance_testing:
            try:
                logger.info(f"Adding DSR/PBO significance testing for {symbol}")
                
                # Import enhanced evaluator
                from .services.enhanced_model_evaluation import EnhancedModelEvaluator
                
                # Get data for significance testing
                df = await analysis_service.data_pipeline.prepare_data_for_analysis(
                    symbol, period, add_features=True
                )
                
                if df is not None and not df.empty and len(df) >= 100:
                    # Prepare target column
                    target_column = 'return_1d'
                    if target_column not in df.columns:
                        df['return_1d'] = df['close'].pct_change()
                        df = df.dropna()
                    
                    # Initialize enhanced evaluator
                    enhanced_evaluator = EnhancedModelEvaluator(
                        spa_threshold=0.05,
                        pbo_threshold=0.2,
                        min_strategies_for_testing=2
                    )
                    
                    # Run significance testing only (extract from base result)
                    enhanced_result = await enhanced_evaluator.evaluate_with_significance_testing(
                        data=df,
                        target_column=target_column,
                        benchmark_returns=None
                    )
                    
                    # Add significance testing results to the base result
                    result['significance_testing'] = {
                        'deflated_sharpe_ratio': {
                            'dsr_value': enhanced_result.significance_metrics.deflated_sharpe,
                            'p_value': enhanced_result.significance_metrics.deflated_sharpe_p_value,
                            'is_significant': enhanced_result.significance_metrics.deflated_sharpe_significant
                        },
                        'backtest_overfitting': {
                            'pbo_estimate': enhanced_result.significance_metrics.pbo_estimate,
                            'is_overfitted': enhanced_result.significance_metrics.pbo_is_overfitted,
                            'interpretation': f"PBO < 0.2 indicates low overfitting risk"
                        },
                        'spa_test': {
                            'p_value': enhanced_result.significance_metrics.spa_p_value,
                            'is_significant': enhanced_result.significance_metrics.spa_is_significant
                        },
                        'deployment_assessment': {
                            'overall_approved': enhanced_result.overall_deployment_approved,
                            'statistical_confidence': enhanced_result.statistical_confidence
                        },
                        'interpretation': enhanced_result.significance_metrics.interpretation
                    }
                    
                    logger.info(f"Significance testing completed. DSR: {enhanced_result.significance_metrics.deflated_sharpe:.4f}, "
                               f"PBO: {enhanced_result.significance_metrics.pbo_estimate:.4f}")
                else:
                    result['significance_testing'] = {
                        'error': 'Insufficient data for significance testing',
                        'data_points': len(df) if df is not None else 0
                    }
                    
            except Exception as e:
                logger.error(f"Error in significance testing for {symbol}: {e}")
                result['significance_testing'] = {
                    'error': f'Significance testing failed: {str(e)}'
                }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced model evaluation for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/recommendation/{symbol}")
async def get_model_recommendation(symbol: str):
    """
    Get model recommendation based on recent evaluation results.
    Returns the best performing model with performance rationale.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Getting model recommendation for {symbol}")
        
        result = await forecasting_service.get_model_recommendation(symbol)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model recommendation for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/batch-evaluate")
async def batch_model_evaluation(request: BatchAnalysisRequest):
    """
    Run advanced model evaluation for multiple symbols.
    Useful for comparing model performance across different assets.
    """
    try:
        if len(request.symbols) > 3:  # Limit for intensive model evaluation
            raise HTTPException(
                status_code=400, 
                detail="Maximum 3 symbols allowed for batch model evaluation (intensive operation)"
            )
        
        logger.info(f"Starting batch model evaluation for {len(request.symbols)} symbols")
        
        results = {}
        errors = {}
        
        for symbol in request.symbols:
            try:
                symbol_upper = symbol.upper()
                result = await forecasting_service.evaluate_advanced_models(
                    symbol=symbol_upper,
                    period="2y",
                    cv_folds=3,  # Reduced for batch processing
                    include_financial_metrics=True
                )
                
                if 'error' in result:
                    errors[symbol_upper] = result
                else:
                    results[symbol_upper] = result
                    
            except Exception as e:
                errors[symbol.upper()] = {
                    'error': 'evaluation_failed',
                    'message': str(e)
                }
        
        return {
            'batch_evaluation_timestamp': datetime.now().isoformat(),
            'requested_symbols': request.symbols,
            'successful_evaluations': len(results),
            'failed_evaluations': len(errors),
            'results': results,
            'errors': errors if errors else None,
            'summary': {
                'best_models': {
                    symbol: result.get('best_model', 'unknown') 
                    for symbol, result in results.items()
                },
                'avg_performance': {
                    'avg_sharpe': sum(
                        result.get('model_performance', {})
                        .get(result.get('best_model', ''), {})
                        .get('financial_metrics', {})
                        .get('sharpe_ratio', 0) for result in results.values()
                    ) / len(results) if results else 0
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch model evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/evaluate-enhanced/{symbol}")
async def evaluate_models_with_significance_testing(
    symbol: str,
    period: str = Query(default="2y", description="Training data period"),
    include_spa_test: bool = Query(default=True, description="Include SPA test"),
    include_dsr_pbo: bool = Query(default=True, description="Include DSR and PBO analysis"),
    spa_threshold: float = Query(default=0.05, description="SPA significance threshold"),
    pbo_threshold: float = Query(default=0.2, description="Maximum acceptable PBO"),
    confidence_level: float = Query(default=0.95, description="Confidence level for tests")
):
    """
    Enhanced model evaluation with DSR/PBO and SPA significance testing.
    
    Provides comprehensive statistical analysis including:
    - Deflated Sharpe Ratio (DSR) with multiple testing correction
    - Probability of Backtest Overfitting (PBO) estimation
    - Superior Predictive Ability (SPA) test for data snooping
    - Deployment gate validation
    - Statistical confidence assessment
    
    This endpoint implements institutional-grade model validation standards.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Starting enhanced model evaluation with significance testing for {symbol}")
        
        # Import enhanced evaluator
        from .services.enhanced_model_evaluation import EnhancedModelEvaluator
        
        # Initialize enhanced evaluator with custom thresholds
        enhanced_evaluator = EnhancedModelEvaluator(
            spa_threshold=spa_threshold,
            pbo_threshold=pbo_threshold,
            min_strategies_for_testing=2  # Allow testing with fewer strategies
        )
        
        # Get data for evaluation
        df = await analysis_service.data_pipeline.prepare_data_for_analysis(
            symbol, period, add_features=True
        )
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {symbol} over period {period}"
            )
        
        if len(df) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for reliable analysis: {len(df)} points (minimum 100 required)"
            )
        
        # Determine target column (returns)
        target_column = 'return_1d'
        if target_column not in df.columns:
            # Calculate returns if not present
            df['return_1d'] = df['close'].pct_change()
            df = df.dropna()
        
        # Run enhanced evaluation with significance testing
        logger.info(f"Running enhanced evaluation with DSR/PBO for {symbol}")
        enhanced_result = await enhanced_evaluator.evaluate_with_significance_testing(
            data=df,
            target_column=target_column,
            benchmark_returns=None  # Will create market-like benchmark
        )
        
        # Format response with comprehensive results
        response = {
            'symbol': symbol,
            'evaluation_period': period,
            'data_points': len(df),
            'timestamp': enhanced_result.timestamp.isoformat(),
            
            # Base model evaluation results
            'model_performance': {
                'best_model': enhanced_result.base_evaluation.best_model,
                'models_tested': len(enhanced_result.base_evaluation.model_metrics),
                'model_metrics': {
                    model: {
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'volatility': metrics.volatility,
                        'max_drawdown': metrics.max_drawdown,
                        'hit_rate': getattr(metrics, 'hit_rate', None)
                    }
                    for model, metrics in enhanced_result.base_evaluation.model_metrics.items()
                }
            },
            
            # Significance testing results
            'significance_analysis': {
                'spa_test': {
                    'p_value': enhanced_result.significance_metrics.spa_p_value,
                    'is_significant': enhanced_result.significance_metrics.spa_is_significant,
                    'threshold': spa_threshold,
                    'passed': enhanced_result.spa_gate_passed
                } if include_spa_test else None,
                
                'deflated_sharpe_ratio': {
                    'dsr_value': enhanced_result.significance_metrics.deflated_sharpe,
                    'p_value': enhanced_result.significance_metrics.deflated_sharpe_p_value,
                    'is_significant': enhanced_result.significance_metrics.deflated_sharpe_significant,
                    'interpretation': 'Lower DSR indicates better risk-adjusted performance after multiple testing correction'
                } if include_dsr_pbo else None,
                
                'backtest_overfitting': {
                    'pbo_estimate': enhanced_result.significance_metrics.pbo_estimate,
                    'is_overfitted': enhanced_result.significance_metrics.pbo_is_overfitted,
                    'threshold': pbo_threshold,
                    'passed': enhanced_result.pbo_gate_passed,
                    'interpretation': f'PBO < {pbo_threshold} indicates low overfitting risk'
                } if include_dsr_pbo else None,
                
                'confidence_level': f'{confidence_level*100}%',
                'strategies_tested': enhanced_result.significance_metrics.n_strategies_tested,
                'interpretation': enhanced_result.significance_metrics.interpretation
            },
            
            # Deployment recommendation
            'deployment_assessment': {
                'overall_approved': enhanced_result.overall_deployment_approved,
                'recommendation': enhanced_result.deployment_recommendation,
                'statistical_confidence': enhanced_result.statistical_confidence,
                'overfitting_risk_score': enhanced_result.overfitting_risk_score,
                'risk_adjusted_ranking': enhanced_result.risk_adjusted_ranking
            },
            
            # Enhanced analysis
            'advanced_metrics': {
                'model_comparison': enhanced_result.model_comparison_analysis,
                'sensitivity_analysis': enhanced_result.sensitivity_analysis
            },
            
            # Configuration
            'test_configuration': {
                'spa_threshold': spa_threshold,
                'pbo_threshold': pbo_threshold,
                'confidence_level': confidence_level,
                'include_spa_test': include_spa_test,
                'include_dsr_pbo': include_dsr_pbo
            }
        }
        
        logger.info(f"Enhanced evaluation completed for {symbol}. "
                   f"Deployment approved: {enhanced_result.overall_deployment_approved}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enhanced model evaluation for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced model evaluation failed: {str(e)}"
        )

@app.get("/analyze/{symbol}/patterns")
async def get_chart_patterns(
    symbol: str,
    period: str = Query(default="3mo", description="Analysis period")
):
    """
    Get comprehensive chart pattern recognition for a symbol
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Getting chart patterns for {symbol}")
        
        # Get data
        df = await analysis_service.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=False)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Pattern recognition
        from .core.pattern_recognition import PatternRecognition
        pattern_analyzer = PatternRecognition()
        patterns = pattern_analyzer.detect_all_patterns(df)
        
        return {
            "symbol": symbol,
            "period": period,
            "data_points": len(df),
            "timestamp": datetime.now().isoformat(),
            "patterns": patterns
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patterns for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{symbol}/advanced")
async def get_advanced_indicators(
    symbol: str,
    period: str = Query(default="6mo", description="Analysis period"),
    indicators: List[str] = Query(default=["ichimoku", "adx", "sar", "mfi", "cci"], description="Advanced indicators to calculate")
):
    """
    Get advanced technical indicators for a symbol
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Getting advanced indicators {indicators} for {symbol}")
        
        # Get data
        df = await analysis_service.data_pipeline.prepare_data_for_analysis(symbol, period, add_features=False)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Advanced indicators
        from .core.advanced_indicators import AdvancedIndicators, CompositeIndicators
        adv_indicators = AdvancedIndicators()
        composite = CompositeIndicators()
        
        results = {
            "symbol": symbol,
            "period": period,
            "data_points": len(df),
            "timestamp": datetime.now().isoformat(),
            "advanced_indicators": {}
        }
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        if "ichimoku" in indicators and len(df) >= 52:
            ichimoku = adv_indicators.ichimoku_cloud(high, low, close)
            results["advanced_indicators"]["ichimoku"] = {
                "tenkan_sen": float(ichimoku['tenkan_sen'].iloc[-1]) if not ichimoku['tenkan_sen'].isna().all() else None,
                "kijun_sen": float(ichimoku['kijun_sen'].iloc[-1]) if not ichimoku['kijun_sen'].isna().all() else None,
                "senkou_span_a": float(ichimoku['senkou_span_a'].iloc[-1]) if not ichimoku['senkou_span_a'].isna().all() else None,
                "senkou_span_b": float(ichimoku['senkou_span_b'].iloc[-1]) if not ichimoku['senkou_span_b'].isna().all() else None,
                "cloud_signal": "BULLISH" if close.iloc[-1] > max(ichimoku['senkou_span_a'].iloc[-1], ichimoku['senkou_span_b'].iloc[-1]) else "BEARISH" if close.iloc[-1] < min(ichimoku['senkou_span_a'].iloc[-1], ichimoku['senkou_span_b'].iloc[-1]) else "NEUTRAL"
            }
        
        if "adx" in indicators:
            adx_data = adv_indicators.adx(high, low, close)
            results["advanced_indicators"]["adx"] = {
                "adx": float(adx_data['adx'].iloc[-1]) if not adx_data['adx'].isna().all() else None,
                "di_plus": float(adx_data['di_plus'].iloc[-1]) if not adx_data['di_plus'].isna().all() else None,
                "di_minus": float(adx_data['di_minus'].iloc[-1]) if not adx_data['di_minus'].isna().all() else None,
                "trend_strength": "VERY_STRONG" if adx_data['adx'].iloc[-1] > 50 else "STRONG" if adx_data['adx'].iloc[-1] > 25 else "WEAK" if not adx_data['adx'].isna().all() else None,
                "trend_direction": "BULLISH" if adx_data['di_plus'].iloc[-1] > adx_data['di_minus'].iloc[-1] else "BEARISH" if not adx_data['di_plus'].isna().all() else None
            }
        
        if "sar" in indicators and len(df) >= 10:
            sar = adv_indicators.parabolic_sar(high, low)
            results["advanced_indicators"]["parabolic_sar"] = {
                "value": float(sar.iloc[-1]) if not sar.isna().all() else None,
                "signal": "BUY" if sar.iloc[-1] < close.iloc[-1] else "SELL" if not sar.isna().all() else "NEUTRAL",
                "stop_loss_level": float(sar.iloc[-1]) if not sar.isna().all() else None
            }
        
        if "mfi" in indicators:
            mfi = adv_indicators.money_flow_index(high, low, close, volume)
            results["advanced_indicators"]["money_flow_index"] = {
                "value": float(mfi.iloc[-1]) if not mfi.isna().all() else None,
                "signal": "OVERBOUGHT" if mfi.iloc[-1] > 80 else "OVERSOLD" if mfi.iloc[-1] < 20 else "NEUTRAL" if not mfi.isna().all() else "NEUTRAL",
                "interpretation": "High money flow" if mfi.iloc[-1] > 50 else "Low money flow" if not mfi.isna().all() else None
            }
        
        if "cci" in indicators:
            cci = adv_indicators.commodity_channel_index(high, low, close)
            results["advanced_indicators"]["cci"] = {
                "value": float(cci.iloc[-1]) if not cci.isna().all() else None,
                "signal": "OVERBOUGHT" if cci.iloc[-1] > 100 else "OVERSOLD" if cci.iloc[-1] < -100 else "NEUTRAL" if not cci.isna().all() else "NEUTRAL",
                "extreme_level": cci.iloc[-1] > 200 or cci.iloc[-1] < -200 if not cci.isna().all() else False
            }
        
        if "vwap" in indicators:
            vwap = adv_indicators.vwap(high, low, close, volume)
            results["advanced_indicators"]["vwap"] = {
                "value": float(vwap.iloc[-1]) if not vwap.isna().all() else None,
                "signal": "ABOVE_VWAP" if close.iloc[-1] > vwap.iloc[-1] else "BELOW_VWAP" if not vwap.isna().all() else "NEUTRAL",
                "distance_percent": float((close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] * 100) if not vwap.isna().all() else None
            }
        
        # Add composite indicators
        if "strength" in indicators:
            market_strength = composite.market_strength_composite(df)
            results["advanced_indicators"]["market_strength"] = market_strength
        
        if "volatility_regime" in indicators:
            volatility_regime = composite.volatility_regime_detector(df)
            results["advanced_indicators"]["volatility_regime"] = volatility_regime
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting advanced indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/score/{symbol}")
async def get_composite_score(symbol: str, period: str = Query(default="6mo", description="Analysis period")):
    """
    Composite score blending technicals, sentiment, and fundamentals.
    Returns component scores, weights, and the final composite.
    """
    try:
        symbol = symbol.upper()
        # Use quick analysis snapshot as the technical base
        quick = await analysis_service.quick_analysis(symbol)
        result = await composite_algo.compute(symbol, quick)
        result.update({
            "period": period,
            "analysis_timestamp": datetime.now().isoformat()
        })
        return result
    except Exception as e:
        logger.error(f"Error computing composite score for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/{symbol}/comprehensive")
async def comprehensive_technical_analysis(
    symbol: str,
    period: str = Query(default="6mo", description="Analysis period"),
    include_patterns: bool = Query(default=True, description="Include pattern recognition"),
    include_advanced: bool = Query(default=True, description="Include advanced indicators")
):
    """
    Get the most comprehensive technical analysis combining all indicators and patterns
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Starting comprehensive technical analysis for {symbol}")
        
        # This will now include patterns and advanced indicators automatically
        result = await analysis_service.comprehensive_analysis(symbol, period)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        # Add analysis metadata
        result['analysis_type'] = 'comprehensive_with_patterns_and_advanced'
        result['includes_patterns'] = include_patterns
        result['includes_advanced_indicators'] = include_advanced
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/{symbol}/multi-factor")
async def multi_factor_analysis_endpoint(
    symbol: str,
    period: str = Query(default="6mo", description="Analysis period"),
    include_features: bool = Query(default=True, description="Include detailed feature breakdown")
):
    """
    Multi-factor analysis combining technical, fundamental, macro, options, sentiment, and event data
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Starting multi-factor analysis for {symbol}")
        
        result = await multi_factor_analysis.calculate_composite_score(symbol, period)
        
        if not include_features:
            # Remove detailed features to reduce response size
            result.pop("multi_factor_features", None)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in multi-factor analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/{symbol}/features")
async def get_multi_factor_features(
    symbol: str,
    period: str = Query(default="6mo", description="Analysis period"),
    categories: Optional[str] = Query(default=None, description="Comma-separated categories: technical,fundamental,macro,options,sentiment,event")
):
    """
    Get detailed multi-factor features for a symbol
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Getting multi-factor features for {symbol}")
        
        features = await multi_factor_analysis.get_multi_factor_features(symbol, period)
        
        # Filter by requested categories if specified
        if categories:
            requested_cats = [cat.strip().lower() for cat in categories.split(",")]
            filtered_features = {
                "symbol": features.symbol,
                "as_of": features.as_of.isoformat(),
            }
            
            if "technical" in requested_cats:
                filtered_features["technical_features"] = features.technical_features
            if "fundamental" in requested_cats:
                filtered_features["fundamental_features"] = features.fundamental_features
            if "macro" in requested_cats:
                filtered_features["macro_features"] = features.macro_features
            if "options" in requested_cats:
                filtered_features["options_features"] = features.options_features
            if "sentiment" in requested_cats:
                filtered_features["sentiment_features"] = features.sentiment_features
            if "event" in requested_cats:
                filtered_features["event_features"] = features.event_features
            if "interaction" in requested_cats:
                filtered_features["interaction_features"] = features.interaction_features
            
            return filtered_features
        
        # Return all features
        return {
            "symbol": features.symbol,
            "as_of": features.as_of.isoformat(),
            "technical_features": features.technical_features,
            "fundamental_features": features.fundamental_features,
            "macro_features": features.macro_features,
            "options_features": features.options_features,
            "sentiment_features": features.sentiment_features,
            "event_features": features.event_features,
            "interaction_features": features.interaction_features,
        }
        
    except Exception as e:
        logger.error(f"Error getting features for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Time Series Cross-Validation Endpoints

@app.get("/tscv/methods")
async def get_tscv_methods():
    """
    Get available time series cross-validation methods and their descriptions.
    """
    try:
        return {
            "available_methods": {
                "walk_forward": {
                    "description": "Walk-forward cross-validation with rolling windows",
                    "suitable_for": "Standard time series forecasting",
                    "benefits": ["Prevents look-ahead bias", "Simulates real trading conditions"]
                },
                "expanding_window": {
                    "description": "Expanding window cross-validation with growing training set", 
                    "suitable_for": "Long-term trend analysis",
                    "benefits": ["Utilizes all historical data", "Good for stable patterns"]
                },
                "rolling_window": {
                    "description": "Rolling window cross-validation with fixed training window size",
                    "suitable_for": "Regime-aware models and recent patterns",
                    "benefits": ["Focuses on recent patterns", "Adaptive to market changes"]
                },
                "regime_aware": {
                    "description": "Market regime-aware cross-validation aligned with volatility transitions",
                    "suitable_for": "Models sensitive to market conditions",
                    "benefits": ["Regime-specific evaluation", "Realistic performance assessment"]
                }
            },
            "default_configuration": {
                "method": "walk_forward",
                "n_splits": 5,
                "train_window_days": 1260,  # 5 years
                "test_window_days": 252,    # 1 year
                "min_train_days": 252,      # 1 year minimum
                "gap_days": 1,              # 1 day gap to prevent leakage
                "step_days": 126,           # 6 months step
                "adaptive_sizing": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting TSCV methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TSCVConfigRequest(BaseModel):
    method: str = "walk_forward"
    n_splits: int = 5
    train_window_days: Optional[int] = 1260
    test_window_days: int = 252
    min_train_days: int = 252
    gap_days: int = 1
    step_days: int = 126
    adaptive_sizing: bool = True


@app.post("/tscv/validate/{symbol}")
async def validate_tscv_configuration(
    symbol: str,
    config: TSCVConfigRequest,
    period: str = Query(default="2y", description="Historical data period")
):
    """
    Validate time series cross-validation configuration for a symbol.
    
    Tests the CV configuration to ensure it produces valid splits with the
    available data for the specified symbol.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Validating TSCV configuration for {symbol}")
        
        # Import TSCV classes
        from .services.time_series_cv import AdvancedTimeSeriesCV, CVConfiguration
        
        # Get data to validate configuration
        data = await forecasting_service.data_pipeline.prepare_data_for_analysis(
            symbol, period, add_features=True
        )
        
        if data is None or len(data) < 100:
            return {
                'error': f'Insufficient data for TSCV validation of {symbol}',
                'symbol': symbol,
                'required_samples': 100,
                'available_samples': len(data) if data is not None else 0
            }
        
        # Create CV configuration
        cv_config = CVConfiguration(**config.dict())
        
        # Initialize TSCV and test configuration
        tscv = AdvancedTimeSeriesCV()
        
        try:
            splits = tscv.create_splits(data, cv_config)
            
            # Validate splits
            validation_results = {
                'symbol': symbol,
                'configuration': config.dict(),
                'validation_status': 'valid',
                'total_data_points': len(data),
                'date_range': {
                    'start': data.index.min().isoformat(),
                    'end': data.index.max().isoformat()
                },
                'splits_generated': len(splits),
                'splits_summary': []
            }
            
            for split in splits:
                validation_results['splits_summary'].append({
                    'fold': split.fold_number,
                    'train_size': split.train_size,
                    'test_size': split.test_size,
                    'train_period': f"{split.train_start.date()} to {split.train_end.date()}",
                    'test_period': f"{split.test_start.date()} to {split.test_end.date()}",
                    'regime': split.regime_label
                })
            
            return validation_results
            
        except Exception as cv_error:
            return {
                'error': 'invalid_configuration',
                'symbol': symbol,
                'configuration': config.dict(),
                'validation_status': 'invalid',
                'message': str(cv_error),
                'suggestions': [
                    'Reduce number of splits',
                    'Decrease training window size',
                    'Increase step size',
                    'Use longer historical period'
                ]
            }
        
    except Exception as e:
        logger.error(f"Error validating TSCV configuration for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tscv/benchmark/{symbol}")
async def benchmark_tscv_methods(
    symbol: str,
    period: str = Query(default="2y", description="Historical data period"),
    quick_test: bool = Query(default=True, description="Run quick benchmark with reduced folds")
):
    """
    Benchmark different TSCV methods for a symbol to help choose the best approach.
    
    Compares walk-forward, expanding window, and rolling window methods using
    a simple baseline model to evaluate which CV strategy works best.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Benchmarking TSCV methods for {symbol}")
        
        # Get data
        data = await forecasting_service.data_pipeline.prepare_data_for_analysis(
            symbol, period, add_features=True
        )
        
        if data is None or len(data) < 200:
            return {
                'error': f'Insufficient data for TSCV benchmarking of {symbol}',
                'symbol': symbol,
                'required_samples': 200,
                'available_samples': len(data) if data is not None else 0
            }
        
        # Import TSCV classes
        from .services.time_series_cv import AdvancedTimeSeriesCV, CVConfiguration
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare features and target
        feature_columns = [col for col in data.columns if col != 'close']
        X = data[feature_columns]
        y = data['close'].pct_change().dropna()
        X = X.iloc[1:]  # Align with y
        
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index].fillna(0)
        y = y.loc[common_index]
        
        # Test different methods
        methods_to_test = ['walk_forward', 'expanding_window', 'rolling_window']
        n_splits = 3 if quick_test else 5
        
        benchmark_results = {
            'symbol': symbol,
            'benchmark_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_samples': len(X),
                'features': len(X.columns),
                'date_range': {
                    'start': X.index.min().isoformat(),
                    'end': X.index.max().isoformat()
                }
            },
            'method_comparison': {},
            'recommendations': []
        }
        
        # Simple model for benchmarking
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        
        for method in methods_to_test:
            try:
                cv_config = CVConfiguration(
                    method=method,
                    n_splits=n_splits,
                    train_window_days=756 if method == 'rolling_window' else None,
                    test_window_days=126,
                    step_days=63,
                    adaptive_sizing=True
                )
                
                tscv = AdvancedTimeSeriesCV()
                cv_results = tscv.evaluate_model_with_cv(
                    model, X, y, cv_config,
                    scoring_metrics=['r2', 'mse']
                )
                
                benchmark_results['method_comparison'][method] = {
                    'successful_folds': len(cv_results.fold_scores),
                    'avg_r2': np.mean([f.get('r2', 0) for f in cv_results.fold_scores]),
                    'r2_std': np.std([f.get('r2', 0) for f in cv_results.fold_scores]),
                    'execution_time': cv_results.execution_time,
                    'configuration': cv_config.__dict__
                }
                
            except Exception as method_error:
                benchmark_results['method_comparison'][method] = {
                    'error': str(method_error),
                    'status': 'failed'
                }
        
        # Generate recommendations
        successful_methods = {
            k: v for k, v in benchmark_results['method_comparison'].items() 
            if 'error' not in v
        }
        
        if successful_methods:
            best_method = max(
                successful_methods.keys(),
                key=lambda m: successful_methods[m]['avg_r2']
            )
            benchmark_results['recommendations'] = [
                f"Best performing method: {best_method}",
                f"Average R²: {successful_methods[best_method]['avg_r2']:.4f}",
                f"Execution time: {successful_methods[best_method]['execution_time']:.2f}s"
            ]
        else:
            benchmark_results['recommendations'] = [
                "All methods failed - consider using longer historical period",
                "Try reducing number of CV folds",
                "Check data quality and completeness"
            ]
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error benchmarking TSCV methods for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data Leakage Audit Endpoints

@app.get("/audit/leakage/{symbol}")
async def audit_data_leakage(
    symbol: str,
    period: str = Query(default="2y", description="Historical data period"),
    target_col: str = Query(default="close", description="Target column for leakage detection")
):
    """
    Perform comprehensive data leakage audit for a symbol.
    
    Detects temporal violations, options data leakage, event timing issues,
    and preprocessing bias to ensure zero look-ahead bias in features.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Starting data leakage audit for {symbol}")
        
        # Import pipeline here to avoid circular imports
        from .core.data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        audit_result = await pipeline.audit_data_leakage(symbol, period, target_col)
        
        # Format response
        return {
            'symbol': symbol,
            'audit_timestamp': audit_result.audit_timestamp.isoformat(),
            'compliance_score': audit_result.compliance_score,
            'dataset_info': audit_result.dataset_info,
            'audit_summary': audit_result.audit_summary,
            'violations': [
                {
                    'violation_id': v.violation_id,
                    'type': v.leakage_type.value,
                    'feature': v.feature_name,
                    'severity': v.severity,
                    'description': v.description,
                    'evidence': v.evidence,
                    'recommendation': v.recommendation
                }
                for v in audit_result.violations
            ],
            'passed_features': audit_result.passed_features,
            'recommendations': audit_result.recommendations,
            'compliance_status': 'PASS' if audit_result.compliance_score >= 0.8 else 'FAIL'
        }
        
    except Exception as e:
        logger.error(f"Error auditing data leakage for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/quick-check/{symbol}")
async def quick_leakage_check(
    symbol: str,
    period: str = Query(default="1y", description="Historical data period"),
    compliance_threshold: float = Query(default=0.8, description="Minimum compliance score (0.0-1.0)")
):
    """
    Quick automated leakage check with pass/fail results.
    
    Provides fast assessment of data pipeline compliance for production readiness.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Running quick leakage check for {symbol}")
        
        # Import pipeline here to avoid circular imports
        from .core.data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        check_result = await pipeline.automated_leakage_check(
            symbol, period, compliance_threshold
        )
        
        return check_result
        
    except Exception as e:
        logger.error(f"Error in quick leakage check for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BatchLeakageRequest(BaseModel):
    symbols: List[str]
    period: Optional[str] = "1y"
    compliance_threshold: Optional[float] = 0.8


@app.post("/audit/batch-check")
async def batch_leakage_check(request: BatchLeakageRequest):
    """
    Batch leakage check across multiple symbols.
    
    Runs automated leakage checks for multiple symbols and provides
    aggregated compliance results for portfolio-wide assessment.
    """
    try:
        if len(request.symbols) > 10:  # Limit for audit operations
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 symbols allowed for batch audit (intensive operation)"
            )
        
        logger.info(f"Starting batch leakage check for {len(request.symbols)} symbols")
        
        # Import pipeline here to avoid circular imports
        from .core.data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        
        results = {}
        failed_symbols = []
        passed_count = 0
        
        for symbol in request.symbols:
            try:
                symbol_upper = symbol.upper()
                check_result = await pipeline.automated_leakage_check(
                    symbol_upper, request.period, request.compliance_threshold
                )
                
                results[symbol_upper] = check_result
                if check_result.get('passed', False):
                    passed_count += 1
                    
            except Exception as symbol_error:
                failed_symbols.append(symbol.upper())
                logger.error(f"Leakage check failed for {symbol}: {symbol_error}")
                results[symbol.upper()] = {
                    'symbol': symbol.upper(),
                    'passed': False,
                    'status': 'ERROR',
                    'error': str(symbol_error)
                }
        
        # Calculate aggregate statistics
        total_symbols = len(request.symbols)
        pass_rate = passed_count / total_symbols if total_symbols > 0 else 0.0
        
        # Aggregate compliance scores
        compliance_scores = [
            r.get('compliance_score', 0.0) 
            for r in results.values() 
            if 'compliance_score' in r
        ]
        avg_compliance = np.mean(compliance_scores) if compliance_scores else 0.0
        
        # Count violation types across all symbols
        all_violation_types = {}
        for result in results.values():
            if 'violations_by_type' in result:
                for vtype, count in result['violations_by_type'].items():
                    all_violation_types[vtype] = all_violation_types.get(vtype, 0) + count
        
        return {
            'batch_check_timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(request.symbols),
            'symbols_passed': passed_count,
            'symbols_failed': total_symbols - passed_count,
            'pass_rate': pass_rate,
            'failed_symbols': failed_symbols,
            'aggregate_compliance': avg_compliance,
            'compliance_threshold': request.compliance_threshold,
            'portfolio_status': 'PASS' if pass_rate >= 0.8 else 'FAIL',
            'aggregate_violations': all_violation_types,
            'recommendations': [
                f"Portfolio pass rate: {pass_rate:.1%}",
                f"Average compliance: {avg_compliance:.1%}",
                "Review failed symbols for critical violations" if failed_symbols else "All symbols passed basic checks"
            ],
            'individual_results': results
        }
        
    except Exception as e:
        logger.error(f"Error in batch leakage check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/status")
async def get_audit_status():
    """
    Get data leakage audit service status and capabilities.
    """
    try:
        return {
            "service": "Data Leakage Audit",
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                "temporal_integrity_check": True,
                "options_lag_validation": True,
                "event_timing_validation": True,
                "preprocessing_leakage_detection": True,
                "feature_engineering_audit": True,
                "automated_compliance_scoring": True
            },
            "violation_types": [
                "temporal_leakage",
                "options_leakage", 
                "event_leakage",
                "preprocessing_leakage",
                "feature_leakage",
                "intraday_leakage"
            ],
            "severity_levels": ["critical", "high", "medium", "low"],
            "compliance_thresholds": {
                "production_ready": 0.9,
                "acceptable": 0.8,
                "needs_review": 0.6,
                "critical": 0.4
            },
            "audit_scope": [
                "Feature naming conventions",
                "Temporal sequence validation", 
                "Options data lag structure",
                "Event announcement vs execution timing",
                "Preprocessing statistics leakage",
                "Target-derived feature detection",
                "Correlation-based leakage detection"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting audit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Feature Selection and Pruning Endpoints

@app.get("/features/analyze/{symbol}")
async def analyze_feature_importance(
    symbol: str,
    period: str = Query(default="2y", description="Historical data period"),
    method: str = Query(default="composite", description="Selection method: composite, shap, rfe, correlation, model")
):
    """
    Analyze feature importance for a symbol using advanced feature selection techniques.
    
    Combines SHAP values, RFE, and collinearity analysis to identify the most important
    features for price prediction while removing redundant features.
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Analyzing feature importance for {symbol} using {method}")
        
        result = await forecasting_service.analyze_feature_importance(
            symbol=symbol,
            period=period,
            method=method
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing feature importance for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/features/optimize/{symbol}")
async def optimize_feature_set(
    symbol: str,
    period: str = Query(default="2y", description="Historical data period"),
    method: str = Query(default="composite", description="Selection method"),
    target_reduction: float = Query(default=0.5, description="Target feature reduction ratio (0.0-1.0)")
):
    """
    Optimize feature set by automatically selecting the best features and removing redundant ones.
    
    Performs comprehensive feature selection to reduce model complexity while maintaining
    or improving predictive performance.
    """
    try:
        symbol = symbol.upper()
        
        if not 0.0 <= target_reduction <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="target_reduction must be between 0.0 and 1.0"
            )
        
        logger.info(f"Optimizing feature set for {symbol} with {target_reduction:.0%} reduction target")
        
        result = await forecasting_service.optimize_feature_set(
            symbol=symbol,
            period=period,
            method=method,
            target_reduction=target_reduction
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error optimizing feature set for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BatchFeatureRequest(BaseModel):
    symbols: List[str]
    period: Optional[str] = "2y"
    method: Optional[str] = "composite"


@app.post("/features/batch-analyze")
async def batch_feature_analysis(request: BatchFeatureRequest):
    """
    Perform batch feature analysis across multiple symbols.
    
    Analyzes feature importance for multiple symbols and provides aggregated insights
    about the most important features across the portfolio.
    """
    try:
        if len(request.symbols) > 5:  # Limit for intensive feature analysis
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 symbols allowed for batch feature analysis (intensive operation)"
            )
        
        logger.info(f"Starting batch feature analysis for {len(request.symbols)} symbols")
        
        result = await forecasting_service.batch_feature_analysis(
            symbols=[s.upper() for s in request.symbols],
            period=request.period,
            method=request.method
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in batch feature analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/status")
async def get_feature_selection_status():
    """
    Get current feature selection service status and configuration.
    """
    try:
        # Import here to avoid circular imports
        from .services.feature_selection import SHAP_AVAILABLE, LIGHTGBM_AVAILABLE
        
        return {
            "service": "Feature Selection and Pruning",
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                "shap_analysis": SHAP_AVAILABLE,
                "lightgbm_support": LIGHTGBM_AVAILABLE,
                "rfe_analysis": True,
                "collinearity_analysis": True,
                "composite_scoring": True
            },
            "available_methods": [
                "composite",
                "shap" if SHAP_AVAILABLE else None,
                "rfe", 
                "correlation",
                "model"
            ],
            "configuration": {
                "correlation_threshold": forecasting_service.feature_selector.correlation_threshold,
                "vif_threshold": forecasting_service.feature_selector.vif_threshold,
                "shap_threshold": forecasting_service.feature_selector.shap_threshold,
                "min_features": forecasting_service.feature_selector.min_features,
                "max_features": forecasting_service.feature_selector.max_features
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting feature selection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# MLflow Tracking and Experiment Management Endpoints

@app.get("/mlflow/experiments")
async def list_mlflow_experiments():
    """
    List all MLflow experiments with summary statistics.
    """
    try:
        experiments = await forecasting_service.mlflow_tracker.list_experiments()
        return {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(experiments),
            "experiments": experiments
        }
        
    except Exception as e:
        logger.error(f"Error listing MLflow experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/experiments/{experiment_name}/runs")
async def list_experiment_runs(
    experiment_name: str,
    limit: int = Query(default=50, description="Maximum number of runs to return")
):
    """
    List runs for a specific experiment.
    """
    try:
        runs = await forecasting_service.mlflow_tracker.list_runs(
            experiment_name=experiment_name,
            limit=limit
        )
        return {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(runs),
            "runs": runs
        }
        
    except Exception as e:
        logger.error(f"Error listing runs for experiment {experiment_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/runs/{run_id}")
async def get_run_details(run_id: str):
    """
    Get detailed information about a specific run.
    """
    try:
        run_details = await forecasting_service.mlflow_tracker.get_run_details(run_id)
        return {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "run_details": run_details
        }
        
    except Exception as e:
        logger.error(f"Error getting run details for {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/models")
async def list_registered_models():
    """
    List all registered models in MLflow model registry.
    """
    try:
        models = await forecasting_service.mlflow_tracker.list_registered_models()
        return {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(models),
            "models": models
        }
        
    except Exception as e:
        logger.error(f"Error listing registered models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/models/{model_name}")
async def get_model_details(model_name: str):
    """
    Get detailed information about a registered model including all versions.
    """
    try:
        model_details = await forecasting_service.mlflow_tracker.get_model_details(model_name)
        return {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "model_details": model_details
        }
        
    except Exception as e:
        logger.error(f"Error getting model details for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ModelPromotionRequest(BaseModel):
    model_name: str
    version: str
    stage: str  # "Staging", "Production", "Archived"
    description: Optional[str] = None


@app.post("/mlflow/models/promote")
async def promote_model(request: ModelPromotionRequest):
    """
    Promote a model version to a new stage (Staging, Production, Archived).
    """
    try:
        if request.stage not in ["Staging", "Production", "Archived"]:
            raise HTTPException(
                status_code=400,
                detail="Stage must be one of: Staging, Production, Archived"
            )
        
        result = await forecasting_service.mlflow_tracker.transition_model_stage(
            model_name=request.model_name,
            version=request.version,
            stage=request.stage,
            description=request.description
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "model_name": request.model_name,
            "version": request.version,
            "new_stage": request.stage,
            "transition_result": result
        }
        
    except Exception as e:
        logger.error(f"Error promoting model {request.model_name} v{request.version}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SearchRunsRequest(BaseModel):
    experiment_names: Optional[List[str]] = None
    filter_string: Optional[str] = None
    order_by: Optional[List[str]] = None
    max_results: Optional[int] = 100


@app.post("/mlflow/search")
async def search_runs(request: SearchRunsRequest):
    """
    Search runs across experiments with flexible filtering and ordering.
    
    Examples:
    - filter_string: "metrics.r2 > 0.5 and params.symbol = 'AAPL'"
    - order_by: ["metrics.sharpe_ratio DESC"]
    """
    try:
        runs = await forecasting_service.mlflow_tracker.search_runs(
            experiment_names=request.experiment_names,
            filter_string=request.filter_string,
            order_by=request.order_by,
            max_results=request.max_results
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "search_criteria": {
                "experiments": request.experiment_names,
                "filter": request.filter_string,
                "order_by": request.order_by,
                "max_results": request.max_results
            },
            "total_results": len(runs),
            "runs": runs
        }
        
    except Exception as e:
        logger.error(f"Error searching runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/leaderboard/{symbol}")
async def get_model_leaderboard(
    symbol: str,
    metric: str = Query(default="sharpe_ratio", description="Metric to rank by"),
    limit: int = Query(default=10, description="Number of top models to return")
):
    """
    Get leaderboard of best performing models for a symbol ranked by specified metric.
    """
    try:
        symbol = symbol.upper()
        
        # Search for runs for this symbol
        search_request = SearchRunsRequest(
            filter_string=f"params.symbol = '{symbol}'",
            order_by=[f"metrics.{metric} DESC"],
            max_results=limit
        )
        
        runs = await forecasting_service.mlflow_tracker.search_runs(
            filter_string=search_request.filter_string,
            order_by=search_request.order_by,
            max_results=search_request.max_results
        )
        
        leaderboard = []
        for i, run in enumerate(runs):
            if 'metrics' in run and metric in run['metrics']:
                leaderboard.append({
                    "rank": i + 1,
                    "run_id": run.get('run_id'),
                    "model_type": run.get('params', {}).get('model_type', 'unknown'),
                    "metric_value": run['metrics'][metric],
                    "experiment_name": run.get('experiment_name'),
                    "start_time": run.get('start_time'),
                    "metrics": run.get('metrics', {}),
                    "parameters": run.get('params', {})
                })
        
        return {
            "symbol": symbol,
            "metric": metric,
            "timestamp": datetime.now().isoformat(),
            "leaderboard": leaderboard,
            "total_models": len(leaderboard)
        }
        
    except Exception as e:
        logger.error(f"Error getting model leaderboard for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlflow/status")
async def get_mlflow_status():
    """
    Get MLflow tracking service status and configuration.
    """
    try:
        status = await forecasting_service.mlflow_tracker.get_tracking_status()
        return {
            "service": "MLflow Tracking",
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "tracking_uri": forecasting_service.mlflow_tracker.tracking_uri,
            "capabilities": {
                "experiment_tracking": True,
                "model_registry": True,
                "artifact_storage": True,
                "model_versioning": True,
                "stage_transitions": True,
                "run_comparison": True,
                "metric_logging": True,
                "parameter_logging": True
            },
            "supported_stages": ["None", "Staging", "Production", "Archived"],
            "default_artifacts_path": forecasting_service.mlflow_tracker.artifacts_path
        }
        
    except Exception as e:
        logger.error(f"Error getting MLflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Deployment Pipeline Endpoints

@app.post("/deploy/staging/{model_name}")
async def deploy_model_to_staging(
    model_name: str,
    version: str = Query(..., description="Model version to deploy"),
    strategy: str = Query(default="blue_green", description="Deployment strategy")
):
    """
    Deploy model to staging environment for testing.
    
    Supports blue_green, canary, and immediate deployment strategies.
    """
    try:
        logger.info(f"Deploying {model_name} v{version} to staging")
        
        result = await forecasting_service.deploy_model_to_staging(
            model_name=model_name,
            version=version,
            strategy=strategy
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error deploying {model_name} v{version} to staging: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deploy/production/{model_name}")
async def promote_model_to_production(
    model_name: str,
    version: str = Query(..., description="Model version to promote"),
    strategy: str = Query(default="blue_green", description="Deployment strategy"),
    min_sharpe_ratio: float = Query(default=1.0, description="Minimum Sharpe ratio"),
    min_hit_rate: float = Query(default=0.55, description="Minimum hit rate"),
    max_drawdown: float = Query(default=0.15, description="Maximum drawdown"),
    min_r2_score: float = Query(default=0.4, description="Minimum R² score"),
    min_days_in_staging: int = Query(default=7, description="Minimum days in staging")
):
    """
    Promote model from staging to production with validation.
    
    Validates promotion criteria before deploying to production.
    """
    try:
        logger.info(f"Promoting {model_name} v{version} to production")
        
        promotion_criteria = {
            'min_sharpe_ratio': min_sharpe_ratio,
            'min_hit_rate': min_hit_rate,
            'max_drawdown': max_drawdown,
            'min_r2_score': min_r2_score,
            'min_days_in_staging': min_days_in_staging
        }
        
        result = await forecasting_service.promote_model_to_production(
            model_name=model_name,
            version=version,
            strategy=strategy,
            promotion_criteria=promotion_criteria
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error promoting {model_name} v{version} to production: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deploy/shadow/{model_name}")
async def create_shadow_deployment(
    model_name: str,
    version: str = Query(..., description="Model version to deploy"),
    traffic_pct: float = Query(default=0.1, description="Percentage of traffic to shadow (0.0-1.0)")
):
    """
    Create shadow deployment for testing without affecting production traffic.
    
    Shadow deployments receive a copy of production traffic for testing.
    """
    try:
        if not 0.0 <= traffic_pct <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="traffic_pct must be between 0.0 and 1.0"
            )
        
        logger.info(f"Creating shadow deployment for {model_name} v{version}")
        
        result = await forecasting_service.create_shadow_deployment(
            model_name=model_name,
            version=version,
            traffic_pct=traffic_pct
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error creating shadow deployment for {model_name} v{version}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deploy/rollback/{deployment_id}")
async def rollback_deployment(deployment_id: str):
    """
    Rollback a deployment to previous version.
    
    Safely reverts a deployment and restores previous model version.
    """
    try:
        logger.info(f"Rolling back deployment {deployment_id}")
        
        result = await forecasting_service.rollback_deployment(deployment_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error rolling back deployment {deployment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deploy/status/{deployment_id}")
async def get_deployment_status(deployment_id: str):
    """
    Get status of a specific deployment.
    
    Returns detailed information about deployment progress and health.
    """
    try:
        result = await forecasting_service.get_deployment_status(deployment_id)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployment status for {deployment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deploy/list")
async def list_active_deployments():
    """
    List all active deployments across all stages.
    
    Provides overview of current deployment status.
    """
    try:
        result = await forecasting_service.list_active_deployments()
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing active deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deploy/health")
async def get_endpoint_health(
    stage: str = Query(default="all", description="Stage to check: staging, production, shadow, all")
):
    """
    Get health status of endpoints for specified stage or all stages.
    
    Performs health checks and returns status for model serving endpoints.
    """
    try:
        result = await forecasting_service.get_endpoint_health(stage)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting endpoint health for stage {stage}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DeploymentConfigRequest(BaseModel):
    model_name: str
    version: str
    strategy: str = "blue_green"
    promotion_criteria: Optional[Dict[str, float]] = None
    shadow_traffic_pct: Optional[float] = 0.1


@app.post("/deploy/batch")
async def batch_deployment(request: DeploymentConfigRequest):
    """
    Perform batch deployment operations.
    
    Deploys model through staging to production pipeline with automated validation.
    """
    try:
        if len(request.model_name) == 0:
            raise HTTPException(
                status_code=400,
                detail="model_name is required"
            )
        
        logger.info(f"Starting batch deployment for {request.model_name} v{request.version}")
        
        results = {}
        
        # Step 1: Deploy to staging
        staging_result = await forecasting_service.deploy_model_to_staging(
            model_name=request.model_name,
            version=request.version,
            strategy=request.strategy
        )
        results['staging'] = staging_result
        
        if staging_result.get('status') == 'success':
            # Step 2: Create shadow deployment for testing
            shadow_result = await forecasting_service.create_shadow_deployment(
                model_name=request.model_name,
                version=request.version,
                traffic_pct=request.shadow_traffic_pct
            )
            results['shadow'] = shadow_result
            
            # Step 3: Promote to production (optional, based on criteria)
            if request.promotion_criteria:
                production_result = await forecasting_service.promote_model_to_production(
                    model_name=request.model_name,
                    version=request.version,
                    strategy=request.strategy,
                    promotion_criteria=request.promotion_criteria
                )
                results['production'] = production_result
        
        return {
            'batch_deployment_id': f"batch_{request.model_name}_{request.version}_{int(time.time())}",
            'model_name': request.model_name,
            'version': request.version,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in batch deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deploy/pipeline-status")
async def get_deployment_pipeline_status():
    """
    Get overall deployment pipeline status and configuration.
    
    Provides system-wide deployment capability information.
    """
    try:
        return {
            "service": "Model Deployment Pipeline",
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                "staging_deployment": True,
                "production_deployment": True,
                "shadow_deployment": True,
                "blue_green_strategy": True,
                "canary_strategy": True,
                "rollback_support": True,
                "health_monitoring": True,
                "promotion_validation": True
            },
            "deployment_strategies": [
                "blue_green",
                "canary", 
                "rolling",
                "immediate"
            ],
            "deployment_stages": [
                "staging",
                "production",
                "shadow",
                "archived"
            ],
            "default_criteria": {
                "min_sharpe_ratio": 1.0,
                "min_hit_rate": 0.55,
                "max_drawdown": 0.15,
                "min_r2_score": 0.4,
                "min_days_in_staging": 7,
                "max_latency_ms": 500.0,
                "min_uptime_pct": 99.0
            },
            "port_configuration": {
                "staging_port": 8004,
                "production_port": 8005,
                "shadow_port": 8007
            }
        }


# Model Drift Monitoring Endpoints

@app.post("/drift/baseline/{model_name}")
async def store_model_baseline(
    model_name: str,
    symbol: str = Query(..., description="Trading symbol"),
    period: str = Query(default="2y", description="Historical data period for baseline")
):
    """
    Store baseline feature distributions for drift monitoring.
    
    This establishes the reference distribution for detecting future drift.
    """
    try:
        logger.info(f"Storing baseline for {model_name}_{symbol}")
        
        result = await forecasting_service.store_model_baseline(
            model_name=model_name,
            symbol=symbol,
            period=period
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing baseline for {model_name}_{symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/analyze/{model_name}")
async def analyze_model_drift(
    model_name: str,
    symbol: str = Query(..., description="Trading symbol"),
    model_version: str = Query(default="latest", description="Model version"),
    period: str = Query(default="30d", description="Period for current data analysis")
):
    """
    Analyze model drift using PSI and KS tests.
    
    Compares current feature distributions against stored baseline
    to detect data distribution changes that may affect model performance.
    """
    try:
        logger.info(f"Analyzing drift for {model_name}_{symbol}")
        
        result = await forecasting_service.analyze_model_drift(
            model_name=model_name,
            symbol=symbol,
            model_version=model_version,
            period=period
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing drift for {model_name}_{symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/status")
async def get_drift_monitoring_status():
    """
    Get overall drift monitoring status across all models.
    
    Provides system-wide view of drift monitoring health and statistics.
    """
    try:
        result = await forecasting_service.get_drift_monitoring_status()
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting drift monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DriftThresholdsRequest(BaseModel):
    psi_low: Optional[float] = None
    psi_medium: Optional[float] = None
    psi_high: Optional[float] = None
    ks_low: Optional[float] = None
    ks_medium: Optional[float] = None
    ks_high: Optional[float] = None
    ks_p_value: Optional[float] = None
    min_samples: Optional[int] = None


@app.post("/drift/configure")
async def configure_drift_thresholds(request: DriftThresholdsRequest):
    """
    Configure drift detection thresholds.
    
    Allows fine-tuning of drift sensitivity for different models and use cases.
    """
    try:
        # Filter out None values
        thresholds = {k: v for k, v in request.dict().items() if v is not None}
        
        if not thresholds:
            raise HTTPException(
                status_code=400,
                detail="At least one threshold parameter must be provided"
            )
        
        result = await forecasting_service.configure_drift_thresholds(thresholds)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring drift thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BatchDriftRequest(BaseModel):
    models: List[Dict[str, str]]
    period: Optional[str] = "30d"


@app.post("/drift/batch")
async def batch_drift_analysis(request: BatchDriftRequest):
    """
    Perform batch drift analysis across multiple models.
    
    Efficiently analyzes drift for multiple model-symbol combinations
    and provides aggregated insights.
    """
    try:
        if len(request.models) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 models allowed for batch drift analysis"
            )
        
        # Validate model entries
        for model_info in request.models:
            if 'model_name' not in model_info or 'symbol' not in model_info:
                raise HTTPException(
                    status_code=400,
                    detail="Each model entry must have 'model_name' and 'symbol' keys"
                )
        
        logger.info(f"Starting batch drift analysis for {len(request.models)} models")
        
        result = await forecasting_service.batch_drift_analysis(
            models=request.models,
            period=request.period
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch drift analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift/capabilities")
async def get_drift_monitoring_capabilities():
    """
    Get drift monitoring system capabilities and configuration.
    
    Provides information about available drift detection methods and settings.
    """
    try:
        return {
            "service": "Model Drift Monitoring",
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                "population_stability_index": True,
                "kolmogorov_smirnov_test": True,
                "automated_alerting": True,
                "retraining_workflow_trigger": True,
                "batch_analysis": True,
                "configurable_thresholds": True,
                "baseline_storage": True,
                "distribution_comparison": True
            },
            "drift_metrics": {
                "psi": {
                    "description": "Population Stability Index for feature distribution monitoring",
                    "thresholds": {
                        "low": "< 0.1 (stable)",
                        "medium": "0.1 - 0.2 (minor drift)",
                        "high": "0.2 - 0.25 (significant drift)",
                        "critical": "> 0.25 (major drift)"
                    }
                },
                "ks_test": {
                    "description": "Kolmogorov-Smirnov test for data distribution skew detection",
                    "thresholds": {
                        "low": "< 0.1 (stable)",
                        "medium": "0.1 - 0.2 (minor drift)",
                        "high": "0.2 - 0.3 (significant drift)",
                        "critical": "> 0.3 (major drift)"
                    }
                }
            },
            "severity_levels": ["low", "medium", "high", "critical"],
            "drift_status_levels": ["stable", "minor_drift", "significant_drift", "critical_drift"],
            "automated_actions": {
                "critical_drift": "Trigger immediate retraining workflow",
                "significant_drift": "Schedule retraining within 1-2 weeks",
                "minor_drift": "Increase monitoring frequency",
                "stable": "Continue regular monitoring"
            },
            "configuration": {
                "default_thresholds": {
                    "psi_low": 0.1,
                    "psi_medium": 0.2,
                    "psi_high": 0.25,
                    "ks_low": 0.1,
                    "ks_medium": 0.2,
                    "ks_high": 0.3,
                    "ks_p_value": 0.05,
                    "min_samples": 100
                },
                "monitoring_frequency": {
                    "critical": "daily",
                    "significant": "weekly", 
                    "minor": "bi-weekly",
                    "stable": "monthly"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting drift monitoring capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Signal-to-Execution Latency Tracking Endpoints

@app.post("/signals/generate/{symbol}")
async def generate_trading_signal(
    symbol: str,
    signal_type: str = Query(default="auto", description="Signal type: buy, sell, auto"),
    confidence_threshold: float = Query(default=0.7, ge=0.0, le=1.0, description="Minimum model confidence")
):
    """
    Generate trading signal with latency tracking.
    
    Creates a trading signal based on model forecasts and begins tracking
    the signal-to-execution latency for alpha decay analysis.
    """
    try:
        result = await forecasting_service.generate_trading_signal(
            symbol=symbol,
            signal_type=signal_type,
            model_confidence_threshold=confidence_threshold
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating trading signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signals/track-transmission/{signal_id}")
async def track_signal_transmission(signal_id: str):
    """
    Track signal transmission to strategy service.
    
    Logs the transmission checkpoint for latency measurement.
    """
    try:
        result = await forecasting_service.track_signal_transmission(signal_id)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking signal transmission: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signals/alpha-decay/{signal_id}")
async def calculate_signal_alpha_decay(
    signal_id: str,
    current_price: float = Query(..., description="Current market price"),
    holding_period_hours: float = Query(default=24.0, ge=0.1, le=168.0, description="Holding period for analysis")
):
    """
    Calculate alpha decay for a signal.
    
    Measures how much trading alpha was lost due to execution delays
    by comparing potential returns vs actual returns.
    """
    try:
        result = await forecasting_service.calculate_signal_alpha_decay(
            signal_id=signal_id,
            current_price=current_price,
            holding_period_hours=holding_period_hours
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating alpha decay: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latency/performance")
async def get_latency_performance(
    lookback_minutes: int = Query(default=60, ge=1, le=1440, description="Analysis window in minutes")
):
    """
    Get latency performance analysis.
    
    Provides comprehensive analysis of signal-to-execution latency
    including bottleneck identification and performance metrics.
    """
    try:
        result = await forecasting_service.get_latency_performance(lookback_minutes)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting latency performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latency/alpha-decay")
async def get_alpha_decay_performance(
    lookback_hours: int = Query(default=24, ge=1, le=168, description="Analysis window in hours")
):
    """
    Get alpha decay performance analysis.
    
    Provides analysis of alpha decay patterns and correlations
    with execution latency across all tracked signals.
    """
    try:
        result = await forecasting_service.get_alpha_decay_performance(lookback_hours)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting alpha decay performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latency/bottlenecks")
async def identify_execution_bottlenecks():
    """
    Identify execution bottlenecks in the signal pipeline.
    
    Analyzes latency measurements to identify the slowest components
    and stages in the signal-to-execution pipeline with optimization recommendations.
    """
    try:
        result = await forecasting_service.identify_execution_bottlenecks()
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error identifying bottlenecks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latency/capabilities")
async def get_latency_tracking_capabilities():
    """
    Get latency tracking system capabilities and configuration.
    
    Provides information about tracked stages, metrics, and analysis features.
    """
    try:
        return {
            "service": "Signal-to-Execution Latency Tracking",
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                "microsecond_precision": True,
                "cross_service_tracking": True,
                "alpha_decay_analysis": True,
                "bottleneck_identification": True,
                "real_time_monitoring": True,
                "performance_analytics": True
            },
            "tracked_stages": [
                {
                    "stage": "signal_generated",
                    "description": "Initial signal creation in analysis service",
                    "component": "analysis_service"
                },
                {
                    "stage": "signal_validated", 
                    "description": "Signal validation and quality checks",
                    "component": "analysis_service"
                },
                {
                    "stage": "signal_transmitted",
                    "description": "Signal sent to strategy service",
                    "component": "communication_layer"
                },
                {
                    "stage": "order_received",
                    "description": "Order received by strategy service",
                    "component": "strategy_service"
                },
                {
                    "stage": "order_validated",
                    "description": "Order validation and risk checks",
                    "component": "strategy_service"
                },
                {
                    "stage": "order_submitted",
                    "description": "Order submitted to execution engine",
                    "component": "execution_engine"
                },
                {
                    "stage": "order_executed",
                    "description": "Order filled in market",
                    "component": "execution_engine"
                },
                {
                    "stage": "execution_confirmed",
                    "description": "Execution confirmation received",
                    "component": "execution_engine"
                }
            ],
            "metrics": {
                "latency_measurement": "Microsecond precision timing between stages",
                "alpha_decay": "Lost alpha due to execution delays (basis points)",
                "decay_rate": "Alpha decay per millisecond of latency",
                "execution_quality": "How well execution matched signal intent",
                "bottleneck_analysis": "Identification of slowest pipeline components"
            },
            "signal_types": ["buy", "sell", "hold", "rebalance"],
            "analysis_features": {
                "real_time_monitoring": "Live latency tracking and alerts",
                "historical_analysis": "Trend analysis over time",
                "correlation_analysis": "Latency vs alpha decay correlations",
                "component_breakdown": "Latency by service/component",
                "optimization_recommendations": "Actionable performance improvements"
            },
            "performance_thresholds": {
                "excellent_latency": "< 10ms total pipeline",
                "good_latency": "10-50ms total pipeline",
                "acceptable_latency": "50-100ms total pipeline",
                "poor_latency": "> 100ms total pipeline",
                "critical_alpha_decay": "> 10bps per 100ms latency"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting latency tracking capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interpretability/explain/{symbol}")
async def explain_model_predictions(
    symbol: str,
    explainer_type: str = Query(default="shap_tree", description="Explainer type: shap_tree, shap_linear, shap_kernel, lime_tabular"),
    sample_size: int = Query(default=100, description="Number of samples to explain"),
    period: str = Query(default="1y", description="Period for training data")
):
    """
    Generate explanations for model predictions using SHAP or LIME.
    
    This endpoint provides feature attribution analysis to understand:
    - Which features most influence model predictions
    - How feature values impact the prediction direction
    - Global vs local feature importance patterns
    - Model behavior across different market conditions
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Explaining model predictions for {symbol}")
        
        result = await forecasting_service.explain_model_predictions(
            symbol=symbol,
            explainer_type=explainer_type,
            sample_size=sample_size,
            period=period
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining model predictions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interpretability/attribution-trends/{symbol}")
async def analyze_feature_attribution_trends(
    symbol: str,
    lookback_days: int = Query(default=90, description="Number of days to analyze"),
    regime_aware: bool = Query(default=True, description="Include regime-specific analysis")
):
    """
    Analyze feature attribution trends over time and across market regimes.
    
    This endpoint tracks how feature importance changes over time by:
    - Analyzing attribution trends across time windows
    - Computing feature stability metrics
    - Identifying regime-specific attribution patterns
    - Measuring model behavioral consistency
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Analyzing feature attribution trends for {symbol}")
        
        result = await forecasting_service.analyze_feature_attribution_trends(
            symbol=symbol,
            lookback_days=lookback_days,
            regime_aware=regime_aware
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing attribution trends for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interpretability/model-stability/{symbol}")
async def validate_model_stability(
    symbol: str,
    stability_period: int = Query(default=60, description="Number of days for stability analysis"),
    explainer_type: str = Query(default="shap_tree", description="Explainer type for analysis")
):
    """
    Validate model stability through interpretability analysis.
    
    This endpoint assesses model reliability by:
    - Measuring feature attribution consistency over time
    - Detecting model behavior shifts
    - Computing stability scores and confidence metrics
    - Identifying periods of model uncertainty
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Validating model stability for {symbol}")
        
        # Get the model explainer from forecasting service
        if not hasattr(forecasting_service, 'model_explainer'):
            raise HTTPException(
                status_code=500, 
                detail="Model explainer not initialized in forecasting service"
            )
        
        # Get training data for stability analysis
        data = await forecasting_service.data_pipeline.prepare_data_for_analysis(
            symbol, f"{stability_period * 2}d", add_features=True
        )
        
        if data is None or len(data) < stability_period:
            raise HTTPException(
                status_code=400,
                detail=f'Insufficient data for stability analysis of {symbol}'
            )
        
        # Get best model
        cache_key = f"{symbol}_best_model"
        if cache_key not in forecasting_service.model_cache:
            await forecasting_service.train_models(symbol, period=f"{stability_period * 2}d")
            
        if cache_key not in forecasting_service.model_cache:
            raise HTTPException(
                status_code=400,
                detail=f'No trained model available for {symbol}'
            )
            
        model = forecasting_service.model_cache[cache_key]
        
        # Prepare features
        feature_columns = [col for col in data.columns if col != 'close']
        X = data[feature_columns].fillna(0)
        
        # Validate model stability
        stability_result = await forecasting_service.model_explainer.validate_model_stability(
            model, X, stability_period
        )
        
        # Format response
        return {
            'symbol': symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'stability_period': stability_period,
            'model_type': type(model).__name__,
            'stability_metrics': {
                'overall_stability_score': float(stability_result.overall_stability),
                'attribution_consistency': float(stability_result.attribution_consistency),
                'prediction_variance': float(stability_result.prediction_variance),
                'feature_stability': {
                    feat.feature_name: {
                        'stability_score': float(feat.stability_score),
                        'variance': float(feat.variance),
                        'drift_detected': feat.drift_detected
                    }
                    for feat in stability_result.feature_stability[:20]  # Top 20 features
                }
            },
            'stability_windows': [
                {
                    'period_start': window.period_start.isoformat(),
                    'period_end': window.period_end.isoformat(),
                    'stability_score': float(window.stability_score),
                    'sample_count': window.sample_count
                }
                for window in stability_result.stability_windows
            ],
            'drift_alerts': [
                {
                    'feature_name': alert.feature_name,
                    'drift_score': float(alert.drift_score),
                    'detection_date': alert.detection_date.isoformat(),
                    'severity': alert.severity
                }
                for alert in stability_result.drift_alerts
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating model stability for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interpretability/compliance-report/{symbol}")
async def generate_compliance_report(
    symbol: str,
    report_period: int = Query(default=90, description="Number of days for compliance analysis"),
    include_validations: bool = Query(default=True, description="Include model validation metrics")
):
    """
    Generate regulatory compliance report for model validation.
    
    This endpoint creates comprehensive documentation for regulatory requirements:
    - Model performance and stability metrics
    - Feature attribution analysis and consistency
    - Risk assessment and validation results
    - Interpretability evidence for model decisions
    """
    try:
        symbol = symbol.upper()
        logger.info(f"Generating compliance report for {symbol}")
        
        # Get the model explainer from forecasting service
        if not hasattr(forecasting_service, 'model_explainer'):
            raise HTTPException(
                status_code=500, 
                detail="Model explainer not initialized in forecasting service"
            )
        
        # Get training data
        data = await forecasting_service.data_pipeline.prepare_data_for_analysis(
            symbol, f"{report_period * 2}d", add_features=True
        )
        
        if data is None or len(data) < report_period:
            raise HTTPException(
                status_code=400,
                detail=f'Insufficient data for compliance report of {symbol}'
            )
        
        # Get best model
        cache_key = f"{symbol}_best_model"
        if cache_key not in forecasting_service.model_cache:
            await forecasting_service.train_models(symbol, period=f"{report_period * 2}d")
            
        if cache_key not in forecasting_service.model_cache:
            raise HTTPException(
                status_code=400,
                detail=f'No trained model available for {symbol}'
            )
            
        model = forecasting_service.model_cache[cache_key]
        
        # Prepare features
        feature_columns = [col for col in data.columns if col != 'close']
        X = data[feature_columns].fillna(0)
        
        # Generate compliance report
        compliance_report = await forecasting_service.model_explainer.generate_compliance_report(
            model, X, symbol, report_period
        )
        
        # Format response
        return {
            'symbol': symbol,
            'report_timestamp': datetime.now().isoformat(),
            'report_period': report_period,
            'model_type': type(model).__name__,
            'compliance_summary': {
                'model_id': compliance_report.model_id,
                'validation_status': compliance_report.validation_status,
                'compliance_score': float(compliance_report.compliance_score),
                'last_validation_date': compliance_report.last_validation_date.isoformat(),
                'next_validation_due': compliance_report.next_validation_due.isoformat()
            },
            'model_documentation': {
                'purpose': compliance_report.model_documentation.purpose,
                'methodology': compliance_report.model_documentation.methodology,
                'limitations': compliance_report.model_documentation.limitations,
                'assumptions': compliance_report.model_documentation.assumptions,
                'validation_approach': compliance_report.model_documentation.validation_approach
            },
            'performance_metrics': {
                'training_performance': {
                    metric: float(value) for metric, value in 
                    compliance_report.performance_metrics.training_performance.items()
                },
                'validation_performance': {
                    metric: float(value) for metric, value in 
                    compliance_report.performance_metrics.validation_performance.items()
                },
                'stability_metrics': {
                    metric: float(value) for metric, value in 
                    compliance_report.performance_metrics.stability_metrics.items()
                }
            },
            'interpretability_analysis': {
                'global_feature_importance': {
                    feat.feature_name: float(feat.importance_score)
                    for feat in compliance_report.interpretability_analysis.global_feature_importance[:15]
                },
                'feature_stability_scores': {
                    feat.feature_name: float(feat.stability_score)
                    for feat in compliance_report.interpretability_analysis.feature_stability[:15]
                },
                'attribution_consistency': float(compliance_report.interpretability_analysis.attribution_consistency)
            },
            'risk_assessment': {
                'model_risk_rating': compliance_report.risk_assessment.model_risk_rating,
                'risk_factors': compliance_report.risk_assessment.risk_factors,
                'mitigation_measures': compliance_report.risk_assessment.mitigation_measures,
                'monitoring_requirements': compliance_report.risk_assessment.monitoring_requirements
            },
            'validation_results': {
                'backtesting_results': {
                    metric: float(value) for metric, value in 
                    compliance_report.validation_results.backtesting_results.items()
                },
                'stress_testing_results': {
                    scenario: {metric: float(value) for metric, value in results.items()}
                    for scenario, results in compliance_report.validation_results.stress_testing_results.items()
                },
                'model_limitations': compliance_report.validation_results.model_limitations,
                'validation_conclusions': compliance_report.validation_results.validation_conclusions
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating compliance report for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
