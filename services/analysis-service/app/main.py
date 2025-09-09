"""
Analysis Service API
Provides comprehensive technical and statistical analysis for trading
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from .services.analysis_service import ComprehensiveAnalysisService
from .services.forecasting_service import ForecastingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
analysis_service = ComprehensiveAnalysisService()
forecasting_service = ForecastingService()

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
            "market_data_service": "http://localhost:8002"
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
        
        result = await forecasting_service.generate_forecast(
            symbol=symbol,
            model_type=model_type,
            prediction_horizon=horizon,
            period=period,
            retrain=retrain
        )
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ML forecast for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )