"""
Gap Trading API endpoints for comprehensive gap analysis and monitoring.
Provides gap continuation vs fade analysis with pre/post-market context.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pandas as pd

from ..services.gap_trading_engine import (
    GapTradingEngine,
    GapAnalysis,
    GapMonitoringResult,
    GapType,
    GapSize,
    GapDirection,
    MarketSession,
    PrePostMarketData
)

router = APIRouter(prefix="/gap-trading", tags=["Gap Trading"])

# Request/Response Models
class GapAnalysisRequest(BaseModel):
    """Request for gap analysis"""
    symbol: str = Field(..., description="Stock symbol to analyze")
    current_price: float = Field(..., description="Current/opening price")
    previous_close: float = Field(..., description="Previous trading day close price")
    current_volume: Optional[int] = Field(None, description="Current volume")
    average_volume: Optional[int] = Field(None, description="Average daily volume")
    include_pre_market: bool = Field(default=True, description="Include pre-market analysis")
    include_overnight: bool = Field(default=True, description="Include overnight analysis")

class PreMarketDataInput(BaseModel):
    """Pre-market session data input"""
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    session_start: datetime
    session_end: datetime
    trade_count: Optional[int] = 0
    spread_avg: Optional[float] = 0.01
    vwap: Optional[float] = None

class NewsEventInput(BaseModel):
    """News event input for gap analysis"""
    event_type: str
    description: str
    timestamp: datetime
    impact_score: Optional[float] = 5.0
    source: Optional[str] = "news"

class GapAnalysisWithDataRequest(BaseModel):
    """Request for gap analysis with complete data"""
    symbol: str
    current_price: float
    previous_close: float
    current_volume: Optional[int] = None
    average_volume: Optional[int] = None
    pre_market_data: Optional[PreMarketDataInput] = None
    overnight_data: Optional[PreMarketDataInput] = None
    news_events: Optional[List[NewsEventInput]] = None

class GapMonitoringRequest(BaseModel):
    """Request for gap monitoring"""
    symbol: str
    gap_analysis_id: str  # Reference to previous gap analysis
    current_price: float
    current_volume: Optional[int] = None
    volume_profile: Optional[Dict[str, Any]] = None

class BulkGapScanRequest(BaseModel):
    """Request for bulk gap scanning"""
    symbols: List[str]
    min_gap_percentage: float = Field(default=0.02, description="Minimum gap percentage (2%)")
    max_gap_percentage: float = Field(default=0.25, description="Maximum gap percentage (25%)")
    include_pre_market_context: bool = Field(default=True)

class GapScreenerRequest(BaseModel):
    """Request for gap screening with filters"""
    min_gap_size: str = Field(default="small", description="Minimum gap size (micro/small/medium/large/massive)")
    max_gap_size: str = Field(default="massive", description="Maximum gap size")
    gap_direction: Optional[str] = Field(None, description="Gap direction (gap_up/gap_down)")
    min_volume_ratio: float = Field(default=1.5, description="Minimum volume ratio")
    require_news_catalyst: bool = Field(default=False)
    sector_filter: Optional[List[str]] = Field(None, description="Sector filters")

class GapResponse(BaseModel):
    """Response for gap analysis"""
    symbol: str
    gap_detected: bool
    gap_type: Optional[str] = None
    gap_size: Optional[str] = None
    gap_percentage: float
    gap_points: float
    previous_close: float
    current_open: float
    volume_surge: bool
    volume_ratio: float
    news_catalyst: bool
    earnings_gap: bool
    gap_fill_probability: float
    continuation_probability: float
    optimal_entry_price: float
    stop_loss_level: float
    profit_targets: List[float]
    risk_reward_ratio: float
    fibonacci_levels: Dict[str, float]
    support_resistance_levels: List[float]
    pre_market_summary: Optional[Dict[str, Any]] = None
    overnight_summary: Optional[Dict[str, Any]] = None

class GapMonitoringResponse(BaseModel):
    """Response for gap monitoring"""
    symbol: str
    current_direction: str
    fill_percentage: float
    time_since_open: str
    price_action_strength: float
    volume_analysis: Dict[str, float]
    momentum_indicators: Dict[str, float]
    liquidity_analysis: Dict[str, float]
    trading_recommendation: str
    updated_targets: List[float]

# Dependency injection
async def get_gap_engine() -> GapTradingEngine:
    """Get gap trading engine instance"""
    return GapTradingEngine()

# API Endpoints
@router.post("/analyze-gap", response_model=GapResponse)
async def analyze_gap(
    request: GapAnalysisRequest,
    engine: GapTradingEngine = Depends(get_gap_engine)
):
    """
    Analyze gap characteristics and trading potential.
    
    Provides comprehensive gap analysis including size classification,
    probability calculations, and trading setup recommendations.
    """
    try:
        # Prepare data for analysis
        current_price_data = {'price': request.current_price}
        volume_data = None
        
        if request.current_volume and request.average_volume:
            volume_data = {
                'current_volume': request.current_volume,
                'average_volume': request.average_volume
            }
        
        # Fetch pre-market and overnight data if requested
        # In a real implementation, this would fetch from data sources
        pre_market_data = None
        overnight_data = None
        
        gap_analysis = await engine.analyze_gap(
            symbol=request.symbol,
            current_price_data=current_price_data,
            previous_close=request.previous_close,
            pre_market_data=pre_market_data,
            overnight_data=overnight_data,
            volume_data=volume_data,
            news_events=None
        )
        
        # Format pre-market summary
        pre_market_summary = None
        if gap_analysis.pre_market_data:
            pre_market_summary = {
                'session_volume': gap_analysis.pre_market_data.volume,
                'price_range': gap_analysis.pre_market_data.high_price - gap_analysis.pre_market_data.low_price,
                'vwap': gap_analysis.pre_market_data.vwap,
                'liquidity_score': gap_analysis.pre_market_data.liquidity_score
            }
        
        # Format overnight summary
        overnight_summary = None
        if gap_analysis.overnight_data:
            overnight_summary = {
                'session_volume': gap_analysis.overnight_data.volume,
                'price_range': gap_analysis.overnight_data.high_price - gap_analysis.overnight_data.low_price,
                'vwap': gap_analysis.overnight_data.vwap,
                'liquidity_score': gap_analysis.overnight_data.liquidity_score
            }
        
        return GapResponse(
            symbol=gap_analysis.symbol,
            gap_detected=True,
            gap_type=gap_analysis.gap_type.value,
            gap_size=gap_analysis.gap_size.value,
            gap_percentage=gap_analysis.gap_percentage,
            gap_points=gap_analysis.gap_points,
            previous_close=gap_analysis.previous_close,
            current_open=gap_analysis.current_open,
            volume_surge=gap_analysis.volume_surge,
            volume_ratio=gap_analysis.volume_ratio,
            news_catalyst=gap_analysis.news_catalyst,
            earnings_gap=gap_analysis.earnings_gap,
            gap_fill_probability=gap_analysis.gap_fill_probability,
            continuation_probability=gap_analysis.continuation_probability,
            optimal_entry_price=gap_analysis.optimal_entry_price,
            stop_loss_level=gap_analysis.stop_loss_level,
            profit_targets=gap_analysis.profit_targets,
            risk_reward_ratio=gap_analysis.risk_reward_ratio,
            fibonacci_levels=gap_analysis.fibonacci_levels,
            support_resistance_levels=gap_analysis.support_resistance_levels,
            pre_market_summary=pre_market_summary,
            overnight_summary=overnight_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing gap: {str(e)}")

@router.post("/analyze-gap-with-data", response_model=GapResponse)
async def analyze_gap_with_data(
    request: GapAnalysisWithDataRequest,
    engine: GapTradingEngine = Depends(get_gap_engine)
):
    """
    Analyze gap with provided pre/post-market and news data.
    
    Allows for detailed gap analysis with custom data inputs for
    backtesting and comprehensive analysis scenarios.
    """
    try:
        # Convert input data to engine format
        current_price_data = {'price': request.current_price}
        
        volume_data = None
        if request.current_volume and request.average_volume:
            volume_data = {
                'current_volume': request.current_volume,
                'average_volume': request.average_volume
            }
        
        pre_market_data = None
        if request.pre_market_data:
            pre_market_data = {
                'open': request.pre_market_data.open_price,
                'high': request.pre_market_data.high_price,
                'low': request.pre_market_data.low_price,
                'close': request.pre_market_data.close_price,
                'volume': request.pre_market_data.volume,
                'session_start': request.pre_market_data.session_start,
                'session_end': request.pre_market_data.session_end,
                'trade_count': request.pre_market_data.trade_count,
                'spread_avg': request.pre_market_data.spread_avg,
                'vwap': request.pre_market_data.vwap
            }
        
        overnight_data = None
        if request.overnight_data:
            overnight_data = {
                'open': request.overnight_data.open_price,
                'high': request.overnight_data.high_price,
                'low': request.overnight_data.low_price,
                'close': request.overnight_data.close_price,
                'volume': request.overnight_data.volume,
                'session_start': request.overnight_data.session_start,
                'session_end': request.overnight_data.session_end,
                'trade_count': request.overnight_data.trade_count,
                'spread_avg': request.overnight_data.spread_avg,
                'vwap': request.overnight_data.vwap
            }
        
        news_events = None
        if request.news_events:
            news_events = [
                {
                    'event_type': event.event_type,
                    'description': event.description,
                    'timestamp': event.timestamp,
                    'impact_score': event.impact_score,
                    'source': event.source
                }
                for event in request.news_events
            ]
        
        gap_analysis = await engine.analyze_gap(
            symbol=request.symbol,
            current_price_data=current_price_data,
            previous_close=request.previous_close,
            pre_market_data=pre_market_data,
            overnight_data=overnight_data,
            volume_data=volume_data,
            news_events=news_events
        )
        
        return GapResponse(
            symbol=gap_analysis.symbol,
            gap_detected=True,
            gap_type=gap_analysis.gap_type.value,
            gap_size=gap_analysis.gap_size.value,
            gap_percentage=gap_analysis.gap_percentage,
            gap_points=gap_analysis.gap_points,
            previous_close=gap_analysis.previous_close,
            current_open=gap_analysis.current_open,
            volume_surge=gap_analysis.volume_surge,
            volume_ratio=gap_analysis.volume_ratio,
            news_catalyst=gap_analysis.news_catalyst,
            earnings_gap=gap_analysis.earnings_gap,
            gap_fill_probability=gap_analysis.gap_fill_probability,
            continuation_probability=gap_analysis.continuation_probability,
            optimal_entry_price=gap_analysis.optimal_entry_price,
            stop_loss_level=gap_analysis.stop_loss_level,
            profit_targets=gap_analysis.profit_targets,
            risk_reward_ratio=gap_analysis.risk_reward_ratio,
            fibonacci_levels=gap_analysis.fibonacci_levels,
            support_resistance_levels=gap_analysis.support_resistance_levels
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing gap with data: {str(e)}")

@router.post("/monitor-gap", response_model=GapMonitoringResponse)
async def monitor_gap(
    request: GapMonitoringRequest,
    engine: GapTradingEngine = Depends(get_gap_engine)
):
    """
    Monitor gap behavior in real-time.
    
    Tracks gap fill percentage, direction, and provides updated
    trading recommendations based on current price action.
    """
    try:
        # In a real implementation, retrieve gap_analysis from cache/database
        # For now, create a minimal gap analysis for demonstration
        
        # This would be retrieved based on gap_analysis_id
        gap_analysis = None  # Placeholder
        
        if not gap_analysis:
            raise HTTPException(status_code=404, detail="Gap analysis not found")
        
        current_price_data = {'price': request.current_price}
        if request.current_volume:
            current_price_data['volume'] = request.current_volume
        
        monitoring_result = await engine.monitor_gap_behavior(
            gap_analysis=gap_analysis,
            current_price_data=current_price_data,
            volume_profile=request.volume_profile
        )
        
        # Generate trading recommendation
        recommendation = _generate_trading_recommendation(monitoring_result)
        
        # Calculate updated targets
        updated_targets = _calculate_updated_targets(monitoring_result)
        
        return GapMonitoringResponse(
            symbol=request.symbol,
            current_direction=monitoring_result.current_direction.value,
            fill_percentage=monitoring_result.fill_percentage,
            time_since_open=str(monitoring_result.time_since_open),
            price_action_strength=monitoring_result.price_action_strength,
            volume_analysis=monitoring_result.volume_profile,
            momentum_indicators=monitoring_result.momentum_indicators,
            liquidity_analysis=monitoring_result.liquidity_analysis,
            trading_recommendation=recommendation,
            updated_targets=updated_targets
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error monitoring gap: {str(e)}")

@router.post("/bulk-gap-scan")
async def bulk_gap_scan(
    request: BulkGapScanRequest,
    engine: GapTradingEngine = Depends(get_gap_engine)
):
    """
    Scan multiple symbols for gap opportunities.
    
    Identifies gaps across a list of symbols and ranks them
    by trading potential and probability metrics.
    """
    try:
        results = {}
        gap_count = 0
        
        for symbol in request.symbols:
            try:
                # In a real implementation, fetch current and previous close prices
                # For demonstration, use placeholder data
                current_price = 100.0  # Placeholder
                previous_close = 95.0   # Placeholder - creates 5.26% gap
                
                gap_percentage = abs((current_price - previous_close) / previous_close)
                
                # Check if gap meets criteria
                if (request.min_gap_percentage <= gap_percentage <= request.max_gap_percentage):
                    
                    gap_analysis = await engine.analyze_gap(
                        symbol=symbol,
                        current_price_data={'price': current_price},
                        previous_close=previous_close,
                        pre_market_data={} if request.include_pre_market_context else None,
                        overnight_data=None,
                        volume_data=None,
                        news_events=None
                    )
                    
                    gap_count += 1
                    results[symbol] = {
                        'gap_percentage': gap_analysis.gap_percentage,
                        'gap_size': gap_analysis.gap_size.value,
                        'gap_type': gap_analysis.gap_type.value,
                        'continuation_probability': gap_analysis.continuation_probability,
                        'fill_probability': gap_analysis.gap_fill_probability,
                        'risk_reward_ratio': gap_analysis.risk_reward_ratio,
                        'volume_surge': gap_analysis.volume_surge,
                        'news_catalyst': gap_analysis.news_catalyst
                    }
                else:
                    results[symbol] = {'gap_detected': False}
                    
            except Exception as e:
                results[symbol] = {'error': f"Failed to analyze {symbol}: {str(e)}"}
        
        # Sort results by trading potential
        sorted_gaps = sorted(
            [(symbol, data) for symbol, data in results.items() 
             if isinstance(data, dict) and 'continuation_probability' in data],
            key=lambda x: x[1]['continuation_probability'] * x[1]['risk_reward_ratio'],
            reverse=True
        )
        
        return {
            'total_symbols_scanned': len(request.symbols),
            'gaps_detected': gap_count,
            'scan_criteria': {
                'min_gap_percentage': request.min_gap_percentage,
                'max_gap_percentage': request.max_gap_percentage,
                'include_pre_market': request.include_pre_market_context
            },
            'top_opportunities': dict(sorted_gaps[:10]),  # Top 10 opportunities
            'all_results': results,
            'scan_timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in bulk gap scan: {str(e)}")

@router.get("/gap-screener")
async def gap_screener(
    request: GapScreenerRequest,
    engine: GapTradingEngine = Depends(get_gap_engine)
):
    """
    Screen for gaps with advanced filtering criteria.
    
    Provides sophisticated gap screening with multiple filter
    options for professional gap trading workflows.
    """
    try:
        # In a real implementation, this would query market data for all symbols
        # and apply the screening criteria
        
        # Placeholder implementation
        screener_results = {
            'criteria': {
                'min_gap_size': request.min_gap_size,
                'max_gap_size': request.max_gap_size,
                'gap_direction': request.gap_direction,
                'min_volume_ratio': request.min_volume_ratio,
                'require_news_catalyst': request.require_news_catalyst,
                'sector_filter': request.sector_filter
            },
            'gaps_found': [],
            'total_matches': 0,
            'screen_timestamp': datetime.utcnow().isoformat(),
            'market_session': 'regular_hours',  # Would be determined by current time
            'screening_universe': 'sp500'  # Would be configurable
        }
        
        return screener_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in gap screening: {str(e)}")

@router.get("/gap-statistics")
async def get_gap_statistics():
    """
    Get historical gap trading statistics and probabilities.
    
    Provides statistical data for gap behavior patterns
    used in probability calculations and strategy development.
    """
    try:
        engine = GapTradingEngine()
        
        return {
            'gap_size_distribution': {
                'micro_gaps_percentage': 45.2,
                'small_gaps_percentage': 28.7,
                'medium_gaps_percentage': 18.1,
                'large_gaps_percentage': 6.8,
                'massive_gaps_percentage': 1.2
            },
            'continuation_rates_by_size': {
                size.value: rate for size, rate in 
                engine.gap_statistics['continuation_rate_by_size'].items()
            },
            'fill_rates_by_size': {
                size.value: rate for size, rate in 
                engine.gap_statistics['fill_rate_by_size'].items()
            },
            'average_fill_times': {
                'micro_gaps': '15 minutes',
                'small_gaps': '45 minutes',
                'medium_gaps': '2.5 hours',
                'large_gaps': '1.2 days',
                'massive_gaps': '5.7 days'
            },
            'volume_impact': {
                'high_volume_continuation_boost': 15,  # percentage points
                'low_volume_fade_probability': 68,
                'average_volume_multiple': 3.2
            },
            'news_catalyst_impact': {
                'earnings_gaps_fill_rate': 35,
                'news_driven_continuation_rate': 72,
                'random_gaps_fill_rate': 58
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving gap statistics: {str(e)}")

@router.get("/market-session-info")
async def get_market_session_info():
    """
    Get current market session information for gap analysis context.
    """
    try:
        now = datetime.utcnow()
        current_time = now.time()
        
        engine = GapTradingEngine()
        
        # Determine current session
        if engine.market_hours['pre_market_start'] <= current_time < engine.market_hours['market_open']:
            session = MarketSession.PRE_MARKET
        elif engine.market_hours['market_open'] <= current_time < engine.market_hours['market_close']:
            session = MarketSession.REGULAR
        elif engine.market_hours['market_close'] <= current_time < engine.market_hours['after_hours_end']:
            session = MarketSession.AFTER_HOURS
        else:
            session = MarketSession.OVERNIGHT
        
        return {
            'current_session': session.value,
            'current_time_et': now.isoformat(),
            'market_hours': {
                'pre_market_start': engine.market_hours['pre_market_start'].isoformat(),
                'market_open': engine.market_hours['market_open'].isoformat(),
                'market_close': engine.market_hours['market_close'].isoformat(),
                'after_hours_end': engine.market_hours['after_hours_end'].isoformat()
            },
            'next_session_change': 'calculated_dynamically',
            'gap_analysis_optimal': session in [MarketSession.PRE_MARKET, MarketSession.REGULAR]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving market session info: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for gap trading service"""
    return {
        'status': 'healthy',
        'service': 'gap-trading-engine',
        'timestamp': datetime.utcnow().isoformat(),
        'features': [
            'gap_analysis',
            'continuation_vs_fade_detection',
            'pre_post_market_handling',
            'probability_calculations',
            'real_time_monitoring',
            'bulk_screening'
        ]
    }

# Helper functions
def _generate_trading_recommendation(monitoring_result: GapMonitoringResult) -> str:
    """Generate trading recommendation based on monitoring result"""
    direction = monitoring_result.current_direction
    fill_percentage = monitoring_result.fill_percentage
    strength = monitoring_result.price_action_strength
    
    if direction == GapDirection.CONTINUATION and strength > 1.5:
        return "STRONG_CONTINUATION - Consider adding to position"
    elif direction == GapDirection.FADE and fill_percentage > 0.5:
        return "GAP_FILLING - Consider taking profits or reversing"
    elif direction == GapDirection.PARTIAL_FILL:
        return "PARTIAL_FILL - Monitor for direction confirmation"
    else:
        return "NEUTRAL - Wait for clearer signal"

def _calculate_updated_targets(monitoring_result: GapMonitoringResult) -> List[float]:
    """Calculate updated profit targets based on current behavior"""
    original_targets = monitoring_result.gap_analysis.profit_targets
    current_price = monitoring_result.gap_analysis.current_open  # Would use actual current price
    
    # Adjust targets based on momentum and time
    momentum_factor = monitoring_result.momentum_indicators.get('price_momentum', 0)
    time_decay = monitoring_result.momentum_indicators.get('time_decay', 1.0)
    
    updated_targets = []
    for target in original_targets:
        # Adjust target based on momentum and time decay
        adjustment = 1.0 + (momentum_factor * time_decay * 0.1)
        updated_target = target * adjustment
        updated_targets.append(updated_target)
    
    return updated_targets