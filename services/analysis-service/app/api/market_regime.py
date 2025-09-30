"""
Market Regime Analysis API endpoints for event trade execution filtering.
Provides regime-based filtering to ensure trades are only executed in favorable conditions.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from ..services.market_regime_filter import (
    MarketRegimeFilter,
    RegimeAnalysis,
    TradeExecutionDecision,
    MarketRegime,
    RegimeFavorability,
    EventType,
    RegimeIndicators
)

router = APIRouter(prefix="/market-regime", tags=["Market Regime"])

# Request/Response Models
class MarketDataInput(BaseModel):
    """Market data input for regime analysis"""
    vix: Optional[float] = 20.0
    vix9d_vix_ratio: Optional[float] = 1.0
    spy_price: Optional[float] = 400.0
    spy_20ma: Optional[float] = 395.0
    spy_50ma: Optional[float] = 390.0
    spy_200ma: Optional[float] = 380.0
    sector_rotation_score: Optional[float] = 0.5
    credit_spreads: Optional[float] = 100.0
    advance_decline_ratio: Optional[float] = 1.0
    new_highs_lows_ratio: Optional[float] = 1.0
    volume_breadth: Optional[float] = 0.5
    sector_breadth: Optional[float] = 0.5
    liquidity_score: Optional[float] = 0.5
    correlation_level: Optional[float] = 0.5

class RegimeAnalysisRequest(BaseModel):
    """Request for market regime analysis"""
    market_data: MarketDataInput
    lookback_periods: Optional[Dict[str, int]] = Field(
        default={'short_term': 5, 'medium_term': 20, 'long_term': 60}
    )
    include_forecasting: bool = Field(default=True, description="Include regime forecasting")

class TradeExecutionRequest(BaseModel):
    """Request for trade execution evaluation"""
    symbol: str = Field(..., description="Stock symbol")
    event_type: str = Field(..., description="Event type (earnings, fda_approval, etc.)")
    event_details: Dict[str, Any] = Field(..., description="Event details")
    market_data: MarketDataInput
    risk_tolerance: str = Field(default='moderate', description="Risk tolerance (conservative/moderate/aggressive)")
    force_analysis: bool = Field(default=False, description="Force analysis even if regime is unfavorable")

class BulkTradeEvaluationRequest(BaseModel):
    """Request for bulk trade evaluation"""
    trades: List[Dict[str, Any]] = Field(..., description="List of potential trades")
    market_data: MarketDataInput
    risk_tolerance: str = Field(default='moderate')
    max_approved_trades: Optional[int] = Field(None, description="Maximum number of trades to approve")

class RegimeBacktestRequest(BaseModel):
    """Request for regime-based backtesting"""
    start_date: datetime
    end_date: datetime
    event_types: List[str]
    risk_tolerance: str = Field(default='moderate')
    regime_filter_enabled: bool = Field(default=True)

class RegimeResponse(BaseModel):
    """Response for regime analysis"""
    timestamp: str
    primary_regime: str
    secondary_regimes: List[str]
    regime_strength: float
    regime_duration_days: int
    transition_probability: float
    volatility_percentile: float
    trend_strength: float
    mean_reversion_tendency: float
    breakout_potential: float
    tail_risk: float
    correlation_risk: float
    liquidity_risk: float
    overall_favorability: str
    event_favorability: Dict[str, str]
    regime_indicators: Dict[str, float]

class TradeExecutionResponse(BaseModel):
    """Response for trade execution evaluation"""
    symbol: str
    event_type: str
    execution_approved: bool
    favorability_score: float
    risk_adjustment_factor: float
    position_size_modifier: float
    approval_reasons: List[str]
    rejection_reasons: List[str]
    risk_mitigation_required: List[str]
    recommended_entry_timing: Optional[str]
    regime_stop_loss: Optional[float]
    profit_target_adjustment: Optional[float]
    regime_context: Dict[str, Any]

class BulkEvaluationResponse(BaseModel):
    """Response for bulk trade evaluation"""
    total_trades_evaluated: int
    trades_approved: int
    approval_rate: float
    approved_trades: List[Dict[str, Any]]
    rejected_trades: List[Dict[str, Any]]
    regime_summary: Dict[str, Any]
    risk_budget_utilization: float

# Dependency injection
async def get_regime_filter() -> MarketRegimeFilter:
    """Get market regime filter instance"""
    return MarketRegimeFilter()

# API Endpoints
@router.post("/analyze-regime", response_model=RegimeResponse)
async def analyze_regime(
    request: RegimeAnalysisRequest,
    filter_engine: MarketRegimeFilter = Depends(get_regime_filter)
):
    """
    Analyze current market regime conditions.
    
    Provides comprehensive market regime analysis including primary/secondary
    regimes, favorability assessments, and risk metrics.
    """
    try:
        # Convert input to market data dictionary
        market_data = request.market_data.dict()
        
        # Add derived indicators
        market_data['trend_score'] = _calculate_trend_score(market_data)
        market_data['risk_appetite'] = _calculate_risk_appetite_score(market_data)
        market_data['vol_clustering'] = _estimate_vol_clustering(market_data)
        market_data['regime_duration_days'] = 10  # Would be calculated from historical data
        
        # Perform regime analysis
        regime_analysis = await filter_engine.analyze_market_regime(
            market_data=market_data,
            lookback_periods=request.lookback_periods
        )
        
        # Format indicators for response
        regime_indicators = {
            'vix_level': market_data['vix'],
            'vix_term_structure': market_data['vix9d_vix_ratio'],
            'market_trend': market_data['trend_score'],
            'sector_rotation': market_data['sector_rotation_score'],
            'credit_spreads': market_data['credit_spreads'],
            'liquidity_conditions': market_data['liquidity_score'],
            'correlation_regime': market_data['correlation_level']
        }
        
        return RegimeResponse(
            timestamp=regime_analysis.timestamp.isoformat(),
            primary_regime=regime_analysis.primary_regime.value,
            secondary_regimes=[regime.value for regime in regime_analysis.secondary_regimes],
            regime_strength=regime_analysis.regime_strength,
            regime_duration_days=regime_analysis.regime_duration.days,
            transition_probability=regime_analysis.regime_transition_probability,
            volatility_percentile=regime_analysis.volatility_percentile,
            trend_strength=regime_analysis.trend_strength,
            mean_reversion_tendency=regime_analysis.mean_reversion_tendency,
            breakout_potential=regime_analysis.breakout_potential,
            tail_risk=regime_analysis.tail_risk,
            correlation_risk=regime_analysis.correlation_risk,
            liquidity_risk=regime_analysis.liquidity_risk,
            overall_favorability=regime_analysis.overall_favorability.value,
            event_favorability={
                event_type.value: favorability.value
                for event_type, favorability in regime_analysis.event_specific_favorability.items()
            },
            regime_indicators=regime_indicators
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing market regime: {str(e)}")

@router.post("/evaluate-trade-execution", response_model=TradeExecutionResponse)
async def evaluate_trade_execution(
    request: TradeExecutionRequest,
    filter_engine: MarketRegimeFilter = Depends(get_regime_filter)
):
    """
    Evaluate whether to execute an event trade based on market regime.
    
    Analyzes market conditions and determines if the regime is favorable
    for executing the specified event trade.
    """
    try:
        # Convert inputs
        market_data = request.market_data.dict()
        market_data['trend_score'] = _calculate_trend_score(market_data)
        market_data['risk_appetite'] = _calculate_risk_appetite_score(market_data)
        market_data['vol_clustering'] = _estimate_vol_clustering(market_data)
        market_data['regime_duration_days'] = 10
        
        # Analyze regime
        regime_analysis = await filter_engine.analyze_market_regime(market_data=market_data)
        
        # Convert event type
        try:
            event_type_enum = EventType(request.event_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {request.event_type}")
        
        # Evaluate trade execution
        execution_decision = await filter_engine.evaluate_trade_execution(
            symbol=request.symbol,
            event_type=event_type_enum,
            event_details=request.event_details,
            regime_analysis=regime_analysis,
            risk_tolerance=request.risk_tolerance
        )
        
        # Override rejection if forced
        if request.force_analysis and not execution_decision.execution_approved:
            execution_decision.execution_approved = True
            execution_decision.approval_reasons.append("FORCED EXECUTION - Regime filter overridden")
            execution_decision.position_size_modifier *= 0.5  # Reduce size for forced trades
        
        # Prepare regime context
        regime_context = {
            'primary_regime': regime_analysis.primary_regime.value,
            'regime_strength': regime_analysis.regime_strength,
            'overall_favorability': regime_analysis.overall_favorability.value,
            'event_specific_favorability': regime_analysis.event_specific_favorability[event_type_enum].value,
            'volatility_percentile': regime_analysis.volatility_percentile,
            'tail_risk': regime_analysis.tail_risk
        }
        
        return TradeExecutionResponse(
            symbol=execution_decision.symbol,
            event_type=execution_decision.event_type.value,
            execution_approved=execution_decision.execution_approved,
            favorability_score=execution_decision.favorability_score,
            risk_adjustment_factor=execution_decision.risk_adjustment_factor,
            position_size_modifier=execution_decision.position_size_modifier,
            approval_reasons=execution_decision.approval_reasons,
            rejection_reasons=execution_decision.rejection_reasons,
            risk_mitigation_required=execution_decision.risk_mitigation_required,
            recommended_entry_timing=execution_decision.recommended_entry_timing,
            regime_stop_loss=execution_decision.regime_stop_loss,
            profit_target_adjustment=execution_decision.regime_profit_target_adjustment,
            regime_context=regime_context
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating trade execution: {str(e)}")

@router.post("/bulk-evaluate-trades", response_model=BulkEvaluationResponse)
async def bulk_evaluate_trades(
    request: BulkTradeEvaluationRequest,
    filter_engine: MarketRegimeFilter = Depends(get_regime_filter)
):
    """
    Evaluate multiple potential trades for execution approval.
    
    Efficiently processes multiple trade opportunities and ranks them
    by regime favorability and risk-adjusted return potential.
    """
    try:
        # Analyze regime once for all trades
        market_data = request.market_data.dict()
        market_data['trend_score'] = _calculate_trend_score(market_data)
        market_data['risk_appetite'] = _calculate_risk_appetite_score(market_data)
        market_data['vol_clustering'] = _estimate_vol_clustering(market_data)
        market_data['regime_duration_days'] = 10
        
        regime_analysis = await filter_engine.analyze_market_regime(market_data=market_data)
        
        approved_trades = []
        rejected_trades = []
        total_risk_budget = 0.0
        
        # Evaluate each trade
        for trade_data in request.trades:
            try:
                symbol = trade_data.get('symbol', 'UNKNOWN')
                event_type_str = trade_data.get('event_type', 'earnings')
                event_details = trade_data.get('event_details', {})
                
                # Convert event type
                try:
                    event_type_enum = EventType(event_type_str.lower())
                except ValueError:
                    rejected_trades.append({
                        'symbol': symbol,
                        'event_type': event_type_str,
                        'rejection_reason': f'Invalid event type: {event_type_str}',
                        'favorability_score': 0.0
                    })
                    continue
                
                # Evaluate execution
                execution_decision = await filter_engine.evaluate_trade_execution(
                    symbol=symbol,
                    event_type=event_type_enum,
                    event_details=event_details,
                    regime_analysis=regime_analysis,
                    risk_tolerance=request.risk_tolerance
                )
                
                trade_result = {
                    'symbol': symbol,
                    'event_type': event_type_str,
                    'favorability_score': execution_decision.favorability_score,
                    'position_size_modifier': execution_decision.position_size_modifier,
                    'risk_adjustment_factor': execution_decision.risk_adjustment_factor,
                    'approval_reasons': execution_decision.approval_reasons,
                    'rejection_reasons': execution_decision.rejection_reasons
                }
                
                if execution_decision.execution_approved:
                    total_risk_budget += execution_decision.position_size_modifier
                    approved_trades.append(trade_result)
                else:
                    rejected_trades.append(trade_result)
                    
            except Exception as e:
                rejected_trades.append({
                    'symbol': trade_data.get('symbol', 'UNKNOWN'),
                    'event_type': trade_data.get('event_type', 'unknown'),
                    'rejection_reason': f'Evaluation error: {str(e)}',
                    'favorability_score': 0.0
                })
        
        # Sort approved trades by favorability score
        approved_trades.sort(key=lambda x: x['favorability_score'], reverse=True)
        
        # Apply maximum trade limit if specified
        if request.max_approved_trades and len(approved_trades) > request.max_approved_trades:
            excess_trades = approved_trades[request.max_approved_trades:]
            approved_trades = approved_trades[:request.max_approved_trades]
            
            # Move excess to rejected with explanation
            for trade in excess_trades:
                trade['rejection_reason'] = 'Exceeded maximum approved trades limit'
                rejected_trades.append(trade)
        
        # Calculate metrics
        total_evaluated = len(request.trades)
        total_approved = len(approved_trades)
        approval_rate = total_approved / total_evaluated if total_evaluated > 0 else 0.0
        
        # Risk budget utilization (assume max budget of 10.0)
        max_risk_budget = 10.0
        risk_utilization = min(1.0, total_risk_budget / max_risk_budget)
        
        # Regime summary
        regime_summary = {
            'primary_regime': regime_analysis.primary_regime.value,
            'overall_favorability': regime_analysis.overall_favorability.value,
            'regime_strength': regime_analysis.regime_strength,
            'market_stress_level': (regime_analysis.tail_risk + regime_analysis.correlation_risk) / 2,
            'recommended_position_sizing': 'reduced' if regime_analysis.volatility_percentile > 0.7 else 'normal'
        }
        
        return BulkEvaluationResponse(
            total_trades_evaluated=total_evaluated,
            trades_approved=total_approved,
            approval_rate=approval_rate,
            approved_trades=approved_trades,
            rejected_trades=rejected_trades,
            regime_summary=regime_summary,
            risk_budget_utilization=risk_utilization
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in bulk trade evaluation: {str(e)}")

@router.get("/current-regime-status")
async def get_current_regime_status(
    filter_engine: MarketRegimeFilter = Depends(get_regime_filter)
):
    """
    Get current market regime status for quick reference.
    
    Provides a simplified view of current market conditions
    for rapid decision making.
    """
    try:
        # Use default market data (in production, this would fetch real-time data)
        market_data = {
            'vix': 22.5,
            'vix9d_vix_ratio': 0.95,
            'trend_score': 0.3,
            'risk_appetite': 0.6,
            'vol_clustering': 0.4,
            'sector_rotation_score': 0.5,
            'credit_spreads': 120.0,
            'liquidity_score': 0.7,
            'correlation_level': 0.6,
            'regime_duration_days': 15
        }
        
        regime_analysis = await filter_engine.analyze_market_regime(market_data=market_data)
        
        # Simplified status
        trading_recommendation = _get_trading_recommendation(regime_analysis)
        risk_level = _assess_risk_level(regime_analysis)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'regime_status': regime_analysis.primary_regime.value,
            'favorability': regime_analysis.overall_favorability.value,
            'trading_recommendation': trading_recommendation,
            'risk_level': risk_level,
            'vix_level': market_data['vix'],
            'market_trend': market_data['trend_score'],
            'liquidity_score': market_data['liquidity_score'],
            'key_risks': _identify_key_risks(regime_analysis),
            'optimal_event_types': _get_optimal_event_types(regime_analysis)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting regime status: {str(e)}")

@router.get("/regime-history")
async def get_regime_history(
    days_back: int = Query(30, description="Number of days to look back"),
    filter_engine: MarketRegimeFilter = Depends(get_regime_filter)
):
    """
    Get historical regime analysis for backtesting and research.
    
    Provides historical regime data for strategy development
    and performance attribution analysis.
    """
    try:
        # In a real implementation, this would query historical data
        # For now, return simulated historical data
        
        history = []
        for i in range(days_back):
            date = datetime.utcnow() - timedelta(days=i)
            
            # Simulate regime changes
            if i < 10:
                regime = MarketRegime.BULL_MARKET
                favorability = RegimeFavorability.FAVORABLE
            elif i < 20:
                regime = MarketRegime.SIDEWAYS_MARKET
                favorability = RegimeFavorability.NEUTRAL
            else:
                regime = MarketRegime.HIGH_VOLATILITY
                favorability = RegimeFavorability.UNFAVORABLE
            
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'regime': regime.value,
                'favorability': favorability.value,
                'vix_level': 20.0 + np.random.normal(0, 5),
                'trade_approval_rate': 0.7 if favorability in [RegimeFavorability.FAVORABLE, RegimeFavorability.HIGHLY_FAVORABLE] else 0.3
            })
        
        return {
            'period': f'{days_back} days',
            'regime_changes': len(set(item['regime'] for item in history)),
            'average_approval_rate': np.mean([item['trade_approval_rate'] for item in history]),
            'history': list(reversed(history))  # Most recent first
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving regime history: {str(e)}")

@router.get("/event-type-favorability")
async def get_event_type_favorability(
    filter_engine: MarketRegimeFilter = Depends(get_regime_filter)
):
    """
    Get current favorability ratings for all event types.
    
    Shows which types of events are most favorable to trade
    in the current market regime.
    """
    try:
        # Use current market conditions
        market_data = {
            'vix': 20.0,
            'trend_score': 0.4,
            'risk_appetite': 0.6,
            'vol_clustering': 0.3,
            'regime_duration_days': 12
        }
        
        regime_analysis = await filter_engine.analyze_market_regime(market_data=market_data)
        
        # Format favorability by event type
        event_favorability = []
        for event_type, favorability in regime_analysis.event_specific_favorability.items():
            favorability_score = _favorability_to_score(favorability)
            
            event_favorability.append({
                'event_type': event_type.value,
                'favorability': favorability.value,
                'favorability_score': favorability_score,
                'recommended': favorability_score >= 0.6,
                'description': _get_event_description(event_type)
            })
        
        # Sort by favorability score
        event_favorability.sort(key=lambda x: x['favorability_score'], reverse=True)
        
        return {
            'current_regime': regime_analysis.primary_regime.value,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'event_favorability': event_favorability,
            'regime_summary': {
                'strength': regime_analysis.regime_strength,
                'volatility_percentile': regime_analysis.volatility_percentile,
                'tail_risk': regime_analysis.tail_risk
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting event favorability: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for market regime service"""
    return {
        'status': 'healthy',
        'service': 'market-regime-filter',
        'timestamp': datetime.utcnow().isoformat(),
        'features': [
            'regime_analysis',
            'trade_execution_filtering',
            'favorability_assessment',
            'risk_adjustment',
            'bulk_evaluation',
            'historical_analysis'
        ]
    }

# Helper functions
def _calculate_trend_score(market_data: Dict) -> float:
    """Calculate trend score from market data"""
    spy_price = market_data.get('spy_price', 400)
    spy_20ma = market_data.get('spy_20ma', 395)
    spy_50ma = market_data.get('spy_50ma', 390)
    spy_200ma = market_data.get('spy_200ma', 380)
    
    # Simple trend calculation
    short_trend = (spy_price - spy_20ma) / spy_20ma
    medium_trend = (spy_20ma - spy_50ma) / spy_50ma
    long_trend = (spy_50ma - spy_200ma) / spy_200ma
    
    return (short_trend + medium_trend + long_trend) / 3

def _calculate_risk_appetite_score(market_data: Dict) -> float:
    """Calculate risk appetite from market indicators"""
    vix = market_data.get('vix', 20)
    advance_decline = market_data.get('advance_decline_ratio', 1.0)
    
    # Lower VIX and positive breadth = higher risk appetite
    vix_component = max(0, 1.0 - (vix - 15) / 20)
    breadth_component = min(1.0, advance_decline)
    
    return (vix_component + breadth_component) / 2

def _estimate_vol_clustering(market_data: Dict) -> float:
    """Estimate volatility clustering"""
    vix = market_data.get('vix', 20)
    vix_term_structure = market_data.get('vix9d_vix_ratio', 1.0)
    
    # Inverted term structure suggests clustering
    clustering = 1.0 - vix_term_structure if vix_term_structure < 1.0 else 0.0
    return max(0, min(1, clustering + (vix - 20) / 30))

def _get_trading_recommendation(regime_analysis: RegimeAnalysis) -> str:
    """Get overall trading recommendation"""
    if regime_analysis.overall_favorability == RegimeFavorability.HIGHLY_FAVORABLE:
        return "AGGRESSIVE_TRADING"
    elif regime_analysis.overall_favorability == RegimeFavorability.FAVORABLE:
        return "NORMAL_TRADING"
    elif regime_analysis.overall_favorability == RegimeFavorability.NEUTRAL:
        return "SELECTIVE_TRADING"
    elif regime_analysis.overall_favorability == RegimeFavorability.UNFAVORABLE:
        return "DEFENSIVE_MODE"
    else:
        return "AVOID_TRADING"

def _assess_risk_level(regime_analysis: RegimeAnalysis) -> str:
    """Assess overall risk level"""
    risk_score = (regime_analysis.tail_risk + regime_analysis.correlation_risk + 
                  regime_analysis.liquidity_risk) / 3
    
    if risk_score > 0.7:
        return "HIGH"
    elif risk_score > 0.4:
        return "MODERATE"
    else:
        return "LOW"

def _identify_key_risks(regime_analysis: RegimeAnalysis) -> List[str]:
    """Identify key market risks"""
    risks = []
    
    if regime_analysis.tail_risk > 0.6:
        risks.append("High tail risk - potential for extreme moves")
    if regime_analysis.correlation_risk > 0.7:
        risks.append("High correlation risk - diversification breakdown")
    if regime_analysis.liquidity_risk > 0.5:
        risks.append("Liquidity concerns - wider spreads and slippage")
    if regime_analysis.volatility_percentile > 0.8:
        risks.append("Elevated volatility - increased position risk")
    if regime_analysis.regime_transition_probability > 0.7:
        risks.append("Regime change likely - strategy adjustments needed")
    
    return risks if risks else ["No major risks identified"]

def _get_optimal_event_types(regime_analysis: RegimeAnalysis) -> List[str]:
    """Get optimal event types for current regime"""
    optimal = []
    
    for event_type, favorability in regime_analysis.event_specific_favorability.items():
        if favorability in [RegimeFavorability.FAVORABLE, RegimeFavorability.HIGHLY_FAVORABLE]:
            optimal.append(event_type.value)
    
    return optimal if optimal else ["None - defensive positioning recommended"]

def _favorability_to_score(favorability: RegimeFavorability) -> float:
    """Convert favorability enum to numerical score"""
    scores = {
        RegimeFavorability.HIGHLY_UNFAVORABLE: 0.1,
        RegimeFavorability.UNFAVORABLE: 0.3,
        RegimeFavorability.NEUTRAL: 0.5,
        RegimeFavorability.FAVORABLE: 0.7,
        RegimeFavorability.HIGHLY_FAVORABLE: 0.9
    }
    return scores[favorability]

def _get_event_description(event_type: EventType) -> str:
    """Get description for event type"""
    descriptions = {
        EventType.EARNINGS: "Quarterly earnings announcements",
        EventType.FDA_APPROVAL: "FDA drug/device approvals",
        EventType.MERGER_ACQUISITION: "M&A announcements",
        EventType.PRODUCT_LAUNCH: "New product launches",
        EventType.REGULATORY: "Regulatory decisions",
        EventType.ANALYST_UPGRADE: "Analyst upgrades",
        EventType.ANALYST_DOWNGRADE: "Analyst downgrades",
        EventType.GUIDANCE: "Management guidance updates",
        EventType.SPINOFF: "Corporate spinoffs",
        EventType.DIVIDEND: "Dividend announcements"
    }
    return descriptions.get(event_type, "Unknown event type")