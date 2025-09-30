"""
FastAPI endpoints for Event-Driven Strategy Analysis

Provides REST API endpoints for:
- CAR (Cumulative Abnormal Return) analysis
- Event regime identification
- Market microstructure analysis
- Integrated event-microstructure studies
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

from ..services.car_analysis import (
    CARAnalyzer, EventRegimeIdentifier, EventData, EventType, Sector, CARResults,
    create_car_analyzer, create_regime_identifier
)
from ..services.market_microstructure import (
    MicrostructureAnalyzer, EventMicrostructureIntegrator, OrderFlowData,
    LiquidityMetrics, VolumeProfile, MicrostructureRegime,
    create_microstructure_analyzer, create_event_microstructure_integrator
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/event-analysis", tags=["Event Analysis"])

# Pydantic models for API
class EventDataRequest(BaseModel):
    symbol: str
    event_type: str
    event_date: datetime
    sector: Optional[str] = None
    event_magnitude: Optional[float] = None
    pre_event_volume: Optional[float] = None
    event_description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CARAnalysisRequest(BaseModel):
    events: List[EventDataRequest]
    event_type: str
    sector: Optional[str] = None
    start_date: datetime
    end_date: datetime
    symbols: Optional[List[str]] = None

class CARAnalysisResponse(BaseModel):
    event_type: str
    sector: Optional[str]
    optimal_holding_period: int
    expected_return: float
    return_volatility: float
    skewness: float
    kurtosis: float
    sharpe_ratio: float
    hit_rate: float
    statistical_significance: Dict[str, float]
    regime_parameters: Dict[str, float]
    car_values: List[float]
    profit_distribution: Dict[str, float]

class OrderFlowRequest(BaseModel):
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    trade_direction: Optional[int] = None

class LiquidityAnalysisRequest(BaseModel):
    order_flow_data: List[OrderFlowRequest]
    trade_data: List[Dict[str, Any]]
    symbol: str
    analysis_window_minutes: int = 5

class LiquidityAnalysisResponse(BaseModel):
    bid_ask_spread: float
    spread_bps: float
    effective_spread: float
    price_impact: float
    market_depth: float
    order_imbalance: float
    liquidity_regime: str
    execution_difficulty: str
    optimal_execution_strategy: Dict[str, Any]

class EventMicrostructureAnalysisRequest(BaseModel):
    event_type: str
    sector: Optional[str] = None
    symbol: str
    event_date: datetime
    pre_event_data: List[OrderFlowRequest]
    post_event_data: List[OrderFlowRequest]
    trade_data: List[Dict[str, Any]]

# Dependency injection
async def get_car_analyzer():
    return await create_car_analyzer()

async def get_regime_identifier():
    return await create_regime_identifier()

async def get_microstructure_analyzer():
    return await create_microstructure_analyzer()

async def get_event_microstructure_integrator():
    return await create_event_microstructure_integrator()

# API Endpoints
@router.post("/car-analysis", response_model=CARAnalysisResponse)
async def perform_car_analysis(
    request: CARAnalysisRequest,
    car_analyzer: CARAnalyzer = Depends(get_car_analyzer)
):
    """
    Perform Cumulative Abnormal Return (CAR) analysis for specific event type/sector
    
    This endpoint analyzes historical events to determine optimal holding periods,
    expected returns, and risk characteristics for event-driven trading strategies.
    """
    try:
        # Convert request events to EventData objects
        events = []
        for event_req in request.events:
            event_data = EventData(
                symbol=event_req.symbol,
                event_type=EventType(event_req.event_type),
                event_date=event_req.event_date,
                sector=Sector(event_req.sector) if event_req.sector else None,
                event_magnitude=event_req.event_magnitude,
                pre_event_volume=event_req.pre_event_volume,
                event_description=event_req.event_description,
                metadata=event_req.metadata
            )
            events.append(event_data)
        
        # TODO: Fetch price and market data from database
        # This is a placeholder - in production, you'd fetch from your data store
        price_data = pd.DataFrame()  # Replace with actual data fetching
        market_data = pd.DataFrame()  # Replace with actual market index data
        
        if price_data.empty or market_data.empty:
            raise HTTPException(
                status_code=400,
                detail="Insufficient price or market data for analysis period"
            )
        
        # Perform CAR analysis
        results = await car_analyzer.calculate_car(
            events=events,
            price_data=price_data,
            market_data=market_data,
            event_type=EventType(request.event_type),
            sector=Sector(request.sector) if request.sector else None
        )
        
        return CARAnalysisResponse(
            event_type=request.event_type,
            sector=request.sector,
            optimal_holding_period=results.optimal_holding_period,
            expected_return=results.expected_return,
            return_volatility=results.return_volatility,
            skewness=results.skewness,
            kurtosis=results.kurtosis,
            sharpe_ratio=results.sharpe_ratio,
            hit_rate=results.hit_rate,
            statistical_significance=results.statistical_significance,
            regime_parameters=results.regime_parameters,
            car_values=results.car_values.tolist(),
            profit_distribution=results.profit_distribution
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"CAR analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during CAR analysis")

@router.get("/regime-parameters/{event_type}")
async def get_regime_parameters(
    event_type: str,
    sector: Optional[str] = Query(None),
    regime_identifier: EventRegimeIdentifier = Depends(get_regime_identifier)
):
    """
    Get pre-calculated regime parameters for specific event type/sector combination
    
    Returns optimal trading parameters derived from historical CAR analysis.
    """
    try:
        event_enum = EventType(event_type)
        sector_enum = Sector(sector) if sector else None
        
        parameters = regime_identifier.get_regime_parameters(event_enum, sector_enum)
        
        if parameters is None:
            raise HTTPException(
                status_code=404,
                detail=f"No regime parameters found for {event_type}/{sector}"
            )
        
        return {
            "event_type": event_type,
            "sector": sector,
            "parameters": parameters
        }
        
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type '{event_type}' or sector '{sector}'"
        )

@router.post("/liquidity-analysis", response_model=LiquidityAnalysisResponse)
async def analyze_liquidity(
    request: LiquidityAnalysisRequest,
    analyzer: MicrostructureAnalyzer = Depends(get_microstructure_analyzer)
):
    """
    Analyze market liquidity and microstructure conditions
    
    Provides comprehensive liquidity metrics and optimal execution strategies.
    """
    try:
        # Convert request data to internal format
        order_flow_data = [
            OrderFlowData(
                timestamp=data.timestamp,
                symbol=data.symbol,
                bid_price=data.bid_price,
                ask_price=data.ask_price,
                bid_size=data.bid_size,
                ask_size=data.ask_size,
                last_price=data.last_price,
                volume=data.volume,
                trade_direction=data.trade_direction
            )
            for data in request.order_flow_data
        ]
        
        # Convert trade data to DataFrame
        trade_df = pd.DataFrame(request.trade_data)
        
        # Calculate liquidity metrics
        liquidity_metrics = await analyzer.calculate_liquidity_metrics(
            order_flow_data, trade_df
        )
        
        # Analyze volume profile if trade data available
        volume_profile = None
        if not trade_df.empty and 'price' in trade_df.columns and 'volume' in trade_df.columns:
            volume_profile = await analyzer.analyze_volume_profile(trade_df)
        else:
            # Create dummy volume profile for regime identification
            volume_profile = VolumeProfile(
                price_levels=np.array([0]), volume_by_price=np.array([0]),
                poc=0, value_area_high=0, value_area_low=0,
                value_area_volume_pct=0, developing_poc=0,
                volume_imbalance_areas=[]
            )
        
        # Identify microstructure regime
        regime = analyzer.identify_microstructure_regime(
            liquidity_metrics, volume_profile, {"volatility": 0.2}
        )
        
        # Assess execution difficulty
        if liquidity_metrics.spread_bps > 20:
            execution_difficulty = "high"
        elif liquidity_metrics.spread_bps > 10:
            execution_difficulty = "medium"
        else:
            execution_difficulty = "low"
        
        return LiquidityAnalysisResponse(
            bid_ask_spread=liquidity_metrics.bid_ask_spread,
            spread_bps=liquidity_metrics.spread_bps,
            effective_spread=liquidity_metrics.effective_spread,
            price_impact=liquidity_metrics.price_impact,
            market_depth=liquidity_metrics.market_depth,
            order_imbalance=liquidity_metrics.order_imbalance,
            liquidity_regime=regime.liquidity_level,
            execution_difficulty=execution_difficulty,
            optimal_execution_strategy=regime.optimal_execution_strategy
        )
        
    except Exception as e:
        logger.error(f"Liquidity analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during liquidity analysis")

@router.post("/event-microstructure-impact")
async def analyze_event_microstructure_impact(
    request: EventMicrostructureAnalysisRequest,
    integrator: EventMicrostructureIntegrator = Depends(get_event_microstructure_integrator)
):
    """
    Analyze how events impact market microstructure
    
    Studies the changes in liquidity, spreads, and execution conditions around events.
    """
    try:
        # Convert request data
        pre_event_data = [
            OrderFlowData(
                timestamp=data.timestamp,
                symbol=data.symbol,
                bid_price=data.bid_price,
                ask_price=data.ask_price,
                bid_size=data.bid_size,
                ask_size=data.ask_size,
                last_price=data.last_price,
                volume=data.volume,
                trade_direction=data.trade_direction
            )
            for data in request.pre_event_data
        ]
        
        post_event_data = [
            OrderFlowData(
                timestamp=data.timestamp,
                symbol=data.symbol,
                bid_price=data.bid_price,
                ask_price=data.ask_price,
                bid_size=data.bid_size,
                ask_size=data.ask_size,
                last_price=data.last_price,
                volume=data.volume,
                trade_direction=data.trade_direction
            )
            for data in request.post_event_data
        ]
        
        trade_df = pd.DataFrame(request.trade_data)
        
        # Perform integrated analysis
        impact_analysis = await integrator.analyze_event_microstructure_impact(
            event_type=EventType(request.event_type),
            sector=Sector(request.sector) if request.sector else None,
            pre_event_data=pre_event_data,
            post_event_data=post_event_data,
            event_trade_data=trade_df
        )
        
        return impact_analysis
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Event microstructure analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during event-microstructure analysis")

@router.get("/event-types")
async def get_supported_event_types():
    """Get list of supported event types for analysis"""
    return {
        "event_types": [event_type.value for event_type in EventType],
        "sectors": [sector.value for sector in Sector]
    }

@router.get("/health")
async def health_check():
    """Health check endpoint for event analysis service"""
    return {"status": "healthy", "service": "event_analysis"}

@router.post("/batch-regime-analysis")
async def perform_batch_regime_analysis(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    symbols: Optional[List[str]] = Query(None),
    regime_identifier: EventRegimeIdentifier = Depends(get_regime_identifier)
):
    """
    Perform batch regime analysis across all event types and sectors
    
    This is typically run periodically to update regime parameters based on
    the latest historical data.
    """
    try:
        # TODO: Fetch historical events and price data from database
        # This would typically query your event database and market data
        historical_events = []  # Replace with actual data fetching
        price_data = pd.DataFrame()  # Replace with actual price data
        market_data = pd.DataFrame()  # Replace with actual market data
        
        if not historical_events:
            raise HTTPException(
                status_code=400,
                detail="No historical events found for the specified period"
            )
        
        # Perform regime identification
        regimes = await regime_identifier.identify_regimes_by_event_sector(
            historical_events, price_data, market_data
        )
        
        # Update cache
        await regime_identifier.update_regime_cache(regimes)
        
        # Format response
        regime_summary = {}
        for (event_type, sector), results in regimes.items():
            key = f"{event_type.value}_{sector.value if sector else 'all'}"
            regime_summary[key] = {
                "optimal_holding_period": results.optimal_holding_period,
                "expected_return": results.expected_return,
                "sharpe_ratio": results.sharpe_ratio,
                "hit_rate": results.hit_rate
            }
        
        return {
            "analysis_period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "regimes_analyzed": len(regimes),
            "regime_summary": regime_summary,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Batch regime analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch analysis")