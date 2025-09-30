"""
API endpoints for Execution Realism Framework.

Provides REST API access to realistic trading execution simulation:
- Latency modeling and network delays
- Order book simulation and queue position
- Market halts and trading session management
- Slippage and market impact modeling
- Transaction cost analysis
- Realistic backtesting with execution constraints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from services.execution_realism import (
    ExecutionSimulator, RealisticBacktester, LatencyProfile, LiquidityProfile,
    Order, OrderType, OrderSide, MarketData, LatencySimulator, MarketHaltSimulator,
    SlippageModel, TransactionCostModel
)

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models
class OrderRequest(BaseModel):
    """Request model for order submission."""
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side: 'buy' or 'sell'")
    order_type: str = Field(..., description="Order type: 'market', 'limit', 'stop', 'stop_limit'")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, description="Limit price (for limit orders)")
    stop_price: Optional[float] = Field(None, description="Stop price (for stop orders)")
    
    @validator('side')
    def validate_side(cls, v):
        if v.lower() not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        return v.lower()
    
    @validator('order_type')
    def validate_order_type(cls, v):
        allowed_types = ['market', 'limit', 'stop', 'stop_limit']
        if v.lower() not in allowed_types:
            raise ValueError(f"Order type must be one of {allowed_types}")
        return v.lower()


class MarketDataPoint(BaseModel):
    """Market data point for execution simulation."""
    timestamp: datetime
    symbol: str
    bid: float = Field(..., gt=0, description="Best bid price")
    ask: float = Field(..., gt=0, description="Best ask price")
    last_price: float = Field(..., gt=0, description="Last trade price")
    volume: float = Field(..., ge=0, description="Trade volume")
    bid_size: Optional[float] = Field(100, ge=0, description="Bid size")
    ask_size: Optional[float] = Field(100, ge=0, description="Ask size")
    vwap: Optional[float] = Field(None, description="Volume weighted average price")
    
    @validator('ask')
    def validate_spread(cls, v, values):
        if 'bid' in values and v <= values['bid']:
            raise ValueError("Ask price must be greater than bid price")
        return v


class LatencyProfileRequest(BaseModel):
    """Request model for latency profile configuration."""
    market_data_latency_ms: float = Field(2.0, ge=0.1, le=100, description="Market data latency")
    order_latency_ms: float = Field(5.0, ge=0.1, le=100, description="Order submission latency")
    fill_latency_ms: float = Field(3.0, ge=0.1, le=100, description="Fill confirmation latency")
    cancel_latency_ms: float = Field(4.0, ge=0.1, le=100, description="Order cancellation latency")
    jitter_ms: float = Field(1.0, ge=0.0, le=10, description="Latency jitter")


class LiquidityProfileRequest(BaseModel):
    """Request model for liquidity profile configuration."""
    average_bid_ask_spread_bps: float = Field(5.0, ge=0.1, le=1000, description="Average spread in bps")
    average_depth_usd: float = Field(100000.0, ge=1000, le=10000000, description="Average depth per level")
    price_levels: int = Field(5, ge=1, le=20, description="Number of price levels")
    depth_decay_factor: float = Field(0.7, ge=0.1, le=1.0, description="Depth decay factor")
    liquidity_regeneration_rate: float = Field(0.1, ge=0.01, le=1.0, description="Liquidity regeneration rate")


class ExecutionSimulationRequest(BaseModel):
    """Request model for execution simulation."""
    orders: List[OrderRequest] = Field(..., description="Orders to simulate")
    market_data: List[MarketDataPoint] = Field(..., description="Market data stream")
    latency_profile: Optional[LatencyProfileRequest] = Field(None, description="Latency configuration")
    liquidity_profile: Optional[LiquidityProfileRequest] = Field(None, description="Liquidity configuration")
    
    enable_halts: bool = Field(True, description="Enable trading halt simulation")
    enable_slippage: bool = Field(True, description="Enable slippage modeling")
    enable_transaction_costs: bool = Field(True, description="Enable transaction cost calculation")


class BacktestRequest(BaseModel):
    """Request model for realistic backtesting."""
    strategy_signals: List[Dict[str, Any]] = Field(..., description="Strategy signals data")
    market_data: List[MarketDataPoint] = Field(..., description="Historical market data")
    universe: List[str] = Field(..., description="Trading universe symbols")
    initial_capital: float = Field(1000000.0, gt=0, description="Starting capital")
    
    latency_profile: Optional[LatencyProfileRequest] = Field(None, description="Latency configuration")
    liquidity_profile: Optional[LiquidityProfileRequest] = Field(None, description="Liquidity configuration")
    
    start_date: Optional[datetime] = Field(None, description="Backtest start date")
    end_date: Optional[datetime] = Field(None, description="Backtest end date")


class SlippageAnalysisRequest(BaseModel):
    """Request model for slippage analysis."""
    orders: List[OrderRequest] = Field(..., description="Orders to analyze")
    market_data: List[MarketDataPoint] = Field(..., description="Market data")
    average_daily_volume_usd: Optional[float] = Field(1000000.0, description="Average daily volume in USD")


class ExecutionResponse(BaseModel):
    """Response model for execution operations."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime
    processing_time_ms: Optional[float] = None


# Utility functions
def convert_order_request_to_order(order_req: OrderRequest) -> Order:
    """Convert OrderRequest to Order object."""
    return Order(
        order_id=order_req.order_id,
        symbol=order_req.symbol,
        side=OrderSide.BUY if order_req.side == 'buy' else OrderSide.SELL,
        order_type=OrderType(order_req.order_type.upper()),
        quantity=order_req.quantity,
        price=order_req.price,
        stop_price=order_req.stop_price
    )


def convert_market_data_point(md_point: MarketDataPoint) -> MarketData:
    """Convert MarketDataPoint to MarketData object."""
    return MarketData(
        timestamp=md_point.timestamp,
        symbol=md_point.symbol,
        bid=md_point.bid,
        ask=md_point.ask,
        last_price=md_point.last_price,
        volume=md_point.volume,
        bid_size=md_point.bid_size or 100,
        ask_size=md_point.ask_size or 100,
        vwap=md_point.vwap
    )


# API Endpoints

@router.post("/simulate", response_model=ExecutionResponse)
async def simulate_execution(request: ExecutionSimulationRequest):
    """
    Simulate order execution with realistic market conditions.
    
    Models latency, slippage, market impact, queue position, and transaction costs
    to provide realistic execution simulation for strategy evaluation.
    """
    start_time = datetime.now()
    
    try:
        # Validate input
        if not request.orders:
            raise HTTPException(status_code=400, detail="At least one order required")
        
        if not request.market_data:
            raise HTTPException(status_code=400, detail="Market data required")
        
        # Create profiles
        latency_profile = None
        if request.latency_profile:
            latency_profile = LatencyProfile(
                market_data_latency_ms=request.latency_profile.market_data_latency_ms,
                order_latency_ms=request.latency_profile.order_latency_ms,
                fill_latency_ms=request.latency_profile.fill_latency_ms,
                cancel_latency_ms=request.latency_profile.cancel_latency_ms,
                jitter_ms=request.latency_profile.jitter_ms
            )
        
        liquidity_profile = None
        if request.liquidity_profile:
            liquidity_profile = LiquidityProfile(
                average_bid_ask_spread_bps=request.liquidity_profile.average_bid_ask_spread_bps,
                average_depth_usd=request.liquidity_profile.average_depth_usd,
                price_levels=request.liquidity_profile.price_levels,
                depth_decay_factor=request.liquidity_profile.depth_decay_factor,
                liquidity_regeneration_rate=request.liquidity_profile.liquidity_regeneration_rate
            )
        
        # Initialize execution simulator
        simulator = ExecutionSimulator(latency_profile, liquidity_profile)
        
        # Convert market data
        market_data_objects = [convert_market_data_point(md) for md in request.market_data]
        market_data_by_symbol = {}
        for md in market_data_objects:
            if md.symbol not in market_data_by_symbol:
                market_data_by_symbol[md.symbol] = []
            market_data_by_symbol[md.symbol].append(md)
        
        # Sort market data by timestamp
        for symbol in market_data_by_symbol:
            market_data_by_symbol[symbol].sort(key=lambda x: x.timestamp)
        
        # Simulate order execution
        execution_results = []
        fills = []
        
        for order_req in request.orders:
            order = convert_order_request_to_order(order_req)
            
            # Get corresponding market data
            if order.symbol in market_data_by_symbol:
                # Find closest market data point
                symbol_data = market_data_by_symbol[order.symbol]
                closest_md = min(symbol_data, key=lambda x: abs(
                    x.timestamp.timestamp() - datetime.now().timestamp()
                ))
                
                # Submit order
                execution_result = simulator.submit_order(
                    order, closest_md, closest_md.timestamp
                )
                execution_results.append(execution_result)
                
                # Update market data stream
                for md in symbol_data:
                    simulator.update_market_data(md)
        
        # Get execution summary
        execution_summary = simulator.get_execution_summary()
        
        # Calculate performance metrics
        total_slippage = sum([
            result.get('slippage', 0) for result in execution_results
            if result.get('slippage') is not None
        ])
        
        avg_execution_delay = np.mean([
            (fill.timestamp - order.timestamp).total_seconds()
            for fill in simulator.fills
            for order in simulator.pending_orders.values()
            if fill.order_id == order.order_id and order.timestamp
        ]) if simulator.fills else 0
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExecutionResponse(
            success=True,
            message=f"Simulated execution for {len(request.orders)} orders",
            data={
                'execution_results': execution_results,
                'execution_summary': execution_summary,
                'performance_metrics': {
                    'total_slippage': total_slippage,
                    'avg_execution_delay_sec': avg_execution_delay,
                    'orders_processed': len(request.orders),
                    'fills_generated': len(simulator.fills)
                },
                'simulator_config': {
                    'latency_enabled': latency_profile is not None,
                    'liquidity_modeling': liquidity_profile is not None,
                    'halts_enabled': request.enable_halts,
                    'slippage_enabled': request.enable_slippage,
                    'transaction_costs_enabled': request.enable_transaction_costs
                }
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Execution simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Execution simulation failed: {str(e)}")


@router.post("/backtest", response_model=ExecutionResponse)
async def realistic_backtest(request: BacktestRequest):
    """
    Run realistic backtest with execution constraints.
    
    Simulates strategy execution with realistic latency, slippage, transaction costs,
    and market microstructure effects for accurate performance evaluation.
    """
    start_time = datetime.now()
    
    try:
        # Validate input
        if not request.strategy_signals:
            raise HTTPException(status_code=400, detail="Strategy signals required")
        
        if not request.market_data:
            raise HTTPException(status_code=400, detail="Market data required")
        
        # Create profiles
        latency_profile = None
        if request.latency_profile:
            latency_profile = LatencyProfile(
                market_data_latency_ms=request.latency_profile.market_data_latency_ms,
                order_latency_ms=request.latency_profile.order_latency_ms,
                fill_latency_ms=request.latency_profile.fill_latency_ms,
                cancel_latency_ms=request.latency_profile.cancel_latency_ms,
                jitter_ms=request.latency_profile.jitter_ms
            )
        
        liquidity_profile = None
        if request.liquidity_profile:
            liquidity_profile = LiquidityProfile(
                average_bid_ask_spread_bps=request.liquidity_profile.average_bid_ask_spread_bps,
                average_depth_usd=request.liquidity_profile.average_depth_usd,
                price_levels=request.liquidity_profile.price_levels,
                depth_decay_factor=request.liquidity_profile.depth_decay_factor,
                liquidity_regeneration_rate=request.liquidity_profile.liquidity_regeneration_rate
            )
        
        # Initialize simulator and backtester
        simulator = ExecutionSimulator(latency_profile, liquidity_profile)
        
        # Set date range
        start_date = request.start_date or datetime.now() - timedelta(days=365)
        end_date = request.end_date or datetime.now()
        
        backtester = RealisticBacktester(
            execution_simulator=simulator,
            universe=request.universe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Prepare data
        signals_df = pd.DataFrame(request.strategy_signals)
        market_df = pd.DataFrame([{
            'timestamp': md.timestamp,
            'symbol': md.symbol,
            'bid': md.bid,
            'ask': md.ask,
            'last': md.last_price,
            'volume': md.volume,
            'bid_size': md.bid_size,
            'ask_size': md.ask_size
        } for md in request.market_data])
        
        # Run backtest
        backtest_results = backtester.run_strategy_backtest(
            strategy_signals=signals_df,
            market_data=market_df,
            initial_capital=request.initial_capital
        )
        
        # Calculate performance metrics
        portfolio_history = backtest_results['portfolio_history']
        total_return = backtest_results['total_return']
        
        # Calculate additional metrics
        returns = portfolio_history['portfolio_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
        max_drawdown = (portfolio_history['portfolio_value'] / 
                       portfolio_history['portfolio_value'].cummax() - 1).min()
        
        execution_stats = backtest_results['execution_stats']
        avg_slippage = execution_stats['slippage'].mean() if len(execution_stats) > 0 else 0
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExecutionResponse(
            success=True,
            message=f"Realistic backtest completed for {len(request.universe)} symbols",
            data={
                'performance_metrics': {
                    'total_return': total_return,
                    'annualized_volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'final_portfolio_value': portfolio_history['portfolio_value'].iloc[-1],
                    'avg_slippage': avg_slippage
                },
                'execution_summary': backtest_results['execution_summary'],
                'backtest_period': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': (end_date - start_date).days
                },
                'trading_activity': {
                    'total_trades': len(execution_stats),
                    'total_volume': execution_stats['quantity'].sum() if len(execution_stats) > 0 else 0,
                    'total_fees': execution_stats['fees'].sum() if len(execution_stats) > 0 else 0
                },
                'final_positions': backtest_results['final_positions'],
                'initial_capital': request.initial_capital
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Realistic backtesting failed: {e}")
        raise HTTPException(status_code=500, detail=f"Realistic backtesting failed: {str(e)}")


@router.post("/slippage-analysis", response_model=ExecutionResponse)
async def analyze_slippage(request: SlippageAnalysisRequest):
    """
    Analyze expected slippage and market impact for orders.
    
    Estimates slippage based on order size, market conditions, and liquidity
    to help optimize execution strategies.
    """
    start_time = datetime.now()
    
    try:
        # Initialize slippage model
        slippage_model = SlippageModel()
        
        # Analyze each order
        slippage_analysis = []
        
        for order_req in request.orders:
            order = convert_order_request_to_order(order_req)
            
            # Find corresponding market data
            symbol_market_data = [md for md in request.market_data if md.symbol == order.symbol]
            
            if symbol_market_data:
                market_data = convert_market_data_point(symbol_market_data[0])
                
                # Calculate slippage components
                market_impact_slippage = slippage_model.calculate_slippage(
                    order, market_data, request.average_daily_volume_usd
                )
                bid_ask_impact = slippage_model.calculate_bid_ask_impact(order, market_data)
                
                # Calculate order statistics
                order_value = order.quantity * market_data.last_price
                participation_rate = order_value / request.average_daily_volume_usd
                
                slippage_analysis.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'order_value': order_value,
                    'participation_rate': participation_rate,
                    'market_impact_slippage': market_impact_slippage,
                    'bid_ask_impact': bid_ask_impact,
                    'total_slippage': market_impact_slippage + bid_ask_impact,
                    'slippage_bps': ((market_impact_slippage + bid_ask_impact) / market_data.last_price) * 10000,
                    'market_data': {
                        'bid': market_data.bid,
                        'ask': market_data.ask,
                        'last_price': market_data.last_price,
                        'spread_bps': ((market_data.ask - market_data.bid) / market_data.last_price) * 10000
                    }
                })
        
        # Calculate aggregate statistics
        total_slippage = sum(analysis['total_slippage'] for analysis in slippage_analysis)
        avg_slippage_bps = np.mean([analysis['slippage_bps'] for analysis in slippage_analysis])
        max_slippage_bps = max([analysis['slippage_bps'] for analysis in slippage_analysis])
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExecutionResponse(
            success=True,
            message=f"Slippage analysis completed for {len(request.orders)} orders",
            data={
                'order_analysis': slippage_analysis,
                'aggregate_statistics': {
                    'total_slippage': total_slippage,
                    'avg_slippage_bps': avg_slippage_bps,
                    'max_slippage_bps': max_slippage_bps,
                    'total_order_value': sum(analysis['order_value'] for analysis in slippage_analysis),
                    'avg_participation_rate': np.mean([analysis['participation_rate'] for analysis in slippage_analysis])
                },
                'model_parameters': {
                    'linear_impact_bps': slippage_model.linear_impact_bps,
                    'sqrt_impact_bps': slippage_model.sqrt_impact_bps,
                    'temporary_impact_decay': slippage_model.temporary_impact_decay
                }
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Slippage analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Slippage analysis failed: {str(e)}")


@router.post("/latency-test")
async def test_latency_simulation(latency_profile: LatencyProfileRequest):
    """
    Test latency simulation with specified parameters.
    
    Generates sample latency measurements to validate latency modeling
    and understand execution timing characteristics.
    """
    start_time = datetime.now()
    
    try:
        # Create latency profile
        profile = LatencyProfile(
            market_data_latency_ms=latency_profile.market_data_latency_ms,
            order_latency_ms=latency_profile.order_latency_ms,
            fill_latency_ms=latency_profile.fill_latency_ms,
            cancel_latency_ms=latency_profile.cancel_latency_ms,
            jitter_ms=latency_profile.jitter_ms
        )
        
        # Initialize latency simulator
        latency_simulator = LatencySimulator(profile)
        
        # Generate sample latencies
        n_samples = 1000
        
        market_data_latencies = [latency_simulator.get_market_data_latency() for _ in range(n_samples)]
        order_latencies = [latency_simulator.get_order_latency() for _ in range(n_samples)]
        fill_latencies = [latency_simulator.get_fill_latency() for _ in range(n_samples)]
        cancel_latencies = [latency_simulator.get_cancel_latency() for _ in range(n_samples)]
        
        # Calculate statistics
        latency_stats = {
            'market_data': {
                'mean': np.mean(market_data_latencies),
                'std': np.std(market_data_latencies),
                'min': np.min(market_data_latencies),
                'max': np.max(market_data_latencies),
                'percentiles': {
                    'p50': np.percentile(market_data_latencies, 50),
                    'p95': np.percentile(market_data_latencies, 95),
                    'p99': np.percentile(market_data_latencies, 99)
                }
            },
            'order_submission': {
                'mean': np.mean(order_latencies),
                'std': np.std(order_latencies),
                'min': np.min(order_latencies),
                'max': np.max(order_latencies),
                'percentiles': {
                    'p50': np.percentile(order_latencies, 50),
                    'p95': np.percentile(order_latencies, 95),
                    'p99': np.percentile(order_latencies, 99)
                }
            },
            'fill_confirmation': {
                'mean': np.mean(fill_latencies),
                'std': np.std(fill_latencies),
                'min': np.min(fill_latencies),
                'max': np.max(fill_latencies),
                'percentiles': {
                    'p50': np.percentile(fill_latencies, 50),
                    'p95': np.percentile(fill_latencies, 95),
                    'p99': np.percentile(fill_latencies, 99)
                }
            },
            'order_cancellation': {
                'mean': np.mean(cancel_latencies),
                'std': np.std(cancel_latencies),
                'min': np.min(cancel_latencies),
                'max': np.max(cancel_latencies),
                'percentiles': {
                    'p50': np.percentile(cancel_latencies, 50),
                    'p95': np.percentile(cancel_latencies, 95),
                    'p99': np.percentile(cancel_latencies, 99)
                }
            }
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExecutionResponse(
            success=True,
            message=f"Latency simulation test completed with {n_samples} samples",
            data={
                'latency_statistics': latency_stats,
                'profile_configuration': {
                    'market_data_latency_ms': profile.market_data_latency_ms,
                    'order_latency_ms': profile.order_latency_ms,
                    'fill_latency_ms': profile.fill_latency_ms,
                    'cancel_latency_ms': profile.cancel_latency_ms,
                    'jitter_ms': profile.jitter_ms
                },
                'sample_size': n_samples
            },
            timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Latency test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Latency test failed: {str(e)}")


@router.get("/execution-profiles")
async def get_execution_profiles():
    """Get available execution profiles and their characteristics."""
    
    return {
        'latency_profiles': {
            'low_latency': {
                'description': 'High-frequency trading setup',
                'market_data_latency_ms': 0.5,
                'order_latency_ms': 1.0,
                'fill_latency_ms': 0.8,
                'cancel_latency_ms': 1.2,
                'jitter_ms': 0.2
            },
            'retail': {
                'description': 'Typical retail trading latency',
                'market_data_latency_ms': 50.0,
                'order_latency_ms': 100.0,
                'fill_latency_ms': 75.0,
                'cancel_latency_ms': 80.0,
                'jitter_ms': 20.0
            },
            'institutional': {
                'description': 'Institutional trading latency',
                'market_data_latency_ms': 5.0,
                'order_latency_ms': 10.0,
                'fill_latency_ms': 8.0,
                'cancel_latency_ms': 12.0,
                'jitter_ms': 2.0
            }
        },
        'liquidity_profiles': {
            'large_cap': {
                'description': 'Large cap stocks with high liquidity',
                'average_bid_ask_spread_bps': 2.0,
                'average_depth_usd': 500000.0,
                'price_levels': 10,
                'depth_decay_factor': 0.8
            },
            'mid_cap': {
                'description': 'Mid cap stocks with moderate liquidity',
                'average_bid_ask_spread_bps': 8.0,
                'average_depth_usd': 100000.0,
                'price_levels': 5,
                'depth_decay_factor': 0.6
            },
            'small_cap': {
                'description': 'Small cap stocks with limited liquidity',
                'average_bid_ask_spread_bps': 25.0,
                'average_depth_usd': 25000.0,
                'price_levels': 3,
                'depth_decay_factor': 0.4
            }
        },
        'market_impact_factors': {
            'order_size': 'Larger orders create more market impact',
            'participation_rate': 'Higher participation in daily volume increases impact',
            'market_conditions': 'Volatile markets have higher impact',
            'time_of_day': 'Opening/closing periods have different impact',
            'liquidity': 'Less liquid stocks have higher impact'
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for execution realism service."""
    return {
        'status': 'healthy',
        'service': 'execution-realism',
        'timestamp': datetime.now(),
        'available_endpoints': [
            'simulate',
            'backtest',
            'slippage-analysis',
            'latency-test',
            'execution-profiles'
        ]
    }


@router.get("/documentation")
async def get_api_documentation():
    """Get comprehensive API documentation for execution realism endpoints."""
    
    return {
        'overview': 'Execution Realism Framework for realistic trading simulation and backtesting',
        'purpose': 'Bridge the gap between idealized backtesting and live trading reality',
        'components': {
            'latency_modeling': {
                'description': 'Simulates network and processing delays',
                'factors': ['Market data feed latency', 'Order submission delays', 'Fill confirmations', 'Cancellation timing'],
                'impact': 'Affects order timing and market opportunity capture'
            },
            'queue_position': {
                'description': 'Models order book position and fill probability',
                'factors': ['Price level aggressiveness', 'Queue position', 'Market volume', 'Time in queue'],
                'impact': 'Determines fill likelihood and timing for limit orders'
            },
            'market_impact': {
                'description': 'Models price impact from order execution',
                'factors': ['Order size', 'Market liquidity', 'Participation rate', 'Temporary vs permanent impact'],
                'impact': 'Affects execution price and slippage costs'
            },
            'transaction_costs': {
                'description': 'Comprehensive cost modeling',
                'factors': ['Commissions', 'Regulatory fees', 'Borrowing costs', 'Bid-ask spreads'],
                'impact': 'Reduces net returns and affects strategy profitability'
            },
            'market_halts': {
                'description': 'Trading session and halt management',
                'factors': ['Market hours', 'Trading halts', 'Pre/post market sessions'],
                'impact': 'Affects order execution timing and strategy implementation'
            }
        },
        'endpoints': {
            '/simulate': {
                'method': 'POST',
                'description': 'Simulate order execution with realistic constraints',
                'use_cases': ['Order execution analysis', 'Strategy validation', 'Cost estimation']
            },
            '/backtest': {
                'method': 'POST',
                'description': 'Run realistic backtests with execution constraints',
                'use_cases': ['Strategy performance evaluation', 'Risk assessment', 'Implementation shortfall analysis']
            },
            '/slippage-analysis': {
                'method': 'POST',
                'description': 'Analyze expected slippage and market impact',
                'use_cases': ['Trade cost analysis', 'Order sizing optimization', 'Execution strategy selection']
            },
            '/latency-test': {
                'method': 'POST',
                'description': 'Test latency modeling with custom parameters',
                'use_cases': ['Infrastructure assessment', 'Latency optimization', 'Execution timing analysis']
            }
        },
        'best_practices': [
            'Use realistic latency profiles for your trading infrastructure',
            'Account for market impact in large order execution',
            'Consider transaction costs in strategy evaluation',
            'Model queue position for limit order strategies',
            'Test execution during different market conditions',
            'Validate backtest results with live trading performance'
        ],
        'common_applications': [
            'Algorithmic trading strategy validation',
            'Execution algorithm development',
            'Risk management and cost analysis',
            'Infrastructure performance assessment',
            'Regulatory compliance and best execution analysis'
        ]
    }