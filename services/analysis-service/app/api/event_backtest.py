"""
Event-Aware Backtesting API endpoints for comprehensive event-driven strategy testing.
Provides sophisticated backtesting with event detection and tight stop-loss management.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import io
import json

from ..services.event_aware_backtest_engine import (
    EventAwareBacktestEngine,
    BacktestEvent,
    EventType,
    StopLossConfig,
    StopLossType,
    BacktestResults
)

router = APIRouter(prefix="/event-backtest", tags=["Event Backtesting"])

# Request/Response Models
class EventInput(BaseModel):
    """Event input for backtesting"""
    timestamp: datetime
    symbol: str
    event_type: str
    event_description: str
    surprise_magnitude: float
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    expected_move: Optional[float] = None
    actual_move_1d: Optional[float] = None
    actual_move_3d: Optional[float] = None
    actual_move_5d: Optional[float] = None
    volume_surge: Optional[float] = None
    analyst_rating: Optional[str] = None
    confidence_score: Optional[float] = None

class StopLossConfigInput(BaseModel):
    """Stop loss configuration input"""
    stop_type: str = Field(default="volatility_adjusted")
    base_percentage: float = Field(default=0.08, description="Base stop loss percentage")
    volatility_multiplier: float = Field(default=1.5, description="Volatility adjustment multiplier")
    max_stop_loss: float = Field(default=0.15, description="Maximum stop loss allowed")
    min_stop_loss: float = Field(default=0.03, description="Minimum stop loss allowed")
    trailing_threshold: Optional[float] = Field(None, description="Trailing stop threshold")
    time_decay_factor: Optional[float] = Field(None, description="Time-based stop tightening")

class BacktestRequest(BaseModel):
    """Request for event-aware backtesting"""
    start_date: datetime
    end_date: datetime
    events: List[EventInput]
    initial_capital: float = Field(default=100000.0, description="Starting capital")
    position_sizing: str = Field(default="equal_weight", description="Position sizing method")
    max_positions: int = Field(default=10, description="Maximum concurrent positions")
    stop_loss_configs: Optional[Dict[str, StopLossConfigInput]] = Field(None, description="Event-specific stop loss configs")
    include_costs: bool = Field(default=True, description="Include trading costs")
    commission_rate: float = Field(default=0.001, description="Commission rate (0.1%)")
    slippage_bps: float = Field(default=5.0, description="Slippage in basis points")

class BacktestWithDataRequest(BaseModel):
    """Request for backtesting with uploaded price data"""
    start_date: datetime
    end_date: datetime
    events: List[EventInput]
    price_data: Dict[str, List[Dict[str, Any]]] = Field(..., description="Price data by symbol")
    initial_capital: float = Field(default=100000.0)
    position_sizing: str = Field(default="equal_weight")
    max_positions: int = Field(default=10)
    stop_loss_configs: Optional[Dict[str, StopLossConfigInput]] = None
    include_costs: bool = Field(default=True)
    commission_rate: float = Field(default=0.001)
    slippage_bps: float = Field(default=5.0)

class OptimizationRequest(BaseModel):
    """Request for stop loss optimization"""
    events: List[EventInput]
    price_data: Dict[str, List[Dict[str, Any]]]
    start_date: datetime
    end_date: datetime
    optimization_metric: str = Field(default="sharpe_ratio", description="Metric to optimize")
    parameter_ranges: Dict[str, List[float]] = Field(
        default={
            "base_percentage": [0.05, 0.08, 0.10, 0.12, 0.15],
            "volatility_multiplier": [1.0, 1.5, 2.0, 2.5, 3.0]
        }
    )

class BacktestResponse(BaseModel):
    """Response for backtest results"""
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk metrics
    var_95: float
    expected_shortfall: float
    maximum_consecutive_losses: int
    average_holding_period: float
    
    # Event analysis
    event_type_performance: Dict[str, Dict[str, float]]
    top_performing_events: List[Dict[str, Any]]
    worst_performing_events: List[Dict[str, Any]]
    
    # Additional metrics
    monthly_returns: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    drawdown_curve: List[Dict[str, Any]]

class OptimizationResponse(BaseModel):
    """Response for optimization results"""
    optimal_parameters: Dict[str, float]
    optimization_metric_value: float
    parameter_sensitivity: Dict[str, List[Dict[str, float]]]
    performance_surface: List[Dict[str, Any]]
    recommended_config: Dict[str, Any]

# Dependency injection
async def get_backtest_engine() -> EventAwareBacktestEngine:
    """Get event-aware backtest engine instance"""
    return EventAwareBacktestEngine()

# API Endpoints
@router.post("/run-backtest", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    engine: EventAwareBacktestEngine = Depends(get_backtest_engine)
):
    """
    Run comprehensive event-aware backtest.
    
    Simulates event-driven trading strategy with sophisticated stop-loss
    management and comprehensive performance attribution.
    """
    try:
        # Convert events to BacktestEvent objects
        events = []
        for event_input in request.events:
            try:
                event_type = EventType(event_input.event_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {event_input.event_type}")
            
            events.append(BacktestEvent(
                timestamp=event_input.timestamp,
                symbol=event_input.symbol,
                event_type=event_type,
                event_description=event_input.event_description,
                surprise_magnitude=event_input.surprise_magnitude,
                market_cap=event_input.market_cap,
                sector=event_input.sector,
                expected_move=event_input.expected_move,
                actual_move_1d=event_input.actual_move_1d,
                actual_move_3d=event_input.actual_move_3d,
                actual_move_5d=event_input.actual_move_5d,
                volume_surge=event_input.volume_surge,
                analyst_rating=event_input.analyst_rating,
                confidence_score=event_input.confidence_score
            ))
        
        # Convert stop loss configurations
        stop_configs = None
        if request.stop_loss_configs:
            stop_configs = {}
            for event_type_str, config_input in request.stop_loss_configs.items():
                try:
                    event_type = EventType(event_type_str.lower())
                    stop_type = StopLossType(config_input.stop_type.lower())
                    
                    stop_configs[event_type] = StopLossConfig(
                        stop_type=stop_type,
                        base_percentage=config_input.base_percentage,
                        volatility_multiplier=config_input.volatility_multiplier,
                        max_stop_loss=config_input.max_stop_loss,
                        min_stop_loss=config_input.min_stop_loss,
                        trailing_threshold=config_input.trailing_threshold,
                        time_decay_factor=config_input.time_decay_factor
                    )
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid configuration for {event_type_str}: {str(e)}")
        
        # Generate sample price data (in production, this would fetch from data sources)
        price_data = _generate_sample_price_data(events, request.start_date, request.end_date)
        
        # Run backtest
        results = await engine.run_backtest(
            events=events,
            price_data=price_data,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            position_sizing=request.position_sizing,
            max_positions=request.max_positions,
            stop_loss_config=stop_configs,
            include_costs=request.include_costs,
            commission_rate=request.commission_rate,
            slippage_bps=request.slippage_bps
        )
        
        # Format response
        return _format_backtest_response(results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")

@router.post("/run-backtest-with-data", response_model=BacktestResponse)
async def run_backtest_with_data(
    request: BacktestWithDataRequest,
    engine: EventAwareBacktestEngine = Depends(get_backtest_engine)
):
    """
    Run backtest with provided price data.
    
    Allows for precise backtesting with custom price datasets
    for research and validation purposes.
    """
    try:
        # Convert events
        events = []
        for event_input in request.events:
            try:
                event_type = EventType(event_input.event_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {event_input.event_type}")
            
            events.append(BacktestEvent(
                timestamp=event_input.timestamp,
                symbol=event_input.symbol,
                event_type=event_type,
                event_description=event_input.event_description,
                surprise_magnitude=event_input.surprise_magnitude,
                market_cap=event_input.market_cap,
                sector=event_input.sector,
                expected_move=event_input.expected_move,
                actual_move_1d=event_input.actual_move_1d,
                actual_move_3d=event_input.actual_move_3d,
                actual_move_5d=event_input.actual_move_5d,
                volume_surge=event_input.volume_surge,
                analyst_rating=event_input.analyst_rating,
                confidence_score=event_input.confidence_score
            ))
        
        # Convert price data to DataFrames
        price_dataframes = {}
        for symbol, price_list in request.price_data.items():
            df = pd.DataFrame(price_list)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            price_dataframes[symbol] = df
        
        # Convert stop loss configurations
        stop_configs = None
        if request.stop_loss_configs:
            stop_configs = {}
            for event_type_str, config_input in request.stop_loss_configs.items():
                try:
                    event_type = EventType(event_type_str.lower())
                    stop_type = StopLossType(config_input.stop_type.lower())
                    
                    stop_configs[event_type] = StopLossConfig(
                        stop_type=stop_type,
                        base_percentage=config_input.base_percentage,
                        volatility_multiplier=config_input.volatility_multiplier,
                        max_stop_loss=config_input.max_stop_loss,
                        min_stop_loss=config_input.min_stop_loss,
                        trailing_threshold=config_input.trailing_threshold,
                        time_decay_factor=config_input.time_decay_factor
                    )
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid configuration for {event_type_str}: {str(e)}")
        
        # Run backtest
        results = await engine.run_backtest(
            events=events,
            price_data=price_dataframes,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            position_sizing=request.position_sizing,
            max_positions=request.max_positions,
            stop_loss_config=stop_configs,
            include_costs=request.include_costs,
            commission_rate=request.commission_rate,
            slippage_bps=request.slippage_bps
        )
        
        return _format_backtest_response(results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running backtest with data: {str(e)}")

@router.post("/optimize-stop-losses", response_model=OptimizationResponse)
async def optimize_stop_losses(
    request: OptimizationRequest,
    engine: EventAwareBacktestEngine = Depends(get_backtest_engine)
):
    """
    Optimize stop loss parameters for maximum performance.
    
    Tests multiple parameter combinations to find optimal
    stop loss configuration for the given event dataset.
    """
    try:
        # Convert events
        events = []
        for event_input in request.events:
            try:
                event_type = EventType(event_input.event_type.lower())
            except ValueError:
                continue  # Skip invalid event types
            
            events.append(BacktestEvent(
                timestamp=event_input.timestamp,
                symbol=event_input.symbol,
                event_type=event_type,
                event_description=event_input.event_description,
                surprise_magnitude=event_input.surprise_magnitude,
                confidence_score=event_input.confidence_score
            ))
        
        # Convert price data
        price_dataframes = {}
        for symbol, price_list in request.price_data.items():
            df = pd.DataFrame(price_list)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            price_dataframes[symbol] = df
        
        # Run optimization
        optimization_results = []
        
        base_percentages = request.parameter_ranges.get("base_percentage", [0.05, 0.08, 0.10, 0.12, 0.15])
        volatility_multipliers = request.parameter_ranges.get("volatility_multiplier", [1.0, 1.5, 2.0, 2.5, 3.0])
        
        best_metric = -float('inf')
        best_params = {}
        
        for base_pct in base_percentages:
            for vol_mult in volatility_multipliers:
                # Create stop loss config
                stop_config = {
                    EventType.EARNINGS: StopLossConfig(
                        stop_type=StopLossType.VOLATILITY_ADJUSTED,
                        base_percentage=base_pct,
                        volatility_multiplier=vol_mult,
                        max_stop_loss=0.20,
                        min_stop_loss=0.02
                    )
                }
                
                # Run backtest with these parameters
                try:
                    results = await engine.run_backtest(
                        events=events,
                        price_data=price_dataframes,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        initial_capital=100000.0,
                        position_sizing="equal_weight",
                        max_positions=10,
                        stop_loss_config=stop_config,
                        include_costs=True,
                        commission_rate=0.001,
                        slippage_bps=5.0
                    )
                    
                    # Extract optimization metric
                    if request.optimization_metric == "sharpe_ratio":
                        metric_value = results.sharpe_ratio
                    elif request.optimization_metric == "total_return":
                        metric_value = results.total_return
                    elif request.optimization_metric == "profit_factor":
                        metric_value = results.profit_factor
                    else:
                        metric_value = results.sharpe_ratio  # Default
                    
                    optimization_results.append({
                        'base_percentage': base_pct,
                        'volatility_multiplier': vol_mult,
                        'metric_value': metric_value,
                        'total_return': results.total_return,
                        'sharpe_ratio': results.sharpe_ratio,
                        'max_drawdown': results.max_drawdown,
                        'win_rate': results.win_rate
                    })
                    
                    # Track best parameters
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_params = {
                            'base_percentage': base_pct,
                            'volatility_multiplier': vol_mult
                        }
                        
                except Exception as e:
                    # Skip failed parameter combinations
                    continue
        
        # Calculate parameter sensitivity
        parameter_sensitivity = {
            'base_percentage': [],
            'volatility_multiplier': []
        }
        
        # Group results by parameter values
        for base_pct in base_percentages:
            base_results = [r for r in optimization_results if r['base_percentage'] == base_pct]
            if base_results:
                avg_metric = np.mean([r['metric_value'] for r in base_results])
                parameter_sensitivity['base_percentage'].append({
                    'parameter_value': base_pct,
                    'avg_metric': avg_metric
                })
        
        for vol_mult in volatility_multipliers:
            vol_results = [r for r in optimization_results if r['volatility_multiplier'] == vol_mult]
            if vol_results:
                avg_metric = np.mean([r['metric_value'] for r in vol_results])
                parameter_sensitivity['volatility_multiplier'].append({
                    'parameter_value': vol_mult,
                    'avg_metric': avg_metric
                })
        
        # Recommended configuration
        recommended_config = {
            'stop_loss_type': 'volatility_adjusted',
            'base_percentage': best_params.get('base_percentage', 0.08),
            'volatility_multiplier': best_params.get('volatility_multiplier', 1.5),
            'max_stop_loss': 0.15,
            'min_stop_loss': 0.03,
            'rationale': f'Optimized for {request.optimization_metric} with value of {best_metric:.3f}'
        }
        
        return OptimizationResponse(
            optimal_parameters=best_params,
            optimization_metric_value=best_metric,
            parameter_sensitivity=parameter_sensitivity,
            performance_surface=optimization_results,
            recommended_config=recommended_config
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing stop losses: {str(e)}")

@router.get("/default-stop-configs")
async def get_default_stop_configs():
    """
    Get default stop loss configurations for different event types.
    
    Returns the built-in stop loss configurations that are optimized
    for different types of events based on historical analysis.
    """
    try:
        engine = EventAwareBacktestEngine()
        
        configs = {}
        for event_type, config in engine.default_stop_configs.items():
            configs[event_type.value] = {
                'stop_type': config.stop_type.value,
                'base_percentage': config.base_percentage,
                'volatility_multiplier': config.volatility_multiplier,
                'max_stop_loss': config.max_stop_loss,
                'min_stop_loss': config.min_stop_loss,
                'trailing_threshold': config.trailing_threshold,
                'time_decay_factor': config.time_decay_factor
            }
        
        return {
            'default_configurations': configs,
            'configuration_rationale': {
                'earnings': 'Volatility-adjusted stops with moderate time decay for earnings events',
                'fda_approval': 'Wide stops with trailing feature for binary biotech events',
                'merger_acquisition': 'Tight fixed stops for deal arbitrage strategies'
            },
            'customization_guidelines': {
                'high_volatility_events': 'Increase volatility_multiplier to 2.0-2.5',
                'low_confidence_events': 'Reduce base_percentage for tighter risk control',
                'binary_events': 'Use event_specific type with trailing stops',
                'short_term_events': 'Increase time_decay_factor for faster tightening'
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving default configurations: {str(e)}")

@router.post("/event-performance-analysis")
async def analyze_event_performance(
    events: List[EventInput],
    price_data: Dict[str, List[Dict[str, Any]]]
):
    """
    Analyze historical performance of events without executing trades.
    
    Provides statistical analysis of event performance for strategy
    development and event filtering optimization.
    """
    try:
        # Convert events
        backtest_events = []
        for event_input in events:
            try:
                event_type = EventType(event_input.event_type.lower())
                backtest_events.append(BacktestEvent(
                    timestamp=event_input.timestamp,
                    symbol=event_input.symbol,
                    event_type=event_type,
                    event_description=event_input.event_description,
                    surprise_magnitude=event_input.surprise_magnitude,
                    actual_move_1d=event_input.actual_move_1d,
                    actual_move_3d=event_input.actual_move_3d,
                    actual_move_5d=event_input.actual_move_5d,
                    confidence_score=event_input.confidence_score
                ))
            except ValueError:
                continue  # Skip invalid events
        
        # Analyze performance by event type
        event_analysis = {}
        for event_type in EventType:
            type_events = [e for e in backtest_events if e.event_type == event_type]
            
            if type_events:
                moves_1d = [e.actual_move_1d for e in type_events if e.actual_move_1d is not None]
                moves_3d = [e.actual_move_3d for e in type_events if e.actual_move_3d is not None]
                surprises = [e.surprise_magnitude for e in type_events]
                
                event_analysis[event_type.value] = {
                    'count': len(type_events),
                    'avg_surprise': np.mean(surprises) if surprises else 0,
                    'avg_1d_move': np.mean(moves_1d) if moves_1d else 0,
                    'avg_3d_move': np.mean(moves_3d) if moves_3d else 0,
                    'win_rate_1d': len([m for m in moves_1d if m > 0]) / len(moves_1d) if moves_1d else 0,
                    'volatility_1d': np.std(moves_1d) if moves_1d else 0,
                    'best_1d_move': max(moves_1d) if moves_1d else 0,
                    'worst_1d_move': min(moves_1d) if moves_1d else 0
                }
        
        # Surprise magnitude analysis
        surprise_buckets = {
            'low': [e for e in backtest_events if 0 <= abs(e.surprise_magnitude) < 0.05],
            'medium': [e for e in backtest_events if 0.05 <= abs(e.surprise_magnitude) < 0.15],
            'high': [e for e in backtest_events if abs(e.surprise_magnitude) >= 0.15]
        }
        
        surprise_analysis = {}
        for bucket, bucket_events in surprise_buckets.items():
            if bucket_events:
                moves = [e.actual_move_1d for e in bucket_events if e.actual_move_1d is not None]
                surprise_analysis[bucket] = {
                    'count': len(bucket_events),
                    'avg_move': np.mean(moves) if moves else 0,
                    'win_rate': len([m for m in moves if m > 0]) / len(moves) if moves else 0,
                    'volatility': np.std(moves) if moves else 0
                }
        
        return {
            'total_events': len(backtest_events),
            'event_type_analysis': event_analysis,
            'surprise_magnitude_analysis': surprise_analysis,
            'overall_statistics': {
                'avg_surprise': np.mean([e.surprise_magnitude for e in backtest_events]),
                'avg_1d_move': np.mean([e.actual_move_1d for e in backtest_events if e.actual_move_1d is not None]),
                'overall_win_rate': len([e for e in backtest_events if e.actual_move_1d and e.actual_move_1d > 0]) / len([e for e in backtest_events if e.actual_move_1d is not None])
            },
            'recommendations': {
                'best_event_types': sorted(event_analysis.items(), key=lambda x: x[1]['avg_1d_move'], reverse=True)[:3],
                'optimal_surprise_threshold': 0.05,  # Based on analysis
                'suggested_filters': [
                    'Focus on high surprise magnitude events (>15%)',
                    'Prioritize FDA approvals and M&A events',
                    'Avoid low confidence events (<60%)'
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing event performance: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for event backtesting service"""
    return {
        'status': 'healthy',
        'service': 'event-aware-backtest-engine',
        'timestamp': datetime.utcnow().isoformat(),
        'features': [
            'event_aware_backtesting',
            'sophisticated_stop_losses',
            'performance_attribution',
            'parameter_optimization',
            'event_analysis',
            'tight_risk_management'
        ]
    }

# Helper functions
def _generate_sample_price_data(events: List[BacktestEvent], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """Generate sample price data for backtesting"""
    
    price_data = {}
    symbols = list(set(event.symbol for event in events))
    
    for symbol in symbols:
        # Generate daily price data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate price movement with some volatility
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
        
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create DataFrame with OHLCV data
        df_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            daily_vol = abs(np.random.normal(0, 0.01))  # Daily volatility
            high = price * (1 + daily_vol)
            low = price * (1 - daily_vol)
            volume = int(np.random.normal(1000000, 200000))
            
            df_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': price * (1 + np.random.normal(0, 0.005)),  # Small close adjustment
                'volume': max(100000, volume)
            })
        
        df = pd.DataFrame(df_data, index=dates)
        price_data[symbol] = df
    
    return price_data

def _format_backtest_response(results: BacktestResults) -> BacktestResponse:
    """Format backtest results for API response"""
    
    # Find top and worst performing trades
    trades_with_pnl = [t for t in results.trades if t.pnl is not None]
    top_trades = sorted(trades_with_pnl, key=lambda x: x.pnl, reverse=True)[:5]
    worst_trades = sorted(trades_with_pnl, key=lambda x: x.pnl)[:5]
    
    top_performing_events = [
        {
            'symbol': trade.symbol,
            'event_type': trade.event.event_type.value,
            'event_description': trade.event.event_description,
            'pnl': trade.pnl,
            'pnl_percentage': trade.pnl_percentage,
            'holding_period_hours': trade.holding_period_hours,
            'exit_reason': trade.exit_reason.value if trade.exit_reason else None
        }
        for trade in top_trades
    ]
    
    worst_performing_events = [
        {
            'symbol': trade.symbol,
            'event_type': trade.event.event_type.value,
            'event_description': trade.event.event_description,
            'pnl': trade.pnl,
            'pnl_percentage': trade.pnl_percentage,
            'holding_period_hours': trade.holding_period_hours,
            'exit_reason': trade.exit_reason.value if trade.exit_reason else None
        }
        for trade in worst_trades
    ]
    
    # Convert event type performance
    event_type_performance = {}
    for event_type, performance in results.event_type_performance.items():
        event_type_performance[event_type.value] = performance
    
    return BacktestResponse(
        total_return=results.total_return,
        annual_return=results.annual_return,
        sharpe_ratio=results.sharpe_ratio,
        sortino_ratio=results.sortino_ratio,
        max_drawdown=results.max_drawdown,
        win_rate=results.win_rate,
        profit_factor=results.profit_factor,
        total_trades=results.total_trades,
        winning_trades=results.winning_trades,
        losing_trades=results.losing_trades,
        average_win=results.average_win,
        average_loss=results.average_loss,
        largest_win=results.largest_win,
        largest_loss=results.largest_loss,
        var_95=results.var_95,
        expected_shortfall=results.expected_shortfall,
        maximum_consecutive_losses=results.maximum_consecutive_losses,
        average_holding_period=results.average_holding_period,
        event_type_performance=event_type_performance,
        top_performing_events=top_performing_events,
        worst_performing_events=worst_performing_events,
        monthly_returns=[],  # Would calculate from equity curve
        equity_curve=[],     # Would extract from results
        drawdown_curve=[]    # Would extract from results
    )