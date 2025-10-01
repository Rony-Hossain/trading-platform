"""
Strategy Service - Backtesting and Strategy Development Engine
Provides backtesting capabilities, strategy optimization, and paper trading
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel
import pandas as pd

from .core.database import get_db, create_tables
from .engines.backtest_engine import BacktestEngine
from .engines.portfolio_manager import PortfolioManager
from .engines.risk_manager import RiskManager
from .engines.market_impact_model import (
    MarketImpactModel, MarketMicrostructure, OrderProfile, 
    ImpactModel, ExecutionStyle
)
from .engines.portfolio_constructor import create_portfolio_allocation
from .execution.borrow_checker import (
    BorrowChecker, BorrowRequest, create_borrow_checker, 
    validate_portfolio_shorts, BorrowStatus
)
from .strategies.strategy_loader import StrategyLoader
from .schemas import (
    Strategy, StrategyCreate, StrategyUpdate,
    Backtest, BacktestCreate, BacktestResult,
    Trade, Position, PortfolioSnapshot,
    RiskMetrics, PerformanceMetrics,
    OptimizationRequest, OptimizationResult
)

# Simple ErrorResponse for now
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
backtest_engine = BacktestEngine()
portfolio_manager = PortfolioManager()
risk_manager = RiskManager()
market_impact_model = MarketImpactModel()
strategy_loader = StrategyLoader()
borrow_checker = create_borrow_checker({'enable_mock': True})

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    logger.info("Strategy Service started")
    yield
    # Shutdown
    logger.info("Strategy Service stopped")

app = FastAPI(
    title="Strategy Service", 
    description="Backtesting engine and strategy development platform",
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

# Request models
class BacktestRequest(BaseModel):
    strategy_id: int
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    parameters: Optional[Dict[str, Any]] = None

class OptimizationRequest(BaseModel):
    strategy_id: int
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    parameter_ranges: Dict[str, Dict[str, float]]  # {"param": {"min": 1, "max": 10, "step": 1}}
    objective: str = "sharpe_ratio"  # sharpe_ratio, total_return, max_drawdown

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Strategy Service",
        "status": "running",
        "version": "1.0.0", 
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "strategies": "/strategies",
            "backtest": "/backtest",
            "optimize": "/optimize",
            "results": "/results/{backtest_id}",
            "performance": "/performance/{backtest_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "strategy-service",
        "timestamp": datetime.now().isoformat(),
        "engines": {
            "backtest": backtest_engine.is_healthy(),
            "portfolio": portfolio_manager.is_healthy(),
            "risk": risk_manager.is_healthy()
        }
    }

@app.get("/strategies", response_model=List[Strategy])
async def get_strategies(
    user_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None), 
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get available strategies"""
    try:
        return strategy_loader.get_strategies(
            db, user_id, category, skip, limit
        )
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/strategies", response_model=Strategy)
async def create_strategy(
    strategy: StrategyCreate,
    db: Session = Depends(get_db)
):
    """Create a new trading strategy"""
    try:
        return strategy_loader.create_strategy(db, strategy)
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/strategies/{strategy_id}", response_model=Strategy)
async def get_strategy(
    strategy_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific strategy"""
    try:
        strategy = strategy_loader.get_strategy(db, strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/strategies/{strategy_id}", response_model=Strategy)
async def update_strategy(
    strategy_id: int,
    strategy_update: StrategyUpdate,
    db: Session = Depends(get_db)
):
    """Update a strategy"""
    try:
        strategy = strategy_loader.update_strategy(
            db, strategy_id, strategy_update
        )
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        return strategy
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest", response_model=Backtest)
async def start_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start a new backtest"""
    try:
        # Validate strategy exists
        strategy = strategy_loader.get_strategy(db, request.strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Create backtest record
        backtest_create = BacktestCreate(
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            start_date=datetime.fromisoformat(request.start_date).date(),
            end_date=datetime.fromisoformat(request.end_date).date(),
            initial_capital=request.initial_capital,
            parameters=request.parameters or {},
            status="pending"
        )
        
        backtest = backtest_engine.create_backtest(db, backtest_create)
        
        # Start backtest in background
        background_tasks.add_task(
            backtest_engine.run_backtest_async,
            backtest.id, db
        )
        
        return backtest
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/{backtest_id}", response_model=Backtest)
async def get_backtest(
    backtest_id: int,
    db: Session = Depends(get_db)
):
    """Get backtest details"""
    try:
        backtest = backtest_engine.get_backtest(db, backtest_id)
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        return backtest
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/{backtest_id}/results", response_model=BacktestResult)
async def get_backtest_results(
    backtest_id: int,
    include_trades: bool = Query(True),
    include_metrics: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get detailed backtest results"""
    try:
        return backtest_engine.get_backtest_results(
            db, backtest_id, include_trades, include_metrics
        )
    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/{backtest_id}/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    backtest_id: int,
    benchmark_symbol: Optional[str] = Query("SPY", description="Benchmark symbol for comparison"),
    db: Session = Depends(get_db)
):
    """Get performance analysis for backtest"""
    try:
        return await backtest_engine.calculate_performance_metrics(
            db, backtest_id, benchmark_symbol
        )
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/{backtest_id}/trades", response_model=List[Trade])
async def get_backtest_trades(
    backtest_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
    db: Session = Depends(get_db)
):
    """Get trades from backtest"""
    try:
        return backtest_engine.get_backtest_trades(
            db, backtest_id, skip, limit
        )
    except Exception as e:
        logger.error(f"Error getting backtest trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize", response_model=OptimizationResult)
async def optimize_strategy(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Optimize strategy parameters"""
    try:
        # Validate strategy exists
        strategy = strategy_loader.get_strategy(db, request.strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Start optimization in background
        optimization_id = backtest_engine.start_optimization(db, request)
        
        background_tasks.add_task(
            backtest_engine.run_optimization_async,
            optimization_id, db
        )
        
        return {
            "optimization_id": optimization_id,
            "status": "started",
            "started_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize/{optimization_id}")
async def get_optimization_results(
    optimization_id: int,
    db: Session = Depends(get_db)
):
    """Get optimization results"""
    try:
        return backtest_engine.get_optimization_results(db, optimization_id)
    except Exception as e:
        logger.error(f"Error getting optimization results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk/{backtest_id}", response_model=RiskMetrics)
async def get_risk_analysis(
    backtest_id: int,
    confidence_level: float = Query(0.95, ge=0.90, le=0.99),
    db: Session = Depends(get_db)
):
    """Get risk analysis for backtest"""
    try:
        return await risk_manager.analyze_backtest_risk(
            db, backtest_id, confidence_level
        )
    except Exception as e:
        logger.error(f"Error analyzing risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/{backtest_id}/snapshots", response_model=List[PortfolioSnapshot])
async def get_portfolio_snapshots(
    backtest_id: int,
    frequency: str = Query("daily", description="daily, weekly, monthly"),
    db: Session = Depends(get_db)
):
    """Get portfolio value snapshots over time"""
    try:
        return portfolio_manager.get_portfolio_snapshots(
            db, backtest_id, frequency
        )
    except Exception as e:
        logger.error(f"Error getting portfolio snapshots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare")
async def compare_strategies(
    backtest_ids: str = Query(..., description="Comma-separated backtest IDs"),
    metrics: str = Query("total_return,sharpe_ratio,max_drawdown", description="Comma-separated metrics"),
    db: Session = Depends(get_db)
):
    """Compare multiple strategy backtests"""
    try:
        backtest_id_list = [int(x.strip()) for x in backtest_ids.split(",")]
        metric_list = [x.strip() for x in metrics.split(",")]
        
        return await backtest_engine.compare_backtests(
            db, backtest_id_list, metric_list
        )
    except Exception as e:
        logger.error(f"Error comparing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leaderboard")
async def get_strategy_leaderboard(
    metric: str = Query("sharpe_ratio", description="Ranking metric"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get strategy leaderboard"""
    try:
        return backtest_engine.get_strategy_leaderboard(
            db, metric, symbol, timeframe, limit
        )
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/paper-trading/{strategy_id}")
async def get_paper_trading_status(
    strategy_id: int,
    db: Session = Depends(get_db)
):
    """Get paper trading status for strategy"""
    try:
        return portfolio_manager.get_paper_trading_status(db, strategy_id)
    except Exception as e:
        logger.error(f"Error getting paper trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/paper-trading/{strategy_id}/start")
async def start_paper_trading(
    strategy_id: int,
    initial_capital: float = Query(100000.0, gt=0),
    symbols: str = Query(..., description="Comma-separated symbols"),
    db: Session = Depends(get_db)
):
    """Start paper trading for strategy"""
    try:
        symbol_list = [x.strip() for x in symbols.split(",")]
        
        return portfolio_manager.start_paper_trading(
            db, strategy_id, initial_capital, symbol_list
        )
    except Exception as e:
        logger.error(f"Error starting paper trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/paper-trading/{strategy_id}/stop")
async def stop_paper_trading(
    strategy_id: int,
    db: Session = Depends(get_db)
):
    """Stop paper trading for strategy"""
    try:
        return portfolio_manager.stop_paper_trading(db, strategy_id)
    except Exception as e:
        logger.error(f"Error stopping paper trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_service_stats(db: Session = Depends(get_db)):
    """Get service statistics"""
    try:
        return {
            "total_strategies": strategy_loader.get_strategy_count(db),
            "total_backtests": backtest_engine.get_backtest_count(db),
            "active_paper_trading": portfolio_manager.get_active_paper_trading_count(db),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting service stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Market Impact and Execution Cost Modeling Endpoints

class MarketDataRequest(BaseModel):
    symbol: str
    current_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    average_daily_volume: float
    average_daily_value: float
    volatility_annualized: float
    market_cap: Optional[float] = None
    tick_size: float = 0.01

class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    target_price: Optional[float] = None
    execution_style: str = "moderate"  # aggressive, moderate, passive, twap, vwap
    time_horizon_minutes: float = 60.0
    participation_rate: float = 0.2
    urgency_factor: float = 0.5
    risk_aversion: float = 1.0
    max_adv_participation: float = 0.25

@app.post("/market-impact/estimate")
async def estimate_market_impact(
    market_data: MarketDataRequest,
    order: OrderRequest,
    model: str = Query(default="hybrid", description="Impact model: almgren_chriss, square_root, linear, power_law, hybrid")
):
    """
    Estimate market impact and execution costs for an order.
    
    Uses advanced market microstructure models including Almgren-Chriss and square root models
    to provide realistic execution cost estimates based on order size, market liquidity,
    and execution strategy.
    """
    try:
        # Convert request models to internal types
        market_microstructure = MarketMicrostructure(
            symbol=market_data.symbol,
            current_price=market_data.current_price,
            bid_price=market_data.bid_price,
            ask_price=market_data.ask_price,
            bid_size=market_data.bid_size,
            ask_size=market_data.ask_size,
            average_daily_volume=market_data.average_daily_volume,
            average_daily_value=market_data.average_daily_value,
            volatility_annualized=market_data.volatility_annualized,
            market_cap=market_data.market_cap,
            tick_size=market_data.tick_size
        )
        
        # Validate execution style
        try:
            execution_style = ExecutionStyle(order.execution_style.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid execution style: {order.execution_style}. Valid options: aggressive, moderate, passive, twap, vwap"
            )
        
        order_profile = OrderProfile(
            symbol=order.symbol,
            side=order.side.lower(),
            quantity=order.quantity,
            target_price=order.target_price,
            execution_style=execution_style,
            time_horizon_minutes=order.time_horizon_minutes,
            participation_rate=order.participation_rate,
            urgency_factor=order.urgency_factor,
            risk_aversion=order.risk_aversion,
            max_adv_participation=order.max_adv_participation
        )
        
        # Validate impact model
        try:
            impact_model = ImpactModel(model.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid impact model: {model}. Valid options: almgren_chriss, square_root, linear, power_law, hybrid"
            )
        
        # Estimate market impact
        impact_estimate = market_impact_model.estimate_market_impact(
            market_microstructure, order_profile, impact_model
        )
        
        return impact_estimate.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error estimating market impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/market-impact/optimize-execution")
async def optimize_execution_strategy(
    market_data: MarketDataRequest,
    order: OrderRequest
):
    """
    Optimize execution strategy to minimize total execution costs.
    
    Analyzes different execution styles (aggressive, moderate, passive, TWAP, VWAP)
    and recommends the optimal approach based on market conditions and order characteristics.
    """
    try:
        # Convert request models to internal types
        market_microstructure = MarketMicrostructure(
            symbol=market_data.symbol,
            current_price=market_data.current_price,
            bid_price=market_data.bid_price,
            ask_price=market_data.ask_price,
            bid_size=market_data.bid_size,
            ask_size=market_data.ask_size,
            average_daily_volume=market_data.average_daily_volume,
            average_daily_value=market_data.average_daily_value,
            volatility_annualized=market_data.volatility_annualized,
            market_cap=market_data.market_cap,
            tick_size=market_data.tick_size
        )
        
        try:
            execution_style = ExecutionStyle(order.execution_style.lower())
        except ValueError:
            execution_style = ExecutionStyle.MODERATE
        
        order_profile = OrderProfile(
            symbol=order.symbol,
            side=order.side.lower(),
            quantity=order.quantity,
            target_price=order.target_price,
            execution_style=execution_style,
            time_horizon_minutes=order.time_horizon_minutes,
            participation_rate=order.participation_rate,
            urgency_factor=order.urgency_factor,
            risk_aversion=order.risk_aversion,
            max_adv_participation=order.max_adv_participation
        )
        
        # Optimize execution strategy
        optimization_result = market_impact_model.optimize_execution_strategy(
            market_microstructure, order_profile
        )
        
        return optimization_result
        
    except Exception as e:
        logger.error(f"Error optimizing execution strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-impact/capabilities")
async def get_market_impact_capabilities():
    """
    Get market impact modeling capabilities and configuration.
    
    Provides information about available impact models, execution styles, and parameters.
    """
    try:
        return {
            "service": "Market Impact and Execution Cost Modeling",
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "capabilities": {
                "almgren_chriss_model": True,
                "square_root_model": True,
                "linear_model": True,
                "power_law_model": True,
                "hybrid_model": True,
                "execution_optimization": True,
                "microstructure_analysis": True,
                "adv_based_sizing": True,
                "volatility_adjustments": True,
                "bid_ask_spread_modeling": True
            },
            "impact_models": {
                "almgren_chriss": {
                    "description": "Optimal execution model with temporary and permanent impact",
                    "best_for": "Large institutional orders with flexible timing",
                    "parameters": ["eta", "gamma", "sigma_factor"]
                },
                "square_root": {
                    "description": "Square root market impact model (Barra/MSCI)",
                    "best_for": "General purpose impact estimation",
                    "parameters": ["alpha", "beta", "min_impact"]
                },
                "linear": {
                    "description": "Simple linear impact model",
                    "best_for": "Quick estimates and smaller orders",
                    "parameters": ["base_impact_per_adv"]
                },
                "power_law": {
                    "description": "Power law impact model",
                    "best_for": "Academic research and specialized analysis",
                    "parameters": ["alpha", "base_coefficient"]
                },
                "hybrid": {
                    "description": "Adaptive model combining Almgren-Chriss and square root",
                    "best_for": "Production trading systems",
                    "parameters": ["dynamic_weighting_based_on_order_size"]
                }
            },
            "execution_styles": {
                "aggressive": {
                    "description": "Market orders, immediate execution",
                    "typical_cost": "High impact, low timing risk",
                    "best_for": "Urgent orders, high alpha signals"
                },
                "moderate": {
                    "description": "Mix of market and limit orders",
                    "typical_cost": "Balanced impact and timing risk",
                    "best_for": "General purpose trading"
                },
                "passive": {
                    "description": "Limit orders, patient execution",
                    "typical_cost": "Low impact, higher timing risk",
                    "best_for": "Large orders, portfolio rebalancing"
                },
                "twap": {
                    "description": "Time-weighted average price execution",
                    "typical_cost": "Moderate impact, controlled timing",
                    "best_for": "Regular execution over time"
                },
                "vwap": {
                    "description": "Volume-weighted average price execution",
                    "typical_cost": "Follows market volume patterns",
                    "best_for": "Large orders following market rhythm"
                }
            },
            "cost_components": {
                "temporary_impact": "Immediate price movement that reverts",
                "permanent_impact": "Lasting price change from information revelation",
                "bid_ask_cost": "Cost of crossing the bid-ask spread",
                "timing_cost": "Risk of adverse price movement during execution",
                "opportunity_cost": "Cost of delayed execution"
            },
            "market_microstructure_factors": {
                "adv_participation": "Order size relative to average daily volume",
                "spread_volatility": "Variability in bid-ask spreads",
                "depth_imbalance": "Difference between bid and ask quantities",
                "liquidity_score": "Overall market liquidity assessment",
                "volatility_regime": "Current market volatility environment"
            },
            "optimization_features": {
                "execution_style_comparison": "Compare all execution styles for optimal choice",
                "cost_savings_estimation": "Quantify potential savings from optimization",
                "market_condition_adaptation": "Adjust recommendations based on current conditions",
                "risk_cost_tradeoff": "Balance execution speed vs. cost"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting market impact capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio Construction Endpoints

class PortfolioAllocationRequest(BaseModel):
    symbols: List[str]
    returns_data: Dict[str, List[float]]  # symbol -> list of daily returns
    returns_dates: Optional[List[str]] = None  # ISO date strings
    current_positions: Optional[Dict[str, float]] = None  # symbol -> weight
    config: Optional[Dict[str, Any]] = None  # Portfolio configuration overrides
    
    class Config:
        schema_extra = {
            "example": {
                "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
                "returns_data": {
                    "AAPL": [0.01, -0.005, 0.02, 0.015, -0.01],
                    "GOOGL": [0.015, 0.005, -0.01, 0.02, 0.005],
                    "MSFT": [0.005, 0.01, 0.015, -0.005, 0.02],
                    "TSLA": [0.03, -0.02, 0.05, -0.01, 0.015],
                    "AMZN": [0.02, 0.01, -0.015, 0.025, -0.005]
                },
                "current_positions": {
                    "AAPL": 0.2,
                    "GOOGL": 0.2,
                    "MSFT": 0.2,
                    "TSLA": 0.2,
                    "AMZN": 0.2
                },
                "config": {
                    "target_volatility": 0.12,
                    "max_single_weight": 0.06,
                    "covariance_method": "ledoit_wolf"
                }
            }
        }

@app.post("/portfolio/allocate")
async def create_portfolio_allocation_endpoint(request: PortfolioAllocationRequest):
    """
    Create optimal portfolio allocation using ERC with volatility targeting.
    
    This endpoint implements Equal Risk Contribution (ERC) portfolio optimization with:
    - Ledoit-Wolf or OAS covariance estimation
    - Volatility targeting (default 10% annual)
    - Exposure and correlation caps
    - Risk metrics and constraint monitoring
    
    Returns portfolio weights, risk metrics, and constraint validation flags.
    """
    try:
        # Convert returns data to pandas Series
        asset_returns = {}
        
        if request.returns_dates:
            # Use provided dates
            dates = pd.to_datetime(request.returns_dates)
        else:
            # Generate date range
            dates = pd.date_range(
                end=datetime.now(),
                periods=len(next(iter(request.returns_data.values()))),
                freq='D'
            )
        
        for symbol in request.symbols:
            if symbol in request.returns_data:
                returns_list = request.returns_data[symbol]
                asset_returns[symbol] = pd.Series(returns_list, index=dates[:len(returns_list)])
        
        if not asset_returns:
            raise HTTPException(
                status_code=400,
                detail="No valid returns data provided for any symbols"
            )
        
        # Call portfolio construction
        result = create_portfolio_allocation(
            asset_returns=asset_returns,
            current_positions=request.current_positions,
            config_overrides=request.config
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating portfolio allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/config-schema")
async def get_portfolio_config_schema():
    """
    Get portfolio construction configuration schema.
    
    Returns available configuration parameters and their default values.
    """
    try:
        return {
            "configuration_parameters": {
                "target_volatility": {
                    "description": "Target annual portfolio volatility",
                    "type": "float",
                    "default": 0.10,
                    "range": [0.05, 0.50],
                    "example": 0.12
                },
                "vol_tolerance": {
                    "description": "Tolerance around target volatility",
                    "type": "float", 
                    "default": 0.01,
                    "range": [0.005, 0.05],
                    "example": 0.015
                },
                "covariance_method": {
                    "description": "Covariance estimation method",
                    "type": "string",
                    "default": "ledoit_wolf",
                    "options": ["ledoit_wolf", "oas", "empirical"],
                    "example": "ledoit_wolf"
                },
                "lookback_days": {
                    "description": "Number of days for return history",
                    "type": "integer",
                    "default": 252,
                    "range": [60, 1000],
                    "example": 180
                },
                "max_single_weight": {
                    "description": "Maximum weight per asset",
                    "type": "float",
                    "default": 0.05,
                    "range": [0.01, 0.50],
                    "example": 0.08
                },
                "max_sector_weight": {
                    "description": "Maximum weight per sector",
                    "type": "float",
                    "default": 0.25,
                    "range": [0.05, 1.0],
                    "example": 0.30
                },
                "max_beta": {
                    "description": "Maximum portfolio beta",
                    "type": "float",
                    "default": 1.2,
                    "range": [0.5, 2.0],
                    "example": 1.5
                },
                "max_turnover": {
                    "description": "Maximum portfolio turnover per rebalance",
                    "type": "float",
                    "default": 0.5,
                    "range": [0.1, 2.0],
                    "example": 0.3
                },
                "risk_aversion": {
                    "description": "Risk aversion parameter for optimization",
                    "type": "float",
                    "default": 1e-6,
                    "range": [1e-8, 1e-3],
                    "example": 5e-6
                }
            },
            "constraint_types": {
                "exposure_constraints": {
                    "max_long_exposure": "Maximum long exposure (default 1.0)",
                    "max_short_exposure": "Maximum short exposure (default 0.3)"
                },
                "sector_constraints": {
                    "description": "Per-sector exposure limits",
                    "automatic": "Detected from asset metadata"
                },
                "beta_constraints": {
                    "description": "Portfolio beta limits",
                    "requires": "Asset beta data"
                }
            },
            "risk_metrics_calculated": {
                "expected_volatility": "Annualized portfolio volatility",
                "diversification_ratio": "Ratio of weighted avg vol to portfolio vol",
                "effective_assets": "Effective number of assets (concentration measure)",
                "max_drawdown": "Historical maximum drawdown",
                "vol_target_achieved": "Whether volatility target was achieved"
            },
            "example_request": {
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "returns_data": {
                    "AAPL": [0.01, -0.005, 0.02],
                    "GOOGL": [0.015, 0.005, -0.01], 
                    "MSFT": [0.005, 0.01, 0.015]
                },
                "config": {
                    "target_volatility": 0.15,
                    "max_single_weight": 0.4,
                    "covariance_method": "ledoit_wolf"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio config schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Borrow/Locate & Hard-To-Borrow Fees Endpoints

class BorrowCheckRequest(BaseModel):
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str = "market"
    urgency: str = "normal"
    account_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "quantity": 1000,
                "side": "sell",
                "order_type": "market",
                "urgency": "normal"
            }
        }

class PortfolioBorrowRequest(BaseModel):
    positions: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "positions": [
                    {"symbol": "AAPL", "side": "sell", "quantity": 1000, "position_value": 150000, "holding_days": 30},
                    {"symbol": "TSLA", "side": "sell", "quantity": 500, "position_value": 100000, "holding_days": 15},
                    {"symbol": "MSFT", "side": "buy", "quantity": 800, "position_value": 240000, "holding_days": 45}
                ]
            }
        }

@app.post("/borrow/check")
async def check_borrow_availability(request: BorrowCheckRequest):
    """
    Check borrow availability and fees for a single position.
    
    This endpoint validates whether a short position can be executed by checking:
    - Borrow availability from configured providers
    - Current borrow rates and fees
    - Quantity limits and constraints
    
    Returns borrow status, fees, and validation results.
    """
    try:
        borrow_request = BorrowRequest(
            symbol=request.symbol,
            quantity=request.quantity,
            side=request.side,
            order_type=request.order_type,
            urgency=request.urgency,
            account_id=request.account_id
        )
        
        # Check borrow availability
        borrow_info = await borrow_checker.check_borrow_availability(borrow_request)
        
        # Validate the order
        validation = await borrow_checker.validate_short_order(
            request.symbol, 
            request.quantity, 
            request.account_id
        )
        
        return {
            "symbol": request.symbol,
            "quantity": request.quantity,
            "side": request.side,
            "borrow_info": {
                "status": borrow_info.status.value,
                "available": borrow_info.is_available,
                "quantity_available": borrow_info.quantity_available,
                "borrow_rate_annual": borrow_info.borrow_rate,
                "locate_fee": borrow_info.locate_fee,
                "rebate_rate": borrow_info.rebate_rate,
                "daily_cost_per_dollar": borrow_info.daily_cost_per_dollar,
                "provider": borrow_info.source,
                "updated_at": borrow_info.updated_at.isoformat()
            },
            "validation": validation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking borrow availability: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/borrow/calculate-costs")
async def calculate_borrow_costs(
    symbol: str,
    position_value: float,
    holding_days: int,
    side: str = "sell"
):
    """
    Calculate detailed borrow costs for a position.
    
    Provides comprehensive cost breakdown including:
    - Daily borrowing fees
    - Total cost over holding period
    - Locate fees
    - Net position value after costs
    """
    try:
        if side.lower() == 'buy':
            return {
                "symbol": symbol,
                "position_value": position_value,
                "holding_days": holding_days,
                "costs": {
                    "daily_fee": 0.0,
                    "total_borrow_cost": 0.0,
                    "locate_fee": 0.0,
                    "net_position_value": position_value,
                    "borrow_rate_annual": 0.0,
                    "note": "No borrow costs for long positions"
                }
            }
        
        # Get borrow info for short position
        borrow_request = BorrowRequest(
            symbol=symbol,
            quantity=int(abs(position_value) / 100),  # Estimate shares
            side=side
        )
        
        borrow_info = await borrow_checker.check_borrow_availability(borrow_request)
        costs = borrow_checker.calculate_borrow_cost(symbol, position_value, holding_days, borrow_info)
        
        return {
            "symbol": symbol,
            "position_value": position_value,
            "holding_days": holding_days,
            "side": side,
            "borrow_available": borrow_info.is_available,
            "costs": costs,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating borrow costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/borrow/validate-portfolio")
async def validate_portfolio_borrow(request: PortfolioBorrowRequest):
    """
    Validate borrow availability for an entire portfolio.
    
    Checks all short positions in a portfolio and provides:
    - Individual position validation
    - Portfolio-level summary statistics
    - Cost calculations for all positions
    - Blocked position identification
    """
    try:
        # Validate short positions
        portfolio_validation = await validate_portfolio_shorts(request.positions, borrow_checker)
        
        # Calculate costs for all positions
        costs = await borrow_checker.get_borrow_costs_batch(request.positions)
        
        # Combine results
        result = portfolio_validation.copy()
        result['borrow_costs'] = costs
        
        # Calculate total portfolio costs
        total_daily_fees = sum(cost.get('daily_fee', 0) for cost in costs.values())
        total_borrow_costs = sum(cost.get('total_borrow_cost', 0) for cost in costs.values())
        total_locate_fees = sum(cost.get('locate_fee', 0) for cost in costs.values())
        
        result['portfolio_summary'] = {
            **result['summary'],
            'total_daily_fees': total_daily_fees,
            'total_borrow_costs': total_borrow_costs,
            'total_locate_fees': total_locate_fees,
            'total_position_value': sum(p.get('position_value', 0) for p in request.positions),
            'cost_as_percentage': (total_borrow_costs / sum(abs(p.get('position_value', 1)) for p in request.positions)) * 100
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating portfolio borrow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/borrow/block-symbol")
async def block_symbol_for_shorting(
    symbol: str,
    reason: str = "Risk management decision"
):
    """
    Block a symbol from short selling.
    
    Adds a symbol to the blocked list, preventing any short orders
    from being validated for execution.
    """
    try:
        borrow_checker.block_symbol(symbol, reason)
        
        return {
            "symbol": symbol,
            "status": "blocked",
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error blocking symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/borrow/unblock-symbol")
async def unblock_symbol_for_shorting(symbol: str):
    """
    Unblock a symbol for short selling.
    
    Removes a symbol from the blocked list, allowing short orders
    to be validated normally.
    """
    try:
        borrow_checker.unblock_symbol(symbol)
        
        return {
            "symbol": symbol,
            "status": "unblocked",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error unblocking symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/borrow/status")
async def get_borrow_service_status():
    """
    Get borrow service status and configuration.
    
    Returns information about available providers, blocked symbols,
    and service health status.
    """
    try:
        return {
            "service": "Borrow/Locate & Hard-To-Borrow Fees",
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "providers": [provider.name for provider in borrow_checker.providers],
            "blocked_symbols": list(borrow_checker.blocked_symbols),
            "capabilities": {
                "borrow_availability_check": True,
                "cost_calculation": True,
                "portfolio_validation": True,
                "real_time_rates": True,
                "multiple_providers": True,
                "symbol_blocking": True,
                "fee_embedding": True
            },
            "borrow_statuses": {
                "available": "Borrow is available for short selling",
                "unavailable": "Borrow is not available",
                "limited": "Limited quantity available",
                "hard_to_borrow": "Available but expensive (HTB)",
                "easy_to_borrow": "Available at low cost (ETB)",
                "unknown": "Status could not be determined"
            },
            "fee_components": {
                "borrow_rate": "Annual percentage rate for borrowing",
                "locate_fee": "One-time fee for locating shares",
                "rebate_rate": "Interest earned on cash collateral",
                "daily_fee": "Daily cost per dollar of position"
            },
            "validation_gates": {
                "availability_check": "Blocks orders if borrow unavailable",
                "quantity_limits": "Enforces maximum borrowable quantity",
                "blocked_symbols": "Prevents trading in risk-managed symbols",
                "fee_disclosure": "Records all costs in trade records"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting borrow service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="info")