"""
REST API Endpoints for OOS Validation and Paper Trading

This module provides comprehensive API endpoints for out-of-sample validation
and paper trading functionality with rigorous statistical testing requirements.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import asyncio

from ..services.oos_validation import (
    OOSValidator, ValidationThresholds, ValidationResults, ValidationStatus,
    BenchmarkType, create_oos_validator, require_oos_validation
)
from ..services.paper_trading import (
    PaperTradingEngine, OrderType, OrderSide, MarketData,
    create_paper_trading_engine
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/validation", tags=["Strategy Validation"])

# Pydantic models for API
class ValidationThresholdsRequest(BaseModel):
    min_sharpe_ratio: float = Field(default=1.0, ge=0.1, le=5.0)
    min_t_statistic: float = Field(default=2.0, ge=1.0, le=5.0)
    min_hit_rate: float = Field(default=0.55, ge=0.5, le=1.0)
    max_drawdown_threshold: float = Field(default=0.15, ge=0.01, le=0.5)
    min_calmar_ratio: float = Field(default=0.5, ge=0.1, le=2.0)
    min_information_ratio: float = Field(default=0.3, ge=0.0, le=2.0)
    min_validation_period_months: int = Field(default=6, ge=3, le=24)
    min_trades: int = Field(default=20, ge=10, le=1000)
    significance_level: float = Field(default=0.05, ge=0.01, le=0.1)

class StrategySignalRequest(BaseModel):
    date: datetime
    symbol: str
    signal: float = Field(ge=-1.0, le=1.0)  # Normalized signal strength
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OOSValidationRequest(BaseModel):
    strategy_id: str = Field(min_length=1, max_length=100)
    signals: List[StrategySignalRequest] = Field(min_items=10)
    oos_start_date: datetime
    validation_period_months: int = Field(default=12, ge=6, le=24)
    benchmark_symbols: List[str] = Field(default=["SPY"])
    custom_thresholds: Optional[ValidationThresholdsRequest] = None

class ValidationResultsResponse(BaseModel):
    strategy_id: str
    validation_status: str
    oos_period_start: datetime
    oos_period_end: datetime
    strategy_performance: Dict[str, Any]
    benchmark_comparison: Dict[str, Any]
    statistical_tests: List[Dict[str, Any]]
    information_ratio: float
    excess_return_t_stat: float
    overfitting_score: float
    validation_summary: Dict[str, Any]
    recommendations: List[str]
    risk_warnings: List[str]

class PaperTradingOrderRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=10)
    side: str = Field(regex="^(buy|sell)$")
    quantity: int = Field(gt=0, le=100000)
    order_type: str = Field(default="market", regex="^(market|limit|stop|stop_limit)$")
    price: Optional[float] = Field(default=None, gt=0)
    stop_price: Optional[float] = Field(default=None, gt=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MarketDataUpdate(BaseModel):
    symbol: str = Field(min_length=1, max_length=10)
    timestamp: datetime
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    last: float = Field(gt=0)
    volume: int = Field(ge=0)
    bid_size: int = Field(default=100, ge=0)
    ask_size: int = Field(default=100, ge=0)

class PaperAccountResponse(BaseModel):
    account_id: str
    strategy_id: str
    account_value: Dict[str, float]
    performance: Dict[str, float]
    positions: List[Dict[str, Any]]
    open_orders: List[Dict[str, Any]]
    order_count: int
    last_updated: str

# Global instances
paper_trading_engine = None
oos_validators_cache: Dict[str, OOSValidator] = {}

async def get_paper_trading_engine():
    """Get or create paper trading engine instance"""
    global paper_trading_engine
    if paper_trading_engine is None:
        paper_trading_engine = await create_paper_trading_engine(initial_balance=100000.0)
    return paper_trading_engine

async def get_oos_validator(thresholds: Optional[ValidationThresholds] = None):
    """Get or create OOS validator with caching"""
    global oos_validators_cache
    
    # Use default thresholds if none provided
    if thresholds is None:
        thresholds = ValidationThresholds()
    
    # Create cache key from thresholds
    cache_key = f"{thresholds.min_sharpe_ratio}_{thresholds.min_t_statistic}_{thresholds.min_hit_rate}"
    
    if cache_key not in oos_validators_cache:
        oos_validators_cache[cache_key] = await create_oos_validator(thresholds)
    
    return oos_validators_cache[cache_key]

# Validation endpoints
@router.post("/oos-validate", response_model=ValidationResultsResponse)
async def perform_oos_validation(
    request: OOSValidationRequest,
    background_tasks: BackgroundTasks
):
    """
    Perform comprehensive out-of-sample validation of a trading strategy
    
    This endpoint implements rigorous statistical testing to prevent overfitting
    and ensures strategies meet minimum performance thresholds before production.
    """
    try:
        logger.info(f"Starting OOS validation for strategy {request.strategy_id}")
        
        # Convert thresholds if provided
        thresholds = None
        if request.custom_thresholds:
            thresholds = ValidationThresholds(
                min_sharpe_ratio=request.custom_thresholds.min_sharpe_ratio,
                min_t_statistic=request.custom_thresholds.min_t_statistic,
                min_hit_rate=request.custom_thresholds.min_hit_rate,
                max_drawdown_threshold=request.custom_thresholds.max_drawdown_threshold,
                min_calmar_ratio=request.custom_thresholds.min_calmar_ratio,
                min_information_ratio=request.custom_thresholds.min_information_ratio,
                min_validation_period_months=request.custom_thresholds.min_validation_period_months,
                min_trades=request.custom_thresholds.min_trades,
                significance_level=request.custom_thresholds.significance_level
            )
        
        # Get validator
        validator = await get_oos_validator(thresholds)
        
        # Convert signals to DataFrame
        signals_data = []
        for signal in request.signals:
            signals_data.append({
                'date': signal.date,
                'symbol': signal.symbol,
                'signal': signal.signal,
                'confidence': signal.confidence,
                'metadata': signal.metadata
            })
        
        signals_df = pd.DataFrame(signals_data)
        
        # TODO: In production, fetch real price and benchmark data
        # For now, generate sample data for demonstration
        price_data = await _generate_sample_price_data(
            signals_df['symbol'].unique().tolist(),
            request.oos_start_date,
            request.oos_start_date + timedelta(days=request.validation_period_months * 30)
        )
        
        benchmark_data = await _generate_sample_benchmark_data(
            request.benchmark_symbols[0],
            request.oos_start_date,
            request.oos_start_date + timedelta(days=request.validation_period_months * 30)
        )
        
        # Perform validation
        validation_results = await validator.validate_strategy(
            strategy_id=request.strategy_id,
            strategy_signals=signals_df,
            price_data=price_data,
            benchmark_data=benchmark_data,
            oos_start_date=request.oos_start_date,
            validation_period_months=request.validation_period_months
        )
        
        # Format response
        response = ValidationResultsResponse(
            strategy_id=validation_results.strategy_id,
            validation_status=validation_results.validation_status.value,
            oos_period_start=validation_results.oos_period_start,
            oos_period_end=validation_results.oos_period_end,
            strategy_performance={
                "total_return": validation_results.strategy_performance.total_return,
                "annualized_return": validation_results.strategy_performance.annualized_return,
                "sharpe_ratio": validation_results.strategy_performance.sharpe_ratio,
                "max_drawdown": validation_results.strategy_performance.max_drawdown,
                "hit_rate": validation_results.strategy_performance.hit_rate,
                "total_trades": validation_results.strategy_performance.total_trades,
                "calmar_ratio": validation_results.strategy_performance.calmar_ratio
            },
            benchmark_comparison={
                bench.benchmark_type.value: {
                    "excess_return": validation_results.strategy_performance.annualized_return - bench.performance_metrics.annualized_return,
                    "excess_sharpe": validation_results.strategy_performance.sharpe_ratio - bench.performance_metrics.sharpe_ratio,
                    "tracking_error": bench.tracking_error
                }
                for bench in validation_results.benchmark_performances
            },
            statistical_tests=[
                {
                    "test_type": test.test_type,
                    "t_statistic": test.t_statistic,
                    "p_value": test.p_value,
                    "is_significant": test.is_significant,
                    "confidence_interval": [test.confidence_interval_lower, test.confidence_interval_upper]
                }
                for test in validation_results.statistical_tests
            ],
            information_ratio=validation_results.information_ratio,
            excess_return_t_stat=validation_results.excess_return_t_stat,
            overfitting_score=validation_results.overfitting_score,
            validation_summary=validation_results.validation_summary,
            recommendations=validation_results.recommendations,
            risk_warnings=validation_results.risk_warnings
        )
        
        logger.info(f"OOS validation complete for {request.strategy_id}: {validation_results.validation_status.value}")
        return response
        
    except Exception as e:
        logger.error(f"OOS validation failed for {request.strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/validation-status/{strategy_id}")
async def get_validation_status(strategy_id: str):
    """Get current validation status for a strategy"""
    try:
        # TODO: Query from database
        return {
            "strategy_id": strategy_id,
            "status": "not_validated",
            "message": "Strategy has not been validated yet"
        }
    except Exception as e:
        logger.error(f"Failed to get validation status for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/validation-requirements")
async def get_validation_requirements():
    """Get current validation requirements and thresholds"""
    default_thresholds = ValidationThresholds()
    
    return {
        "mandatory_requirements": {
            "min_sharpe_ratio": default_thresholds.min_sharpe_ratio,
            "min_t_statistic": default_thresholds.min_t_statistic,
            "min_hit_rate": default_thresholds.min_hit_rate,
            "max_drawdown_threshold": default_thresholds.max_drawdown_threshold,
            "min_trades": default_thresholds.min_trades,
            "statistical_significance": f"p < {default_thresholds.significance_level}"
        },
        "recommended_thresholds": {
            "min_calmar_ratio": default_thresholds.min_calmar_ratio,
            "min_information_ratio": default_thresholds.min_information_ratio,
            "min_validation_period_months": default_thresholds.min_validation_period_months
        },
        "overfitting_prevention": {
            "description": "Strategies with high overfitting risk will be flagged",
            "risk_factors": [
                "Extremely high Sharpe ratio (>3.0)",
                "Very high hit rate (>80%)",
                "Low number of trades (<50)",
                "Short validation period (<6 months)",
                "Extreme return skewness"
            ]
        }
    }

# Paper trading endpoints
@router.post("/paper-trading/create-account")
async def create_paper_account(
    strategy_id: str = Query(..., min_length=1, max_length=100),
    initial_balance: float = Query(default=100000.0, ge=10000.0, le=10000000.0)
):
    """Create new paper trading account for strategy validation"""
    try:
        engine = await get_paper_trading_engine()
        engine.initial_balance = initial_balance  # Update for this account
        
        account = await engine.create_paper_account(strategy_id)
        
        return {
            "account_id": account.account_id,
            "strategy_id": account.strategy_id,
            "initial_balance": account.initial_balance,
            "status": "active",
            "created_at": account.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create paper account for {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/paper-trading/{account_id}/place-order")
async def place_paper_order(
    account_id: str,
    order_request: PaperTradingOrderRequest
):
    """Place paper trading order"""
    try:
        engine = await get_paper_trading_engine()
        
        # Convert string enums
        side = OrderSide.BUY if order_request.side.lower() == "buy" else OrderSide.SELL
        order_type = OrderType(order_request.order_type.lower())
        
        order = await engine.place_order(
            account_id=account_id,
            symbol=order_request.symbol,
            side=side,
            quantity=order_request.quantity,
            order_type=order_type,
            price=order_request.price,
            stop_price=order_request.stop_price,
            metadata=order_request.metadata
        )
        
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "status": order.status.value,
            "created_at": order.created_at.isoformat(),
            "estimated_commission": order.commission
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to place order for account {account_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/paper-trading/{account_id}/cancel-order/{order_id}")
async def cancel_paper_order(account_id: str, order_id: str):
    """Cancel pending paper trading order"""
    try:
        engine = await get_paper_trading_engine()
        success = await engine.cancel_order(account_id, order_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Order not found or cannot be cancelled")
        
        return {"message": f"Order {order_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/paper-trading/market-data/update")
async def update_market_data(market_data: MarketDataUpdate):
    """Update market data for paper trading simulation"""
    try:
        engine = await get_paper_trading_engine()
        
        market_snapshot = MarketData(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            bid=market_data.bid,
            ask=market_data.ask,
            last=market_data.last,
            volume=market_data.volume,
            bid_size=market_data.bid_size,
            ask_size=market_data.ask_size
        )
        
        await engine.update_market_data(market_data.symbol, market_snapshot)
        
        return {
            "message": f"Market data updated for {market_data.symbol}",
            "timestamp": market_data.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update market data for {market_data.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/paper-trading/{account_id}/summary", response_model=PaperAccountResponse)
async def get_paper_account_summary(account_id: str):
    """Get comprehensive paper trading account summary"""
    try:
        engine = await get_paper_trading_engine()
        summary = engine.get_account_summary(account_id)
        
        return PaperAccountResponse(
            account_id=summary["account_id"],
            strategy_id=summary["strategy_id"],
            account_value=summary["account_value"],
            performance=summary["performance"],
            positions=summary["positions"],
            open_orders=summary["open_orders"],
            order_count=summary["order_count"],
            last_updated=summary["last_updated"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get account summary for {account_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/paper-trading/{account_id}/performance")
async def get_paper_trading_performance(
    account_id: str,
    start_date: Optional[datetime] = Query(None)
):
    """Get detailed performance metrics for paper trading account"""
    try:
        engine = await get_paper_trading_engine()
        performance = await engine.get_performance_metrics(account_id, start_date)
        
        return {
            "account_id": account_id,
            "performance_metrics": performance,
            "meets_validation_thresholds": {
                "sharpe_ratio": performance["sharpe_ratio"] >= 1.0,
                "win_rate": performance["win_rate"] >= 0.55,
                "total_trades": performance["total_trades"] >= 20,
                "max_drawdown": performance["max_drawdown"] <= 0.15
            },
            "analysis_date": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get performance metrics for {account_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Strategy deployment with validation enforcement
@router.post("/deploy-strategy/{strategy_id}")
@require_oos_validation(min_sharpe=1.0, min_t_stat=2.0)
async def deploy_strategy(strategy_id: str, deployment_config: Dict[str, Any] = None):
    """
    Deploy strategy to production (requires passing OOS validation)
    
    This endpoint is decorated with @require_oos_validation to enforce
    that strategies must pass validation before production deployment.
    """
    try:
        # This function will only execute if validation requirements are met
        logger.info(f"Deploying strategy {strategy_id} to production")
        
        return {
            "strategy_id": strategy_id,
            "status": "deployed",
            "message": "Strategy successfully deployed to production",
            "deployment_time": datetime.now().isoformat(),
            "validation_enforced": True
        }
        
    except ValueError as e:
        # Validation enforcement errors
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to deploy strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint for validation service"""
    return {
        "status": "healthy",
        "service": "validation_api",
        "validation_enforced": True,
        "paper_trading_active": paper_trading_engine is not None
    }

# Helper functions
async def _generate_sample_price_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Generate sample price data for validation (replace with real data fetch)"""
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    price_data = []
    
    for symbol in symbols:
        # Generate realistic price movements
        initial_price = np.random.uniform(50, 200)
        prices = [initial_price]
        
        for i in range(len(date_range) - 1):
            # Random walk with slight positive drift
            change = np.random.normal(0.0005, 0.02)  # 0.05% drift, 2% daily vol
            new_price = max(prices[-1] * (1 + change), 1.0)
            prices.append(new_price)
        
        # Calculate returns
        returns = [0.0] + [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        for i, (date, price, ret) in enumerate(zip(date_range, prices, returns)):
            price_data.append({
                'date': date,
                'symbol': symbol,
                'price': price,
                'return': ret,
                'volume': int(np.random.lognormal(12, 0.5))
            })
    
    return pd.DataFrame(price_data)

async def _generate_sample_benchmark_data(
    benchmark_symbol: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Generate sample benchmark data (replace with real data fetch)"""
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate market index with lower volatility
    initial_value = 300
    values = [initial_value]
    
    for i in range(len(date_range) - 1):
        change = np.random.normal(0.0003, 0.015)  # Lower vol for market
        new_value = max(values[-1] * (1 + change), 100)
        values.append(new_value)
    
    returns = [0.0] + [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
    
    benchmark_data = []
    for date, value, ret in zip(date_range, values, returns):
        benchmark_data.append({
            'date': date,
            'symbol': benchmark_symbol,
            'price': value,
            'return': ret
        })
    
    return pd.DataFrame(benchmark_data)

# Advanced Execution Modeling Endpoints

@router.post("/execution/calculate-costs")
async def calculate_execution_costs(
    symbol: str,
    quantity: int,
    price: float,
    side: str,
    exchange_type: str = "NYSE"
):
    """Calculate comprehensive execution costs including exchange fees"""
    try:
        from ..services.execution_modeling import create_execution_engine, ExchangeType
        
        # Convert string to ExchangeType enum
        exchange_enum = ExchangeType(exchange_type.upper())
        execution_engine = create_execution_engine(exchange_enum)
        
        # Calculate total costs
        costs = execution_engine.calculate_total_costs(symbol, quantity, price, side.lower())
        
        return {
            "success": True,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "side": side,
            "exchange_type": exchange_type,
            "costs": costs
        }
        
    except Exception as e:
        logger.error(f"Cost calculation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/execution/simulate-fill")
async def simulate_order_fill(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    limit_price: Optional[float] = None,
    exchange_type: str = "NYSE",
    bid_price: float = 100.0,
    ask_price: float = 100.05,
    bid_size: int = 1000,
    ask_size: int = 1000
):
    """Simulate realistic order fill with depth-aware execution"""
    try:
        from ..services.execution_modeling import (
            create_execution_engine, ExchangeType, OrderExecutionType, MarketDepth
        )
        from datetime import datetime
        
        # Convert enums
        exchange_enum = ExchangeType(exchange_type.upper())
        order_type_enum = OrderExecutionType.MARKET if order_type.lower() == "market" else OrderExecutionType.LIMIT
        
        execution_engine = create_execution_engine(exchange_enum)
        
        # Create market depth
        market_depth = MarketDepth(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=(bid_price + ask_price) / 2,
            volume=10000
        )
        
        # Execute order simulation
        result = await execution_engine.execute_order_async(
            symbol=symbol,
            side=side.lower(),
            quantity=quantity,
            order_type=order_type_enum,
            limit_price=limit_price,
            market_depth=market_depth
        )
        
        return {
            "success": True,
            "execution_result": result,
            "market_conditions": {
                "bid_price": bid_price,
                "ask_price": ask_price,
                "spread_bps": ((ask_price - bid_price) / bid_price) * 10000,
                "bid_size": bid_size,
                "ask_size": ask_size
            }
        }
        
    except Exception as e:
        logger.error(f"Fill simulation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/execution/fee-structures")
async def get_fee_structures():
    """Get comprehensive fee structures for all supported exchanges"""
    try:
        from ..services.execution_modeling import ExchangeType, create_execution_engine
        
        fee_structures = {}
        
        for exchange in ExchangeType:
            engine = create_execution_engine(exchange)
            fee_calc = engine.fee_calculator
            
            if hasattr(fee_calc, 'exchange_fees') and exchange in fee_calc.exchange_fees:
                fee_structure = fee_calc.exchange_fees[exchange]
                fee_structures[exchange.value] = {
                    "maker_fee": fee_structure.maker_fee,
                    "taker_fee": fee_structure.taker_fee,
                    "sec_fee": fee_structure.sec_fee,
                    "taf_fee": fee_structure.taf_fee,
                    "finra_orf": fee_structure.finra_orf,
                    "clearing_fee": fee_structure.clearing_fee,
                    "min_fee": getattr(fee_structure, 'min_fee', 0.0),
                    "max_fee": getattr(fee_structure, 'max_fee', float('inf'))
                }
        
        return {
            "success": True,
            "fee_structures": fee_structures,
            "description": "Comprehensive exchange fee structures including maker/taker fees, SEC fees, TAF, FINRA ORF, and clearing fees"
        }
        
    except Exception as e:
        logger.error(f"Fee structure retrieval failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@router.post("/execution/paper-trading-with-fees")
async def create_advanced_paper_account(
    strategy_id: str,
    initial_balance: float = 100000.0,
    exchange_type: str = "NYSE"
):
    """Create paper trading account with advanced execution modeling"""
    try:
        from ..services.execution_modeling import ExchangeType
        from ..services.paper_trading import create_paper_trading_engine
        
        # Convert to enum
        exchange_enum = ExchangeType(exchange_type.upper())
        
        # Create engine with advanced execution
        engine = await create_paper_trading_engine(initial_balance, exchange_enum)
        
        # Create account
        account = await engine.create_paper_account(strategy_id)
        
        return {
            "success": True,
            "account": {
                "account_id": account.account_id,
                "strategy_id": account.strategy_id,
                "initial_balance": account.initial_balance,
                "cash_balance": account.cash_balance,
                "total_value": account.total_value,
                "buying_power": account.buying_power,
                "exchange_type": exchange_type
            },
            "execution_engine": {
                "type": "AdvancedExecutionEngine",
                "exchange": exchange_type,
                "features": [
                    "Depth-aware slippage modeling",
                    "Comprehensive exchange fees",
                    "Fill probability curves",
                    "Realistic market impact",
                    "SEC/TAF/FINRA fee calculation"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Advanced paper account creation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }