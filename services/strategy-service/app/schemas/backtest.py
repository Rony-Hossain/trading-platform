"""
Strategy Service Schemas - Data models for backtesting and strategy management
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

class TradeType(str, Enum):
    BUY = "buy"
    SELL = "sell"

class StrategyStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"

class Trade(BaseModel):
    symbol: str
    trade_type: TradeType
    quantity: int
    price: Decimal
    timestamp: datetime
    commission: Optional[Decimal] = None

class Position(BaseModel):
    symbol: str
    quantity: int
    average_price: Decimal
    current_price: Optional[Decimal] = None
    market_value: Optional[Decimal] = None
    unrealized_pnl: Optional[Decimal] = None

class PerformanceMetrics(BaseModel):
    total_return: Decimal
    total_return_percent: Decimal
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    volatility: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
    profit_factor: Optional[Decimal] = None

class RiskMetrics(BaseModel):
    var_95: Optional[Decimal] = None
    var_99: Optional[Decimal] = None
    expected_shortfall: Optional[Decimal] = None
    beta: Optional[Decimal] = None
    correlation: Optional[Decimal] = None

class BacktestCreate(BaseModel):
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: Decimal
    symbols: List[str]
    parameters: Dict[str, Any] = {}

class BacktestResult(BaseModel):
    id: str
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: Decimal
    final_capital: Decimal
    total_trades: int
    performance_metrics: PerformanceMetrics
    risk_metrics: RiskMetrics
    trades: List[Trade]
    created_at: datetime

class Backtest(BaseModel):
    id: str
    strategy_name: str
    status: StrategyStatus
    start_date: date
    end_date: date
    initial_capital: Decimal
    parameters: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[BacktestResult] = None

class OptimizationRequest(BaseModel):
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: Decimal
    symbols: List[str]
    parameter_ranges: Dict[str, Any]
    optimization_metric: str = "sharpe_ratio"
    max_iterations: int = 100