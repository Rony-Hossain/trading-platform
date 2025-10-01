from .backtest import (
    Backtest, BacktestCreate, BacktestResult, Trade, Position,
    PerformanceMetrics, OptimizationRequest, RiskMetrics,
    TradeType, StrategyStatus
)

# Additional schema classes that main.py expects
Strategy = Backtest  # Alias for now
StrategyCreate = BacktestCreate  # Alias for now
StrategyUpdate = BacktestCreate  # Alias for now
PortfolioSnapshot = dict  # Simple dict for now
OptimizationResult = dict  # Simple dict for now