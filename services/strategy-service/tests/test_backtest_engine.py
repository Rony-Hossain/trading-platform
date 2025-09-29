import asyncio
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "app" / "engines" / "backtest_engine.py"


def _ensure_schema_module() -> None:
    if "services.strategy_service.app.models.schemas" in sys.modules:
        return

    services_mod = sys.modules.setdefault("services", types.ModuleType("services"))
    if not hasattr(services_mod, "__path__"):
        services_mod.__path__ = []  # type: ignore[attr-defined]

    strategy_pkg = sys.modules.setdefault("services.strategy_service", types.ModuleType("services.strategy_service"))
    if not hasattr(strategy_pkg, "__path__"):
        strategy_pkg.__path__ = [str(ROOT)]  # type: ignore[attr-defined]

    app_pkg = sys.modules.setdefault("services.strategy_service.app", types.ModuleType("services.strategy_service.app"))
    if not hasattr(app_pkg, "__path__"):
        app_pkg.__path__ = [str(ROOT / "app")]  # type: ignore[attr-defined]

    engines_pkg = sys.modules.setdefault("services.strategy_service.app.engines", types.ModuleType("services.strategy_service.app.engines"))
    if not hasattr(engines_pkg, "__path__"):
        engines_pkg.__path__ = [str(ROOT / "app" / "engines")]  # type: ignore[attr-defined]

    models_pkg = sys.modules.setdefault("services.strategy_service.app.models", types.ModuleType("services.strategy_service.app.models"))
    if not hasattr(models_pkg, "__path__"):
        models_pkg.__path__ = [str(ROOT / "app" / "models")]  # type: ignore[attr-defined]

    schemas_mod = types.ModuleType("services.strategy_service.app.models.schemas")

    @dataclass
    class Trade:
        symbol: str
        side: str
        quantity: int
        price: float
        timestamp: Any
        commission: float = 0.0
        slippage: float = 0.0
        total_cost: float = 0.0

    @dataclass
    class Position:
        symbol: str
        quantity: int
        average_price: float

    @dataclass
    class Backtest:
        id: int
        strategy_id: int
        symbol: str
        start_date: Any
        end_date: Any
        initial_capital: float
        parameters: Optional[Dict[str, Any]] = None

    @dataclass
    class BacktestCreate:
        strategy_id: int
        symbol: str
        start_date: Any
        end_date: Any
        initial_capital: float
        parameters: Optional[Dict[str, Any]] = None

    @dataclass
    class PerformanceMetrics:
        total_return: float = 0.0

    @dataclass
    class OptimizationRequest:
        strategy_id: int

    @dataclass
    class RiskMetrics:
        backtest_id: int
        value_at_risk: Optional[float] = None

    @dataclass
    class PortfolioSnapshot:
        timestamp: Any
        portfolio_value: float

    @dataclass
    class BacktestResult:
        backtest_id: int
        total_return: float
        annual_return: float
        sharpe_ratio: float
        max_drawdown: float
        win_rate: float
        profit_factor: float
        total_trades: int
        final_portfolio_value: float
        trades: List[Trade]
        daily_returns: List[float]
        benchmark_returns: List[float]
        metadata: Dict[str, Any] = field(default_factory=dict)
        drawdown_series: Optional[List[float]] = None

    schemas_mod.Trade = Trade
    schemas_mod.Position = Position
    schemas_mod.Backtest = Backtest
    schemas_mod.BacktestCreate = BacktestCreate
    schemas_mod.BacktestResult = BacktestResult
    schemas_mod.PerformanceMetrics = PerformanceMetrics
    schemas_mod.OptimizationRequest = OptimizationRequest
    schemas_mod.RiskMetrics = RiskMetrics
    schemas_mod.PortfolioSnapshot = PortfolioSnapshot

    sys.modules["services.strategy_service.app.models.schemas"] = schemas_mod


def _load_backtest_engine():
    _ensure_schema_module()
    name = "services.strategy_service.app.engines.backtest_engine"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


engine_module = _load_backtest_engine()
BacktestEngine = engine_module.BacktestEngine


class SingleBuyStrategy:
    def __init__(self, quantity: int, symbol: str = "TEST") -> None:
        self.quantity = quantity
        self.symbol = symbol
        self.has_bought = False

    def generate_signals(self, history: pd.DataFrame, state) -> dict:
        current_position = state.positions.get(self.symbol, 0)
        if not self.has_bought and current_position <= 0:
            self.has_bought = True
            return {"BUY": {"quantity": self.quantity}}
        return {}


@pytest.mark.asyncio
async def test_daily_loss_circuit_breaker_triggers_and_flattens(monkeypatch):
    engine = BacktestEngine()

    price_series = pd.DataFrame(
        {
            "open": [100.0, 80.0],
            "high": [100.0, 80.0],
            "low": [100.0, 80.0],
            "close": [100.0, 80.0],
            "volume": [1_000, 1_000],
        },
        index=pd.to_datetime(["2023-01-02", "2023-01-03"]),
    )

    benchmark = price_series.copy()

    async def fake_load_market_data(symbol: str, start_date, end_date):
        return price_series.copy() if symbol == "TEST" else benchmark.copy()

    monkeypatch.setattr(engine, "load_market_data", fake_load_market_data)
    monkeypatch.setattr(engine, "load_strategy", lambda sid, params: SingleBuyStrategy(quantity=10))

    backtest = SimpleNamespace(
        id=1,
        strategy_id=1,
        symbol="TEST",
        start_date=datetime(2023, 1, 2),
        end_date=datetime(2023, 1, 3),
        initial_capital=100_000.0,
        parameters={
            "risk": {
                "daily_loss_limit": 0.05,
                "resume_after_hit": False,
            }
        },
    )

    result = await engine.run_backtest(backtest, db=None)

    breakers = result.metadata["circuit_breakers"]
    assert any(entry["type"] == "daily_loss" for entry in breakers)
    assert len(result.trades) >= 2  # initial BUY plus forced liquidation


@pytest.mark.asyncio
async def test_slippage_configuration_applied(monkeypatch):
    engine = BacktestEngine()

    price_series = pd.DataFrame(
        {
            "open": [50.0],
            "high": [50.0],
            "low": [50.0],
            "close": [50.0],
            "volume": [1_000],
        },
        index=pd.to_datetime(["2023-01-02"]),
    )

    async def fake_load_market_data(symbol: str, start_date, end_date):
        return price_series.copy()

    monkeypatch.setattr(engine, "load_market_data", fake_load_market_data)
    monkeypatch.setattr(engine, "load_strategy", lambda sid, params: SingleBuyStrategy(quantity=10))

    slippage_cfg = {"type": "per_share", "per_share": 1.5}

    backtest = SimpleNamespace(
        id=2,
        strategy_id=1,
        symbol="TEST",
        start_date=datetime(2023, 1, 2),
        end_date=datetime(2023, 1, 2),
        initial_capital=10_000.0,
        parameters={
            "execution": {"slippage": slippage_cfg}
        },
    )

    result = await engine.run_backtest(backtest, db=None)

    assert result.metadata["slippage_config"] == slippage_cfg
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert pytest.approx(trade.slippage, rel=1e-6) == 1.5 * trade.quantity
