"""
Scenario Simulation Engine
Replays historical crises and generates synthetic shocks for stress testing

Scenarios:
- 2008 Financial Crisis
- 2020 COVID Crash
- 2010 Flash Crash
- Rate Shock
- Geopolitical Crisis
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """Types of market scenarios"""
    FINANCIAL_CRISIS = "financial_crisis"
    COVID_CRASH = "covid_crash"
    FLASH_CRASH = "flash_crash"
    RATE_SHOCK = "rate_shock"
    GEOPOLITICAL = "geopolitical"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CUSTOM = "custom"


class ShockMagnitude(str, Enum):
    """Shock severity levels"""
    MILD = "mild"  # 1 standard deviation
    MODERATE = "moderate"  # 2 standard deviations
    SEVERE = "severe"  # 3 standard deviations
    EXTREME = "extreme"  # 4+ standard deviations


@dataclass
class MarketConditions:
    """Market conditions during scenario"""
    volatility_multiplier: float  # 1.0 = normal, 2.0 = 2x normal vol
    liquidity_multiplier: float  # 1.0 = normal, 0.5 = half normal liquidity
    spread_multiplier: float  # 1.0 = normal, 3.0 = 3x normal spread
    correlation_spike: float  # 0.0-1.0, how much correlations increase
    circuit_breaker_triggered: bool = False
    trading_halts: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result of scenario simulation"""
    scenario_id: str
    scenario_name: str
    duration_minutes: int
    pnl_impact: float
    max_drawdown: float
    volatility_realized: float
    sharpe_ratio: float
    risk_limit_breaches: int
    execution_failures: int
    system_alerts: int
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]


class ScenarioEngine:
    """
    Scenario simulation engine for stress testing

    Capabilities:
    - Replay historical crises
    - Generate synthetic shocks
    - Multi-asset simulation
    - Liquidity stress testing
    - System performance monitoring
    """

    def __init__(self):
        self.scenarios = self._init_scenarios()
        logger.info("Scenario engine initialized with %d scenarios", len(self.scenarios))

    def _init_scenarios(self) -> Dict[str, Dict]:
        """Initialize historical scenario definitions"""
        return {
            "2008_financial_crisis": {
                "name": "2008 Financial Crisis (Lehman Collapse)",
                "start_date": "2008-09-15",
                "duration_days": 30,
                "conditions": MarketConditions(
                    volatility_multiplier=4.0,  # VIX hit 80
                    liquidity_multiplier=0.3,  # Severe liquidity freeze
                    spread_multiplier=5.0,  # Spreads widened dramatically
                    correlation_spike=0.95,  # Everything moved together
                    circuit_breaker_triggered=True,
                    trading_halts=["LEH", "AIG", "WM"]
                ),
                "equity_drop_pct": -30.0,
                "credit_spread_widen_bps": 600,
                "description": "Lehman bankruptcy, credit freeze, panic selling"
            },
            "2020_covid_crash": {
                "name": "2020 COVID-19 Market Crash",
                "start_date": "2020-02-20",
                "duration_days": 20,
                "conditions": MarketConditions(
                    volatility_multiplier=3.5,
                    liquidity_multiplier=0.4,
                    spread_multiplier=4.0,
                    correlation_spike=0.90,
                    circuit_breaker_triggered=True,
                    trading_halts=[]
                ),
                "equity_drop_pct": -34.0,
                "volume_surge": 10.0,  # 10x normal volume
                "description": "Pandemic shutdown, fastest bear market in history"
            },
            "2010_flash_crash": {
                "name": "2010 Flash Crash",
                "start_date": "2010-05-06",
                "duration_minutes": 30,
                "conditions": MarketConditions(
                    volatility_multiplier=10.0,
                    liquidity_multiplier=0.1,  # Liquidity evaporated
                    spread_multiplier=20.0,
                    correlation_spike=0.99,
                    circuit_breaker_triggered=False,
                    trading_halts=["SPY", "QQQ"]
                ),
                "equity_drop_pct": -9.0,  # Drop in minutes
                "recovery_minutes": 20,  # Rapid recovery
                "description": "Algorithmic trading failure, liquidity vacuum"
            },
            "rate_shock": {
                "name": "Rate Shock (200 bps hike)",
                "duration_days": 5,
                "conditions": MarketConditions(
                    volatility_multiplier=2.5,
                    liquidity_multiplier=0.6,
                    spread_multiplier=2.5,
                    correlation_spike=0.70,
                    circuit_breaker_triggered=False,
                    trading_halts=[]
                ),
                "equity_drop_pct": -15.0,
                "rate_change_bps": 200,  # 2% rate hike
                "bond_drop_pct": -8.0,
                "description": "Unexpected Fed rate hike, yield curve inversion"
            },
            "geopolitical_crisis": {
                "name": "Geopolitical Crisis",
                "duration_days": 10,
                "conditions": MarketConditions(
                    volatility_multiplier=3.0,
                    liquidity_multiplier=0.5,
                    spread_multiplier=3.5,
                    correlation_spike=0.85,
                    circuit_breaker_triggered=False,
                    trading_halts=["RSX"]  # Russia ETF
                ),
                "equity_drop_pct": -12.0,
                "oil_spike_pct": 40.0,
                "safe_haven_rally_pct": 8.0,  # Gold/Treasuries
                "description": "International conflict, market uncertainty"
            }
        }

    async def run_scenario(
        self,
        scenario_name: str,
        portfolio: Dict[str, Any],
        strategies: List[str],
        duration_minutes: Optional[int] = None
    ) -> ScenarioResult:
        """
        Run a scenario simulation

        Args:
            scenario_name: Name of scenario to run
            portfolio: Current portfolio state
            strategies: List of strategy names to simulate
            duration_minutes: Override scenario duration

        Returns:
            ScenarioResult with detailed metrics
        """
        logger.info(f"Running scenario: {scenario_name}")

        # Get scenario definition
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]
        conditions = scenario["conditions"]

        # Simulate market conditions
        market_data = self._generate_market_data(scenario, duration_minutes)

        # Simulate portfolio performance
        pnl_impact, max_drawdown = self._simulate_portfolio(
            portfolio, market_data, conditions
        )

        # Simulate strategy performance
        strategy_results = self._simulate_strategies(
            strategies, market_data, conditions
        )

        # Simulate execution failures
        execution_failures = self._simulate_execution_failures(conditions)

        # Simulate system alerts
        system_alerts = self._check_alert_triggers(
            pnl_impact, max_drawdown, conditions
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            pnl_impact, max_drawdown, execution_failures, conditions
        )

        # Calculate risk limit breaches
        risk_breaches = self._count_risk_breaches(max_drawdown, conditions)

        result = ScenarioResult(
            scenario_id=f"sim_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            scenario_name=scenario["name"],
            duration_minutes=duration_minutes or scenario.get("duration_days", 1) * 24 * 60,
            pnl_impact=pnl_impact,
            max_drawdown=max_drawdown,
            volatility_realized=conditions.volatility_multiplier,
            sharpe_ratio=self._calculate_sharpe(pnl_impact, conditions),
            risk_limit_breaches=risk_breaches,
            execution_failures=execution_failures,
            system_alerts=system_alerts,
            recommendations=recommendations,
            detailed_metrics={
                "market_conditions": {
                    "volatility_multiplier": conditions.volatility_multiplier,
                    "liquidity_multiplier": conditions.liquidity_multiplier,
                    "spread_multiplier": conditions.spread_multiplier,
                    "correlation_spike": conditions.correlation_spike
                },
                "strategy_results": strategy_results,
                "equity_drop_pct": scenario.get("equity_drop_pct", 0),
                "circuit_breaker": conditions.circuit_breaker_triggered
            }
        )

        logger.info(f"Scenario complete: PnL impact ${pnl_impact:,.2f}, MaxDD {max_drawdown:.2f}%")

        return result

    async def generate_synthetic_shock(
        self,
        shock_type: ScenarioType,
        magnitude: ShockMagnitude = ShockMagnitude.MODERATE,
        duration_days: int = 5,
        affected_sectors: Optional[List[str]] = None
    ) -> ScenarioResult:
        """
        Generate a custom synthetic shock scenario

        Args:
            shock_type: Type of shock
            magnitude: Severity level
            duration_days: How long the shock lasts
            affected_sectors: Which sectors are most affected

        Returns:
            ScenarioResult with simulation outcome
        """
        logger.info(f"Generating synthetic {shock_type} shock (magnitude: {magnitude})")

        # Map magnitude to multipliers
        magnitude_map = {
            ShockMagnitude.MILD: 1.5,
            ShockMagnitude.MODERATE: 2.5,
            ShockMagnitude.SEVERE: 4.0,
            ShockMagnitude.EXTREME: 6.0
        }
        multiplier = magnitude_map[magnitude]

        # Create conditions
        conditions = MarketConditions(
            volatility_multiplier=multiplier,
            liquidity_multiplier=1.0 / multiplier,
            spread_multiplier=multiplier,
            correlation_spike=0.5 + (multiplier / 10),
            circuit_breaker_triggered=(magnitude == ShockMagnitude.EXTREME)
        )

        # Generate scenario
        scenario = {
            "name": f"Synthetic {shock_type.value} ({magnitude.value})",
            "duration_days": duration_days,
            "conditions": conditions,
            "equity_drop_pct": -5 * multiplier,
            "affected_sectors": affected_sectors or []
        }

        # Run simulation
        return await self.run_scenario(
            f"synthetic_{shock_type.value}",
            portfolio={},  # Placeholder
            strategies=[],
            duration_minutes=duration_days * 24 * 60
        )

    def _generate_market_data(
        self,
        scenario: Dict,
        duration_minutes: Optional[int]
    ) -> pd.DataFrame:
        """Generate synthetic market data for scenario"""
        duration = duration_minutes or scenario.get("duration_days", 1) * 24 * 60

        # Create time series
        timestamps = pd.date_range(
            start=datetime.now(),
            periods=duration,
            freq='1min'
        )

        # Generate price path with volatility
        volatility = scenario["conditions"].volatility_multiplier
        equity_drop = scenario.get("equity_drop_pct", -10) / 100

        returns = np.random.normal(
            loc=equity_drop / duration,  # Drift towards drop
            scale=volatility * 0.01,  # Scaled volatility
            size=duration
        )

        prices = 100 * (1 + returns).cumprod()

        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volatility': volatility,
            'liquidity': scenario["conditions"].liquidity_multiplier
        })

    def _simulate_portfolio(
        self,
        portfolio: Dict,
        market_data: pd.DataFrame,
        conditions: MarketConditions
    ) -> tuple:
        """Simulate portfolio performance"""
        # Simplified simulation
        equity_drop = market_data['price'].iloc[-1] / market_data['price'].iloc[0] - 1

        # Apply leverage and correlation effects
        pnl_impact = equity_drop * 100000  # Assume $100k portfolio
        max_drawdown = equity_drop * 100  # Percentage

        # Worse drawdown with higher correlation
        max_drawdown *= (1 + conditions.correlation_spike)

        return pnl_impact, max_drawdown

    def _simulate_strategies(
        self,
        strategies: List[str],
        market_data: pd.DataFrame,
        conditions: MarketConditions
    ) -> Dict[str, Dict]:
        """Simulate strategy performance"""
        results = {}

        for strategy in strategies:
            # Different strategies behave differently in stress
            if "momentum" in strategy.lower():
                # Momentum suffers in reversals
                pnl = -abs(market_data['price'].iloc[-1] - market_data['price'].iloc[0]) * 1000
            elif "mean_reversion" in strategy.lower():
                # Mean reversion can profit from oversold bounces
                pnl = abs(market_data['price'].iloc[-1] - market_data['price'].iloc[0]) * 500
            else:
                pnl = 0

            results[strategy] = {
                "pnl": pnl,
                "trades": int(len(market_data) / 60),  # Assume hourly trades
                "success_rate": 0.4 if conditions.volatility_multiplier > 3 else 0.6
            }

        return results

    def _simulate_execution_failures(self, conditions: MarketConditions) -> int:
        """Simulate execution failures during stress"""
        # More failures with lower liquidity
        failure_rate = (1.0 - conditions.liquidity_multiplier) * 10
        return int(failure_rate)

    def _check_alert_triggers(
        self,
        pnl_impact: float,
        max_drawdown: float,
        conditions: MarketConditions
    ) -> int:
        """Count how many alerts would trigger"""
        alerts = 0

        if abs(max_drawdown) > 10:
            alerts += 1  # Drawdown alert
        if abs(pnl_impact) > 50000:
            alerts += 1  # Large loss alert
        if conditions.circuit_breaker_triggered:
            alerts += 1  # Circuit breaker alert
        if conditions.volatility_multiplier > 3:
            alerts += 1  # High volatility alert
        if conditions.liquidity_multiplier < 0.5:
            alerts += 1  # Low liquidity alert

        return alerts

    def _count_risk_breaches(self, max_drawdown: float, conditions: MarketConditions) -> int:
        """Count risk limit breaches"""
        breaches = 0

        # Check various risk limits
        if abs(max_drawdown) > 15:  # 15% max drawdown limit
            breaches += 1
        if abs(max_drawdown) > 20:  # 20% circuit breaker
            breaches += 1
        if conditions.circuit_breaker_triggered:
            breaches += 1

        return breaches

    def _generate_recommendations(
        self,
        pnl_impact: float,
        max_drawdown: float,
        execution_failures: int,
        conditions: MarketConditions
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if abs(max_drawdown) > 15:
            recommendations.append("Increase cash buffer to 20% for liquidity reserves")
            recommendations.append("Tighten stop losses during high volatility periods")

        if execution_failures > 5:
            recommendations.append("Review execution algorithms for stress scenarios")
            recommendations.append("Add alternative liquidity providers")

        if conditions.liquidity_multiplier < 0.5:
            recommendations.append("Reduce position sizes during low liquidity")
            recommendations.append("Use limit orders instead of market orders")

        if conditions.volatility_multiplier > 3:
            recommendations.append("Scale down leverage during high volatility")
            recommendations.append("Increase monitoring frequency")

        if conditions.correlation_spike > 0.85:
            recommendations.append("Diversification benefits reduced - review hedges")
            recommendations.append("Consider cross-asset diversification")

        return recommendations

    def _calculate_sharpe(self, pnl: float, conditions: MarketConditions) -> float:
        """Calculate Sharpe ratio for scenario"""
        # Simplified calculation
        annual_return = pnl / 100000 * 252  # Annualize
        annual_vol = conditions.volatility_multiplier * 0.15  # Base vol 15%

        return annual_return / annual_vol if annual_vol > 0 else 0.0


# Singleton instance
_scenario_engine: Optional[ScenarioEngine] = None


def get_scenario_engine() -> ScenarioEngine:
    """Get or create scenario engine singleton"""
    global _scenario_engine
    if _scenario_engine is None:
        _scenario_engine = ScenarioEngine()
    return _scenario_engine
