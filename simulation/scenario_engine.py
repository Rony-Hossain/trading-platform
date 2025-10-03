"""
Scenario Simulation Engine

Runs trading strategies through extreme market scenarios to validate
risk management and identify weaknesses.

Features:
- Scenario packs (flash crash, high volatility, correlation breakdown)
- Automated breach detection
- Jira ticket creation for violations
- PDF report generation
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
scenarios_run = Counter('scenario_engine_runs_total', 'Total scenario runs', ['scenario_id', 'status'])
breach_detected = Counter('scenario_engine_breaches_total', 'Breaches detected', ['scenario_id', 'breach_type'])
simulation_duration = Histogram('scenario_engine_duration_seconds', 'Simulation duration', ['scenario_id'])


class BreachSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Scenario:
    """Market scenario definition"""
    name: str
    description: str
    scenario_id: str
    market_conditions: Dict[str, Any]
    expected_behavior: Dict[str, Any]
    severity: str
    tags: List[str]


@dataclass
class SimulationResult:
    """Simulation result"""
    scenario_id: str
    strategy_id: str
    run_id: str
    start_time: datetime
    end_time: datetime

    # Performance metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float

    # Risk metrics
    circuit_breaker_triggered: bool
    positions_flattened: bool
    risk_limits_breached: bool

    # Position metrics
    avg_position_size: float
    max_position_size: float
    position_sizing_reduced: bool

    # Recovery metrics
    recovery_time_minutes: float
    alpha_degradation: float

    # Additional
    num_trades: int
    total_pnl: float
    metadata: Dict[str, Any]


@dataclass
class Breach:
    """Breach of expected behavior"""
    breach_type: str
    expected: Any
    actual: Any
    severity: BreachSeverity
    description: str
    remediation: str


@dataclass
class BreachReport:
    """Breach report"""
    simulation_run_id: str
    scenario_name: str
    strategy_id: str
    timestamp: datetime
    breaches: List[Breach]
    ticket_id: Optional[str] = None
    report_path: Optional[str] = None


class ScenarioEngine:
    """Scenario simulation engine"""

    def __init__(self, scenario_dir: str = "simulation/packs"):
        self.scenario_dir = Path(scenario_dir)
        self.scenarios: Dict[str, Scenario] = {}
        self.load_scenarios()

    def load_scenarios(self):
        """Load scenario definitions from JSON files"""
        if not self.scenario_dir.exists():
            logger.warning(f"Scenario directory {self.scenario_dir} does not exist")
            return

        for scenario_file in self.scenario_dir.glob("*.json"):
            try:
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)

                scenario = Scenario(**scenario_data)
                self.scenarios[scenario.scenario_id] = scenario

                logger.info(f"Loaded scenario: {scenario.name} ({scenario.scenario_id})")

            except Exception as e:
                logger.error(f"Error loading scenario {scenario_file}: {e}")

    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get scenario by ID"""
        return self.scenarios.get(scenario_id)

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """List all available scenarios"""
        return [
            {
                "scenario_id": s.scenario_id,
                "name": s.name,
                "description": s.description,
                "severity": s.severity,
                "tags": s.tags
            }
            for s in self.scenarios.values()
        ]

    def run_simulation(
        self,
        scenario_id: str,
        strategy_id: str,
        strategy_func: Any,
        **kwargs
    ) -> SimulationResult:
        """
        Run simulation with scenario

        Args:
            scenario_id: Scenario ID
            strategy_id: Strategy ID
            strategy_func: Strategy function to run
            **kwargs: Additional parameters

        Returns:
            SimulationResult
        """
        import time
        start_time = time.time()

        scenario = self.get_scenario(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")

        logger.info(f"Running simulation: {scenario.name} with strategy {strategy_id}")

        try:
            # Generate market data based on scenario
            market_data = self._generate_market_data(scenario)

            # Run strategy
            result = strategy_func(market_data, **kwargs)

            # Create simulation result
            simulation_result = SimulationResult(
                scenario_id=scenario_id,
                strategy_id=strategy_id,
                run_id=f"{scenario_id}_{strategy_id}_{int(time.time())}",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(seconds=time.time() - start_time),
                total_return=result.get('total_return', 0.0),
                sharpe_ratio=result.get('sharpe_ratio', 0.0),
                max_drawdown=result.get('max_drawdown', 0.0),
                volatility=result.get('volatility', 0.0),
                circuit_breaker_triggered=result.get('circuit_breaker_triggered', False),
                positions_flattened=result.get('positions_flattened', False),
                risk_limits_breached=result.get('risk_limits_breached', False),
                avg_position_size=result.get('avg_position_size', 0.0),
                max_position_size=result.get('max_position_size', 0.0),
                position_sizing_reduced=result.get('position_sizing_reduced', False),
                recovery_time_minutes=result.get('recovery_time_minutes', 0.0),
                alpha_degradation=result.get('alpha_degradation', 0.0),
                num_trades=result.get('num_trades', 0),
                total_pnl=result.get('total_pnl', 0.0),
                metadata=result.get('metadata', {})
            )

            duration = time.time() - start_time
            simulation_duration.labels(scenario_id=scenario_id).observe(duration)
            scenarios_run.labels(scenario_id=scenario_id, status='success').inc()

            logger.info(f"Simulation completed in {duration:.2f}s")

            return simulation_result

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            scenarios_run.labels(scenario_id=scenario_id, status='failure').inc()
            raise

    def _generate_market_data(self, scenario: Scenario) -> pd.DataFrame:
        """Generate synthetic market data based on scenario"""
        conditions = scenario.market_conditions

        if scenario.scenario_id == "flash_crash_2010":
            return self._generate_flash_crash_data(conditions)
        elif scenario.scenario_id == "high_vix_regime":
            return self._generate_high_vol_data(conditions)
        elif scenario.scenario_id == "correlation_breakdown":
            return self._generate_correlation_breakdown_data(conditions)
        else:
            raise ValueError(f"Unknown scenario type: {scenario.scenario_id}")

    def _generate_flash_crash_data(self, conditions: Dict) -> pd.DataFrame:
        """Generate flash crash market data"""
        initial_price = conditions['initial_price']
        drop_magnitude = conditions['drop_magnitude']
        drop_duration = conditions['drop_duration_minutes']
        recovery_duration = conditions['recovery_duration_minutes']

        total_minutes = 390  # Trading day
        timestamps = pd.date_range('2024-01-01 09:30', periods=total_minutes, freq='1min')

        prices = []
        crash_start = 120  # 2 hours into trading

        for i in range(total_minutes):
            if i < crash_start:
                # Normal pre-crash
                price = initial_price + np.random.normal(0, 0.5)
            elif i < crash_start + drop_duration:
                # Flash crash
                progress = (i - crash_start) / drop_duration
                price = initial_price * (1 - drop_magnitude * progress)
            elif i < crash_start + drop_duration + recovery_duration:
                # Recovery
                recovery_progress = (i - crash_start - drop_duration) / recovery_duration
                crash_price = initial_price * (1 - drop_magnitude)
                price = crash_price + (initial_price - crash_price) * recovery_progress
            else:
                # Post-recovery
                price = initial_price + np.random.normal(0, 0.5)

            prices.append(max(price, 0.01))

        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': [int(np.random.normal(10000, 2000)) for _ in range(total_minutes)]
        })

    def _generate_high_vol_data(self, conditions: Dict) -> pd.DataFrame:
        """Generate high volatility regime data"""
        vix_level = conditions['vix_level']
        duration_days = conditions['duration_days']

        # Convert VIX to daily volatility
        daily_vol = vix_level / 100
        minute_vol = daily_vol / np.sqrt(390)

        total_minutes = duration_days * 390
        timestamps = pd.date_range('2024-01-01 09:30', periods=total_minutes, freq='1min')

        current_price = 100.0
        prices = []

        for _ in range(total_minutes):
            ret = np.random.normal(0, minute_vol)
            current_price *= (1 + ret)
            prices.append(max(current_price, 0.01))

        return pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': [int(np.random.normal(25000, 5000)) for _ in range(total_minutes)]
        })

    def _generate_correlation_breakdown_data(self, conditions: Dict) -> pd.DataFrame:
        """Generate correlation breakdown data"""
        initial_corr = conditions['initial_correlation']
        crisis_corr = conditions['crisis_correlation']
        transition_days = conditions['transition_days']
        duration_days = conditions['duration_days']

        total_days = transition_days + duration_days
        total_minutes = total_days * 390
        timestamps = pd.date_range('2024-01-01 09:30', periods=total_minutes, freq='1min')

        # Generate correlated asset prices
        np.random.seed(42)

        # Two assets with changing correlation
        returns1 = np.random.normal(0, 0.01, total_minutes)

        # Gradually increase correlation
        correlations = np.linspace(initial_corr, crisis_corr, transition_days * 390)
        correlations = np.concatenate([correlations, np.ones(duration_days * 390) * crisis_corr])

        returns2 = []
        for i, corr in enumerate(correlations):
            # Generate correlated return
            independent = np.random.normal(0, 0.01)
            correlated_return = corr * returns1[i] + np.sqrt(1 - corr**2) * independent
            returns2.append(correlated_return)

        prices1 = 100 * np.cumprod(1 + returns1)
        prices2 = 100 * np.cumprod(1 + np.array(returns2))

        return pd.DataFrame({
            'timestamp': timestamps,
            'asset1_price': prices1,
            'asset2_price': prices2,
            'correlation': correlations
        })

    def detect_breaches(
        self,
        simulation_result: SimulationResult,
        scenario: Scenario
    ) -> List[Breach]:
        """
        Detect breaches of expected behavior

        Args:
            simulation_result: Simulation result
            scenario: Scenario definition

        Returns:
            List of breaches
        """
        breaches = []
        expected = scenario.expected_behavior

        # Check max drawdown
        if 'max_drawdown' in expected:
            expected_dd = expected['max_drawdown']
            actual_dd = abs(simulation_result.max_drawdown)

            if actual_dd > expected_dd:
                breaches.append(Breach(
                    breach_type='max_drawdown_breach',
                    expected=expected_dd,
                    actual=actual_dd,
                    severity=BreachSeverity.HIGH,
                    description=f"Max drawdown {actual_dd:.2%} exceeded limit {expected_dd:.2%}",
                    remediation="Review position sizing and stop-loss levels"
                ))
                breach_detected.labels(scenario_id=scenario.scenario_id, breach_type='max_drawdown').inc()

        # Check circuit breaker
        if 'circuit_breaker_triggers' in expected:
            if expected['circuit_breaker_triggers'] and not simulation_result.circuit_breaker_triggered:
                breaches.append(Breach(
                    breach_type='circuit_breaker_not_triggered',
                    expected=True,
                    actual=False,
                    severity=BreachSeverity.CRITICAL,
                    description="Circuit breaker should have triggered but didn't",
                    remediation="Review circuit breaker thresholds and logic"
                ))
                breach_detected.labels(scenario_id=scenario.scenario_id, breach_type='circuit_breaker').inc()

        # Check position flattening
        if 'positions_flattened' in expected:
            if expected['positions_flattened'] and not simulation_result.positions_flattened:
                breaches.append(Breach(
                    breach_type='positions_not_flattened',
                    expected=True,
                    actual=False,
                    severity=BreachSeverity.HIGH,
                    description="Positions should have been flattened during crisis",
                    remediation="Review position management during extreme events"
                ))
                breach_detected.labels(scenario_id=scenario.scenario_id, breach_type='position_flatten').inc()

        # Check position sizing reduction
        if 'position_sizing_reduced' in expected:
            if expected['position_sizing_reduced'] and not simulation_result.position_sizing_reduced:
                breaches.append(Breach(
                    breach_type='position_sizing_not_reduced',
                    expected=True,
                    actual=False,
                    severity=BreachSeverity.MEDIUM,
                    description="Position sizing should be reduced in high volatility",
                    remediation="Implement volatility-based position sizing"
                ))
                breach_detected.labels(scenario_id=scenario.scenario_id, breach_type='position_sizing').inc()

        # Check alpha degradation
        if 'alpha_degradation_acceptable' in expected:
            acceptable_degradation = expected['alpha_degradation_acceptable']
            actual_degradation = simulation_result.alpha_degradation

            if actual_degradation > acceptable_degradation:
                breaches.append(Breach(
                    breach_type='alpha_degradation_excessive',
                    expected=acceptable_degradation,
                    actual=actual_degradation,
                    severity=BreachSeverity.MEDIUM,
                    description=f"Alpha degradation {actual_degradation:.2%} exceeds acceptable {acceptable_degradation:.2%}",
                    remediation="Review strategy robustness in different regimes"
                ))
                breach_detected.labels(scenario_id=scenario.scenario_id, breach_type='alpha_degradation').inc()

        # Check recovery time
        if 'recovery_time_minutes' in expected:
            expected_recovery = expected['recovery_time_minutes']
            actual_recovery = simulation_result.recovery_time_minutes

            if actual_recovery > expected_recovery:
                breaches.append(Breach(
                    breach_type='slow_recovery',
                    expected=expected_recovery,
                    actual=actual_recovery,
                    severity=BreachSeverity.MEDIUM,
                    description=f"Recovery time {actual_recovery:.0f} min exceeds target {expected_recovery:.0f} min",
                    remediation="Improve recovery procedures and position rebalancing"
                ))
                breach_detected.labels(scenario_id=scenario.scenario_id, breach_type='recovery_time').inc()

        logger.info(f"Detected {len(breaches)} breaches for scenario {scenario.scenario_id}")

        return breaches


def generate_breach_report(
    simulation_result: SimulationResult,
    scenario: Scenario,
    breaches: List[Breach]
) -> BreachReport:
    """
    Generate breach report and create remediation ticket

    Args:
        simulation_result: Simulation result
        scenario: Scenario
        breaches: List of breaches

    Returns:
        BreachReport with ticket ID and report path
    """
    if not breaches:
        logger.info("No breaches detected")
        return BreachReport(
            simulation_run_id=simulation_result.run_id,
            scenario_name=scenario.name,
            strategy_id=simulation_result.strategy_id,
            timestamp=datetime.utcnow(),
            breaches=[]
        )

    # Create Jira ticket (mock implementation)
    ticket_id = create_jira_ticket(scenario, breaches)

    # Generate PDF report (mock implementation)
    report_path = generate_pdf_report(simulation_result, scenario, breaches)

    report = BreachReport(
        simulation_run_id=simulation_result.run_id,
        scenario_name=scenario.name,
        strategy_id=simulation_result.strategy_id,
        timestamp=datetime.utcnow(),
        breaches=breaches,
        ticket_id=ticket_id,
        report_path=report_path
    )

    logger.warning(f"Breach report generated: {len(breaches)} breaches, ticket: {ticket_id}")

    return report


def create_jira_ticket(scenario: Scenario, breaches: List[Breach]) -> str:
    """
    Create Jira ticket for breaches (mock implementation)

    In production, this would use Jira API
    """
    # Mock ticket creation
    ticket_id = f"RISK-{int(datetime.utcnow().timestamp())}"

    description = f"""
Simulation Breach: {scenario.name}

Scenario: {scenario.description}
Severity: {scenario.severity}

Breaches Detected:
"""

    for i, breach in enumerate(breaches, 1):
        description += f"""
{i}. {breach.breach_type}
   - Expected: {breach.expected}
   - Actual: {breach.actual}
   - Severity: {breach.severity.value}
   - Description: {breach.description}
   - Remediation: {breach.remediation}
"""

    logger.info(f"Mock Jira ticket created: {ticket_id}")
    logger.debug(f"Ticket description:\n{description}")

    return ticket_id


def generate_pdf_report(
    simulation_result: SimulationResult,
    scenario: Scenario,
    breaches: List[Breach]
) -> str:
    """
    Generate PDF breach report (mock implementation)

    In production, this would use ReportLab or similar
    """
    report_filename = f"breach_report_{simulation_result.run_id}.pdf"
    report_path = f"simulation/reports/{report_filename}"

    logger.info(f"Mock PDF report generated: {report_path}")

    return report_path
