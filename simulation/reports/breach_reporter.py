"""
Breach Reporter

Generates detailed reports for scenario breaches and manages remediation workflow.
"""
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import json

from simulation.scenario_engine import Breach, BreachReport, SimulationResult, Scenario

logger = logging.getLogger(__name__)


class BreachReporter:
    """Enhanced breach reporting with multiple output formats"""

    def __init__(self, output_dir: str = "simulation/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        simulation_result: SimulationResult,
        scenario: Scenario,
        breaches: List[Breach]
    ) -> str:
        """Generate Markdown breach report"""

        report = f"""# Simulation Breach Report

## Scenario: {scenario.name}

**Scenario ID**: `{scenario.scenario_id}`
**Description**: {scenario.description}
**Severity**: {scenario.severity}
**Run ID**: `{simulation_result.run_id}`
**Strategy**: {simulation_result.strategy_id}
**Timestamp**: {datetime.utcnow().isoformat()}

---

## Simulation Results

| Metric | Value |
|--------|-------|
| Total Return | {simulation_result.total_return:.2%} |
| Sharpe Ratio | {simulation_result.sharpe_ratio:.2f} |
| Max Drawdown | {simulation_result.max_drawdown:.2%} |
| Volatility | {simulation_result.volatility:.2%} |
| Num Trades | {simulation_result.num_trades} |
| Total P&L | ${simulation_result.total_pnl:,.2f} |

---

## Breach Summary

**Total Breaches**: {len(breaches)}

"""

        if not breaches:
            report += "\n✅ **No breaches detected** - All acceptance criteria met!\n"
        else:
            # Group breaches by severity
            critical = [b for b in breaches if b.severity.value == "CRITICAL"]
            high = [b for b in breaches if b.severity.value == "HIGH"]
            medium = [b for b in breaches if b.severity.value == "MEDIUM"]
            low = [b for b in breaches if b.severity.value == "LOW"]

            report += f"""
### Breakdown by Severity

- 🔴 **CRITICAL**: {len(critical)}
- 🟠 **HIGH**: {len(high)}
- 🟡 **MEDIUM**: {len(medium)}
- 🟢 **LOW**: {len(low)}

---

## Breach Details

"""

            for i, breach in enumerate(breaches, 1):
                severity_icon = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠",
                    "MEDIUM": "🟡",
                    "LOW": "🟢"
                }.get(breach.severity.value, "⚪")

                report += f"""
### {severity_icon} Breach {i}: {breach.breach_type}

**Severity**: {breach.severity.value}

**Description**: {breach.description}

**Expected**: `{breach.expected}`
**Actual**: `{breach.actual}`

**Recommended Remediation**:
{breach.remediation}

---

"""

        report += f"""
## Expected Behavior

```json
{json.dumps(scenario.expected_behavior, indent=2)}
```

## Market Conditions

```json
{json.dumps(scenario.market_conditions, indent=2)}
```

---

*Report generated at {datetime.utcnow().isoformat()}*
"""

        # Save report
        filename = f"{simulation_result.run_id}_breach_report.md"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(report)

        logger.info(f"Markdown report saved: {filepath}")

        return str(filepath)

    def generate_json_report(
        self,
        simulation_result: SimulationResult,
        scenario: Scenario,
        breaches: List[Breach]
    ) -> str:
        """Generate JSON breach report for programmatic access"""

        report_data = {
            "scenario": {
                "scenario_id": scenario.scenario_id,
                "name": scenario.name,
                "description": scenario.description,
                "severity": scenario.severity,
                "tags": scenario.tags
            },
            "simulation": {
                "run_id": simulation_result.run_id,
                "strategy_id": simulation_result.strategy_id,
                "start_time": simulation_result.start_time.isoformat(),
                "end_time": simulation_result.end_time.isoformat(),
                "results": {
                    "total_return": simulation_result.total_return,
                    "sharpe_ratio": simulation_result.sharpe_ratio,
                    "max_drawdown": simulation_result.max_drawdown,
                    "volatility": simulation_result.volatility,
                    "circuit_breaker_triggered": simulation_result.circuit_breaker_triggered,
                    "positions_flattened": simulation_result.positions_flattened,
                    "risk_limits_breached": simulation_result.risk_limits_breached,
                    "avg_position_size": simulation_result.avg_position_size,
                    "recovery_time_minutes": simulation_result.recovery_time_minutes,
                    "alpha_degradation": simulation_result.alpha_degradation,
                    "num_trades": simulation_result.num_trades,
                    "total_pnl": simulation_result.total_pnl
                }
            },
            "breaches": [
                {
                    "breach_type": b.breach_type,
                    "expected": str(b.expected),
                    "actual": str(b.actual),
                    "severity": b.severity.value,
                    "description": b.description,
                    "remediation": b.remediation
                }
                for b in breaches
            ],
            "summary": {
                "total_breaches": len(breaches),
                "breaches_by_severity": {
                    "CRITICAL": len([b for b in breaches if b.severity.value == "CRITICAL"]),
                    "HIGH": len([b for b in breaches if b.severity.value == "HIGH"]),
                    "MEDIUM": len([b for b in breaches if b.severity.value == "MEDIUM"]),
                    "LOW": len([b for b in breaches if b.severity.value == "LOW"])
                },
                "passed": len(breaches) == 0
            },
            "metadata": {
                "generated_at": datetime.utcnow().isoformat()
            }
        }

        # Save JSON report
        filename = f"{simulation_result.run_id}_breach_report.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"JSON report saved: {filepath}")

        return str(filepath)

    def generate_html_report(
        self,
        simulation_result: SimulationResult,
        scenario: Scenario,
        breaches: List[Breach]
    ) -> str:
        """Generate HTML breach report for web viewing"""

        # Simple HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Breach Report - {scenario.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .breach {{ border-left: 4px solid #e74c3c; padding: 15px; margin: 15px 0; background: #fff5f5; }}
        .breach.CRITICAL {{ border-color: #c0392b; background: #ffebee; }}
        .breach.HIGH {{ border-color: #e67e22; background: #fff3e0; }}
        .breach.MEDIUM {{ border-color: #f39c12; background: #fff8e1; }}
        .breach.LOW {{ border-color: #27ae60; background: #e8f5e9; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 3px; color: white; font-size: 0.9em; }}
        .badge.CRITICAL {{ background: #c0392b; }}
        .badge.HIGH {{ background: #e67e22; }}
        .badge.MEDIUM {{ background: #f39c12; }}
        .badge.LOW {{ background: #27ae60; }}
        .passed {{ color: #27ae60; font-size: 1.5em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Simulation Breach Report</h1>
        <h2>Scenario: {scenario.name}</h2>
        <p><strong>Description:</strong> {scenario.description}</p>
        <p><strong>Run ID:</strong> {simulation_result.run_id}</p>
        <p><strong>Strategy:</strong> {simulation_result.strategy_id}</p>
        <p><strong>Timestamp:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>

        <h2>📊 Simulation Results</h2>
        <div class="metric">
            <div class="metric-label">Total Return</div>
            <div class="metric-value">{simulation_result.total_return:.2%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{simulation_result.sharpe_ratio:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value">{simulation_result.max_drawdown:.2%}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Volatility</div>
            <div class="metric-value">{simulation_result.volatility:.2%}</div>
        </div>

        <h2>⚠️ Breach Summary</h2>
        <p><strong>Total Breaches:</strong> {len(breaches)}</p>
"""

        if not breaches:
            html += '<p class="passed">✅ No breaches detected - All acceptance criteria met!</p>'
        else:
            html += "<h3>Breach Details</h3>"

            for i, breach in enumerate(breaches, 1):
                html += f"""
        <div class="breach {breach.severity.value}">
            <h4>Breach {i}: {breach.breach_type} <span class="badge {breach.severity.value}">{breach.severity.value}</span></h4>
            <p><strong>Description:</strong> {breach.description}</p>
            <p><strong>Expected:</strong> {breach.expected} | <strong>Actual:</strong> {breach.actual}</p>
            <p><strong>Remediation:</strong> {breach.remediation}</p>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""

        # Save HTML report
        filename = f"{simulation_result.run_id}_breach_report.html"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(html)

        logger.info(f"HTML report saved: {filepath}")

        return str(filepath)

    def generate_all_reports(
        self,
        simulation_result: SimulationResult,
        scenario: Scenario,
        breaches: List[Breach]
    ) -> Dict[str, str]:
        """Generate all report formats"""

        return {
            "markdown": self.generate_markdown_report(simulation_result, scenario, breaches),
            "json": self.generate_json_report(simulation_result, scenario, breaches),
            "html": self.generate_html_report(simulation_result, scenario, breaches)
        }
