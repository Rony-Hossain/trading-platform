"""
Alpha Decay vs Latency Analysis Job

Emits per-strategy alpha loss metrics vs latency buckets.
Sets up metric: alpha_decay_slope{strategy=...}
Alerts on threshold breaches requiring page & memo.

Acceptance: Alerts fire correctly; weekly review exported
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class LatencyBucket(Enum):
    """Latency bucket definitions"""
    FAST = "0-100ms"
    MEDIUM = "100-500ms"
    SLOW = "500-2000ms"
    VERY_SLOW = "2000ms+"


@dataclass
class AlphaDecayMetric:
    """Alpha decay metric for a specific strategy and latency bucket"""
    strategy_id: str
    strategy_name: str
    latency_bucket: LatencyBucket
    alpha_decay_slope: float  # Alpha loss per ms of latency
    total_alpha_loss: float  # Total alpha lost in this bucket
    trade_count: int
    avg_latency_ms: float
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary for export"""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "latency_bucket": self.latency_bucket.value,
            "alpha_decay_slope": self.alpha_decay_slope,
            "total_alpha_loss": self.total_alpha_loss,
            "trade_count": self.trade_count,
            "avg_latency_ms": self.avg_latency_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AlphaDecayAlert:
    """Alert for alpha decay threshold breach"""
    strategy_id: str
    strategy_name: str
    current_slope: float
    threshold: float
    breach_percentage: float
    action_required: str
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary for alerting"""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "current_slope": self.current_slope,
            "threshold": self.threshold,
            "breach_percentage": self.breach_percentage,
            "action_required": self.action_required,
            "severity": "CRITICAL" if self.breach_percentage > 50 else "WARNING",
            "timestamp": self.timestamp.isoformat()
        }


class AlphaDecayAnalyzer:
    """
    Analyzes alpha decay vs latency across strategies.

    Calculates how much alpha is lost as latency increases,
    emitting per-strategy metrics and alerts.
    """

    def __init__(self,
                 alpha_decay_threshold: float = 0.05,  # 5 bps alpha loss per 100ms
                 lookback_days: int = 7):
        """
        Initialize alpha decay analyzer.

        Args:
            alpha_decay_threshold: Alert threshold for alpha decay slope
            lookback_days: Number of days to analyze
        """
        self.alpha_decay_threshold = alpha_decay_threshold
        self.lookback_days = lookback_days
        self.metrics_history: List[AlphaDecayMetric] = []
        self.alerts_history: List[AlphaDecayAlert] = []

    def categorize_latency(self, latency_ms: float) -> LatencyBucket:
        """Categorize latency into buckets"""
        if latency_ms < 100:
            return LatencyBucket.FAST
        elif latency_ms < 500:
            return LatencyBucket.MEDIUM
        elif latency_ms < 2000:
            return LatencyBucket.SLOW
        else:
            return LatencyBucket.VERY_SLOW

    def calculate_alpha_decay(self,
                             trades_df: pd.DataFrame,
                             strategy_id: str,
                             strategy_name: str) -> List[AlphaDecayMetric]:
        """
        Calculate alpha decay metrics for a strategy.

        Args:
            trades_df: DataFrame with columns: timestamp, latency_ms, realized_alpha, expected_alpha
            strategy_id: Strategy identifier
            strategy_name: Strategy name

        Returns:
            List of AlphaDecayMetric per latency bucket
        """
        if trades_df.empty:
            logger.warning(f"No trades for strategy {strategy_name}")
            return []

        # Calculate alpha loss for each trade
        trades_df['alpha_loss'] = trades_df['expected_alpha'] - trades_df['realized_alpha']

        # Categorize by latency bucket
        trades_df['latency_bucket'] = trades_df['latency_ms'].apply(self.categorize_latency)

        metrics = []

        # Calculate metrics per bucket
        for bucket in LatencyBucket:
            bucket_trades = trades_df[trades_df['latency_bucket'] == bucket]

            if len(bucket_trades) < 2:
                continue

            # Calculate alpha decay slope (alpha loss per ms)
            latencies = bucket_trades['latency_ms'].values
            alpha_losses = bucket_trades['alpha_loss'].values

            # Linear regression: alpha_loss = slope * latency + intercept
            if len(latencies) > 1:
                slope, intercept = np.polyfit(latencies, alpha_losses, 1)
            else:
                slope = 0.0

            metric = AlphaDecayMetric(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                latency_bucket=bucket,
                alpha_decay_slope=slope,
                total_alpha_loss=alpha_losses.sum(),
                trade_count=len(bucket_trades),
                avg_latency_ms=latencies.mean(),
                timestamp=datetime.now()
            )

            metrics.append(metric)
            self.metrics_history.append(metric)

        return metrics

    def check_thresholds(self,
                        metrics: List[AlphaDecayMetric]) -> List[AlphaDecayAlert]:
        """
        Check if alpha decay slopes breach thresholds.

        Args:
            metrics: List of alpha decay metrics to check

        Returns:
            List of alerts for threshold breaches
        """
        alerts = []

        for metric in metrics:
            # Normalize slope to per-100ms basis for comparison
            slope_per_100ms = metric.alpha_decay_slope * 100

            if abs(slope_per_100ms) > self.alpha_decay_threshold:
                breach_pct = ((abs(slope_per_100ms) - self.alpha_decay_threshold) /
                             self.alpha_decay_threshold * 100)

                alert = AlphaDecayAlert(
                    strategy_id=metric.strategy_id,
                    strategy_name=metric.strategy_name,
                    current_slope=slope_per_100ms,
                    threshold=self.alpha_decay_threshold,
                    breach_percentage=breach_pct,
                    action_required=(
                        f"Strategy {metric.strategy_name} is losing "
                        f"{abs(slope_per_100ms):.4f} bps alpha per 100ms in "
                        f"{metric.latency_bucket.value} bucket. "
                        f"Investigate latency optimization opportunities."
                    ),
                    timestamp=datetime.now()
                )

                alerts.append(alert)
                self.alerts_history.append(alert)

                logger.critical(
                    f"ALPHA DECAY ALERT: {alert.strategy_name} - "
                    f"Slope: {alert.current_slope:.4f} bps/100ms "
                    f"(Threshold: {alert.threshold:.4f})"
                )

        return alerts

    def emit_metrics(self, metrics: List[AlphaDecayMetric]) -> Dict:
        """
        Emit metrics in Prometheus format.

        Args:
            metrics: List of alpha decay metrics

        Returns:
            Dictionary with Prometheus-formatted metrics
        """
        prometheus_metrics = []

        for metric in metrics:
            # Main metric: alpha_decay_slope{strategy="...", bucket="..."}
            prometheus_metrics.append({
                "metric": "alpha_decay_slope",
                "labels": {
                    "strategy": metric.strategy_name,
                    "strategy_id": metric.strategy_id,
                    "latency_bucket": metric.latency_bucket.value
                },
                "value": metric.alpha_decay_slope,
                "timestamp": metric.timestamp.timestamp()
            })

            # Supporting metric: total_alpha_loss
            prometheus_metrics.append({
                "metric": "total_alpha_loss",
                "labels": {
                    "strategy": metric.strategy_name,
                    "strategy_id": metric.strategy_id,
                    "latency_bucket": metric.latency_bucket.value
                },
                "value": metric.total_alpha_loss,
                "timestamp": metric.timestamp.timestamp()
            })

            # Supporting metric: trade_count
            prometheus_metrics.append({
                "metric": "alpha_decay_trade_count",
                "labels": {
                    "strategy": metric.strategy_name,
                    "strategy_id": metric.strategy_id,
                    "latency_bucket": metric.latency_bucket.value
                },
                "value": metric.trade_count,
                "timestamp": metric.timestamp.timestamp()
            })

        return {
            "metrics": prometheus_metrics,
            "timestamp": datetime.now().isoformat()
        }

    def generate_weekly_report(self,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict:
        """
        Generate weekly review report.

        Args:
            start_date: Start of report period (defaults to 7 days ago)
            end_date: End of report period (defaults to now)

        Returns:
            Weekly report dictionary
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        # Filter metrics and alerts for period
        period_metrics = [
            m for m in self.metrics_history
            if start_date <= m.timestamp <= end_date
        ]

        period_alerts = [
            a for a in self.alerts_history
            if start_date <= a.timestamp <= end_date
        ]

        # Calculate summary statistics
        total_alpha_loss = sum(m.total_alpha_loss for m in period_metrics)
        avg_decay_slope = np.mean([m.alpha_decay_slope for m in period_metrics]) if period_metrics else 0

        # Group by strategy
        strategy_summary = {}
        for metric in period_metrics:
            if metric.strategy_id not in strategy_summary:
                strategy_summary[metric.strategy_id] = {
                    "strategy_name": metric.strategy_name,
                    "total_alpha_loss": 0,
                    "buckets": {},
                    "alert_count": 0
                }

            strategy_summary[metric.strategy_id]["total_alpha_loss"] += metric.total_alpha_loss
            strategy_summary[metric.strategy_id]["buckets"][metric.latency_bucket.value] = {
                "alpha_decay_slope": metric.alpha_decay_slope,
                "total_alpha_loss": metric.total_alpha_loss,
                "trade_count": metric.trade_count,
                "avg_latency_ms": metric.avg_latency_ms
            }

        # Add alert counts
        for alert in period_alerts:
            if alert.strategy_id in strategy_summary:
                strategy_summary[alert.strategy_id]["alert_count"] += 1

        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_metrics_collected": len(period_metrics),
                "total_alerts_fired": len(period_alerts),
                "total_alpha_loss": total_alpha_loss,
                "average_decay_slope": avg_decay_slope,
                "strategies_analyzed": len(strategy_summary)
            },
            "strategy_breakdown": strategy_summary,
            "alerts": [a.to_dict() for a in period_alerts],
            "top_alpha_losers": sorted(
                [
                    {
                        "strategy_id": sid,
                        "strategy_name": data["strategy_name"],
                        "total_alpha_loss": data["total_alpha_loss"]
                    }
                    for sid, data in strategy_summary.items()
                ],
                key=lambda x: x["total_alpha_loss"],
                reverse=True
            )[:10],
            "generated_at": datetime.now().isoformat()
        }

        return report

    def export_weekly_report(self, output_path: str) -> str:
        """
        Export weekly report to file.

        Args:
            output_path: Path to export report

        Returns:
            Path to exported report
        """
        report = self.generate_weekly_report()

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Weekly alpha decay report exported to {output_path}")

        return output_path


def run_alpha_decay_analysis(strategies: List[Dict],
                             trades_data: Dict[str, pd.DataFrame],
                             alpha_decay_threshold: float = 0.05) -> Dict:
    """
    Run alpha decay analysis job for all strategies.

    Args:
        strategies: List of strategy dictionaries with id and name
        trades_data: Dict mapping strategy_id to trades DataFrame
        alpha_decay_threshold: Alert threshold

    Returns:
        Analysis results with metrics and alerts
    """
    analyzer = AlphaDecayAnalyzer(alpha_decay_threshold=alpha_decay_threshold)

    all_metrics = []
    all_alerts = []

    for strategy in strategies:
        strategy_id = strategy['id']
        strategy_name = strategy['name']

        if strategy_id not in trades_data:
            logger.warning(f"No trades data for strategy {strategy_name}")
            continue

        # Calculate alpha decay metrics
        metrics = analyzer.calculate_alpha_decay(
            trades_data[strategy_id],
            strategy_id,
            strategy_name
        )

        # Check thresholds and generate alerts
        alerts = analyzer.check_thresholds(metrics)

        all_metrics.extend(metrics)
        all_alerts.extend(alerts)

    # Emit metrics
    prometheus_metrics = analyzer.emit_metrics(all_metrics)

    return {
        "metrics": [m.to_dict() for m in all_metrics],
        "alerts": [a.to_dict() for a in all_alerts],
        "prometheus_metrics": prometheus_metrics,
        "summary": {
            "total_strategies": len(strategies),
            "strategies_with_data": len([s for s in strategies if s['id'] in trades_data]),
            "total_metrics": len(all_metrics),
            "total_alerts": len(all_alerts)
        },
        "analyzer": analyzer  # Return for weekly report generation
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Mock data for testing
    strategies = [
        {"id": "strat_001", "name": "MomentumAlpha"},
        {"id": "strat_002", "name": "MeanReversion"}
    ]

    # Generate mock trades data
    np.random.seed(42)
    trades_data = {}

    for strategy in strategies:
        n_trades = 100
        trades_data[strategy['id']] = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=n_trades, freq='1H'),
            'latency_ms': np.random.lognormal(5, 1, n_trades),
            'expected_alpha': np.random.normal(0.001, 0.0005, n_trades),
            'realized_alpha': np.random.normal(0.0008, 0.0006, n_trades)
        })

    # Run analysis
    results = run_alpha_decay_analysis(strategies, trades_data)

    print(f"\nAlpha Decay Analysis Results:")
    print(f"Metrics collected: {results['summary']['total_metrics']}")
    print(f"Alerts fired: {results['summary']['total_alerts']}")

    if results['alerts']:
        print("\nAlerts:")
        for alert in results['alerts']:
            print(f"  - {alert['strategy_name']}: {alert['action_required']}")

    # Generate weekly report
    analyzer = results['analyzer']
    report = analyzer.generate_weekly_report()
    print(f"\nWeekly Report Summary:")
    print(f"Total alpha loss: {report['summary']['total_alpha_loss']:.6f}")
    print(f"Strategies analyzed: {report['summary']['strategies_analyzed']}")
