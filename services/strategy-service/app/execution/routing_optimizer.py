"""
Routing Optimizer
Analyzes historical routing decisions to optimize venue selection
Performs hindsight analysis to validate routing accuracy (target: ≥95%)
"""
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionOutcome:
    """Actual execution outcome"""
    venue: str
    fill_price: float
    fill_time: datetime
    slippage_bps: float
    total_cost_bps: float  # Including fees
    filled_quantity: int


@dataclass
class HindsightAnalysis:
    """Hindsight analysis results"""
    routing_decision_venue: str
    actual_best_venue: str
    was_optimal: bool
    cost_delta_bps: float
    potential_savings_bps: float
    decision_quality_score: float


class RoutingOptimizer:
    """
    Analyzes routing decisions and optimizes venue selection
    """

    def __init__(self):
        self.execution_history = []
        self.routing_history = []

    def record_execution(
        self,
        order_id: str,
        routing_decision_venue: str,
        outcomes: Dict[str, ExecutionOutcome]
    ):
        """
        Record execution outcomes across venues for hindsight analysis

        Args:
            order_id: Order identifier
            routing_decision_venue: Venue selected by router
            outcomes: Actual/simulated outcomes for all venues
        """
        self.execution_history.append({
            "order_id": order_id,
            "timestamp": datetime.utcnow(),
            "routing_decision": routing_decision_venue,
            "outcomes": outcomes
        })

    def analyze_hindsight(
        self,
        lookback_days: int = 30
    ) -> Dict[str, any]:
        """
        Perform hindsight analysis on routing decisions

        Args:
            lookback_days: Number of days to analyze

        Returns:
            Analysis results including routing accuracy
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Filter recent executions
        recent_executions = [
            ex for ex in self.execution_history
            if ex['timestamp'] >= cutoff_date
        ]

        if not recent_executions:
            return {
                "routing_accuracy": 0.0,
                "avg_cost_delta_bps": 0.0,
                "total_potential_savings_bps": 0.0,
                "analysis_count": 0
            }

        # Analyze each execution
        analyses = []
        for execution in recent_executions:
            analysis = self._analyze_single_execution(execution)
            analyses.append(analysis)

        # Calculate aggregate metrics
        routing_accuracy = sum(1 for a in analyses if a.was_optimal) / len(analyses)

        avg_cost_delta = np.mean([a.cost_delta_bps for a in analyses])

        total_potential_savings = sum(a.potential_savings_bps for a in analyses)

        # Venue-specific accuracy
        venue_accuracy = self._calculate_venue_accuracy(analyses)

        # Decision quality distribution
        quality_scores = [a.decision_quality_score for a in analyses]

        results = {
            "routing_accuracy": routing_accuracy,
            "avg_cost_delta_bps": avg_cost_delta,
            "total_potential_savings_bps": total_potential_savings,
            "analysis_count": len(analyses),
            "venue_accuracy": venue_accuracy,
            "quality_score_distribution": {
                "p25": np.percentile(quality_scores, 25),
                "p50": np.percentile(quality_scores, 50),
                "p75": np.percentile(quality_scores, 75),
                "p95": np.percentile(quality_scores, 95)
            }
        }

        logger.info(
            f"Hindsight analysis ({lookback_days} days): "
            f"Routing accuracy = {routing_accuracy:.1%}, "
            f"Avg cost delta = {avg_cost_delta:.2f} bps"
        )

        # Assert routing accuracy meets target
        if routing_accuracy < 0.95:
            logger.warning(
                f"Routing accuracy {routing_accuracy:.1%} below target 95%"
            )

        return results

    def _analyze_single_execution(
        self,
        execution: Dict
    ) -> HindsightAnalysis:
        """Analyze a single execution for optimal venue selection"""
        routing_decision_venue = execution['routing_decision']
        outcomes = execution['outcomes']

        # Find actual best venue (lowest total cost)
        best_venue = min(
            outcomes.items(),
            key=lambda x: x[1].total_cost_bps
        )[0]

        best_cost = outcomes[best_venue].total_cost_bps
        decision_cost = outcomes[routing_decision_venue].total_cost_bps

        # Calculate metrics
        was_optimal = (routing_decision_venue == best_venue)
        cost_delta = decision_cost - best_cost
        potential_savings = max(cost_delta, 0.0)

        # Decision quality score (0-1, higher is better)
        # Score based on how close to optimal
        if was_optimal:
            quality_score = 1.0
        else:
            # Penalize based on cost difference
            quality_score = max(0.0, 1.0 - (cost_delta / 10.0))

        return HindsightAnalysis(
            routing_decision_venue=routing_decision_venue,
            actual_best_venue=best_venue,
            was_optimal=was_optimal,
            cost_delta_bps=cost_delta,
            potential_savings_bps=potential_savings,
            decision_quality_score=quality_score
        )

    def _calculate_venue_accuracy(
        self,
        analyses: List[HindsightAnalysis]
    ) -> Dict[str, float]:
        """Calculate routing accuracy per venue"""
        venue_stats = {}

        for analysis in analyses:
            venue = analysis.routing_decision_venue

            if venue not in venue_stats:
                venue_stats[venue] = {"total": 0, "optimal": 0}

            venue_stats[venue]["total"] += 1

            if analysis.was_optimal:
                venue_stats[venue]["optimal"] += 1

        # Calculate accuracy
        venue_accuracy = {}
        for venue, stats in venue_stats.items():
            accuracy = stats["optimal"] / stats["total"] if stats["total"] > 0 else 0.0
            venue_accuracy[venue] = accuracy

        return venue_accuracy

    def calculate_slippage_improvement(
        self,
        baseline_slippage_bps: float,
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate slippage improvement vs baseline

        Args:
            baseline_slippage_bps: Baseline slippage (e.g., from random routing)
            lookback_days: Analysis period

        Returns:
            Slippage metrics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

        # Get recent executions
        recent = [
            ex for ex in self.execution_history
            if ex['timestamp'] >= cutoff_date
        ]

        if not recent:
            return {
                "avg_slippage_bps": 0.0,
                "slippage_improvement_pct": 0.0,
                "slippage_reduction_bps": 0.0
            }

        # Calculate average slippage for SOR decisions
        slippages = []
        for execution in recent:
            venue = execution['routing_decision']
            outcome = execution['outcomes'][venue]
            slippages.append(outcome.slippage_bps)

        avg_slippage = np.mean(slippages)

        # Calculate improvement
        slippage_reduction = baseline_slippage_bps - avg_slippage
        improvement_pct = (slippage_reduction / baseline_slippage_bps) * 100

        logger.info(
            f"Slippage analysis: SOR avg = {avg_slippage:.2f} bps, "
            f"Baseline = {baseline_slippage_bps:.2f} bps, "
            f"Improvement = {improvement_pct:.1f}%"
        )

        # Assert slippage improvement meets target
        if improvement_pct < 10.0:
            logger.warning(
                f"Slippage improvement {improvement_pct:.1f}% below target 10%"
            )

        return {
            "avg_slippage_bps": avg_slippage,
            "baseline_slippage_bps": baseline_slippage_bps,
            "slippage_improvement_pct": improvement_pct,
            "slippage_reduction_bps": slippage_reduction
        }

    def generate_optimization_report(
        self,
        lookback_days: int = 30
    ) -> Dict:
        """Generate comprehensive optimization report"""
        hindsight = self.analyze_hindsight(lookback_days)

        # Calculate slippage improvement (using historical baseline)
        baseline_slippage = 5.0  # Example: 5 bps baseline
        slippage_metrics = self.calculate_slippage_improvement(
            baseline_slippage,
            lookback_days
        )

        # Venue cost analysis
        venue_costs = self._analyze_venue_costs()

        report = {
            "analysis_period_days": lookback_days,
            "generated_at": datetime.utcnow().isoformat(),
            "routing_performance": hindsight,
            "slippage_metrics": slippage_metrics,
            "venue_cost_analysis": venue_costs,
            "recommendations": self._generate_recommendations(hindsight)
        }

        return report

    def _analyze_venue_costs(self) -> Dict[str, Dict]:
        """Analyze costs per venue"""
        venue_costs = {}

        for execution in self.execution_history:
            for venue, outcome in execution['outcomes'].items():
                if venue not in venue_costs:
                    venue_costs[venue] = {
                        "total_cost_bps": [],
                        "slippage_bps": [],
                        "count": 0
                    }

                venue_costs[venue]["total_cost_bps"].append(outcome.total_cost_bps)
                venue_costs[venue]["slippage_bps"].append(outcome.slippage_bps)
                venue_costs[venue]["count"] += 1

        # Calculate averages
        venue_summary = {}
        for venue, costs in venue_costs.items():
            venue_summary[venue] = {
                "avg_total_cost_bps": np.mean(costs["total_cost_bps"]),
                "avg_slippage_bps": np.mean(costs["slippage_bps"]),
                "execution_count": costs["count"]
            }

        return venue_summary

    def _generate_recommendations(
        self,
        hindsight: Dict
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Check routing accuracy
        if hindsight["routing_accuracy"] < 0.95:
            recommendations.append(
                f"Routing accuracy ({hindsight['routing_accuracy']:.1%}) below target. "
                "Consider adjusting venue weights or improving market data quality."
            )

        # Check venue-specific accuracy
        for venue, accuracy in hindsight.get("venue_accuracy", {}).items():
            if accuracy < 0.90:
                recommendations.append(
                    f"Venue {venue} has low routing accuracy ({accuracy:.1%}). "
                    "Review venue profile parameters."
                )

        # Check cost savings opportunity
        if hindsight["total_potential_savings_bps"] > 100:
            recommendations.append(
                f"Potential cost savings of {hindsight['total_potential_savings_bps']:.2f} bps available. "
                "Review and update routing weights."
            )

        if not recommendations:
            recommendations.append("Routing performance is optimal. No changes needed.")

        return recommendations


if __name__ == "__main__":
    """Example usage"""
    logging.basicConfig(level=logging.INFO)

    optimizer = RoutingOptimizer()

    # Simulate some executions
    for i in range(100):
        # Simulate outcomes for different venues
        outcomes = {
            "NASDAQ": ExecutionOutcome(
                venue="NASDAQ",
                fill_price=185.50 + np.random.randn() * 0.02,
                fill_time=datetime.utcnow(),
                slippage_bps=2.5 + np.random.randn() * 0.5,
                total_cost_bps=3.4 + np.random.randn() * 0.3,
                filled_quantity=500
            ),
            "NYSE": ExecutionOutcome(
                venue="NYSE",
                fill_price=185.50 + np.random.randn() * 0.02,
                fill_time=datetime.utcnow(),
                slippage_bps=2.7 + np.random.randn() * 0.5,
                total_cost_bps=3.2 + np.random.randn() * 0.3,
                filled_quantity=500
            ),
            "IEX": ExecutionOutcome(
                venue="IEX",
                fill_price=185.50 + np.random.randn() * 0.03,
                fill_time=datetime.utcnow(),
                slippage_bps=3.0 + np.random.randn() * 0.6,
                total_cost_bps=3.9 + np.random.randn() * 0.4,
                filled_quantity=500
            )
        }

        # Simulate routing decision (90% accuracy simulation)
        best_venue = min(outcomes.items(), key=lambda x: x[1].total_cost_bps)[0]
        decision = best_venue if np.random.rand() > 0.1 else np.random.choice(list(outcomes.keys()))

        optimizer.record_execution(f"order_{i}", decision, outcomes)

    # Generate report
    report = optimizer.generate_optimization_report(lookback_days=30)

    print(f"\n{'='*60}")
    print("OPTIMIZATION REPORT")
    print(f"{'='*60}")
    print(f"\nRouting Accuracy: {report['routing_performance']['routing_accuracy']:.1%}")
    print(f"Avg Cost Delta: {report['routing_performance']['avg_cost_delta_bps']:.2f} bps")
    print(f"Slippage Improvement: {report['slippage_metrics']['slippage_improvement_pct']:.1f}%")
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
