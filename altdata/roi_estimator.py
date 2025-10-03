"""
Alt-Data ROI Estimator

Estimates ROI of alternative data subscriptions to make data-driven
onboarding decisions.

Gate Decision: Onboard only if IR_uplift_per_$ ≥ threshold
"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import numpy as np
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Metrics
roi_calculations = Counter('altdata_roi_calculations_total', 'Total ROI calculations')
onboarding_decisions = Counter('altdata_onboarding_decisions_total', 'Onboarding decisions', ['decision'])


class OnboardingDecision(str, Enum):
    ONBOARD = "ONBOARD"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


@dataclass
class ROIEstimate:
    """ROI estimate for alt-data source"""
    data_source: str
    annual_cost: float
    baseline_sharpe: float
    altdata_sharpe: float
    portfolio_capital: float

    # Calculated metrics
    sharpe_uplift: float
    annual_benefit: float
    roi_percent: float
    ir_uplift_per_dollar: float
    break_even_capital: float

    # Decision
    decision: OnboardingDecision
    decision_reason: str

    # Metadata
    calculation_date: datetime
    assumptions: Dict[str, Any]


class AltDataROIEstimator:
    """Alt-data ROI estimator"""

    def __init__(
        self,
        threshold_ir_per_10k: float = 0.1,
        assumed_volatility: float = 0.15,
        risk_free_rate: float = 0.04
    ):
        """
        Initialize ROI estimator

        Args:
            threshold_ir_per_10k: Minimum IR uplift per $10K (default: 0.1)
            assumed_volatility: Portfolio volatility assumption (default: 15%)
            risk_free_rate: Risk-free rate (default: 4%)
        """
        self.threshold_ir_per_10k = threshold_ir_per_10k
        self.assumed_volatility = assumed_volatility
        self.risk_free_rate = risk_free_rate

    def estimate_roi(
        self,
        data_source: str,
        altdata_cost_annual: float,
        baseline_sharpe: float,
        altdata_sharpe: float,
        portfolio_capital: float
    ) -> ROIEstimate:
        """
        Estimate ROI of alt-data subscription

        Args:
            data_source: Data source name
            altdata_cost_annual: Annual data cost ($)
            baseline_sharpe: Baseline Sharpe ratio
            altdata_sharpe: Sharpe ratio with alt-data
            portfolio_capital: Portfolio capital ($)

        Returns:
            ROIEstimate with decision
        """
        roi_calculations.inc()

        # Calculate uplift
        sharpe_uplift = altdata_sharpe - baseline_sharpe

        # Convert Sharpe to expected return (simplified)
        baseline_return = baseline_sharpe * self.assumed_volatility
        altdata_return = altdata_sharpe * self.assumed_volatility
        return_uplift = altdata_return - baseline_return

        # Calculate annual benefit
        annual_benefit = portfolio_capital * return_uplift

        # Calculate ROI
        roi = (annual_benefit - altdata_cost_annual) / altdata_cost_annual if altdata_cost_annual > 0 else 0

        # Calculate IR uplift per dollar
        ir_uplift_per_dollar = sharpe_uplift / altdata_cost_annual if altdata_cost_annual > 0 else 0

        # Gate decision
        threshold_ir_per_dollar = self.threshold_ir_per_10k / 10000

        # Break-even capital
        break_even_capital = altdata_cost_annual / return_uplift if return_uplift > 0 else float('inf')

        # Make decision
        if sharpe_uplift <= 0:
            decision = OnboardingDecision.REJECT
            decision_reason = f"No Sharpe uplift (uplift: {sharpe_uplift:.4f})"
        elif ir_uplift_per_dollar < threshold_ir_per_dollar:
            decision = OnboardingDecision.REJECT
            decision_reason = f"IR uplift per $ ({ir_uplift_per_dollar:.2e}) below threshold ({threshold_ir_per_dollar:.2e})"
        elif annual_benefit < altdata_cost_annual:
            decision = OnboardingDecision.REVIEW
            decision_reason = f"Annual benefit (${annual_benefit:,.0f}) < cost (${altdata_cost_annual:,.0f}) at current capital"
        else:
            decision = OnboardingDecision.ONBOARD
            decision_reason = f"IR uplift per $ ({ir_uplift_per_dollar:.2e}) ≥ threshold, ROI: {roi*100:.1f}%"

        onboarding_decisions.labels(decision=decision.value).inc()

        estimate = ROIEstimate(
            data_source=data_source,
            annual_cost=altdata_cost_annual,
            baseline_sharpe=baseline_sharpe,
            altdata_sharpe=altdata_sharpe,
            portfolio_capital=portfolio_capital,
            sharpe_uplift=sharpe_uplift,
            annual_benefit=annual_benefit,
            roi_percent=roi * 100,
            ir_uplift_per_dollar=ir_uplift_per_dollar,
            break_even_capital=break_even_capital,
            decision=decision,
            decision_reason=decision_reason,
            calculation_date=datetime.utcnow(),
            assumptions={
                'volatility': self.assumed_volatility,
                'risk_free_rate': self.risk_free_rate,
                'threshold_ir_per_10k': self.threshold_ir_per_10k
            }
        )

        logger.info(f"ROI estimate for {data_source}: {decision.value} - {decision_reason}")

        return estimate

    def sensitivity_analysis(
        self,
        data_source: str,
        altdata_cost_annual: float,
        baseline_sharpe: float,
        altdata_sharpe: float,
        capital_range: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on portfolio capital

        Args:
            data_source: Data source name
            altdata_cost_annual: Annual data cost
            baseline_sharpe: Baseline Sharpe
            altdata_sharpe: Sharpe with alt-data
            capital_range: (min, max) capital range (default: 1M to 100M)

        Returns:
            Sensitivity analysis results
        """
        if capital_range is None:
            capital_range = (1_000_000, 100_000_000)

        min_capital, max_capital = capital_range
        capital_levels = np.logspace(
            np.log10(min_capital),
            np.log10(max_capital),
            num=20
        )

        results = []

        for capital in capital_levels:
            estimate = self.estimate_roi(
                data_source=data_source,
                altdata_cost_annual=altdata_cost_annual,
                baseline_sharpe=baseline_sharpe,
                altdata_sharpe=altdata_sharpe,
                portfolio_capital=capital
            )

            results.append({
                'capital': capital,
                'annual_benefit': estimate.annual_benefit,
                'roi_percent': estimate.roi_percent,
                'decision': estimate.decision.value
            })

        # Find break-even capital
        sharpe_uplift = altdata_sharpe - baseline_sharpe
        return_uplift = sharpe_uplift * self.assumed_volatility
        break_even = altdata_cost_annual / return_uplift if return_uplift > 0 else float('inf')

        return {
            'data_source': data_source,
            'sensitivity_results': results,
            'break_even_capital': break_even,
            'min_capital_tested': min_capital,
            'max_capital_tested': max_capital
        }

    def compare_sources(
        self,
        comparisons: list[Dict[str, Any]],
        portfolio_capital: float
    ) -> Dict[str, Any]:
        """
        Compare multiple alt-data sources

        Args:
            comparisons: List of dicts with 'data_source', 'annual_cost', 'baseline_sharpe', 'altdata_sharpe'
            portfolio_capital: Portfolio capital

        Returns:
            Comparison results ranked by ROI
        """
        estimates = []

        for comp in comparisons:
            estimate = self.estimate_roi(
                data_source=comp['data_source'],
                altdata_cost_annual=comp['annual_cost'],
                baseline_sharpe=comp['baseline_sharpe'],
                altdata_sharpe=comp['altdata_sharpe'],
                portfolio_capital=portfolio_capital
            )
            estimates.append(estimate)

        # Sort by ROI
        estimates.sort(key=lambda x: x.roi_percent, reverse=True)

        # Identify best options
        onboard_candidates = [e for e in estimates if e.decision == OnboardingDecision.ONBOARD]
        review_candidates = [e for e in estimates if e.decision == OnboardingDecision.REVIEW]

        return {
            'portfolio_capital': portfolio_capital,
            'num_sources_evaluated': len(estimates),
            'onboard_candidates': len(onboard_candidates),
            'review_candidates': len(review_candidates),
            'best_source': estimates[0].data_source if estimates else None,
            'best_roi': estimates[0].roi_percent if estimates else None,
            'all_estimates': [asdict(e) for e in estimates]
        }

    def quarterly_review(
        self,
        data_source: str,
        projected_benefit: float,
        actual_benefit: float,
        annual_cost: float
    ) -> Dict[str, Any]:
        """
        Quarterly ROI review (actual vs projected)

        Args:
            data_source: Data source name
            projected_benefit: Projected quarterly benefit
            actual_benefit: Actual quarterly benefit
            annual_cost: Annual cost

        Returns:
            Review results with recommendation
        """
        variance = actual_benefit - projected_benefit
        variance_pct = (variance / projected_benefit * 100) if projected_benefit > 0 else 0

        quarterly_cost = annual_cost / 4
        actual_roi = (actual_benefit - quarterly_cost) / quarterly_cost * 100 if quarterly_cost > 0 else 0

        # Recommendation
        if actual_benefit < quarterly_cost:
            recommendation = "CONSIDER_CANCELLATION"
            reason = f"Actual benefit (${actual_benefit:,.0f}) < quarterly cost (${quarterly_cost:,.0f})"
        elif variance_pct < -20:
            recommendation = "REVIEW_REQUIRED"
            reason = f"Actual benefit {variance_pct:.1f}% below projection"
        elif variance_pct < -50:
            recommendation = "IMMEDIATE_REVIEW"
            reason = f"Actual benefit significantly below projection ({variance_pct:.1f}%)"
        else:
            recommendation = "CONTINUE"
            reason = f"Performance meeting expectations (ROI: {actual_roi:.1f}%)"

        return {
            'data_source': data_source,
            'review_date': datetime.utcnow().date(),
            'projected_benefit': projected_benefit,
            'actual_benefit': actual_benefit,
            'variance': variance,
            'variance_percent': variance_pct,
            'quarterly_cost': quarterly_cost,
            'actual_roi_percent': actual_roi,
            'recommendation': recommendation,
            'reason': reason
        }


def format_roi_report(estimate: ROIEstimate) -> str:
    """Format ROI estimate as readable report"""

    report = f"""
==============================================
Alt-Data ROI Estimate
==============================================

Data Source: {estimate.data_source}
Calculation Date: {estimate.calculation_date.strftime('%Y-%m-%d')}

BASELINE PERFORMANCE
-------------------
Sharpe Ratio: {estimate.baseline_sharpe:.3f}

WITH ALT-DATA
-------------
Sharpe Ratio: {estimate.altdata_sharpe:.3f}
Sharpe Uplift: {estimate.sharpe_uplift:.3f}

FINANCIAL ANALYSIS
------------------
Portfolio Capital: ${estimate.portfolio_capital:,.0f}
Annual Data Cost: ${estimate.annual_cost:,.0f}
Annual Benefit: ${estimate.annual_benefit:,.0f}
ROI: {estimate.roi_percent:.1f}%

EFFICIENCY METRICS
------------------
IR Uplift per $: {estimate.ir_uplift_per_dollar:.2e}
Threshold: {estimate.assumptions['threshold_ir_per_10k'] / 10000:.2e}
Break-even Capital: ${estimate.break_even_capital:,.0f}

DECISION
--------
{estimate.decision.value}: {estimate.decision_reason}

ASSUMPTIONS
-----------
Volatility: {estimate.assumptions['volatility']:.1%}
Risk-free Rate: {estimate.assumptions['risk_free_rate']:.1%}

==============================================
"""

    return report
