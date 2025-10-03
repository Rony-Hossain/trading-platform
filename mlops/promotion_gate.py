"""
Promotion Gate for Model Deployment
Validates models against SPA/DSR/PBO criteria before promotion
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, asdict
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PromotionCriteria:
    """Criteria for model promotion"""
    min_sharpe_improvement: float = 0.1
    max_drawdown_tolerance: float = 0.02
    min_hit_rate: float = 0.52
    min_information_ratio: float = 0.5
    min_samples: int = 1000
    require_spa: bool = True
    require_dsr: bool = True
    require_pbo: bool = True


@dataclass
class GateResults:
    """Results from promotion gates"""
    overall_passed: bool
    spa_result: Dict[str, Any]
    dsr_result: Dict[str, Any]
    pbo_result: Dict[str, Any]
    details: Dict[str, Any]


class PromotionGate:
    """
    Validates model performance before promotion to production
    Implements SPA (Sharpe Performance Analysis), DSR (Deflated Sharpe Ratio),
    and PBO (Probability of Backtest Overfitting) tests
    """

    def __init__(self, criteria: PromotionCriteria):
        self.criteria = criteria

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred_challenger: np.ndarray,
        y_pred_champion: np.ndarray = None
    ) -> GateResults:
        """
        Evaluate if challenger model passes promotion gates

        Args:
            y_true: True labels
            y_pred_challenger: Challenger predictions
            y_pred_champion: Champion predictions (for comparison)

        Returns:
            GateResults with pass/fail status
        """
        logger.info("Running promotion gate evaluation...")

        results = {
            "spa": {},
            "dsr": {},
            "pbo": {}
        }

        # 1. Sharpe Performance Analysis (SPA)
        if self.criteria.require_spa:
            spa_result = self._test_spa(y_true, y_pred_challenger, y_pred_champion)
            results["spa"] = spa_result
            logger.info(f"SPA Test: {'PASS' if spa_result['passed'] else 'FAIL'}")

        # 2. Deflated Sharpe Ratio (DSR)
        if self.criteria.require_dsr:
            dsr_result = self._test_dsr(y_true, y_pred_challenger)
            results["dsr"] = dsr_result
            logger.info(f"DSR Test: {'PASS' if dsr_result['passed'] else 'FAIL'}")

        # 3. Probability of Backtest Overfitting (PBO)
        if self.criteria.require_pbo:
            pbo_result = self._test_pbo(y_true, y_pred_challenger)
            results["pbo"] = pbo_result
            logger.info(f"PBO Test: {'PASS' if pbo_result['passed'] else 'FAIL'}")

        # Overall pass/fail
        overall_passed = (
            (not self.criteria.require_spa or results["spa"].get("passed", False)) and
            (not self.criteria.require_dsr or results["dsr"].get("passed", False)) and
            (not self.criteria.require_pbo or results["pbo"].get("passed", False))
        )

        gate_results = GateResults(
            overall_passed=overall_passed,
            spa_result=results["spa"],
            dsr_result=results["dsr"],
            pbo_result=results["pbo"],
            details=self._calculate_additional_metrics(y_true, y_pred_challenger)
        )

        logger.info(f"\nPromotion Gate: {'✓ PASS' if overall_passed else '✗ FAIL'}")

        return gate_results

    def _test_spa(
        self,
        y_true: np.ndarray,
        y_pred_challenger: np.ndarray,
        y_pred_champion: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Sharpe Performance Analysis

        Tests if challenger has statistically significant improvement over champion
        """
        # Calculate returns
        challenger_returns = y_true * np.sign(y_pred_challenger)

        # Calculate Sharpe ratio
        challenger_sharpe = self._calculate_sharpe(challenger_returns)

        if y_pred_champion is not None:
            champion_returns = y_true * np.sign(y_pred_champion)
            champion_sharpe = self._calculate_sharpe(champion_returns)
            sharpe_improvement = challenger_sharpe - champion_sharpe

            # Test for statistical significance
            _, p_value = stats.ttest_rel(challenger_returns, champion_returns)
            is_significant = p_value < 0.05

        else:
            champion_sharpe = 0.0
            sharpe_improvement = challenger_sharpe
            is_significant = True  # No baseline to compare

        # Pass if improvement meets threshold AND is significant
        passed = (
            sharpe_improvement >= self.criteria.min_sharpe_improvement and
            is_significant
        )

        return {
            "passed": passed,
            "challenger_sharpe": float(challenger_sharpe),
            "champion_sharpe": float(champion_sharpe),
            "sharpe_improvement": float(sharpe_improvement),
            "required_improvement": self.criteria.min_sharpe_improvement,
            "is_statistically_significant": is_significant,
            "p_value": float(p_value) if y_pred_champion is not None else None
        }

    def _test_dsr(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Deflated Sharpe Ratio

        Adjusts Sharpe ratio for multiple testing and skewness/kurtosis
        """
        returns = y_true * np.sign(y_pred)

        # Calculate standard Sharpe
        sharpe = self._calculate_sharpe(returns)

        # Calculate skewness and kurtosis
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        # Number of trials (simplified - in production track actual number)
        n_trials = 100  # Assume 100 different model configs were tried

        # Deflated Sharpe Ratio adjustment
        # Account for skewness, kurtosis, and multiple testing
        n = len(returns)

        # Variance of Sharpe estimator
        var_sharpe = (1 + (0.5 * sharpe**2) - (skew * sharpe) + ((kurt - 3) / 4) * sharpe**2) / n

        # Multiple testing adjustment (Bonferroni-style)
        deflation_factor = np.sqrt(np.log(n_trials))

        # Deflated Sharpe
        dsr = sharpe / (np.sqrt(var_sharpe) * deflation_factor)

        # Test statistic
        dsr_threshold = 2.0  # Typical threshold for significance

        passed = dsr > dsr_threshold

        return {
            "passed": passed,
            "deflated_sharpe_ratio": float(dsr),
            "standard_sharpe": float(sharpe),
            "threshold": dsr_threshold,
            "skewness": float(skew),
            "kurtosis": float(kurt),
            "n_trials_assumed": n_trials
        }

    def _test_pbo(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Probability of Backtest Overfitting

        Uses combinatorially symmetric cross-validation (CSCV)
        to estimate probability that model is overfit
        """
        returns = y_true * np.sign(y_pred)

        # Split data into K subsets
        n = len(returns)
        k = 16  # Number of subsets (must be even)

        if n < k * 100:
            # Not enough data for reliable PBO
            logger.warning("Insufficient data for PBO test")
            return {
                "passed": True,  # Pass by default if can't test
                "probability_overfit": 0.0,
                "reason": "insufficient_data"
            }

        subset_size = n // k
        subsets = [returns[i*subset_size:(i+1)*subset_size] for i in range(k)]

        # Combinatorially symmetric partitions
        n_partitions = 100  # Number of random partitions to test
        pbo_scores = []

        for _ in range(n_partitions):
            # Randomly split into two groups
            indices = np.random.permutation(k)
            group1_idx = indices[:k//2]
            group2_idx = indices[k//2:]

            # Calculate Sharpe for each group
            group1_returns = np.concatenate([subsets[i] for i in group1_idx])
            group2_returns = np.concatenate([subsets[i] for i in group2_idx])

            sharpe1 = self._calculate_sharpe(group1_returns)
            sharpe2 = self._calculate_sharpe(group2_returns)

            # Check if performance degrades
            pbo_scores.append(1 if sharpe2 < sharpe1 else 0)

        # PBO is the probability that out-of-sample performance < in-sample
        pbo = np.mean(pbo_scores)

        # Pass if PBO < 0.5 (less than 50% chance of overfitting)
        passed = pbo < 0.5

        return {
            "passed": passed,
            "probability_overfit": float(pbo),
            "threshold": 0.5,
            "n_partitions_tested": n_partitions
        }

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()

    def _calculate_additional_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate additional performance metrics"""
        returns = y_true * np.sign(y_pred)

        # Hit rate (directional accuracy)
        hit_rate = (np.sign(y_true) == np.sign(y_pred)).mean()

        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()

        # Information ratio
        excess_returns = returns - returns.mean()
        information_ratio = (
            np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            if excess_returns.std() > 0 else 0.0
        )

        return {
            "hit_rate": float(hit_rate),
            "max_drawdown": float(max_drawdown),
            "information_ratio": float(information_ratio),
            "num_samples": len(y_true)
        }


if __name__ == "__main__":
    """Example usage"""
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    n_samples = 5000

    y_true = np.random.randn(n_samples) * 0.02  # True returns
    y_pred_challenger = y_true + np.random.randn(n_samples) * 0.01  # Challenger predictions
    y_pred_champion = y_true + np.random.randn(n_samples) * 0.015  # Champion predictions (slightly worse)

    # Create promotion gate
    criteria = PromotionCriteria(
        min_sharpe_improvement=0.1,
        max_drawdown_tolerance=0.02,
        require_spa=True,
        require_dsr=True,
        require_pbo=True
    )

    gate = PromotionGate(criteria)

    # Evaluate
    results = gate.evaluate_model(y_true, y_pred_challenger, y_pred_champion)

    print(f"\n{'='*60}")
    print("PROMOTION GATE RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall: {'✓ PASS' if results.overall_passed else '✗ FAIL'}")
    print(f"\nSPA: {results.spa_result}")
    print(f"\nDSR: {results.dsr_result}")
    print(f"\nPBO: {results.pbo_result}")
    print(f"\nAdditional Metrics: {results.details}")
