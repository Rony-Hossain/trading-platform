"""
Deflated Sharpe Ratio & Probability of Backtest Overfitting (PBO) Framework

Implements advanced overfitting detection methods to identify when strategy performance
is likely due to data mining rather than genuine skill. Based on López de Prado (2014)
and Marcos et al. research.

Part of Phase 3B institutional statistical rigor framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, date
from scipy import stats
from scipy.stats import norm, skew, kurtosis
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from sklearn.model_selection import ParameterGrid
import itertools

logger = logging.getLogger(__name__)

class OverfittingMethod(Enum):
    """Overfitting detection methods"""
    DEFLATED_SHARPE = "deflated_sharpe"
    PBO_HAIRCUT = "pbo_haircut"
    PBO_PROBABILITY = "pbo_probability" 
    COMBINATORIAL_PURGE = "combinatorial_purge"

class SelectionMethod(Enum):
    """Strategy selection methods for PBO analysis"""
    MAX_SHARPE = "max_sharpe"
    MAX_RETURN = "max_return"
    MIN_VOLATILITY = "min_volatility"
    MAX_CALMAR = "max_calmar"
    CUSTOM_SCORE = "custom_score"

@dataclass
class BacktestConfiguration:
    """Configuration for a single backtest"""
    strategy_name: str
    parameters: Dict[str, Any]
    returns: np.ndarray
    sharpe_ratio: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    calmar_ratio: float
    
    # Additional metrics for analysis
    skewness: Optional[float] = None
    excess_kurtosis: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    
    def __post_init__(self):
        """Calculate additional metrics if not provided"""
        if self.skewness is None:
            self.skewness = skew(self.returns)
        if self.excess_kurtosis is None:
            self.excess_kurtosis = kurtosis(self.returns, fisher=True)
        if self.var_95 is None:
            self.var_95 = np.percentile(self.returns, 5)
        if self.cvar_95 is None:
            # Conditional VaR (Expected Shortfall)
            self.cvar_95 = np.mean(self.returns[self.returns <= self.var_95])

@dataclass
class DeflatedSharpeResult:
    """Results from Deflated Sharpe Ratio analysis"""
    observed_sharpe: float
    deflated_sharpe: float
    trials_factor: float
    length_factor: float
    variance_factor: float
    skewness_kurtosis_factor: float
    
    # Statistical inference
    p_value: float
    is_significant_95: bool
    is_significant_99: bool
    
    # Metadata
    n_trials: int
    n_observations: int
    strategy_name: str = ""
    
    @property
    def deflation_magnitude(self) -> float:
        """How much the Sharpe ratio was deflated"""
        if self.observed_sharpe == 0:
            return 0.0
        return (self.observed_sharpe - self.deflated_sharpe) / abs(self.observed_sharpe)

@dataclass
class PBOResult:
    """Results from Probability of Backtest Overfitting analysis"""
    pbo_probability: float
    performance_haircut: float
    rank_correlation: float
    
    # Strategy selection analysis
    best_is_strategy: str
    best_oos_strategy: str
    selection_consistency: float
    
    # Distribution analysis
    is_performance_distribution: np.ndarray
    oos_performance_distribution: np.ndarray
    
    # Statistical tests
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    
    # Metadata
    n_trials: int
    n_strategies_per_trial: int
    selection_method: SelectionMethod
    
    @property
    def overfitting_risk(self) -> str:
        """Classify overfitting risk level"""
        if self.pbo_probability >= 0.7:
            return "HIGH"
        elif self.pbo_probability >= 0.5:
            return "MODERATE"
        elif self.pbo_probability >= 0.3:
            return "LOW"
        else:
            return "MINIMAL"

class OverfittingDetector:
    """
    Advanced overfitting detection using Deflated Sharpe Ratio and PBO analysis
    """
    
    def __init__(self,
                 n_bootstrap_trials: int = 1000,
                 significance_level: float = 0.05,
                 random_seed: Optional[int] = None):
        
        self.n_bootstrap_trials = n_bootstrap_trials
        self.significance_level = significance_level
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def deflated_sharpe_ratio(self,
                            observed_sharpe: float,
                            n_trials: int,
                            n_observations: int,
                            returns: Optional[np.ndarray] = None,
                            skewness: Optional[float] = None,
                            kurtosis: Optional[float] = None,
                            strategy_name: str = "") -> DeflatedSharpeResult:
        """
        Calculate Deflated Sharpe Ratio accounting for multiple testing and non-normality
        
        Based on Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
        """
        logger.info(f"Calculating Deflated Sharpe Ratio for {strategy_name}")
        
        # Calculate higher moments if not provided
        if returns is not None:
            if skewness is None:
                skewness = stats.skew(returns)
            if kurtosis is None:
                kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        else:
            skewness = skewness or 0.0
            kurtosis = kurtosis or 0.0
        
        # Trials factor - accounts for multiple testing
        trials_factor = np.sqrt(np.log(n_trials))
        
        # Length factor - accounts for sample size
        length_factor = 1.0 / np.sqrt(n_observations)
        
        # Variance factor from higher moments
        variance_factor = np.sqrt(
            1 + 
            (1 - skewness * observed_sharpe) / 4 +
            ((kurtosis - 1) * observed_sharpe**2) / 24
        )
        
        # Combined skewness-kurtosis adjustment
        skewness_kurtosis_factor = (
            (skewness / 6) * observed_sharpe**2 +
            ((kurtosis - 3) / 24) * observed_sharpe**3
        )
        
        # Deflated Sharpe Ratio
        deflated_sharpe = (
            observed_sharpe - 
            trials_factor * length_factor * variance_factor -
            skewness_kurtosis_factor
        )
        
        # Statistical significance test
        # Under null hypothesis, deflated Sharpe ~ N(0,1)
        p_value = 2 * (1 - norm.cdf(abs(deflated_sharpe)))  # Two-tailed test
        
        result = DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=deflated_sharpe,
            trials_factor=trials_factor,
            length_factor=length_factor,
            variance_factor=variance_factor,
            skewness_kurtosis_factor=skewness_kurtosis_factor,
            p_value=p_value,
            is_significant_95=p_value < 0.05,
            is_significant_99=p_value < 0.01,
            n_trials=n_trials,
            n_observations=n_observations,
            strategy_name=strategy_name
        )
        
        logger.info(f"Deflated Sharpe: {deflated_sharpe:.4f} (original: {observed_sharpe:.4f}), "
                   f"p-value: {p_value:.4f}")
        
        return result
    
    def probability_backtest_overfitting(self,
                                       backtest_configs: List[BacktestConfiguration],
                                       selection_method: SelectionMethod = SelectionMethod.MAX_SHARPE,
                                       n_splits: int = 16) -> PBOResult:
        """
        Calculate Probability of Backtest Overfitting using combinatorial cross-validation
        
        Based on Bailey et al. (2016) "The Probability of Backtest Overfitting"
        """
        logger.info(f"Calculating PBO for {len(backtest_configs)} strategies with {n_splits} splits")
        
        if len(backtest_configs) < 2:
            raise ValueError("Need at least 2 backtest configurations for PBO analysis")
        
        # Validate that all configurations have same length
        n_observations = len(backtest_configs[0].returns)
        if not all(len(config.returns) == n_observations for config in backtest_configs):
            raise ValueError("All backtest configurations must have same number of observations")
        
        # Generate combinatorial splits
        is_performance = []
        oos_performance = []
        best_is_strategies = []
        best_oos_strategies = []
        
        # Use combinatorial purged cross-validation
        split_combinations = self._generate_combinatorial_splits(n_observations, n_splits)
        
        for train_indices, test_indices in split_combinations:
            # Calculate in-sample performance
            is_performances = {}
            oos_performances = {}
            
            for config in backtest_configs:
                train_returns = config.returns[train_indices]
                test_returns = config.returns[test_indices]
                
                # Calculate performance metrics for both periods
                is_perf = self._calculate_performance_metric(train_returns, selection_method)
                oos_perf = self._calculate_performance_metric(test_returns, selection_method)
                
                is_performances[config.strategy_name] = is_perf
                oos_performances[config.strategy_name] = oos_perf
            
            # Select best strategy based on in-sample performance
            best_is_strategy = max(is_performances, key=is_performances.get)
            best_oos_strategy = max(oos_performances, key=oos_performances.get)
            
            # Record results
            is_performance.append(is_performances[best_is_strategy])
            oos_performance.append(oos_performances[best_is_strategy])  # OOS performance of IS-selected strategy
            best_is_strategies.append(best_is_strategy)
            best_oos_strategies.append(best_oos_strategy)
        
        # Convert to numpy arrays
        is_performance = np.array(is_performance)
        oos_performance = np.array(oos_performance)
        
        # Calculate PBO probability
        pbo_probability = np.mean(oos_performance <= 0)
        
        # Calculate performance haircut
        performance_haircut = np.mean(is_performance) - np.mean(oos_performance)
        
        # Rank correlation between IS and OOS performance
        try:
            rank_correlation, _ = stats.spearmanr(is_performance, oos_performance)
            if np.isnan(rank_correlation):
                rank_correlation = 0.0
        except:
            rank_correlation = 0.0
        
        # Selection consistency
        most_common_strategy = max(set(best_is_strategies), key=best_is_strategies.count)
        selection_consistency = best_is_strategies.count(most_common_strategy) / len(best_is_strategies)
        
        # Wilcoxon signed-rank test for IS vs OOS performance
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(is_performance, oos_performance)
        except:
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
        
        result = PBOResult(
            pbo_probability=pbo_probability,
            performance_haircut=performance_haircut,
            rank_correlation=rank_correlation,
            best_is_strategy=most_common_strategy,
            best_oos_strategy=max(set(best_oos_strategies), key=best_oos_strategies.count),
            selection_consistency=selection_consistency,
            is_performance_distribution=is_performance,
            oos_performance_distribution=oos_performance,
            wilcoxon_statistic=wilcoxon_stat,
            wilcoxon_p_value=wilcoxon_p,
            n_trials=len(split_combinations),
            n_strategies_per_trial=len(backtest_configs),
            selection_method=selection_method
        )
        
        logger.info(f"PBO Analysis completed: Probability={pbo_probability:.3f}, "
                   f"Haircut={performance_haircut:.4f}, Risk={result.overfitting_risk}")
        
        return result
    
    def comprehensive_overfitting_analysis(self,
                                         backtest_configs: List[BacktestConfiguration],
                                         n_trials_estimate: Optional[int] = None) -> Dict[str, Any]:
        """
        Comprehensive overfitting analysis combining multiple methods
        """
        logger.info(f"Running comprehensive overfitting analysis on {len(backtest_configs)} strategies")
        
        if n_trials_estimate is None:
            # Estimate number of trials based on parameter space
            n_trials_estimate = self._estimate_trials_from_configs(backtest_configs)
        
        results = {
            "metadata": {
                "analysis_date": datetime.utcnow().isoformat(),
                "n_strategies": len(backtest_configs),
                "estimated_trials": n_trials_estimate,
                "n_observations": len(backtest_configs[0].returns) if backtest_configs else 0
            },
            "deflated_sharpe_results": {},
            "pbo_analysis": {},
            "overfitting_summary": {},
            "recommendations": []
        }
        
        # 1. Deflated Sharpe Ratio analysis for each strategy
        deflated_results = {}
        for config in backtest_configs:
            deflated_result = self.deflated_sharpe_ratio(
                observed_sharpe=config.sharpe_ratio,
                n_trials=n_trials_estimate,
                n_observations=len(config.returns),
                returns=config.returns,
                strategy_name=config.strategy_name
            )
            deflated_results[config.strategy_name] = deflated_result
        
        results["deflated_sharpe_results"] = deflated_results
        
        # 2. PBO analysis across all strategies
        if len(backtest_configs) >= 2:
            try:
                pbo_result = self.probability_backtest_overfitting(
                    backtest_configs=backtest_configs,
                    selection_method=SelectionMethod.MAX_SHARPE
                )
                results["pbo_analysis"] = pbo_result
            except Exception as e:
                logger.error(f"PBO analysis failed: {e}")
                results["pbo_analysis"] = None
        
        # 3. Generate overfitting summary and recommendations
        results["overfitting_summary"] = self._generate_overfitting_summary(results)
        results["recommendations"] = self._generate_overfitting_recommendations(results)
        
        return results
    
    def _generate_combinatorial_splits(self, n_observations: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate combinatorial purged cross-validation splits"""
        
        # Calculate split size
        split_size = n_observations // n_splits
        
        splits = []
        for i in range(n_splits):
            # Test set: one split
            test_start = i * split_size
            test_end = min((i + 1) * split_size, n_observations)
            test_indices = np.arange(test_start, test_end)
            
            # Training set: all other data
            train_indices = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, n_observations)
            ])
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        # Add additional combinations for robustness
        # Use random splits as well
        for _ in range(min(8, n_splits)):
            indices = np.random.permutation(n_observations)
            split_point = n_observations // 2
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]
            splits.append((train_indices, test_indices))
        
        return splits
    
    def _calculate_performance_metric(self, returns: np.ndarray, method: SelectionMethod) -> float:
        """Calculate performance metric for strategy selection"""
        
        if len(returns) == 0:
            return 0.0
        
        if method == SelectionMethod.MAX_SHARPE:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            return (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        
        elif method == SelectionMethod.MAX_RETURN:
            return np.mean(returns) * 252  # Annualized
        
        elif method == SelectionMethod.MIN_VOLATILITY:
            return -np.std(returns, ddof=1) * np.sqrt(252)  # Negative for maximization
        
        elif method == SelectionMethod.MAX_CALMAR:
            annual_return = np.mean(returns) * 252
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = np.min(drawdown)
            return annual_return / abs(max_dd) if max_dd != 0 else 0.0
        
        else:  # Custom score - could be implemented based on specific needs
            return np.mean(returns) * 252  # Default to annual return
    
    def _estimate_trials_from_configs(self, configs: List[BacktestConfiguration]) -> int:
        """Estimate number of trials based on parameter configurations"""
        
        # Extract unique parameter combinations
        all_params = []
        for config in configs:
            param_tuple = tuple(sorted(config.parameters.items()))
            all_params.append(param_tuple)
        
        unique_params = set(all_params)
        
        # If we have explicit parameter grids, estimate combinations
        if len(unique_params) > 1:
            # Try to estimate the parameter space size
            param_keys = set()
            for config in configs:
                param_keys.update(config.parameters.keys())
            
            if param_keys:
                # Conservative estimate: assume 5-10 values per parameter
                avg_values_per_param = max(5, len(configs) ** (1/len(param_keys)))
                estimated_trials = int(avg_values_per_param ** len(param_keys))
                
                # Cap at reasonable maximum
                return min(estimated_trials, 10000)
        
        # Default estimate based on number of configurations
        return max(len(configs), 100)
    
    def _generate_overfitting_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of overfitting analysis"""
        
        summary = {
            "overall_overfitting_risk": "UNKNOWN",
            "significant_strategies_deflated": [],
            "overfitted_strategies": [],
            "robust_strategies": [],
            "key_metrics": {}
        }
        
        # Analyze deflated Sharpe results
        deflated_results = results.get("deflated_sharpe_results", {})
        significant_count = 0
        overfitted_count = 0
        
        for strategy_name, deflated_result in deflated_results.items():
            if deflated_result.is_significant_95:
                summary["significant_strategies_deflated"].append(strategy_name)
                significant_count += 1
            
            # Consider overfitted if deflation is severe (>50% reduction)
            if deflated_result.deflation_magnitude > 0.5:
                summary["overfitted_strategies"].append(strategy_name)
                overfitted_count += 1
            elif deflated_result.is_significant_95 and deflated_result.deflation_magnitude < 0.2:
                summary["robust_strategies"].append(strategy_name)
        
        # Analyze PBO results
        pbo_result = results.get("pbo_analysis")
        if pbo_result:
            summary["key_metrics"]["pbo_probability"] = pbo_result.pbo_probability
            summary["key_metrics"]["performance_haircut"] = pbo_result.performance_haircut
            summary["key_metrics"]["pbo_risk_level"] = pbo_result.overfitting_risk
            
            # Overall risk assessment
            if pbo_result.pbo_probability >= 0.7 or overfitted_count > significant_count:
                summary["overall_overfitting_risk"] = "HIGH"
            elif pbo_result.pbo_probability >= 0.5 or overfitted_count > 0:
                summary["overall_overfitting_risk"] = "MODERATE"
            elif significant_count > 0:
                summary["overall_overfitting_risk"] = "LOW"
            else:
                summary["overall_overfitting_risk"] = "MINIMAL"
        
        summary["key_metrics"]["n_significant_deflated"] = significant_count
        summary["key_metrics"]["n_overfitted"] = overfitted_count
        summary["key_metrics"]["n_robust"] = len(summary["robust_strategies"])
        
        return summary
    
    def _generate_overfitting_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on overfitting analysis"""
        
        recommendations = []
        summary = results.get("overfitting_summary", {})
        pbo_result = results.get("pbo_analysis")
        
        overall_risk = summary.get("overall_overfitting_risk", "UNKNOWN")
        
        if overall_risk == "HIGH":
            recommendations.extend([
                "HIGH OVERFITTING RISK DETECTED - Exercise extreme caution",
                "Consider significantly larger out-of-sample validation periods",
                "Reduce parameter optimization complexity",
                "Implement walk-forward analysis for validation",
                "Consider ensemble methods to reduce overfitting"
            ])
        
        elif overall_risk == "MODERATE":
            recommendations.extend([
                "Moderate overfitting risk - Additional validation recommended",
                "Extend out-of-sample testing period",
                "Consider paper trading before live deployment",
                "Monitor strategies closely for performance degradation"
            ])
        
        elif overall_risk == "LOW":
            recommendations.extend([
                "Low overfitting risk - Strategies appear robust",
                "Continue with standard validation procedures",
                "Monitor performance regularly for early warning signs"
            ])
        
        # Specific recommendations based on deflated Sharpe results
        deflated_results = results.get("deflated_sharpe_results", {})
        avg_deflation = np.mean([r.deflation_magnitude for r in deflated_results.values()])
        
        if avg_deflation > 0.3:
            recommendations.append(
                f"Average Sharpe deflation of {avg_deflation:.1%} suggests excessive optimization"
            )
        
        # PBO-specific recommendations
        if pbo_result:
            if pbo_result.performance_haircut > 0.02:  # 2% annual
                recommendations.append(
                    f"Large performance haircut ({pbo_result.performance_haircut:.2%}) indicates backtest inflation"
                )
            
            if pbo_result.rank_correlation < 0.5:
                recommendations.append(
                    "Poor IS/OOS correlation suggests unstable strategy ranking"
                )
        
        # Strategy-specific recommendations
        if summary.get("robust_strategies"):
            recommendations.append(
                f"Focus deployment on robust strategies: {', '.join(summary['robust_strategies'][:3])}"
            )
        
        if summary.get("overfitted_strategies"):
            recommendations.append(
                f"Avoid overfitted strategies: {', '.join(summary['overfitted_strategies'][:3])}"
            )
        
        return recommendations

# Utility functions for creating backtest configurations

def create_backtest_configuration(strategy_name: str,
                                parameters: Dict[str, Any],
                                returns: np.ndarray) -> BacktestConfiguration:
    """Create a BacktestConfiguration from returns and parameters"""
    
    # Calculate performance metrics
    annual_return = np.mean(returns) * 252
    annual_volatility = np.std(returns, ddof=1) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    
    # Calculate max drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    return BacktestConfiguration(
        strategy_name=strategy_name,
        parameters=parameters,
        returns=returns,
        sharpe_ratio=sharpe_ratio,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio
    )

def generate_parameter_grid_configs(base_returns: np.ndarray,
                                  parameter_grid: Dict[str, List[Any]],
                                  strategy_name_template: str = "strategy_{}_{}") -> List[BacktestConfiguration]:
    """
    Generate backtest configurations from parameter grid
    
    This simulates the process of parameter optimization by creating
    synthetic variations of returns based on parameter combinations.
    """
    
    configs = []
    param_combinations = list(ParameterGrid(parameter_grid))
    
    for i, params in enumerate(param_combinations):
        # Create synthetic returns variation
        # In practice, this would come from actual backtesting with different parameters
        noise_factor = np.random.normal(0, 0.1, len(base_returns))  # Add parameter-dependent noise
        modified_returns = base_returns + noise_factor * np.std(base_returns) * 0.2
        
        strategy_name = strategy_name_template.format(i, "_".join(f"{k}{v}" for k, v in params.items()))
        
        config = create_backtest_configuration(
            strategy_name=strategy_name,
            parameters=params,
            returns=modified_returns
        )
        configs.append(config)
    
    return configs

# Example usage functions

def simulate_overfitting_example() -> Dict[str, Any]:
    """
    Simulate an example of overfitting detection analysis
    """
    np.random.seed(42)
    
    # Generate base strategy returns
    n_obs = 252 * 2  # 2 years of daily data
    base_returns = np.random.normal(0.0008, 0.02, n_obs)  # Modest positive returns
    
    # Create parameter grid (simulating optimization)
    parameter_grid = {
        'lookback_period': [10, 20, 50, 100],
        'threshold': [0.01, 0.02, 0.05],
        'rebalance_freq': [1, 5, 10]
    }
    
    # Generate backtest configurations
    configs = generate_parameter_grid_configs(base_returns, parameter_grid)
    
    # Add some genuinely different strategies
    configs.extend([
        create_backtest_configuration(
            "momentum_strategy",
            {"type": "momentum", "period": 20},
            np.random.normal(0.0012, 0.025, n_obs)
        ),
        create_backtest_configuration(
            "mean_reversion_strategy", 
            {"type": "mean_reversion", "period": 50},
            np.random.normal(0.0006, 0.018, n_obs)
        )
    ])
    
    # Run overfitting analysis
    detector = OverfittingDetector(n_bootstrap_trials=500)  # Reduced for example
    results = detector.comprehensive_overfitting_analysis(configs)
    
    return results

if __name__ == "__main__":
    # Run example analysis
    example_results = simulate_overfitting_example()
    
    print("Overfitting Detection Analysis Results:")
    print("=" * 50)
    
    summary = example_results["overfitting_summary"]
    print(f"Overall Risk: {summary['overall_overfitting_risk']}")
    print(f"Significant Strategies (Deflated): {len(summary['significant_strategies_deflated'])}")
    print(f"Overfitted Strategies: {len(summary['overfitted_strategies'])}")
    print(f"Robust Strategies: {len(summary['robust_strategies'])}")
    
    if example_results["pbo_analysis"]:
        pbo = example_results["pbo_analysis"]
        print(f"PBO Probability: {pbo.pbo_probability:.3f}")
        print(f"Performance Haircut: {pbo.performance_haircut:.4f}")
        
    print("\nRecommendations:")
    for rec in example_results["recommendations"]:
        print(f"  • {rec}")