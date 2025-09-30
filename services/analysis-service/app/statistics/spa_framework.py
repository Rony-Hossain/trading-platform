"""
White's Reality Check & Superior Predictive Ability (SPA) Test Framework

Implements rigorous statistical significance testing for trading strategies to prevent
false discoveries and data mining bias. Based on White (2000) Reality Check and
Hansen (2005) Superior Predictive Ability test.

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
from scipy.stats import norm
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class TestType(Enum):
    """Statistical test types"""
    REALITY_CHECK = "reality_check"  # White's Reality Check
    SPA_CONSISTENT = "spa_consistent"  # SPA with consistent p-values
    SPA_LOWER = "spa_lower"  # SPA lower test
    SPA_UPPER = "spa_upper"  # SPA upper test

class BootstrapMethod(Enum):
    """Bootstrap resampling methods"""
    STATIONARY = "stationary"  # Standard stationary bootstrap
    CIRCULAR = "circular"  # Circular block bootstrap
    MOVING_BLOCK = "moving_block"  # Moving block bootstrap
    WILD = "wild"  # Wild bootstrap

@dataclass
class PerformanceMetrics:
    """Strategy performance metrics for testing"""
    strategy_name: str
    returns: np.ndarray
    benchmark_returns: Optional[np.ndarray] = None
    excess_returns: Optional[np.ndarray] = None
    sharpe_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.excess_returns is None and self.benchmark_returns is not None:
            self.excess_returns = self.returns - self.benchmark_returns
        
        if self.sharpe_ratio is None:
            self.sharpe_ratio = self._calculate_sharpe_ratio()
        
        if self.information_ratio is None and self.excess_returns is not None:
            self.information_ratio = self._calculate_information_ratio()
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(self.returns) == 0:
            return 0.0
        
        mean_return = np.mean(self.returns) * 252  # Annualize
        volatility = np.std(self.returns, ddof=1) * np.sqrt(252)
        
        return mean_return / volatility if volatility > 0 else 0.0
    
    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio vs benchmark"""
        if self.excess_returns is None or len(self.excess_returns) == 0:
            return 0.0
        
        mean_excess = np.mean(self.excess_returns) * 252
        tracking_error = np.std(self.excess_returns, ddof=1) * np.sqrt(252)
        
        return mean_excess / tracking_error if tracking_error > 0 else 0.0

@dataclass 
class SPATestResult:
    """Results from SPA statistical test"""
    test_type: TestType
    test_statistic: float
    p_value: float
    critical_value_95: float
    critical_value_99: float
    is_significant_95: bool
    is_significant_99: bool
    
    # Bootstrap details
    bootstrap_iterations: int
    bootstrap_distribution: np.ndarray
    
    # Multiple testing correction
    bonferroni_p_value: Optional[float] = None
    fdr_p_value: Optional[float] = None
    
    # Test metadata
    strategy_name: str = ""
    benchmark_name: str = ""
    sample_size: int = 0
    test_period_start: Optional[date] = None
    test_period_end: Optional[date] = None
    
    # Diagnostic information
    effective_sample_size: Optional[int] = None
    autocorrelation_adjustment: bool = False
    heteroskedasticity_robust: bool = False

@dataclass
class MultipleTestingResults:
    """Results from multiple testing correction"""
    strategy_names: List[str]
    raw_p_values: np.ndarray
    bonferroni_p_values: np.ndarray
    fdr_p_values: np.ndarray
    significant_strategies_raw: List[str]
    significant_strategies_bonferroni: List[str] 
    significant_strategies_fdr: List[str]
    family_wise_error_rate: float
    false_discovery_rate: float

class SPATestFramework:
    """
    Superior Predictive Ability Test Framework
    
    Implements White's Reality Check and Hansen's SPA test for rigorous
    statistical validation of trading strategy performance.
    """
    
    def __init__(self,
                 bootstrap_iterations: int = 10000,
                 block_length: Optional[int] = None,
                 bootstrap_method: BootstrapMethod = BootstrapMethod.STATIONARY,
                 significance_levels: List[float] = [0.05, 0.01],
                 n_jobs: int = -1):
        
        self.bootstrap_iterations = bootstrap_iterations
        self.block_length = block_length
        self.bootstrap_method = bootstrap_method
        self.significance_levels = significance_levels
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
        # Cache for bootstrap distributions
        self._bootstrap_cache: Dict[str, np.ndarray] = {}
        
    def reality_check_test(self,
                          strategy_metrics: List[PerformanceMetrics],
                          benchmark_metrics: PerformanceMetrics,
                          test_statistic: str = "sharpe_ratio") -> SPATestResult:
        """
        White's Reality Check Test
        
        Tests the null hypothesis that the best strategy does not outperform
        the benchmark, accounting for data mining bias.
        """
        logger.info(f"Running Reality Check test on {len(strategy_metrics)} strategies")
        
        # Calculate performance statistics
        strategy_stats = self._calculate_test_statistics(
            strategy_metrics, benchmark_metrics, test_statistic
        )
        
        # Get the maximum test statistic (best strategy)
        max_stat = np.max(strategy_stats)
        best_strategy_idx = np.argmax(strategy_stats)
        
        # Bootstrap distribution under null hypothesis
        bootstrap_dist = self._bootstrap_reality_check(
            strategy_metrics, benchmark_metrics, test_statistic
        )
        
        # Calculate p-value
        p_value = np.mean(bootstrap_dist >= max_stat)
        
        # Critical values
        critical_95 = np.percentile(bootstrap_dist, 95)
        critical_99 = np.percentile(bootstrap_dist, 99)
        
        result = SPATestResult(
            test_type=TestType.REALITY_CHECK,
            test_statistic=max_stat,
            p_value=p_value,
            critical_value_95=critical_95,
            critical_value_99=critical_99,
            is_significant_95=p_value < 0.05,
            is_significant_99=p_value < 0.01,
            bootstrap_iterations=self.bootstrap_iterations,
            bootstrap_distribution=bootstrap_dist,
            strategy_name=strategy_metrics[best_strategy_idx].strategy_name,
            benchmark_name=benchmark_metrics.strategy_name,
            sample_size=len(strategy_metrics[0].returns)
        )
        
        logger.info(f"Reality Check test completed. Best strategy: {result.strategy_name}, "
                   f"p-value: {result.p_value:.4f}")
        
        return result
    
    def spa_test(self,
                strategy_metrics: List[PerformanceMetrics],
                benchmark_metrics: PerformanceMetrics,
                test_statistic: str = "sharpe_ratio",
                test_type: TestType = TestType.SPA_CONSISTENT) -> SPATestResult:
        """
        Superior Predictive Ability (SPA) Test
        
        More powerful than Reality Check, with better finite sample properties
        and consistent p-values under general conditions.
        """
        logger.info(f"Running SPA test ({test_type.value}) on {len(strategy_metrics)} strategies")
        
        # Calculate relative performance statistics
        relative_stats = self._calculate_relative_performance(
            strategy_metrics, benchmark_metrics, test_statistic
        )
        
        # Calculate SPA test statistic
        if test_type == TestType.SPA_CONSISTENT:
            test_stat = self._spa_consistent_statistic(relative_stats)
        elif test_type == TestType.SPA_LOWER:
            test_stat = self._spa_lower_statistic(relative_stats)
        elif test_type == TestType.SPA_UPPER:
            test_stat = self._spa_upper_statistic(relative_stats)
        else:
            raise ValueError(f"Unknown SPA test type: {test_type}")
        
        # Bootstrap distribution
        bootstrap_dist = self._bootstrap_spa_test(
            strategy_metrics, benchmark_metrics, test_statistic, test_type
        )
        
        # Calculate p-value
        if test_type == TestType.SPA_LOWER:
            p_value = np.mean(bootstrap_dist <= test_stat)
        else:
            p_value = np.mean(bootstrap_dist >= test_stat)
        
        # Critical values
        if test_type == TestType.SPA_LOWER:
            critical_95 = np.percentile(bootstrap_dist, 5)
            critical_99 = np.percentile(bootstrap_dist, 1)
        else:
            critical_95 = np.percentile(bootstrap_dist, 95)
            critical_99 = np.percentile(bootstrap_dist, 99)
        
        result = SPATestResult(
            test_type=test_type,
            test_statistic=test_stat,
            p_value=p_value,
            critical_value_95=critical_95,
            critical_value_99=critical_99,
            is_significant_95=p_value < 0.05,
            is_significant_99=p_value < 0.01,
            bootstrap_iterations=self.bootstrap_iterations,
            bootstrap_distribution=bootstrap_dist,
            benchmark_name=benchmark_metrics.strategy_name,
            sample_size=len(strategy_metrics[0].returns)
        )
        
        logger.info(f"SPA test completed. Test statistic: {result.test_statistic:.4f}, "
                   f"p-value: {result.p_value:.4f}")
        
        return result
    
    def multiple_testing_correction(self,
                                  strategy_results: List[SPATestResult],
                                  method: str = "both") -> MultipleTestingResults:
        """
        Apply multiple testing corrections to control family-wise error rate
        
        Args:
            strategy_results: Individual test results for each strategy
            method: "bonferroni", "fdr", or "both"
        """
        logger.info(f"Applying multiple testing correction to {len(strategy_results)} strategies")
        
        strategy_names = [r.strategy_name for r in strategy_results]
        raw_p_values = np.array([r.p_value for r in strategy_results])
        
        # Bonferroni correction
        bonferroni_p_values = np.minimum(raw_p_values * len(raw_p_values), 1.0)
        
        # False Discovery Rate (Benjamini-Hochberg)
        fdr_p_values = self._benjamini_hochberg_correction(raw_p_values)
        
        # Determine significant strategies
        significant_raw = [name for name, p in zip(strategy_names, raw_p_values) if p < 0.05]
        significant_bonferroni = [name for name, p in zip(strategy_names, bonferroni_p_values) if p < 0.05]
        significant_fdr = [name for name, p in zip(strategy_names, fdr_p_values) if p < 0.05]
        
        # Calculate error rates
        family_wise_error_rate = np.min(bonferroni_p_values)
        false_discovery_rate = len(significant_fdr) / max(len(significant_raw), 1)
        
        results = MultipleTestingResults(
            strategy_names=strategy_names,
            raw_p_values=raw_p_values,
            bonferroni_p_values=bonferroni_p_values,
            fdr_p_values=fdr_p_values,
            significant_strategies_raw=significant_raw,
            significant_strategies_bonferroni=significant_bonferroni,
            significant_strategies_fdr=significant_fdr,
            family_wise_error_rate=family_wise_error_rate,
            false_discovery_rate=false_discovery_rate
        )
        
        logger.info(f"Multiple testing correction completed. "
                   f"Significant strategies - Raw: {len(significant_raw)}, "
                   f"Bonferroni: {len(significant_bonferroni)}, "
                   f"FDR: {len(significant_fdr)}")
        
        return results
    
    def comprehensive_strategy_testing(self,
                                     strategy_metrics: List[PerformanceMetrics],
                                     benchmark_metrics: PerformanceMetrics,
                                     include_individual_tests: bool = True) -> Dict[str, Any]:
        """
        Comprehensive statistical testing framework for strategy validation
        
        Combines Reality Check, SPA tests, and multiple testing corrections
        for rigorous strategy evaluation.
        """
        logger.info(f"Running comprehensive testing on {len(strategy_metrics)} strategies")
        
        results = {
            "test_summary": {
                "n_strategies": len(strategy_metrics),
                "benchmark": benchmark_metrics.strategy_name,
                "sample_size": len(strategy_metrics[0].returns),
                "test_date": datetime.utcnow().isoformat()
            }
        }
        
        # 1. White's Reality Check
        try:
            reality_check = self.reality_check_test(strategy_metrics, benchmark_metrics)
            results["reality_check"] = reality_check
        except Exception as e:
            logger.error(f"Reality Check test failed: {e}")
            results["reality_check"] = None
        
        # 2. SPA Tests (multiple variants)
        spa_results = {}
        for test_type in [TestType.SPA_CONSISTENT, TestType.SPA_LOWER, TestType.SPA_UPPER]:
            try:
                spa_result = self.spa_test(strategy_metrics, benchmark_metrics, test_type=test_type)
                spa_results[test_type.value] = spa_result
            except Exception as e:
                logger.error(f"SPA test {test_type.value} failed: {e}")
                spa_results[test_type.value] = None
        
        results["spa_tests"] = spa_results
        
        # 3. Individual strategy testing (if requested)
        if include_individual_tests:
            individual_results = []
            for strategy in strategy_metrics:
                try:
                    individual_result = self.spa_test(
                        [strategy], benchmark_metrics, test_type=TestType.SPA_CONSISTENT
                    )
                    individual_result.strategy_name = strategy.strategy_name
                    individual_results.append(individual_result)
                except Exception as e:
                    logger.error(f"Individual test for {strategy.strategy_name} failed: {e}")
            
            # 4. Multiple testing correction
            if individual_results:
                try:
                    multiple_testing = self.multiple_testing_correction(individual_results)
                    results["multiple_testing"] = multiple_testing
                    results["individual_tests"] = individual_results
                except Exception as e:
                    logger.error(f"Multiple testing correction failed: {e}")
                    results["multiple_testing"] = None
                    results["individual_tests"] = individual_results
        
        # 5. Summary statistics
        results["summary"] = self._generate_test_summary(results)
        
        logger.info("Comprehensive strategy testing completed")
        return results
    
    def _calculate_test_statistics(self,
                                 strategy_metrics: List[PerformanceMetrics],
                                 benchmark_metrics: PerformanceMetrics,
                                 statistic: str) -> np.ndarray:
        """Calculate test statistics for strategies vs benchmark"""
        
        if statistic == "sharpe_ratio":
            stats = np.array([m.sharpe_ratio for m in strategy_metrics])
            benchmark_stat = benchmark_metrics.sharpe_ratio
        elif statistic == "information_ratio":
            stats = np.array([m.information_ratio for m in strategy_metrics])
            benchmark_stat = benchmark_metrics.information_ratio
        elif statistic == "mean_return":
            stats = np.array([np.mean(m.returns) for m in strategy_metrics])
            benchmark_stat = np.mean(benchmark_metrics.returns)
        else:
            raise ValueError(f"Unknown test statistic: {statistic}")
        
        # Return difference from benchmark
        return stats - benchmark_stat
    
    def _calculate_relative_performance(self,
                                      strategy_metrics: List[PerformanceMetrics],
                                      benchmark_metrics: PerformanceMetrics,
                                      statistic: str) -> np.ndarray:
        """Calculate relative performance matrix for SPA test"""
        
        n_strategies = len(strategy_metrics)
        sample_size = len(strategy_metrics[0].returns)
        
        # Create relative performance matrix (T x k)
        relative_performance = np.zeros((sample_size, n_strategies))
        
        for i, strategy in enumerate(strategy_metrics):
            if statistic == "excess_return":
                # Use excess returns directly
                if strategy.excess_returns is not None:
                    relative_performance[:, i] = strategy.excess_returns
                else:
                    relative_performance[:, i] = strategy.returns - benchmark_metrics.returns
            elif statistic == "sharpe_ratio":
                # For SPA test, use excess returns rather than rolling Sharpe
                # Rolling Sharpe differences are complex and can cause numerical issues
                relative_performance[:, i] = strategy.returns - benchmark_metrics.returns
            else:
                # Default to excess returns
                relative_performance[:, i] = strategy.returns - benchmark_metrics.returns
        
        return relative_performance
    
    def _spa_consistent_statistic(self, relative_performance: np.ndarray) -> float:
        """Calculate SPA consistent test statistic"""
        if relative_performance.ndim == 1:
            # Single strategy case
            sample_mean = np.mean(relative_performance)
            sample_var = np.var(relative_performance, ddof=1)
            sample_size = len(relative_performance)
            
            if sample_var <= 0 or sample_mean <= 0:
                return 0.0
            
            return np.sqrt(sample_size) * sample_mean / np.sqrt(sample_var)
        
        sample_size, n_strategies = relative_performance.shape
        
        # Sample means
        sample_means = np.mean(relative_performance, axis=0)
        
        # Sample variance-covariance matrix
        if n_strategies == 1:
            sample_var = np.var(relative_performance[:, 0], ddof=1)
        else:
            sample_cov = np.cov(relative_performance.T)
            
        # Consistent test statistic
        max_mean = np.max(sample_means)
        
        if max_mean <= 0:
            return 0.0
        
        # Find strategy with maximum mean
        max_idx = np.argmax(sample_means)
        
        if n_strategies == 1:
            sample_var_max = sample_var
        else:
            sample_var_max = sample_cov[max_idx, max_idx] if sample_cov.ndim > 0 else sample_cov
        
        if sample_var_max <= 0:
            return 0.0
        
        test_stat = np.sqrt(sample_size) * max_mean / np.sqrt(sample_var_max)
        return test_stat
    
    def _spa_lower_statistic(self, relative_performance: np.ndarray) -> float:
        """Calculate SPA lower test statistic"""
        if relative_performance.ndim == 1:
            # Single strategy case
            sample_mean = np.mean(relative_performance)
            sample_std = np.std(relative_performance, ddof=1)
            sample_size = len(relative_performance)
            
            if sample_std == 0:
                return 0.0
            
            return sample_mean / (sample_std / np.sqrt(sample_size))
        
        sample_size, n_strategies = relative_performance.shape
        sample_means = np.mean(relative_performance, axis=0)
        
        # Standard errors
        sample_std = np.std(relative_performance, axis=0, ddof=1)
        standard_errors = sample_std / np.sqrt(sample_size)
        
        # t-statistics
        t_stats = np.where(standard_errors > 0, sample_means / standard_errors, 0)
        
        return np.max(t_stats)
    
    def _spa_upper_statistic(self, relative_performance: np.ndarray) -> float:
        """Calculate SPA upper test statistic"""
        if relative_performance.ndim == 1:
            # Single strategy case
            sample_mean = np.mean(relative_performance)
            sample_std = np.std(relative_performance, ddof=1)
            sample_size = len(relative_performance)
            
            if sample_mean <= 0 or sample_std == 0:
                return 0.0
            
            return sample_mean / (sample_std / np.sqrt(sample_size))
        
        sample_size, n_strategies = relative_performance.shape
        sample_means = np.mean(relative_performance, axis=0)
        
        # Only consider strategies with positive sample means
        positive_means = sample_means[sample_means > 0]
        
        if len(positive_means) == 0:
            return 0.0
        
        # Standard errors for positive means only
        positive_indices = sample_means > 0
        sample_std = np.std(relative_performance[:, positive_indices], axis=0, ddof=1)
        standard_errors = sample_std / np.sqrt(sample_size)
        
        # t-statistics for positive means
        t_stats = np.where(standard_errors > 0, positive_means / standard_errors, 0)
        
        return np.max(t_stats)
    
    def _bootstrap_reality_check(self,
                                strategy_metrics: List[PerformanceMetrics],
                                benchmark_metrics: PerformanceMetrics,
                                test_statistic: str) -> np.ndarray:
        """Bootstrap distribution for Reality Check test"""
        
        logger.debug(f"Generating bootstrap distribution with {self.bootstrap_iterations} iterations")
        
        # Prepare data for bootstrap
        n_strategies = len(strategy_metrics)
        sample_size = len(strategy_metrics[0].returns)
        
        # Create returns matrix
        strategy_returns = np.array([m.returns for m in strategy_metrics]).T
        benchmark_returns = benchmark_metrics.returns
        
        # Bootstrap iterations
        bootstrap_stats = np.zeros(self.bootstrap_iterations)
        
        for i in range(self.bootstrap_iterations):
            # Generate bootstrap sample
            bootstrap_indices = self._generate_bootstrap_indices(sample_size)
            
            # Bootstrap strategy and benchmark returns
            bootstrap_strategy_returns = strategy_returns[bootstrap_indices]
            bootstrap_benchmark_returns = benchmark_returns[bootstrap_indices]
            
            # Calculate bootstrap test statistics
            bootstrap_test_stats = np.zeros(n_strategies)
            
            for j in range(n_strategies):
                if test_statistic == "sharpe_ratio":
                    strategy_sharpe = self._calculate_sharpe(bootstrap_strategy_returns[:, j])
                    benchmark_sharpe = self._calculate_sharpe(bootstrap_benchmark_returns)
                    bootstrap_test_stats[j] = strategy_sharpe - benchmark_sharpe
                elif test_statistic == "mean_return":
                    strategy_mean = np.mean(bootstrap_strategy_returns[:, j])
                    benchmark_mean = np.mean(bootstrap_benchmark_returns)
                    bootstrap_test_stats[j] = strategy_mean - benchmark_mean
            
            # Store maximum test statistic
            bootstrap_stats[i] = np.max(bootstrap_test_stats)
        
        return bootstrap_stats
    
    def _bootstrap_spa_test(self,
                           strategy_metrics: List[PerformanceMetrics],
                           benchmark_metrics: PerformanceMetrics,
                           test_statistic: str,
                           test_type: TestType) -> np.ndarray:
        """Bootstrap distribution for SPA test"""
        
        logger.debug(f"Generating SPA bootstrap distribution with {self.bootstrap_iterations} iterations")
        
        sample_size = len(strategy_metrics[0].returns)
        
        # Get original relative performance
        relative_performance = self._calculate_relative_performance(
            strategy_metrics, benchmark_metrics, test_statistic
        )
        
        # Center the data (impose null hypothesis)
        sample_means = np.mean(relative_performance, axis=0)
        centered_performance = relative_performance - sample_means
        
        bootstrap_stats = np.zeros(self.bootstrap_iterations)
        
        for i in range(self.bootstrap_iterations):
            # Generate bootstrap indices
            bootstrap_indices = self._generate_bootstrap_indices(sample_size)
            
            # Bootstrap centered data
            bootstrap_data = centered_performance[bootstrap_indices]
            
            # Calculate bootstrap test statistic
            if test_type == TestType.SPA_CONSISTENT:
                bootstrap_stats[i] = self._spa_consistent_statistic(bootstrap_data)
            elif test_type == TestType.SPA_LOWER:
                bootstrap_stats[i] = self._spa_lower_statistic(bootstrap_data)
            elif test_type == TestType.SPA_UPPER:
                bootstrap_stats[i] = self._spa_upper_statistic(bootstrap_data)
        
        return bootstrap_stats
    
    def _generate_bootstrap_indices(self, sample_size: int) -> np.ndarray:
        """Generate bootstrap indices based on chosen method"""
        
        if self.bootstrap_method == BootstrapMethod.STATIONARY:
            return self._stationary_bootstrap_indices(sample_size)
        elif self.bootstrap_method == BootstrapMethod.CIRCULAR:
            return self._circular_bootstrap_indices(sample_size)
        elif self.bootstrap_method == BootstrapMethod.MOVING_BLOCK:
            return self._moving_block_bootstrap_indices(sample_size)
        else:
            # Default to simple random resampling
            return np.random.choice(sample_size, size=sample_size, replace=True)
    
    def _stationary_bootstrap_indices(self, sample_size: int) -> np.ndarray:
        """Stationary bootstrap with random block lengths"""
        if self.block_length is None:
            # Optimal block length for time series
            self.block_length = int(np.ceil(sample_size ** (1/3)))
        
        indices = []
        while len(indices) < sample_size:
            # Random starting point
            start = np.random.randint(0, sample_size)
            
            # Geometric block length
            block_len = np.random.geometric(1.0 / self.block_length)
            block_len = min(block_len, sample_size - len(indices))
            
            # Add block indices (with wraparound)
            for i in range(block_len):
                indices.append((start + i) % sample_size)
        
        return np.array(indices[:sample_size])
    
    def _circular_bootstrap_indices(self, sample_size: int) -> np.ndarray:
        """Circular block bootstrap"""
        if self.block_length is None:
            self.block_length = int(np.ceil(sample_size ** (1/3)))
        
        indices = []
        while len(indices) < sample_size:
            start = np.random.randint(0, sample_size)
            block_len = min(self.block_length, sample_size - len(indices))
            
            for i in range(block_len):
                indices.append((start + i) % sample_size)
        
        return np.array(indices[:sample_size])
    
    def _moving_block_bootstrap_indices(self, sample_size: int) -> np.ndarray:
        """Moving block bootstrap"""
        if self.block_length is None:
            self.block_length = int(np.ceil(sample_size ** (1/3)))
        
        n_blocks = int(np.ceil(sample_size / self.block_length))
        indices = []
        
        for _ in range(n_blocks):
            start = np.random.randint(0, sample_size - self.block_length + 1)
            indices.extend(range(start, start + self.block_length))
        
        return np.array(indices[:sample_size])
    
    def _benjamini_hochberg_correction(self, p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Benjamini-Hochberg false discovery rate correction"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # Calculate adjusted p-values
        adjusted_p_values = np.zeros(n)
        
        for i in range(n-1, -1, -1):
            if i == n-1:
                adjusted_p_values[sorted_indices[i]] = sorted_p_values[i]
            else:
                adjusted_p_values[sorted_indices[i]] = min(
                    sorted_p_values[i] * n / (i + 1),
                    adjusted_p_values[sorted_indices[i + 1]]
                )
        
        return np.minimum(adjusted_p_values, 1.0)
    
    def _rolling_sharpe(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling Sharpe ratio"""
        if len(returns) < window:
            return np.array([self._calculate_sharpe(returns)])
        
        rolling_sharpe = []
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            rolling_sharpe.append(self._calculate_sharpe(window_returns))
        
        # Pad with first value to match original length
        padding = [rolling_sharpe[0]] * (window - 1)
        return np.array(padding + rolling_sharpe)
    
    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)
        
        if std_excess == 0:
            return 0.0
        
        return mean_excess / std_excess * np.sqrt(252)  # Annualized
    
    def _generate_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all test results"""
        summary = {
            "reality_check_significant": False,
            "spa_tests_significant": {},
            "n_significant_strategies_raw": 0,
            "n_significant_strategies_corrected": 0,
            "strongest_evidence": None,
            "recommendations": []
        }
        
        # Reality Check results
        if results.get("reality_check"):
            rc = results["reality_check"]
            summary["reality_check_significant"] = rc.is_significant_95
            if rc.is_significant_95:
                summary["strongest_evidence"] = "reality_check"
        
        # SPA test results
        for test_name, spa_result in results.get("spa_tests", {}).items():
            if spa_result:
                summary["spa_tests_significant"][test_name] = spa_result.is_significant_95
                if spa_result.is_significant_95 and not summary["strongest_evidence"]:
                    summary["strongest_evidence"] = test_name
        
        # Multiple testing results
        if results.get("multiple_testing"):
            mt = results["multiple_testing"]
            summary["n_significant_strategies_raw"] = len(mt.significant_strategies_raw)
            summary["n_significant_strategies_corrected"] = len(mt.significant_strategies_fdr)
        
        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(results)
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check if any tests show significance
        any_significant = False
        
        if results.get("reality_check") and results["reality_check"].is_significant_95:
            any_significant = True
            recommendations.append("Reality Check test shows significant outperformance")
        
        for test_name, spa_result in results.get("spa_tests", {}).items():
            if spa_result and spa_result.is_significant_95:
                any_significant = True
                recommendations.append(f"SPA {test_name} test shows significant outperformance")
        
        if results.get("multiple_testing"):
            mt = results["multiple_testing"]
            if len(mt.significant_strategies_fdr) > 0:
                recommendations.append(f"{len(mt.significant_strategies_fdr)} strategies remain significant after FDR correction")
            else:
                recommendations.append("No strategies remain significant after multiple testing correction")
        
        if not any_significant:
            recommendations.extend([
                "No statistically significant outperformance detected",
                "Consider larger sample size or different strategies",
                "Avoid data mining bias in strategy selection"
            ])
        
        return recommendations

# Utility functions for integration

def create_performance_metrics_from_returns(strategy_name: str,
                                          returns: np.ndarray,
                                          benchmark_returns: Optional[np.ndarray] = None) -> PerformanceMetrics:
    """Create PerformanceMetrics object from return series"""
    return PerformanceMetrics(
        strategy_name=strategy_name,
        returns=returns,
        benchmark_returns=benchmark_returns
    )

def load_strategy_returns_from_dataframe(df: pd.DataFrame,
                                       strategy_columns: List[str],
                                       benchmark_column: str = "benchmark") -> Tuple[List[PerformanceMetrics], PerformanceMetrics]:
    """Load strategy and benchmark returns from DataFrame"""
    
    # Create strategy metrics
    strategy_metrics = []
    for col in strategy_columns:
        if col in df.columns:
            returns = df[col].dropna().values
            metrics = create_performance_metrics_from_returns(col, returns)
            strategy_metrics.append(metrics)
    
    # Create benchmark metrics
    if benchmark_column in df.columns:
        benchmark_returns = df[benchmark_column].dropna().values
        benchmark_metrics = create_performance_metrics_from_returns(benchmark_column, benchmark_returns)
    else:
        # Use zero returns as benchmark
        benchmark_returns = np.zeros(len(df))
        benchmark_metrics = create_performance_metrics_from_returns("zero_benchmark", benchmark_returns)
    
    return strategy_metrics, benchmark_metrics