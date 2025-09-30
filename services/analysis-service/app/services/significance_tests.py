"""
Statistical Significance Testing Framework

Implements sophisticated statistical tests for trading strategy evaluation:
- White's Reality Check / Superior Predictive Ability (SPA) test
- Deflated Sharpe Ratio (DSR) for multiple testing correction
- Probability of Backtest Overfitting (PBO) estimation
- Cluster-robust statistics for correlated returns

References:
- White (2000): A Reality Check for Data Snooping
- Hansen (2005): A Test for Superior Predictive Ability  
- Bailey & Lopez de Prado (2014): The Deflated Sharpe Ratio
- Bailey et al. (2017): The Probability of Backtest Overfitting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import norm, t as t_dist
from scipy.optimize import minimize_scalar
import warnings
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignificanceTestResult:
    """Result container for significance tests."""
    test_name: str
    test_statistic: float
    p_value: float
    critical_value: Optional[float]
    is_significant: bool
    confidence_level: float
    additional_stats: Dict[str, float]
    interpretation: str
    timestamp: datetime


@dataclass
class SPATestResult:
    """Superior Predictive Ability test result."""
    spa_statistic: float
    spa_p_value: float
    rc_statistic: float  # Reality Check statistic
    rc_p_value: float
    bootstrap_iterations: int
    benchmark_performance: float
    best_strategy_performance: float
    num_strategies: int
    is_significant: bool
    interpretation: str


@dataclass
class DeflatedSharpeResult:
    """Deflated Sharpe Ratio test result."""
    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    trials: int
    length: int
    skewness: float
    kurtosis: float
    is_significant: bool
    interpretation: str


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting result."""
    pbo_estimate: float
    phi_estimate: float
    logits: np.ndarray
    omega_matrix: np.ndarray
    is_overfitted: bool
    max_pbo_threshold: float
    interpretation: str


class WhiteRealityCheck:
    """
    Implementation of White's Reality Check and Hansen's SPA test.
    
    Tests whether the best performing strategy is significantly better 
    than a benchmark after accounting for data snooping bias.
    """
    
    def __init__(self, benchmark_returns: np.ndarray, strategy_returns: np.ndarray, 
                 bootstrap_iterations: int = 10000, block_length: Optional[int] = None):
        """
        Initialize the Reality Check test.
        
        Args:
            benchmark_returns: Returns of benchmark strategy
            strategy_returns: Returns matrix (strategies x time)
            bootstrap_iterations: Number of bootstrap iterations
            block_length: Block length for stationary bootstrap (auto if None)
        """
        self.benchmark_returns = np.array(benchmark_returns)
        self.strategy_returns = np.array(strategy_returns)
        if self.strategy_returns.ndim == 1:
            self.strategy_returns = self.strategy_returns.reshape(1, -1)
        
        self.n_strategies, self.n_periods = self.strategy_returns.shape
        self.bootstrap_iterations = bootstrap_iterations
        self.block_length = block_length or self._optimal_block_length()
        
        # Calculate performance differences
        self.performance_diffs = self._calculate_performance_differences()
        
    def _optimal_block_length(self) -> int:
        """Calculate optimal block length using Politis & White (2004) method."""
        # Simplified estimation - can be enhanced with more sophisticated methods
        return max(1, int(np.ceil(self.n_periods ** (1/3))))
    
    def _calculate_performance_differences(self) -> np.ndarray:
        """Calculate performance differences between strategies and benchmark."""
        benchmark_perf = np.mean(self.benchmark_returns)
        strategy_perfs = np.mean(self.strategy_returns, axis=1)
        return strategy_perfs - benchmark_perf
    
    def _stationary_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """
        Generate stationary bootstrap sample.
        
        Args:
            data: Time series data
            
        Returns:
            Bootstrap sample of same length
        """
        n = len(data)
        bootstrap_sample = np.zeros(n)
        
        i = 0
        while i < n:
            # Random starting point
            start = np.random.randint(0, n)
            
            # Geometric block length
            block_len = np.random.geometric(1 / self.block_length)
            block_len = min(block_len, n - i)
            
            # Copy block (with wrap-around)
            for j in range(block_len):
                bootstrap_sample[i + j] = data[(start + j) % n]
            
            i += block_len
            
        return bootstrap_sample
    
    def _bootstrap_max_statistic(self) -> np.ndarray:
        """Generate bootstrap distribution of maximum test statistic."""
        max_statistics = np.zeros(self.bootstrap_iterations)
        
        for b in range(self.bootstrap_iterations):
            # Bootstrap benchmark returns
            bootstrap_benchmark = self._stationary_bootstrap(self.benchmark_returns)
            
            # Bootstrap strategy returns
            bootstrap_strategies = np.zeros_like(self.strategy_returns)
            for s in range(self.n_strategies):
                bootstrap_strategies[s] = self._stationary_bootstrap(self.strategy_returns[s])
            
            # Calculate bootstrap performance differences
            benchmark_perf = np.mean(bootstrap_benchmark)
            strategy_perfs = np.mean(bootstrap_strategies, axis=1)
            bootstrap_diffs = strategy_perfs - benchmark_perf
            
            # Center the bootstrap differences
            centered_diffs = bootstrap_diffs - self.performance_diffs
            
            # Calculate test statistics (t-statistics)
            test_stats = np.zeros(self.n_strategies)
            for s in range(self.n_strategies):
                strategy_benchmark_diff = bootstrap_strategies[s] - bootstrap_benchmark
                if np.std(strategy_benchmark_diff) > 0:
                    test_stats[s] = (centered_diffs[s] * np.sqrt(self.n_periods) / 
                                   np.std(strategy_benchmark_diff))
            
            max_statistics[b] = np.max(test_stats)
            
        return max_statistics
    
    def run_test(self, confidence_level: float = 0.95) -> SPATestResult:
        """
        Run the SPA test.
        
        Args:
            confidence_level: Confidence level for the test
            
        Returns:
            SPATestResult object with test results
        """
        # Calculate original test statistics
        test_statistics = np.zeros(self.n_strategies)
        
        for s in range(self.n_strategies):
            strategy_benchmark_diff = self.strategy_returns[s] - self.benchmark_returns
            if np.std(strategy_benchmark_diff) > 0:
                test_statistics[s] = (self.performance_diffs[s] * np.sqrt(self.n_periods) / 
                                    np.std(strategy_benchmark_diff))
        
        # Reality Check statistic (original White test)
        rc_statistic = np.max(test_statistics)
        
        # SPA statistic (Hansen improvement)
        positive_stats = test_statistics[test_statistics > 0]
        spa_statistic = np.max(positive_stats) if len(positive_stats) > 0 else 0
        
        # Generate bootstrap distribution
        bootstrap_max_stats = self._bootstrap_max_statistic()
        
        # Calculate p-values
        rc_p_value = np.mean(bootstrap_max_stats >= rc_statistic)
        spa_p_value = np.mean(bootstrap_max_stats >= spa_statistic)
        
        # Determine significance
        alpha = 1 - confidence_level
        is_significant = spa_p_value < alpha
        
        # Generate interpretation
        interpretation = self._generate_interpretation(spa_p_value, alpha, 
                                                     self.n_strategies, is_significant)
        
        return SPATestResult(
            spa_statistic=spa_statistic,
            spa_p_value=spa_p_value,
            rc_statistic=rc_statistic,
            rc_p_value=rc_p_value,
            bootstrap_iterations=self.bootstrap_iterations,
            benchmark_performance=np.mean(self.benchmark_returns),
            best_strategy_performance=np.max(np.mean(self.strategy_returns, axis=1)),
            num_strategies=self.n_strategies,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    def _generate_interpretation(self, p_value: float, alpha: float, 
                               n_strategies: int, is_significant: bool) -> str:
        """Generate interpretation of test results."""
        if is_significant:
            return (f"SPA test (p={p_value:.4f}) indicates significant outperformance "
                   f"after correcting for data snooping across {n_strategies} strategies. "
                   f"The best strategy likely has genuine predictive ability.")
        else:
            return (f"SPA test (p={p_value:.4f}) suggests the best strategy's outperformance "
                   f"could be due to data snooping across {n_strategies} strategies. "
                   f"No evidence of genuine predictive ability.")


class DeflatedSharpe:
    """
    Implementation of the Deflated Sharpe Ratio test.
    
    Corrects Sharpe ratio for multiple testing and non-normal returns,
    providing a more conservative estimate of strategy performance.
    """
    
    @staticmethod
    def calculate_deflated_sharpe(returns: np.ndarray, trials: int, 
                                 length: Optional[int] = None) -> DeflatedSharpeResult:
        """
        Calculate the Deflated Sharpe Ratio.
        
        Args:
            returns: Strategy returns
            trials: Number of trials/strategies tested
            length: Sample length (uses actual length if None)
            
        Returns:
            DeflatedSharpeResult object
        """
        returns = np.array(returns)
        n = length or len(returns)
        
        # Calculate observed Sharpe ratio
        observed_sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
        
        # Calculate moments
        skewness = stats.skew(returns)
        excess_kurtosis = stats.kurtosis(returns)
        
        # Calculate deflated Sharpe ratio
        deflated_sharpe = DeflatedSharpe._deflate_sharpe_ratio(
            observed_sharpe, n, trials, skewness, excess_kurtosis
        )
        
        # Calculate p-value
        p_value = 1 - norm.cdf(deflated_sharpe)
        
        # Determine significance
        is_significant = p_value < 0.05
        
        # Generate interpretation
        interpretation = DeflatedSharpe._generate_interpretation(
            observed_sharpe, deflated_sharpe, p_value, trials, is_significant
        )
        
        return DeflatedSharpeResult(
            observed_sharpe=observed_sharpe,
            deflated_sharpe=deflated_sharpe,
            p_value=p_value,
            trials=trials,
            length=n,
            skewness=skewness,
            kurtosis=excess_kurtosis,
            is_significant=is_significant,
            interpretation=interpretation
        )
    
    @staticmethod
    def _deflate_sharpe_ratio(sharpe: float, n: int, trials: int, 
                             skewness: float, kurtosis: float) -> float:
        """
        Calculate the deflated Sharpe ratio.
        
        Args:
            sharpe: Observed Sharpe ratio
            n: Sample length
            trials: Number of trials
            skewness: Returns skewness
            kurtosis: Returns excess kurtosis
            
        Returns:
            Deflated Sharpe ratio
        """
        # Expected maximum Sharpe ratio under null
        gamma = 0.5772156649  # Euler-Mascheroni constant
        expected_max_sr = (1 - gamma) * norm.ppf(1 - 1/trials) + gamma * norm.ppf(1 - 1/(trials * np.e))
        
        # Variance of maximum Sharpe ratio
        var_max_sr = (1 - gamma) * norm.ppf(1 - 1/trials)**2 + gamma * norm.ppf(1 - 1/(trials * np.e))**2 - expected_max_sr**2
        var_max_sr += np.pi**2 / 6
        
        # Sharpe ratio adjustment for non-normality
        sr_adjustment = (1 + (skewness * sharpe)/6 + ((kurtosis - 3) * sharpe**2)/24)
        
        # Deflated Sharpe ratio
        deflated_sr = (sharpe - expected_max_sr / np.sqrt(n)) / np.sqrt(var_max_sr / n) * sr_adjustment
        
        return deflated_sr
    
    @staticmethod
    def _generate_interpretation(observed_sharpe: float, deflated_sharpe: float, 
                               p_value: float, trials: int, is_significant: bool) -> str:
        """Generate interpretation of deflated Sharpe ratio results."""
        if is_significant:
            return (f"Deflated Sharpe ratio {deflated_sharpe:.3f} (p={p_value:.4f}) "
                   f"indicates significant performance after correcting for {trials} trials. "
                   f"Original Sharpe ratio {observed_sharpe:.3f} is likely genuine.")
        else:
            return (f"Deflated Sharpe ratio {deflated_sharpe:.3f} (p={p_value:.4f}) "
                   f"suggests observed Sharpe ratio {observed_sharpe:.3f} may be inflated "
                   f"due to multiple testing across {trials} strategies.")


class PBOEstimator:
    """
    Probability of Backtest Overfitting (PBO) estimator.
    
    Estimates the probability that an observed backtest performance
    is due to overfitting rather than genuine predictive ability.
    """
    
    def __init__(self, returns_matrix: np.ndarray, threshold: float = 0.0):
        """
        Initialize PBO estimator.
        
        Args:
            returns_matrix: Matrix of strategy returns (strategies x time)
            threshold: Performance threshold (default 0 for positive returns)
        """
        self.returns_matrix = np.array(returns_matrix)
        if self.returns_matrix.ndim == 1:
            self.returns_matrix = self.returns_matrix.reshape(1, -1)
        
        self.n_strategies, self.n_periods = self.returns_matrix.shape
        self.threshold = threshold
        
    def estimate_pbo(self, n_splits: int = 16) -> PBOResult:
        """
        Estimate the Probability of Backtest Overfitting.
        
        Args:
            n_splits: Number of splits for combinatorial sampling
            
        Returns:
            PBOResult object
        """
        # Generate combinatorial splits
        splits = self._generate_combinatorial_splits(n_splits)
        
        # Calculate logits for each split
        logits = self._calculate_logits(splits)
        
        # Estimate omega matrix (correlation matrix of logits)
        omega_matrix = np.corrcoef(logits)
        
        # Calculate PBO estimate
        pbo_estimate = self._calculate_pbo(logits)
        
        # Calculate phi (expected logit under null)
        phi_estimate = np.mean(logits)
        
        # Determine if overfitted
        max_pbo_threshold = 0.2  # Common threshold
        is_overfitted = pbo_estimate > max_pbo_threshold
        
        # Generate interpretation
        interpretation = self._generate_interpretation(pbo_estimate, max_pbo_threshold, 
                                                     is_overfitted, self.n_strategies)
        
        return PBOResult(
            pbo_estimate=pbo_estimate,
            phi_estimate=phi_estimate,
            logits=logits,
            omega_matrix=omega_matrix,
            is_overfitted=is_overfitted,
            max_pbo_threshold=max_pbo_threshold,
            interpretation=interpretation
        )
    
    def _generate_combinatorial_splits(self, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate combinatorial splits of the data."""
        splits = []
        split_size = self.n_periods // 2
        
        for i in range(n_splits):
            # Random split
            indices = np.random.permutation(self.n_periods)
            train_idx = indices[:split_size]
            test_idx = indices[split_size:2*split_size]
            
            splits.append((train_idx, test_idx))
            
        return splits
    
    def _calculate_logits(self, splits: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Calculate logits for each split."""
        logits = np.zeros(len(splits))
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # In-sample performance
            is_performance = np.mean(self.returns_matrix[:, train_idx], axis=1)
            
            # Select best strategy based on in-sample performance
            best_strategy_idx = np.argmax(is_performance)
            
            # Out-of-sample performance of selected strategy
            oos_performance = np.mean(self.returns_matrix[best_strategy_idx, test_idx])
            
            # Calculate logit
            if oos_performance > self.threshold:
                wins = 1
            else:
                wins = 0
                
            # Logit calculation (with small epsilon to avoid log(0))
            epsilon = 1e-10
            p = (wins + epsilon) / (1 + 2 * epsilon)
            logits[i] = np.log(p / (1 - p))
            
        return logits
    
    def _calculate_pbo(self, logits: np.ndarray) -> float:
        """Calculate PBO estimate from logits."""
        # PBO is the probability that logit <= 0
        return np.mean(logits <= 0)
    
    def _generate_interpretation(self, pbo: float, threshold: float, 
                               is_overfitted: bool, n_strategies: int) -> str:
        """Generate interpretation of PBO results."""
        if is_overfitted:
            return (f"PBO estimate {pbo:.3f} exceeds threshold {threshold}, "
                   f"indicating high probability of overfitting across {n_strategies} strategies. "
                   f"Backtest results may not generalize to live trading.")
        else:
            return (f"PBO estimate {pbo:.3f} below threshold {threshold}, "
                   f"suggesting lower probability of overfitting. "
                   f"Backtest results more likely to be genuine.")


class ClusterRobustStatistics:
    """
    Cluster-robust statistical tests for correlated observations.
    
    Useful for event studies and other analyses where observations
    may be clustered in time or by other characteristics.
    """
    
    @staticmethod
    def cluster_robust_ttest(returns: np.ndarray, clusters: np.ndarray, 
                           null_mean: float = 0.0) -> Tuple[float, float]:
        """
        Calculate cluster-robust t-test.
        
        Args:
            returns: Return observations
            clusters: Cluster assignments
            null_mean: Null hypothesis mean
            
        Returns:
            Tuple of (t_statistic, p_value)
        """
        returns = np.array(returns)
        clusters = np.array(clusters)
        
        # Calculate cluster means
        unique_clusters = np.unique(clusters)
        cluster_means = np.array([np.mean(returns[clusters == c]) for c in unique_clusters])
        
        # Number of clusters and observations
        n_clusters = len(unique_clusters)
        n_obs = len(returns)
        
        # Overall mean
        overall_mean = np.mean(returns)
        
        # Calculate cluster-robust standard error
        cluster_variance = np.sum((cluster_means - overall_mean)**2) / (n_clusters - 1)
        cluster_se = np.sqrt(cluster_variance / n_clusters)
        
        # Cluster-robust t-statistic
        t_stat = (overall_mean - null_mean) / cluster_se
        
        # Degrees of freedom (number of clusters - 1)
        df = n_clusters - 1
        
        # Two-tailed p-value
        p_value = 2 * (1 - t_dist.cdf(np.abs(t_stat), df))
        
        return t_stat, p_value
    
    @staticmethod
    def clustered_car_test(abnormal_returns: np.ndarray, event_dates: np.ndarray,
                          event_window: Tuple[int, int] = (-1, 1)) -> Dict[str, float]:
        """
        Calculate clustered cumulative abnormal returns test.
        
        Args:
            abnormal_returns: Matrix of abnormal returns (events x time)
            event_dates: Array of event dates
            event_window: Event window (start, end) relative to event date
            
        Returns:
            Dictionary with test statistics
        """
        # Calculate CARs for each event
        start_offset, end_offset = event_window
        cars = np.sum(abnormal_returns[:, start_offset:end_offset+1], axis=1)
        
        # Create time-based clusters (e.g., monthly)
        clusters = np.array([pd.to_datetime(date).to_period('M').ordinal 
                           for date in event_dates])
        
        # Calculate cluster-robust statistics
        t_stat, p_value = ClusterRobustStatistics.cluster_robust_ttest(cars, clusters)
        
        # Additional statistics
        mean_car = np.mean(cars)
        std_car = np.std(cars, ddof=1)
        n_events = len(cars)
        n_clusters = len(np.unique(clusters))
        
        return {
            'mean_car': mean_car,
            'std_car': std_car,
            't_stat_clustered': t_stat,
            'p_value_clustered': p_value,
            'n_events': n_events,
            'n_clusters': n_clusters,
            'car_window': f"({start_offset}, {end_offset})"
        }


class SignificanceTestSuite:
    """
    Comprehensive significance testing suite for trading strategies.
    
    Combines multiple tests to provide robust evaluation of strategy performance
    while controlling for multiple testing, data snooping, and overfitting.
    """
    
    def __init__(self, strategy_returns: np.ndarray, benchmark_returns: np.ndarray,
                 strategy_names: Optional[List[str]] = None):
        """
        Initialize the test suite.
        
        Args:
            strategy_returns: Matrix of strategy returns (strategies x time)
            benchmark_returns: Benchmark returns
            strategy_names: Optional names for strategies
        """
        self.strategy_returns = np.array(strategy_returns)
        if self.strategy_returns.ndim == 1:
            self.strategy_returns = self.strategy_returns.reshape(1, -1)
            
        self.benchmark_returns = np.array(benchmark_returns)
        self.n_strategies, self.n_periods = self.strategy_returns.shape
        
        self.strategy_names = (strategy_names or 
                             [f"Strategy_{i+1}" for i in range(self.n_strategies)])
    
    def run_comprehensive_test(self, confidence_level: float = 0.95,
                             bootstrap_iterations: int = 10000) -> Dict[str, any]:
        """
        Run comprehensive significance testing suite.
        
        Args:
            confidence_level: Confidence level for tests
            bootstrap_iterations: Bootstrap iterations for SPA test
            
        Returns:
            Dictionary with all test results
        """
        results = {
            'summary': self._calculate_summary_statistics(),
            'spa_test': None,
            'deflated_sharpe': [],
            'pbo_test': None,
            'deployment_recommendation': None
        }
        
        # 1. SPA Test
        try:
            spa_test = WhiteRealityCheck(
                self.benchmark_returns, 
                self.strategy_returns,
                bootstrap_iterations=bootstrap_iterations
            )
            results['spa_test'] = spa_test.run_test(confidence_level)
        except Exception as e:
            logger.error(f"SPA test failed: {e}")
            results['spa_test'] = None
        
        # 2. Deflated Sharpe Ratio for each strategy
        for i, returns in enumerate(self.strategy_returns):
            try:
                dsr_result = DeflatedSharpe.calculate_deflated_sharpe(
                    returns, self.n_strategies
                )
                results['deflated_sharpe'].append({
                    'strategy': self.strategy_names[i],
                    'result': dsr_result
                })
            except Exception as e:
                logger.error(f"Deflated Sharpe test failed for strategy {i}: {e}")
        
        # 3. PBO Test
        try:
            pbo_estimator = PBOEstimator(self.strategy_returns)
            results['pbo_test'] = pbo_estimator.estimate_pbo()
        except Exception as e:
            logger.error(f"PBO test failed: {e}")
            results['pbo_test'] = None
        
        # 4. Deployment Recommendation
        results['deployment_recommendation'] = self._generate_deployment_recommendation(results)
        
        return results
    
    def _calculate_summary_statistics(self) -> Dict[str, any]:
        """Calculate summary statistics for all strategies."""
        stats = {}
        
        for i, returns in enumerate(self.strategy_returns):
            strategy_stats = {
                'mean_return': np.mean(returns),
                'volatility': np.std(returns, ddof=1),
                'sharpe_ratio': np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(returns),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns)
            }
            stats[self.strategy_names[i]] = strategy_stats
        
        # Benchmark statistics
        stats['benchmark'] = {
            'mean_return': np.mean(self.benchmark_returns),
            'volatility': np.std(self.benchmark_returns, ddof=1),
            'sharpe_ratio': (np.mean(self.benchmark_returns) / 
                           np.std(self.benchmark_returns, ddof=1) * np.sqrt(252)),
            'max_drawdown': self._calculate_max_drawdown(self.benchmark_returns)
        }
        
        return stats
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _generate_deployment_recommendation(self, results: Dict[str, any]) -> Dict[str, any]:
        """Generate deployment recommendation based on test results."""
        recommendation = {
            'deploy': False,
            'confidence': 'low',
            'reasons': [],
            'warnings': [],
            'best_strategy': None
        }
        
        # Check SPA test
        if results['spa_test'] and results['spa_test'].is_significant:
            recommendation['reasons'].append("SPA test indicates significant outperformance")
            spa_pass = True
        else:
            recommendation['warnings'].append("SPA test suggests potential data snooping")
            spa_pass = False
        
        # Check PBO
        if results['pbo_test']:
            if results['pbo_test'].pbo_estimate <= 0.2:  # Standard threshold
                recommendation['reasons'].append(f"Low PBO risk ({results['pbo_test'].pbo_estimate:.3f})")
                pbo_pass = True
            else:
                recommendation['warnings'].append(f"High PBO risk ({results['pbo_test'].pbo_estimate:.3f})")
                pbo_pass = False
        else:
            pbo_pass = False
        
        # Check deflated Sharpe ratios
        significant_strategies = [
            dsr['strategy'] for dsr in results['deflated_sharpe'] 
            if dsr['result'].is_significant
        ]
        
        if significant_strategies:
            recommendation['reasons'].append(f"Significant deflated Sharpe ratios: {significant_strategies}")
            dsr_pass = True
            recommendation['best_strategy'] = significant_strategies[0]  # First significant
        else:
            recommendation['warnings'].append("No strategies with significant deflated Sharpe ratios")
            dsr_pass = False
        
        # Overall recommendation
        if spa_pass and pbo_pass and dsr_pass:
            recommendation['deploy'] = True
            recommendation['confidence'] = 'high'
        elif (spa_pass and pbo_pass) or (spa_pass and dsr_pass) or (pbo_pass and dsr_pass):
            recommendation['deploy'] = True
            recommendation['confidence'] = 'medium'
        else:
            recommendation['deploy'] = False
            recommendation['confidence'] = 'low'
        
        return recommendation


# Utility functions for API integration

def validate_deployment_gates(spa_p_value: float, pbo_estimate: float,
                            spa_threshold: float = 0.05, 
                            pbo_threshold: float = 0.2) -> Dict[str, bool]:
    """
    Validate deployment gates based on significance tests.
    
    Args:
        spa_p_value: SPA test p-value
        pbo_estimate: PBO estimate
        spa_threshold: SPA p-value threshold (default 0.05)
        pbo_threshold: Maximum allowed PBO (default 0.2)
        
    Returns:
        Dictionary with gate results
    """
    return {
        'spa_gate_passed': spa_p_value < spa_threshold,
        'pbo_gate_passed': pbo_estimate <= pbo_threshold,
        'overall_gate_passed': (spa_p_value < spa_threshold and 
                               pbo_estimate <= pbo_threshold),
        'spa_p_value': spa_p_value,
        'pbo_estimate': pbo_estimate,
        'thresholds': {
            'spa_threshold': spa_threshold,
            'pbo_threshold': pbo_threshold
        }
    }


def format_test_results_for_api(results: Dict[str, any]) -> Dict[str, any]:
    """Format test results for API response."""
    formatted = {
        'timestamp': datetime.now().isoformat(),
        'summary': results.get('summary', {}),
        'tests': {}
    }
    
    # Format SPA test
    if results.get('spa_test'):
        spa = results['spa_test']
        formatted['tests']['spa'] = {
            'spa_statistic': spa.spa_statistic,
            'spa_p_value': spa.spa_p_value,
            'is_significant': spa.is_significant,
            'num_strategies': spa.num_strategies,
            'interpretation': spa.interpretation
        }
    
    # Format deflated Sharpe tests
    if results.get('deflated_sharpe'):
        formatted['tests']['deflated_sharpe'] = []
        for dsr in results['deflated_sharpe']:
            result = dsr['result']
            formatted['tests']['deflated_sharpe'].append({
                'strategy': dsr['strategy'],
                'observed_sharpe': result.observed_sharpe,
                'deflated_sharpe': result.deflated_sharpe,
                'p_value': result.p_value,
                'is_significant': result.is_significant,
                'interpretation': result.interpretation
            })
    
    # Format PBO test
    if results.get('pbo_test'):
        pbo = results['pbo_test']
        formatted['tests']['pbo'] = {
            'pbo_estimate': pbo.pbo_estimate,
            'is_overfitted': pbo.is_overfitted,
            'max_threshold': pbo.max_pbo_threshold,
            'interpretation': pbo.interpretation
        }
    
    # Format deployment recommendation
    if results.get('deployment_recommendation'):
        formatted['deployment_recommendation'] = results['deployment_recommendation']
    
    return formatted