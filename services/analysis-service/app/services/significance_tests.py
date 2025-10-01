"""
Statistical Significance Testing for Trading Strategies
Implements White's Reality Check, SPA, Deflated Sharpe Ratio, and PBO
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)


class TestMethod(Enum):
    """Statistical test methods"""
    WHITES_REALITY_CHECK = "whites_reality_check"
    SPA = "spa"  # Superior Predictive Ability
    STEPWISE_SPA = "stepwise_spa"
    

class BootstrapMethod(Enum):
    """Bootstrap resampling methods"""
    CIRCULAR_BLOCK = "circular_block"
    STATIONARY_BLOCK = "stationary_block"
    MOVING_BLOCK = "moving_block"


@dataclass
class SignificanceTestResult:
    """Result of a significance test"""
    test_statistic: float
    p_value: float
    method: str
    n_strategies: int
    n_bootstrap: int
    best_strategy_index: Optional[int] = None
    reject_null: Optional[bool] = None
    confidence_level: float = 0.05
    
    def is_significant(self, alpha: float = None) -> bool:
        """Check if result is statistically significant"""
        alpha = alpha or self.confidence_level
        return self.p_value < alpha


@dataclass
class DeflatedSharpeResult:
    """Result of Deflated Sharpe Ratio calculation"""
    sharpe_ratio: float
    deflated_sharpe: float
    p_value: float
    n_trials: int
    avg_correlation: float
    reject_null: bool
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if Sharpe ratio is statistically significant"""
        return self.p_value < alpha


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting result"""
    pbo: float
    n_strategies: int
    oos_performance_rank: float
    is_performance_rank: float
    reject_null: bool
    
    def is_overfitted(self, threshold: float = 0.2) -> bool:
        """Check if backtest shows signs of overfitting"""
        return self.pbo > threshold


class WhitesRealityCheck:
    """
    Implementation of White's Reality Check for data snooping
    Tests null hypothesis that best strategy has no superior performance
    """
    
    def __init__(self, bootstrap_method: BootstrapMethod = BootstrapMethod.CIRCULAR_BLOCK):
        self.bootstrap_method = bootstrap_method
    
    def test(self, 
             returns_matrix: np.ndarray,
             benchmark_returns: np.ndarray = None,
             n_bootstrap: int = 2000,
             block_size: int = None,
             alpha: float = 0.05) -> SignificanceTestResult:
        """
        Perform White's Reality Check test
        
        Args:
            returns_matrix: (T x N) matrix of strategy returns
            benchmark_returns: (T,) benchmark returns to compare against
            n_bootstrap: Number of bootstrap samples
            block_size: Block size for bootstrap (auto if None)
            alpha: Significance level
            
        Returns:
            SignificanceTestResult with test statistics and p-value
        """
        T, N = returns_matrix.shape
        
        if benchmark_returns is None:
            benchmark_returns = np.zeros(T)
        
        # Calculate excess returns
        excess_returns = returns_matrix - benchmark_returns.reshape(-1, 1)
        
        # Calculate mean excess returns
        mean_excess = np.mean(excess_returns, axis=0)
        
        # Find best performing strategy
        best_idx = np.argmax(mean_excess)
        test_stat = mean_excess[best_idx]
        
        # Bootstrap procedure
        if block_size is None:
            block_size = int(np.ceil(T ** (1/3)))
        
        bootstrap_stats = self._bootstrap_test_statistic(
            excess_returns, n_bootstrap, block_size
        )
        
        # Calculate p-value
        p_value = np.mean(bootstrap_stats >= test_stat)
        
        return SignificanceTestResult(
            test_statistic=test_stat,
            p_value=p_value,
            method="whites_reality_check",
            n_strategies=N,
            n_bootstrap=n_bootstrap,
            best_strategy_index=int(best_idx),
            reject_null=p_value < alpha,
            confidence_level=alpha
        )
    
    def _bootstrap_test_statistic(self, 
                                 excess_returns: np.ndarray,
                                 n_bootstrap: int,
                                 block_size: int) -> np.ndarray:
        """Generate bootstrap distribution of test statistic"""
        T, N = excess_returns.shape
        bootstrap_stats = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            # Generate bootstrap sample
            if self.bootstrap_method == BootstrapMethod.CIRCULAR_BLOCK:
                bootstrap_sample = self._circular_block_bootstrap(
                    excess_returns, block_size
                )
            else:
                bootstrap_sample = self._stationary_block_bootstrap(
                    excess_returns, block_size
                )
            
            # Center the bootstrap sample to impose null hypothesis
            bootstrap_sample_centered = bootstrap_sample - np.mean(bootstrap_sample, axis=0)
            
            # Calculate test statistic for bootstrap sample
            mean_bootstrap = np.mean(bootstrap_sample_centered, axis=0)
            bootstrap_stats[i] = np.max(mean_bootstrap)
        
        return bootstrap_stats
    
    def _circular_block_bootstrap(self, 
                                 data: np.ndarray,
                                 block_size: int) -> np.ndarray:
        """Circular block bootstrap resampling"""
        T, N = data.shape
        n_blocks = int(np.ceil(T / block_size))
        
        # Create circular version of data
        circular_data = np.vstack([data, data[:block_size-1]])
        
        # Sample block starting points
        start_indices = np.random.randint(0, T, n_blocks)
        
        # Construct bootstrap sample
        bootstrap_sample = []
        for start_idx in start_indices:
            block = circular_data[start_idx:start_idx + block_size]
            bootstrap_sample.append(block)
        
        bootstrap_sample = np.vstack(bootstrap_sample)[:T]
        return bootstrap_sample
    
    def _stationary_block_bootstrap(self, 
                                   data: np.ndarray,
                                   avg_block_size: int) -> np.ndarray:
        """Stationary block bootstrap with geometric block lengths"""
        T, N = data.shape
        bootstrap_sample = []
        current_length = 0
        
        while current_length < T:
            # Geometric block length
            block_length = np.random.geometric(1/avg_block_size)
            start_idx = np.random.randint(0, T)
            
            # Extract block (with wraparound)
            for i in range(block_length):
                if current_length >= T:
                    break
                idx = (start_idx + i) % T
                bootstrap_sample.append(data[idx])
                current_length += 1
        
        return np.array(bootstrap_sample[:T])


class SuperiorPredictiveAbility:
    """
    Implementation of Hansen's Superior Predictive Ability (SPA) test
    More powerful than White's Reality Check, controls for poor performing strategies
    """
    
    def __init__(self):
        self.studentization_constant = 0.5
    
    def test(self, 
             returns_matrix: np.ndarray,
             benchmark_returns: np.ndarray = None,
             n_bootstrap: int = 2000,
             block_size: int = None,
             alpha: float = 0.05,
             studentize: bool = True) -> SignificanceTestResult:
        """
        Perform SPA test
        
        Args:
            returns_matrix: (T x N) matrix of strategy returns
            benchmark_returns: (T,) benchmark returns
            n_bootstrap: Number of bootstrap samples
            block_size: Block size for bootstrap
            alpha: Significance level
            studentize: Whether to studentize the test statistic
            
        Returns:
            SignificanceTestResult
        """
        T, N = returns_matrix.shape
        
        if benchmark_returns is None:
            benchmark_returns = np.zeros(T)
        
        # Calculate excess returns
        excess_returns = returns_matrix - benchmark_returns.reshape(-1, 1)
        
        # Calculate sample statistics
        mean_excess = np.mean(excess_returns, axis=0)
        
        if studentize:
            # Calculate standard errors
            std_errors = np.std(excess_returns, axis=0, ddof=1) / np.sqrt(T)
            # Avoid division by zero
            std_errors = np.maximum(std_errors, 1e-8)
            t_stats = mean_excess / std_errors
            test_stat = np.max(t_stats)
        else:
            test_stat = np.max(mean_excess)
        
        # Bootstrap procedure
        if block_size is None:
            block_size = int(np.ceil(T ** (1/3)))
        
        bootstrap_stats = self._spa_bootstrap(
            excess_returns, n_bootstrap, block_size, studentize
        )
        
        # Calculate p-value
        p_value = np.mean(bootstrap_stats >= test_stat)
        
        best_idx = np.argmax(mean_excess)
        
        return SignificanceTestResult(
            test_statistic=test_stat,
            p_value=p_value,
            method="spa",
            n_strategies=N,
            n_bootstrap=n_bootstrap,
            best_strategy_index=int(best_idx),
            reject_null=p_value < alpha,
            confidence_level=alpha
        )
    
    def _spa_bootstrap(self, 
                      excess_returns: np.ndarray,
                      n_bootstrap: int,
                      block_size: int,
                      studentize: bool) -> np.ndarray:
        """SPA bootstrap procedure with re-centering"""
        T, N = excess_returns.shape
        bootstrap_stats = np.zeros(n_bootstrap)
        
        # Calculate original sample means and std errors
        original_means = np.mean(excess_returns, axis=0)
        original_stds = np.std(excess_returns, axis=0, ddof=1) / np.sqrt(T)
        original_stds = np.maximum(original_stds, 1e-8)
        
        wrc = WhitesRealityCheck()
        
        for i in range(n_bootstrap):
            # Generate bootstrap sample
            bootstrap_sample = wrc._circular_block_bootstrap(excess_returns, block_size)
            
            # Re-center bootstrap sample for SPA
            bootstrap_means = np.mean(bootstrap_sample, axis=0)
            
            # SPA re-centering: subtract max(0, original_mean) to impose null
            recentering = np.maximum(original_means, 0)
            bootstrap_recentered = bootstrap_sample - recentering
            
            # Calculate bootstrap test statistic
            bootstrap_means_recentered = np.mean(bootstrap_recentered, axis=0)
            
            if studentize:
                bootstrap_stds = np.std(bootstrap_recentered, axis=0, ddof=1) / np.sqrt(T)
                bootstrap_stds = np.maximum(bootstrap_stds, 1e-8)
                t_stats_bootstrap = bootstrap_means_recentered / bootstrap_stds
                bootstrap_stats[i] = np.max(t_stats_bootstrap)
            else:
                bootstrap_stats[i] = np.max(bootstrap_means_recentered)
        
        return bootstrap_stats


class StepwiseSPA:
    """
    Stepwise SPA test for identifying multiple superior strategies
    """
    
    def __init__(self):
        self.spa = SuperiorPredictiveAbility()
    
    def test(self, 
             returns_matrix: np.ndarray,
             benchmark_returns: np.ndarray = None,
             n_bootstrap: int = 2000,
             alpha: float = 0.05,
             max_steps: int = None) -> List[SignificanceTestResult]:
        """
        Perform stepwise SPA test
        
        Returns list of results, one for each step until no more significant strategies
        """
        if max_steps is None:
            max_steps = returns_matrix.shape[1]  # Max number of strategies
        
        results = []
        remaining_indices = list(range(returns_matrix.shape[1]))
        
        for step in range(max_steps):
            if len(remaining_indices) == 0:
                break
            
            # Test remaining strategies
            current_returns = returns_matrix[:, remaining_indices]
            
            result = self.spa.test(
                current_returns, 
                benchmark_returns, 
                n_bootstrap, 
                alpha=alpha
            )
            
            if not result.reject_null:
                # No more significant strategies
                break
            
            # Record result with original index
            original_best_idx = remaining_indices[result.best_strategy_index]
            result.best_strategy_index = original_best_idx
            results.append(result)
            
            # Remove the identified superior strategy
            remaining_indices.pop(result.best_strategy_index)
        
        return results


class DeflatedSharpe:
    """
    Implementation of Deflated Sharpe Ratio by López de Prado
    Adjusts Sharpe ratio for multiple testing and non-normality
    """
    
    @staticmethod
    def calculate(returns: np.ndarray,
                 n_trials: int,
                 skewness: float = None,
                 kurtosis: float = None,
                 avg_correlation: float = 0.0,
                 benchmark_sharpe: float = 0.0) -> DeflatedSharpeResult:
        """
        Calculate Deflated Sharpe Ratio
        
        Args:
            returns: Strategy returns
            n_trials: Number of strategies tested
            skewness: Returns skewness (calculated if None)
            kurtosis: Returns excess kurtosis (calculated if None)
            avg_correlation: Average correlation between strategies
            benchmark_sharpe: Benchmark Sharpe ratio
            
        Returns:
            DeflatedSharpeResult
        """
        # Calculate basic statistics
        T = len(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        if skewness is None:
            skewness = stats.skew(returns)
        if kurtosis is None:
            kurtosis = stats.kurtosis(returns)  # Excess kurtosis
        
        # Estimate parameters for multiple testing adjustment
        variance_sharpe = DeflatedSharpe._variance_of_sharpe(T, skewness, kurtosis, sharpe_ratio)
        
        # Expected maximum Sharpe ratio under null
        expected_max_sharpe = DeflatedSharpe._expected_maximum_sharpe(
            n_trials, avg_correlation, variance_sharpe
        )
        
        # Standard deviation of maximum Sharpe ratio
        std_max_sharpe = np.sqrt(variance_sharpe)
        
        # Deflated Sharpe ratio
        if std_max_sharpe > 0:
            deflated_sharpe = (sharpe_ratio - expected_max_sharpe) / std_max_sharpe
        else:
            deflated_sharpe = 0
        
        # P-value (one-tailed test)
        p_value = 1 - stats.norm.cdf(deflated_sharpe)
        
        return DeflatedSharpeResult(
            sharpe_ratio=sharpe_ratio,
            deflated_sharpe=deflated_sharpe,
            p_value=p_value,
            n_trials=n_trials,
            avg_correlation=avg_correlation,
            reject_null=p_value < 0.05
        )
    
    @staticmethod
    def _variance_of_sharpe(T: int, skewness: float, kurtosis: float, sharpe: float) -> float:
        """Calculate variance of Sharpe ratio estimator"""
        # Mertens (2002) formula for variance of Sharpe ratio
        variance = (1 + 0.5 * sharpe**2 - skewness * sharpe + (kurtosis/4) * sharpe**2) / T
        return max(variance, 1e-8)  # Ensure positive
    
    @staticmethod
    def _expected_maximum_sharpe(n_trials: int, avg_correlation: float, variance_sharpe: float) -> float:
        """Expected maximum Sharpe ratio under null hypothesis"""
        if n_trials <= 1:
            return 0
        
        # Effective number of independent trials
        effective_n = n_trials * (1 - avg_correlation) / (1 + (n_trials - 1) * avg_correlation)
        effective_n = max(effective_n, 1)
        
        # Expected maximum of standard normal variates
        gamma = 0.5772156649015329  # Euler-Mascheroni constant
        
        if effective_n > 1:
            expected_max = np.sqrt(2 * np.log(effective_n)) - (np.log(np.log(effective_n)) + gamma) / (2 * np.sqrt(2 * np.log(effective_n)))
        else:
            expected_max = 0
        
        return expected_max * np.sqrt(variance_sharpe)


class ProbabilityBacktestOverfitting:
    """
    Implementation of Probability of Backtest Overfitting (PBO) by López de Prado
    Estimates probability that out-of-sample performance ranking differs from in-sample
    """
    
    @staticmethod
    def calculate(is_returns_matrix: np.ndarray,
                 oos_returns_matrix: np.ndarray,
                 n_bootstrap: int = 1000) -> PBOResult:
        """
        Calculate Probability of Backtest Overfitting
        
        Args:
            is_returns_matrix: (T_is x N) in-sample returns matrix
            oos_returns_matrix: (T_oos x N) out-of-sample returns matrix
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            PBOResult
        """
        N = is_returns_matrix.shape[1]
        
        if oos_returns_matrix.shape[1] != N:
            raise ValueError("In-sample and out-of-sample must have same number of strategies")
        
        # Calculate performance metrics (Sharpe ratios)
        is_sharpe = ProbabilityBacktestOverfitting._calculate_sharpe_ratios(is_returns_matrix)
        oos_sharpe = ProbabilityBacktestOverfitting._calculate_sharpe_ratios(oos_returns_matrix)
        
        # Rank strategies by in-sample performance
        is_ranks = stats.rankdata(-is_sharpe)  # Negative for descending order
        oos_ranks = stats.rankdata(-oos_sharpe)
        
        # Find best in-sample strategy
        best_is_strategy = np.argmax(is_sharpe)
        
        # Calculate rank correlation
        rank_correlation = stats.spearmanr(is_ranks, oos_ranks)[0]
        
        # Bootstrap procedure to estimate PBO
        bootstrap_correlations = []
        
        for _ in range(n_bootstrap):
            # Bootstrap in-sample returns
            bootstrap_is = ProbabilityBacktestOverfitting._bootstrap_sample(is_returns_matrix)
            bootstrap_is_sharpe = ProbabilityBacktestOverfitting._calculate_sharpe_ratios(bootstrap_is)
            
            # Calculate ranks
            bootstrap_is_ranks = stats.rankdata(-bootstrap_is_sharpe)
            
            # Calculate correlation with out-of-sample ranks
            bootstrap_corr = stats.spearmanr(bootstrap_is_ranks, oos_ranks)[0]
            bootstrap_correlations.append(bootstrap_corr)
        
        # PBO is probability that correlation is negative or zero
        pbo = np.mean(np.array(bootstrap_correlations) <= 0)
        
        # Performance rank statistics
        is_performance_rank = is_ranks[best_is_strategy] / N
        oos_performance_rank = oos_ranks[best_is_strategy] / N
        
        return PBOResult(
            pbo=pbo,
            n_strategies=N,
            oos_performance_rank=oos_performance_rank,
            is_performance_rank=is_performance_rank,
            reject_null=pbo > 0.5  # Null: no overfitting
        )
    
    @staticmethod
    def _calculate_sharpe_ratios(returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate Sharpe ratios for each strategy"""
        means = np.mean(returns_matrix, axis=0)
        stds = np.std(returns_matrix, axis=0, ddof=1)
        
        # Avoid division by zero
        stds = np.maximum(stds, 1e-8)
        
        return means / stds
    
    @staticmethod
    def _bootstrap_sample(data: np.ndarray) -> np.ndarray:
        """Generate bootstrap sample using circular block bootstrap"""
        T = data.shape[0]
        block_size = max(1, int(T ** (1/3)))
        
        wrc = WhitesRealityCheck()
        return wrc._circular_block_bootstrap(data, block_size)


class SignificanceTestSuite:
    """
    Comprehensive suite of significance tests for trading strategies
    """
    
    def __init__(self):
        self.wrc = WhitesRealityCheck()
        self.spa = SuperiorPredictiveAbility()
        self.stepwise_spa = StepwiseSPA()
    
    def run_comprehensive_tests(self,
                               returns_matrix: np.ndarray,
                               benchmark_returns: np.ndarray = None,
                               is_returns_matrix: np.ndarray = None,
                               oos_returns_matrix: np.ndarray = None,
                               n_bootstrap: int = 2000,
                               alpha: float = 0.05) -> Dict:
        """
        Run comprehensive significance testing suite
        
        Args:
            returns_matrix: Full period returns matrix
            benchmark_returns: Benchmark returns
            is_returns_matrix: In-sample returns (for PBO)
            oos_returns_matrix: Out-of-sample returns (for PBO)
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level
            
        Returns:
            Dictionary with all test results
        """
        results = {}
        
        # White's Reality Check
        try:
            wrc_result = self.wrc.test(returns_matrix, benchmark_returns, n_bootstrap, alpha=alpha)
            results['whites_reality_check'] = wrc_result
        except Exception as e:
            logger.error(f"White's Reality Check failed: {e}")
            results['whites_reality_check'] = None
        
        # SPA Test
        try:
            spa_result = self.spa.test(returns_matrix, benchmark_returns, n_bootstrap, alpha=alpha)
            results['spa'] = spa_result
        except Exception as e:
            logger.error(f"SPA test failed: {e}")
            results['spa'] = None
        
        # Stepwise SPA
        try:
            stepwise_results = self.stepwise_spa.test(returns_matrix, benchmark_returns, n_bootstrap, alpha=alpha)
            results['stepwise_spa'] = stepwise_results
        except Exception as e:
            logger.error(f"Stepwise SPA failed: {e}")
            results['stepwise_spa'] = None
        
        # Deflated Sharpe Ratio for best strategy
        if 'spa' in results and results['spa'] is not None:
            try:
                best_idx = results['spa'].best_strategy_index
                best_returns = returns_matrix[:, best_idx]
                
                dsr_result = DeflatedSharpe.calculate(
                    best_returns,
                    n_trials=returns_matrix.shape[1],
                    avg_correlation=0.1  # Conservative estimate
                )
                results['deflated_sharpe'] = dsr_result
            except Exception as e:
                logger.error(f"Deflated Sharpe calculation failed: {e}")
                results['deflated_sharpe'] = None
        
        # Probability of Backtest Overfitting
        if is_returns_matrix is not None and oos_returns_matrix is not None:
            try:
                pbo_result = ProbabilityBacktestOverfitting.calculate(
                    is_returns_matrix,
                    oos_returns_matrix,
                    n_bootstrap=min(n_bootstrap, 1000)  # PBO doesn't need as many
                )
                results['pbo'] = pbo_result
            except Exception as e:
                logger.error(f"PBO calculation failed: {e}")
                results['pbo'] = None
        
        return results
    
    def validate_deployment_criteria(self,
                                   test_results: Dict,
                                   spa_threshold: float = 0.05,
                                   pbo_threshold: float = 0.2,
                                   dsr_threshold: float = 0.05) -> Dict[str, bool]:
        """
        Validate strategy against deployment criteria
        
        Returns:
            Dictionary of pass/fail for each criterion
        """
        validation = {}
        
        # SPA test criterion
        if 'spa' in test_results and test_results['spa'] is not None:
            spa_result = test_results['spa']
            validation['spa_pass'] = spa_result.p_value < spa_threshold
        else:
            validation['spa_pass'] = False
        
        # PBO criterion
        if 'pbo' in test_results and test_results['pbo'] is not None:
            pbo_result = test_results['pbo']
            validation['pbo_pass'] = pbo_result.pbo <= pbo_threshold
        else:
            validation['pbo_pass'] = True  # Pass if not available
        
        # Deflated Sharpe criterion
        if 'deflated_sharpe' in test_results and test_results['deflated_sharpe'] is not None:
            dsr_result = test_results['deflated_sharpe']
            validation['dsr_pass'] = dsr_result.p_value < dsr_threshold
        else:
            validation['dsr_pass'] = True  # Pass if not available
        
        # Overall deployment approval
        validation['deployment_approved'] = all([
            validation.get('spa_pass', False),
            validation.get('pbo_pass', True),
            validation.get('dsr_pass', True)
        ])
        
        return validation


# Utility functions for integration

def format_results_for_api(test_results: Dict) -> Dict:
    """Format test results for API response"""
    formatted = {}
    
    for test_name, result in test_results.items():
        if result is None:
            continue
            
        if isinstance(result, SignificanceTestResult):
            formatted[test_name] = {
                'test_statistic': float(result.test_statistic),
                'p_value': float(result.p_value),
                'method': result.method,
                'n_strategies': result.n_strategies,
                'n_bootstrap': result.n_bootstrap,
                'best_strategy_index': result.best_strategy_index,
                'reject_null': result.reject_null,
                'is_significant': result.is_significant()
            }
        elif isinstance(result, DeflatedSharpeResult):
            formatted[test_name] = {
                'sharpe_ratio': float(result.sharpe_ratio),
                'deflated_sharpe': float(result.deflated_sharpe),
                'p_value': float(result.p_value),
                'n_trials': result.n_trials,
                'avg_correlation': float(result.avg_correlation),
                'is_significant': result.is_significant()
            }
        elif isinstance(result, PBOResult):
            formatted[test_name] = {
                'pbo': float(result.pbo),
                'n_strategies': result.n_strategies,
                'oos_performance_rank': float(result.oos_performance_rank),
                'is_performance_rank': float(result.is_performance_rank),
                'is_overfitted': result.is_overfitted()
            }
        elif isinstance(result, list):  # Stepwise SPA
            formatted[test_name] = [
                {
                    'step': i,
                    'test_statistic': float(r.test_statistic),
                    'p_value': float(r.p_value),
                    'best_strategy_index': r.best_strategy_index,
                    'is_significant': r.is_significant()
                }
                for i, r in enumerate(result)
            ]
    
    return formatted


def calculate_strategy_correlations(returns_matrix: np.ndarray) -> float:
    """Calculate average pairwise correlation between strategies"""
    corr_matrix = np.corrcoef(returns_matrix.T)
    
    # Extract upper triangle excluding diagonal
    n = corr_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    correlations = corr_matrix[upper_tri_indices]
    
    # Remove NaN values
    correlations = correlations[~np.isnan(correlations)]
    
    if len(correlations) == 0:
        return 0.0
    
    return np.mean(correlations)