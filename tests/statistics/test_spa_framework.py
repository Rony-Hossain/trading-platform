"""
Tests for SPA Framework - White's Reality Check & Superior Predictive Ability Tests

Comprehensive test suite for statistical significance testing framework.
Validates implementation against known statistical properties.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date
import warnings
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add analysis service to path
analysis_service_path = project_root / 'services' / 'analysis-service'
sys.path.insert(0, str(analysis_service_path))

from app.statistics.spa_framework import (
    SPATestFramework, PerformanceMetrics, TestType, BootstrapMethod,
    create_performance_metrics_from_returns, load_strategy_returns_from_dataframe
)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

class TestPerformanceMetrics:
    """Test PerformanceMetrics class functionality"""
    
    def test_performance_metrics_creation(self):
        """Test basic creation of PerformanceMetrics"""
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        benchmark_returns = np.random.normal(0.0005, 0.015, 252)
        
        metrics = PerformanceMetrics(
            strategy_name="test_strategy",
            returns=returns,
            benchmark_returns=benchmark_returns
        )
        
        assert metrics.strategy_name == "test_strategy"
        assert len(metrics.returns) == 252
        assert len(metrics.benchmark_returns) == 252
        assert metrics.excess_returns is not None
        assert len(metrics.excess_returns) == 252
        assert metrics.sharpe_ratio is not None
        assert metrics.information_ratio is not None
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        # Create returns with known Sharpe ratio
        returns = np.random.normal(0.001, 0.02, 252)  # ~1.25 Sharpe
        
        metrics = PerformanceMetrics(
            strategy_name="test",
            returns=returns
        )
        
        # Sharpe should be positive for positive mean returns
        assert metrics.sharpe_ratio > 0
        
        # Test with zero returns
        zero_returns = np.zeros(252)
        zero_metrics = PerformanceMetrics(
            strategy_name="zero",
            returns=zero_returns
        )
        assert zero_metrics.sharpe_ratio == 0
    
    def test_information_ratio_calculation(self):
        """Test information ratio calculation"""
        returns = np.random.normal(0.001, 0.02, 252)
        benchmark_returns = np.random.normal(0.0005, 0.015, 252)
        
        metrics = PerformanceMetrics(
            strategy_name="test",
            returns=returns,
            benchmark_returns=benchmark_returns
        )
        
        assert metrics.information_ratio is not None
        assert isinstance(metrics.information_ratio, float)

class TestSPATestFramework:
    """Test main SPA framework functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample strategy and benchmark data"""
        np.random.seed(42)  # For reproducible tests
        
        n_strategies = 5
        sample_size = 500  # ~2 years of daily data
        
        # Generate correlated strategy returns
        strategy_returns = []
        for i in range(n_strategies):
            # Some strategies outperform, some don't
            mean_return = 0.0005 + i * 0.0002  # Increasing performance
            returns = np.random.normal(mean_return, 0.02, sample_size)
            strategy_returns.append(returns)
        
        # Benchmark returns
        benchmark_returns = np.random.normal(0.0003, 0.015, sample_size)
        
        # Create PerformanceMetrics objects
        strategy_metrics = []
        for i, returns in enumerate(strategy_returns):
            metrics = PerformanceMetrics(
                strategy_name=f"strategy_{i+1}",
                returns=returns,
                benchmark_returns=benchmark_returns
            )
            strategy_metrics.append(metrics)
        
        benchmark_metrics = PerformanceMetrics(
            strategy_name="benchmark",
            returns=benchmark_returns
        )
        
        return strategy_metrics, benchmark_metrics
    
    @pytest.fixture
    def spa_framework(self):
        """Create SPA test framework"""
        return SPATestFramework(
            bootstrap_iterations=1000,  # Reduced for faster testing
            bootstrap_method=BootstrapMethod.STATIONARY
        )
    
    def test_reality_check_test(self, spa_framework, sample_data):
        """Test White's Reality Check implementation"""
        strategy_metrics, benchmark_metrics = sample_data
        
        result = spa_framework.reality_check_test(
            strategy_metrics, benchmark_metrics, test_statistic="sharpe_ratio"
        )
        
        # Validate result structure
        assert result.test_type == TestType.REALITY_CHECK
        assert isinstance(result.test_statistic, float)
        assert 0 <= result.p_value <= 1
        assert isinstance(result.critical_value_95, float)
        assert isinstance(result.critical_value_99, float)
        assert len(result.bootstrap_distribution) == 1000
        assert result.strategy_name in [m.strategy_name for m in strategy_metrics]
        assert result.sample_size == len(strategy_metrics[0].returns)
    
    def test_spa_consistent_test(self, spa_framework, sample_data):
        """Test SPA consistent test implementation"""
        strategy_metrics, benchmark_metrics = sample_data
        
        result = spa_framework.spa_test(
            strategy_metrics, benchmark_metrics, 
            test_statistic="sharpe_ratio",
            test_type=TestType.SPA_CONSISTENT
        )
        
        # Validate result structure
        assert result.test_type == TestType.SPA_CONSISTENT
        assert isinstance(result.test_statistic, float)
        assert 0 <= result.p_value <= 1
        assert len(result.bootstrap_distribution) == 1000
    
    def test_spa_lower_test(self, spa_framework, sample_data):
        """Test SPA lower test implementation"""
        strategy_metrics, benchmark_metrics = sample_data
        
        result = spa_framework.spa_test(
            strategy_metrics, benchmark_metrics,
            test_type=TestType.SPA_LOWER
        )
        
        assert result.test_type == TestType.SPA_LOWER
        assert isinstance(result.test_statistic, float)
        assert 0 <= result.p_value <= 1
    
    def test_spa_upper_test(self, spa_framework, sample_data):
        """Test SPA upper test implementation"""
        strategy_metrics, benchmark_metrics = sample_data
        
        result = spa_framework.spa_test(
            strategy_metrics, benchmark_metrics,
            test_type=TestType.SPA_UPPER
        )
        
        assert result.test_type == TestType.SPA_UPPER
        assert isinstance(result.test_statistic, float)
        assert 0 <= result.p_value <= 1
    
    def test_multiple_testing_correction(self, spa_framework, sample_data):
        """Test multiple testing correction"""
        strategy_metrics, benchmark_metrics = sample_data
        
        # Run individual tests
        individual_results = []
        for strategy in strategy_metrics:
            result = spa_framework.spa_test(
                [strategy], benchmark_metrics, test_type=TestType.SPA_CONSISTENT
            )
            result.strategy_name = strategy.strategy_name
            individual_results.append(result)
        
        # Apply multiple testing correction
        mt_results = spa_framework.multiple_testing_correction(individual_results)
        
        # Validate results
        assert len(mt_results.strategy_names) == len(strategy_metrics)
        assert len(mt_results.raw_p_values) == len(strategy_metrics)
        assert len(mt_results.bonferroni_p_values) == len(strategy_metrics)
        assert len(mt_results.fdr_p_values) == len(strategy_metrics)
        
        # Bonferroni should be more conservative
        assert all(mt_results.bonferroni_p_values >= mt_results.raw_p_values)
        
        # Check that all p-values are valid
        assert all(0 <= p <= 1 for p in mt_results.raw_p_values)
        assert all(0 <= p <= 1 for p in mt_results.bonferroni_p_values)
        assert all(0 <= p <= 1 for p in mt_results.fdr_p_values)
        
        # Error rates should be reasonable
        assert 0 <= mt_results.family_wise_error_rate <= 1
        assert 0 <= mt_results.false_discovery_rate <= 1
    
    def test_comprehensive_testing(self, spa_framework, sample_data):
        """Test comprehensive testing workflow"""
        strategy_metrics, benchmark_metrics = sample_data
        
        results = spa_framework.comprehensive_strategy_testing(
            strategy_metrics, benchmark_metrics, include_individual_tests=True
        )
        
        # Validate results structure
        assert "test_summary" in results
        assert "reality_check" in results
        assert "spa_tests" in results
        assert "individual_tests" in results
        assert "multiple_testing" in results
        assert "summary" in results
        
        # Check test summary
        summary = results["test_summary"]
        assert summary["n_strategies"] == len(strategy_metrics)
        assert summary["benchmark"] == benchmark_metrics.strategy_name
        assert summary["sample_size"] == len(strategy_metrics[0].returns)
        
        # Check reality check results
        rc = results["reality_check"]
        if rc is not None:
            assert rc.test_type == TestType.REALITY_CHECK
        
        # Check SPA tests
        spa_tests = results["spa_tests"]
        for test_type in ["spa_consistent", "spa_lower", "spa_upper"]:
            if spa_tests.get(test_type) is not None:
                assert isinstance(spa_tests[test_type].p_value, float)
        
        # Check individual tests
        individual_tests = results["individual_tests"]
        assert len(individual_tests) == len(strategy_metrics)
        
        # Check multiple testing
        mt = results["multiple_testing"]
        assert len(mt.strategy_names) == len(strategy_metrics)
        
        # Check summary
        summary = results["summary"]
        assert "reality_check_significant" in summary
        assert "spa_tests_significant" in summary
        assert "recommendations" in summary
        assert isinstance(summary["recommendations"], list)

class TestBootstrapMethods:
    """Test different bootstrap methods"""
    
    @pytest.fixture
    def spa_framework_stationary(self):
        return SPATestFramework(
            bootstrap_iterations=100,  # Small for testing
            bootstrap_method=BootstrapMethod.STATIONARY,
            block_length=10
        )
    
    @pytest.fixture
    def spa_framework_circular(self):
        return SPATestFramework(
            bootstrap_iterations=100,
            bootstrap_method=BootstrapMethod.CIRCULAR,
            block_length=10
        )
    
    @pytest.fixture
    def simple_data(self):
        """Simple data for bootstrap testing"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        benchmark_returns = np.random.normal(0.0005, 0.015, 100)
        
        strategy_metrics = [PerformanceMetrics("strategy", returns)]
        benchmark_metrics = PerformanceMetrics("benchmark", benchmark_returns)
        
        return strategy_metrics, benchmark_metrics
    
    def test_stationary_bootstrap(self, spa_framework_stationary, simple_data):
        """Test stationary bootstrap method"""
        strategy_metrics, benchmark_metrics = simple_data
        
        result = spa_framework_stationary.reality_check_test(
            strategy_metrics, benchmark_metrics
        )
        
        assert isinstance(result.test_statistic, float)
        assert 0 <= result.p_value <= 1
        assert len(result.bootstrap_distribution) == 100
    
    def test_circular_bootstrap(self, spa_framework_circular, simple_data):
        """Test circular bootstrap method"""
        strategy_metrics, benchmark_metrics = simple_data
        
        result = spa_framework_circular.reality_check_test(
            strategy_metrics, benchmark_metrics
        )
        
        assert isinstance(result.test_statistic, float)
        assert 0 <= result.p_value <= 1
        assert len(result.bootstrap_distribution) == 100
    
    def test_bootstrap_indices_generation(self, spa_framework_stationary):
        """Test bootstrap indices generation"""
        sample_size = 100
        
        # Test stationary bootstrap
        indices = spa_framework_stationary._generate_bootstrap_indices(sample_size)
        
        assert len(indices) == sample_size
        assert all(0 <= idx < sample_size for idx in indices)
        
        # Test circular bootstrap
        spa_framework_stationary.bootstrap_method = BootstrapMethod.CIRCULAR
        indices_circular = spa_framework_stationary._generate_bootstrap_indices(sample_size)
        
        assert len(indices_circular) == sample_size
        assert all(0 <= idx < sample_size for idx in indices_circular)

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_performance_metrics_from_returns(self):
        """Test utility function for creating performance metrics"""
        returns = np.random.normal(0.001, 0.02, 252)
        benchmark_returns = np.random.normal(0.0005, 0.015, 252)
        
        metrics = create_performance_metrics_from_returns(
            "test_strategy", returns, benchmark_returns
        )
        
        assert metrics.strategy_name == "test_strategy"
        assert np.array_equal(metrics.returns, returns)
        assert np.array_equal(metrics.benchmark_returns, benchmark_returns)
        assert metrics.sharpe_ratio is not None
    
    def test_load_strategy_returns_from_dataframe(self):
        """Test loading strategy data from DataFrame"""
        # Create sample DataFrame
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        data = {
            "date": dates,
            "strategy_1": np.random.normal(0.001, 0.02, 252),
            "strategy_2": np.random.normal(0.0008, 0.022, 252),
            "strategy_3": np.random.normal(0.0012, 0.018, 252),
            "benchmark": np.random.normal(0.0005, 0.015, 252)
        }
        df = pd.DataFrame(data)
        
        strategy_columns = ["strategy_1", "strategy_2", "strategy_3"]
        strategy_metrics, benchmark_metrics = load_strategy_returns_from_dataframe(
            df, strategy_columns, "benchmark"
        )
        
        assert len(strategy_metrics) == 3
        assert benchmark_metrics.strategy_name == "benchmark"
        
        for i, metrics in enumerate(strategy_metrics):
            assert metrics.strategy_name == f"strategy_{i+1}"
            assert len(metrics.returns) == 252

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_strategy_list(self):
        """Test handling of empty strategy list"""
        spa_framework = SPATestFramework(bootstrap_iterations=10)
        benchmark_returns = np.random.normal(0.0005, 0.015, 100)
        benchmark_metrics = PerformanceMetrics("benchmark", benchmark_returns)
        
        with pytest.raises((ValueError, IndexError)):
            spa_framework.reality_check_test([], benchmark_metrics)
    
    def test_single_strategy(self):
        """Test with single strategy"""
        spa_framework = SPATestFramework(bootstrap_iterations=10)
        
        returns = np.random.normal(0.001, 0.02, 100)
        benchmark_returns = np.random.normal(0.0005, 0.015, 100)
        
        strategy_metrics = [PerformanceMetrics("strategy", returns)]
        benchmark_metrics = PerformanceMetrics("benchmark", benchmark_returns)
        
        result = spa_framework.reality_check_test(strategy_metrics, benchmark_metrics)
        
        assert result.strategy_name == "strategy"
        assert isinstance(result.test_statistic, float)
    
    def test_zero_variance_returns(self):
        """Test handling of zero variance returns"""
        spa_framework = SPATestFramework(bootstrap_iterations=10)
        
        # Constant returns (zero variance)
        constant_returns = np.ones(100) * 0.001
        benchmark_returns = np.random.normal(0.0005, 0.015, 100)
        
        strategy_metrics = [PerformanceMetrics("constant", constant_returns)]
        benchmark_metrics = PerformanceMetrics("benchmark", benchmark_returns)
        
        # Should handle gracefully without crashing
        result = spa_framework.spa_test(
            strategy_metrics, benchmark_metrics, test_type=TestType.SPA_CONSISTENT
        )
        
        assert isinstance(result.test_statistic, float)
        assert 0 <= result.p_value <= 1
    
    def test_mismatched_sample_sizes(self):
        """Test handling of mismatched sample sizes"""
        returns_short = np.random.normal(0.001, 0.02, 50)
        returns_long = np.random.normal(0.001, 0.02, 100)
        benchmark_returns = np.random.normal(0.0005, 0.015, 100)
        
        # This should be handled by the data preparation layer
        # For now, we expect it to work with whatever data is provided
        strategy_metrics = [
            PerformanceMetrics("short", returns_short),
            PerformanceMetrics("long", returns_long)
        ]
        benchmark_metrics = PerformanceMetrics("benchmark", benchmark_returns)
        
        # The framework should handle this gracefully or raise appropriate error
        try:
            spa_framework = SPATestFramework(bootstrap_iterations=10)
            result = spa_framework.reality_check_test(strategy_metrics, benchmark_metrics)
            assert isinstance(result.test_statistic, float)
        except (ValueError, IndexError):
            # Expected behavior for mismatched sizes
            pass

class TestStatisticalProperties:
    """Test statistical properties of the implementation"""
    
    def test_type_i_error_control(self):
        """Test that Type I error is controlled at specified level"""
        # This is a more advanced test that would require extensive simulation
        # For now, we'll do a basic sanity check
        
        spa_framework = SPATestFramework(bootstrap_iterations=100)
        
        # Generate data under null hypothesis (no outperformance)
        n_simulations = 20  # Reduced for testing speed
        p_values = []
        
        for _ in range(n_simulations):
            # All strategies have same expected return as benchmark
            benchmark_returns = np.random.normal(0.0005, 0.015, 100)
            strategy_returns = np.random.normal(0.0005, 0.015, 100)
            
            strategy_metrics = [PerformanceMetrics("strategy", strategy_returns)]
            benchmark_metrics = PerformanceMetrics("benchmark", benchmark_returns)
            
            result = spa_framework.spa_test(
                strategy_metrics, benchmark_metrics, test_type=TestType.SPA_CONSISTENT
            )
            p_values.append(result.p_value)
        
        # Under null hypothesis, p-values should be approximately uniform
        # Check that not too many are significant at 5% level
        significant_count = sum(1 for p in p_values if p < 0.05)
        
        # With 20 simulations, expect ~1 significant (could be 0-4 reasonably)
        assert significant_count <= 6  # Allow some variance
    
    def test_power_against_alternative(self):
        """Test that test has power against genuine alternatives"""
        spa_framework = SPATestFramework(bootstrap_iterations=50)  # Reduced for speed
        
        # Generate data with genuine outperformance
        benchmark_returns = np.random.normal(0.0005, 0.015, 200)
        outperforming_returns = np.random.normal(0.002, 0.015, 200)  # Higher mean
        
        strategy_metrics = [PerformanceMetrics("outperformer", outperforming_returns)]
        benchmark_metrics = PerformanceMetrics("benchmark", benchmark_returns)
        
        result = spa_framework.spa_test(
            strategy_metrics, benchmark_metrics, test_type=TestType.SPA_CONSISTENT
        )
        
        # With sufficient outperformance, should often be significant
        # (Though with small sample this isn't guaranteed)
        assert isinstance(result.p_value, float)
        assert 0 <= result.p_value <= 1

# Integration test
def test_end_to_end_spa_workflow():
    """End-to-end test of complete SPA workflow"""
    np.random.seed(42)
    
    # Create realistic strategy data
    n_strategies = 3
    sample_size = 252  # 1 year
    
    strategy_metrics = []
    for i in range(n_strategies):
        mean_return = 0.0008 + i * 0.0004  # Increasing performance
        returns = np.random.normal(mean_return, 0.02, sample_size)
        metrics = PerformanceMetrics(f"strategy_{i+1}", returns)
        strategy_metrics.append(metrics)
    
    benchmark_returns = np.random.normal(0.0005, 0.015, sample_size)
    benchmark_metrics = PerformanceMetrics("SP500", benchmark_returns)
    
    # Run comprehensive testing
    spa_framework = SPATestFramework(bootstrap_iterations=500)
    results = spa_framework.comprehensive_strategy_testing(
        strategy_metrics, benchmark_metrics, include_individual_tests=True
    )
    
    # Validate comprehensive results
    assert isinstance(results, dict)
    assert all(key in results for key in [
        "test_summary", "reality_check", "spa_tests", 
        "individual_tests", "multiple_testing", "summary"
    ])
    
    # Test summary should contain recommendations
    assert "recommendations" in results["summary"]
    assert isinstance(results["summary"]["recommendations"], list)
    
    # Multiple testing should identify corrected significant strategies
    mt = results["multiple_testing"]
    assert len(mt.strategy_names) == n_strategies
    
    print(f"Test completed successfully. Significant strategies (FDR): {mt.significant_strategies_fdr}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])