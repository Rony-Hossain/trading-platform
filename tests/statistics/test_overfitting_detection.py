"""
Comprehensive test suite for overfitting detection framework.

Tests Deflated Sharpe Ratio, Probability of Backtest Overfitting (PBO),
and comprehensive overfitting analysis methods.

References:
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: 
  correcting for selection bias, backtest overfitting, and non-normality.
- Bailey, D. H., et al. (2017). The probability of backtest overfitting.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# Add analysis service to path
sys.path.append(str(Path(__file__).parent.parent.parent / "services" / "analysis-service"))

from app.statistics.overfitting_detection import (
    OverfittingDetector,
    DeflatedSharpeResult,
    PBOResult,
    BacktestConfiguration,
    create_backtest_configuration
)


class TestOverfittingDetector:
    """Test suite for OverfittingDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create OverfittingDetector instance for testing."""
        return OverfittingDetector()
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return series for testing."""
        np.random.seed(42)
        n_obs = 252
        returns = np.random.normal(0.001, 0.02, n_obs)
        dates = pd.date_range('2023-01-01', periods=n_obs, freq='D')
        return pd.Series(returns, index=dates)
    
    @pytest.fixture
    def sample_benchmark_returns(self):
        """Generate sample benchmark return series."""
        np.random.seed(43)
        n_obs = 252
        returns = np.random.normal(0.0005, 0.015, n_obs)
        dates = pd.date_range('2023-01-01', periods=n_obs, freq='D')
        return pd.Series(returns, index=dates)
    
    def test_deflated_sharpe_ratio_basic(self, detector, sample_returns):
        """Test basic deflated Sharpe ratio calculation."""
        result = detector.deflated_sharpe_ratio(
            returns=sample_returns,
            n_trials=100,
            n_observations=252
        )
        
        assert isinstance(result, DeflatedSharpeResult)
        assert result.sharpe_ratio is not None
        assert result.deflated_sharpe is not None
        assert result.p_value is not None
        assert 0 <= result.p_value <= 1
        assert result.is_significant_95 in [True, False]
        assert result.variance_inflation_factor >= 1.0
        assert result.expected_max_sharpe >= 0
    
    def test_deflated_sharpe_ratio_with_skewness_kurtosis(self, detector, sample_returns):
        """Test deflated Sharpe with non-normal returns."""
        np.random.seed(44)
        skewed_returns = np.random.gamma(2, 0.01, 252) - 0.02
        skewed_returns = pd.Series(skewed_returns, index=sample_returns.index)
        
        result = detector.deflated_sharpe_ratio(
            returns=skewed_returns,
            n_trials=50,
            n_observations=252
        )
        
        assert result.skewness is not None
        assert result.kurtosis is not None
        assert abs(result.skewness) > 0.1  # Should detect skewness
        assert result.kurtosis > 3.0  # Should detect excess kurtosis
    
    def test_deflated_sharpe_ratio_with_benchmark(self, detector, sample_returns, sample_benchmark_returns):
        """Test deflated Sharpe with benchmark."""
        result = detector.deflated_sharpe_ratio(
            returns=sample_returns,
            benchmark_returns=sample_benchmark_returns,
            n_trials=50,
            n_observations=252
        )
        
        assert result.sharpe_ratio is not None
        assert result.deflated_sharpe is not None
        assert result.excess_returns_mean is not None
        assert result.excess_returns_std is not None
    
    def test_deflated_sharpe_ratio_edge_cases(self, detector):
        """Test edge cases for deflated Sharpe ratio."""
        np.random.seed(45)
        
        # Test with zero volatility
        zero_vol_returns = pd.Series([0.001] * 100)
        result = detector.deflated_sharpe_ratio(zero_vol_returns, n_trials=10, n_observations=100)
        assert np.isinf(result.sharpe_ratio) or np.isnan(result.sharpe_ratio)
        
        # Test with negative mean returns
        negative_returns = pd.Series(np.random.normal(-0.001, 0.02, 100))
        result = detector.deflated_sharpe_ratio(negative_returns, n_trials=10, n_observations=100)
        assert result.sharpe_ratio < 0
    
    def test_probability_backtest_overfitting_basic(self, detector):
        """Test basic PBO calculation."""
        np.random.seed(46)
        n_strategies = 20
        n_obs = 252
        
        strategy_returns = {}
        for i in range(n_strategies):
            returns = np.random.normal(0.0005 + i*0.0001, 0.02, n_obs)
            dates = pd.date_range('2023-01-01', periods=n_obs, freq='D')
            strategy_returns[f'strategy_{i}'] = pd.Series(returns, index=dates)
        
        result = detector.probability_backtest_overfitting(
            strategy_returns=strategy_returns,
            n_splits=5
        )
        
        assert isinstance(result, PBOResult)
        assert 0 <= result.pbo_probability <= 1
        assert result.n_strategies == n_strategies
        assert result.n_splits == 5
        assert len(result.is_rank_degradation) == 5
        assert result.median_rank_degradation is not None
        assert result.performance_degradation is not None
    
    def test_probability_backtest_overfitting_perfect_strategies(self, detector):
        """Test PBO with strategies that maintain performance."""
        np.random.seed(47)
        n_strategies = 10
        n_obs = 252
        
        strategy_returns = {}
        for i in range(n_strategies):
            returns = np.random.normal(0.002, 0.01, n_obs)  # All good strategies
            dates = pd.date_range('2023-01-01', periods=n_obs, freq='D')
            strategy_returns[f'strategy_{i}'] = pd.Series(returns, index=dates)
        
        result = detector.probability_backtest_overfitting(
            strategy_returns=strategy_returns,
            n_splits=3
        )
        
        # Should have low PBO for consistently good strategies
        assert result.pbo_probability <= 0.5
    
    def test_probability_backtest_overfitting_overfitted_strategies(self, detector):
        """Test PBO with clearly overfitted strategies."""
        np.random.seed(48)
        n_strategies = 15
        n_obs = 252
        
        strategy_returns = {}
        dates = pd.date_range('2023-01-01', periods=n_obs, freq='D')
        
        for i in range(n_strategies):
            if i < 5:
                # First 5 strategies: good in first half, poor in second half
                first_half = np.random.normal(0.003, 0.015, n_obs//2)
                second_half = np.random.normal(-0.001, 0.025, n_obs//2)
                returns = np.concatenate([first_half, second_half])
            else:
                # Rest: consistently poor
                returns = np.random.normal(-0.0005, 0.02, n_obs)
            
            strategy_returns[f'strategy_{i}'] = pd.Series(returns, index=dates)
        
        result = detector.probability_backtest_overfitting(
            strategy_returns=strategy_returns,
            n_splits=4
        )
        
        # Should have high PBO for overfitted strategies
        assert result.pbo_probability >= 0.3
    
    def test_comprehensive_overfitting_analysis(self, detector):
        """Test comprehensive analysis combining all methods."""
        np.random.seed(49)
        n_strategies = 10
        n_obs = 252
        
        strategy_returns = {}
        dates = pd.date_range('2023-01-01', periods=n_obs, freq='D')
        
        for i in range(n_strategies):
            returns = np.random.normal(0.001 + i*0.0002, 0.02, n_obs)
            strategy_returns[f'strategy_{i}'] = pd.Series(returns, index=dates)
        
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, n_obs), 
            index=dates
        )
        
        results = detector.comprehensive_overfitting_analysis(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            n_trials=50,
            n_pbo_splits=4
        )
        
        assert 'deflated_sharpe_results' in results
        assert 'pbo_result' in results
        assert 'risk_assessment' in results
        assert 'recommendations' in results
        
        assert len(results['deflated_sharpe_results']) == n_strategies
        assert isinstance(results['pbo_result'], PBOResult)
        
        risk_assessment = results['risk_assessment']
        assert 'overall_risk_level' in risk_assessment
        assert risk_assessment['overall_risk_level'] in ['LOW', 'MEDIUM', 'HIGH']
        assert 'individual_strategy_risks' in risk_assessment
        
        recommendations = results['recommendations']
        assert 'primary_recommendation' in recommendations
        assert 'specific_actions' in recommendations
        assert isinstance(recommendations['specific_actions'], list)
    
    def test_risk_assessment_logic(self, detector):
        """Test risk assessment logic with controlled inputs."""
        # Test HIGH risk scenario
        high_risk_ds_results = [
            MagicMock(is_significant_95=False, p_value=0.8, deflated_sharpe=-1.0),
            MagicMock(is_significant_95=False, p_value=0.9, deflated_sharpe=-0.5),
        ]
        high_risk_pbo = MagicMock(pbo_probability=0.9, performance_degradation=0.8)
        
        risk_assessment = detector._assess_overfitting_risk(
            high_risk_ds_results, high_risk_pbo
        )
        
        assert risk_assessment['overall_risk_level'] == 'HIGH'
        assert risk_assessment['high_pbo_probability'] is True
        assert risk_assessment['multiple_non_significant_strategies'] is True
        
        # Test LOW risk scenario
        low_risk_ds_results = [
            MagicMock(is_significant_95=True, p_value=0.01, deflated_sharpe=2.0),
            MagicMock(is_significant_95=True, p_value=0.02, deflated_sharpe=1.8),
        ]
        low_risk_pbo = MagicMock(pbo_probability=0.1, performance_degradation=0.1)
        
        risk_assessment = detector._assess_overfitting_risk(
            low_risk_ds_results, low_risk_pbo
        )
        
        assert risk_assessment['overall_risk_level'] == 'LOW'
        assert risk_assessment['high_pbo_probability'] is False
        assert risk_assessment['multiple_non_significant_strategies'] is False


class TestUtilityFunctions:
    """Test utility functions for overfitting detection."""
    
    def test_create_backtest_configuration(self):
        """Test backtest configuration creation."""
        config = create_backtest_configuration(
            strategy_name="test_strategy",
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-12-31'),
            test_start=pd.Timestamp('2021-01-01'),
            test_end=pd.Timestamp('2021-03-31')
        )
        
        assert isinstance(config, BacktestConfiguration)
        assert config.strategy_name == "test_strategy"
        assert config.train_start < config.train_end
        assert config.test_start < config.test_end
        assert config.train_end <= config.test_start


class TestDataClassesSerialization:
    """Test data classes can be properly serialized."""
    
    def test_deflated_sharpe_result_serialization(self):
        """Test DeflatedSharpeResult can be converted to dict."""
        result = DeflatedSharpeResult(
            sharpe_ratio=1.5,
            deflated_sharpe=0.8,
            p_value=0.05,
            is_significant_95=True,
            variance_inflation_factor=1.2,
            expected_max_sharpe=0.9,
            skewness=-0.1,
            kurtosis=3.2,
            excess_returns_mean=0.001,
            excess_returns_std=0.02
        )
        
        result_dict = asdict(result)
        assert 'sharpe_ratio' in result_dict
        assert 'deflated_sharpe' in result_dict
        assert 'p_value' in result_dict
        assert result_dict['is_significant_95'] is True
    
    def test_pbo_result_serialization(self):
        """Test PBOResult can be converted to dict."""
        result = PBOResult(
            pbo_probability=0.6,
            n_strategies=10,
            n_splits=5,
            is_rank_degradation=[True, False, True, False, True],
            median_rank_degradation=0.3,
            performance_degradation=0.25
        )
        
        result_dict = asdict(result)
        assert 'pbo_probability' in result_dict
        assert 'n_strategies' in result_dict
        assert 'is_rank_degradation' in result_dict
        assert len(result_dict['is_rank_degradation']) == 5
    
    def test_backtest_configuration_serialization(self):
        """Test BacktestConfiguration can be converted to dict."""
        config = BacktestConfiguration(
            strategy_name="test_strategy",
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-12-31'),
            test_start=pd.Timestamp('2021-01-01'),
            test_end=pd.Timestamp('2021-03-31')
        )
        
        config_dict = asdict(config)
        assert 'strategy_name' in config_dict
        assert 'train_start' in config_dict
        assert 'train_end' in config_dict
        assert 'test_start' in config_dict
        assert 'test_end' in config_dict


class TestStatisticalProperties:
    """Test statistical properties of the overfitting detection methods."""
    
    def test_deflated_sharpe_type_i_error_control(self):
        """Test that deflated Sharpe controls Type I error under null."""
        detector = OverfittingDetector()
        np.random.seed(50)
        
        n_simulations = 100
        n_trials = 50
        false_positives = 0
        
        for _ in range(n_simulations):
            # Generate returns under null hypothesis (no skill)
            returns = pd.Series(np.random.normal(0, 0.02, 252))
            
            result = detector.deflated_sharpe_ratio(
                returns=returns,
                n_trials=n_trials,
                n_observations=252
            )
            
            if result.is_significant_95:
                false_positives += 1
        
        type_i_error_rate = false_positives / n_simulations
        
        # Should be close to 5% under null hypothesis
        assert 0.01 <= type_i_error_rate <= 0.15  # Allow some Monte Carlo variation
    
    def test_pbo_identifies_overfitting(self):
        """Test that PBO correctly identifies overfitted strategies."""
        detector = OverfittingDetector()
        np.random.seed(51)
        
        # Create clearly overfitted strategies
        n_strategies = 20
        n_obs = 500
        dates = pd.date_range('2020-01-01', periods=n_obs, freq='D')
        
        strategy_returns = {}
        split_point = n_obs // 2
        
        for i in range(n_strategies):
            # Good performance in first half, poor in second half
            first_half = np.random.normal(0.004, 0.015, split_point)
            second_half = np.random.normal(-0.002, 0.025, n_obs - split_point)
            returns = np.concatenate([first_half, second_half])
            
            strategy_returns[f'overfitted_{i}'] = pd.Series(returns, index=dates)
        
        result = detector.probability_backtest_overfitting(
            strategy_returns=strategy_returns,
            n_splits=4
        )
        
        # Should detect high probability of overfitting
        assert result.pbo_probability >= 0.5
        assert result.performance_degradation >= 0.3
    
    def test_deflated_sharpe_variance_inflation(self):
        """Test variance inflation factor calculation."""
        detector = OverfittingDetector()
        np.random.seed(52)
        
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # Test with different trial counts
        result_few_trials = detector.deflated_sharpe_ratio(
            returns=returns, n_trials=10, n_observations=252
        )
        
        result_many_trials = detector.deflated_sharpe_ratio(
            returns=returns, n_trials=1000, n_observations=252
        )
        
        # VIF should increase with more trials
        assert result_many_trials.variance_inflation_factor > result_few_trials.variance_inflation_factor
        
        # Expected max Sharpe should increase with more trials
        assert result_many_trials.expected_max_sharpe > result_few_trials.expected_max_sharpe


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])