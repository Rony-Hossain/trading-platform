# White's Reality Check & SPA Test Framework Guide

## Overview

The Superior Predictive Ability (SPA) test framework provides rigorous statistical validation for trading strategies, controlling for data mining bias and multiple testing problems. This implementation follows White (2000) Reality Check and Hansen (2005) SPA test methodologies.

## Key Features

### 1. Statistical Tests
- **White's Reality Check**: Tests null hypothesis that the best strategy does not outperform benchmark
- **SPA Consistent Test**: More powerful test with consistent p-values under general conditions  
- **SPA Lower/Upper Tests**: Alternative formulations with different power properties
- **Multiple Testing Correction**: Bonferroni and False Discovery Rate (FDR) adjustments

### 2. Bootstrap Methods
- **Stationary Bootstrap**: Handles time series autocorrelation with random block lengths
- **Circular Block Bootstrap**: Fixed block length with wraparound sampling
- **Moving Block Bootstrap**: Non-overlapping blocks for robustness
- **Configurable Block Length**: Automatic or manual selection

### 3. Performance Metrics
- Sharpe ratio, Information ratio, Excess returns
- Maximum drawdown, Win rate, Volatility measures
- Automatic calculation with proper annualization

## Quick Start

```python
from app.statistics.spa_framework import SPATestFramework, PerformanceMetrics

# Create performance metrics
strategy_returns = np.random.normal(0.001, 0.02, 252)
benchmark_returns = np.random.normal(0.0005, 0.015, 252)

strategy_metrics = [PerformanceMetrics("MyStrategy", strategy_returns)]
benchmark_metrics = PerformanceMetrics("SP500", benchmark_returns)

# Initialize framework
spa_framework = SPATestFramework(bootstrap_iterations=10000)

# Run comprehensive testing
results = spa_framework.comprehensive_strategy_testing(
    strategy_metrics, benchmark_metrics, include_individual_tests=True
)

print(f"Reality Check p-value: {results['reality_check'].p_value:.4f}")
print(f"SPA Consistent p-value: {results['spa_tests']['spa_consistent'].p_value:.4f}")
```

## Command Line Usage

```bash
# Validate specific strategies
python scripts/validate_strategy_significance.py \
  --strategies SPY_MOMENTUM,QQQ_MOMENTUM,VIX_MEAN_REVERT \
  --benchmark SP500 \
  --bootstrap-iterations 10000 \
  --test-types reality_check,spa_consistent

# Quick validation with fewer iterations
python scripts/validate_strategy_significance.py \
  --strategies MY_STRATEGY \
  --bootstrap-iterations 1000 \
  --test-types reality_check

# Custom date range
python scripts/validate_strategy_significance.py \
  --strategies MOMENTUM_STRATEGY \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --output-report momentum_validation.json
```

## Interpretation Guidelines

### P-Value Interpretation
- **p < 0.05**: Evidence of significant outperformance at 5% level
- **p < 0.01**: Strong evidence of outperformance at 1% level  
- **p > 0.05**: No significant evidence of outperformance

### Multiple Testing Results
- **Raw p-values**: Individual strategy significance (uncorrected)
- **Bonferroni correction**: Conservative family-wise error rate control
- **FDR correction**: Controls expected proportion of false discoveries
- **Recommended**: Use FDR-corrected results for practical decisions

### Assessment Categories
- **STRONG_EVIDENCE**: Significant after multiple testing correction
- **MODERATE_EVIDENCE**: Significant in individual tests but not after correction
- **NO_EVIDENCE**: No statistically significant outperformance detected

## Bootstrap Configuration

### Optimal Settings
```python
# For daily data (252 observations per year)
spa_framework = SPATestFramework(
    bootstrap_iterations=10000,  # Production setting
    bootstrap_method=BootstrapMethod.STATIONARY,
    block_length=None,  # Auto-select based on sample size
    significance_levels=[0.05, 0.01]
)

# For high-frequency data or larger samples
spa_framework = SPATestFramework(
    bootstrap_iterations=5000,   # Faster for large datasets
    bootstrap_method=BootstrapMethod.CIRCULAR,
    block_length=20,  # Manual selection
)
```

### Block Length Selection
- **Automatic**: `int(sample_size ** (1/3))` - optimal for most cases
- **Manual**: Set based on expected autocorrelation structure
- **Daily data**: 10-30 blocks typically adequate
- **High-frequency**: May need longer blocks (50-100)

## Statistical Properties

### Type I Error Control
- Framework controls Type I error at specified significance level
- Bootstrap provides asymptotically valid p-values
- Multiple testing corrections prevent inflated error rates

### Power Properties
- **Reality Check**: Good power against genuine alternatives
- **SPA Consistent**: Generally most powerful, robust to poor alternatives
- **SPA Lower**: Conservative, good when many strategies tested
- **SPA Upper**: Focus on best-performing strategies only

### Sample Size Requirements
- **Minimum**: 100 observations for basic validity
- **Recommended**: 252+ observations (1+ years daily data)
- **Large sample**: 500+ observations for reliable inference
- **Very large**: 1000+ observations for precise p-values

## Integration with Trading Pipeline

### 1. Strategy Development Phase
```python
# Test strategy during development
validation_results = spa_framework.comprehensive_strategy_testing(
    [candidate_strategy], benchmark, include_individual_tests=False
)

if validation_results["reality_check"].is_significant_95:
    print("Strategy shows promise - proceed with additional testing")
else:
    print("No significant edge detected - revise strategy")
```

### 2. Portfolio Construction
```python
# Test multiple strategies for portfolio inclusion  
all_strategies = [strategy1, strategy2, strategy3, ...]
results = spa_framework.comprehensive_strategy_testing(
    all_strategies, benchmark, include_individual_tests=True
)

# Select strategies that survive multiple testing correction
selected_strategies = results["multiple_testing"]["significant_strategies_fdr"]
print(f"Selected {len(selected_strategies)} strategies for portfolio")
```

### 3. Production Monitoring
```python
# Regular validation of live strategies
monthly_results = spa_framework.reality_check_test(
    live_strategies, benchmark, test_statistic="sharpe_ratio"
)

if monthly_results.p_value > 0.10:
    print("WARNING: Strategy performance deteriorating")
```

## Common Pitfalls and Solutions

### 1. Data Mining Bias
**Problem**: Testing many strategies without proper correction
**Solution**: Always apply multiple testing correction when testing multiple strategies

### 2. Insufficient Sample Size
**Problem**: Unreliable results with small samples
**Solution**: Require minimum 252 observations, prefer 500+ for robust inference

### 3. Look-Ahead Bias
**Problem**: Using future information in strategy development
**Solution**: Integrate with feature contracts and PIT enforcement systems

### 4. Autocorrelation Issues
**Problem**: Standard bootstrap invalid for time series
**Solution**: Use stationary or block bootstrap methods, set appropriate block length

### 5. Benchmark Selection
**Problem**: Inappropriate benchmark leading to misleading results  
**Solution**: Choose relevant benchmark (market index, risk-free rate, etc.)

## Performance Optimization

### 1. Parallel Processing
```python
spa_framework = SPATestFramework(
    bootstrap_iterations=10000,
    n_jobs=8  # Use 8 CPU cores
)
```

### 2. Reduced Iterations for Development
```python
# Fast testing during development
spa_framework = SPATestFramework(bootstrap_iterations=1000)

# Production validation
spa_framework = SPATestFramework(bootstrap_iterations=10000)
```

### 3. Caching Bootstrap Distributions
The framework automatically caches bootstrap distributions to avoid recomputation for similar tests.

## Validation Checklist

Before deploying strategies based on SPA test results:

- [ ] Sample size ≥ 252 observations
- [ ] Bootstrap iterations ≥ 5000 (preferably 10000)
- [ ] Appropriate block length for data frequency
- [ ] Multiple testing correction applied if testing >1 strategy
- [ ] Benchmark properly selected and validated
- [ ] Results interpreted in context of economic significance
- [ ] Strategy not over-fitted to test period
- [ ] Out-of-sample validation performed

## References

1. White, H. (2000). "A Reality Check for Data Snooping." Econometrica, 68(5), 1097-1126.
2. Hansen, P. R. (2005). "A Test for Superior Predictive Ability." Journal of Business & Economic Statistics, 23(4), 365-380.
3. Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap." Journal of the American Statistical Association, 89(428), 1303-1313.

## Support

For technical support or questions about the SPA framework:
- Review test cases in `tests/statistics/test_spa_framework.py`
- Check implementation details in `app/statistics/spa_framework.py`
- Run validation examples in `scripts/validate_strategy_significance.py`