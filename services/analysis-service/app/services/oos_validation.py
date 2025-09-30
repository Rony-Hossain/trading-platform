"""
Out-of-Sample (OOS) Strategy Validation Framework

This module implements comprehensive out-of-sample validation for trading strategies
with rigorous statistical testing to prevent overfitting and ensure robust performance
before production deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import asyncpg
from ..core.database import get_database_url
from .car_analysis import EventType, Sector, EventData, CARResults

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"

class BenchmarkType(Enum):
    """Benchmark strategy types"""
    BUY_AND_HOLD = "buy_and_hold"
    SIMPLE_FACTOR = "simple_factor"
    MARKET_INDEX = "market_index"
    RISK_PARITY = "risk_parity"
    MOMENTUM = "momentum"

@dataclass
class ValidationThresholds:
    """OOS validation performance thresholds"""
    min_sharpe_ratio: float = 1.0
    min_t_statistic: float = 2.0
    min_hit_rate: float = 0.55
    max_drawdown_threshold: float = 0.15
    min_calmar_ratio: float = 0.5
    min_information_ratio: float = 0.3
    min_validation_period_months: int = 6
    min_trades: int = 20
    significance_level: float = 0.05

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    hit_rate: float
    total_trades: int
    average_holding_period: float
    profit_factor: float
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk
    daily_returns: np.ndarray
    cumulative_returns: np.ndarray
    trade_returns: List[float]

@dataclass
class BenchmarkPerformance:
    """Benchmark performance for comparison"""
    benchmark_type: BenchmarkType
    performance_metrics: StrategyPerformance
    correlation_with_strategy: float
    beta_to_benchmark: float
    tracking_error: float

@dataclass
class StatisticalTestResults:
    """Statistical significance test results"""
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_interval_lower: float
    confidence_interval_upper: float
    degrees_of_freedom: int
    test_type: str
    null_hypothesis: str

@dataclass
class ValidationResults:
    """Comprehensive OOS validation results"""
    strategy_id: str
    validation_status: ValidationStatus
    oos_period_start: datetime
    oos_period_end: datetime
    strategy_performance: StrategyPerformance
    benchmark_performances: List[BenchmarkPerformance]
    statistical_tests: List[StatisticalTestResults]
    information_ratio: float
    excess_return_t_stat: float
    overfitting_score: float
    validation_summary: Dict[str, Any]
    recommendations: List[str]
    risk_warnings: List[str]

class OOSValidator:
    """Out-of-Sample validation engine"""
    
    def __init__(self, thresholds: Optional[ValidationThresholds] = None):
        self.thresholds = thresholds or ValidationThresholds()
        self.trading_days_per_year = 252
        
    async def validate_strategy(
        self,
        strategy_id: str,
        strategy_signals: pd.DataFrame,
        price_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        oos_start_date: datetime,
        validation_period_months: int = 12
    ) -> ValidationResults:
        """
        Comprehensive OOS validation of a trading strategy
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_signals: DataFrame with columns: date, symbol, signal, confidence
            price_data: Historical price data
            benchmark_data: Benchmark price data
            oos_start_date: Start date for OOS period
            validation_period_months: Length of validation period
            
        Returns:
            ValidationResults with comprehensive analysis
        """
        logger.info(f"Starting OOS validation for strategy {strategy_id}")
        
        # Define validation period
        oos_end_date = oos_start_date + timedelta(days=validation_period_months * 30)
        
        # Filter data to OOS period
        oos_signals = strategy_signals[
            (strategy_signals['date'] >= oos_start_date) & 
            (strategy_signals['date'] <= oos_end_date)
        ].copy()
        
        oos_prices = price_data[
            (price_data['date'] >= oos_start_date) & 
            (price_data['date'] <= oos_end_date)
        ].copy()
        
        oos_benchmark = benchmark_data[
            (benchmark_data['date'] >= oos_start_date) & 
            (benchmark_data['date'] <= oos_end_date)
        ].copy()
        
        if len(oos_signals) < self.thresholds.min_trades:
            return ValidationResults(
                strategy_id=strategy_id,
                validation_status=ValidationStatus.FAILED,
                oos_period_start=oos_start_date,
                oos_period_end=oos_end_date,
                strategy_performance=None,
                benchmark_performances=[],
                statistical_tests=[],
                information_ratio=0.0,
                excess_return_t_stat=0.0,
                overfitting_score=1.0,
                validation_summary={"error": "Insufficient trades for validation"},
                recommendations=["Increase signal frequency or extend validation period"],
                risk_warnings=["Strategy has insufficient trading activity"]
            )
        
        # Calculate strategy performance
        strategy_performance = await self._calculate_strategy_performance(
            oos_signals, oos_prices
        )
        
        # Calculate benchmark performances
        benchmark_performances = await self._calculate_benchmark_performances(
            oos_benchmark, oos_prices, strategy_performance
        )
        
        # Statistical significance testing
        statistical_tests = await self._perform_statistical_tests(
            strategy_performance, benchmark_performances
        )
        
        # Information ratio calculation
        information_ratio = self._calculate_information_ratio(
            strategy_performance, benchmark_performances[0] if benchmark_performances else None
        )
        
        # Excess return t-statistic
        excess_return_t_stat = self._calculate_excess_return_t_stat(
            strategy_performance, benchmark_performances[0] if benchmark_performances else None
        )
        
        # Overfitting assessment
        overfitting_score = await self._assess_overfitting_risk(
            strategy_performance, validation_period_months
        )
        
        # Determine validation status
        validation_status = self._determine_validation_status(
            strategy_performance, statistical_tests, information_ratio, 
            excess_return_t_stat, overfitting_score
        )
        
        # Generate summary and recommendations
        validation_summary = self._generate_validation_summary(
            strategy_performance, benchmark_performances, statistical_tests
        )
        
        recommendations = self._generate_recommendations(
            strategy_performance, validation_status, statistical_tests
        )
        
        risk_warnings = self._generate_risk_warnings(
            strategy_performance, overfitting_score, statistical_tests
        )
        
        results = ValidationResults(
            strategy_id=strategy_id,
            validation_status=validation_status,
            oos_period_start=oos_start_date,
            oos_period_end=oos_end_date,
            strategy_performance=strategy_performance,
            benchmark_performances=benchmark_performances,
            statistical_tests=statistical_tests,
            information_ratio=information_ratio,
            excess_return_t_stat=excess_return_t_stat,
            overfitting_score=overfitting_score,
            validation_summary=validation_summary,
            recommendations=recommendations,
            risk_warnings=risk_warnings
        )
        
        # Persist validation results
        await self._persist_validation_results(results)
        
        logger.info(f"OOS validation complete for {strategy_id}: {validation_status.value}")
        return results
    
    async def _calculate_strategy_performance(
        self, 
        signals: pd.DataFrame, 
        prices: pd.DataFrame
    ) -> StrategyPerformance:
        """Calculate comprehensive strategy performance metrics"""
        
        # Merge signals with prices
        merged_data = signals.merge(
            prices, on=['date', 'symbol'], how='left'
        ).sort_values(['date', 'symbol'])
        
        # Calculate position returns
        merged_data['next_return'] = merged_data.groupby('symbol')['return'].shift(-1)
        merged_data['position_return'] = merged_data['signal'] * merged_data['next_return']
        
        # Daily portfolio returns (equal weight across positions)
        daily_returns = merged_data.groupby('date')['position_return'].mean()
        daily_returns = daily_returns.fillna(0)
        
        # Cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # Trade-level returns
        trade_returns = merged_data['position_return'].dropna().tolist()
        
        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(daily_returns)) - 1
        annualized_volatility = daily_returns.std() * np.sqrt(self.trading_days_per_year)
        
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Maximum drawdown
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Hit rate
        winning_trades = sum(1 for ret in trade_returns if ret > 0)
        hit_rate = winning_trades / len(trade_returns) if trade_returns else 0
        
        # Average holding period (assuming daily rebalancing)
        average_holding_period = 1.0  # Daily signals
        
        # Profit factor
        gross_profit = sum(ret for ret in trade_returns if ret > 0)
        gross_loss = abs(sum(ret for ret in trade_returns if ret < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Higher moments
        skewness = stats.skew(daily_returns)
        kurtosis = stats.kurtosis(daily_returns)
        
        # Risk metrics
        var_95 = np.percentile(daily_returns, 5)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        
        return StrategyPerformance(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            total_trades=len(trade_returns),
            average_holding_period=average_holding_period,
            profit_factor=profit_factor,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            daily_returns=daily_returns.values,
            cumulative_returns=cumulative_returns.values,
            trade_returns=trade_returns
        )
    
    async def _calculate_benchmark_performances(
        self,
        benchmark_data: pd.DataFrame,
        price_data: pd.DataFrame,
        strategy_performance: StrategyPerformance
    ) -> List[BenchmarkPerformance]:
        """Calculate performance for various benchmark strategies"""
        
        benchmarks = []
        
        # Buy and Hold benchmark
        buy_hold_returns = benchmark_data['return'].fillna(0)
        buy_hold_perf = self._calculate_performance_from_returns(buy_hold_returns)
        
        correlation = np.corrcoef(strategy_performance.daily_returns, buy_hold_returns)[0, 1]
        beta = np.cov(strategy_performance.daily_returns, buy_hold_returns)[0, 1] / np.var(buy_hold_returns)
        tracking_error = np.std(strategy_performance.daily_returns - buy_hold_returns) * np.sqrt(self.trading_days_per_year)
        
        benchmarks.append(BenchmarkPerformance(
            benchmark_type=BenchmarkType.BUY_AND_HOLD,
            performance_metrics=buy_hold_perf,
            correlation_with_strategy=correlation,
            beta_to_benchmark=beta,
            tracking_error=tracking_error
        ))
        
        # Simple momentum benchmark (if sufficient data)
        if len(price_data) >= 21:  # Need at least 21 days for momentum
            momentum_signals = self._generate_momentum_signals(price_data)
            momentum_returns = self._calculate_strategy_returns_from_signals(momentum_signals, price_data)
            momentum_perf = self._calculate_performance_from_returns(momentum_returns)
            
            mom_correlation = np.corrcoef(strategy_performance.daily_returns, momentum_returns)[0, 1]
            mom_beta = np.cov(strategy_performance.daily_returns, momentum_returns)[0, 1] / np.var(momentum_returns)
            mom_tracking_error = np.std(strategy_performance.daily_returns - momentum_returns) * np.sqrt(self.trading_days_per_year)
            
            benchmarks.append(BenchmarkPerformance(
                benchmark_type=BenchmarkType.MOMENTUM,
                performance_metrics=momentum_perf,
                correlation_with_strategy=mom_correlation,
                beta_to_benchmark=mom_beta,
                tracking_error=mom_tracking_error
            ))
        
        return benchmarks
    
    def _calculate_performance_from_returns(self, returns: pd.Series) -> StrategyPerformance:
        """Calculate performance metrics from return series"""
        
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(self.trading_days_per_year)
        
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        hit_rate = sum(1 for ret in returns if ret > 0) / len(returns) if len(returns) > 0 else 0
        
        return StrategyPerformance(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            total_trades=len(returns),
            average_holding_period=1.0,
            profit_factor=0.0,
            skewness=stats.skew(returns),
            kurtosis=stats.kurtosis(returns),
            var_95=np.percentile(returns, 5),
            cvar_95=returns[returns <= np.percentile(returns, 5)].mean(),
            daily_returns=returns.values,
            cumulative_returns=cumulative_returns.values,
            trade_returns=returns.tolist()
        )
    
    def _generate_momentum_signals(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate simple momentum signals as benchmark"""
        
        signals = []
        for symbol in price_data['symbol'].unique():
            symbol_data = price_data[price_data['symbol'] == symbol].sort_values('date')
            symbol_data['momentum_20'] = symbol_data['price'].rolling(20).mean()
            symbol_data['signal'] = np.where(
                symbol_data['price'] > symbol_data['momentum_20'], 1, -1
            )
            
            for _, row in symbol_data.iterrows():
                if not pd.isna(row['signal']):
                    signals.append({
                        'date': row['date'],
                        'symbol': row['symbol'],
                        'signal': row['signal'],
                        'confidence': 0.5
                    })
        
        return pd.DataFrame(signals)
    
    def _calculate_strategy_returns_from_signals(
        self, 
        signals: pd.DataFrame, 
        prices: pd.DataFrame
    ) -> pd.Series:
        """Calculate strategy returns from signals and prices"""
        
        merged_data = signals.merge(prices, on=['date', 'symbol'], how='left')
        merged_data['next_return'] = merged_data.groupby('symbol')['return'].shift(-1)
        merged_data['position_return'] = merged_data['signal'] * merged_data['next_return']
        
        daily_returns = merged_data.groupby('date')['position_return'].mean().fillna(0)
        return daily_returns
    
    async def _perform_statistical_tests(
        self,
        strategy_performance: StrategyPerformance,
        benchmark_performances: List[BenchmarkPerformance]
    ) -> List[StatisticalTestResults]:
        """Perform statistical significance tests"""
        
        tests = []
        
        # Test against zero (absolute performance)
        strategy_returns = strategy_performance.daily_returns
        t_stat_zero, p_val_zero = stats.ttest_1samp(strategy_returns, 0)
        
        tests.append(StatisticalTestResults(
            t_statistic=t_stat_zero,
            p_value=p_val_zero,
            is_significant=p_val_zero < self.thresholds.significance_level,
            confidence_interval_lower=np.percentile(strategy_returns, 2.5),
            confidence_interval_upper=np.percentile(strategy_returns, 97.5),
            degrees_of_freedom=len(strategy_returns) - 1,
            test_type="One-sample t-test vs zero",
            null_hypothesis="Strategy returns equal zero"
        ))
        
        # Test against benchmarks
        for benchmark in benchmark_performances:
            if len(benchmark.performance_metrics.daily_returns) == len(strategy_returns):
                excess_returns = strategy_returns - benchmark.performance_metrics.daily_returns
                t_stat_bench, p_val_bench = stats.ttest_1samp(excess_returns, 0)
                
                tests.append(StatisticalTestResults(
                    t_statistic=t_stat_bench,
                    p_value=p_val_bench,
                    is_significant=p_val_bench < self.thresholds.significance_level,
                    confidence_interval_lower=np.percentile(excess_returns, 2.5),
                    confidence_interval_upper=np.percentile(excess_returns, 97.5),
                    degrees_of_freedom=len(excess_returns) - 1,
                    test_type=f"Paired t-test vs {benchmark.benchmark_type.value}",
                    null_hypothesis=f"No excess return over {benchmark.benchmark_type.value}"
                ))
        
        return tests
    
    def _calculate_information_ratio(
        self,
        strategy_performance: StrategyPerformance,
        benchmark_performance: Optional[BenchmarkPerformance]
    ) -> float:
        """Calculate information ratio vs benchmark"""
        
        if not benchmark_performance:
            return 0.0
        
        strategy_returns = strategy_performance.daily_returns
        benchmark_returns = benchmark_performance.performance_metrics.daily_returns
        
        if len(strategy_returns) != len(benchmark_returns):
            return 0.0
        
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
        
        if tracking_error == 0:
            return 0.0
        
        excess_return = np.mean(excess_returns) * self.trading_days_per_year
        return excess_return / tracking_error
    
    def _calculate_excess_return_t_stat(
        self,
        strategy_performance: StrategyPerformance,
        benchmark_performance: Optional[BenchmarkPerformance]
    ) -> float:
        """Calculate t-statistic for excess returns"""
        
        if not benchmark_performance:
            return 0.0
        
        strategy_returns = strategy_performance.daily_returns
        benchmark_returns = benchmark_performance.performance_metrics.daily_returns
        
        if len(strategy_returns) != len(benchmark_returns):
            return 0.0
        
        excess_returns = strategy_returns - benchmark_returns
        t_stat, _ = stats.ttest_1samp(excess_returns, 0)
        
        return t_stat
    
    async def _assess_overfitting_risk(
        self,
        strategy_performance: StrategyPerformance,
        validation_period_months: int
    ) -> float:
        """Assess overfitting risk based on various factors"""
        
        # Factors that increase overfitting risk
        overfitting_score = 0.0
        
        # High Sharpe ratio may indicate overfitting
        if strategy_performance.sharpe_ratio > 3.0:
            overfitting_score += 0.3
        elif strategy_performance.sharpe_ratio > 2.0:
            overfitting_score += 0.1
        
        # Very high hit rate may indicate overfitting
        if strategy_performance.hit_rate > 0.8:
            overfitting_score += 0.2
        elif strategy_performance.hit_rate > 0.7:
            overfitting_score += 0.1
        
        # Low number of trades increases overfitting risk
        if strategy_performance.total_trades < 50:
            overfitting_score += 0.3
        elif strategy_performance.total_trades < 100:
            overfitting_score += 0.1
        
        # Short validation period increases risk
        if validation_period_months < 6:
            overfitting_score += 0.3
        elif validation_period_months < 12:
            overfitting_score += 0.1
        
        # Extreme skewness may indicate overfitting
        if abs(strategy_performance.skewness) > 2.0:
            overfitting_score += 0.1
        
        return min(overfitting_score, 1.0)  # Cap at 1.0
    
    def _determine_validation_status(
        self,
        strategy_performance: StrategyPerformance,
        statistical_tests: List[StatisticalTestResults],
        information_ratio: float,
        excess_return_t_stat: float,
        overfitting_score: float
    ) -> ValidationStatus:
        """Determine overall validation status"""
        
        # Check minimum requirements
        requirements_met = []
        
        # Sharpe ratio requirement
        requirements_met.append(strategy_performance.sharpe_ratio >= self.thresholds.min_sharpe_ratio)
        
        # T-statistic requirement (check against best benchmark)
        requirements_met.append(abs(excess_return_t_stat) >= self.thresholds.min_t_statistic)
        
        # Hit rate requirement
        requirements_met.append(strategy_performance.hit_rate >= self.thresholds.min_hit_rate)
        
        # Drawdown requirement
        requirements_met.append(strategy_performance.max_drawdown <= self.thresholds.max_drawdown_threshold)
        
        # Information ratio requirement
        requirements_met.append(information_ratio >= self.thresholds.min_information_ratio)
        
        # Statistical significance requirement
        significant_tests = sum(1 for test in statistical_tests if test.is_significant)
        requirements_met.append(significant_tests > 0)
        
        # Minimum trades requirement
        requirements_met.append(strategy_performance.total_trades >= self.thresholds.min_trades)
        
        # Check overfitting risk
        low_overfitting_risk = overfitting_score <= 0.5
        
        # Determine status
        if all(requirements_met) and low_overfitting_risk:
            return ValidationStatus.PASSED
        elif sum(requirements_met) >= len(requirements_met) * 0.7 and overfitting_score <= 0.7:
            return ValidationStatus.REQUIRES_REVIEW
        else:
            return ValidationStatus.FAILED
    
    def _generate_validation_summary(
        self,
        strategy_performance: StrategyPerformance,
        benchmark_performances: List[BenchmarkPerformance],
        statistical_tests: List[StatisticalTestResults]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        
        summary = {
            "strategy_metrics": {
                "annualized_return": strategy_performance.annualized_return,
                "sharpe_ratio": strategy_performance.sharpe_ratio,
                "max_drawdown": strategy_performance.max_drawdown,
                "hit_rate": strategy_performance.hit_rate,
                "total_trades": strategy_performance.total_trades,
                "calmar_ratio": strategy_performance.calmar_ratio
            },
            "benchmark_comparison": {},
            "statistical_significance": {
                "significant_tests": sum(1 for test in statistical_tests if test.is_significant),
                "total_tests": len(statistical_tests),
                "best_t_statistic": max([test.t_statistic for test in statistical_tests], default=0)
            }
        }
        
        # Benchmark comparisons
        for benchmark in benchmark_performances:
            summary["benchmark_comparison"][benchmark.benchmark_type.value] = {
                "excess_return": strategy_performance.annualized_return - benchmark.performance_metrics.annualized_return,
                "excess_sharpe": strategy_performance.sharpe_ratio - benchmark.performance_metrics.sharpe_ratio,
                "correlation": benchmark.correlation_with_strategy,
                "tracking_error": benchmark.tracking_error
            }
        
        return summary
    
    def _generate_recommendations(
        self,
        strategy_performance: StrategyPerformance,
        validation_status: ValidationStatus,
        statistical_tests: List[StatisticalTestResults]
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        if validation_status == ValidationStatus.PASSED:
            recommendations.append("Strategy passes all validation criteria and is approved for production deployment")
            recommendations.append("Consider gradual position sizing ramp-up to validate live performance")
            
        elif validation_status == ValidationStatus.REQUIRES_REVIEW:
            recommendations.append("Strategy shows promise but requires manual review before deployment")
            
            if strategy_performance.sharpe_ratio < self.thresholds.min_sharpe_ratio:
                recommendations.append(f"Improve risk-adjusted returns (current Sharpe: {strategy_performance.sharpe_ratio:.2f}, required: {self.thresholds.min_sharpe_ratio})")
            
            if strategy_performance.max_drawdown > self.thresholds.max_drawdown_threshold:
                recommendations.append(f"Reduce maximum drawdown (current: {strategy_performance.max_drawdown:.2%}, threshold: {self.thresholds.max_drawdown_threshold:.2%})")
                
        else:  # FAILED
            recommendations.append("Strategy fails validation criteria and should not be deployed")
            recommendations.append("Consider strategy refinement or additional training data")
            
            if not any(test.is_significant for test in statistical_tests):
                recommendations.append("Results are not statistically significant - strategy may not have genuine alpha")
        
        return recommendations
    
    def _generate_risk_warnings(
        self,
        strategy_performance: StrategyPerformance,
        overfitting_score: float,
        statistical_tests: List[StatisticalTestResults]
    ) -> List[str]:
        """Generate risk warnings"""
        
        warnings = []
        
        if overfitting_score > 0.7:
            warnings.append("HIGH OVERFITTING RISK: Strategy may not perform well in live trading")
        elif overfitting_score > 0.5:
            warnings.append("MODERATE OVERFITTING RISK: Monitor live performance closely")
        
        if strategy_performance.total_trades < 50:
            warnings.append("LOW SAMPLE SIZE: Results based on limited number of trades")
        
        if abs(strategy_performance.skewness) > 2.0:
            warnings.append("EXTREME SKEWNESS: Return distribution may have fat tails")
        
        if strategy_performance.max_drawdown > 0.2:
            warnings.append("HIGH DRAWDOWN RISK: Strategy experiences large losses")
        
        if not any(test.is_significant for test in statistical_tests):
            warnings.append("NO STATISTICAL SIGNIFICANCE: Results may be due to chance")
        
        return warnings
    
    async def _persist_validation_results(self, results: ValidationResults):
        """Persist validation results to database"""
        
        try:
            conn = await asyncpg.connect(get_database_url())
            
            # Create validation results table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS oos_validation_results (
                    id SERIAL PRIMARY KEY,
                    strategy_id VARCHAR(255) NOT NULL,
                    validation_status VARCHAR(50) NOT NULL,
                    oos_period_start TIMESTAMP NOT NULL,
                    oos_period_end TIMESTAMP NOT NULL,
                    annualized_return FLOAT,
                    sharpe_ratio FLOAT,
                    max_drawdown FLOAT,
                    information_ratio FLOAT,
                    excess_return_t_stat FLOAT,
                    overfitting_score FLOAT,
                    validation_summary JSONB,
                    recommendations TEXT[],
                    risk_warnings TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX(strategy_id, created_at)
                )
            """)
            
            # Insert validation results
            await conn.execute("""
                INSERT INTO oos_validation_results 
                (strategy_id, validation_status, oos_period_start, oos_period_end,
                 annualized_return, sharpe_ratio, max_drawdown, information_ratio,
                 excess_return_t_stat, overfitting_score, validation_summary,
                 recommendations, risk_warnings)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                results.strategy_id,
                results.validation_status.value,
                results.oos_period_start,
                results.oos_period_end,
                results.strategy_performance.annualized_return,
                results.strategy_performance.sharpe_ratio,
                results.strategy_performance.max_drawdown,
                results.information_ratio,
                results.excess_return_t_stat,
                results.overfitting_score,
                results.validation_summary,
                results.recommendations,
                results.risk_warnings
            )
            
            await conn.close()
            logger.info(f"Validation results persisted for strategy {results.strategy_id}")
            
        except Exception as e:
            logger.error(f"Failed to persist validation results: {e}")

# Factory functions
async def create_oos_validator(thresholds: Optional[ValidationThresholds] = None) -> OOSValidator:
    """Create OOS validator with custom thresholds"""
    return OOSValidator(thresholds)

# Validation enforcement decorator
def require_oos_validation(min_sharpe: float = 1.0, min_t_stat: float = 2.0):
    """Decorator to enforce OOS validation before strategy deployment"""
    
    def decorator(strategy_deploy_func):
        async def wrapper(strategy_id: str, *args, **kwargs):
            # Check if strategy has passed validation
            conn = await asyncpg.connect(get_database_url())
            
            validation_result = await conn.fetchrow("""
                SELECT validation_status, created_at 
                FROM oos_validation_results 
                WHERE strategy_id = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            """, strategy_id)
            
            await conn.close()
            
            if not validation_result:
                raise ValueError(f"Strategy {strategy_id} has not been validated. OOS validation required before deployment.")
            
            if validation_result['validation_status'] != 'passed':
                raise ValueError(f"Strategy {strategy_id} failed OOS validation. Status: {validation_result['validation_status']}")
            
            # Check if validation is recent (within 90 days)
            validation_age = (datetime.now() - validation_result['created_at']).days
            if validation_age > 90:
                raise ValueError(f"Strategy {strategy_id} validation is outdated ({validation_age} days old). Re-validation required.")
            
            # Proceed with deployment
            return await strategy_deploy_func(strategy_id, *args, **kwargs)
        
        return wrapper
    return decorator