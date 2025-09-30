"""
Monitoring & Attribution System with Alpha-Decay Tracking.

Implements comprehensive monitoring and attribution framework for trading strategies:
1. Alpha-decay detection and tracking over time
2. Multi-factor P&L attribution analysis
3. Performance monitoring with SLOs (Service Level Objectives)
4. Model degradation detection and alerts
5. Risk attribution and factor exposure tracking
6. Real-time performance monitoring and alerting
7. Strategy health scoring and diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, RidgeRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


@dataclass
class AlphaDecayMetrics:
    """Alpha decay analysis results."""
    lookback_periods: List[int]
    alpha_estimates: List[float]
    t_statistics: List[float]
    p_values: List[float]
    decay_rate: float
    half_life_days: Optional[float]
    decay_r_squared: float
    is_significant_decay: bool
    decay_confidence: float


@dataclass
class AttributionResult:
    """P&L attribution analysis result."""
    timestamp: datetime
    total_pnl: float
    factor_attributions: Dict[str, float]
    specific_return: float
    attribution_r_squared: float
    explained_variance: float
    residual_risk: float
    factor_exposures: Dict[str, float]
    factor_contributions: Dict[str, float]


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    period_start: datetime
    period_end: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    average_trade_pnl: float
    trade_count: int
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float


@dataclass
class SLOTarget:
    """Service Level Objective target."""
    metric_name: str
    target_value: float
    tolerance: float
    operator: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    frequency: str  # 'daily', 'weekly', 'monthly'
    description: str


@dataclass
class SLOViolation:
    """SLO violation record."""
    timestamp: datetime
    slo_name: str
    actual_value: float
    target_value: float
    violation_magnitude: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str


@dataclass
class ModelHealthScore:
    """Model health assessment."""
    timestamp: datetime
    overall_score: float  # 0-100
    alpha_decay_score: float
    attribution_quality_score: float
    risk_score: float
    performance_score: float
    data_quality_score: float
    alerts: List[str]
    recommendations: List[str]


class AlphaDecayAnalyzer:
    """Analyzes alpha decay patterns in trading strategies."""
    
    def __init__(
        self,
        min_observations: int = 60,
        confidence_level: float = 0.95,
        decay_significance_threshold: float = 0.05
    ):
        self.min_observations = min_observations
        self.confidence_level = confidence_level
        self.decay_significance_threshold = decay_significance_threshold
        
    def analyze_alpha_decay(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        lookback_periods: List[int] = None
    ) -> AlphaDecayMetrics:
        """
        Analyze alpha decay over different lookback periods.
        
        Args:
            returns: Strategy returns time series
            benchmark_returns: Benchmark returns time series
            lookback_periods: List of lookback periods to analyze
            
        Returns:
            AlphaDecayMetrics with decay analysis results
        """
        
        if lookback_periods is None:
            lookback_periods = [30, 60, 90, 120, 180, 252]
        
        # Align returns
        aligned_data = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < self.min_observations:
            raise ValueError(f"Insufficient data: {len(aligned_data)} < {self.min_observations}")
        
        alpha_estimates = []
        t_statistics = []
        p_values = []
        
        for period in lookback_periods:
            if len(aligned_data) >= period:
                # Calculate rolling alpha over the period
                alpha, t_stat, p_val = self._calculate_rolling_alpha(
                    aligned_data.tail(period)
                )
                alpha_estimates.append(alpha)
                t_statistics.append(t_stat)
                p_values.append(p_val)
            else:
                alpha_estimates.append(np.nan)
                t_statistics.append(np.nan)
                p_values.append(np.nan)
        
        # Fit decay model
        decay_rate, half_life, decay_r2 = self._fit_decay_model(
            lookback_periods, alpha_estimates
        )
        
        # Assess decay significance
        valid_alphas = [a for a in alpha_estimates if not np.isnan(a)]
        if len(valid_alphas) >= 3:
            decay_trend_p = stats.spearmanr(
                lookback_periods[:len(valid_alphas)], valid_alphas
            )[1]
            is_significant_decay = decay_trend_p < self.decay_significance_threshold
            decay_confidence = 1 - decay_trend_p
        else:
            is_significant_decay = False
            decay_confidence = 0.0
        
        return AlphaDecayMetrics(
            lookback_periods=lookback_periods,
            alpha_estimates=alpha_estimates,
            t_statistics=t_statistics,
            p_values=p_values,
            decay_rate=decay_rate,
            half_life_days=half_life,
            decay_r_squared=decay_r2,
            is_significant_decay=is_significant_decay,
            decay_confidence=decay_confidence
        )
    
    def _calculate_rolling_alpha(
        self,
        data: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """Calculate alpha using CAPM regression."""
        
        # Calculate excess returns (assuming risk-free rate = 0 for simplicity)
        strategy_returns = data['strategy'].values
        benchmark_returns = data['benchmark'].values
        
        # CAPM regression: R_strategy = alpha + beta * R_benchmark + epsilon
        X = benchmark_returns.reshape(-1, 1)
        y = strategy_returns
        
        # Fit regression
        reg = LinearRegression().fit(X, y)
        alpha = reg.intercept_
        beta = reg.coef_[0]
        
        # Calculate statistics
        y_pred = reg.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        
        # Standard error of alpha
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        alpha_se = np.sqrt(cov_matrix[0, 0])
        
        # T-statistic and p-value
        t_stat = alpha / alpha_se if alpha_se > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 2))
        
        return alpha, t_stat, p_val
    
    def _fit_decay_model(
        self,
        periods: List[int],
        alphas: List[float]
    ) -> Tuple[float, Optional[float], float]:
        """Fit exponential decay model to alpha estimates."""
        
        # Filter out NaN values
        valid_data = [(p, a) for p, a in zip(periods, alphas) if not np.isnan(a)]
        
        if len(valid_data) < 3:
            return 0.0, None, 0.0
        
        periods_valid = [d[0] for d in valid_data]
        alphas_valid = [d[1] for d in valid_data]
        
        try:
            # Fit exponential decay: alpha(t) = alpha_0 * exp(-lambda * t)
            def decay_func(params, t):
                alpha_0, decay_rate = params
                return alpha_0 * np.exp(-decay_rate * t)
            
            def objective(params):
                predicted = decay_func(params, periods_valid)
                return np.sum((np.array(alphas_valid) - predicted) ** 2)
            
            # Initial guess
            initial_alpha = alphas_valid[0] if alphas_valid[0] > 0 else 0.01
            initial_guess = [initial_alpha, 0.01]
            
            # Optimize
            result = minimize(objective, initial_guess, bounds=[(0, None), (0, 1)])
            
            if result.success:
                alpha_0, decay_rate = result.x
                predicted = decay_func(result.x, periods_valid)
                r_squared = r2_score(alphas_valid, predicted)
                
                # Calculate half-life
                half_life = np.log(2) / decay_rate if decay_rate > 0 else None
                
                return decay_rate, half_life, r_squared
            else:
                return 0.0, None, 0.0
                
        except Exception as e:
            logger.warning(f"Decay model fitting failed: {e}")
            return 0.0, None, 0.0


class FactorAttributionEngine:
    """Multi-factor P&L attribution analysis."""
    
    def __init__(
        self,
        factor_names: List[str],
        attribution_method: str = 'regression',
        regularization_alpha: float = 0.01
    ):
        self.factor_names = factor_names
        self.attribution_method = attribution_method
        self.regularization_alpha = regularization_alpha
        self.scaler = StandardScaler()
        
    def attribute_returns(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        weights: Optional[pd.Series] = None
    ) -> AttributionResult:
        """
        Perform factor attribution analysis.
        
        Args:
            returns: Portfolio/strategy returns
            factor_returns: Factor returns DataFrame
            weights: Optional sample weights
            
        Returns:
            AttributionResult with attribution breakdown
        """
        
        # Align data
        aligned_data = pd.DataFrame({
            'returns': returns,
            **{f'factor_{i}': factor_returns[col] 
               for i, col in enumerate(factor_returns.columns)}
        }).dropna()
        
        if len(aligned_data) < 30:
            raise ValueError("Insufficient data for attribution analysis")
        
        # Prepare regression data
        y = aligned_data['returns'].values
        X = aligned_data[[col for col in aligned_data.columns if col.startswith('factor_')]].values
        
        # Scale factors
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit attribution model
        if self.attribution_method == 'regression':
            model = LinearRegression()
        elif self.attribution_method == 'ridge':
            model = RidgeRegression(alpha=self.regularization_alpha)
        else:
            raise ValueError(f"Unknown attribution method: {self.attribution_method}")
        
        # Fit model with sample weights if provided
        if weights is not None:
            weights_aligned = weights.loc[aligned_data.index].values
            model.fit(X_scaled, y, sample_weight=weights_aligned)
        else:
            model.fit(X_scaled, y)
        
        # Calculate factor exposures (loadings)
        factor_exposures = dict(zip(self.factor_names, model.coef_))
        
        # Calculate factor contributions
        factor_contributions = {}
        total_factor_pnl = 0
        
        for i, factor_name in enumerate(self.factor_names):
            # Factor contribution = exposure * factor_return
            factor_return = aligned_data.iloc[-1][f'factor_{i}']
            contribution = factor_exposures[factor_name] * factor_return
            factor_contributions[factor_name] = contribution
            total_factor_pnl += contribution
        
        # Calculate specific return (alpha)
        total_return = aligned_data['returns'].iloc[-1]
        specific_return = total_return - total_factor_pnl
        
        # Model quality metrics
        y_pred = model.predict(X_scaled)
        r_squared = r2_score(y, y_pred)
        explained_variance = model.score(X_scaled, y)
        residuals = y - y_pred
        residual_risk = np.std(residuals)
        
        return AttributionResult(
            timestamp=aligned_data.index[-1],
            total_pnl=total_return,
            factor_attributions=factor_contributions,
            specific_return=specific_return,
            attribution_r_squared=r_squared,
            explained_variance=explained_variance,
            residual_risk=residual_risk,
            factor_exposures=factor_exposures,
            factor_contributions=factor_contributions
        )
    
    def rolling_attribution(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 60,
        min_periods: int = 30
    ) -> pd.DataFrame:
        """Calculate rolling factor attribution over time."""
        
        attribution_results = []
        
        for i in range(min_periods, len(returns)):
            end_idx = i + 1
            start_idx = max(0, end_idx - window)
            
            window_returns = returns.iloc[start_idx:end_idx]
            window_factors = factor_returns.iloc[start_idx:end_idx]
            
            try:
                attribution = self.attribute_returns(window_returns, window_factors)
                
                result_row = {
                    'timestamp': window_returns.index[-1],
                    'total_pnl': attribution.total_pnl,
                    'specific_return': attribution.specific_return,
                    'r_squared': attribution.attribution_r_squared,
                    'residual_risk': attribution.residual_risk
                }
                
                # Add factor contributions
                for factor_name, contribution in attribution.factor_contributions.items():
                    result_row[f'{factor_name}_contribution'] = contribution
                
                # Add factor exposures
                for factor_name, exposure in attribution.factor_exposures.items():
                    result_row[f'{factor_name}_exposure'] = exposure
                
                attribution_results.append(result_row)
                
            except Exception as e:
                logger.warning(f"Attribution failed for window ending {window_returns.index[-1]}: {e}")
                continue
        
        return pd.DataFrame(attribution_results)


class SLOMonitor:
    """Service Level Objectives monitoring system."""
    
    def __init__(self):
        self.slo_targets = {}
        self.violation_history = []
        
    def add_slo(self, slo: SLOTarget):
        """Add SLO target for monitoring."""
        self.slo_targets[slo.metric_name] = slo
        
    def check_slo_compliance(
        self,
        metrics: Dict[str, float],
        timestamp: datetime
    ) -> List[SLOViolation]:
        """Check SLO compliance and return violations."""
        
        violations = []
        
        for metric_name, metric_value in metrics.items():
            if metric_name in self.slo_targets:
                slo = self.slo_targets[metric_name]
                violation = self._check_single_slo(slo, metric_value, timestamp)
                if violation:
                    violations.append(violation)
                    self.violation_history.append(violation)
        
        return violations
    
    def _check_single_slo(
        self,
        slo: SLOTarget,
        actual_value: float,
        timestamp: datetime
    ) -> Optional[SLOViolation]:
        """Check compliance for a single SLO."""
        
        target = slo.target_value
        tolerance = slo.tolerance
        
        # Check violation based on operator
        is_violation = False
        if slo.operator == 'gt' and actual_value <= target - tolerance:
            is_violation = True
        elif slo.operator == 'lt' and actual_value >= target + tolerance:
            is_violation = True
        elif slo.operator == 'gte' and actual_value < target - tolerance:
            is_violation = True
        elif slo.operator == 'lte' and actual_value > target + tolerance:
            is_violation = True
        elif slo.operator == 'eq' and abs(actual_value - target) > tolerance:
            is_violation = True
        
        if is_violation:
            violation_magnitude = abs(actual_value - target)
            
            # Determine severity
            if violation_magnitude > tolerance * 3:
                severity = 'critical'
            elif violation_magnitude > tolerance * 2:
                severity = 'high'
            elif violation_magnitude > tolerance:
                severity = 'medium'
            else:
                severity = 'low'
            
            return SLOViolation(
                timestamp=timestamp,
                slo_name=slo.metric_name,
                actual_value=actual_value,
                target_value=target,
                violation_magnitude=violation_magnitude,
                severity=severity,
                description=f"{slo.metric_name} SLO violation: {actual_value:.4f} vs target {target:.4f}"
            )
        
        return None
    
    def get_slo_compliance_rate(
        self,
        metric_name: str,
        period_days: int = 30
    ) -> float:
        """Calculate SLO compliance rate over period."""
        
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_violations = [
            v for v in self.violation_history
            if v.slo_name == metric_name and v.timestamp >= cutoff_date
        ]
        
        # Simplified calculation - in practice, would need more sophisticated tracking
        total_checks = period_days  # Assuming daily checks
        violation_count = len(recent_violations)
        
        return max(0.0, 1.0 - violation_count / total_checks)


class PerformanceMonitor:
    """Real-time strategy performance monitoring."""
    
    def __init__(self):
        self.performance_history = []
        self.alert_thresholds = {
            'max_drawdown': -0.15,      # -15% max drawdown
            'sharpe_ratio': 0.5,        # Minimum Sharpe ratio
            'win_rate': 0.4,            # Minimum 40% win rate
            'var_95': -0.05,            # 5% daily VaR limit
            'volatility': 0.25          # 25% annual volatility limit
        }
        
    def calculate_performance_metrics(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if len(returns) < 2:
            raise ValueError("Need at least 2 return observations")
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        if prices is not None:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade-based metrics
        if trades is not None and len(trades) > 0:
            win_rate = (trades['pnl'] > 0).mean()
            winning_trades = trades[trades['pnl'] > 0]['pnl']
            losing_trades = trades[trades['pnl'] < 0]['pnl']
            
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 1
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            average_trade_pnl = trades['pnl'].mean()
            trade_count = len(trades)
        else:
            win_rate = (returns > 0).mean()
            profit_factor = 1.0
            average_trade_pnl = returns.mean()
            trade_count = len(returns)
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return PerformanceMetrics(
            period_start=returns.index[0],
            period_end=returns.index[-1],
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_trade_pnl=average_trade_pnl,
            trade_count=trade_count,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    def generate_alerts(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance alerts based on thresholds."""
        
        alerts = []
        
        if metrics.max_drawdown < self.alert_thresholds['max_drawdown']:
            alerts.append(f"High drawdown alert: {metrics.max_drawdown:.2%} exceeds threshold")
        
        if metrics.sharpe_ratio < self.alert_thresholds['sharpe_ratio']:
            alerts.append(f"Low Sharpe ratio alert: {metrics.sharpe_ratio:.2f} below threshold")
        
        if metrics.win_rate < self.alert_thresholds['win_rate']:
            alerts.append(f"Low win rate alert: {metrics.win_rate:.2%} below threshold")
        
        if metrics.var_95 < self.alert_thresholds['var_95']:
            alerts.append(f"High VaR alert: {metrics.var_95:.2%} exceeds risk limit")
        
        if metrics.volatility > self.alert_thresholds['volatility']:
            alerts.append(f"High volatility alert: {metrics.volatility:.2%} exceeds limit")
        
        return alerts


class ModelHealthAnalyzer:
    """Comprehensive model health assessment."""
    
    def __init__(self):
        self.alpha_analyzer = AlphaDecayAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        
    def assess_model_health(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame,
        timestamp: datetime
    ) -> ModelHealthScore:
        """Assess overall model health and generate score."""
        
        alerts = []
        recommendations = []
        
        # 1. Alpha decay analysis
        try:
            alpha_decay = self.alpha_analyzer.analyze_alpha_decay(
                strategy_returns, benchmark_returns
            )
            
            if alpha_decay.is_significant_decay:
                alpha_decay_score = max(0, 50 - alpha_decay.decay_confidence * 50)
                alerts.append("Significant alpha decay detected")
                recommendations.append("Consider model retraining or strategy adjustment")
            else:
                alpha_decay_score = 85
                
        except Exception as e:
            alpha_decay_score = 50
            alerts.append(f"Alpha decay analysis failed: {str(e)}")
        
        # 2. Attribution quality
        try:
            factor_engine = FactorAttributionEngine(factor_returns.columns.tolist())
            attribution = factor_engine.attribute_returns(strategy_returns, factor_returns)
            
            if attribution.attribution_r_squared > 0.7:
                attribution_quality_score = 90
            elif attribution.attribution_r_squared > 0.5:
                attribution_quality_score = 70
            else:
                attribution_quality_score = 40
                alerts.append("Low attribution model quality")
                
        except Exception as e:
            attribution_quality_score = 50
            alerts.append(f"Attribution analysis failed: {str(e)}")
        
        # 3. Performance analysis
        try:
            performance = self.performance_monitor.calculate_performance_metrics(strategy_returns)
            
            perf_alerts = self.performance_monitor.generate_alerts(performance)
            alerts.extend(perf_alerts)
            
            # Performance scoring
            if performance.sharpe_ratio > 1.5:
                performance_score = 95
            elif performance.sharpe_ratio > 1.0:
                performance_score = 80
            elif performance.sharpe_ratio > 0.5:
                performance_score = 60
            else:
                performance_score = 30
                
        except Exception as e:
            performance_score = 50
            alerts.append(f"Performance analysis failed: {str(e)}")
        
        # 4. Risk assessment
        try:
            recent_vol = strategy_returns.tail(30).std() * np.sqrt(252)
            if recent_vol > 0.3:
                risk_score = 30
                alerts.append("High recent volatility")
            elif recent_vol > 0.2:
                risk_score = 60
            else:
                risk_score = 85
                
        except Exception:
            risk_score = 50
        
        # 5. Data quality (simplified)
        missing_data_pct = strategy_returns.isna().mean()
        if missing_data_pct > 0.05:
            data_quality_score = 40
            alerts.append("High missing data rate")
        else:
            data_quality_score = 90
        
        # Calculate overall score
        weights = [0.25, 0.20, 0.25, 0.20, 0.10]  # Alpha, Attribution, Performance, Risk, Data
        scores = [alpha_decay_score, attribution_quality_score, performance_score, risk_score, data_quality_score]
        overall_score = sum(w * s for w, s in zip(weights, scores))
        
        # Generate recommendations
        if overall_score < 60:
            recommendations.append("Model requires immediate attention")
        elif overall_score < 75:
            recommendations.append("Consider model optimization")
        
        return ModelHealthScore(
            timestamp=timestamp,
            overall_score=overall_score,
            alpha_decay_score=alpha_decay_score,
            attribution_quality_score=attribution_quality_score,
            risk_score=risk_score,
            performance_score=performance_score,
            data_quality_score=data_quality_score,
            alerts=alerts,
            recommendations=recommendations
        )


class ComprehensiveMonitoringSystem:
    """Integrated monitoring and attribution system."""
    
    def __init__(self, factor_names: List[str]):
        self.alpha_analyzer = AlphaDecayAnalyzer()
        self.attribution_engine = FactorAttributionEngine(factor_names)
        self.slo_monitor = SLOMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.health_analyzer = ModelHealthAnalyzer()
        
        # Initialize default SLOs
        self._setup_default_slos()
        
    def _setup_default_slos(self):
        """Setup default SLO targets."""
        default_slos = [
            SLOTarget("sharpe_ratio", 1.0, 0.2, "gte", "daily", "Minimum Sharpe ratio"),
            SLOTarget("max_drawdown", -0.15, 0.05, "gte", "daily", "Maximum drawdown limit"),
            SLOTarget("win_rate", 0.45, 0.05, "gte", "weekly", "Minimum win rate"),
            SLOTarget("volatility", 0.20, 0.05, "lte", "daily", "Maximum volatility")
        ]
        
        for slo in default_slos:
            self.slo_monitor.add_slo(slo)
    
    def run_comprehensive_monitoring(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, any]:
        """Run complete monitoring analysis."""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        results = {
            'timestamp': timestamp,
            'monitoring_results': {}
        }
        
        # 1. Alpha decay analysis
        try:
            alpha_decay = self.alpha_analyzer.analyze_alpha_decay(
                strategy_returns, benchmark_returns
            )
            results['monitoring_results']['alpha_decay'] = alpha_decay
        except Exception as e:
            logger.error(f"Alpha decay analysis failed: {e}")
            results['monitoring_results']['alpha_decay'] = None
        
        # 2. Factor attribution
        try:
            attribution = self.attribution_engine.attribute_returns(
                strategy_returns, factor_returns
            )
            results['monitoring_results']['attribution'] = attribution
        except Exception as e:
            logger.error(f"Attribution analysis failed: {e}")
            results['monitoring_results']['attribution'] = None
        
        # 3. Performance metrics
        try:
            performance = self.performance_monitor.calculate_performance_metrics(strategy_returns)
            results['monitoring_results']['performance'] = performance
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            results['monitoring_results']['performance'] = None
        
        # 4. SLO compliance
        try:
            if results['monitoring_results']['performance']:
                perf = results['monitoring_results']['performance']
                metrics = {
                    'sharpe_ratio': perf.sharpe_ratio,
                    'max_drawdown': perf.max_drawdown,
                    'win_rate': perf.win_rate,
                    'volatility': perf.volatility
                }
                violations = self.slo_monitor.check_slo_compliance(metrics, timestamp)
                results['monitoring_results']['slo_violations'] = violations
        except Exception as e:
            logger.error(f"SLO monitoring failed: {e}")
            results['monitoring_results']['slo_violations'] = []
        
        # 5. Model health assessment
        try:
            health_score = self.health_analyzer.assess_model_health(
                strategy_returns, benchmark_returns, factor_returns, timestamp
            )
            results['monitoring_results']['health_score'] = health_score
        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
            results['monitoring_results']['health_score'] = None
        
        return results
    
    def get_monitoring_summary(self) -> Dict[str, any]:
        """Get high-level monitoring summary."""
        
        return {
            'slo_compliance_rates': {
                slo_name: self.slo_monitor.get_slo_compliance_rate(slo_name)
                for slo_name in self.slo_monitor.slo_targets.keys()
            },
            'total_violations': len(self.slo_monitor.violation_history),
            'recent_violations': len([
                v for v in self.slo_monitor.violation_history
                if v.timestamp >= datetime.now() - timedelta(days=7)
            ]),
            'alert_thresholds': self.performance_monitor.alert_thresholds
        }