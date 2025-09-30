"""
Alpha Decay & Attribution Monitoring Framework

This module provides comprehensive monitoring and attribution analysis for trading strategies,
including alpha decay detection, performance attribution, and Service Level Objectives (SLOs).

Key Features:
1. Alpha Decay Tracking: Monitor strategy performance degradation over time
2. P&L Attribution: Decompose returns into components (alpha, beta, specific risk)
3. Risk Attribution: Factor-based risk decomposition and monitoring
4. Performance SLOs: Service Level Objectives with alerting
5. Real-time Monitoring: Live performance tracking with anomaly detection
6. Regime Detection: Identify market regime changes affecting strategy performance

Applications:
- Strategy performance monitoring
- Risk management and position sizing
- Portfolio construction and rebalancing
- Strategy lifecycle management
- Compliance and reporting

References:
- Grinold, R. C., & Kahn, R. N. (2019). Active Portfolio Management.
- Qian, E., Hua, R., & Sorensen, E. (2007). Quantitative Equity Portfolio Management.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AttributionComponent(Enum):
    """Components for performance attribution."""
    ALPHA = "alpha"                    # Strategy-specific returns
    BETA = "beta"                      # Market exposure returns
    SECTOR = "sector"                  # Sector exposure returns
    STYLE = "style"                    # Style factor returns
    SPECIFIC = "specific"              # Security-specific returns
    INTERACTION = "interaction"        # Interaction effects
    TIMING = "timing"                  # Market timing effects


class RegimeType(Enum):
    """Market regime types."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    CRISIS = "crisis"


@dataclass
class SLOTarget:
    """Service Level Objective target."""
    metric_name: str
    target_value: float
    threshold_type: str = "min"  # "min", "max", "range"
    measurement_window: str = "1D"  # "1H", "1D", "1W", "1M"
    alert_level: AlertLevel = AlertLevel.WARNING
    
    def __post_init__(self):
        if self.threshold_type not in ["min", "max", "range"]:
            raise ValueError("threshold_type must be 'min', 'max', or 'range'")


@dataclass
class PerformanceAlert:
    """Performance monitoring alert."""
    timestamp: datetime
    strategy_id: str
    alert_level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'strategy_id': self.strategy_id,
            'alert_level': self.alert_level.value,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'message': self.message,
            'metadata': self.metadata
        }


@dataclass
class AlphaDecayResult:
    """Results from alpha decay analysis."""
    strategy_id: str
    analysis_period: Tuple[datetime, datetime]
    half_life_days: float
    decay_rate: float
    r_squared: float
    significance_level: float
    is_statistically_significant: bool
    confidence_interval: Tuple[float, float]
    alpha_trajectory: pd.Series
    fitted_decay: pd.Series
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_id': self.strategy_id,
            'analysis_period': [self.analysis_period[0].isoformat(), self.analysis_period[1].isoformat()],
            'half_life_days': self.half_life_days,
            'decay_rate': self.decay_rate,
            'r_squared': self.r_squared,
            'significance_level': self.significance_level,
            'is_statistically_significant': self.is_statistically_significant,
            'confidence_interval': self.confidence_interval,
            'alpha_trajectory': self.alpha_trajectory.to_dict(),
            'fitted_decay': self.fitted_decay.to_dict()
        }


@dataclass
class AttributionResult:
    """Results from performance attribution analysis."""
    strategy_id: str
    period: Tuple[datetime, datetime]
    total_return: float
    attribution_breakdown: Dict[AttributionComponent, float]
    factor_exposures: Dict[str, float]
    risk_attribution: Dict[str, float]
    tracking_error: float
    information_ratio: float
    active_return: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_id': self.strategy_id,
            'period': [self.period[0].isoformat(), self.period[1].isoformat()],
            'total_return': self.total_return,
            'attribution_breakdown': {k.value: v for k, v in self.attribution_breakdown.items()},
            'factor_exposures': self.factor_exposures,
            'risk_attribution': self.risk_attribution,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio,
            'active_return': self.active_return
        }


class AlphaDecayMonitor:
    """
    Alpha Decay Monitoring System.
    
    Monitors strategy performance degradation over time using various models
    including exponential decay, linear decay, and regime-based analysis.
    """
    
    def __init__(self, 
                 min_observations: int = 30,
                 significance_level: float = 0.05,
                 rolling_window: int = 252):
        """
        Initialize alpha decay monitor.
        
        Parameters:
        - min_observations: Minimum observations for analysis
        - significance_level: Statistical significance level
        - rolling_window: Rolling window for analysis
        """
        self.min_observations = min_observations
        self.significance_level = significance_level
        self.rolling_window = rolling_window
        
        # Storage for analysis results
        self.decay_results: Dict[str, AlphaDecayResult] = {}
        self.performance_history: Dict[str, pd.DataFrame] = {}
    
    def add_performance_data(self, 
                           strategy_id: str,
                           returns: pd.Series,
                           benchmark_returns: Optional[pd.Series] = None,
                           factor_returns: Optional[pd.DataFrame] = None):
        """
        Add performance data for monitoring.
        
        Parameters:
        - strategy_id: Strategy identifier
        - returns: Strategy returns
        - benchmark_returns: Benchmark returns for active return calculation
        - factor_returns: Factor returns for attribution
        """
        # Calculate active returns if benchmark provided
        if benchmark_returns is not None:
            aligned_returns = returns.align(benchmark_returns, join='inner')
            active_returns = aligned_returns[0] - aligned_returns[1]
        else:
            active_returns = returns
        
        # Store performance data
        perf_data = pd.DataFrame({
            'returns': returns,
            'active_returns': active_returns,
            'cumulative_returns': (1 + returns).cumprod(),
            'rolling_alpha': self._calculate_rolling_alpha(returns, benchmark_returns, factor_returns)
        })
        
        self.performance_history[strategy_id] = perf_data
        
        logger.debug(f"Added {len(returns)} observations for strategy {strategy_id}")
    
    def analyze_alpha_decay(self, 
                          strategy_id: str,
                          model_type: str = "exponential") -> Optional[AlphaDecayResult]:
        """
        Analyze alpha decay for strategy.
        
        Parameters:
        - strategy_id: Strategy to analyze
        - model_type: Decay model ("exponential", "linear", "power")
        
        Returns:
        - Alpha decay analysis results
        """
        if strategy_id not in self.performance_history:
            logger.warning(f"No performance data for strategy {strategy_id}")
            return None
        
        perf_data = self.performance_history[strategy_id]
        
        if len(perf_data) < self.min_observations:
            logger.warning(f"Insufficient data for {strategy_id}: {len(perf_data)} observations")
            return None
        
        alpha_series = perf_data['rolling_alpha'].dropna()
        
        if len(alpha_series) < self.min_observations:
            logger.warning(f"Insufficient alpha data for {strategy_id}")
            return None
        
        # Fit decay model
        if model_type == "exponential":
            decay_result = self._fit_exponential_decay(strategy_id, alpha_series)
        elif model_type == "linear":
            decay_result = self._fit_linear_decay(strategy_id, alpha_series)
        elif model_type == "power":
            decay_result = self._fit_power_decay(strategy_id, alpha_series)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Store results
        self.decay_results[strategy_id] = decay_result
        
        logger.info(f"Alpha decay analysis for {strategy_id}: "
                   f"half-life={decay_result.half_life_days:.1f} days, "
                   f"R²={decay_result.r_squared:.3f}")
        
        return decay_result
    
    def _calculate_rolling_alpha(self, 
                               returns: pd.Series,
                               benchmark_returns: Optional[pd.Series],
                               factor_returns: Optional[pd.DataFrame]) -> pd.Series:
        """Calculate rolling alpha using factor model."""
        if benchmark_returns is None:
            # Use simple rolling mean as alpha proxy
            return returns.rolling(window=30, min_periods=10).mean() * 252
        
        # Calculate rolling alpha using CAPM or factor model
        alpha_series = pd.Series(index=returns.index, dtype=float)
        
        for i in range(30, len(returns)):
            window_returns = returns.iloc[i-30:i]
            window_benchmark = benchmark_returns.iloc[i-30:i]
            
            # Align data
            aligned_data = pd.DataFrame({
                'returns': window_returns,
                'benchmark': window_benchmark
            }).dropna()
            
            if len(aligned_data) < 15:  # Minimum observations
                continue
            
            # Fit CAPM model
            X = sm.add_constant(aligned_data['benchmark'])
            y = aligned_data['returns']
            
            try:
                model = sm.OLS(y, X).fit()
                alpha_annual = model.params[0] * 252  # Annualize alpha
                alpha_series.iloc[i] = alpha_annual
            except:
                continue
        
        return alpha_series
    
    def _fit_exponential_decay(self, 
                             strategy_id: str, 
                             alpha_series: pd.Series) -> AlphaDecayResult:
        """Fit exponential decay model to alpha series."""
        # Time variable (days since start)
        time_days = (alpha_series.index - alpha_series.index[0]).days
        
        # Remove invalid values
        valid_mask = np.isfinite(alpha_series) & np.isfinite(time_days)
        y = alpha_series[valid_mask].values
        t = time_days[valid_mask]
        
        if len(y) < 10:
            raise ValueError("Insufficient valid data points")
        
        # Fit exponential decay: alpha(t) = alpha_0 * exp(-lambda * t)
        def exponential_model(params, t):
            alpha_0, decay_rate = params
            return alpha_0 * np.exp(-decay_rate * t)
        
        def objective(params):
            predicted = exponential_model(params, t)
            return np.sum((y - predicted) ** 2)
        
        # Initial guess
        initial_params = [y[0], 0.001]  # Small initial decay rate
        
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B',
                            bounds=[(None, None), (0, 1)])  # Decay rate must be positive
            
            alpha_0, decay_rate = result.x
            fitted_values = exponential_model(result.x, t)
            
            # Calculate statistics
            ss_res = np.sum((y - fitted_values) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate half-life
            half_life = np.log(2) / decay_rate if decay_rate > 0 else np.inf
            
            # Statistical significance test
            # Use F-test comparing to constant model
            n = len(y)
            f_stat = ((ss_tot - ss_res) / 1) / (ss_res / (n - 2))
            p_value = 1 - stats.f.cdf(f_stat, 1, n - 2)
            is_significant = p_value < self.significance_level
            
            # Confidence interval for decay rate (rough approximation)
            std_error = np.sqrt(ss_res / (n - 2)) / np.std(t)
            t_crit = stats.t.ppf(1 - self.significance_level/2, n - 2)
            ci_lower = decay_rate - t_crit * std_error
            ci_upper = decay_rate + t_crit * std_error
            
            # Create fitted series
            fitted_series = pd.Series(
                exponential_model(result.x, time_days),
                index=alpha_series.index
            )
            
            return AlphaDecayResult(
                strategy_id=strategy_id,
                analysis_period=(alpha_series.index[0], alpha_series.index[-1]),
                half_life_days=half_life,
                decay_rate=decay_rate,
                r_squared=r_squared,
                significance_level=p_value,
                is_statistically_significant=is_significant,
                confidence_interval=(ci_lower, ci_upper),
                alpha_trajectory=alpha_series,
                fitted_decay=fitted_series
            )
            
        except Exception as e:
            logger.error(f"Failed to fit exponential decay for {strategy_id}: {e}")
            # Return default result
            return AlphaDecayResult(
                strategy_id=strategy_id,
                analysis_period=(alpha_series.index[0], alpha_series.index[-1]),
                half_life_days=np.inf,
                decay_rate=0.0,
                r_squared=0.0,
                significance_level=1.0,
                is_statistically_significant=False,
                confidence_interval=(0.0, 0.0),
                alpha_trajectory=alpha_series,
                fitted_decay=pd.Series(np.full_like(alpha_series, np.mean(y)), index=alpha_series.index)
            )
    
    def _fit_linear_decay(self, 
                        strategy_id: str, 
                        alpha_series: pd.Series) -> AlphaDecayResult:
        """Fit linear decay model to alpha series."""
        time_days = (alpha_series.index - alpha_series.index[0]).days
        
        # Remove invalid values
        valid_mask = np.isfinite(alpha_series) & np.isfinite(time_days)
        y = alpha_series[valid_mask].values
        t = time_days[valid_mask]
        
        if len(y) < 10:
            raise ValueError("Insufficient valid data points")
        
        # Fit linear model: alpha(t) = alpha_0 + beta * t
        X = sm.add_constant(t)
        model = sm.OLS(y, X).fit()
        
        alpha_0 = model.params[0]
        decay_rate = -model.params[1]  # Negative slope indicates decay
        
        # Calculate half-life for linear decay
        if decay_rate > 0 and alpha_0 > 0:
            half_life = alpha_0 / (2 * decay_rate)
        else:
            half_life = np.inf
        
        # Create fitted series
        fitted_series = pd.Series(
            model.predict(sm.add_constant(time_days)),
            index=alpha_series.index
        )
        
        return AlphaDecayResult(
            strategy_id=strategy_id,
            analysis_period=(alpha_series.index[0], alpha_series.index[-1]),
            half_life_days=half_life,
            decay_rate=decay_rate,
            r_squared=model.rsquared,
            significance_level=model.pvalues[1],  # P-value for slope
            is_statistically_significant=model.pvalues[1] < self.significance_level,
            confidence_interval=tuple(model.conf_int().iloc[1]),
            alpha_trajectory=alpha_series,
            fitted_decay=fitted_series
        )
    
    def _fit_power_decay(self, 
                       strategy_id: str, 
                       alpha_series: pd.Series) -> AlphaDecayResult:
        """Fit power decay model to alpha series."""
        time_days = (alpha_series.index - alpha_series.index[0]).days + 1  # Add 1 to avoid log(0)
        
        # Remove invalid values
        valid_mask = (np.isfinite(alpha_series) & np.isfinite(time_days) & 
                     (alpha_series > 0))  # Power model requires positive values
        y = alpha_series[valid_mask].values
        t = time_days[valid_mask]
        
        if len(y) < 10:
            raise ValueError("Insufficient valid data points")
        
        # Fit power model: alpha(t) = alpha_0 * t^(-beta)
        # Use log-linear regression: log(alpha) = log(alpha_0) - beta * log(t)
        X = sm.add_constant(np.log(t))
        model = sm.OLS(np.log(y), X).fit()
        
        log_alpha_0 = model.params[0]
        power_exponent = -model.params[1]
        
        alpha_0 = np.exp(log_alpha_0)
        
        # Calculate half-life for power decay
        if power_exponent > 0:
            half_life = (2 ** (1/power_exponent)) - 1
        else:
            half_life = np.inf
        
        # Create fitted series
        fitted_series = pd.Series(
            alpha_0 * (time_days ** (-power_exponent)),
            index=alpha_series.index
        )
        
        return AlphaDecayResult(
            strategy_id=strategy_id,
            analysis_period=(alpha_series.index[0], alpha_series.index[-1]),
            half_life_days=half_life,
            decay_rate=power_exponent,
            r_squared=model.rsquared,
            significance_level=model.pvalues[1],
            is_statistically_significant=model.pvalues[1] < self.significance_level,
            confidence_interval=tuple(model.conf_int().iloc[1]),
            alpha_trajectory=alpha_series,
            fitted_decay=fitted_series
        )
    
    def detect_regime_changes(self, 
                            strategy_id: str,
                            lookback_days: int = 60) -> List[Tuple[datetime, RegimeType]]:
        """
        Detect regime changes in strategy performance.
        
        Parameters:
        - strategy_id: Strategy to analyze
        - lookback_days: Lookback window for regime detection
        
        Returns:
        - List of (timestamp, regime_type) tuples
        """
        if strategy_id not in self.performance_history:
            return []
        
        perf_data = self.performance_history[strategy_id]
        returns = perf_data['returns']
        
        regime_changes = []
        
        # Rolling statistics for regime detection
        rolling_mean = returns.rolling(window=lookback_days).mean()
        rolling_vol = returns.rolling(window=lookback_days).std()
        rolling_sharpe = rolling_mean / rolling_vol
        
        # Detect regime changes based on statistical changes
        for i in range(lookback_days, len(returns)):
            current_date = returns.index[i]
            
            # Check for volatility regime change
            recent_vol = rolling_vol.iloc[i]
            historical_vol = rolling_vol.iloc[:i-lookback_days].median()
            
            if recent_vol > 1.5 * historical_vol:
                regime_changes.append((current_date, RegimeType.HIGH_VOLATILITY))
            elif recent_vol < 0.5 * historical_vol:
                regime_changes.append((current_date, RegimeType.LOW_VOLATILITY))
            
            # Check for performance regime change
            recent_sharpe = rolling_sharpe.iloc[i]
            historical_sharpe = rolling_sharpe.iloc[:i-lookback_days].median()
            
            if recent_sharpe < 0 and historical_sharpe > 0:
                regime_changes.append((current_date, RegimeType.BEAR_MARKET))
            elif recent_sharpe > 0 and historical_sharpe < 0:
                regime_changes.append((current_date, RegimeType.BULL_MARKET))
        
        return regime_changes


class PerformanceAttributor:
    """
    Performance Attribution System.
    
    Decomposes strategy returns into various components using factor models
    and provides risk attribution analysis.
    """
    
    def __init__(self):
        """Initialize performance attributor."""
        self.factor_models: Dict[str, Any] = {}
        self.attribution_history: Dict[str, List[AttributionResult]] = {}
    
    def add_factor_model(self, 
                        model_name: str,
                        factor_returns: pd.DataFrame,
                        factor_names: List[str]):
        """
        Add factor model for attribution.
        
        Parameters:
        - model_name: Name of the factor model
        - factor_returns: Factor return data
        - factor_names: Names of factors
        """
        self.factor_models[model_name] = {
            'factor_returns': factor_returns,
            'factor_names': factor_names
        }
        
        logger.info(f"Added factor model '{model_name}' with {len(factor_names)} factors")
    
    def attribute_performance(self,
                            strategy_id: str,
                            strategy_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            factor_model_name: str = "default",
                            holdings: Optional[pd.DataFrame] = None) -> AttributionResult:
        """
        Perform comprehensive performance attribution.
        
        Parameters:
        - strategy_id: Strategy identifier
        - strategy_returns: Strategy returns
        - benchmark_returns: Benchmark returns
        - factor_model_name: Factor model to use
        - holdings: Portfolio holdings for detailed attribution
        
        Returns:
        - Attribution analysis results
        """
        # Calculate basic metrics
        total_return = (1 + strategy_returns).prod() - 1
        active_returns = strategy_returns - benchmark_returns
        active_return = active_returns.mean() * 252  # Annualized
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        # Factor-based attribution
        attribution_breakdown = {}
        factor_exposures = {}
        risk_attribution = {}
        
        if factor_model_name in self.factor_models:
            factor_data = self.factor_models[factor_model_name]
            factor_returns = factor_data['factor_returns']
            factor_names = factor_data['factor_names']
            
            # Align data
            common_dates = strategy_returns.index.intersection(factor_returns.index)
            if len(common_dates) > 30:  # Minimum observations
                strategy_aligned = strategy_returns.loc[common_dates]
                factors_aligned = factor_returns.loc[common_dates, factor_names]
                
                # Fit factor model
                X = sm.add_constant(factors_aligned)
                model = sm.OLS(strategy_aligned, X).fit()
                
                # Extract attribution components
                alpha = model.params[0] * 252  # Annualized alpha
                factor_betas = model.params[1:]
                
                attribution_breakdown[AttributionComponent.ALPHA] = alpha
                
                # Factor contributions
                for i, factor_name in enumerate(factor_names):
                    factor_return = factors_aligned[factor_name].mean() * 252
                    factor_contribution = factor_betas.iloc[i] * factor_return
                    
                    # Categorize factors
                    if 'market' in factor_name.lower() or 'mkt' in factor_name.lower():
                        attribution_breakdown[AttributionComponent.BETA] = factor_contribution
                    elif 'sector' in factor_name.lower():
                        current_sector = attribution_breakdown.get(AttributionComponent.SECTOR, 0)
                        attribution_breakdown[AttributionComponent.SECTOR] = current_sector + factor_contribution
                    else:
                        current_style = attribution_breakdown.get(AttributionComponent.STYLE, 0)
                        attribution_breakdown[AttributionComponent.STYLE] = current_style + factor_contribution
                    
                    factor_exposures[factor_name] = factor_betas.iloc[i]
                
                # Risk attribution
                factor_cov = factors_aligned.cov() * 252  # Annualized
                portfolio_var = np.dot(factor_betas, np.dot(factor_cov, factor_betas))
                idiosyncratic_var = (model.resid.std() ** 2) * 252
                
                total_var = portfolio_var + idiosyncratic_var
                
                risk_attribution['factor_risk'] = portfolio_var / total_var if total_var > 0 else 0
                risk_attribution['specific_risk'] = idiosyncratic_var / total_var if total_var > 0 else 0
                
                # Individual factor risk contributions
                for i, factor_name in enumerate(factor_names):
                    factor_var = (factor_betas.iloc[i] ** 2) * factor_cov.iloc[i, i]
                    risk_attribution[f'{factor_name}_risk'] = factor_var / total_var if total_var > 0 else 0
        
        else:
            # Simple CAPM attribution
            X = sm.add_constant(benchmark_returns)
            model = sm.OLS(strategy_returns, X).fit()
            
            alpha = model.params[0] * 252
            beta = model.params[1]
            
            attribution_breakdown[AttributionComponent.ALPHA] = alpha
            attribution_breakdown[AttributionComponent.BETA] = beta * benchmark_returns.mean() * 252
            
            factor_exposures['market_beta'] = beta
        
        # Specific return component (residual)
        explained_return = sum(attribution_breakdown.values())
        attribution_breakdown[AttributionComponent.SPECIFIC] = total_return * 252 - explained_return
        
        result = AttributionResult(
            strategy_id=strategy_id,
            period=(strategy_returns.index[0], strategy_returns.index[-1]),
            total_return=total_return,
            attribution_breakdown=attribution_breakdown,
            factor_exposures=factor_exposures,
            risk_attribution=risk_attribution,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            active_return=active_return
        )
        
        # Store result
        if strategy_id not in self.attribution_history:
            self.attribution_history[strategy_id] = []
        self.attribution_history[strategy_id].append(result)
        
        logger.info(f"Attribution analysis for {strategy_id}: "
                   f"Alpha={attribution_breakdown.get(AttributionComponent.ALPHA, 0):.2%}, "
                   f"IR={information_ratio:.2f}")
        
        return result


class SLOMonitor:
    """
    Service Level Objectives (SLO) Monitoring System.
    
    Monitors strategy performance against predefined SLOs and generates
    alerts when thresholds are breached.
    """
    
    def __init__(self):
        """Initialize SLO monitor."""
        self.slo_targets: Dict[str, List[SLOTarget]] = {}
        self.performance_metrics: Dict[str, pd.DataFrame] = {}
        self.alerts_history: List[PerformanceAlert] = []
    
    def add_slo_target(self, strategy_id: str, slo_target: SLOTarget):
        """
        Add SLO target for strategy.
        
        Parameters:
        - strategy_id: Strategy identifier
        - slo_target: SLO target specification
        """
        if strategy_id not in self.slo_targets:
            self.slo_targets[strategy_id] = []
        
        self.slo_targets[strategy_id].append(slo_target)
        logger.info(f"Added SLO target for {strategy_id}: {slo_target.metric_name}")
    
    def update_metrics(self, 
                      strategy_id: str,
                      metrics: Dict[str, float],
                      timestamp: Optional[datetime] = None):
        """
        Update performance metrics for strategy.
        
        Parameters:
        - strategy_id: Strategy identifier
        - metrics: Dictionary of metric values
        - timestamp: Metric timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize metrics DataFrame if needed
        if strategy_id not in self.performance_metrics:
            self.performance_metrics[strategy_id] = pd.DataFrame()
        
        # Add new metrics
        metrics_row = pd.DataFrame([metrics], index=[timestamp])
        
        if self.performance_metrics[strategy_id].empty:
            self.performance_metrics[strategy_id] = metrics_row
        else:
            self.performance_metrics[strategy_id] = pd.concat([
                self.performance_metrics[strategy_id], 
                metrics_row
            ])
        
        # Check SLO violations
        self._check_slo_violations(strategy_id, metrics, timestamp)
    
    def _check_slo_violations(self, 
                            strategy_id: str,
                            current_metrics: Dict[str, float],
                            timestamp: datetime):
        """Check for SLO violations and generate alerts."""
        if strategy_id not in self.slo_targets:
            return
        
        for slo_target in self.slo_targets[strategy_id]:
            metric_name = slo_target.metric_name
            
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            threshold = slo_target.target_value
            
            # Check threshold violation
            violation = False
            
            if slo_target.threshold_type == "min" and current_value < threshold:
                violation = True
                message = f"{metric_name} ({current_value:.4f}) below minimum threshold ({threshold:.4f})"
            
            elif slo_target.threshold_type == "max" and current_value > threshold:
                violation = True
                message = f"{metric_name} ({current_value:.4f}) above maximum threshold ({threshold:.4f})"
            
            if violation:
                alert = PerformanceAlert(
                    timestamp=timestamp,
                    strategy_id=strategy_id,
                    alert_level=slo_target.alert_level,
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=threshold,
                    message=message,
                    metadata={
                        'measurement_window': slo_target.measurement_window,
                        'threshold_type': slo_target.threshold_type
                    }
                )
                
                self.alerts_history.append(alert)
                
                logger.warning(f"SLO violation for {strategy_id}: {message}")
    
    def get_slo_compliance(self, 
                         strategy_id: str,
                         lookback_days: int = 30) -> Dict[str, float]:
        """
        Calculate SLO compliance rates.
        
        Parameters:
        - strategy_id: Strategy to analyze
        - lookback_days: Lookback period for compliance calculation
        
        Returns:
        - Dictionary of compliance rates by metric
        """
        if strategy_id not in self.performance_metrics:
            return {}
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_metrics = self.performance_metrics[strategy_id].loc[cutoff_date:]
        
        if recent_metrics.empty:
            return {}
        
        compliance_rates = {}
        
        if strategy_id in self.slo_targets:
            for slo_target in self.slo_targets[strategy_id]:
                metric_name = slo_target.metric_name
                
                if metric_name not in recent_metrics.columns:
                    continue
                
                metric_values = recent_metrics[metric_name].dropna()
                
                if len(metric_values) == 0:
                    continue
                
                # Calculate compliance rate
                if slo_target.threshold_type == "min":
                    compliant = (metric_values >= slo_target.target_value).sum()
                elif slo_target.threshold_type == "max":
                    compliant = (metric_values <= slo_target.target_value).sum()
                else:
                    compliant = len(metric_values)  # Default to 100% for range
                
                compliance_rate = compliant / len(metric_values)
                compliance_rates[metric_name] = compliance_rate
        
        return compliance_rates
    
    def get_recent_alerts(self, 
                         strategy_id: Optional[str] = None,
                         hours: int = 24) -> List[PerformanceAlert]:
        """
        Get recent alerts.
        
        Parameters:
        - strategy_id: Optional strategy filter
        - hours: Lookback hours
        
        Returns:
        - List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alerts_history
            if alert.timestamp >= cutoff_time
        ]
        
        if strategy_id is not None:
            recent_alerts = [
                alert for alert in recent_alerts
                if alert.strategy_id == strategy_id
            ]
        
        return recent_alerts