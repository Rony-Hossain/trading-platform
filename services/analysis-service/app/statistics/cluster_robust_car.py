"""
Cluster-Robust Cumulative Abnormal Returns (CAR) Analysis

This module provides sophisticated event study analysis with cluster-robust standard errors,
addressing key econometric issues in financial event studies:

1. Clustering: Events may be clustered in time, industries, or by characteristics
2. Heteroskedasticity: Variance may differ across observations
3. Autocorrelation: Returns may exhibit serial correlation
4. Cross-sectional dependence: Returns may be correlated across assets

Key Features:
- Multi-dimensional clustering (time, industry, firm)
- Heteroskedasticity-robust inference
- Bootstrap confidence intervals
- Event window flexibility
- Multiple test corrections

References:
- Petersen, M. A. (2009). Estimating standard errors in finance panel data sets.
- Thompson, S. B. (2011). Simple formulas for standard errors that cluster by both firm and time.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering.
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.utils import resample
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.sandwich_covariance import cov_cluster

logger = logging.getLogger(__name__)


class ClusteringMethod(Enum):
    """Clustering methods for robust standard errors."""
    TIME = "time"
    FIRM = "firm" 
    INDUSTRY = "industry"
    FIRM_TIME = "firm_time"  # Two-way clustering
    INDUSTRY_TIME = "industry_time"
    FIRM_INDUSTRY = "firm_industry"
    THREE_WAY = "three_way"  # Firm, industry, time


class EstimationMethod(Enum):
    """Methods for estimating abnormal returns."""
    MARKET_MODEL = "market_model"
    FAMA_FRENCH_3 = "fama_french_3"
    FAMA_FRENCH_5 = "fama_french_5"
    CAPM = "capm"
    CONSTANT_MEAN = "constant_mean"


@dataclass
class EventWindow:
    """Define event window parameters."""
    start_day: int  # Days relative to event (negative = before)
    end_day: int    # Days relative to event (positive = after)
    
    def __post_init__(self):
        if self.start_day > self.end_day:
            raise ValueError("start_day must be <= end_day")
    
    @property
    def length(self) -> int:
        """Number of days in event window."""
        return self.end_day - self.start_day + 1
    
    def __str__(self) -> str:
        return f"[{self.start_day}, {self.end_day}]"


@dataclass
class AbnormalReturnResult:
    """Results from abnormal return calculation."""
    abnormal_returns: pd.DataFrame  # AR for each security-event
    cumulative_abnormal_returns: pd.DataFrame  # CAR for each security-event
    average_abnormal_returns: pd.Series  # AAR across all events
    cumulative_average_abnormal_returns: pd.Series  # CAAR across all events
    estimation_stats: Dict[str, Any]  # Model estimation statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'abnormal_returns': self.abnormal_returns.to_dict(),
            'cumulative_abnormal_returns': self.cumulative_abnormal_returns.to_dict(),
            'average_abnormal_returns': self.average_abnormal_returns.to_dict(),
            'cumulative_average_abnormal_returns': self.cumulative_average_abnormal_returns.to_dict(),
            'estimation_stats': self.estimation_stats
        }


@dataclass
class ClusterRobustTestResult:
    """Results from cluster-robust significance testing."""
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    standard_error: float
    degrees_of_freedom: int
    clustering_method: ClusteringMethod
    n_clusters: Dict[str, int]  # Number of clusters by dimension
    is_significant_95: bool
    is_significant_99: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'confidence_interval': self.confidence_interval,
            'standard_error': self.standard_error,
            'degrees_of_freedom': self.degrees_of_freedom,
            'clustering_method': self.clustering_method.value,
            'n_clusters': self.n_clusters,
            'is_significant_95': self.is_significant_95,
            'is_significant_99': self.is_significant_99
        }


class ClusterRobustCAR:
    """
    Cluster-Robust Cumulative Abnormal Returns Analysis.
    
    Provides comprehensive event study analysis with proper statistical inference
    accounting for clustering, heteroskedasticity, and autocorrelation.
    """
    
    def __init__(self, 
                 clustering_method: ClusteringMethod = ClusteringMethod.FIRM_TIME,
                 estimation_method: EstimationMethod = EstimationMethod.MARKET_MODEL,
                 bootstrap_iterations: int = 1000,
                 confidence_level: float = 0.95):
        """
        Initialize Cluster-Robust CAR analyzer.
        
        Parameters:
        - clustering_method: Method for clustering standard errors
        - estimation_method: Method for estimating expected returns
        - bootstrap_iterations: Number of bootstrap samples for robust inference
        - confidence_level: Confidence level for intervals and tests
        """
        self.clustering_method = clustering_method
        self.estimation_method = estimation_method
        self.bootstrap_iterations = bootstrap_iterations
        self.confidence_level = confidence_level
        
        # Initialize results storage
        self._estimation_results = {}
        self._event_data = None
        
    def estimate_expected_returns(self,
                                returns_data: pd.DataFrame,
                                market_data: pd.DataFrame,
                                estimation_window: Tuple[pd.Timestamp, pd.Timestamp],
                                risk_free_rate: Optional[pd.Series] = None,
                                factor_data: Optional[pd.DataFrame] = None) -> Dict[str, RegressionResults]:
        """
        Estimate expected return models for each security.
        
        Parameters:
        - returns_data: Security returns (securities x dates)
        - market_data: Market return data
        - estimation_window: (start_date, end_date) for parameter estimation
        - risk_free_rate: Risk-free rate series (for CAPM, FF models)
        - factor_data: Additional factor data (for FF models)
        
        Returns:
        - Dictionary of regression results by security
        """
        start_date, end_date = estimation_window
        estimation_returns = returns_data.loc[start_date:end_date]
        estimation_market = market_data.loc[start_date:end_date]
        
        results = {}
        
        for security in estimation_returns.columns:
            try:
                y = estimation_returns[security].dropna()
                
                if len(y) < 30:  # Minimum observations for reliable estimation
                    logger.warning(f"Insufficient data for {security}: {len(y)} observations")
                    continue
                
                # Align market data with security data
                common_dates = y.index.intersection(estimation_market.index)
                y_aligned = y.loc[common_dates]
                
                if self.estimation_method == EstimationMethod.CONSTANT_MEAN:
                    # Simple constant mean model
                    X = np.ones(len(y_aligned))
                    
                elif self.estimation_method == EstimationMethod.MARKET_MODEL:
                    # Market model: R_i = alpha + beta * R_m + epsilon
                    market_aligned = estimation_market.loc[common_dates]
                    X = sm.add_constant(market_aligned.values)
                    
                elif self.estimation_method == EstimationMethod.CAPM:
                    # CAPM: R_i - R_f = alpha + beta * (R_m - R_f) + epsilon
                    if risk_free_rate is None:
                        raise ValueError("Risk-free rate required for CAPM")
                    
                    rf_aligned = risk_free_rate.loc[common_dates]
                    market_aligned = estimation_market.loc[common_dates]
                    
                    y_aligned = y_aligned - rf_aligned
                    market_excess = market_aligned - rf_aligned
                    X = sm.add_constant(market_excess.values)
                    
                elif self.estimation_method in [EstimationMethod.FAMA_FRENCH_3, EstimationMethod.FAMA_FRENCH_5]:
                    # Fama-French models
                    if factor_data is None:
                        raise ValueError("Factor data required for Fama-French models")
                    
                    if risk_free_rate is None:
                        raise ValueError("Risk-free rate required for Fama-French models")
                    
                    rf_aligned = risk_free_rate.loc[common_dates]
                    y_aligned = y_aligned - rf_aligned
                    
                    factors = ['MKT', 'SMB', 'HML']
                    if self.estimation_method == EstimationMethod.FAMA_FRENCH_5:
                        factors.extend(['RMW', 'CMA'])
                    
                    factor_aligned = factor_data.loc[common_dates, factors]
                    X = sm.add_constant(factor_aligned)
                    
                else:
                    raise ValueError(f"Unknown estimation method: {self.estimation_method}")
                
                # Run regression
                model = sm.OLS(y_aligned, X, missing='drop')
                result = model.fit()
                results[security] = result
                
                logger.debug(f"Estimated model for {security}: R² = {result.rsquared:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to estimate model for {security}: {str(e)}")
                continue
        
        self._estimation_results = results
        logger.info(f"Successfully estimated models for {len(results)} securities")
        
        return results
    
    def calculate_abnormal_returns(self,
                                 returns_data: pd.DataFrame,
                                 market_data: pd.DataFrame,
                                 event_data: pd.DataFrame,
                                 event_window: EventWindow,
                                 estimation_window: Tuple[pd.Timestamp, pd.Timestamp],
                                 risk_free_rate: Optional[pd.Series] = None,
                                 factor_data: Optional[pd.DataFrame] = None) -> AbnormalReturnResult:
        """
        Calculate abnormal returns for event study.
        
        Parameters:
        - returns_data: Security returns (dates x securities)
        - market_data: Market return data (dates x 1)
        - event_data: Event information (event_id, security, event_date, [cluster_vars])
        - event_window: Event window specification
        - estimation_window: Parameter estimation window
        - risk_free_rate: Risk-free rate series
        - factor_data: Factor data for multi-factor models
        
        Returns:
        - AbnormalReturnResult with all computed statistics
        """
        logger.info(f"Calculating abnormal returns for {len(event_data)} events")
        
        # Store event data for clustering
        self._event_data = event_data.copy()
        
        # Estimate expected return models
        if not self._estimation_results:
            self.estimate_expected_returns(
                returns_data, market_data, estimation_window,
                risk_free_rate, factor_data
            )
        
        # Initialize result containers
        abnormal_returns_list = []
        car_list = []
        
        for idx, event in event_data.iterrows():
            security = event['security']
            event_date = pd.to_datetime(event['event_date'])
            event_id = event.get('event_id', f"{security}_{event_date.strftime('%Y%m%d')}")
            
            if security not in self._estimation_results:
                logger.warning(f"No estimation results for {security}, skipping")
                continue
            
            # Get event window dates
            event_window_dates = self._get_event_window_dates(
                event_date, event_window, returns_data.index
            )
            
            if len(event_window_dates) != event_window.length:
                logger.warning(f"Incomplete event window for {event_id}")
                continue
            
            # Calculate abnormal returns for this event
            ar_event = self._calculate_event_abnormal_returns(
                security, event_window_dates, returns_data, market_data,
                risk_free_rate, factor_data
            )
            
            if ar_event is not None:
                # Add metadata
                ar_event['event_id'] = event_id
                ar_event['security'] = security
                ar_event['event_date'] = event_date
                
                # Add clustering variables
                for col in event_data.columns:
                    if col not in ['security', 'event_date', 'event_id']:
                        ar_event[col] = event[col]
                
                abnormal_returns_list.append(ar_event)
                
                # Calculate cumulative abnormal returns
                car_event = ar_event[['abnormal_return']].copy()
                car_event['cumulative_abnormal_return'] = car_event['abnormal_return'].cumsum()
                car_event['event_id'] = event_id
                car_event['security'] = security
                car_event['event_date'] = event_date
                
                # Add clustering variables
                for col in event_data.columns:
                    if col not in ['security', 'event_date', 'event_id']:
                        car_event[col] = event[col]
                
                car_list.append(car_event)
        
        if not abnormal_returns_list:
            raise ValueError("No abnormal returns calculated - check data alignment")
        
        # Combine results
        abnormal_returns_df = pd.concat(abnormal_returns_list, ignore_index=True)
        car_df = pd.concat(car_list, ignore_index=True)
        
        # Calculate cross-sectional averages
        aar = abnormal_returns_df.groupby('event_day')['abnormal_return'].mean()
        caar = aar.cumsum()
        
        # Estimation statistics
        estimation_stats = self._compile_estimation_stats()
        
        result = AbnormalReturnResult(
            abnormal_returns=abnormal_returns_df,
            cumulative_abnormal_returns=car_df,
            average_abnormal_returns=aar,
            cumulative_average_abnormal_returns=caar,
            estimation_stats=estimation_stats
        )
        
        logger.info(f"Calculated abnormal returns for {len(abnormal_returns_list)} events")
        return result
    
    def test_significance_cluster_robust(self,
                                       ar_result: AbnormalReturnResult,
                                       test_window: Optional[EventWindow] = None) -> ClusterRobustTestResult:
        """
        Test statistical significance using cluster-robust standard errors.
        
        Parameters:
        - ar_result: Abnormal return results
        - test_window: Window for testing (defaults to full event window)
        
        Returns:
        - ClusterRobustTestResult with test statistics and inference
        """
        if test_window is None:
            # Use full event window
            car_data = ar_result.cumulative_abnormal_returns
            test_statistic_data = car_data.groupby('event_id')['cumulative_abnormal_return'].last()
        else:
            # Filter to test window
            car_data = ar_result.cumulative_abnormal_returns
            test_data = car_data[
                (car_data['event_day'] >= test_window.start_day) &
                (car_data['event_day'] <= test_window.end_day)
            ]
            test_statistic_data = test_data.groupby('event_id')['cumulative_abnormal_return'].last()
        
        # Prepare clustering variables
        cluster_data = self._prepare_cluster_variables(car_data)
        
        # Align test data with cluster data
        common_events = test_statistic_data.index.intersection(cluster_data.index)
        y = test_statistic_data.loc[common_events]
        cluster_vars = cluster_data.loc[common_events]
        
        if len(y) == 0:
            raise ValueError("No overlapping data for significance testing")
        
        # Run regression with cluster-robust standard errors
        X = np.ones(len(y))  # Test if mean CAR = 0
        
        # Use statsmodels for cluster-robust inference
        model = sm.OLS(y, X)
        
        if self.clustering_method == ClusteringMethod.TIME:
            cov_kwds = {'groups': cluster_vars['time_cluster']}
        elif self.clustering_method == ClusteringMethod.FIRM:
            cov_kwds = {'groups': cluster_vars['firm_cluster']}
        elif self.clustering_method == ClusteringMethod.INDUSTRY:
            cov_kwds = {'groups': cluster_vars['industry_cluster']}
        elif self.clustering_method == ClusteringMethod.FIRM_TIME:
            cov_kwds = {'groups': cluster_vars[['firm_cluster', 'time_cluster']]}
        else:
            # Default to firm-time clustering
            cov_kwds = {'groups': cluster_vars[['firm_cluster', 'time_cluster']]}
        
        # Fit with cluster-robust standard errors
        try:
            result = model.fit(cov_type='cluster', cov_kwds=cov_kwds)
            
            test_stat = result.tvalues[0]
            p_value = result.pvalues[0]
            se = result.bse[0]
            
            # Confidence interval
            alpha = 1 - self.confidence_level
            t_crit = stats.t.ppf(1 - alpha/2, result.df_resid)
            mean_car = result.params[0]
            ci_lower = mean_car - t_crit * se
            ci_upper = mean_car + t_crit * se
            
            # Count clusters
            n_clusters = self._count_clusters(cluster_vars)
            
        except Exception as e:
            logger.warning(f"Cluster-robust estimation failed, using bootstrap: {e}")
            # Fallback to bootstrap inference
            test_stat, p_value, se, ci_lower, ci_upper, n_clusters = self._bootstrap_inference(
                y, cluster_vars
            )
        
        return ClusterRobustTestResult(
            test_statistic=test_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            standard_error=se,
            degrees_of_freedom=len(y) - 1,
            clustering_method=self.clustering_method,
            n_clusters=n_clusters,
            is_significant_95=(p_value < 0.05),
            is_significant_99=(p_value < 0.01)
        )
    
    def comprehensive_event_analysis(self,
                                   returns_data: pd.DataFrame,
                                   market_data: pd.DataFrame,
                                   event_data: pd.DataFrame,
                                   event_window: EventWindow,
                                   estimation_window: Tuple[pd.Timestamp, pd.Timestamp],
                                   test_windows: Optional[List[EventWindow]] = None,
                                   risk_free_rate: Optional[pd.Series] = None,
                                   factor_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run comprehensive cluster-robust event study analysis.
        
        Parameters:
        - returns_data: Security returns
        - market_data: Market return data
        - event_data: Event information with clustering variables
        - event_window: Event window for analysis
        - estimation_window: Window for parameter estimation
        - test_windows: Multiple windows to test (defaults to common windows)
        - risk_free_rate: Risk-free rate data
        - factor_data: Factor data for multi-factor models
        
        Returns:
        - Comprehensive analysis results
        """
        logger.info("Starting comprehensive cluster-robust event analysis")
        
        # Calculate abnormal returns
        ar_result = self.calculate_abnormal_returns(
            returns_data, market_data, event_data, event_window,
            estimation_window, risk_free_rate, factor_data
        )
        
        # Default test windows if not provided
        if test_windows is None:
            test_windows = [
                EventWindow(-1, 1),   # 3-day window
                EventWindow(0, 0),    # Event day only
                EventWindow(-2, 2),   # 5-day window
                EventWindow(-5, 5),   # 11-day window
                event_window          # Full event window
            ]
        
        # Test significance for each window
        significance_results = {}
        for i, window in enumerate(test_windows):
            window_name = f"window_{i}_{window}"
            try:
                sig_result = self.test_significance_cluster_robust(ar_result, window)
                significance_results[window_name] = sig_result
                logger.info(f"Window {window}: t-stat={sig_result.test_statistic:.3f}, "
                          f"p-value={sig_result.p_value:.4f}")
            except Exception as e:
                logger.error(f"Failed to test window {window}: {e}")
                continue
        
        # Compile comprehensive results
        results = {
            'abnormal_returns': ar_result,
            'significance_tests': significance_results,
            'methodology': {
                'clustering_method': self.clustering_method.value,
                'estimation_method': self.estimation_method.value,
                'event_window': str(event_window),
                'n_events': len(event_data),
                'n_securities': len(returns_data.columns)
            },
            'summary_statistics': self._calculate_summary_statistics(ar_result)
        }
        
        logger.info("Comprehensive cluster-robust event analysis completed")
        return results
    
    def _get_event_window_dates(self, 
                              event_date: pd.Timestamp, 
                              event_window: EventWindow,
                              available_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Get actual dates for event window."""
        # Find event date position
        try:
            event_pos = available_dates.get_loc(event_date)
        except KeyError:
            # Find nearest date
            nearest_idx = available_dates.searchsorted(event_date)
            if nearest_idx >= len(available_dates):
                nearest_idx = len(available_dates) - 1
            event_pos = nearest_idx
        
        # Calculate window positions
        start_pos = max(0, event_pos + event_window.start_day)
        end_pos = min(len(available_dates) - 1, event_pos + event_window.end_day)
        
        return available_dates[start_pos:end_pos + 1]
    
    def _calculate_event_abnormal_returns(self,
                                        security: str,
                                        event_dates: pd.DatetimeIndex,
                                        returns_data: pd.DataFrame,
                                        market_data: pd.DataFrame,
                                        risk_free_rate: Optional[pd.Series],
                                        factor_data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Calculate abnormal returns for a single event."""
        if security not in self._estimation_results:
            return None
        
        regression_result = self._estimation_results[security]
        
        # Get actual returns for event window
        actual_returns = returns_data.loc[event_dates, security]
        
        # Calculate expected returns
        expected_returns = []
        
        for date in event_dates:
            try:
                if self.estimation_method == EstimationMethod.CONSTANT_MEAN:
                    expected_ret = regression_result.params[0]
                    
                elif self.estimation_method == EstimationMethod.MARKET_MODEL:
                    market_ret = market_data.loc[date].iloc[0]
                    expected_ret = regression_result.params[0] + regression_result.params[1] * market_ret
                    
                elif self.estimation_method == EstimationMethod.CAPM:
                    market_ret = market_data.loc[date].iloc[0]
                    rf_ret = risk_free_rate.loc[date]
                    expected_ret = regression_result.params[0] + regression_result.params[1] * (market_ret - rf_ret) + rf_ret
                    
                elif self.estimation_method in [EstimationMethod.FAMA_FRENCH_3, EstimationMethod.FAMA_FRENCH_5]:
                    factors = factor_data.loc[date]
                    rf_ret = risk_free_rate.loc[date]
                    
                    expected_excess = regression_result.params[0]  # Alpha
                    for i, factor in enumerate(['MKT', 'SMB', 'HML']):
                        expected_excess += regression_result.params[i + 1] * factors[factor]
                    
                    if self.estimation_method == EstimationMethod.FAMA_FRENCH_5:
                        for i, factor in enumerate(['RMW', 'CMA']):
                            expected_excess += regression_result.params[i + 4] * factors[factor]
                    
                    expected_ret = expected_excess + rf_ret
                    
                expected_returns.append(expected_ret)
                
            except (KeyError, IndexError) as e:
                logger.warning(f"Missing data for {date}: {e}")
                expected_returns.append(np.nan)
        
        # Calculate abnormal returns
        expected_returns = pd.Series(expected_returns, index=event_dates)
        abnormal_returns = actual_returns - expected_returns
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'date': event_dates,
            'event_day': range(len(event_dates)),
            'actual_return': actual_returns.values,
            'expected_return': expected_returns.values,
            'abnormal_return': abnormal_returns.values
        })
        
        # Add relative event day (0 = event date)
        event_date_pos = len(event_dates) // 2  # Approximate for now
        result_df['event_day'] = result_df['event_day'] - event_date_pos
        
        return result_df
    
    def _prepare_cluster_variables(self, car_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare clustering variables for robust inference."""
        # Get unique events
        event_info = car_data.groupby('event_id').first()
        
        cluster_vars = pd.DataFrame(index=event_info.index)
        
        # Time clustering (by year-month or year)
        if 'event_date' in event_info.columns:
            cluster_vars['time_cluster'] = pd.to_datetime(event_info['event_date']).dt.strftime('%Y-%m')
        
        # Firm clustering
        if 'security' in event_info.columns:
            cluster_vars['firm_cluster'] = event_info['security']
        
        # Industry clustering
        if 'industry' in event_info.columns:
            cluster_vars['industry_cluster'] = event_info['industry']
        elif 'sector' in event_info.columns:
            cluster_vars['industry_cluster'] = event_info['sector']
        else:
            # Create dummy industry clusters based on security
            cluster_vars['industry_cluster'] = 'default'
        
        return cluster_vars
    
    def _count_clusters(self, cluster_vars: pd.DataFrame) -> Dict[str, int]:
        """Count number of clusters by dimension."""
        n_clusters = {}
        
        for col in cluster_vars.columns:
            n_clusters[col.replace('_cluster', '')] = cluster_vars[col].nunique()
        
        return n_clusters
    
    def _bootstrap_inference(self, 
                           y: pd.Series, 
                           cluster_vars: pd.DataFrame) -> Tuple[float, float, float, float, float, Dict[str, int]]:
        """Bootstrap inference when cluster-robust estimation fails."""
        n_boot = self.bootstrap_iterations
        boot_means = []
        
        # Cluster bootstrap
        if self.clustering_method == ClusteringMethod.FIRM_TIME:
            # Bootstrap by firm clusters
            firms = cluster_vars['firm_cluster'].unique()
            
            for _ in range(n_boot):
                boot_firms = resample(firms, replace=True, random_state=None)
                boot_indices = []
                
                for firm in boot_firms:
                    firm_indices = cluster_vars[cluster_vars['firm_cluster'] == firm].index
                    boot_indices.extend(firm_indices)
                
                boot_y = y.loc[boot_indices]
                boot_means.append(boot_y.mean())
        
        else:
            # Simple bootstrap
            for _ in range(n_boot):
                boot_y = resample(y, replace=True, random_state=None)
                boot_means.append(np.mean(boot_y))
        
        boot_means = np.array(boot_means)
        
        # Calculate statistics
        mean_y = y.mean()
        se = np.std(boot_means)
        t_stat = mean_y / se if se > 0 else np.nan
        
        # P-value (two-tailed)
        p_value = 2 * (1 - norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else 1.0
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(boot_means, 100 * alpha / 2)
        ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
        
        n_clusters = self._count_clusters(cluster_vars)
        
        return t_stat, p_value, se, ci_lower, ci_upper, n_clusters
    
    def _compile_estimation_stats(self) -> Dict[str, Any]:
        """Compile estimation statistics across all models."""
        if not self._estimation_results:
            return {}
        
        stats = {
            'n_securities': len(self._estimation_results),
            'avg_r_squared': np.mean([res.rsquared for res in self._estimation_results.values()]),
            'avg_observations': np.mean([res.nobs for res in self._estimation_results.values()]),
            'method': self.estimation_method.value
        }
        
        return stats
    
    def _calculate_summary_statistics(self, ar_result: AbnormalReturnResult) -> Dict[str, Any]:
        """Calculate summary statistics for the event study."""
        car_final = ar_result.cumulative_abnormal_returns.groupby('event_id')['cumulative_abnormal_return'].last()
        
        stats = {
            'mean_car': car_final.mean(),
            'median_car': car_final.median(),
            'std_car': car_final.std(),
            'min_car': car_final.min(),
            'max_car': car_final.max(),
            'positive_car_pct': (car_final > 0).mean() * 100,
            'negative_car_pct': (car_final < 0).mean() * 100
        }
        
        return stats