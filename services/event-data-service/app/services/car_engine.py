"""
Cumulative Abnormal Returns (CAR) Engine with Cluster-Robust Statistics

Enhanced event study analysis with cluster-robust standard errors to handle
correlation in observations. Implements sophisticated statistical methods
for accurate event impact assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import t as t_dist
from sklearn.linear_model import LinearRegression
import logging
import warnings

logger = logging.getLogger(__name__)


@dataclass
class EventWindow:
    """Event window specification."""
    pre_event_days: int
    post_event_days: int
    
    @property
    def total_days(self) -> int:
        return self.pre_event_days + 1 + self.post_event_days
    
    @property
    def event_day_index(self) -> int:
        return self.pre_event_days


@dataclass
class EstimationWindow:
    """Estimation window for normal performance model."""
    days_before_event: int = 250
    days_before_gap: int = 10  # Gap between estimation and event window
    
    def get_estimation_period(self, event_date: datetime) -> Tuple[datetime, datetime]:
        """Get estimation period dates."""
        end_date = event_date - timedelta(days=self.days_before_gap)
        start_date = end_date - timedelta(days=self.days_before_event)
        return start_date, end_date


@dataclass
class CARResult:
    """Cumulative Abnormal Returns analysis result."""
    # Basic CAR statistics
    cumulative_abnormal_returns: np.ndarray
    car_values: Dict[str, float]  # CAR for different windows
    mean_car: float
    median_car: float
    
    # Standard statistics
    standard_error: float
    t_statistic: float
    p_value: float
    
    # Cluster-robust statistics
    cluster_robust_se: float
    cluster_robust_t_stat: float
    cluster_robust_p_value: float
    n_clusters: int
    
    # Additional metrics
    abnormal_returns: np.ndarray
    n_events: int
    event_window: EventWindow
    
    # Significance flags
    is_significant_standard: bool
    is_significant_clustered: bool
    
    # Event characteristics
    positive_events: int
    negative_events: int
    event_dates: List[datetime]
    symbols: List[str]


@dataclass
class EventStudyInput:
    """Input data for event study analysis."""
    returns: pd.DataFrame  # Stock returns (date x symbol)
    market_returns: pd.Series  # Market returns (date)
    event_data: pd.DataFrame  # Event data (symbol, date, event_type, etc.)
    event_window: EventWindow
    estimation_window: EstimationWindow


class ClusterRobustCAR:
    """
    Cluster-robust Cumulative Abnormal Returns analysis.
    
    Implements event study methodology with cluster-robust standard errors
    to account for correlation in event observations (e.g., industry clustering,
    time clustering, etc.).
    """
    
    def __init__(self, cluster_method: str = 'time'):
        """
        Initialize CAR engine.
        
        Args:
            cluster_method: Clustering method ('time', 'industry', 'size', 'custom')
        """
        self.cluster_method = cluster_method
        
    def calculate_abnormal_returns(self, 
                                 stock_returns: pd.Series,
                                 market_returns: pd.Series,
                                 estimation_window: Tuple[datetime, datetime],
                                 model: str = 'market_model') -> pd.Series:
        """
        Calculate abnormal returns using specified model.
        
        Args:
            stock_returns: Stock returns time series
            market_returns: Market returns time series
            estimation_window: Estimation period (start, end)
            model: Model type ('market_model', 'mean_adjusted', 'market_adjusted')
            
        Returns:
            Abnormal returns time series
        """
        start_date, end_date = estimation_window
        
        # Filter estimation period data
        est_stock = stock_returns[(stock_returns.index >= start_date) & 
                                 (stock_returns.index <= end_date)]
        est_market = market_returns[(market_returns.index >= start_date) & 
                                   (market_returns.index <= end_date)]
        
        if len(est_stock) < 50:  # Minimum estimation period
            logger.warning(f"Short estimation period: {len(est_stock)} observations")
        
        if model == 'market_model':
            # Estimate market model: R_it = α_i + β_i * R_mt + ε_it
            aligned_data = pd.concat([est_stock, est_market], axis=1, join='inner')
            if len(aligned_data) < 30:
                raise ValueError("Insufficient data for market model estimation")
            
            X = aligned_data.iloc[:, 1].values.reshape(-1, 1)  # Market returns
            y = aligned_data.iloc[:, 0].values  # Stock returns
            
            model_fit = LinearRegression().fit(X, y)
            alpha = model_fit.intercept_
            beta = model_fit.coef_[0]
            
            # Calculate abnormal returns for full period
            expected_returns = alpha + beta * market_returns
            abnormal_returns = stock_returns - expected_returns
            
        elif model == 'mean_adjusted':
            # Mean-adjusted model: AR_it = R_it - μ_i
            mean_return = est_stock.mean()
            abnormal_returns = stock_returns - mean_return
            
        elif model == 'market_adjusted':
            # Market-adjusted model: AR_it = R_it - R_mt
            abnormal_returns = stock_returns - market_returns
            
        else:
            raise ValueError(f"Unknown model: {model}")
        
        return abnormal_returns.dropna()
    
    def extract_event_window_returns(self,
                                   abnormal_returns: pd.Series,
                                   event_date: datetime,
                                   event_window: EventWindow) -> np.ndarray:
        """Extract abnormal returns for event window."""
        start_date = event_date - timedelta(days=event_window.pre_event_days)
        end_date = event_date + timedelta(days=event_window.post_event_days)
        
        event_returns = abnormal_returns[(abnormal_returns.index >= start_date) & 
                                       (abnormal_returns.index <= end_date)]
        
        if len(event_returns) != event_window.total_days:
            logger.warning(f"Expected {event_window.total_days} days, got {len(event_returns)}")
        
        return event_returns.values
    
    def create_clusters(self, 
                       event_data: pd.DataFrame,
                       cluster_method: str = None) -> np.ndarray:
        """
        Create cluster assignments for events.
        
        Args:
            event_data: DataFrame with event information
            cluster_method: Clustering method override
            
        Returns:
            Array of cluster assignments
        """
        method = cluster_method or self.cluster_method
        
        if method == 'time':
            # Group by month-year
            dates = pd.to_datetime(event_data['event_date'])
            clusters = dates.dt.to_period('M').astype(str)
            
        elif method == 'industry':
            # Cluster by industry (requires industry column)
            if 'industry' not in event_data.columns:
                logger.warning("Industry column not found, falling back to time clustering")
                return self.create_clusters(event_data, 'time')
            clusters = event_data['industry']
            
        elif method == 'size':
            # Cluster by market cap quartiles
            if 'market_cap' not in event_data.columns:
                logger.warning("Market cap column not found, falling back to time clustering")
                return self.create_clusters(event_data, 'time')
            clusters = pd.qcut(event_data['market_cap'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
        elif method == 'custom':
            # Use custom cluster column
            if 'cluster' not in event_data.columns:
                raise ValueError("Custom clustering requires 'cluster' column in event_data")
            clusters = event_data['cluster']
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Convert to numeric codes for easier handling
        cluster_codes = pd.Categorical(clusters).codes
        return cluster_codes
    
    def calculate_cluster_robust_se(self,
                                  abnormal_returns_matrix: np.ndarray,
                                  clusters: np.ndarray) -> Tuple[float, int]:
        """
        Calculate cluster-robust standard errors.
        
        Args:
            abnormal_returns_matrix: Matrix of abnormal returns (events x time)
            clusters: Cluster assignments
            
        Returns:
            Tuple of (cluster_robust_se, n_clusters)
        """
        # Calculate cumulative abnormal returns for each event
        cars = np.sum(abnormal_returns_matrix, axis=1)
        
        # Get unique clusters
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)
        
        # Calculate cluster means
        cluster_means = np.array([
            np.mean(cars[clusters == cluster]) 
            for cluster in unique_clusters
        ])
        
        # Overall mean
        overall_mean = np.mean(cars)
        
        # Cluster-robust variance calculation
        cluster_variance = np.sum((cluster_means - overall_mean)**2) / (n_clusters - 1)
        cluster_robust_se = np.sqrt(cluster_variance / n_clusters)
        
        return cluster_robust_se, n_clusters
    
    def calculate_car_statistics(self,
                                abnormal_returns_matrix: np.ndarray,
                                clusters: np.ndarray,
                                event_window: EventWindow) -> Dict[str, float]:
        """Calculate various CAR statistics."""
        n_events, n_days = abnormal_returns_matrix.shape
        
        # Cumulative abnormal returns for each event
        car_matrix = np.cumsum(abnormal_returns_matrix, axis=1)
        
        # Mean CAR across all events
        mean_car = np.mean(car_matrix, axis=0)
        
        # Standard error (traditional)
        car_final = car_matrix[:, -1]  # Final CAR for each event
        standard_se = np.std(car_final, ddof=1) / np.sqrt(n_events)
        
        # Cluster-robust standard error
        cluster_robust_se, n_clusters = self.calculate_cluster_robust_se(
            abnormal_returns_matrix, clusters
        )
        
        # Test statistics
        mean_car_final = np.mean(car_final)
        t_stat_standard = mean_car_final / standard_se if standard_se > 0 else 0
        t_stat_clustered = mean_car_final / cluster_robust_se if cluster_robust_se > 0 else 0
        
        # P-values
        p_value_standard = 2 * (1 - t_dist.cdf(np.abs(t_stat_standard), n_events - 1))
        p_value_clustered = 2 * (1 - t_dist.cdf(np.abs(t_stat_clustered), n_clusters - 1))
        
        # Additional CAR windows
        car_windows = {}
        event_day_idx = event_window.event_day_index
        
        # Event day only
        car_windows['event_day'] = np.mean(abnormal_returns_matrix[:, event_day_idx])
        
        # Pre-event window
        if event_window.pre_event_days > 0:
            car_windows['pre_event'] = np.mean(np.sum(
                abnormal_returns_matrix[:, :event_day_idx], axis=1
            ))
        
        # Post-event window
        if event_window.post_event_days > 0:
            car_windows['post_event'] = np.mean(np.sum(
                abnormal_returns_matrix[:, event_day_idx+1:], axis=1
            ))
        
        # Short-term window (-1, +1)
        if event_window.pre_event_days >= 1 and event_window.post_event_days >= 1:
            car_windows['short_term'] = np.mean(np.sum(
                abnormal_returns_matrix[:, event_day_idx-1:event_day_idx+2], axis=1
            ))
        
        return {
            'mean_car': mean_car,
            'car_final': mean_car_final,
            'median_car': np.median(car_final),
            'standard_se': standard_se,
            'cluster_robust_se': cluster_robust_se,
            't_stat_standard': t_stat_standard,
            't_stat_clustered': t_stat_clustered,
            'p_value_standard': p_value_standard,
            'p_value_clustered': p_value_clustered,
            'n_clusters': n_clusters,
            'car_windows': car_windows,
            'positive_events': np.sum(car_final > 0),
            'negative_events': np.sum(car_final < 0)
        }
    
    def run_event_study(self, study_input: EventStudyInput) -> CARResult:
        """
        Run complete event study analysis.
        
        Args:
            study_input: Event study input data
            
        Returns:
            CARResult with comprehensive analysis
        """
        returns = study_input.returns
        market_returns = study_input.market_returns
        event_data = study_input.event_data
        event_window = study_input.event_window
        estimation_window = study_input.estimation_window
        
        # Store abnormal returns for all events
        abnormal_returns_list = []
        valid_events = []
        
        logger.info(f"Processing {len(event_data)} events")
        
        for idx, event_row in event_data.iterrows():
            symbol = event_row['symbol']
            event_date = pd.to_datetime(event_row['event_date'])
            
            try:
                # Get stock returns
                if symbol not in returns.columns:
                    logger.warning(f"No return data for symbol {symbol}")
                    continue
                
                stock_returns = returns[symbol]
                
                # Calculate estimation window
                est_start, est_end = estimation_window.get_estimation_period(event_date)
                
                # Calculate abnormal returns
                abnormal_returns = self.calculate_abnormal_returns(
                    stock_returns, market_returns, (est_start, est_end)
                )
                
                # Extract event window returns
                event_ar = self.extract_event_window_returns(
                    abnormal_returns, event_date, event_window
                )
                
                if len(event_ar) == event_window.total_days:
                    abnormal_returns_list.append(event_ar)
                    valid_events.append(event_row)
                else:
                    logger.warning(f"Incomplete event window for {symbol} on {event_date}")
                    
            except Exception as e:
                logger.error(f"Error processing event {symbol} on {event_date}: {e}")
                continue
        
        if len(abnormal_returns_list) == 0:
            raise ValueError("No valid events found for analysis")
        
        # Convert to matrix
        abnormal_returns_matrix = np.array(abnormal_returns_list)
        valid_events_df = pd.DataFrame(valid_events)
        
        # Create clusters
        clusters = self.create_clusters(valid_events_df)
        
        # Calculate statistics
        stats = self.calculate_car_statistics(
            abnormal_returns_matrix, clusters, event_window
        )
        
        # Create result object
        result = CARResult(
            cumulative_abnormal_returns=stats['mean_car'],
            car_values=stats['car_windows'],
            mean_car=stats['car_final'],
            median_car=stats['median_car'],
            standard_error=stats['standard_se'],
            t_statistic=stats['t_stat_standard'],
            p_value=stats['p_value_standard'],
            cluster_robust_se=stats['cluster_robust_se'],
            cluster_robust_t_stat=stats['t_stat_clustered'],
            cluster_robust_p_value=stats['p_value_clustered'],
            n_clusters=stats['n_clusters'],
            abnormal_returns=abnormal_returns_matrix,
            n_events=len(abnormal_returns_list),
            event_window=event_window,
            is_significant_standard=stats['p_value_standard'] < 0.05,
            is_significant_clustered=stats['p_value_clustered'] < 0.05,
            positive_events=stats['positive_events'],
            negative_events=stats['negative_events'],
            event_dates=[pd.to_datetime(event['event_date']) for event in valid_events],
            symbols=[event['symbol'] for event in valid_events]
        )
        
        logger.info(f"Event study completed: {result.n_events} events, "
                   f"{result.n_clusters} clusters, "
                   f"CAR={result.mean_car:.4f} "
                   f"(clustered p={result.cluster_robust_p_value:.4f})")
        
        return result
    
    def export_results(self, result: CARResult) -> Dict[str, Any]:
        """Export results in API-friendly format."""
        return {
            'summary': {
                'mean_car': result.mean_car,
                'median_car': result.median_car,
                'n_events': result.n_events,
                'n_clusters': result.n_clusters,
                'positive_events': result.positive_events,
                'negative_events': result.negative_events
            },
            'statistical_tests': {
                'standard': {
                    'standard_error': result.standard_error,
                    't_statistic': result.t_statistic,
                    'p_value': result.p_value,
                    'is_significant': result.is_significant_standard
                },
                'cluster_robust': {
                    'cluster_robust_se': result.cluster_robust_se,
                    't_stat_clustered': result.cluster_robust_t_stat,
                    'p_value_clustered': result.cluster_robust_p_value,
                    'is_significant': result.is_significant_clustered
                }
            },
            'car_by_window': result.car_values,
            'cumulative_returns': result.cumulative_abnormal_returns.tolist(),
            'event_window': {
                'pre_event_days': result.event_window.pre_event_days,
                'post_event_days': result.event_window.post_event_days,
                'total_days': result.event_window.total_days
            },
            'clustering_method': self.cluster_method,
            'analysis_timestamp': datetime.now().isoformat()
        }


# Utility functions for common event study patterns

def earnings_announcement_study(earnings_data: pd.DataFrame,
                              stock_returns: pd.DataFrame,
                              market_returns: pd.Series,
                              pre_days: int = 5,
                              post_days: int = 5) -> CARResult:
    """
    Convenience function for earnings announcement event study.
    
    Args:
        earnings_data: DataFrame with columns ['symbol', 'announcement_date', 'industry']
        stock_returns: DataFrame with stock returns (date x symbol)
        market_returns: Market returns time series
        pre_days: Days before announcement
        post_days: Days after announcement
        
    Returns:
        CARResult object
    """
    # Prepare event data
    event_data = earnings_data.rename(columns={'announcement_date': 'event_date'})
    
    # Set up analysis parameters
    event_window = EventWindow(pre_event_days=pre_days, post_event_days=post_days)
    estimation_window = EstimationWindow()
    
    study_input = EventStudyInput(
        returns=stock_returns,
        market_returns=market_returns,
        event_data=event_data,
        event_window=event_window,
        estimation_window=estimation_window
    )
    
    # Run analysis with industry clustering
    car_engine = ClusterRobustCAR(cluster_method='industry')
    return car_engine.run_event_study(study_input)


def merger_announcement_study(merger_data: pd.DataFrame,
                            stock_returns: pd.DataFrame,
                            market_returns: pd.Series) -> CARResult:
    """
    Convenience function for merger announcement event study.
    """
    event_data = merger_data.rename(columns={'announcement_date': 'event_date'})
    
    # Longer window for merger analysis
    event_window = EventWindow(pre_event_days=10, post_event_days=20)
    estimation_window = EstimationWindow(days_before_event=250)
    
    study_input = EventStudyInput(
        returns=stock_returns,
        market_returns=market_returns,
        event_data=event_data,
        event_window=event_window,
        estimation_window=estimation_window
    )
    
    # Use time clustering for mergers
    car_engine = ClusterRobustCAR(cluster_method='time')
    return car_engine.run_event_study(study_input)


def regulatory_event_study(regulatory_data: pd.DataFrame,
                         stock_returns: pd.DataFrame,
                         market_returns: pd.Series) -> CARResult:
    """
    Convenience function for regulatory event study.
    """
    event_data = regulatory_data.rename(columns={'announcement_date': 'event_date'})
    
    # Medium window for regulatory events
    event_window = EventWindow(pre_event_days=3, post_event_days=10)
    estimation_window = EstimationWindow()
    
    study_input = EventStudyInput(
        returns=stock_returns,
        market_returns=market_returns,
        event_data=event_data,
        event_window=event_window,
        estimation_window=estimation_window
    )
    
    # Use industry clustering for regulatory events
    car_engine = ClusterRobustCAR(cluster_method='industry')
    return car_engine.run_event_study(study_input)