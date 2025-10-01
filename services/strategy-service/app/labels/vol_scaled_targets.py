"""
Vol-Scaled Targets for Trading Strategy Labels

Implementation of volatility-scaled target normalization to stabilize training
across different market regimes and volatility environments.

Key concepts:
- Normalize returns by rolling volatility (σ) or implied volatility (IV30)
- Store both raw and vol-scaled returns for comparison
- Variance reduction validation using Levene's test
- Preserve out-of-sample lift while stabilizing training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.stats import levene
import warnings

from .triple_barrier import (
    TripleBarrierConfig, 
    TripleBarrierLabel, 
    TripleBarrierLabeler
)

logger = logging.getLogger(__name__)


@dataclass
class VolScaledConfig:
    """Configuration for vol-scaled targets."""
    
    # Volatility estimation
    vol_method: Literal['realized', 'ewm', 'garch', 'iv30'] = 'realized'
    vol_lookback: int = 20  # Days for volatility calculation
    vol_min_threshold: float = 0.005  # Minimum volatility (0.5% annualized)
    vol_max_threshold: float = 2.0  # Maximum volatility (200% annualized)
    
    # EWM parameters (if vol_method = 'ewm')
    ewm_halflife: int = 10  # Half-life for exponential weighting
    
    # GARCH parameters (if vol_method = 'garch')
    garch_p: int = 1  # GARCH(p,q) - p parameter
    garch_q: int = 1  # GARCH(p,q) - q parameter
    
    # Scaling parameters
    enable_vol_scaling: bool = True
    scaling_factor: float = 1.0  # Additional scaling factor
    outlier_clip_std: float = 3.0  # Clip outliers beyond N standard deviations
    
    # Validation parameters
    validation_window: int = 252  # Days for validation testing
    levene_significance: float = 0.05  # P-value threshold for Levene test
    
    # Store both versions
    store_raw_returns: bool = True
    store_vol_scaled_returns: bool = True


@dataclass 
class VolScaledLabel:
    """Extended triple barrier label with vol-scaled targets."""
    
    # Original triple barrier fields
    timestamp: datetime
    symbol: str
    entry_price: float
    target_price: float
    stop_loss_price: float
    expiry_time: datetime
    label: int
    hit_barrier: str
    exit_price: float
    exit_time: datetime
    holding_time: int
    
    # Raw returns
    return_pct: float
    raw_return: float = field(init=False)  # Same as return_pct for clarity
    
    # Volatility information
    entry_volatility: float = 0.0  # Volatility at entry
    realized_volatility: float = 0.0  # Realized vol during holding period
    iv30: Optional[float] = None  # Implied volatility if available
    
    # Vol-scaled returns
    vol_scaled_return: float = 0.0  # Return normalized by volatility
    vol_scaling_factor: float = 1.0  # Factor used for scaling
    
    # Metadata
    vol_method: str = 'realized'
    sample_weight: float = 1.0
    upper_hit_ts: Optional[datetime] = None
    lower_hit_ts: Optional[datetime] = None
    time_expiry: Optional[datetime] = None
    
    def __post_init__(self):
        """Set raw_return after initialization."""
        self.raw_return = self.return_pct


class VolatilityEstimator:
    """Volatility estimation using various methods."""
    
    def __init__(self, config: VolScaledConfig):
        self.config = config
        
    def estimate_realized_volatility(self, 
                                   prices: pd.Series,
                                   lookback: Optional[int] = None) -> pd.Series:
        """Calculate realized volatility using log returns."""
        
        lookback = lookback or self.config.vol_lookback
        
        log_returns = np.log(prices / prices.shift(1))
        vol = log_returns.rolling(
            window=lookback, 
            min_periods=max(5, lookback//4)
        ).std()
        
        # Annualize (252 trading days)
        vol = vol * np.sqrt(252)
        
        # Apply bounds
        vol = vol.clip(
            lower=self.config.vol_min_threshold,
            upper=self.config.vol_max_threshold
        )
        
        return vol.fillna(method='bfill').fillna(self.config.vol_min_threshold)
    
    def estimate_ewm_volatility(self, 
                              prices: pd.Series,
                              halflife: Optional[int] = None) -> pd.Series:
        """Calculate EWM volatility."""
        
        halflife = halflife or self.config.ewm_halflife
        
        log_returns = np.log(prices / prices.shift(1))
        vol = log_returns.ewm(halflife=halflife).std()
        
        # Annualize
        vol = vol * np.sqrt(252)
        
        # Apply bounds
        vol = vol.clip(
            lower=self.config.vol_min_threshold,
            upper=self.config.vol_max_threshold
        )
        
        return vol.fillna(method='bfill').fillna(self.config.vol_min_threshold)
    
    def estimate_garch_volatility(self, 
                                prices: pd.Series,
                                p: Optional[int] = None,
                                q: Optional[int] = None) -> pd.Series:
        """
        Estimate GARCH volatility.
        
        Note: This is a simplified implementation. For production,
        consider using arch package for full GARCH modeling.
        """
        
        p = p or self.config.garch_p
        q = q or self.config.garch_q
        
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Simplified GARCH - use EWM with adaptive decay
        # In production, use proper GARCH fitting
        
        # Start with realized vol as base
        base_vol = self.estimate_realized_volatility(prices)
        
        # Add GARCH-like features (simplified)
        squared_returns = (log_returns ** 2) * 252  # Annualized
        
        # EWM of squared returns (volatility proxy)
        vol_proxy = squared_returns.ewm(halflife=self.config.ewm_halflife).mean()
        vol = np.sqrt(vol_proxy)
        
        # Combine with realized vol
        vol = pd.Series(vol, index=log_returns.index)
        vol = vol.reindex(prices.index, method='ffill')
        
        # Apply bounds
        vol = vol.clip(
            lower=self.config.vol_min_threshold,
            upper=self.config.vol_max_threshold
        )
        
        return vol.fillna(method='bfill').fillna(self.config.vol_min_threshold)
    
    def get_volatility(self, 
                      prices: pd.Series,
                      iv30_data: Optional[pd.Series] = None) -> pd.Series:
        """Get volatility based on configured method."""
        
        if self.config.vol_method == 'realized':
            return self.estimate_realized_volatility(prices)
        
        elif self.config.vol_method == 'ewm':
            return self.estimate_ewm_volatility(prices)
        
        elif self.config.vol_method == 'garch':
            return self.estimate_garch_volatility(prices)
        
        elif self.config.vol_method == 'iv30':
            if iv30_data is not None:
                # Use IV30 if available, fallback to realized
                iv30_aligned = iv30_data.reindex(prices.index, method='ffill')
                iv30_aligned = iv30_aligned.fillna(method='bfill')
                
                # Apply bounds to IV30
                iv30_aligned = iv30_aligned.clip(
                    lower=self.config.vol_min_threshold,
                    upper=self.config.vol_max_threshold
                )
                
                # Fill remaining NaNs with realized vol
                mask = iv30_aligned.isna()
                if mask.any():
                    realized_vol = self.estimate_realized_volatility(prices)
                    iv30_aligned[mask] = realized_vol[mask]
                
                return iv30_aligned
            else:
                logger.warning("IV30 data not available, falling back to realized volatility")
                return self.estimate_realized_volatility(prices)
        
        else:
            raise ValueError(f"Unknown volatility method: {self.config.vol_method}")


class VolScaledLabeler:
    """
    Vol-Scaled Triple Barrier Labeler
    
    Extends triple barrier labeling with volatility-scaled targets
    to stabilize training across different market regimes.
    """
    
    def __init__(self, 
                 tb_config: TripleBarrierConfig,
                 vol_config: VolScaledConfig):
        self.tb_config = tb_config
        self.vol_config = vol_config
        self.base_labeler = TripleBarrierLabeler(tb_config)
        self.vol_estimator = VolatilityEstimator(vol_config)
        
    def create_vol_scaled_labels(self,
                               prices: pd.Series,
                               events: pd.DataFrame,
                               symbol: str,
                               iv30_data: Optional[pd.Series] = None) -> List[VolScaledLabel]:
        """
        Create vol-scaled labels using triple barrier method.
        
        Args:
            prices: Price series
            events: Trading events
            symbol: Symbol identifier
            iv30_data: Optional implied volatility data
            
        Returns:
            List of VolScaledLabel objects
        """
        
        # Get base triple barrier labels
        base_labels = self.base_labeler.create_labels(prices, events, symbol)
        
        if not base_labels:
            return []
        
        # Calculate volatility series
        volatility = self.vol_estimator.get_volatility(prices, iv30_data)
        
        vol_scaled_labels = []
        
        for base_label in base_labels:
            
            # Get volatility at entry
            entry_vol = volatility.loc[base_label.timestamp] if base_label.timestamp in volatility.index else self.vol_config.vol_min_threshold
            
            # Calculate realized volatility during holding period
            holding_period_prices = prices[
                (prices.index >= base_label.timestamp) & 
                (prices.index <= base_label.exit_time)
            ]
            
            if len(holding_period_prices) > 1:
                holding_returns = np.log(holding_period_prices / holding_period_prices.shift(1)).dropna()
                realized_vol = holding_returns.std() * np.sqrt(252) if len(holding_returns) > 0 else entry_vol
            else:
                realized_vol = entry_vol
            
            # Calculate vol-scaled return
            raw_return = base_label.return_pct
            
            if self.vol_config.enable_vol_scaling and entry_vol > 0:
                # Scale by entry volatility
                vol_scaled_return = raw_return / entry_vol * self.vol_config.scaling_factor
                vol_scaling_factor = 1.0 / entry_vol * self.vol_config.scaling_factor
            else:
                vol_scaled_return = raw_return
                vol_scaling_factor = 1.0
            
            # Apply outlier clipping
            if self.vol_config.outlier_clip_std > 0:
                vol_scaled_return = np.clip(
                    vol_scaled_return,
                    -self.vol_config.outlier_clip_std,
                    self.vol_config.outlier_clip_std
                )
            
            # Get IV30 if available
            iv30_value = None
            if iv30_data is not None and base_label.timestamp in iv30_data.index:
                iv30_value = iv30_data.loc[base_label.timestamp]
            
            # Create vol-scaled label
            vol_label = VolScaledLabel(
                timestamp=base_label.timestamp,
                symbol=base_label.symbol,
                entry_price=base_label.entry_price,
                target_price=base_label.target_price,
                stop_loss_price=base_label.stop_loss_price,
                expiry_time=base_label.expiry_time,
                label=base_label.label,
                hit_barrier=base_label.hit_barrier,
                exit_price=base_label.exit_price,
                exit_time=base_label.exit_time,
                holding_time=base_label.holding_time,
                return_pct=raw_return,
                entry_volatility=entry_vol,
                realized_volatility=realized_vol,
                iv30=iv30_value,
                vol_scaled_return=vol_scaled_return,
                vol_scaling_factor=vol_scaling_factor,
                vol_method=self.vol_config.vol_method,
                sample_weight=base_label.sample_weight,
                upper_hit_ts=base_label.upper_hit_ts,
                lower_hit_ts=base_label.lower_hit_ts,
                time_expiry=base_label.time_expiry
            )
            
            vol_scaled_labels.append(vol_label)
        
        logger.info(f"Created {len(vol_scaled_labels)} vol-scaled labels for {symbol}")
        return vol_scaled_labels
    
    def validate_variance_reduction(self,
                                  labels: List[VolScaledLabel],
                                  min_samples: int = 30) -> Dict[str, Any]:
        """
        Validate variance reduction using Levene's test.
        
        Tests whether vol-scaling reduces variance in returns
        without losing predictive power.
        """
        
        if len(labels) < min_samples:
            return {
                'error': f'Insufficient samples for validation (need {min_samples}, got {len(labels)})',
                'n_samples': len(labels)
            }
        
        # Extract returns
        raw_returns = np.array([label.raw_return for label in labels])
        vol_scaled_returns = np.array([label.vol_scaled_return for label in labels])
        
        # Remove any inf or nan values
        valid_mask = np.isfinite(raw_returns) & np.isfinite(vol_scaled_returns)
        raw_returns = raw_returns[valid_mask]
        vol_scaled_returns = vol_scaled_returns[valid_mask]
        
        if len(raw_returns) < min_samples:
            return {
                'error': f'Insufficient valid samples after cleaning (need {min_samples}, got {len(raw_returns)})',
                'n_valid_samples': len(raw_returns)
            }
        
        # Calculate basic statistics
        raw_stats = {
            'mean': np.mean(raw_returns),
            'std': np.std(raw_returns),
            'skew': stats.skew(raw_returns),
            'kurtosis': stats.kurtosis(raw_returns)
        }
        
        vol_scaled_stats = {
            'mean': np.mean(vol_scaled_returns),
            'std': np.std(vol_scaled_returns),
            'skew': stats.skew(vol_scaled_returns),
            'kurtosis': stats.kurtosis(vol_scaled_returns)
        }
        
        # Levene's test for equal variances
        # H0: variances are equal
        # H1: variances are different
        try:
            levene_stat, levene_pvalue = levene(raw_returns, vol_scaled_returns)
            variance_reduced = levene_pvalue < self.vol_config.levene_significance
        except Exception as e:
            logger.error(f"Levene test failed: {e}")
            levene_stat, levene_pvalue = np.nan, np.nan
            variance_reduced = False
        
        # Additional variance analysis
        variance_reduction_ratio = vol_scaled_stats['std'] / raw_stats['std'] if raw_stats['std'] > 0 else 1.0
        
        # Check for preserved signal (correlation between raw and scaled returns)
        signal_correlation = np.corrcoef(raw_returns, vol_scaled_returns)[0, 1]
        
        # F-test for variance ratio
        f_statistic = (raw_stats['std'] ** 2) / (vol_scaled_stats['std'] ** 2) if vol_scaled_stats['std'] > 0 else np.inf
        f_pvalue = 2 * min(stats.f.cdf(f_statistic, len(raw_returns)-1, len(vol_scaled_returns)-1),
                          1 - stats.f.cdf(f_statistic, len(raw_returns)-1, len(vol_scaled_returns)-1))
        
        results = {
            'n_samples': len(raw_returns),
            'raw_stats': raw_stats,
            'vol_scaled_stats': vol_scaled_stats,
            'levene_test': {
                'statistic': levene_stat,
                'pvalue': levene_pvalue,
                'variance_reduced': variance_reduced,
                'significance_threshold': self.vol_config.levene_significance
            },
            'variance_analysis': {
                'reduction_ratio': variance_reduction_ratio,
                'variance_reduced_pct': (1 - variance_reduction_ratio) * 100,
                'f_statistic': f_statistic,
                'f_pvalue': f_pvalue
            },
            'signal_preservation': {
                'correlation': signal_correlation,
                'signal_preserved': signal_correlation > 0.7  # Threshold for signal preservation
            },
            'meets_criteria': (
                variance_reduced and 
                variance_reduction_ratio < 1.0 and
                signal_correlation > 0.7
            )
        }
        
        return results
    
    def compare_training_stability(self,
                                 labels: List[VolScaledLabel],
                                 n_windows: int = 5) -> Dict[str, Any]:
        """
        Compare training stability between raw and vol-scaled returns
        across different time windows.
        """
        
        if len(labels) < n_windows * 20:  # Need minimum samples per window
            return {'error': 'Insufficient data for stability analysis'}
        
        # Sort labels by timestamp
        sorted_labels = sorted(labels, key=lambda x: x.timestamp)
        
        # Split into windows
        window_size = len(sorted_labels) // n_windows
        windows = [
            sorted_labels[i*window_size:(i+1)*window_size] 
            for i in range(n_windows)
        ]
        
        raw_window_stats = []
        vol_scaled_window_stats = []
        
        for window in windows:
            if len(window) < 10:  # Skip windows with too few samples
                continue
                
            raw_rets = [label.raw_return for label in window]
            vol_rets = [label.vol_scaled_return for label in window]
            
            raw_window_stats.append({
                'mean': np.mean(raw_rets),
                'std': np.std(raw_rets),
                'sharpe': np.mean(raw_rets) / np.std(raw_rets) if np.std(raw_rets) > 0 else 0
            })
            
            vol_scaled_window_stats.append({
                'mean': np.mean(vol_rets),
                'std': np.std(vol_rets),
                'sharpe': np.mean(vol_rets) / np.std(vol_rets) if np.std(vol_rets) > 0 else 0
            })
        
        if len(raw_window_stats) < 3:
            return {'error': 'Insufficient windows for stability analysis'}
        
        # Calculate stability metrics
        raw_std_of_means = np.std([w['mean'] for w in raw_window_stats])
        raw_std_of_stds = np.std([w['std'] for w in raw_window_stats])
        raw_std_of_sharpes = np.std([w['sharpe'] for w in raw_window_stats])
        
        vol_std_of_means = np.std([w['mean'] for w in vol_scaled_window_stats])
        vol_std_of_stds = np.std([w['std'] for w in vol_scaled_window_stats])
        vol_std_of_sharpes = np.std([w['sharpe'] for w in vol_scaled_window_stats])
        
        stability_improvement = {
            'mean_stability_ratio': vol_std_of_means / raw_std_of_means if raw_std_of_means > 0 else 1.0,
            'std_stability_ratio': vol_std_of_stds / raw_std_of_stds if raw_std_of_stds > 0 else 1.0,
            'sharpe_stability_ratio': vol_std_of_sharpes / raw_std_of_sharpes if raw_std_of_sharpes > 0 else 1.0
        }
        
        results = {
            'n_windows': len(raw_window_stats),
            'window_size': window_size,
            'raw_stability': {
                'std_of_means': raw_std_of_means,
                'std_of_stds': raw_std_of_stds,
                'std_of_sharpes': raw_std_of_sharpes
            },
            'vol_scaled_stability': {
                'std_of_means': vol_std_of_means,
                'std_of_stds': vol_std_of_stds,
                'std_of_sharpes': vol_std_of_sharpes
            },
            'stability_improvement': stability_improvement,
            'training_stabilized': (
                stability_improvement['std_stability_ratio'] < 0.8 and
                stability_improvement['sharpe_stability_ratio'] < 1.2
            )
        }
        
        return results


# Utility functions
def create_vol_scaled_dataset(prices: pd.Series,
                            events: pd.DataFrame,
                            symbol: str,
                            vol_config: Optional[VolScaledConfig] = None,
                            tb_config: Optional[TripleBarrierConfig] = None,
                            iv30_data: Optional[pd.Series] = None) -> Tuple[List[VolScaledLabel], Dict[str, Any]]:
    """
    Convenience function to create vol-scaled dataset with validation.
    """
    
    if vol_config is None:
        vol_config = VolScaledConfig()
    
    if tb_config is None:
        tb_config = TripleBarrierConfig()
    
    labeler = VolScaledLabeler(tb_config, vol_config)
    
    # Create labels
    labels = labeler.create_vol_scaled_labels(prices, events, symbol, iv30_data)
    
    # Validate variance reduction
    validation_results = labeler.validate_variance_reduction(labels)
    
    # Test training stability
    stability_results = labeler.compare_training_stability(labels)
    
    results = {
        'labels': labels,
        'validation': validation_results,
        'stability': stability_results,
        'config': vol_config
    }
    
    return labels, results


# Example usage
def example_vol_scaled_labeling():
    """Example of vol-scaled labeling."""
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    
    # Create price series with time-varying volatility
    vol_regime = np.sin(np.arange(500) * 2 * np.pi / 100) * 0.5 + 1  # Volatility cycle
    base_vol = 0.02
    daily_vol = base_vol * vol_regime
    
    returns = np.random.normal(0.001, daily_vol)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
    
    # Create events
    events = pd.DataFrame({
        'timestamp': dates[::20],  # Every 20th day
        'side': np.random.choice([1, -1], size=len(dates[::20]))
    })
    
    # Create vol-scaled labels
    vol_config = VolScaledConfig(
        vol_method='realized',
        vol_lookback=20,
        enable_vol_scaling=True
    )
    
    labels, results = create_vol_scaled_dataset(
        prices, events, 'TEST', vol_config
    )
    
    print(f"Created {len(labels)} vol-scaled labels")
    print(f"Validation results: {results['validation']}")
    print(f"Stability results: {results['stability']}")
    
    return labels, results


if __name__ == "__main__":
    example_vol_scaled_labeling()