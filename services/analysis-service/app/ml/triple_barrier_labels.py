"""
Triple-Barrier Labels and Meta-Labeling System

This module implements advanced labeling methods for machine learning in trading,
following methodologies from López de Prado's "Advances in Financial Machine Learning".

Key Features:
1. Triple-Barrier Method: Labels based on profit-taking, stop-loss, and time barriers
2. Meta-Labeling: Secondary model to predict probability of correct primary model predictions
3. Dynamic Barriers: Adaptive barriers based on volatility and market conditions
4. Sample Weights: Proper weighting for overlapping samples and label importance
5. Fractional Differentiation: Stationary features while preserving memory

Applications:
- Event-driven trading strategies
- Position sizing and risk management
- Strategy combination and ensemble methods
- Return prediction with proper sample weights

References:
- López de Prado, M. (2018). Advances in Financial Machine Learning.
- López de Prado, M. (2020). Machine Learning for Asset Managers.
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample
import numba

logger = logging.getLogger(__name__)


class BarrierType(Enum):
    """Types of barriers in triple-barrier method."""
    PROFIT_TAKING = "profit_taking"  # Upper barrier
    STOP_LOSS = "stop_loss"         # Lower barrier  
    TIME_LIMIT = "time_limit"       # Vertical barrier


class LabelClass(Enum):
    """Label classes for triple-barrier method."""
    LONG = 1      # Profit-taking barrier hit first
    NEUTRAL = 0   # Time barrier hit first
    SHORT = -1    # Stop-loss barrier hit first


class SamplingMethod(Enum):
    """Methods for handling overlapping samples."""
    CUSUM = "cusum"                    # CUSUM filter for structural breaks
    TICK_IMBALANCE = "tick_imbalance"  # Tick imbalance bars
    VOLUME_IMBALANCE = "volume_imbalance"  # Volume imbalance bars
    TIME_BASED = "time_based"          # Regular time intervals


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""
    profit_taking_multiplier: float = 1.0    # Multiplier for upper barrier
    stop_loss_multiplier: float = 1.0        # Multiplier for lower barrier
    max_holding_period: int = 5               # Maximum holding period (days)
    min_holding_period: int = 1               # Minimum holding period (days)
    volatility_window: int = 20               # Window for volatility estimation
    dynamic_barriers: bool = True             # Use dynamic barriers based on volatility
    barrier_width_factor: float = 1.0        # Factor for barrier width adjustment
    min_sample_weight: float = 0.1            # Minimum sample weight
    
    def __post_init__(self):
        if self.profit_taking_multiplier <= 0:
            raise ValueError("profit_taking_multiplier must be positive")
        if self.stop_loss_multiplier <= 0:
            raise ValueError("stop_loss_multiplier must be positive")
        if self.max_holding_period <= 0:
            raise ValueError("max_holding_period must be positive")


@dataclass
class TripleBarrierLabel:
    """Result from triple-barrier labeling."""
    timestamp: pd.Timestamp
    side: int                    # 1 for long, -1 for short prediction
    barrier_hit: BarrierType     # Which barrier was hit first
    exit_timestamp: pd.Timestamp # When position was closed
    return_realized: float       # Actual return achieved
    holding_period: int          # Days held
    sample_weight: float         # Weight for ML training
    label: LabelClass           # Final label class
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'side': self.side,
            'barrier_hit': self.barrier_hit.value,
            'exit_timestamp': self.exit_timestamp,
            'return_realized': self.return_realized,
            'holding_period': self.holding_period,
            'sample_weight': self.sample_weight,
            'label': self.label.value
        }


@dataclass
class MetaLabelResult:
    """Result from meta-labeling."""
    timestamp: pd.Timestamp
    primary_prediction: float    # Primary model prediction (return forecast)
    primary_confidence: float    # Primary model confidence
    meta_label: int             # 0 = don't trade, 1 = trade
    meta_probability: float     # Probability from meta-model
    actual_outcome: Optional[int] = None  # Actual result (for training)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'primary_prediction': self.primary_prediction,
            'primary_confidence': self.primary_confidence,
            'meta_label': self.meta_label,
            'meta_probability': self.meta_probability,
            'actual_outcome': self.actual_outcome
        }


class TripleBarrierLabeler:
    """
    Triple-Barrier Labeling System.
    
    Implements the triple-barrier method for creating ML training labels
    with proper sample weights and meta-labeling capabilities.
    """
    
    def __init__(self, config: TripleBarrierConfig):
        """Initialize triple-barrier labeler."""
        self.config = config
        self._price_data = None
        self._volatility_estimates = None
        
    def create_labels(self,
                     price_data: pd.Series,
                     events: pd.DataFrame,
                     side_prediction: Optional[pd.Series] = None,
                     barriers: Optional[Tuple[pd.Series, pd.Series]] = None) -> pd.DataFrame:
        """
        Create triple-barrier labels for given events.
        
        Parameters:
        - price_data: Price series (typically close prices)
        - events: DataFrame with columns ['timestamp', 'side'] for each event
        - side_prediction: Optional series with directional predictions (-1, 0, 1)
        - barriers: Optional tuple of (upper_barrier, lower_barrier) series
        
        Returns:
        - DataFrame with triple-barrier labels and sample weights
        """
        logger.info(f"Creating triple-barrier labels for {len(events)} events")
        
        self._price_data = price_data
        
        # Estimate volatility if dynamic barriers are used
        if self.config.dynamic_barriers:
            self._volatility_estimates = self._estimate_volatility(price_data)
        
        # Create barriers if not provided
        if barriers is None:
            upper_barriers, lower_barriers = self._create_dynamic_barriers(
                price_data, events
            )
        else:
            upper_barriers, lower_barriers = barriers
        
        # Process each event
        labels = []
        for idx, event in events.iterrows():
            timestamp = pd.to_datetime(event['timestamp'])
            side = event.get('side', 1)
            
            # Override with side prediction if provided
            if side_prediction is not None and timestamp in side_prediction.index:
                side = side_prediction.loc[timestamp]
            
            # Skip if no clear directional signal
            if side == 0:
                continue
            
            try:
                label = self._apply_triple_barrier(
                    timestamp, side, price_data, 
                    upper_barriers, lower_barriers
                )
                
                if label is not None:
                    labels.append(label)
                    
            except Exception as e:
                logger.warning(f"Failed to create label for {timestamp}: {e}")
                continue
        
        if not labels:
            logger.warning("No labels created")
            return pd.DataFrame()
        
        # Convert to DataFrame
        labels_df = pd.DataFrame([label.to_dict() for label in labels])
        
        # Calculate sample weights
        labels_df = self._calculate_sample_weights(labels_df)
        
        logger.info(f"Created {len(labels_df)} triple-barrier labels")
        return labels_df
    
    def create_meta_labels(self,
                          primary_predictions: pd.Series,
                          actual_outcomes: pd.Series,
                          confidence_threshold: float = 0.5) -> pd.DataFrame:
        """
        Create meta-labels for ensemble learning.
        
        Meta-labeling predicts when the primary model's predictions are reliable.
        
        Parameters:
        - primary_predictions: Primary model predictions
        - actual_outcomes: Actual outcomes (for training)
        - confidence_threshold: Threshold for binary classification
        
        Returns:
        - DataFrame with meta-labels
        """
        logger.info(f"Creating meta-labels for {len(primary_predictions)} predictions")
        
        # Align data
        common_index = primary_predictions.index.intersection(actual_outcomes.index)
        primary_aligned = primary_predictions.loc[common_index]
        outcomes_aligned = actual_outcomes.loc[common_index]
        
        meta_labels = []
        
        for timestamp in common_index:
            prediction = primary_aligned.loc[timestamp]
            outcome = outcomes_aligned.loc[timestamp]
            
            # Primary model confidence (absolute value of prediction)
            confidence = abs(prediction)
            
            # Meta-label: 1 if primary prediction is correct, 0 otherwise
            # For regression predictions, consider sign agreement
            if prediction * outcome > 0:
                meta_label = 1  # Correct direction
            else:
                meta_label = 0  # Incorrect direction
            
            # Meta-probability based on confidence
            meta_prob = min(confidence, 1.0)
            
            result = MetaLabelResult(
                timestamp=timestamp,
                primary_prediction=prediction,
                primary_confidence=confidence,
                meta_label=meta_label,
                meta_probability=meta_prob,
                actual_outcome=outcome
            )
            
            meta_labels.append(result.to_dict())
        
        meta_labels_df = pd.DataFrame(meta_labels)
        logger.info(f"Created {len(meta_labels_df)} meta-labels")
        
        return meta_labels_df
    
    def _estimate_volatility(self, price_data: pd.Series) -> pd.Series:
        """Estimate volatility using various methods."""
        # Simple daily returns volatility
        returns = price_data.pct_change().dropna()
        
        # Rolling standard deviation
        volatility = returns.rolling(
            window=self.config.volatility_window,
            min_periods=max(1, self.config.volatility_window // 2)
        ).std()
        
        # Annualize (assuming daily data)
        volatility = volatility * np.sqrt(252)
        
        # Forward fill missing values
        volatility = volatility.fillna(method='ffill').fillna(volatility.median())
        
        return volatility
    
    def _create_dynamic_barriers(self,
                               price_data: pd.Series,
                               events: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Create dynamic barriers based on volatility."""
        upper_barriers = pd.Series(index=events['timestamp'], dtype=float)
        lower_barriers = pd.Series(index=events['timestamp'], dtype=float)
        
        for timestamp in events['timestamp']:
            timestamp = pd.to_datetime(timestamp)
            
            if timestamp not in price_data.index:
                # Find nearest price
                closest_idx = price_data.index.searchsorted(timestamp)
                if closest_idx >= len(price_data):
                    closest_idx = len(price_data) - 1
                timestamp = price_data.index[closest_idx]
            
            current_price = price_data.loc[timestamp]
            
            # Get current volatility estimate
            if self._volatility_estimates is not None:
                vol = self._volatility_estimates.loc[timestamp]
            else:
                vol = 0.02  # Default 2% daily volatility
            
            # Calculate barrier levels
            barrier_width = vol * self.config.barrier_width_factor
            
            upper_barrier = current_price * (1 + barrier_width * self.config.profit_taking_multiplier)
            lower_barrier = current_price * (1 - barrier_width * self.config.stop_loss_multiplier)
            
            upper_barriers.loc[timestamp] = upper_barrier
            lower_barriers.loc[timestamp] = lower_barrier
        
        return upper_barriers, lower_barriers
    
    def _apply_triple_barrier(self,
                            timestamp: pd.Timestamp,
                            side: int,
                            price_data: pd.Series,
                            upper_barriers: pd.Series,
                            lower_barriers: pd.Series) -> Optional[TripleBarrierLabel]:
        """Apply triple-barrier method to a single event."""
        if timestamp not in price_data.index:
            return None
        
        entry_price = price_data.loc[timestamp]
        upper_barrier = upper_barriers.loc[timestamp]
        lower_barrier = lower_barriers.loc[timestamp]
        
        # Define exit conditions based on side
        if side == 1:  # Long position
            profit_barrier = upper_barrier
            stop_barrier = lower_barrier
        else:  # Short position
            profit_barrier = lower_barrier
            stop_barrier = upper_barrier
        
        # Find exit point
        start_idx = price_data.index.get_loc(timestamp)
        max_idx = min(
            start_idx + self.config.max_holding_period,
            len(price_data) - 1
        )
        min_idx = start_idx + self.config.min_holding_period
        
        barrier_hit = None
        exit_idx = max_idx  # Default to time barrier
        
        # Check price path for barrier hits
        for i in range(min_idx, max_idx + 1):
            if i >= len(price_data):
                break
                
            current_price = price_data.iloc[i]
            
            # Check for barrier hits based on position side
            if side == 1:  # Long position
                if current_price >= profit_barrier:
                    barrier_hit = BarrierType.PROFIT_TAKING
                    exit_idx = i
                    break
                elif current_price <= stop_barrier:
                    barrier_hit = BarrierType.STOP_LOSS
                    exit_idx = i
                    break
            else:  # Short position
                if current_price <= profit_barrier:
                    barrier_hit = BarrierType.PROFIT_TAKING
                    exit_idx = i
                    break
                elif current_price >= stop_barrier:
                    barrier_hit = BarrierType.STOP_LOSS
                    exit_idx = i
                    break
        
        # Default to time barrier if no price barrier hit
        if barrier_hit is None:
            barrier_hit = BarrierType.TIME_LIMIT
        
        # Calculate return
        exit_price = price_data.iloc[exit_idx]
        exit_timestamp = price_data.index[exit_idx]
        
        if side == 1:  # Long position
            return_realized = (exit_price - entry_price) / entry_price
        else:  # Short position
            return_realized = (entry_price - exit_price) / entry_price
        
        holding_period = exit_idx - start_idx
        
        # Determine label class
        if barrier_hit == BarrierType.PROFIT_TAKING:
            label_class = LabelClass.LONG if side == 1 else LabelClass.SHORT
        elif barrier_hit == BarrierType.STOP_LOSS:
            label_class = LabelClass.SHORT if side == 1 else LabelClass.LONG
        else:  # Time barrier
            # Classify based on realized return
            if abs(return_realized) < 0.001:  # Threshold for neutral
                label_class = LabelClass.NEUTRAL
            elif return_realized > 0:
                label_class = LabelClass.LONG
            else:
                label_class = LabelClass.SHORT
        
        return TripleBarrierLabel(
            timestamp=timestamp,
            side=side,
            barrier_hit=barrier_hit,
            exit_timestamp=exit_timestamp,
            return_realized=return_realized,
            holding_period=holding_period,
            sample_weight=1.0,  # Will be calculated later
            label=label_class
        )
    
    def _calculate_sample_weights(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sample weights accounting for overlapping samples."""
        labels_df = labels_df.copy()
        
        # Initialize weights
        labels_df['sample_weight'] = 1.0
        
        # Account for overlapping samples
        labels_df = labels_df.sort_values('timestamp')
        
        for i, row in labels_df.iterrows():
            entry_time = row['timestamp']
            exit_time = row['exit_timestamp']
            
            # Count overlapping samples
            overlap_count = 0
            for j, other_row in labels_df.iterrows():
                if i == j:
                    continue
                
                other_entry = other_row['timestamp']
                other_exit = other_row['exit_timestamp']
                
                # Check for overlap
                if (entry_time <= other_exit and exit_time >= other_entry):
                    overlap_count += 1
            
            # Adjust weight based on overlaps
            if overlap_count > 0:
                weight = 1.0 / (1 + overlap_count)
            else:
                weight = 1.0
            
            # Apply minimum weight threshold
            weight = max(weight, self.config.min_sample_weight)
            
            labels_df.loc[i, 'sample_weight'] = weight
        
        return labels_df


class EventSampler:
    """
    Event sampling methods for machine learning.
    
    Implements various sampling strategies to create training events
    that are more informative and less overlapping.
    """
    
    def __init__(self, method: SamplingMethod = SamplingMethod.CUSUM):
        """Initialize event sampler."""
        self.method = method
    
    def sample_events(self,
                     price_data: pd.Series,
                     threshold: float = 0.02,
                     min_interval: int = 1) -> pd.DataFrame:
        """
        Sample events based on specified method.
        
        Parameters:
        - price_data: Price series
        - threshold: Threshold for event detection
        - min_interval: Minimum interval between events
        
        Returns:
        - DataFrame with event timestamps and metadata
        """
        if self.method == SamplingMethod.CUSUM:
            return self._cusum_filter(price_data, threshold, min_interval)
        elif self.method == SamplingMethod.TIME_BASED:
            return self._time_based_sampling(price_data, min_interval)
        else:
            raise NotImplementedError(f"Sampling method {self.method} not implemented")
    
    def _cusum_filter(self,
                     price_data: pd.Series,
                     threshold: float,
                     min_interval: int) -> pd.DataFrame:
        """CUSUM filter for structural break detection."""
        logger.info(f"Applying CUSUM filter with threshold {threshold}")
        
        returns = price_data.pct_change().dropna()
        
        events = []
        s_pos = 0  # Positive CUSUM
        s_neg = 0  # Negative CUSUM
        last_event_idx = 0
        
        for i, (timestamp, ret) in enumerate(returns.items()):
            # Update CUSUM values
            s_pos = max(0, s_pos + ret)
            s_neg = min(0, s_neg + ret)
            
            # Check for events
            if (s_pos > threshold or s_neg < -threshold) and (i - last_event_idx) >= min_interval:
                # Determine side based on which CUSUM triggered
                if s_pos > threshold:
                    side = 1  # Upward structural break
                else:
                    side = -1  # Downward structural break
                
                events.append({
                    'timestamp': timestamp,
                    'side': side,
                    'cusum_pos': s_pos,
                    'cusum_neg': s_neg
                })
                
                # Reset CUSUM values
                s_pos = 0
                s_neg = 0
                last_event_idx = i
        
        events_df = pd.DataFrame(events)
        logger.info(f"CUSUM filter detected {len(events_df)} events")
        
        return events_df
    
    def _time_based_sampling(self,
                           price_data: pd.Series,
                           interval: int) -> pd.DataFrame:
        """Simple time-based sampling."""
        logger.info(f"Time-based sampling with interval {interval}")
        
        timestamps = price_data.index[::interval]
        
        events = [{
            'timestamp': ts,
            'side': 1  # Default to long bias
        } for ts in timestamps]
        
        events_df = pd.DataFrame(events)
        logger.info(f"Time-based sampling created {len(events_df)} events")
        
        return events_df


@numba.jit(nopython=True)
def _fast_barrier_search(prices: np.ndarray,
                        upper_barrier: float,
                        lower_barrier: float,
                        side: int,
                        min_periods: int) -> Tuple[int, int]:
    """Fast search for barrier hits using Numba."""
    n = len(prices)
    
    for i in range(min_periods, n):
        price = prices[i]
        
        if side == 1:  # Long position
            if price >= upper_barrier:
                return i, 1  # Profit barrier hit
            elif price <= lower_barrier:
                return i, -1  # Stop barrier hit
        else:  # Short position
            if price <= upper_barrier:
                return i, 1  # Profit barrier hit
            elif price >= lower_barrier:
                return i, -1  # Stop barrier hit
    
    return n - 1, 0  # Time barrier hit


def fractional_differentiation(series: pd.Series, 
                              d: float = 0.4,
                              threshold: float = 1e-5) -> pd.Series:
    """
    Apply fractional differentiation to achieve stationarity while preserving memory.
    
    Parameters:
    - series: Time series to differentiate
    - d: Fractional differentiation parameter (0 < d < 1)
    - threshold: Threshold for coefficient truncation
    
    Returns:
    - Fractionally differentiated series
    """
    # Calculate binomial coefficients
    weights = np.array([1.0])
    k = 1
    
    while abs(weights[-1]) > threshold:
        weight_k = -weights[-1] * (d - k + 1) / k
        weights = np.append(weights, weight_k)
        k += 1
    
    # Apply fractional differentiation
    ffd_series = pd.Series(index=series.index, dtype=float)
    
    for i in range(len(weights), len(series)):
        window = series.iloc[i-len(weights)+1:i+1]
        ffd_value = np.sum(weights * window.values[::-1])
        ffd_series.iloc[i] = ffd_value
    
    return ffd_series.dropna()


def calculate_uniqueness_weights(events_df: pd.DataFrame) -> pd.Series:
    """
    Calculate sample uniqueness weights based on label overlap.
    
    Parameters:
    - events_df: DataFrame with event timestamps and exit timestamps
    
    Returns:
    - Series with uniqueness weights
    """
    weights = pd.Series(index=events_df.index, dtype=float)
    
    for i, row in events_df.iterrows():
        # Count concurrent samples
        concurrent = 0
        entry_time = row['timestamp']
        exit_time = row['exit_timestamp']
        
        for j, other_row in events_df.iterrows():
            if i == j:
                continue
            
            other_entry = other_row['timestamp']
            other_exit = other_row['exit_timestamp']
            
            # Check for temporal overlap
            if entry_time <= other_exit and exit_time >= other_entry:
                concurrent += 1
        
        # Weight inversely proportional to concurrent samples
        weights.loc[i] = 1.0 / (1.0 + concurrent)
    
    return weights