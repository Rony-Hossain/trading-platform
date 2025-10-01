"""
Triple-Barrier Labeling and Meta-Labeling for Trading Strategies

Implementation of triple-barrier method from López de Prado's "Advances in Financial Machine Learning"
for creating balanced labeled datasets for machine learning.

Key concepts:
- Triple barriers: upper profit-taking, lower stop-loss, and time-based expiry
- Meta-labeling: secondary model to predict probability of correct directional prediction
- Sample weights: reduce impact of overlapping labels and improve model training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
import warnings

logger = logging.getLogger(__name__)


@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""
    horizon_days: int = 5  # TB_HORIZON_DAYS
    upper_sigma: float = 2.0  # TB_UPPER_SIGMA - profit taking threshold
    lower_sigma: float = 1.5  # TB_LOWER_SIGMA - stop loss threshold
    min_return_threshold: float = 0.005  # Minimum return to consider significant
    volatility_lookback: int = 20  # Days for volatility calculation
    max_holding_time: int = 10  # Maximum holding period in days
    

@dataclass
class TripleBarrierLabel:
    """Result of triple barrier labeling."""
    timestamp: datetime
    symbol: str
    entry_price: float
    target_price: float
    stop_loss_price: float
    expiry_time: datetime
    label: int  # -1, 0, 1 for sell/neutral/buy
    hit_barrier: str  # 'upper', 'lower', 'time'
    exit_price: float
    exit_time: datetime
    holding_time: int  # in trading days
    return_pct: float
    upper_hit_ts: Optional[datetime] = None
    lower_hit_ts: Optional[datetime] = None
    time_expiry: Optional[datetime] = None
    sample_weight: float = 1.0


@dataclass
class MetaLabelResult:
    """Result of meta-labeling process."""
    timestamp: datetime
    symbol: str
    base_signal: int  # Original strategy signal
    meta_probability: float  # Probability from meta-classifier
    meta_prediction: int  # Binary prediction (trade/no-trade)
    features: Dict[str, float]  # Features used for meta-labeling
    confidence: float  # Prediction confidence


class TripleBarrierLabeler:
    """
    Triple Barrier Labeling System
    
    Creates labeled datasets using the triple-barrier method:
    1. Define profit-taking (upper) and stop-loss (lower) barriers based on volatility
    2. Set maximum holding period (time barrier)
    3. Label each observation based on which barrier is hit first
    4. Apply sample weights to reduce overlap impact
    """
    
    def __init__(self, config: TripleBarrierConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_volatility(self, 
                           prices: pd.Series, 
                           lookback: Optional[int] = None) -> pd.Series:
        """Calculate rolling volatility using log returns."""
        lookback = lookback or self.config.volatility_lookback
        
        log_returns = np.log(prices / prices.shift(1))
        volatility = log_returns.rolling(window=lookback, min_periods=lookback//2).std()
        
        # Annualize assuming 252 trading days
        volatility = volatility * np.sqrt(252)
        
        return volatility.fillna(volatility.mean())
    
    def get_barriers(self, 
                     prices: pd.Series,
                     events: pd.DataFrame,
                     volatility: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate triple barriers for each event.
        
        Args:
            prices: Price series indexed by datetime
            events: DataFrame with columns ['timestamp', 'side'] where side is 1 for long, -1 for short
            volatility: Optional pre-calculated volatility series
            
        Returns:
            DataFrame with barriers for each event
        """
        if volatility is None:
            volatility = self.calculate_volatility(prices)
            
        barriers = []
        
        for _, event in events.iterrows():
            timestamp = event['timestamp']
            side = event.get('side', 1)  # Default to long position
            
            if timestamp not in prices.index:
                self.logger.warning(f"Timestamp {timestamp} not found in price series")
                continue
                
            entry_price = prices.loc[timestamp]
            vol = volatility.loc[timestamp]
            
            # Calculate barrier levels
            if side == 1:  # Long position
                upper_barrier = entry_price * (1 + self.config.upper_sigma * vol)
                lower_barrier = entry_price * (1 - self.config.lower_sigma * vol)
            else:  # Short position
                upper_barrier = entry_price * (1 - self.config.upper_sigma * vol)
                lower_barrier = entry_price * (1 + self.config.lower_sigma * vol)
            
            # Time barrier
            time_barrier = timestamp + pd.Timedelta(days=self.config.horizon_days)
            
            barriers.append({
                'timestamp': timestamp,
                'side': side,
                'entry_price': entry_price,
                'upper_barrier': upper_barrier,
                'lower_barrier': lower_barrier,
                'time_barrier': time_barrier,
                'volatility': vol
            })
            
        return pd.DataFrame(barriers)
    
    def apply_barriers(self, 
                       prices: pd.Series,
                       barriers: pd.DataFrame) -> List[TripleBarrierLabel]:
        """Apply triple barriers and generate labels."""
        labels = []
        
        for _, barrier in barriers.iterrows():
            timestamp = barrier['timestamp']
            side = barrier['side']
            entry_price = barrier['entry_price']
            upper_barrier = barrier['upper_barrier']
            lower_barrier = barrier['lower_barrier']
            time_barrier = barrier['time_barrier']
            
            # Get price series from entry to potential exit
            future_prices = prices[prices.index >= timestamp]
            future_prices = future_prices[future_prices.index <= time_barrier]
            
            if len(future_prices) < 2:
                continue
                
            # Find barrier hits
            upper_hits = future_prices[future_prices >= upper_barrier]
            lower_hits = future_prices[future_prices <= lower_barrier]
            
            upper_hit_ts = upper_hits.index[0] if len(upper_hits) > 0 else None
            lower_hit_ts = lower_hits.index[0] if len(lower_hits) > 0 else None
            
            # Determine which barrier was hit first
            hit_times = []
            if upper_hit_ts is not None:
                hit_times.append((upper_hit_ts, 'upper'))
            if lower_hit_ts is not None:
                hit_times.append((lower_hit_ts, 'lower'))
            
            if hit_times:
                # Sort by time and get first hit
                hit_times.sort(key=lambda x: x[0])
                exit_time, hit_barrier = hit_times[0]
                exit_price = prices.loc[exit_time]
            else:
                # Time barrier hit
                exit_time = time_barrier
                if exit_time in prices.index:
                    exit_price = prices.loc[exit_time]
                else:
                    # Get closest available price
                    available_times = prices.index[prices.index <= exit_time]
                    if len(available_times) > 0:
                        exit_time = available_times[-1]
                        exit_price = prices.loc[exit_time]
                    else:
                        continue
                hit_barrier = 'time'
            
            # Calculate return and label
            if side == 1:  # Long position
                return_pct = (exit_price - entry_price) / entry_price
            else:  # Short position
                return_pct = (entry_price - exit_price) / entry_price
            
            # Generate label based on return and barrier hit
            if hit_barrier == 'upper':
                label = side  # Positive return in direction of position
            elif hit_barrier == 'lower':
                label = -side  # Negative return in direction of position
            else:  # Time expiry
                if abs(return_pct) < self.config.min_return_threshold:
                    label = 0  # Neutral
                else:
                    label = 1 if return_pct > 0 else -1
            
            # Calculate holding time
            holding_time = (exit_time - timestamp).days
            
            labels.append(TripleBarrierLabel(
                timestamp=timestamp,
                symbol=getattr(barrier, 'symbol', 'UNKNOWN'),
                entry_price=entry_price,
                target_price=upper_barrier if side == 1 else lower_barrier,
                stop_loss_price=lower_barrier if side == 1 else upper_barrier,
                expiry_time=time_barrier,
                label=label,
                hit_barrier=hit_barrier,
                exit_price=exit_price,
                exit_time=exit_time,
                holding_time=holding_time,
                return_pct=return_pct,
                upper_hit_ts=upper_hit_ts,
                lower_hit_ts=lower_hit_ts,
                time_expiry=time_barrier if hit_barrier == 'time' else None
            ))
            
        return labels
    
    def calculate_sample_weights(self, labels: List[TripleBarrierLabel]) -> List[TripleBarrierLabel]:
        """
        Calculate sample weights to reduce impact of overlapping labels.
        Uses the average uniqueness method from López de Prado.
        """
        if not labels:
            return labels
            
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([{
            'timestamp': label.timestamp,
            'exit_time': label.exit_time,
            'symbol': label.symbol
        } for label in labels])
        
        # Calculate overlaps for each label
        weights = []
        
        for i, label in enumerate(labels):
            start_time = label.timestamp
            end_time = label.exit_time
            
            # Count overlapping labels
            overlaps = 0
            for j, other_label in enumerate(labels):
                if i == j:
                    continue
                    
                other_start = other_label.timestamp
                other_end = other_label.exit_time
                
                # Check for overlap
                if (start_time <= other_end and end_time >= other_start and 
                    label.symbol == other_label.symbol):
                    overlaps += 1
            
            # Weight is inverse of number of overlaps + 1
            weight = 1.0 / (overlaps + 1.0)
            weights.append(weight)
            
        # Update labels with weights
        for i, label in enumerate(labels):
            label.sample_weight = weights[i]
            
        return labels
    
    def create_labels(self, 
                      prices: pd.Series,
                      events: pd.DataFrame,
                      symbol: str = 'UNKNOWN') -> List[TripleBarrierLabel]:
        """
        Main method to create triple barrier labels.
        
        Args:
            prices: Price series indexed by datetime
            events: DataFrame with events to label
            symbol: Symbol identifier
            
        Returns:
            List of TripleBarrierLabel objects
        """
        try:
            # Add symbol to events if not present
            if 'symbol' not in events.columns:
                events = events.copy()
                events['symbol'] = symbol
                
            # Calculate volatility
            volatility = self.calculate_volatility(prices)
            
            # Get barriers
            barriers = self.get_barriers(prices, events, volatility)
            
            if barriers.empty:
                self.logger.warning("No barriers created")
                return []
            
            # Apply barriers
            labels = self.apply_barriers(prices, barriers)
            
            # Calculate sample weights
            labels = self.calculate_sample_weights(labels)
            
            self.logger.info(f"Created {len(labels)} triple barrier labels for {symbol}")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error creating triple barrier labels: {e}")
            return []


class MetaLabeler:
    """
    Meta-Labeling Classifier
    
    Secondary model that predicts the probability that a primary model's 
    directional prediction is correct. Used to filter trading signals and
    improve precision.
    """
    
    def __init__(self, 
                 base_estimator=None,
                 calibration_method: str = 'isotonic',
                 cv_folds: int = 3):
        self.base_estimator = base_estimator or RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.calibration_method = calibration_method
        self.cv_folds = cv_folds
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None
        
    def extract_features(self, 
                        prices: pd.Series,
                        labels: List[TripleBarrierLabel],
                        additional_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Extract features for meta-labeling."""
        
        features_list = []
        
        for label in labels:
            timestamp = label.timestamp
            
            # Get historical prices for feature calculation
            hist_prices = prices[prices.index <= timestamp].tail(20)
            
            if len(hist_prices) < 5:
                continue
                
            # Price-based features
            returns = hist_prices.pct_change().dropna()
            
            features = {
                'timestamp': timestamp,
                'symbol': label.symbol,
                'volatility_5d': returns.tail(5).std() * np.sqrt(252),
                'volatility_20d': returns.std() * np.sqrt(252),
                'momentum_5d': (hist_prices.iloc[-1] / hist_prices.iloc[-5] - 1) if len(hist_prices) >= 5 else 0,
                'momentum_10d': (hist_prices.iloc[-1] / hist_prices.iloc[-10] - 1) if len(hist_prices) >= 10 else 0,
                'rsi_5d': self._calculate_rsi(hist_prices, 5),
                'rsi_14d': self._calculate_rsi(hist_prices, 14),
                'price_distance_ma5': (hist_prices.iloc[-1] / hist_prices.tail(5).mean() - 1),
                'price_distance_ma20': (hist_prices.iloc[-1] / hist_prices.mean() - 1),
                'volume_ratio': 1.0,  # Placeholder - would need volume data
                'true_return': label.return_pct,
                'label': 1 if label.label != 0 else 0,  # Binary: trade vs no-trade
                'sample_weight': label.sample_weight
            }
            
            # Add additional features if provided
            if additional_features is not None and timestamp in additional_features.index:
                additional = additional_features.loc[timestamp].to_dict()
                features.update(additional)
                
            features_list.append(features)
            
        return pd.DataFrame(features_list)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def fit(self, 
            features_df: pd.DataFrame,
            target_col: str = 'label',
            weight_col: str = 'sample_weight') -> 'MetaLabeler':
        """Fit the meta-labeling model."""
        
        # Prepare features and target
        feature_cols = [col for col in features_df.columns 
                       if col not in ['timestamp', 'symbol', target_col, weight_col, 'true_return']]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df[target_col]
        sample_weights = features_df[weight_col] if weight_col in features_df.columns else None
        
        # Fit model with sample weights
        if sample_weights is not None:
            self.model = self.base_estimator.fit(X, y, sample_weight=sample_weights)
        else:
            self.model = self.base_estimator.fit(X, y)
            
        self.is_fitted = True
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False)
            
        return self
    
    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using the meta-labeling model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        feature_cols = [col for col in features_df.columns 
                       if col not in ['timestamp', 'symbol', 'label', 'sample_weight', 'true_return']]
        
        X = features_df[feature_cols].fillna(0)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
            # Return probability of positive class (trade)
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            # For models without predict_proba, use decision function
            scores = self.model.decision_function(X)
            return 1 / (1 + np.exp(-scores))  # Sigmoid transformation
    
    def predict(self, features_df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions using the meta-labeling model."""
        probabilities = self.predict_proba(features_df)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, 
                 features_df: pd.DataFrame,
                 target_col: str = 'label') -> Dict[str, float]:
        """Evaluate the meta-labeling model."""
        
        y_true = features_df[target_col]
        y_proba = self.predict_proba(features_df)
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Calculate metrics
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        
        # Calibration metrics
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        # Calibration slope (should be close to 1 for well-calibrated model)
        calibration_slope = np.polyfit(mean_predicted_value, fraction_of_positives, 1)[0]
        
        return {
            'f1_score': f1,
            'auc_score': auc,
            'calibration_slope': calibration_slope,
            'mean_predicted_prob': y_proba.mean(),
            'positive_rate': y_true.mean()
        }


def create_meta_labels(prices: pd.Series,
                      base_signals: pd.DataFrame,
                      config: TripleBarrierConfig,
                      additional_features: Optional[pd.DataFrame] = None) -> Tuple[List[MetaLabelResult], Dict[str, float]]:
    """
    Complete pipeline for creating meta-labels.
    
    Args:
        prices: Price series
        base_signals: DataFrame with base strategy signals
        config: Triple barrier configuration
        additional_features: Optional additional features for meta-labeling
        
    Returns:
        Tuple of (meta_label_results, performance_metrics)
    """
    
    # Create triple barrier labels
    labeler = TripleBarrierLabeler(config)
    tb_labels = labeler.create_labels(prices, base_signals)
    
    if not tb_labels:
        return [], {}
    
    # Create meta-labeling features
    meta_labeler = MetaLabeler()
    features_df = meta_labeler.extract_features(prices, tb_labels, additional_features)
    
    if features_df.empty:
        return [], {}
    
    # Split into train/test (use first 70% for training)
    split_idx = int(len(features_df) * 0.7)
    train_df = features_df.iloc[:split_idx]
    test_df = features_df.iloc[split_idx:]
    
    # Fit meta-labeling model
    meta_labeler.fit(train_df)
    
    # Generate predictions on test set
    test_probas = meta_labeler.predict_proba(test_df)
    test_predictions = meta_labeler.predict(test_df)
    
    # Create results
    results = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        results.append(MetaLabelResult(
            timestamp=row['timestamp'],
            symbol=row['symbol'],
            base_signal=1,  # Assuming all base signals are positive
            meta_probability=test_probas[i],
            meta_prediction=test_predictions[i],
            features=row.to_dict(),
            confidence=abs(test_probas[i] - 0.5) * 2  # Distance from 0.5, scaled to [0,1]
        ))
    
    # Evaluate performance
    metrics = meta_labeler.evaluate(test_df)
    
    return results, metrics


# Example usage and testing functions
def example_usage():
    """Example usage of triple barrier labeling system."""
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252))),
        index=dates
    )
    
    # Create sample events (trading signals)
    signal_dates = dates[::10]  # Every 10th day
    events = pd.DataFrame({
        'timestamp': signal_dates,
        'side': np.random.choice([1, -1], size=len(signal_dates))
    })
    
    # Configure triple barrier labeling
    config = TripleBarrierConfig(
        horizon_days=5,
        upper_sigma=2.0,
        lower_sigma=1.5,
        volatility_lookback=20
    )
    
    # Create labels
    results, metrics = create_meta_labels(prices, events, config)
    
    print(f"Created {len(results)} meta-labels")
    print(f"Performance metrics: {metrics}")
    
    return results, metrics


if __name__ == "__main__":
    example_usage()