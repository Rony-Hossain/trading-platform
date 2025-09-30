"""
Triple-Barrier Labeling with Meta-Labeling for Advanced Event Trading.

Implements the triple-barrier method from "Advances in Financial Machine Learning" 
by Marcos López de Prado, with meta-labeling for enhanced signal quality.

The triple-barrier method creates labels based on:
1. Profit-taking barrier (upper threshold)
2. Stop-loss barrier (lower threshold) 
3. Time-based barrier (maximum holding period)

Meta-labeling adds a secondary model to filter primary model signals,
improving precision by predicting when the primary model is correct.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


@dataclass
class TripleBarrierLabel:
    """Result of triple-barrier labeling for a single event."""
    timestamp: datetime
    symbol: str
    side: int  # 1 for long, -1 for short, 0 for no position
    target_price: float
    profit_barrier: float
    stop_barrier: float
    time_barrier: datetime
    exit_timestamp: Optional[datetime]
    exit_price: Optional[float]
    exit_reason: Optional[str]  # 'profit', 'stop', 'time'
    return_pct: Optional[float]
    label: Optional[int]  # 1 for profit, -1 for loss, 0 for neutral
    holding_period: Optional[timedelta]
    confidence: Optional[float]


@dataclass
class MetaLabelResult:
    """Result of meta-labeling process."""
    original_signal: int
    meta_prediction: int  # 1 to take signal, 0 to skip
    meta_probability: float
    final_signal: int  # original_signal if meta_prediction==1, else 0
    features: Dict[str, float]
    model_confidence: float


class TripleBarrierLabeler:
    """
    Triple-barrier labeling system for creating machine learning labels
    from price data and trading signals.
    """
    
    def __init__(
        self,
        profit_threshold: float = 0.02,  # 2% profit target
        stop_threshold: float = 0.01,    # 1% stop loss
        max_holding_period: str = '5D',  # 5 days maximum
        min_return_threshold: float = 0.0001,  # Minimum return to consider
        volatility_adjustment: bool = True,
        dynamic_barriers: bool = False
    ):
        self.profit_threshold = profit_threshold
        self.stop_threshold = stop_threshold
        self.max_holding_period = pd.Timedelta(max_holding_period)
        self.min_return_threshold = min_return_threshold
        self.volatility_adjustment = volatility_adjustment
        self.dynamic_barriers = dynamic_barriers
        
    def get_events(
        self,
        prices: pd.Series,
        signals: pd.Series = None,
        volatility: pd.Series = None,
        min_event_separation: str = '1H'
    ) -> pd.DataFrame:
        """
        Extract trading events from price data and optional signals.
        
        Args:
            prices: Time series of prices
            signals: Optional trading signals (-1, 0, 1)
            volatility: Optional volatility estimates for dynamic barriers
            min_event_separation: Minimum time between events
            
        Returns:
            DataFrame with event timestamps and metadata
        """
        events = []
        
        if signals is None:
            # Generate events based on price movements
            returns = prices.pct_change()
            events_idx = self._identify_price_events(returns, volatility)
        else:
            # Use provided signals
            events_idx = signals[signals != 0].index
            
        # Filter events by minimum separation
        min_sep = pd.Timedelta(min_event_separation)
        filtered_events = []
        last_event = None
        
        for event_time in events_idx:
            if last_event is None or (event_time - last_event) >= min_sep:
                filtered_events.append(event_time)
                last_event = event_time
                
        # Create events DataFrame
        for event_time in filtered_events:
            side = 1 if signals is None else signals.loc[event_time]
            vol = volatility.loc[event_time] if volatility is not None else None
            
            events.append({
                'timestamp': event_time,
                'price': prices.loc[event_time],
                'side': side,
                'volatility': vol
            })
            
        return pd.DataFrame(events).set_index('timestamp')
    
    def _identify_price_events(
        self,
        returns: pd.Series,
        volatility: pd.Series = None
    ) -> pd.DatetimeIndex:
        """Identify significant price movement events."""
        if volatility is not None:
            # Use volatility-adjusted thresholds
            threshold = 2.0 * volatility.rolling(20).mean()
            significant_moves = np.abs(returns) > threshold
        else:
            # Use fixed threshold
            threshold = 2.0 * returns.rolling(20).std()
            significant_moves = np.abs(returns) > threshold
            
        return returns[significant_moves].index
    
    def apply_triple_barrier(
        self,
        events: pd.DataFrame,
        prices: pd.DataFrame,
        volatility: pd.Series = None
    ) -> List[TripleBarrierLabel]:
        """
        Apply triple-barrier method to events.
        
        Args:
            events: DataFrame with event data (from get_events)
            prices: DataFrame with OHLCV data
            volatility: Optional volatility estimates
            
        Returns:
            List of TripleBarrierLabel objects
        """
        labels = []
        
        for event_time, event_data in events.iterrows():
            try:
                label = self._process_single_event(
                    event_time, event_data, prices, volatility
                )
                labels.append(label)
            except Exception as e:
                logger.warning(f"Failed to process event at {event_time}: {e}")
                continue
                
        return labels
    
    def _process_single_event(
        self,
        event_time: pd.Timestamp,
        event_data: pd.Series,
        prices: pd.DataFrame,
        volatility: pd.Series = None
    ) -> TripleBarrierLabel:
        """Process a single event through triple-barrier method."""
        
        entry_price = event_data['price']
        side = event_data['side']
        symbol = getattr(event_data, 'symbol', 'unknown')
        
        # Calculate barriers
        barriers = self._calculate_barriers(
            entry_price, side, event_time, volatility
        )
        
        # Find exit point
        exit_info = self._find_exit_point(
            event_time, entry_price, side, barriers, prices
        )
        
        # Calculate return and label
        if exit_info['exit_price'] is not None:
            return_pct = self._calculate_return(
                entry_price, exit_info['exit_price'], side
            )
            label = self._assign_label(return_pct, barriers['profit_threshold'])
        else:
            return_pct = None
            label = None
            
        # Calculate confidence based on barrier distance
        confidence = self._calculate_confidence(
            return_pct, barriers['profit_threshold'], barriers['stop_threshold']
        ) if return_pct is not None else None
        
        return TripleBarrierLabel(
            timestamp=event_time,
            symbol=symbol,
            side=side,
            target_price=entry_price,
            profit_barrier=barriers['profit_barrier'],
            stop_barrier=barriers['stop_barrier'],
            time_barrier=barriers['time_barrier'],
            exit_timestamp=exit_info['exit_timestamp'],
            exit_price=exit_info['exit_price'],
            exit_reason=exit_info['exit_reason'],
            return_pct=return_pct,
            label=label,
            holding_period=exit_info['holding_period'],
            confidence=confidence
        )
    
    def _calculate_barriers(
        self,
        entry_price: float,
        side: int,
        event_time: pd.Timestamp,
        volatility: pd.Series = None
    ) -> Dict[str, Union[float, pd.Timestamp]]:
        """Calculate profit, stop, and time barriers for an event."""
        
        # Adjust thresholds based on volatility if available
        if self.volatility_adjustment and volatility is not None:
            vol_factor = volatility.loc[event_time] / volatility.rolling(20).mean().loc[event_time]
            vol_factor = np.clip(vol_factor, 0.5, 2.0)  # Limit adjustment
            profit_threshold = self.profit_threshold * vol_factor
            stop_threshold = self.stop_threshold * vol_factor
        else:
            profit_threshold = self.profit_threshold
            stop_threshold = self.stop_threshold
            
        # Calculate barrier prices
        if side == 1:  # Long position
            profit_barrier = entry_price * (1 + profit_threshold)
            stop_barrier = entry_price * (1 - stop_threshold)
        else:  # Short position
            profit_barrier = entry_price * (1 - profit_threshold)
            stop_barrier = entry_price * (1 + stop_threshold)
            
        time_barrier = event_time + self.max_holding_period
        
        return {
            'profit_barrier': profit_barrier,
            'stop_barrier': stop_barrier,
            'time_barrier': time_barrier,
            'profit_threshold': profit_threshold,
            'stop_threshold': stop_threshold
        }
    
    def _find_exit_point(
        self,
        entry_time: pd.Timestamp,
        entry_price: float,
        side: int,
        barriers: Dict,
        prices: pd.DataFrame
    ) -> Dict:
        """Find the exit point based on triple barriers."""
        
        # Get price data after entry
        future_prices = prices[prices.index > entry_time].copy()
        
        if future_prices.empty:
            return {
                'exit_timestamp': None,
                'exit_price': None,
                'exit_reason': None,
                'holding_period': None
            }
            
        # Check each time point for barrier breach
        for timestamp, price_data in future_prices.iterrows():
            
            # Use appropriate price for barrier checking
            high_price = price_data.get('high', price_data.get('close', price_data))
            low_price = price_data.get('low', price_data.get('close', price_data))
            close_price = price_data.get('close', price_data)
            
            # Check profit barrier
            if side == 1 and high_price >= barriers['profit_barrier']:
                return {
                    'exit_timestamp': timestamp,
                    'exit_price': barriers['profit_barrier'],
                    'exit_reason': 'profit',
                    'holding_period': timestamp - entry_time
                }
            elif side == -1 and low_price <= barriers['profit_barrier']:
                return {
                    'exit_timestamp': timestamp,
                    'exit_price': barriers['profit_barrier'],
                    'exit_reason': 'profit',
                    'holding_period': timestamp - entry_time
                }
                
            # Check stop barrier
            if side == 1 and low_price <= barriers['stop_barrier']:
                return {
                    'exit_timestamp': timestamp,
                    'exit_price': barriers['stop_barrier'],
                    'exit_reason': 'stop',
                    'holding_period': timestamp - entry_time
                }
            elif side == -1 and high_price >= barriers['stop_barrier']:
                return {
                    'exit_timestamp': timestamp,
                    'exit_price': barriers['stop_barrier'],
                    'exit_reason': 'stop',
                    'holding_period': timestamp - entry_time
                }
                
            # Check time barrier
            if timestamp >= barriers['time_barrier']:
                return {
                    'exit_timestamp': timestamp,
                    'exit_price': close_price,
                    'exit_reason': 'time',
                    'holding_period': timestamp - entry_time
                }
                
        # No barrier hit within available data
        return {
            'exit_timestamp': None,
            'exit_price': None,
            'exit_reason': None,
            'holding_period': None
        }
    
    def _calculate_return(self, entry_price: float, exit_price: float, side: int) -> float:
        """Calculate return for a trade."""
        if side == 1:  # Long
            return (exit_price - entry_price) / entry_price
        else:  # Short
            return (entry_price - exit_price) / entry_price
    
    def _assign_label(self, return_pct: float, profit_threshold: float) -> int:
        """Assign label based on return."""
        if return_pct >= profit_threshold * 0.5:  # 50% of profit threshold
            return 1  # Profit
        elif return_pct <= -profit_threshold * 0.25:  # 25% of profit threshold
            return -1  # Loss
        else:
            return 0  # Neutral
    
    def _calculate_confidence(
        self,
        return_pct: float,
        profit_threshold: float,
        stop_threshold: float
    ) -> float:
        """Calculate confidence score based on barrier distance."""
        if return_pct is None:
            return 0.0
            
        # Confidence based on how far the return is from neutral zone
        abs_return = abs(return_pct)
        max_threshold = max(profit_threshold, stop_threshold)
        
        confidence = min(abs_return / max_threshold, 1.0)
        return confidence


class MetaLabeler:
    """
    Meta-labeling system to improve signal quality by predicting
    when primary model signals should be taken.
    """
    
    def __init__(
        self,
        base_estimator=None,
        cv_folds: int = 5,
        min_precision: float = 0.55,
        feature_importance_threshold: float = 0.01
    ):
        self.base_estimator = base_estimator or RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        self.cv_folds = cv_folds
        self.min_precision = min_precision
        self.feature_importance_threshold = feature_importance_threshold
        self.is_fitted = False
        self.feature_names = []
        
    def prepare_features(
        self,
        labels: List[TripleBarrierLabel],
        prices: pd.DataFrame,
        volume: pd.Series = None,
        volatility: pd.Series = None,
        sentiment: pd.Series = None
    ) -> pd.DataFrame:
        """
        Prepare features for meta-labeling model.
        
        Features include:
        - Primary signal strength
        - Market conditions (volatility, volume)
        - Technical indicators
        - Sentiment indicators (if available)
        """
        features_list = []
        
        for label in labels:
            if label.exit_timestamp is None:
                continue
                
            features = self._extract_features(
                label, prices, volume, volatility, sentiment
            )
            features['target'] = 1 if label.label == 1 else 0  # Binary: profit or not
            features['timestamp'] = label.timestamp
            
            features_list.append(features)
            
        df = pd.DataFrame(features_list)
        df.set_index('timestamp', inplace=True)
        
        return df.dropna()
    
    def _extract_features(
        self,
        label: TripleBarrierLabel,
        prices: pd.DataFrame,
        volume: pd.Series = None,
        volatility: pd.Series = None,
        sentiment: pd.Series = None
    ) -> Dict[str, float]:
        """Extract features for a single label."""
        
        features = {}
        event_time = label.timestamp
        
        # Get historical window
        window_start = event_time - pd.Timedelta('10D')
        hist_prices = prices[window_start:event_time]
        
        if hist_prices.empty:
            return features
            
        # Primary signal features
        features['signal_strength'] = abs(label.side)
        features['signal_direction'] = label.side
        
        # Price-based features
        close_prices = hist_prices['close'] if 'close' in hist_prices.columns else hist_prices.iloc[:, 0]
        features['price_momentum_5d'] = (close_prices.iloc[-1] / close_prices.iloc[-5] - 1) if len(close_prices) >= 5 else 0
        features['price_momentum_10d'] = (close_prices.iloc[-1] / close_prices.iloc[-10] - 1) if len(close_prices) >= 10 else 0
        
        # Volatility features
        returns = close_prices.pct_change().dropna()
        features['realized_vol_5d'] = returns.tail(5).std() * np.sqrt(252) if len(returns) >= 5 else 0
        features['realized_vol_10d'] = returns.tail(10).std() * np.sqrt(252) if len(returns) >= 10 else 0
        
        if volatility is not None and event_time in volatility.index:
            features['implied_vol'] = volatility.loc[event_time]
            features['vol_rank'] = stats.percentileofscore(
                volatility[window_start:event_time].dropna(), 
                volatility.loc[event_time]
            ) / 100.0
        
        # Volume features
        if volume is not None:
            hist_volume = volume[window_start:event_time]
            if not hist_volume.empty and event_time in volume.index:
                features['volume_ratio'] = volume.loc[event_time] / hist_volume.mean()
                features['volume_trend'] = hist_volume.tail(5).mean() / hist_volume.tail(10).mean() if len(hist_volume) >= 10 else 1.0
        
        # Technical indicators
        if len(close_prices) >= 20:
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            features['price_vs_sma20'] = close_prices.iloc[-1] / sma_20 - 1
            
        if len(close_prices) >= 14:
            # Simple RSI calculation
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) else 50
        
        # Sentiment features
        if sentiment is not None:
            hist_sentiment = sentiment[window_start:event_time]
            if not hist_sentiment.empty:
                features['sentiment_level'] = hist_sentiment.iloc[-1] if event_time in sentiment.index else hist_sentiment.mean()
                features['sentiment_trend'] = hist_sentiment.tail(3).mean() - hist_sentiment.tail(10).mean() if len(hist_sentiment) >= 10 else 0
        
        # Market structure features
        if 'high' in hist_prices.columns and 'low' in hist_prices.columns:
            hl_ratio = (hist_prices['high'] / hist_prices['low']).tail(5).mean()
            features['avg_daily_range'] = hl_ratio - 1
        
        # Time-based features
        features['hour_of_day'] = event_time.hour
        features['day_of_week'] = event_time.dayofweek
        features['is_month_end'] = 1 if event_time.day >= 25 else 0
        
        return features
    
    def fit(
        self,
        features_df: pd.DataFrame,
        sample_weights: pd.Series = None
    ) -> Dict[str, float]:
        """
        Fit meta-labeling model.
        
        Args:
            features_df: DataFrame with features and target
            sample_weights: Optional sample weights
            
        Returns:
            Dictionary with model performance metrics
        """
        
        if 'target' not in features_df.columns:
            raise ValueError("features_df must contain 'target' column")
            
        # Prepare data
        X = features_df.drop('target', axis=1)
        y = features_df['target']
        
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Cross-validation for model selection
        cv_scores = cross_val_score(
            self.base_estimator, X, y, cv=self.cv_folds,
            scoring='precision', n_jobs=-1
        )
        
        # Fit final model
        self.base_estimator.fit(X, y, sample_weight=sample_weights)
        self.is_fitted = True
        
        # Calculate performance metrics
        y_pred = self.base_estimator.predict(X)
        y_pred_proba = self.base_estimator.predict_proba(X)[:, 1]
        
        metrics = {
            'cv_precision_mean': cv_scores.mean(),
            'cv_precision_std': cv_scores.std(),
            'train_precision': precision_score(y, y_pred),
            'train_recall': recall_score(y, y_pred),
            'train_f1': f1_score(y, y_pred),
            'feature_importance': dict(zip(
                self.feature_names,
                self.base_estimator.feature_importances_
            ))
        }
        
        logger.info(f"Meta-labeling model fitted with CV precision: {metrics['cv_precision_mean']:.3f} ± {metrics['cv_precision_std']:.3f}")
        
        return metrics
    
    def predict(
        self,
        features: Dict[str, float],
        original_signal: int
    ) -> MetaLabelResult:
        """
        Generate meta-label prediction for a single signal.
        
        Args:
            features: Feature dictionary
            original_signal: Original trading signal (-1, 0, 1)
            
        Returns:
            MetaLabelResult with prediction and metadata
        """
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Prepare feature vector
        X = pd.DataFrame([features], columns=self.feature_names)
        X = X.fillna(X.median())
        
        # Make prediction
        meta_pred = self.base_estimator.predict(X)[0]
        meta_proba = self.base_estimator.predict_proba(X)[0, 1]
        
        # Model confidence based on prediction probability
        model_confidence = max(meta_proba, 1 - meta_proba)
        
        # Final signal: take original signal only if meta-model agrees
        final_signal = original_signal if meta_pred == 1 else 0
        
        return MetaLabelResult(
            original_signal=original_signal,
            meta_prediction=meta_pred,
            meta_probability=meta_proba,
            final_signal=final_signal,
            features=features,
            model_confidence=model_confidence
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        importance_dict = dict(zip(
            self.feature_names,
            self.base_estimator.feature_importances_
        ))
        
        # Filter by importance threshold
        return {
            k: v for k, v in importance_dict.items()
            if v >= self.feature_importance_threshold
        }


class AdvancedLabelingSystem:
    """
    Complete system combining triple-barrier labeling with meta-labeling
    for advanced event trading label generation.
    """
    
    def __init__(
        self,
        labeler_config: Dict = None,
        meta_labeler_config: Dict = None
    ):
        labeler_config = labeler_config or {}
        meta_labeler_config = meta_labeler_config or {}
        
        self.triple_barrier = TripleBarrierLabeler(**labeler_config)
        self.meta_labeler = MetaLabeler(**meta_labeler_config)
        self.labels_history = []
        
    def create_labels(
        self,
        prices: pd.DataFrame,
        signals: pd.Series = None,
        volume: pd.Series = None,
        volatility: pd.Series = None,
        sentiment: pd.Series = None,
        fit_meta_model: bool = True
    ) -> Tuple[List[TripleBarrierLabel], List[MetaLabelResult]]:
        """
        Create triple-barrier labels and apply meta-labeling.
        
        Args:
            prices: OHLCV price data
            signals: Trading signals
            volume: Volume data
            volatility: Volatility estimates
            sentiment: Sentiment indicators
            fit_meta_model: Whether to fit meta-labeling model
            
        Returns:
            Tuple of (triple_barrier_labels, meta_label_results)
        """
        
        # Extract events
        price_series = prices['close'] if 'close' in prices.columns else prices.iloc[:, 0]
        events = self.triple_barrier.get_events(
            price_series, signals, volatility
        )
        
        # Apply triple-barrier labeling
        labels = self.triple_barrier.apply_triple_barrier(
            events, prices, volatility
        )
        
        self.labels_history.extend(labels)
        
        # Prepare features for meta-labeling
        features_df = self.meta_labeler.prepare_features(
            labels, prices, volume, volatility, sentiment
        )
        
        meta_results = []
        
        if fit_meta_model and len(features_df) > 50:  # Minimum samples for training
            # Fit meta-labeling model
            metrics = self.meta_labeler.fit(features_df)
            logger.info(f"Meta-labeling model performance: {metrics}")
            
            # Generate meta-predictions for all signals
            for label in labels:
                if label.timestamp in features_df.index:
                    features = features_df.loc[label.timestamp].drop('target').to_dict()
                    meta_result = self.meta_labeler.predict(features, label.side)
                    meta_results.append(meta_result)
        
        return labels, meta_results
    
    def get_label_statistics(self) -> Dict[str, float]:
        """Get statistics about generated labels."""
        if not self.labels_history:
            return {}
            
        completed_labels = [l for l in self.labels_history if l.label is not None]
        
        if not completed_labels:
            return {}
            
        profit_labels = sum(1 for l in completed_labels if l.label == 1)
        loss_labels = sum(1 for l in completed_labels if l.label == -1)
        neutral_labels = sum(1 for l in completed_labels if l.label == 0)
        
        avg_holding_period = pd.Series([
            l.holding_period.total_seconds() / 3600 for l in completed_labels 
            if l.holding_period is not None
        ]).mean()
        
        profit_exit_reasons = [l.exit_reason for l in completed_labels if l.label == 1]
        
        return {
            'total_labels': len(completed_labels),
            'profit_rate': profit_labels / len(completed_labels),
            'loss_rate': loss_labels / len(completed_labels),
            'neutral_rate': neutral_labels / len(completed_labels),
            'avg_holding_period_hours': avg_holding_period,
            'profit_by_profit_barrier': profit_exit_reasons.count('profit') / max(len(profit_exit_reasons), 1),
            'profit_by_time_barrier': profit_exit_reasons.count('time') / max(len(profit_exit_reasons), 1)
        }