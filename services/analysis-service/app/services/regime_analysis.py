"""
Regime Analysis Service
Implements ATR bands, realized volatility term structure, and HMM state detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import stats

# HMM Libraries (optional)
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not available, using simplified regime detection")

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE_TRENDING = "volatile_trending"
    CRISIS = "crisis"

@dataclass
class RegimeState:
    """Current market regime state"""
    regime: MarketRegime
    confidence: float
    volatility_percentile: float
    trend_strength: float
    atr_position: float
    vol_term_slope: float
    duration_days: int
    timestamp: datetime

@dataclass
class ATRBands:
    """ATR-based volatility bands"""
    upper_1x: float
    lower_1x: float
    upper_2x: float
    lower_2x: float
    position: float  # 0-1, where current price sits in bands
    atr_value: float
    atr_percentile: float

@dataclass
class VolatilityTermStructure:
    """Realized volatility across different time horizons"""
    vol_1d: float
    vol_5d: float
    vol_20d: float
    vol_60d: float
    vol_120d: float
    term_slope: float  # slope from short to long term
    vol_rank: float  # percentile rank of current vol

class RegimeAnalyzer:
    """Comprehensive regime analysis with ATR bands and volatility term structure"""
    
    def __init__(self, atr_period: int = 14, lookback_period: int = 252):
        self.atr_period = atr_period
        self.lookback_period = lookback_period
        
        # Regime thresholds
        self.vol_thresholds = {
            'low': 0.15,
            'medium': 0.25, 
            'high': 0.35,
            'extreme': 0.50
        }
        
        self.trend_thresholds = {
            'weak': 0.01,
            'moderate': 0.02,
            'strong': 0.04
        }
        
    def calculate_atr_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR bands and position indicators"""
        high, low, close = data['high'], data['low'], data['close']
        
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        # Calculate bands
        upper_1x = close + atr
        lower_1x = close - atr
        upper_2x = close + (2 * atr)
        lower_2x = close - (2 * atr)
        
        # Calculate position within bands (0 = lower band, 1 = upper band)
        band_position = (close - lower_1x) / (upper_1x - lower_1x)
        band_position = band_position.clip(0, 1)
        
        # Calculate ATR percentile rank
        atr_percentile = atr.rolling(window=self.lookback_period).rank(pct=True)
        
        return pd.DataFrame({
            'atr': atr,
            'upper_1x': upper_1x,
            'lower_1x': lower_1x,
            'upper_2x': upper_2x,
            'lower_2x': lower_2x,
            'atr_position': band_position,
            'atr_percentile': atr_percentile
        }, index=data.index)
    
    def calculate_volatility_term_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate realized volatility term structure"""
        returns = data['close'].pct_change().dropna()
        
        # Calculate volatility for different periods
        vol_1d = returns.rolling(1).std() * np.sqrt(252)
        vol_5d = returns.rolling(5).std() * np.sqrt(252)
        vol_20d = returns.rolling(20).std() * np.sqrt(252)
        vol_60d = returns.rolling(60).std() * np.sqrt(252)
        vol_120d = returns.rolling(120).std() * np.sqrt(252)
        
        # Calculate term structure slope (long vol - short vol) / short vol
        term_slope = (vol_60d - vol_5d) / vol_5d
        
        # Calculate volatility rank (percentile over lookback period)
        vol_rank = vol_20d.rolling(window=self.lookback_period).rank(pct=True)
        
        return pd.DataFrame({
            'vol_1d': vol_1d,
            'vol_5d': vol_5d,
            'vol_20d': vol_20d,
            'vol_60d': vol_60d,
            'vol_120d': vol_120d,
            'vol_term_slope': term_slope,
            'vol_rank': vol_rank
        }, index=data.index)
    
    def calculate_trend_strength(self, data: pd.DataFrame, periods: List[int] = [20, 60]) -> pd.DataFrame:
        """Calculate trend strength using multiple methods"""
        close = data['close']
        trend_features = pd.DataFrame(index=data.index)
        
        for period in periods:
            # Linear regression slope
            slopes = []
            for i in range(period, len(close)):
                y = close.iloc[i-period:i].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope, _, r_value, _, _ = stats.linregress(x, y)
                    # Normalize by price and weight by R-squared
                    normalized_slope = (slope / y[-1]) * (r_value ** 2)
                    slopes.append(normalized_slope)
                else:
                    slopes.append(0)
            
            # Pad with NaN for initial values
            trend_slopes = [np.nan] * period + slopes
            trend_features[f'trend_strength_{period}d'] = trend_slopes
            
            # Moving average convergence
            sma_short = close.rolling(period // 4).mean()
            sma_long = close.rolling(period).mean()
            ma_convergence = (sma_short - sma_long) / sma_long
            trend_features[f'ma_convergence_{period}d'] = ma_convergence
            
            # Price momentum
            momentum = close.pct_change(period)
            trend_features[f'momentum_{period}d'] = momentum
        
        return trend_features
    
    def calculate_mean_reversion_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators"""
        close = data['close']
        
        # Bollinger Bands mean reversion
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        bb_position = (close - sma_20) / (2 * std_20)
        
        # RSI-like mean reversion
        price_change = close.diff()
        gains = price_change.where(price_change > 0, 0)
        losses = -price_change.where(price_change < 0, 0)
        
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        rsi_normalized = (rsi - 50) / 50  # -1 to 1 scale
        
        # Price vs moving averages
        ma_deviation = {}
        for period in [10, 20, 50]:
            ma = close.rolling(period).mean()
            deviation = (close - ma) / ma
            ma_deviation[f'ma_dev_{period}d'] = deviation
        
        result = pd.DataFrame({
            'bb_position': bb_position,
            'rsi_normalized': rsi_normalized,
            **ma_deviation
        }, index=data.index)
        
        return result
    
    def classify_regime(self, atr_data: pd.DataFrame, vol_data: pd.DataFrame, 
                       trend_data: pd.DataFrame, mean_rev_data: pd.DataFrame) -> pd.DataFrame:
        """Classify market regime based on multiple indicators"""
        
        regimes = []
        confidences = []
        
        for i in range(len(atr_data)):
            # Get current values
            vol_20d = vol_data['vol_20d'].iloc[i] if not pd.isna(vol_data['vol_20d'].iloc[i]) else 0.2
            vol_rank = vol_data['vol_rank'].iloc[i] if not pd.isna(vol_data['vol_rank'].iloc[i]) else 0.5
            atr_percentile = atr_data['atr_percentile'].iloc[i] if not pd.isna(atr_data['atr_percentile'].iloc[i]) else 0.5
            trend_strength = abs(trend_data['trend_strength_20d'].iloc[i]) if not pd.isna(trend_data['trend_strength_20d'].iloc[i]) else 0
            momentum = abs(trend_data['momentum_20d'].iloc[i]) if not pd.isna(trend_data['momentum_20d'].iloc[i]) else 0
            bb_position = abs(mean_rev_data['bb_position'].iloc[i]) if not pd.isna(mean_rev_data['bb_position'].iloc[i]) else 0
            
            # Regime classification logic
            regime = MarketRegime.MEAN_REVERTING  # Default
            confidence = 0.5
            
            # Crisis: High volatility + significant negative momentum
            if vol_20d > self.vol_thresholds['extreme'] and trend_data['momentum_20d'].iloc[i] < -0.1:
                regime = MarketRegime.CRISIS
                confidence = min(0.9, vol_rank + 0.3)
            
            # High Volatility: High vol rank but not crisis
            elif vol_rank > 0.8 or vol_20d > self.vol_thresholds['high']:
                if trend_strength > self.trend_thresholds['moderate']:
                    regime = MarketRegime.VOLATILE_TRENDING
                    confidence = min(0.9, vol_rank + trend_strength / 0.04)
                else:
                    regime = MarketRegime.HIGH_VOLATILITY
                    confidence = vol_rank
            
            # Low Volatility: Low vol rank
            elif vol_rank < 0.2 or vol_20d < self.vol_thresholds['low']:
                regime = MarketRegime.LOW_VOLATILITY
                confidence = 1 - vol_rank
            
            # Trending: Strong trend with moderate volatility
            elif trend_strength > self.trend_thresholds['moderate'] and vol_rank < 0.7:
                regime = MarketRegime.TRENDING
                confidence = min(0.9, trend_strength / 0.04)
            
            # Mean Reverting: High mean reversion signals
            elif bb_position > 1.5 or any(abs(mean_rev_data[col].iloc[i]) > 0.05 for col in mean_rev_data.columns if 'ma_dev' in col):
                regime = MarketRegime.MEAN_REVERTING
                confidence = min(0.9, bb_position / 2)
            
            regimes.append(regime)
            confidences.append(max(0.1, min(0.95, confidence)))
        
        return pd.DataFrame({
            'regime': regimes,
            'confidence': confidences
        }, index=atr_data.index)
    
    def detect_regime_transitions(self, regime_series: pd.Series) -> List[Dict[str, Any]]:
        """Detect regime transitions and their characteristics"""
        transitions = []
        
        for i in range(1, len(regime_series)):
            current_regime = regime_series.iloc[i]
            previous_regime = regime_series.iloc[i-1]
            
            if current_regime != previous_regime:
                # Calculate persistence (how long new regime lasts)
                persistence_days = 1
                for j in range(i + 1, min(i + 30, len(regime_series))):
                    if regime_series.iloc[j] == current_regime:
                        persistence_days += 1
                    else:
                        break
                
                transitions.append({
                    'date': regime_series.index[i],
                    'from_regime': previous_regime.value,
                    'to_regime': current_regime.value,
                    'persistence_days': persistence_days,
                    'transition_index': i
                })
        
        return transitions
    
    def analyze_regime_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Complete regime analysis pipeline"""
        try:
            logger.info("Starting regime feature analysis")
            
            if len(data) < 120:
                raise ValueError("Insufficient data for regime analysis (need at least 120 days)")
            
            # Calculate all features
            atr_features = self.calculate_atr_bands(data)
            vol_features = self.calculate_volatility_term_structure(data)
            trend_features = self.calculate_trend_strength(data)
            mean_rev_features = self.calculate_mean_reversion_signals(data)
            
            # Classify regimes
            regime_classification = self.classify_regime(
                atr_features, vol_features, trend_features, mean_rev_features
            )
            
            # Detect transitions
            transitions = self.detect_regime_transitions(regime_classification['regime'])
            
            # HMM regime detection
            hmm_results = self.detect_hmm_regimes(data)
            
            # Current state
            current_idx = -1
            current_state = RegimeState(
                regime=regime_classification['regime'].iloc[current_idx],
                confidence=regime_classification['confidence'].iloc[current_idx],
                volatility_percentile=vol_features['vol_rank'].iloc[current_idx] if not pd.isna(vol_features['vol_rank'].iloc[current_idx]) else 0.5,
                trend_strength=abs(trend_features['trend_strength_20d'].iloc[current_idx]) if not pd.isna(trend_features['trend_strength_20d'].iloc[current_idx]) else 0,
                atr_position=atr_features['atr_position'].iloc[current_idx] if not pd.isna(atr_features['atr_position'].iloc[current_idx]) else 0.5,
                vol_term_slope=vol_features['vol_term_slope'].iloc[current_idx] if not pd.isna(vol_features['vol_term_slope'].iloc[current_idx]) else 0,
                duration_days=self._calculate_current_regime_duration(regime_classification['regime']),
                timestamp=datetime.now()
            )
            
            # Feature importance analysis
            feature_importance = self._analyze_feature_importance(
                atr_features, vol_features, trend_features, mean_rev_features
            )
            
            # Regime statistics
            regime_stats = self._calculate_regime_statistics(
                regime_classification, vol_features, trend_features
            )
            
            return {
                'current_state': current_state,
                'atr_features': atr_features,
                'volatility_features': vol_features,
                'trend_features': trend_features,
                'mean_reversion_features': mean_rev_features,
                'regime_classification': regime_classification,
                'transitions': transitions,
                'hmm_analysis': hmm_results,
                'feature_importance': feature_importance,
                'regime_statistics': regime_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Regime analysis failed: {str(e)}")
            raise
    
    def _calculate_current_regime_duration(self, regime_series: pd.Series) -> int:
        """Calculate how long current regime has persisted"""
        if len(regime_series) == 0:
            return 0
        
        current_regime = regime_series.iloc[-1]
        duration = 1
        
        for i in range(len(regime_series) - 2, -1, -1):
            if regime_series.iloc[i] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _analyze_feature_importance(self, atr_features: pd.DataFrame, vol_features: pd.DataFrame,
                                  trend_features: pd.DataFrame, mean_rev_features: pd.DataFrame) -> Dict[str, float]:
        """Analyze which features are most important for regime classification"""
        # Simple correlation-based importance
        importance = {}
        
        # ATR features
        atr_vol_corr = abs(atr_features['atr_percentile'].corr(vol_features['vol_rank']))
        importance['atr_bands'] = atr_vol_corr if not pd.isna(atr_vol_corr) else 0
        
        # Volatility term structure
        vol_stability = 1 - vol_features['vol_term_slope'].std() if not pd.isna(vol_features['vol_term_slope'].std()) else 0.5
        importance['vol_term_structure'] = vol_stability
        
        # Trend features
        trend_consistency = 1 - trend_features['trend_strength_20d'].std() if not pd.isna(trend_features['trend_strength_20d'].std()) else 0.5
        importance['trend_strength'] = trend_consistency
        
        # Mean reversion
        mean_rev_signal = abs(mean_rev_features['bb_position'].mean()) if not pd.isna(mean_rev_features['bb_position'].mean()) else 0
        importance['mean_reversion'] = mean_rev_signal
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def detect_hmm_regimes(self, data: pd.DataFrame, n_regimes: int = 3) -> Dict[str, Any]:
        """Detect market regimes using Hidden Markov Model or Gaussian Mixture as fallback"""
        try:
            # Prepare features for regime detection
            returns = data['close'].pct_change().dropna()
            vol = returns.rolling(20).std()
            
            # Create feature matrix
            features = pd.DataFrame({
                'returns': returns,
                'volatility': vol,
                'abs_returns': returns.abs(),
                'volume_change': data['volume'].pct_change() if 'volume' in data.columns else returns * 0
            }).dropna()
            
            if len(features) < 100:
                raise ValueError("Insufficient data for HMM regime detection")
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            hmm_results = {}
            
            if HMM_AVAILABLE:
                # Use HMM if available
                try:
                    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100)
                    model.fit(features_scaled)
                    
                    # Predict regimes
                    hidden_states = model.predict(features_scaled)
                    state_probs = model.predict_proba(features_scaled)
                    
                    # Map HMM states to market regimes
                    state_characteristics = {}
                    for state in range(n_regimes):
                        state_mask = hidden_states == state
                        state_returns = features.loc[features.index[state_mask], 'returns'].mean()
                        state_vol = features.loc[features.index[state_mask], 'volatility'].mean()
                        
                        # Classify HMM state based on characteristics
                        if state_vol > features['volatility'].quantile(0.75):
                            if state_returns < -0.01:
                                regime_type = MarketRegime.CRISIS
                            else:
                                regime_type = MarketRegime.HIGH_VOLATILITY
                        elif state_vol < features['volatility'].quantile(0.25):
                            regime_type = MarketRegime.LOW_VOLATILITY
                        else:
                            if abs(state_returns) > 0.005:
                                regime_type = MarketRegime.TRENDING
                            else:
                                regime_type = MarketRegime.MEAN_REVERTING
                        
                        state_characteristics[state] = {
                            'regime_type': regime_type,
                            'avg_return': state_returns,
                            'avg_volatility': state_vol,
                            'frequency': np.sum(state_mask) / len(hidden_states)
                        }
                    
                    # Map states to regime types
                    regime_mapping = []
                    confidence_scores = []
                    
                    for i, state in enumerate(hidden_states):
                        regime_mapping.append(state_characteristics[state]['regime_type'])
                        confidence_scores.append(state_probs[i, state])
                    
                    hmm_results = {
                        'method': 'HMM',
                        'regimes': regime_mapping,
                        'confidence': confidence_scores,
                        'state_characteristics': state_characteristics,
                        'transition_matrix': model.transmat_.tolist(),
                        'success': True
                    }
                    
                    logger.info("HMM regime detection completed successfully")
                    
                except Exception as e:
                    logger.warning(f"HMM failed, falling back to Gaussian Mixture: {str(e)}")
                    hmm_results = self._fallback_regime_detection(features_scaled, features, n_regimes)
            else:
                # Use Gaussian Mixture Model as fallback
                hmm_results = self._fallback_regime_detection(features_scaled, features, n_regimes)
            
            return hmm_results
            
        except Exception as e:
            logger.error(f"HMM regime detection failed: {str(e)}")
            return {'method': 'failed', 'error': str(e), 'success': False}
    
    def _fallback_regime_detection(self, features_scaled: np.ndarray, features: pd.DataFrame, n_regimes: int) -> Dict[str, Any]:
        """Fallback regime detection using Gaussian Mixture Model"""
        try:
            # Use Gaussian Mixture Model
            gmm = GaussianMixture(n_components=n_regimes, random_state=42)
            gmm.fit(features_scaled)
            
            # Predict clusters
            cluster_labels = gmm.predict(features_scaled)
            cluster_probs = gmm.predict_proba(features_scaled)
            
            # Characterize clusters
            cluster_characteristics = {}
            for cluster in range(n_regimes):
                cluster_mask = cluster_labels == cluster
                cluster_returns = features.loc[features.index[cluster_mask], 'returns'].mean()
                cluster_vol = features.loc[features.index[cluster_mask], 'volatility'].mean()
                
                # Map clusters to regime types
                if cluster_vol > features['volatility'].quantile(0.7):
                    if cluster_returns < -0.01:
                        regime_type = MarketRegime.CRISIS
                    else:
                        regime_type = MarketRegime.HIGH_VOLATILITY
                elif cluster_vol < features['volatility'].quantile(0.3):
                    regime_type = MarketRegime.LOW_VOLATILITY
                else:
                    if abs(cluster_returns) > 0.005:
                        regime_type = MarketRegime.TRENDING
                    else:
                        regime_type = MarketRegime.MEAN_REVERTING
                
                cluster_characteristics[cluster] = {
                    'regime_type': regime_type,
                    'avg_return': cluster_returns,
                    'avg_volatility': cluster_vol,
                    'frequency': np.sum(cluster_mask) / len(cluster_labels)
                }
            
            # Map clusters to regime types
            regime_mapping = []
            confidence_scores = []
            
            for i, cluster in enumerate(cluster_labels):
                regime_mapping.append(cluster_characteristics[cluster]['regime_type'])
                confidence_scores.append(cluster_probs[i, cluster])
            
            logger.info("Gaussian Mixture regime detection completed")
            
            return {
                'method': 'Gaussian_Mixture',
                'regimes': regime_mapping,
                'confidence': confidence_scores,
                'cluster_characteristics': cluster_characteristics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Fallback regime detection failed: {str(e)}")
            return {'method': 'failed', 'error': str(e), 'success': False}
    
    def _calculate_regime_statistics(self, regime_classification: pd.DataFrame,
                                   vol_features: pd.DataFrame, trend_features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for each regime"""
        stats = {}
        
        for regime in MarketRegime:
            regime_mask = regime_classification['regime'] == regime
            regime_data = regime_classification[regime_mask]
            
            if len(regime_data) > 0:
                # Basic statistics
                frequency = len(regime_data) / len(regime_classification)
                avg_confidence = regime_data['confidence'].mean()
                
                # Associated market characteristics
                regime_vol = vol_features.loc[regime_mask, 'vol_20d'].mean() if len(vol_features.loc[regime_mask]) > 0 else 0
                regime_trend = trend_features.loc[regime_mask, 'trend_strength_20d'].mean() if len(trend_features.loc[regime_mask]) > 0 else 0
                
                stats[regime.value] = {
                    'frequency': frequency,
                    'avg_confidence': avg_confidence,
                    'avg_volatility': regime_vol,
                    'avg_trend_strength': abs(regime_trend) if not pd.isna(regime_trend) else 0,
                    'sample_count': len(regime_data)
                }
        
        return stats
    
    def get_regime_based_model_weights(self, current_regime: MarketRegime, confidence: float) -> Dict[str, float]:
        """Get recommended model weights based on current regime"""
        # Default weights
        weights = {'lstm': 0.33, 'gru': 0.33, 'random_forest': 0.34}
        
        # Regime-specific adjustments
        if current_regime == MarketRegime.TRENDING:
            # Neural networks better for trending markets
            weights = {'lstm': 0.4, 'gru': 0.4, 'random_forest': 0.2}
        
        elif current_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
            # Tree-based models better for high volatility
            weights = {'lstm': 0.2, 'gru': 0.2, 'random_forest': 0.6}
        
        elif current_regime == MarketRegime.LOW_VOLATILITY:
            # Neural networks better for low volatility
            weights = {'lstm': 0.45, 'gru': 0.35, 'random_forest': 0.2}
        
        elif current_regime == MarketRegime.MEAN_REVERTING:
            # Balanced with slight preference for tree models
            weights = {'lstm': 0.3, 'gru': 0.3, 'random_forest': 0.4}
        
        elif current_regime == MarketRegime.VOLATILE_TRENDING:
            # Hybrid approach
            weights = {'lstm': 0.35, 'gru': 0.35, 'random_forest': 0.3}
        
        # Adjust weights based on confidence
        if confidence < 0.5:
            # Low confidence - move towards equal weighting
            equal_weight = 1.0 / 3
            adjustment_factor = 1 - confidence
            for model in weights:
                weights[model] = weights[model] * (1 - adjustment_factor) + equal_weight * adjustment_factor
        
        return weights
    
    def validate_regime_detection(self, data: pd.DataFrame, validation_period: int = 60) -> Dict[str, Any]:
        """Validate regime detection accuracy across different market conditions"""
        try:
            logger.info("Starting regime detection validation")
            
            if len(data) < validation_period * 2:
                raise ValueError(f"Insufficient data for validation (need at least {validation_period * 2} days)")
            
            # Split data into validation windows
            window_size = validation_period
            windows = []
            
            for i in range(0, len(data) - window_size, window_size // 2):  # 50% overlap
                window_data = data.iloc[i:i + window_size]
                if len(window_data) >= window_size:
                    windows.append({
                        'data': window_data,
                        'start_date': window_data.index[0],
                        'end_date': window_data.index[-1],
                        'period_name': f"Window_{i//window_size + 1}"
                    })
            
            validation_results = []
            
            for window in windows:
                try:
                    # Analyze regime for this window
                    window_results = self.analyze_regime_features(window['data'])
                    
                    # Calculate validation metrics
                    regime_series = window_results['regime_classification']['regime']
                    confidence_series = window_results['regime_classification']['confidence']
                    
                    # Regime stability (how often regime changes)
                    regime_changes = sum(1 for i in range(1, len(regime_series)) 
                                       if regime_series.iloc[i] != regime_series.iloc[i-1])
                    stability_score = 1 - (regime_changes / len(regime_series))
                    
                    # Confidence consistency
                    avg_confidence = confidence_series.mean()
                    confidence_std = confidence_series.std()
                    
                    # Regime distribution
                    regime_counts = regime_series.value_counts()
                    regime_distribution = {regime.value: count / len(regime_series) 
                                         for regime, count in regime_counts.items()}
                    
                    # Market characteristics during period
                    returns = window['data']['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)
                    total_return = (window['data']['close'].iloc[-1] / window['data']['close'].iloc[0] - 1) * 100
                    max_drawdown = self._calculate_max_drawdown(window['data']['close'])
                    
                    # Validate regime detection against market characteristics
                    validation_score = self._validate_regime_accuracy(
                        regime_series, confidence_series, volatility, total_return, max_drawdown
                    )
                    
                    window_result = {
                        'period': window['period_name'],
                        'start_date': window['start_date'].isoformat(),
                        'end_date': window['end_date'].isoformat(),
                        'regime_stability': stability_score,
                        'avg_confidence': avg_confidence,
                        'confidence_std': confidence_std,
                        'regime_distribution': regime_distribution,
                        'market_characteristics': {
                            'volatility': volatility,
                            'total_return': total_return,
                            'max_drawdown': max_drawdown
                        },
                        'validation_score': validation_score,
                        'regime_transitions': len(window_results['transitions']),
                        'dominant_regime': regime_counts.index[0].value if len(regime_counts) > 0 else None
                    }
                    
                    validation_results.append(window_result)
                    
                except Exception as e:
                    logger.warning(f"Validation failed for window {window['period_name']}: {str(e)}")
                    continue
            
            # Aggregate validation results
            if not validation_results:
                return {'error': 'No valid validation windows', 'results': []}
            
            avg_stability = np.mean([r['regime_stability'] for r in validation_results])
            avg_confidence = np.mean([r['avg_confidence'] for r in validation_results])
            avg_validation_score = np.mean([r['validation_score'] for r in validation_results])
            
            # Regime consistency across periods
            regime_consistency = self._calculate_regime_consistency(validation_results)
            
            # Performance across different market conditions
            condition_performance = self._analyze_performance_by_conditions(validation_results)
            
            return {
                'validation_summary': {
                    'total_windows': len(validation_results),
                    'avg_regime_stability': avg_stability,
                    'avg_confidence': avg_confidence,
                    'avg_validation_score': avg_validation_score,
                    'regime_consistency': regime_consistency
                },
                'condition_performance': condition_performance,
                'detailed_results': validation_results,
                'validation_metrics': {
                    'stability_range': [min(r['regime_stability'] for r in validation_results),
                                      max(r['regime_stability'] for r in validation_results)],
                    'confidence_range': [min(r['avg_confidence'] for r in validation_results),
                                       max(r['avg_confidence'] for r in validation_results)],
                    'best_performing_window': max(validation_results, key=lambda x: x['validation_score'])['period'],
                    'worst_performing_window': min(validation_results, key=lambda x: x['validation_score'])['period']
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Regime detection validation failed: {str(e)}")
            return {
                'error': f'Validation failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for a price series"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _validate_regime_accuracy(self, regime_series: pd.Series, confidence_series: pd.Series,
                                volatility: float, total_return: float, max_drawdown: float) -> float:
        """Validate regime detection accuracy against actual market characteristics"""
        
        score = 0.0
        total_checks = 0
        
        # Check volatility regime consistency
        vol_regimes = [MarketRegime.HIGH_VOLATILITY, MarketRegime.VOLATILE_TRENDING, MarketRegime.CRISIS]
        high_vol_count = sum(1 for regime in regime_series if regime in vol_regimes)
        high_vol_ratio = high_vol_count / len(regime_series)
        
        if volatility > 0.3:  # High volatility period
            if high_vol_ratio > 0.5:  # More than half classified as high vol
                score += 1.0
            total_checks += 1
        elif volatility < 0.15:  # Low volatility period
            low_vol_count = sum(1 for regime in regime_series if regime == MarketRegime.LOW_VOLATILITY)
            if low_vol_count / len(regime_series) > 0.3:
                score += 1.0
            total_checks += 1
        
        # Check trend regime consistency
        trending_regimes = [MarketRegime.TRENDING, MarketRegime.VOLATILE_TRENDING]
        trending_count = sum(1 for regime in regime_series if regime in trending_regimes)
        trending_ratio = trending_count / len(regime_series)
        
        if abs(total_return) > 15:  # Strong trend period
            if trending_ratio > 0.3:
                score += 1.0
            total_checks += 1
        
        # Check crisis regime detection
        if max_drawdown > 0.2:  # Significant drawdown
            crisis_count = sum(1 for regime in regime_series if regime == MarketRegime.CRISIS)
            if crisis_count > 0:
                score += 1.0
            total_checks += 1
        
        # Check confidence appropriateness
        avg_confidence = confidence_series.mean()
        if volatility > 0.25 or abs(total_return) > 20:  # Extreme conditions
            if avg_confidence > 0.6:  # Should be confident in extreme conditions
                score += 0.5
            total_checks += 0.5
        
        return score / total_checks if total_checks > 0 else 0.5
    
    def _calculate_regime_consistency(self, validation_results: List[Dict]) -> float:
        """Calculate consistency of regime detection across validation windows"""
        if len(validation_results) < 2:
            return 1.0
        
        # Compare regime distributions across windows
        consistency_scores = []
        
        for i in range(len(validation_results) - 1):
            current_dist = validation_results[i]['regime_distribution']
            next_dist = validation_results[i + 1]['regime_distribution']
            
            # Calculate similarity between distributions
            all_regimes = set(list(current_dist.keys()) + list(next_dist.keys()))
            similarity = 0.0
            
            for regime in all_regimes:
                current_prob = current_dist.get(regime, 0)
                next_prob = next_dist.get(regime, 0)
                similarity += 1 - abs(current_prob - next_prob)
            
            consistency_scores.append(similarity / len(all_regimes))
        
        return np.mean(consistency_scores)
    
    def _analyze_performance_by_conditions(self, validation_results: List[Dict]) -> Dict[str, Any]:
        """Analyze regime detection performance across different market conditions"""
        
        high_vol_periods = [r for r in validation_results if r['market_characteristics']['volatility'] > 0.25]
        low_vol_periods = [r for r in validation_results if r['market_characteristics']['volatility'] < 0.15]
        trending_periods = [r for r in validation_results if abs(r['market_characteristics']['total_return']) > 10]
        sideways_periods = [r for r in validation_results if abs(r['market_characteristics']['total_return']) < 5]
        crisis_periods = [r for r in validation_results if r['market_characteristics']['max_drawdown'] > 0.15]
        
        def calculate_avg_score(periods):
            return np.mean([p['validation_score'] for p in periods]) if periods else 0.0
        
        return {
            'high_volatility': {
                'count': len(high_vol_periods),
                'avg_validation_score': calculate_avg_score(high_vol_periods),
                'avg_confidence': np.mean([p['avg_confidence'] for p in high_vol_periods]) if high_vol_periods else 0.0
            },
            'low_volatility': {
                'count': len(low_vol_periods),
                'avg_validation_score': calculate_avg_score(low_vol_periods),
                'avg_confidence': np.mean([p['avg_confidence'] for p in low_vol_periods]) if low_vol_periods else 0.0
            },
            'trending': {
                'count': len(trending_periods),
                'avg_validation_score': calculate_avg_score(trending_periods),
                'avg_confidence': np.mean([p['avg_confidence'] for p in trending_periods]) if trending_periods else 0.0
            },
            'sideways': {
                'count': len(sideways_periods),
                'avg_validation_score': calculate_avg_score(sideways_periods),
                'avg_confidence': np.mean([p['avg_confidence'] for p in sideways_periods]) if sideways_periods else 0.0
            },
            'crisis': {
                'count': len(crisis_periods),
                'avg_validation_score': calculate_avg_score(crisis_periods),
                'avg_confidence': np.mean([p['avg_confidence'] for p in crisis_periods]) if crisis_periods else 0.0
            }
        }