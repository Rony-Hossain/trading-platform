"""
Pattern Recognition Module
Advanced chart pattern detection and algorithmic pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.signal import argrelextrema
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PatternRecognition:
    """Advanced pattern recognition for technical analysis"""
    
    def __init__(self, lookback_window: int = 50):
        self.lookback_window = lookback_window
    
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect all chart patterns in the given data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing all detected patterns
        """
        try:
            patterns = {
                'chart_patterns': {},
                'candlestick_patterns': {},
                'trend_patterns': {},
                'reversal_patterns': {},
                'continuation_patterns': {},
                'support_resistance': {},
                'fibonacci_levels': {},
                'volume_patterns': {},
                'pattern_summary': {
                    'total_patterns': 0,
                    'bullish_patterns': 0,
                    'bearish_patterns': 0,
                    'neutral_patterns': 0
                }
            }
            
            # Chart Patterns
            patterns['chart_patterns'] = {
                'head_and_shoulders': self.detect_head_and_shoulders(df),
                'double_top': self.detect_double_top(df),
                'double_bottom': self.detect_double_bottom(df),
                'triangle_ascending': self.detect_ascending_triangle(df),
                'triangle_descending': self.detect_descending_triangle(df),
                'triangle_symmetrical': self.detect_symmetrical_triangle(df),
                'wedge_rising': self.detect_rising_wedge(df),
                'wedge_falling': self.detect_falling_wedge(df),
                'channel_ascending': self.detect_ascending_channel(df),
                'channel_descending': self.detect_descending_channel(df),
                'flag_bull': self.detect_bull_flag(df),
                'flag_bear': self.detect_bear_flag(df),
                'pennant': self.detect_pennant(df)
            }
            
            # Candlestick Patterns
            patterns['candlestick_patterns'] = {
                'doji': self.detect_doji(df),
                'hammer': self.detect_hammer(df),
                'shooting_star': self.detect_shooting_star(df),
                'engulfing_bullish': self.detect_bullish_engulfing(df),
                'engulfing_bearish': self.detect_bearish_engulfing(df),
                'morning_star': self.detect_morning_star(df),
                'evening_star': self.detect_evening_star(df),
                'three_white_soldiers': self.detect_three_white_soldiers(df),
                'three_black_crows': self.detect_three_black_crows(df),
                'spinning_top': self.detect_spinning_top(df)
            }
            
            # Support and Resistance
            patterns['support_resistance'] = self.detect_support_resistance(df)
            
            # Fibonacci Levels
            patterns['fibonacci_levels'] = self.calculate_fibonacci_levels(df)
            
            # Volume Patterns
            patterns['volume_patterns'] = self.detect_volume_patterns(df)
            
            # Trend Patterns
            patterns['trend_patterns'] = {
                'trend_direction': self.detect_trend_direction(df),
                'trend_strength': self.calculate_trend_strength(df),
                'trend_channels': self.detect_trend_channels(df),
                'breakout_signals': self.detect_breakouts(df)
            }
            
            # Calculate pattern summary
            patterns['pattern_summary'] = self._calculate_pattern_summary(patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")
            return {'error': str(e)}
    
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Head and Shoulders pattern"""
        try:
            high = df['high'].values
            low = df['low'].values
            
            if len(high) < self.lookback_window:
                return {'detected': False, 'reason': 'Insufficient data'}
            
            # Find peaks and troughs
            peaks = argrelextrema(high, np.greater, order=5)[0]
            troughs = argrelextrema(low, np.less, order=5)[0]
            
            if len(peaks) < 3:
                return {'detected': False, 'reason': 'Insufficient peaks'}
            
            # Look for H&S pattern in recent peaks
            recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
            
            if len(recent_peaks) == 3:
                left_shoulder = high[recent_peaks[0]]
                head = high[recent_peaks[1]]
                right_shoulder = high[recent_peaks[2]]
                
                # H&S criteria
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
                head_higher = head > left_shoulder and head > right_shoulder
                shoulder_similar = shoulder_diff < 0.05  # Within 5%
                
                if head_higher and shoulder_similar:
                    neckline = min(low[recent_peaks[0]:recent_peaks[1]].min(),
                                 low[recent_peaks[1]:recent_peaks[2]].min())
                    
                    return {
                        'detected': True,
                        'type': 'Head and Shoulders',
                        'signal': 'BEARISH',
                        'confidence': 0.8,
                        'left_shoulder': float(left_shoulder),
                        'head': float(head),
                        'right_shoulder': float(right_shoulder),
                        'neckline': float(neckline),
                        'target_price': float(neckline - (head - neckline)),
                        'pattern_indices': recent_peaks.tolist()
                    }
            
            return {'detected': False, 'reason': 'Pattern criteria not met'}
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return {'detected': False, 'error': str(e)}
    
    def detect_double_top(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Double Top pattern"""
        try:
            high = df['high'].values
            
            peaks = argrelextrema(high, np.greater, order=3)[0]
            
            if len(peaks) < 2:
                return {'detected': False, 'reason': 'Insufficient peaks'}
            
            # Check recent peaks
            if len(peaks) >= 2:
                peak1_idx = peaks[-2]
                peak2_idx = peaks[-1]
                peak1_val = high[peak1_idx]
                peak2_val = high[peak2_idx]
                
                # Double top criteria
                price_diff = abs(peak1_val - peak2_val) / peak1_val
                
                if price_diff < 0.03:  # Within 3%
                    valley_low = high[peak1_idx:peak2_idx].min()
                    
                    return {
                        'detected': True,
                        'type': 'Double Top',
                        'signal': 'BEARISH',
                        'confidence': 0.7,
                        'first_peak': float(peak1_val),
                        'second_peak': float(peak2_val),
                        'valley': float(valley_low),
                        'target_price': float(valley_low - (peak1_val - valley_low)),
                        'pattern_indices': [int(peak1_idx), int(peak2_idx)]
                    }
            
            return {'detected': False, 'reason': 'Pattern criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_double_bottom(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Double Bottom pattern"""
        try:
            low = df['low'].values
            
            troughs = argrelextrema(low, np.less, order=3)[0]
            
            if len(troughs) < 2:
                return {'detected': False, 'reason': 'Insufficient troughs'}
            
            if len(troughs) >= 2:
                trough1_idx = troughs[-2]
                trough2_idx = troughs[-1]
                trough1_val = low[trough1_idx]
                trough2_val = low[trough2_idx]
                
                price_diff = abs(trough1_val - trough2_val) / trough1_val
                
                if price_diff < 0.03:  # Within 3%
                    peak_high = low[trough1_idx:trough2_idx].max()
                    
                    return {
                        'detected': True,
                        'type': 'Double Bottom',
                        'signal': 'BULLISH',
                        'confidence': 0.7,
                        'first_bottom': float(trough1_val),
                        'second_bottom': float(trough2_val),
                        'peak': float(peak_high),
                        'target_price': float(peak_high + (peak_high - trough1_val)),
                        'pattern_indices': [int(trough1_idx), int(trough2_idx)]
                    }
            
            return {'detected': False, 'reason': 'Pattern criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_ascending_triangle(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Ascending Triangle pattern"""
        try:
            if len(df) < 20:
                return {'detected': False, 'reason': 'Insufficient data'}
            
            high = df['high'].values
            low = df['low'].values
            
            # Find recent highs and lows
            recent_data = df.tail(20)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Check for horizontal resistance (similar highs)
            max_high = highs.max()
            high_touches = np.where(highs > max_high * 0.98)[0]
            
            # Check for ascending support (rising lows)
            if len(lows) >= 10:
                x = np.arange(len(lows))
                slope, _, r_value, _, _ = linregress(x, lows)
                
                if len(high_touches) >= 2 and slope > 0 and r_value > 0.5:
                    return {
                        'detected': True,
                        'type': 'Ascending Triangle',
                        'signal': 'BULLISH',
                        'confidence': 0.6,
                        'resistance_level': float(max_high),
                        'support_slope': float(slope),
                        'pattern_strength': float(r_value),
                        'breakout_target': float(max_high * 1.05)
                    }
            
            return {'detected': False, 'reason': 'Pattern criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_descending_triangle(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Descending Triangle pattern"""
        try:
            if len(df) < 20:
                return {'detected': False, 'reason': 'Insufficient data'}
            
            recent_data = df.tail(20)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Check for horizontal support (similar lows)
            min_low = lows.min()
            low_touches = np.where(lows < min_low * 1.02)[0]
            
            # Check for descending resistance (falling highs)
            if len(highs) >= 10:
                x = np.arange(len(highs))
                slope, _, r_value, _, _ = linregress(x, highs)
                
                if len(low_touches) >= 2 and slope < 0 and r_value < -0.5:
                    return {
                        'detected': True,
                        'type': 'Descending Triangle',
                        'signal': 'BEARISH',
                        'confidence': 0.6,
                        'support_level': float(min_low),
                        'resistance_slope': float(slope),
                        'pattern_strength': float(abs(r_value)),
                        'breakdown_target': float(min_low * 0.95)
                    }
            
            return {'detected': False, 'reason': 'Pattern criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_symmetrical_triangle(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Symmetrical Triangle pattern"""
        try:
            if len(df) < 20:
                return {'detected': False, 'reason': 'Insufficient data'}
            
            recent_data = df.tail(20)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            if len(highs) >= 10 and len(lows) >= 10:
                x = np.arange(len(highs))
                
                # Descending highs
                high_slope, _, high_r, _, _ = linregress(x, highs)
                
                # Ascending lows
                low_slope, _, low_r, _, _ = linregress(x, lows)
                
                if (high_slope < 0 and low_slope > 0 and 
                    abs(high_r) > 0.4 and abs(low_r) > 0.4):
                    
                    # Check if lines are converging
                    convergence_point = len(highs) * 2  # Estimate
                    
                    return {
                        'detected': True,
                        'type': 'Symmetrical Triangle',
                        'signal': 'NEUTRAL',
                        'confidence': 0.5,
                        'upper_slope': float(high_slope),
                        'lower_slope': float(low_slope),
                        'convergence_estimate': int(convergence_point),
                        'breakout_direction': 'TBD'
                    }
            
            return {'detected': False, 'reason': 'Pattern criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_doji(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Doji candlestick pattern"""
        try:
            if len(df) < 1:
                return {'detected': False, 'reason': 'No data'}
            
            last_candle = df.iloc[-1]
            open_price = last_candle['open']
            close_price = last_candle['close']
            high_price = last_candle['high']
            low_price = last_candle['low']
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            # Doji criteria: small body relative to range
            if total_range > 0 and body_size / total_range < 0.1:
                return {
                    'detected': True,
                    'type': 'Doji',
                    'signal': 'NEUTRAL',
                    'confidence': 0.6,
                    'body_size_ratio': float(body_size / total_range),
                    'interpretation': 'Market indecision'
                }
            
            return {'detected': False, 'reason': 'Body too large for Doji'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_hammer(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Hammer candlestick pattern"""
        try:
            if len(df) < 1:
                return {'detected': False, 'reason': 'No data'}
            
            last_candle = df.iloc[-1]
            open_price = last_candle['open']
            close_price = last_candle['close']
            high_price = last_candle['high']
            low_price = last_candle['low']
            
            body_size = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)
            total_range = high_price - low_price
            
            # Hammer criteria
            if (total_range > 0 and lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5):
                return {
                    'detected': True,
                    'type': 'Hammer',
                    'signal': 'BULLISH',
                    'confidence': 0.7,
                    'lower_shadow_ratio': float(lower_shadow / total_range),
                    'interpretation': 'Potential reversal from downtrend'
                }
            
            return {'detected': False, 'reason': 'Hammer criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_shooting_star(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Shooting Star candlestick pattern"""
        try:
            if len(df) < 1:
                return {'detected': False, 'reason': 'No data'}
            
            last_candle = df.iloc[-1]
            open_price = last_candle['open']
            close_price = last_candle['close']
            high_price = last_candle['high']
            low_price = last_candle['low']
            
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            total_range = high_price - low_price
            
            # Shooting star criteria
            if (total_range > 0 and upper_shadow > body_size * 2 and 
                lower_shadow < body_size * 0.5):
                return {
                    'detected': True,
                    'type': 'Shooting Star',
                    'signal': 'BEARISH',
                    'confidence': 0.7,
                    'upper_shadow_ratio': float(upper_shadow / total_range),
                    'interpretation': 'Potential reversal from uptrend'
                }
            
            return {'detected': False, 'reason': 'Shooting star criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_bullish_engulfing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Bullish Engulfing pattern"""
        try:
            if len(df) < 2:
                return {'detected': False, 'reason': 'Need at least 2 candles'}
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            # Previous candle bearish
            prev_bearish = prev_candle['close'] < prev_candle['open']
            
            # Current candle bullish
            curr_bullish = curr_candle['close'] > curr_candle['open']
            
            # Current engulfs previous
            engulfs = (curr_candle['open'] < prev_candle['close'] and 
                      curr_candle['close'] > prev_candle['open'])
            
            if prev_bearish and curr_bullish and engulfs:
                return {
                    'detected': True,
                    'type': 'Bullish Engulfing',
                    'signal': 'BULLISH',
                    'confidence': 0.8,
                    'interpretation': 'Strong bullish reversal signal'
                }
            
            return {'detected': False, 'reason': 'Engulfing criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_bearish_engulfing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Bearish Engulfing pattern"""
        try:
            if len(df) < 2:
                return {'detected': False, 'reason': 'Need at least 2 candles'}
            
            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]
            
            # Previous candle bullish
            prev_bullish = prev_candle['close'] > prev_candle['open']
            
            # Current candle bearish
            curr_bearish = curr_candle['close'] < curr_candle['open']
            
            # Current engulfs previous
            engulfs = (curr_candle['open'] > prev_candle['close'] and 
                      curr_candle['close'] < prev_candle['open'])
            
            if prev_bullish and curr_bearish and engulfs:
                return {
                    'detected': True,
                    'type': 'Bearish Engulfing',
                    'signal': 'BEARISH',
                    'confidence': 0.8,
                    'interpretation': 'Strong bearish reversal signal'
                }
            
            return {'detected': False, 'reason': 'Engulfing criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_morning_star(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Morning Star pattern"""
        try:
            if len(df) < 3:
                return {'detected': False, 'reason': 'Need at least 3 candles'}
            
            candle1 = df.iloc[-3]
            candle2 = df.iloc[-2]
            candle3 = df.iloc[-1]
            
            # First candle: bearish
            bearish1 = candle1['close'] < candle1['open']
            
            # Second candle: small body (doji or spinning top)
            body2 = abs(candle2['close'] - candle2['open'])
            range2 = candle2['high'] - candle2['low']
            small_body2 = body2 / range2 < 0.3 if range2 > 0 else False
            
            # Third candle: bullish and closes above midpoint of first
            bullish3 = candle3['close'] > candle3['open']
            closes_high = candle3['close'] > (candle1['open'] + candle1['close']) / 2
            
            if bearish1 and small_body2 and bullish3 and closes_high:
                return {
                    'detected': True,
                    'type': 'Morning Star',
                    'signal': 'BULLISH',
                    'confidence': 0.8,
                    'interpretation': 'Three-candle bullish reversal'
                }
            
            return {'detected': False, 'reason': 'Morning star criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_evening_star(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Evening Star pattern"""
        try:
            if len(df) < 3:
                return {'detected': False, 'reason': 'Need at least 3 candles'}
            
            candle1 = df.iloc[-3]
            candle2 = df.iloc[-2]
            candle3 = df.iloc[-1]
            
            # First candle: bullish
            bullish1 = candle1['close'] > candle1['open']
            
            # Second candle: small body
            body2 = abs(candle2['close'] - candle2['open'])
            range2 = candle2['high'] - candle2['low']
            small_body2 = body2 / range2 < 0.3 if range2 > 0 else False
            
            # Third candle: bearish and closes below midpoint of first
            bearish3 = candle3['close'] < candle3['open']
            closes_low = candle3['close'] < (candle1['open'] + candle1['close']) / 2
            
            if bullish1 and small_body2 and bearish3 and closes_low:
                return {
                    'detected': True,
                    'type': 'Evening Star',
                    'signal': 'BEARISH',
                    'confidence': 0.8,
                    'interpretation': 'Three-candle bearish reversal'
                }
            
            return {'detected': False, 'reason': 'Evening star criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_three_white_soldiers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Three White Soldiers pattern"""
        try:
            if len(df) < 3:
                return {'detected': False, 'reason': 'Need at least 3 candles'}
            
            candles = df.iloc[-3:].copy()
            
            # All three candles must be bullish
            all_bullish = all(candles['close'] > candles['open'])
            
            # Each candle opens within previous body and closes higher
            progressive = True
            for i in range(1, 3):
                prev_open = candles.iloc[i-1]['open']
                prev_close = candles.iloc[i-1]['close']
                curr_open = candles.iloc[i]['open']
                curr_close = candles.iloc[i]['close']
                
                if not (prev_open < curr_open < prev_close and curr_close > prev_close):
                    progressive = False
                    break
            
            if all_bullish and progressive:
                return {
                    'detected': True,
                    'type': 'Three White Soldiers',
                    'signal': 'BULLISH',
                    'confidence': 0.8,
                    'interpretation': 'Strong bullish continuation'
                }
            
            return {'detected': False, 'reason': 'Three white soldiers criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_three_black_crows(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Three Black Crows pattern"""
        try:
            if len(df) < 3:
                return {'detected': False, 'reason': 'Need at least 3 candles'}
            
            candles = df.iloc[-3:].copy()
            
            # All three candles must be bearish
            all_bearish = all(candles['close'] < candles['open'])
            
            # Each candle opens within previous body and closes lower
            progressive = True
            for i in range(1, 3):
                prev_open = candles.iloc[i-1]['open']
                prev_close = candles.iloc[i-1]['close']
                curr_open = candles.iloc[i]['open']
                curr_close = candles.iloc[i]['close']
                
                if not (prev_close < curr_open < prev_open and curr_close < prev_close):
                    progressive = False
                    break
            
            if all_bearish and progressive:
                return {
                    'detected': True,
                    'type': 'Three Black Crows',
                    'signal': 'BEARISH',
                    'confidence': 0.8,
                    'interpretation': 'Strong bearish continuation'
                }
            
            return {'detected': False, 'reason': 'Three black crows criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_spinning_top(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Spinning Top pattern"""
        try:
            if len(df) < 1:
                return {'detected': False, 'reason': 'No data'}
            
            last_candle = df.iloc[-1]
            open_price = last_candle['open']
            close_price = last_candle['close']
            high_price = last_candle['high']
            low_price = last_candle['low']
            
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            total_range = high_price - low_price
            
            # Spinning top criteria
            if (total_range > 0 and body_size / total_range < 0.3 and 
                upper_shadow > body_size and lower_shadow > body_size):
                return {
                    'detected': True,
                    'type': 'Spinning Top',
                    'signal': 'NEUTRAL',
                    'confidence': 0.5,
                    'body_ratio': float(body_size / total_range),
                    'interpretation': 'Market indecision with long shadows'
                }
            
            return {'detected': False, 'reason': 'Spinning top criteria not met'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect support and resistance levels"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Find peaks and troughs
            peaks = argrelextrema(high, np.greater, order=3)[0]
            troughs = argrelextrema(low, np.less, order=3)[0]
            
            # Cluster similar levels
            resistance_levels = []
            support_levels = []
            
            if len(peaks) > 0:
                resistance_candidates = high[peaks]
                # Group similar resistance levels
                for level in resistance_candidates:
                    similar_count = np.sum(np.abs(resistance_candidates - level) < level * 0.02)
                    if similar_count >= 2:
                        resistance_levels.append(float(level))
            
            if len(troughs) > 0:
                support_candidates = low[troughs]
                # Group similar support levels
                for level in support_candidates:
                    similar_count = np.sum(np.abs(support_candidates - level) < level * 0.02)
                    if similar_count >= 2:
                        support_levels.append(float(level))
            
            current_price = float(close[-1])
            
            # Find nearest levels
            nearest_resistance = None
            nearest_support = None
            
            if resistance_levels:
                resistance_above = [r for r in resistance_levels if r > current_price]
                if resistance_above:
                    nearest_resistance = min(resistance_above)
            
            if support_levels:
                support_below = [s for s in support_levels if s < current_price]
                if support_below:
                    nearest_support = max(support_below)
            
            return {
                'resistance_levels': sorted(list(set(resistance_levels)), reverse=True)[:5],
                'support_levels': sorted(list(set(support_levels)), reverse=True)[:5],
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'current_price': current_price,
                'resistance_distance': float((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None,
                'support_distance': float((current_price - nearest_support) / current_price * 100) if nearest_support else None
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Fibonacci retracement levels"""
        try:
            high = df['high'].max()
            low = df['low'].min()
            
            diff = high - low
            
            # Standard Fibonacci levels
            levels = {
                '0%': float(high),
                '23.6%': float(high - 0.236 * diff),
                '38.2%': float(high - 0.382 * diff),
                '50%': float(high - 0.5 * diff),
                '61.8%': float(high - 0.618 * diff),
                '78.6%': float(high - 0.786 * diff),
                '100%': float(low)
            }
            
            current_price = float(df['close'].iloc[-1])
            
            # Find current level
            current_level = None
            for level_name, level_price in levels.items():
                if abs(current_price - level_price) < diff * 0.02:  # Within 2%
                    current_level = level_name
                    break
            
            return {
                'fibonacci_levels': levels,
                'range_high': float(high),
                'range_low': float(low),
                'range_size': float(diff),
                'current_price': current_price,
                'current_level': current_level
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def detect_volume_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect volume-based patterns"""
        try:
            volume = df['volume'].values
            close = df['close'].values
            
            if len(volume) < 20:
                return {'insufficient_data': True}
            
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            
            # Volume surge
            volume_surge = current_volume > avg_volume * 2
            
            # Price vs Volume correlation
            price_change = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            volume_change = (volume[-1] - volume[-2]) / volume[-2] if len(volume) > 1 else 0
            
            # On Balance Volume trend
            obv = np.zeros(len(close))
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            
            obv_trend = 'BULLISH' if obv[-1] > obv[-10] else 'BEARISH' if obv[-1] < obv[-10] else 'NEUTRAL'
            
            return {
                'current_volume': int(current_volume),
                'average_volume_20': float(avg_volume),
                'volume_ratio': float(current_volume / avg_volume),
                'volume_surge': volume_surge,
                'price_volume_correlation': {
                    'price_change': float(price_change * 100),
                    'volume_change': float(volume_change * 100),
                    'correlation': 'POSITIVE' if price_change * volume_change > 0 else 'NEGATIVE'
                },
                'obv_trend': obv_trend,
                'obv_current': float(obv[-1])
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def detect_trend_direction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect overall trend direction"""
        try:
            close = df['close'].values
            
            if len(close) < 20:
                return {'trend': 'UNKNOWN', 'reason': 'Insufficient data'}
            
            # Multiple timeframe analysis
            short_ma = np.mean(close[-5:])
            medium_ma = np.mean(close[-10:])
            long_ma = np.mean(close[-20:])
            
            # Linear regression trend
            x = np.arange(len(close))
            slope, _, r_value, _, _ = linregress(x, close)
            
            # Higher highs and higher lows check
            recent_highs = [close[i] for i in argrelextrema(close, np.greater, order=2)[0][-3:]]
            recent_lows = [close[i] for i in argrelextrema(close, np.less, order=2)[0][-3:]]
            
            higher_highs = len(recent_highs) >= 2 and all(recent_highs[i] >= recent_highs[i-1] for i in range(1, len(recent_highs)))
            higher_lows = len(recent_lows) >= 2 and all(recent_lows[i] >= recent_lows[i-1] for i in range(1, len(recent_lows)))
            lower_highs = len(recent_highs) >= 2 and all(recent_highs[i] <= recent_highs[i-1] for i in range(1, len(recent_highs)))
            lower_lows = len(recent_lows) >= 2 and all(recent_lows[i] <= recent_lows[i-1] for i in range(1, len(recent_lows)))
            
            # Determine trend
            if short_ma > medium_ma > long_ma and slope > 0 and higher_highs and higher_lows:
                trend = 'STRONG_UPTREND'
                strength = min(100, abs(slope) * 1000)
            elif short_ma > medium_ma and slope > 0:
                trend = 'UPTREND'
                strength = min(80, abs(slope) * 1000)
            elif short_ma < medium_ma < long_ma and slope < 0 and lower_highs and lower_lows:
                trend = 'STRONG_DOWNTREND'
                strength = min(100, abs(slope) * 1000)
            elif short_ma < medium_ma and slope < 0:
                trend = 'DOWNTREND'
                strength = min(80, abs(slope) * 1000)
            else:
                trend = 'SIDEWAYS'
                strength = 0
            
            return {
                'trend': trend,
                'strength': float(strength),
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'short_ma': float(short_ma),
                'medium_ma': float(medium_ma),
                'long_ma': float(long_ma),
                'higher_highs': higher_highs,
                'higher_lows': higher_lows
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend strength using ADX-like calculation"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            if len(close) < 14:
                return {'trend_strength': 'UNKNOWN', 'reason': 'Insufficient data'}
            
            # True Range calculation
            tr = np.maximum(high[1:] - low[1:], 
                           np.maximum(np.abs(high[1:] - close[:-1]), 
                                    np.abs(low[1:] - close[:-1])))
            
            # Directional movement
            dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                              np.maximum(high[1:] - high[:-1], 0), 0)
            dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                               np.maximum(low[:-1] - low[1:], 0), 0)
            
            # Smoothed averages
            period = 14
            atr = np.convolve(tr, np.ones(period)/period, mode='valid')
            di_plus = 100 * np.convolve(dm_plus, np.ones(period)/period, mode='valid') / atr
            di_minus = 100 * np.convolve(dm_minus, np.ones(period)/period, mode='valid') / atr
            
            # ADX calculation
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
            adx = np.mean(dx[-period:]) if len(dx) >= period else np.mean(dx)
            
            # Trend strength interpretation
            if adx > 50:
                strength = 'VERY_STRONG'
            elif adx > 25:
                strength = 'STRONG'
            elif adx > 15:
                strength = 'MODERATE'
            else:
                strength = 'WEAK'
            
            return {
                'adx_value': float(adx),
                'trend_strength': strength,
                'di_plus': float(di_plus[-1]) if len(di_plus) > 0 else 0,
                'di_minus': float(di_minus[-1]) if len(di_minus) > 0 else 0,
                'directional_bias': 'BULLISH' if di_plus[-1] > di_minus[-1] else 'BEARISH' if len(di_plus) > 0 else 'NEUTRAL'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def detect_trend_channels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect trend channels"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            if len(close) < 20:
                return {'detected': False, 'reason': 'Insufficient data'}
            
            # Use recent data for channel detection
            recent_data = 20
            recent_highs = high[-recent_data:]
            recent_lows = low[-recent_data:]
            x = np.arange(recent_data)
            
            # Linear regression on highs and lows
            high_slope, high_intercept, high_r, _, _ = linregress(x, recent_highs)
            low_slope, low_intercept, low_r, _, _ = linregress(x, recent_lows)
            
            # Check if slopes are similar (parallel channel)
            slope_diff = abs(high_slope - low_slope)
            
            if slope_diff < abs(high_slope) * 0.5 and abs(high_r) > 0.5 and abs(low_r) > 0.5:
                # Calculate current channel levels
                current_upper = high_intercept + high_slope * (recent_data - 1)
                current_lower = low_intercept + low_slope * (recent_data - 1)
                channel_width = current_upper - current_lower
                
                current_price = close[-1]
                position_in_channel = (current_price - current_lower) / channel_width
                
                return {
                    'detected': True,
                    'type': 'ASCENDING' if high_slope > 0 and low_slope > 0 else 'DESCENDING' if high_slope < 0 and low_slope < 0 else 'HORIZONTAL',
                    'upper_channel': float(current_upper),
                    'lower_channel': float(current_lower),
                    'channel_width': float(channel_width),
                    'position_in_channel': float(position_in_channel),
                    'upper_slope': float(high_slope),
                    'lower_slope': float(low_slope),
                    'channel_strength': float((abs(high_r) + abs(low_r)) / 2)
                }
            
            return {'detected': False, 'reason': 'No clear channel pattern'}
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def detect_breakouts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect price breakouts"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            if len(close) < 20:
                return {'breakout_detected': False, 'reason': 'Insufficient data'}
            
            current_price = close[-1]
            current_volume = volume[-1]
            avg_volume = np.mean(volume[-20:])
            
            # Resistance/Support levels
            resistance_level = np.max(high[-20:-1])  # Exclude current high
            support_level = np.min(low[-20:-1])     # Exclude current low
            
            # Breakout detection
            resistance_breakout = current_price > resistance_level and current_volume > avg_volume * 1.5
            support_breakout = current_price < support_level and current_volume > avg_volume * 1.5
            
            # Volatility expansion
            recent_volatility = np.std(close[-5:])
            avg_volatility = np.std(close[-20:])
            volatility_expansion = recent_volatility > avg_volatility * 1.5
            
            breakout_type = None
            if resistance_breakout:
                breakout_type = 'UPSIDE_BREAKOUT'
            elif support_breakout:
                breakout_type = 'DOWNSIDE_BREAKOUT'
            
            return {
                'breakout_detected': breakout_type is not None,
                'breakout_type': breakout_type,
                'resistance_level': float(resistance_level),
                'support_level': float(support_level),
                'current_price': float(current_price),
                'volume_confirmation': current_volume > avg_volume * 1.5,
                'volume_ratio': float(current_volume / avg_volume),
                'volatility_expansion': volatility_expansion,
                'breakout_strength': float((current_volume / avg_volume) * (recent_volatility / avg_volatility)) if breakout_type else 0
            }
            
        except Exception as e:
            return {'breakout_detected': False, 'error': str(e)}
    
    def _calculate_pattern_summary(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for all detected patterns"""
        try:
            total_patterns = 0
            bullish_patterns = 0
            bearish_patterns = 0
            neutral_patterns = 0
            
            # Count chart patterns
            for pattern_name, pattern_data in patterns.get('chart_patterns', {}).items():
                if isinstance(pattern_data, dict) and pattern_data.get('detected', False):
                    total_patterns += 1
                    signal = pattern_data.get('signal', 'NEUTRAL')
                    if signal == 'BULLISH':
                        bullish_patterns += 1
                    elif signal == 'BEARISH':
                        bearish_patterns += 1
                    else:
                        neutral_patterns += 1
            
            # Count candlestick patterns
            for pattern_name, pattern_data in patterns.get('candlestick_patterns', {}).items():
                if isinstance(pattern_data, dict) and pattern_data.get('detected', False):
                    total_patterns += 1
                    signal = pattern_data.get('signal', 'NEUTRAL')
                    if signal == 'BULLISH':
                        bullish_patterns += 1
                    elif signal == 'BEARISH':
                        bearish_patterns += 1
                    else:
                        neutral_patterns += 1
            
            # Overall sentiment
            if bullish_patterns > bearish_patterns:
                overall_sentiment = 'BULLISH'
            elif bearish_patterns > bullish_patterns:
                overall_sentiment = 'BEARISH'
            else:
                overall_sentiment = 'NEUTRAL'
            
            return {
                'total_patterns': total_patterns,
                'bullish_patterns': bullish_patterns,
                'bearish_patterns': bearish_patterns,
                'neutral_patterns': neutral_patterns,
                'overall_sentiment': overall_sentiment,
                'bullish_percentage': float(bullish_patterns / total_patterns * 100) if total_patterns > 0 else 0,
                'bearish_percentage': float(bearish_patterns / total_patterns * 100) if total_patterns > 0 else 0
            }
            
        except Exception as e:
            return {
                'total_patterns': 0,
                'error': str(e)
            }
    
    # Additional helper methods for specific patterns
    def detect_rising_wedge(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Rising Wedge pattern - placeholder implementation"""
        return {'detected': False, 'reason': 'Pattern detection not implemented yet'}
    
    def detect_falling_wedge(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Falling Wedge pattern - placeholder implementation"""
        return {'detected': False, 'reason': 'Pattern detection not implemented yet'}
    
    def detect_ascending_channel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Ascending Channel pattern - placeholder implementation"""
        return {'detected': False, 'reason': 'Pattern detection not implemented yet'}
    
    def detect_descending_channel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Descending Channel pattern - placeholder implementation"""
        return {'detected': False, 'reason': 'Pattern detection not implemented yet'}
    
    def detect_bull_flag(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Bull Flag pattern - placeholder implementation"""
        return {'detected': False, 'reason': 'Pattern detection not implemented yet'}
    
    def detect_bear_flag(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Bear Flag pattern - placeholder implementation"""
        return {'detected': False, 'reason': 'Pattern detection not implemented yet'}
    
    def detect_pennant(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Pennant pattern - placeholder implementation"""
        return {'detected': False, 'reason': 'Pattern detection not implemented yet'}