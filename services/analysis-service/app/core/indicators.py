"""
Technical Indicators Module
Implements common technical analysis indicators for stock market analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging

# Import advanced modules
from .pattern_recognition import PatternRecognition
from .advanced_indicators import AdvancedIndicators, CompositeIndicators

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical analysis indicators implementation"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """Exponential Moving Average"""
        if alpha is None:
            alpha = 2 / (window + 1)
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'middle': sma,
            'upper': upper_band,
            'lower': lower_band,
            'width': upper_band - lower_band,
            'percent_b': (data - lower_band) / (upper_band - lower_band)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def momentum(data: pd.Series, window: int = 10) -> pd.Series:
        """Price Momentum"""
        return data - data.shift(window)
    
    @staticmethod
    def rate_of_change(data: pd.Series, window: int = 10) -> pd.Series:
        """Rate of Change"""
        return ((data - data.shift(window)) / data.shift(window)) * 100

class TechnicalAnalysis:
    """High-level technical analysis class combining multiple indicators"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.pattern_recognition = PatternRecognition()
        self.advanced_indicators = AdvancedIndicators()
        self.composite_indicators = CompositeIndicators()
    
    def analyze(self, df: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """
        Comprehensive technical analysis of price data
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            symbol: Stock symbol for reference
            
        Returns:
            Dictionary containing all technical indicators and signals
        """
        try:
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Calculate all indicators
            analysis_result = {
                'symbol': symbol,
                'data_points': len(df),
                'date_range': {
                    'start': df.index.min().isoformat() if hasattr(df.index.min(), 'isoformat') else str(df.index.min()),
                    'end': df.index.max().isoformat() if hasattr(df.index.max(), 'isoformat') else str(df.index.max())
                },
                'current_price': float(close.iloc[-1]),
                
                # Moving Averages
                'moving_averages': {
                    'sma_20': float(self.indicators.sma(close, 20).iloc[-1]) if len(close) >= 20 else None,
                    'sma_50': float(self.indicators.sma(close, 50).iloc[-1]) if len(close) >= 50 else None,
                    'sma_200': float(self.indicators.sma(close, 200).iloc[-1]) if len(close) >= 200 else None,
                    'ema_12': float(self.indicators.ema(close, 12).iloc[-1]) if len(close) >= 12 else None,
                    'ema_26': float(self.indicators.ema(close, 26).iloc[-1]) if len(close) >= 26 else None,
                },
                
                # Oscillators
                'oscillators': {
                    'rsi_14': float(self.indicators.rsi(close, 14).iloc[-1]) if len(close) >= 14 else None,
                },
                
                # MACD
                'macd': {},
                
                # Bollinger Bands
                'bollinger_bands': {},
                
                # Volume indicators
                'volume_analysis': {
                    'current_volume': int(volume.iloc[-1]),
                    'avg_volume_20': float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else None,
                    'volume_ratio': float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else None,
                },
                
                # Price statistics
                'price_stats': {
                    'volatility_20d': float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)) if len(close) >= 20 else None,
                    'high_52w': float(high.rolling(252).max().iloc[-1]) if len(high) >= 252 else float(high.max()),
                    'low_52w': float(low.rolling(252).min().iloc[-1]) if len(low) >= 252 else float(low.min()),
                }
            }
            
            # Calculate MACD if enough data
            if len(close) >= 26:
                macd_data = self.indicators.macd(close)
                analysis_result['macd'] = {
                    'macd': float(macd_data['macd'].iloc[-1]),
                    'signal': float(macd_data['signal'].iloc[-1]) if len(macd_data['signal'].dropna()) > 0 else None,
                    'histogram': float(macd_data['histogram'].iloc[-1]) if len(macd_data['histogram'].dropna()) > 0 else None,
                }
            
            # Calculate Bollinger Bands if enough data
            if len(close) >= 20:
                bb_data = self.indicators.bollinger_bands(close)
                analysis_result['bollinger_bands'] = {
                    'upper': float(bb_data['upper'].iloc[-1]),
                    'middle': float(bb_data['middle'].iloc[-1]),
                    'lower': float(bb_data['lower'].iloc[-1]),
                    'width': float(bb_data['width'].iloc[-1]),
                    'percent_b': float(bb_data['percent_b'].iloc[-1]),
                }
            
            # Generate trading signals
            analysis_result['signals'] = self._generate_signals(analysis_result)
            
            # Add pattern recognition
            try:
                patterns = self.pattern_recognition.detect_all_patterns(df)
                analysis_result['patterns'] = patterns
            except Exception as e:
                logger.error(f"Error in pattern recognition: {e}")
                analysis_result['patterns'] = {'error': str(e)}
            
            # Add advanced indicators
            try:
                # Ichimoku Cloud
                if len(close) >= 52:
                    ichimoku = self.advanced_indicators.ichimoku_cloud(high, low, close)
                    analysis_result['ichimoku'] = {
                        'tenkan_sen': float(ichimoku['tenkan_sen'].iloc[-1]) if not ichimoku['tenkan_sen'].isna().all() else None,
                        'kijun_sen': float(ichimoku['kijun_sen'].iloc[-1]) if not ichimoku['kijun_sen'].isna().all() else None,
                        'senkou_span_a': float(ichimoku['senkou_span_a'].iloc[-1]) if not ichimoku['senkou_span_a'].isna().all() else None,
                        'senkou_span_b': float(ichimoku['senkou_span_b'].iloc[-1]) if not ichimoku['senkou_span_b'].isna().all() else None
                    }
                
                # ADX
                adx_data = self.advanced_indicators.adx(high, low, close)
                analysis_result['adx'] = {
                    'adx': float(adx_data['adx'].iloc[-1]) if not adx_data['adx'].isna().all() else None,
                    'di_plus': float(adx_data['di_plus'].iloc[-1]) if not adx_data['di_plus'].isna().all() else None,
                    'di_minus': float(adx_data['di_minus'].iloc[-1]) if not adx_data['di_minus'].isna().all() else None
                }
                
                # Parabolic SAR
                if len(high) >= 10:
                    sar = self.advanced_indicators.parabolic_sar(high, low)
                    analysis_result['parabolic_sar'] = {
                        'value': float(sar.iloc[-1]) if not sar.isna().all() else None,
                        'signal': 'BUY' if sar.iloc[-1] < close.iloc[-1] else 'SELL' if not sar.isna().all() else 'NEUTRAL'
                    }
                
                # Money Flow Index
                if 'volume' in df.columns:
                    mfi = self.advanced_indicators.money_flow_index(high, low, close, volume)
                    analysis_result['money_flow_index'] = {
                        'value': float(mfi.iloc[-1]) if not mfi.isna().all() else None,
                        'signal': 'SELL' if mfi.iloc[-1] > 80 else 'BUY' if mfi.iloc[-1] < 20 else 'NEUTRAL' if not mfi.isna().all() else 'NEUTRAL'
                    }
                
                # Commodity Channel Index
                cci = self.advanced_indicators.commodity_channel_index(high, low, close)
                analysis_result['cci'] = {
                    'value': float(cci.iloc[-1]) if not cci.isna().all() else None,
                    'signal': 'SELL' if cci.iloc[-1] > 100 else 'BUY' if cci.iloc[-1] < -100 else 'NEUTRAL' if not cci.isna().all() else 'NEUTRAL'
                }
                
            except Exception as e:
                logger.error(f"Error calculating advanced indicators: {e}")
                analysis_result['advanced_indicators_error'] = str(e)
            
            # Add composite indicators
            try:
                # Market strength composite
                market_strength = self.composite_indicators.market_strength_composite(df)
                analysis_result['market_strength'] = market_strength
                
                # Volatility regime
                volatility_regime = self.composite_indicators.volatility_regime_detector(df)
                analysis_result['volatility_regime'] = volatility_regime
                
                # Momentum divergence
                divergence = self.composite_indicators.momentum_divergence_detector(df)
                analysis_result['momentum_divergence'] = divergence
                
            except Exception as e:
                logger.error(f"Error calculating composite indicators: {e}")
                analysis_result['composite_indicators_error'] = str(e)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'data_points': len(df) if df is not None else 0
            }
    
    def _generate_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on technical indicators"""
        signals = {
            'overall_signal': 'NEUTRAL',
            'strength': 0,  # -100 to 100
            'individual_signals': {}
        }
        
        score = 0
        signal_count = 0
        
        try:
            # RSI signals
            rsi = analysis.get('oscillators', {}).get('rsi_14')
            if rsi is not None:
                if rsi > 70:
                    signals['individual_signals']['rsi'] = 'SELL'
                    score -= 2
                elif rsi < 30:
                    signals['individual_signals']['rsi'] = 'BUY'
                    score += 2
                else:
                    signals['individual_signals']['rsi'] = 'NEUTRAL'
                signal_count += 1
            
            # Moving Average signals
            current_price = analysis.get('current_price')
            sma_20 = analysis.get('moving_averages', {}).get('sma_20')
            sma_50 = analysis.get('moving_averages', {}).get('sma_50')
            
            if current_price and sma_20:
                if current_price > sma_20:
                    signals['individual_signals']['sma_20'] = 'BUY'
                    score += 1
                else:
                    signals['individual_signals']['sma_20'] = 'SELL'
                    score -= 1
                signal_count += 1
            
            if sma_20 and sma_50:
                if sma_20 > sma_50:
                    signals['individual_signals']['ma_cross'] = 'BUY'
                    score += 1
                else:
                    signals['individual_signals']['ma_cross'] = 'SELL'
                    score -= 1
                signal_count += 1
            
            # MACD signals
            macd_data = analysis.get('macd', {})
            macd_line = macd_data.get('macd')
            signal_line = macd_data.get('signal')
            
            if macd_line is not None and signal_line is not None:
                if macd_line > signal_line:
                    signals['individual_signals']['macd'] = 'BUY'
                    score += 1
                else:
                    signals['individual_signals']['macd'] = 'SELL'
                    score -= 1
                signal_count += 1
            
            # Bollinger Bands signals
            bb_data = analysis.get('bollinger_bands', {})
            percent_b = bb_data.get('percent_b')
            
            if percent_b is not None:
                if percent_b > 1:
                    signals['individual_signals']['bollinger'] = 'SELL'
                    score -= 1
                elif percent_b < 0:
                    signals['individual_signals']['bollinger'] = 'BUY'
                    score += 1
                else:
                    signals['individual_signals']['bollinger'] = 'NEUTRAL'
                signal_count += 1
            
            # Calculate overall signal
            if signal_count > 0:
                strength = (score / signal_count) * 20  # Scale to -100 to 100
                signals['strength'] = min(100, max(-100, strength))
                
                if strength > 30:
                    signals['overall_signal'] = 'STRONG_BUY'
                elif strength > 10:
                    signals['overall_signal'] = 'BUY'
                elif strength < -30:
                    signals['overall_signal'] = 'STRONG_SELL'
                elif strength < -10:
                    signals['overall_signal'] = 'SELL'
                else:
                    signals['overall_signal'] = 'NEUTRAL'
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            signals['error'] = str(e)
        
        return signals