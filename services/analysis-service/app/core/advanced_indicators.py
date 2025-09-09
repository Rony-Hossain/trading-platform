"""
Advanced Technical Indicators Module
Comprehensive collection of advanced technical analysis indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AdvancedIndicators:
    """Advanced technical indicators for comprehensive market analysis"""
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series,
                      tenkan_period: int = 9, kijun_period: int = 26, 
                      senkou_span_b_period: int = 52) -> Dict[str, pd.Series]:
        """
        Ichimoku Cloud indicator system
        """
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A) - shifted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B) - shifted 26 periods ahead
        senkou_span_b = ((high.rolling(senkou_span_b_period).max() + 
                         low.rolling(senkou_span_b_period).min()) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span) - shifted 26 periods behind
        chikou_span = close.shift(-kijun_period)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, 
                      initial_af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """
        Parabolic Stop and Reverse (SAR)
        """
        length = len(high)
        sar = np.zeros(length)
        trend = np.ones(length, dtype=int)  # 1 for uptrend, -1 for downtrend
        af = np.full(length, initial_af)
        ep = np.zeros(length)  # Extreme Point
        
        # Initialize
        sar[0] = low.iloc[0]
        trend[0] = 1
        ep[0] = high.iloc[0]
        
        for i in range(1, length):
            # Previous values
            prev_sar = sar[i-1]
            prev_trend = trend[i-1]
            prev_af = af[i-1]
            prev_ep = ep[i-1]
            
            if prev_trend == 1:  # Uptrend
                # Calculate SAR
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # Ensure SAR doesn't go above low of current or previous period
                sar[i] = min(sar[i], low.iloc[i], low.iloc[i-1] if i > 0 else low.iloc[i])
                
                # Check for trend reversal
                if low.iloc[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = prev_ep
                    af[i] = initial_af
                    ep[i] = low.iloc[i]
                else:
                    trend[i] = 1
                    if high.iloc[i] > prev_ep:
                        ep[i] = high.iloc[i]
                        af[i] = min(prev_af + initial_af, max_af)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
            else:  # Downtrend
                # Calculate SAR
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # Ensure SAR doesn't go below high of current or previous period
                sar[i] = max(sar[i], high.iloc[i], high.iloc[i-1] if i > 0 else high.iloc[i])
                
                # Check for trend reversal
                if high.iloc[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = prev_ep
                    af[i] = initial_af
                    ep[i] = high.iloc[i]
                else:
                    trend[i] = -1
                    if low.iloc[i] < prev_ep:
                        ep[i] = low.iloc[i]
                        af[i] = min(prev_af + initial_af, max_af)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
        
        return pd.Series(sar, index=high.index)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        Average Directional Index (ADX) with +DI and -DI
        """
        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = np.where((high.diff() > low.diff().abs()), np.maximum(high.diff(), 0), 0)
        dm_minus = np.where((low.diff().abs() > high.diff()), np.maximum(low.diff().abs(), 0), 0)
        
        # Smoothed values using Wilder's smoothing
        atr = true_range.rolling(period).mean()
        di_plus = 100 * (pd.Series(dm_plus).rolling(period).mean() / atr)
        di_minus = 100 * (pd.Series(dm_minus).rolling(period).mean() / atr)
        
        # ADX calculation
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()
        
        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus,
            'atr': atr
        }
    
    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI)
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, 
                        volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI)
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
    
    @staticmethod
    def kaufman_adaptive_ma(close: pd.Series, period: int = 10, fast_sc: float = 2, slow_sc: float = 30) -> pd.Series:
        """
        Kaufman Adaptive Moving Average (KAMA)
        """
        change = np.abs(close - close.shift(period))
        volatility = np.abs(close - close.shift(1)).rolling(period).sum()
        
        efficiency_ratio = change / volatility
        
        fast_alpha = 2 / (fast_sc + 1)
        slow_alpha = 2 / (slow_sc + 1)
        
        smoothing_constant = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[period-1] = close.iloc[:period].mean()
        
        for i in range(period, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + smoothing_constant.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    @staticmethod
    def awesome_oscillator(high: pd.Series, low: pd.Series, 
                          fast_period: int = 5, slow_period: int = 34) -> pd.Series:
        """
        Awesome Oscillator (AO)
        """
        median_price = (high + low) / 2
        ao = median_price.rolling(fast_period).mean() - median_price.rolling(slow_period).mean()
        return ao
    
    @staticmethod
    def klinger_volume_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                                 volume: pd.Series, fast: int = 34, slow: int = 55, signal: int = 13) -> Dict[str, pd.Series]:
        """
        Klinger Volume Oscillator
        """
        hlc = (high + low + close) / 3
        dm = hlc.diff()
        
        # Trend calculation
        trend = np.where(dm > 0, 1, -1)
        
        # Volume Force
        vf = volume * trend * np.abs(2 * ((close - low) - (high - close)) / (high - low)) * 100
        
        # EMAs of volume force
        kvo = vf.ewm(span=fast).mean() - vf.ewm(span=slow).mean()
        kvo_signal = kvo.ewm(span=signal).mean()
        
        return {
            'kvo': kvo,
            'signal': kvo_signal
        }
    
    @staticmethod
    def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                           period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """
        Ultimate Oscillator
        """
        prior_close = close.shift(1)
        
        # True Low and Buying Pressure
        true_low = pd.concat([low, prior_close], axis=1).min(axis=1)
        buying_pressure = close - true_low
        
        # True Range
        true_range = pd.concat([high - low, 
                              np.abs(high - prior_close), 
                              np.abs(low - prior_close)], axis=1).max(axis=1)
        
        # Calculate averages for each period
        avg_bp1 = buying_pressure.rolling(period1).sum()
        avg_tr1 = true_range.rolling(period1).sum()
        
        avg_bp2 = buying_pressure.rolling(period2).sum()
        avg_tr2 = true_range.rolling(period2).sum()
        
        avg_bp3 = buying_pressure.rolling(period3).sum()
        avg_tr3 = true_range.rolling(period3).sum()
        
        # Ultimate Oscillator calculation
        uo = 100 * ((4 * (avg_bp1 / avg_tr1)) + (2 * (avg_bp2 / avg_tr2)) + (avg_bp3 / avg_tr3)) / 7
        
        return uo
    
    @staticmethod
    def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, 
                          volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow (CMF)
        """
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)  # Fill NaN values when high == low
        
        mfv = mfm * volume
        cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
        
        return cmf
    
    @staticmethod
    def ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Ease of Movement (EOM)
        """
        distance_moved = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_height = (volume / 100000000) / (high - low)
        
        one_period_eom = distance_moved / box_height
        eom = one_period_eom.rolling(period).mean()
        
        return eom
    
    @staticmethod
    def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
        """
        Force Index
        """
        fi = volume * (close - close.shift(1))
        return fi.rolling(period).mean()
    
    @staticmethod
    def negative_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Negative Volume Index (NVI)
        """
        nvi = pd.Series(index=close.index, dtype=float)
        nvi.iloc[0] = 1000
        
        for i in range(1, len(close)):
            if volume.iloc[i] < volume.iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return nvi
    
    @staticmethod
    def positive_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Positive Volume Index (PVI)
        """
        pvi = pd.Series(index=close.index, dtype=float)
        pvi.iloc[0] = 1000
        
        for i in range(1, len(close)):
            if volume.iloc[i] > volume.iloc[i-1]:
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        return pvi
    
    @staticmethod
    def accumulation_distribution_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Accumulation/Distribution Line
        """
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        
        mfv = mfm * volume
        ad_line = mfv.cumsum()
        
        return ad_line
    
    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        Aroon Indicator
        """
        aroon_up = high.rolling(period + 1).apply(
            lambda x: (period - np.argmax(x)) / period * 100, raw=True
        )
        
        aroon_down = low.rolling(period + 1).apply(
            lambda x: (period - np.argmin(x)) / period * 100, raw=True
        )
        
        aroon_oscillator = aroon_up - aroon_down
        
        return {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator
        }
    
    @staticmethod
    def balance_of_power(open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        Balance of Power (BOP)
        """
        bop = (close - open_price) / (high - low)
        return bop.fillna(0)
    
    @staticmethod
    def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13) -> Dict[str, pd.Series]:
        """
        Elder Ray Index (Bull Power and Bear Power)
        """
        ema = close.ewm(span=period).mean()
        bull_power = high - ema
        bear_power = low - ema
        
        return {
            'bull_power': bull_power,
            'bear_power': bear_power,
            'ema': ema
        }
    
    @staticmethod
    def detrended_price_oscillator(close: pd.Series, period: int = 20) -> pd.Series:
        """
        Detrended Price Oscillator (DPO)
        """
        sma = close.rolling(period).mean()
        shift_period = period // 2 + 1
        dpo = close - sma.shift(shift_period)
        return dpo
    
    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, 
                        period: int = 20, multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Keltner Channels
        """
        typical_price = (high + low + close) / 3
        middle_line = typical_price.ewm(span=period).mean()
        
        # True Range for ATR
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period).mean()
        
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        return {
            'upper': upper_channel,
            'middle': middle_line,
            'lower': lower_channel,
            'width': upper_channel - lower_channel
        }
    
    @staticmethod
    def mass_index(high: pd.Series, low: pd.Series, period: int = 9, sum_period: int = 25) -> pd.Series:
        """
        Mass Index
        """
        high_low_range = high - low
        ema1 = high_low_range.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        
        mass_ratio = ema1 / ema2
        mass_index = mass_ratio.rolling(sum_period).sum()
        
        return mass_index
    
    @staticmethod
    def price_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """
        Price Channels (Donchian Channels)
        """
        upper_channel = high.rolling(period).max()
        lower_channel = low.rolling(period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'upper': upper_channel,
            'middle': middle_channel,
            'lower': lower_channel,
            'width': upper_channel - lower_channel
        }
    
    @staticmethod
    def trix(close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        TRIX Oscillator
        """
        # Triple smoothed exponential moving average
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        # TRIX as rate of change of triple EMA
        trix = ema3.pct_change() * 10000
        trix_signal = trix.ewm(span=9).mean()
        
        return {
            'trix': trix,
            'signal': trix_signal,
            'histogram': trix - trix_signal
        }
    
    @staticmethod
    def volatility_system(high: pd.Series, low: pd.Series, close: pd.Series, 
                         period: int = 20, multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Volatility-based trading system
        """
        # Average True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        # Volatility channels
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Trend following signals
        trend = pd.Series(index=close.index, dtype=int)
        stop_loss = pd.Series(index=close.index, dtype=float)
        
        for i in range(1, len(close)):
            if close.iloc[i] > upper_band.iloc[i]:
                trend.iloc[i] = 1  # Bullish
                stop_loss.iloc[i] = lower_band.iloc[i]
            elif close.iloc[i] < lower_band.iloc[i]:
                trend.iloc[i] = -1  # Bearish
                stop_loss.iloc[i] = upper_band.iloc[i]
            else:
                trend.iloc[i] = trend.iloc[i-1] if not pd.isna(trend.iloc[i-1]) else 0
                stop_loss.iloc[i] = stop_loss.iloc[i-1] if not pd.isna(stop_loss.iloc[i-1]) else close.iloc[i]
        
        return {
            'upper_band': upper_band,
            'lower_band': lower_band,
            'atr': atr,
            'trend': trend,
            'stop_loss': stop_loss
        }

class CompositeIndicators:
    """Composite indicators combining multiple technical analysis methods"""
    
    @staticmethod
    def market_strength_composite(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Composite market strength indicator
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            indicators = AdvancedIndicators()
            
            # Calculate various strength indicators
            rsi = ((close.diff().where(close.diff() > 0, 0).rolling(14).mean()) /
                  (close.diff().abs().rolling(14).mean())) * 100
            
            adx_data = indicators.adx(high, low, close)
            mfi = indicators.money_flow_index(high, low, close, volume)
            ao = indicators.awesome_oscillator(high, low)
            
            # Composite score (normalize and combine)
            scores = []
            
            if not rsi.isna().all():
                rsi_score = (rsi.iloc[-1] - 50) / 50  # -1 to 1
                scores.append(rsi_score)
            
            if not adx_data['adx'].isna().all():
                adx_val = adx_data['adx'].iloc[-1]
                di_diff = adx_data['di_plus'].iloc[-1] - adx_data['di_minus'].iloc[-1]
                adx_score = (di_diff / 100) * min(adx_val / 25, 1)  # Weighted by trend strength
                scores.append(adx_score)
            
            if not mfi.isna().all():
                mfi_score = (mfi.iloc[-1] - 50) / 50
                scores.append(mfi_score)
            
            if not ao.isna().all():
                ao_score = np.tanh(ao.iloc[-1] / ao.std())  # Normalize using tanh
                scores.append(ao_score)
            
            composite_score = np.mean(scores) if scores else 0
            
            # Interpret score
            if composite_score > 0.6:
                strength = "VERY_STRONG_BULLISH"
            elif composite_score > 0.3:
                strength = "STRONG_BULLISH"
            elif composite_score > 0.1:
                strength = "BULLISH"
            elif composite_score < -0.6:
                strength = "VERY_STRONG_BEARISH"
            elif composite_score < -0.3:
                strength = "STRONG_BEARISH"
            elif composite_score < -0.1:
                strength = "BEARISH"
            else:
                strength = "NEUTRAL"
            
            return {
                'composite_score': float(composite_score),
                'strength': strength,
                'individual_scores': {
                    'rsi_score': float(scores[0]) if len(scores) > 0 else None,
                    'adx_score': float(scores[1]) if len(scores) > 1 else None,
                    'mfi_score': float(scores[2]) if len(scores) > 2 else None,
                    'ao_score': float(scores[3]) if len(scores) > 3 else None
                },
                'confidence': min(100, abs(composite_score) * 100)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def volatility_regime_detector(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect current volatility regime
        """
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Various volatility measures
            returns = close.pct_change().dropna()
            
            # Historical volatility
            hist_vol = returns.rolling(20).std() * np.sqrt(252)
            
            # Parkinson volatility
            park_vol = np.sqrt((np.log(high / low) ** 2).rolling(20).mean() * 252 / (4 * np.log(2)))
            
            # Garman-Klass volatility
            gk_vol = np.sqrt(((np.log(high / low) ** 2) - 
                            (2 * np.log(2) - 1) * (np.log(close / low) ** 2)).rolling(20).mean() * 252)
            
            # Current volatility measures
            current_hist_vol = hist_vol.iloc[-1] if not hist_vol.isna().all() else 0
            current_park_vol = park_vol.iloc[-1] if not park_vol.isna().all() else 0
            current_gk_vol = gk_vol.iloc[-1] if not gk_vol.isna().all() else 0
            
            # Volatility percentiles
            hist_vol_pct = (current_hist_vol > hist_vol.quantile(0.8)) * 2 - 1
            park_vol_pct = (current_park_vol > park_vol.quantile(0.8)) * 2 - 1
            gk_vol_pct = (current_gk_vol > gk_vol.quantile(0.8)) * 2 - 1
            
            vol_score = (hist_vol_pct + park_vol_pct + gk_vol_pct) / 3
            
            # Regime classification
            if vol_score > 0.5:
                regime = "HIGH_VOLATILITY"
            elif vol_score < -0.5:
                regime = "LOW_VOLATILITY"
            else:
                regime = "NORMAL_VOLATILITY"
            
            return {
                'volatility_regime': regime,
                'volatility_score': float(vol_score),
                'current_volatility': {
                    'historical': float(current_hist_vol),
                    'parkinson': float(current_park_vol),
                    'garman_klass': float(current_gk_vol)
                },
                'volatility_percentiles': {
                    'historical': float(hist_vol_pct),
                    'parkinson': float(park_vol_pct),
                    'garman_klass': float(gk_vol_pct)
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def momentum_divergence_detector(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect momentum divergences
        """
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            if len(df) < 50:
                return {'divergence_detected': False, 'reason': 'Insufficient data'}
            
            # Price peaks and troughs
            price_peaks = high.rolling(5).max() == high
            price_troughs = low.rolling(5).min() == low
            
            # Momentum indicators
            rsi = close.diff().where(close.diff() > 0, 0).rolling(14).mean() / close.diff().abs().rolling(14).mean() * 100
            macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
            
            # Find recent peaks/troughs
            recent_peaks = df[price_peaks].tail(2)
            recent_troughs = df[price_troughs].tail(2)
            
            divergences = []
            
            # Bullish divergence check (price makes lower low, momentum makes higher low)
            if len(recent_troughs) == 2:
                price_lower_low = recent_troughs['low'].iloc[1] < recent_troughs['low'].iloc[0]
                rsi_higher_low = recent_troughs['close'].iloc[1] > recent_troughs['close'].iloc[0]  # Simplified
                
                if price_lower_low and rsi_higher_low:
                    divergences.append({
                        'type': 'BULLISH_DIVERGENCE',
                        'strength': 'MODERATE',
                        'indicator': 'RSI'
                    })
            
            # Bearish divergence check (price makes higher high, momentum makes lower high)
            if len(recent_peaks) == 2:
                price_higher_high = recent_peaks['high'].iloc[1] > recent_peaks['high'].iloc[0]
                rsi_lower_high = recent_peaks['close'].iloc[1] < recent_peaks['close'].iloc[0]  # Simplified
                
                if price_higher_high and rsi_lower_high:
                    divergences.append({
                        'type': 'BEARISH_DIVERGENCE',
                        'strength': 'MODERATE',
                        'indicator': 'RSI'
                    })
            
            return {
                'divergences_detected': len(divergences) > 0,
                'divergences': divergences,
                'total_divergences': len(divergences)
            }
            
        except Exception as e:
            return {'divergences_detected': False, 'error': str(e)}