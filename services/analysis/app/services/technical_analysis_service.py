import pandas as pd
import numpy as np
import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

# Technical analysis libraries
import ta
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator

from ..models import (
    TechnicalAnalysisResponse, 
    MovingAverages, 
    Oscillators, 
    MACD as MACDModel,
    Signals,
    AdvancedIndicatorsResponse,
    IndicatorValue
)
from ..cache import AnalysisCacheService, AnalysisCacheKeys
from ..config import settings

logger = structlog.get_logger(__name__)

class TechnicalAnalysisService:
    def __init__(self, db: AsyncSession, redis_client):
        self.db = db
        self.cache = AnalysisCacheService(redis_client)
        self.market_data_client = httpx.AsyncClient(
            base_url=settings.MARKET_DATA_API_URL,
            timeout=30.0
        )
    
    async def get_technical_analysis(self, symbol: str, period: str = "6mo") -> Optional[TechnicalAnalysisResponse]:
        """Get comprehensive technical analysis"""
        cache_key = AnalysisCacheKeys.technical_analysis(symbol, period)
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug("Technical analysis cache hit", symbol=symbol, period=period)
            return TechnicalAnalysisResponse(**cached_result)
        
        try:
            # Get historical data from Market Data API
            historical_data = await self._fetch_historical_data(symbol, period)
            if not historical_data or len(historical_data) < settings.MIN_DATA_POINTS:
                logger.warning("Insufficient data for technical analysis", symbol=symbol, data_points=len(historical_data) if historical_data else 0)
                return None
            
            # Convert to DataFrame
            df = self._prepare_dataframe(historical_data)
            
            # Calculate technical indicators
            current_price = float(df['close'].iloc[-1])
            moving_averages = self._calculate_moving_averages(df)
            oscillators = self._calculate_oscillators(df)
            macd = self._calculate_macd(df)
            signals = self._generate_signals(df, moving_averages, oscillators, macd)
            
            # Create response
            result = TechnicalAnalysisResponse(
                symbol=symbol,
                current_price=current_price,
                moving_averages=moving_averages,
                oscillators=oscillators,
                macd=macd,
                signals=signals,
                calculated_at=datetime.now()
            )
            
            # Cache the result
            await self.cache.set(cache_key, result.dict(), settings.TECHNICAL_ANALYSIS_CACHE_TTL)
            
            return result
            
        except Exception as e:
            logger.error("Error calculating technical analysis", symbol=symbol, error=str(e))
            return None
    
    async def get_advanced_indicators(
        self, 
        symbol: str, 
        period: str = "6mo", 
        indicator_list: Optional[List[str]] = None
    ) -> AdvancedIndicatorsResponse:
        """Get advanced technical indicators"""
        indicators_key = ",".join(sorted(indicator_list)) if indicator_list else "all"
        cache_key = AnalysisCacheKeys.advanced_indicators(symbol, period, indicators_key)
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug("Advanced indicators cache hit", symbol=symbol)
            return AdvancedIndicatorsResponse(**cached_result)
        
        try:
            # Get historical data
            historical_data = await self._fetch_historical_data(symbol, period)
            if not historical_data or len(historical_data) < settings.MIN_DATA_POINTS:
                return AdvancedIndicatorsResponse(
                    symbol=symbol,
                    indicators={},
                    calculated_at=datetime.now()
                )
            
            # Convert to DataFrame
            df = self._prepare_dataframe(historical_data)
            
            # Calculate advanced indicators
            indicators = {}
            
            # Bollinger Bands
            if not indicator_list or "bollinger" in indicator_list:
                bb = BollingerBands(close=df['close'], window=20, window_dev=2)
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_middle'] = bb.bollinger_mavg()
                
                current_price = df['close'].iloc[-1]
                bb_upper = df['bb_upper'].iloc[-1]
                bb_lower = df['bb_lower'].iloc[-1]
                bb_middle = df['bb_middle'].iloc[-1]
                
                # Determine signal
                if current_price > bb_upper:
                    signal = "SELL"  # Overbought
                elif current_price < bb_lower:
                    signal = "BUY"   # Oversold
                else:
                    signal = "NEUTRAL"
                
                indicators["bollinger_bands"] = IndicatorValue(
                    value=float(current_price),
                    signal=signal,
                    metadata={
                        "upper_band": float(bb_upper),
                        "lower_band": float(bb_lower),
                        "middle_band": float(bb_middle),
                        "bandwidth": float((bb_upper - bb_lower) / bb_middle * 100)
                    }
                )
            
            # Average True Range (ATR)
            if not indicator_list or "atr" in indicator_list:
                atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
                current_atr = atr.iloc[-1]
                avg_atr = atr.rolling(50).mean().iloc[-1]
                
                # Higher ATR = higher volatility
                signal = "NEUTRAL"
                if current_atr > avg_atr * 1.5:
                    signal = "HIGH_VOLATILITY"
                elif current_atr < avg_atr * 0.5:
                    signal = "LOW_VOLATILITY"
                
                indicators["atr"] = IndicatorValue(
                    value=float(current_atr),
                    signal=signal,
                    metadata={
                        "average_atr": float(avg_atr),
                        "volatility_ratio": float(current_atr / avg_atr)
                    }
                )
            
            # Volume indicators
            if not indicator_list or "volume" in indicator_list:
                # On-Balance Volume
                obv = ta.volume.on_balance_volume(df['close'], df['volume'])
                obv_sma = obv.rolling(20).mean()
                
                current_obv = obv.iloc[-1]
                current_obv_sma = obv_sma.iloc[-1]
                
                signal = "BUY" if current_obv > current_obv_sma else "SELL"
                
                indicators["obv"] = IndicatorValue(
                    value=float(current_obv),
                    signal=signal,
                    metadata={
                        "obv_sma": float(current_obv_sma),
                        "trend": "up" if current_obv > current_obv_sma else "down"
                    }
                )
            
            # Fibonacci retracement levels
            if not indicator_list or "fibonacci" in indicator_list:
                # Calculate over recent swing
                recent_data = df.tail(50)
                high_price = recent_data['high'].max()
                low_price = recent_data['low'].min()
                current_price = df['close'].iloc[-1]
                
                diff = high_price - low_price
                fib_levels = {
                    "0.0": float(high_price),
                    "23.6": float(high_price - 0.236 * diff),
                    "38.2": float(high_price - 0.382 * diff),
                    "50.0": float(high_price - 0.5 * diff),
                    "61.8": float(high_price - 0.618 * diff),
                    "100.0": float(low_price)
                }
                
                # Determine which level current price is near
                signal = "NEUTRAL"
                for level_name, level_price in fib_levels.items():
                    if abs(current_price - level_price) / current_price < 0.02:  # Within 2%
                        if level_name in ["23.6", "38.2"]:
                            signal = "SUPPORT"
                        elif level_name in ["61.8", "50.0"]:
                            signal = "RESISTANCE"
                        break
                
                indicators["fibonacci"] = IndicatorValue(
                    value=float(current_price),
                    signal=signal,
                    metadata=fib_levels
                )
            
            # Commodity Channel Index (CCI)
            if not indicator_list or "cci" in indicator_list:
                cci = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
                current_cci = cci.iloc[-1]
                
                if current_cci > 100:
                    signal = "SELL"  # Overbought
                elif current_cci < -100:
                    signal = "BUY"   # Oversold
                else:
                    signal = "NEUTRAL"
                
                indicators["cci"] = IndicatorValue(
                    value=float(current_cci),
                    signal=signal,
                    metadata={
                        "overbought_threshold": 100,
                        "oversold_threshold": -100
                    }
                )
            
            result = AdvancedIndicatorsResponse(
                symbol=symbol,
                indicators=indicators,
                calculated_at=datetime.now()
            )
            
            # Cache the result
            await self.cache.set(cache_key, result.dict(), settings.ADVANCED_INDICATORS_CACHE_TTL)
            
            return result
            
        except Exception as e:
            logger.error("Error calculating advanced indicators", symbol=symbol, error=str(e))
            return AdvancedIndicatorsResponse(
                symbol=symbol,
                indicators={},
                calculated_at=datetime.now()
            )
    
    async def _fetch_historical_data(self, symbol: str, period: str) -> Optional[List[Dict]]:
        """Fetch historical data from Market Data API"""
        try:
            response = await self.market_data_client.get(f"/stocks/{symbol}/history", params={"period": period})
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error("Failed to fetch historical data", symbol=symbol, error=str(e))
            return None
    
    def _prepare_dataframe(self, historical_data: List[Dict]) -> pd.DataFrame:
        """Convert historical data to pandas DataFrame"""
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()
        
        # Ensure numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df = dropna(df)
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> MovingAverages:
        """Calculate moving averages"""
        try:
            sma_20 = SMAIndicator(close=df['close'], window=20).sma_indicator().iloc[-1]
            sma_50 = SMAIndicator(close=df['close'], window=50).sma_indicator().iloc[-1] if len(df) >= 50 else None
            sma_200 = SMAIndicator(close=df['close'], window=200).sma_indicator().iloc[-1] if len(df) >= 200 else None
            ema_12 = EMAIndicator(close=df['close'], window=12).ema_indicator().iloc[-1]
            ema_26 = EMAIndicator(close=df['close'], window=26).ema_indicator().iloc[-1]
            
            return MovingAverages(
                sma_20=float(sma_20) if not pd.isna(sma_20) else None,
                sma_50=float(sma_50) if sma_50 is not None and not pd.isna(sma_50) else None,
                sma_200=float(sma_200) if sma_200 is not None and not pd.isna(sma_200) else None,
                ema_12=float(ema_12) if not pd.isna(ema_12) else None,
                ema_26=float(ema_26) if not pd.isna(ema_26) else None
            )
        except Exception as e:
            logger.error("Error calculating moving averages", error=str(e))
            return MovingAverages()
    
    def _calculate_oscillators(self, df: pd.DataFrame) -> Oscillators:
        """Calculate oscillator indicators"""
        try:
            rsi = RSIIndicator(close=df['close'], window=14).rsi().iloc[-1]
            
            # Stochastic oscillator
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            stoch_k = stoch.stoch().iloc[-1] if len(df) >= 14 else None
            stoch_d = stoch.stoch_signal().iloc[-1] if len(df) >= 14 else None
            
            # Williams %R
            williams_r = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r().iloc[-1]
            
            return Oscillators(
                rsi_14=float(rsi) if not pd.isna(rsi) else None,
                stochastic_k=float(stoch_k) if stoch_k is not None and not pd.isna(stoch_k) else None,
                stochastic_d=float(stoch_d) if stoch_d is not None and not pd.isna(stoch_d) else None,
                williams_r=float(williams_r) if not pd.isna(williams_r) else None
            )
        except Exception as e:
            logger.error("Error calculating oscillators", error=str(e))
            return Oscillators()
    
    def _calculate_macd(self, df: pd.DataFrame) -> MACDModel:
        """Calculate MACD indicator"""
        try:
            macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            macd_line = macd_indicator.macd().iloc[-1]
            signal_line = macd_indicator.macd_signal().iloc[-1]
            histogram = macd_indicator.macd_diff().iloc[-1]
            
            return MACDModel(
                macd=float(macd_line) if not pd.isna(macd_line) else None,
                signal=float(signal_line) if not pd.isna(signal_line) else None,
                histogram=float(histogram) if not pd.isna(histogram) else None
            )
        except Exception as e:
            logger.error("Error calculating MACD", error=str(e))
            return MACDModel()
    
    def _generate_signals(
        self, 
        df: pd.DataFrame, 
        ma: MovingAverages, 
        osc: Oscillators, 
        macd: MACDModel
    ) -> Signals:
        """Generate trading signals based on technical indicators"""
        try:
            signals = []
            buy_signals = 0
            sell_signals = 0
            
            current_price = df['close'].iloc[-1]
            
            # Moving average signals
            if ma.sma_20 and ma.sma_50:
                if ma.sma_20 > ma.sma_50:
                    signals.append("SMA20 > SMA50: Bullish")
                    buy_signals += 1
                else:
                    signals.append("SMA20 < SMA50: Bearish")
                    sell_signals += 1
            
            if ma.sma_20:
                if current_price > ma.sma_20:
                    signals.append("Price > SMA20: Bullish")
                    buy_signals += 1
                else:
                    signals.append("Price < SMA20: Bearish")
                    sell_signals += 1
            
            # RSI signals
            if osc.rsi_14:
                if osc.rsi_14 > 70:
                    signals.append("RSI > 70: Overbought")
                    sell_signals += 1
                elif osc.rsi_14 < 30:
                    signals.append("RSI < 30: Oversold")
                    buy_signals += 1
                else:
                    signals.append("RSI neutral")
            
            # MACD signals
            if macd.macd and macd.signal:
                if macd.macd > macd.signal:
                    signals.append("MACD > Signal: Bullish")
                    buy_signals += 1
                else:
                    signals.append("MACD < Signal: Bearish")
                    sell_signals += 1
            
            # Determine overall signal
            net_signal = buy_signals - sell_signals
            total_signals = buy_signals + sell_signals
            
            if net_signal >= 2:
                overall_signal = "STRONG_BUY" if net_signal >= 3 else "BUY"
            elif net_signal <= -2:
                overall_signal = "STRONG_SELL" if net_signal <= -3 else "SELL"
            else:
                overall_signal = "HOLD"
            
            # Calculate signal strength
            strength = abs(net_signal) / max(total_signals, 1) if total_signals > 0 else 0.0
            
            individual_signals = {f"signal_{i+1}": signal for i, signal in enumerate(signals[:5])}
            
            return Signals(
                overall_signal=overall_signal,
                strength=strength,
                individual_signals=individual_signals
            )
            
        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return Signals(overall_signal="HOLD", strength=0.0, individual_signals={})