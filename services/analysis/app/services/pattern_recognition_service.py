import pandas as pd
import numpy as np
import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import httpx
from scipy.signal import find_peaks, argrelextrema

from ..models import ChartPatternsResponse, ChartPattern
from ..cache import AnalysisCacheService, AnalysisCacheKeys
from ..config import settings

logger = structlog.get_logger(__name__)

class PatternRecognitionService:
    def __init__(self, db, redis_client):
        self.db = db
        self.cache = AnalysisCacheService(redis_client)
        self.market_data_client = httpx.AsyncClient(
            base_url=settings.MARKET_DATA_API_URL,
            timeout=30.0
        )
    
    async def get_chart_patterns(self, symbol: str, period: str = "3mo") -> ChartPatternsResponse:
        """Detect chart patterns"""
        cache_key = AnalysisCacheKeys.chart_patterns(symbol, period)
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            logger.debug("Chart patterns cache hit", symbol=symbol, period=period)
            return ChartPatternsResponse(**cached_result)
        
        try:
            # Get historical data
            historical_data = await self._fetch_historical_data(symbol, period)
            if not historical_data or len(historical_data) < 50:
                return ChartPatternsResponse(
                    symbol=symbol,
                    patterns=[],
                    calculated_at=datetime.now()
                )
            
            # Convert to DataFrame
            df = self._prepare_dataframe(historical_data)
            
            # Detect patterns
            patterns = []
            
            # Head and Shoulders
            head_shoulders = self._detect_head_and_shoulders(df)
            patterns.extend(head_shoulders)
            
            # Double Top/Bottom
            double_patterns = self._detect_double_top_bottom(df)
            patterns.extend(double_patterns)
            
            # Triangle patterns
            triangle_patterns = self._detect_triangles(df)
            patterns.extend(triangle_patterns)
            
            # Support and Resistance
            support_resistance = self._detect_support_resistance(df)
            patterns.extend(support_resistance)
            
            # Flag and Pennant
            flag_patterns = self._detect_flags_pennants(df)
            patterns.extend(flag_patterns)
            
            result = ChartPatternsResponse(
                symbol=symbol,
                patterns=patterns,
                calculated_at=datetime.now()
            )
            
            # Cache the result
            await self.cache.set(cache_key, result.dict(), settings.PATTERN_CACHE_TTL)
            
            return result
            
        except Exception as e:
            logger.error("Error detecting chart patterns", symbol=symbol, error=str(e))
            return ChartPatternsResponse(
                symbol=symbol,
                patterns=[],
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
            logger.error("Failed to fetch historical data for patterns", symbol=symbol, error=str(e))
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
        
        # Remove any NaN values
        df = df.dropna()
        
        return df
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Head and Shoulders pattern"""
        patterns = []
        
        try:
            # Find peaks in the price data
            highs = df['high'].values
            peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
            
            if len(peaks) < 3:
                return patterns
            
            # Look for head and shoulders pattern
            for i in range(1, len(peaks) - 1):
                left_shoulder = peaks[i-1]
                head = peaks[i]
                right_shoulder = peaks[i+1]
                
                left_height = highs[left_shoulder]
                head_height = highs[head]
                right_height = highs[right_shoulder]
                
                # Check if head is higher than both shoulders
                if (head_height > left_height and head_height > right_height and
                    abs(left_height - right_height) / head_height < 0.05):  # Shoulders roughly equal
                    
                    # Calculate neckline (support level)
                    start_date = df.index[left_shoulder]
                    end_date = df.index[right_shoulder]
                    
                    # Find the lowest point between shoulders for neckline
                    neckline_section = df.loc[start_date:end_date]
                    neckline_level = neckline_section['low'].min()
                    
                    # Target price (head height - neckline)
                    target_price = neckline_level - (head_height - neckline_level)
                    
                    patterns.append(ChartPattern(
                        pattern_type="head_and_shoulders",
                        direction="bearish",
                        confidence=0.7,
                        start_date=start_date,
                        end_date=end_date,
                        target_price=float(target_price),
                        description=f"Head and Shoulders pattern with neckline at {neckline_level:.2f}"
                    ))
            
            # Look for Inverse Head and Shoulders (bullish)
            lows = df['low'].values
            troughs, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
            
            if len(troughs) >= 3:
                for i in range(1, len(troughs) - 1):
                    left_shoulder = troughs[i-1]
                    head = troughs[i]
                    right_shoulder = troughs[i+1]
                    
                    left_depth = lows[left_shoulder]
                    head_depth = lows[head]
                    right_depth = lows[right_shoulder]
                    
                    # Check if head is lower than both shoulders
                    if (head_depth < left_depth and head_depth < right_depth and
                        abs(left_depth - right_depth) / head_depth < 0.05):
                        
                        start_date = df.index[left_shoulder]
                        end_date = df.index[right_shoulder]
                        
                        # Neckline is resistance level
                        neckline_section = df.loc[start_date:end_date]
                        neckline_level = neckline_section['high'].max()
                        
                        target_price = neckline_level + (neckline_level - head_depth)
                        
                        patterns.append(ChartPattern(
                            pattern_type="inverse_head_and_shoulders",
                            direction="bullish",
                            confidence=0.7,
                            start_date=start_date,
                            end_date=end_date,
                            target_price=float(target_price),
                            description=f"Inverse Head and Shoulders with neckline at {neckline_level:.2f}"
                        ))
        
        except Exception as e:
            logger.error("Error detecting head and shoulders", error=str(e))
        
        return patterns
    
    def _detect_double_top_bottom(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Double Top and Double Bottom patterns"""
        patterns = []
        
        try:
            # Double Top (bearish)
            highs = df['high'].values
            peaks, _ = find_peaks(highs, distance=10, prominence=np.std(highs) * 0.3)
            
            for i in range(len(peaks) - 1):
                for j in range(i + 1, len(peaks)):
                    peak1_height = highs[peaks[i]]
                    peak2_height = highs[peaks[j]]
                    
                    # Check if peaks are roughly equal height
                    if abs(peak1_height - peak2_height) / peak1_height < 0.03:
                        start_date = df.index[peaks[i]]
                        end_date = df.index[peaks[j]]
                        
                        # Find the valley between peaks
                        valley_section = df.loc[start_date:end_date]
                        valley_level = valley_section['low'].min()
                        
                        # Target price below valley
                        avg_peak = (peak1_height + peak2_height) / 2
                        target_price = valley_level - (avg_peak - valley_level)
                        
                        patterns.append(ChartPattern(
                            pattern_type="double_top",
                            direction="bearish",
                            confidence=0.6,
                            start_date=start_date,
                            end_date=end_date,
                            target_price=float(target_price),
                            description=f"Double Top at {avg_peak:.2f} with support at {valley_level:.2f}"
                        ))
                        break
            
            # Double Bottom (bullish)
            lows = df['low'].values
            troughs, _ = find_peaks(-lows, distance=10, prominence=np.std(lows) * 0.3)
            
            for i in range(len(troughs) - 1):
                for j in range(i + 1, len(troughs)):
                    trough1_depth = lows[troughs[i]]
                    trough2_depth = lows[troughs[j]]
                    
                    if abs(trough1_depth - trough2_depth) / trough1_depth < 0.03:
                        start_date = df.index[troughs[i]]
                        end_date = df.index[troughs[j]]
                        
                        # Find the peak between troughs
                        peak_section = df.loc[start_date:end_date]
                        peak_level = peak_section['high'].max()
                        
                        avg_trough = (trough1_depth + trough2_depth) / 2
                        target_price = peak_level + (peak_level - avg_trough)
                        
                        patterns.append(ChartPattern(
                            pattern_type="double_bottom",
                            direction="bullish",
                            confidence=0.6,
                            start_date=start_date,
                            end_date=end_date,
                            target_price=float(target_price),
                            description=f"Double Bottom at {avg_trough:.2f} with resistance at {peak_level:.2f}"
                        ))
                        break
        
        except Exception as e:
            logger.error("Error detecting double top/bottom", error=str(e))
        
        return patterns
    
    def _detect_triangles(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Triangle patterns (Ascending, Descending, Symmetrical)"""
        patterns = []
        
        try:
            # Use recent data for triangle detection
            recent_df = df.tail(50)
            
            if len(recent_df) < 20:
                return patterns
            
            # Find trend lines
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            x = np.arange(len(recent_df))
            
            # Fit trend lines to highs and lows
            high_slope, high_intercept = np.polyfit(x, highs, 1)
            low_slope, low_intercept = np.polyfit(x, lows, 1)
            
            # Determine triangle type
            if abs(high_slope) < 0.01 and low_slope > 0.01:
                # Ascending triangle (bullish)
                pattern_type = "ascending_triangle"
                direction = "bullish"
                confidence = 0.5
                description = "Ascending triangle - horizontal resistance, rising support"
                
            elif high_slope < -0.01 and abs(low_slope) < 0.01:
                # Descending triangle (bearish)
                pattern_type = "descending_triangle"
                direction = "bearish"
                confidence = 0.5
                description = "Descending triangle - falling resistance, horizontal support"
                
            elif high_slope < -0.01 and low_slope > 0.01:
                # Symmetrical triangle (direction depends on breakout)
                pattern_type = "symmetrical_triangle"
                direction = "neutral"
                confidence = 0.4
                description = "Symmetrical triangle - converging trend lines"
            else:
                return patterns
            
            start_date = recent_df.index[0]
            end_date = recent_df.index[-1]
            
            # Estimate breakout target
            triangle_height = recent_df['high'].max() - recent_df['low'].min()
            current_price = recent_df['close'].iloc[-1]
            
            if direction == "bullish":
                target_price = current_price + triangle_height
            elif direction == "bearish":
                target_price = current_price - triangle_height
            else:
                target_price = current_price
            
            patterns.append(ChartPattern(
                pattern_type=pattern_type,
                direction=direction,
                confidence=confidence,
                start_date=start_date,
                end_date=end_date,
                target_price=float(target_price),
                description=description
            ))
        
        except Exception as e:
            logger.error("Error detecting triangles", error=str(e))
        
        return patterns
    
    def _detect_support_resistance(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Support and Resistance levels"""
        patterns = []
        
        try:
            # Use recent data
            recent_df = df.tail(100)
            
            # Find potential support levels (price bounces up from these levels)
            lows = recent_df['low'].values
            support_levels = []
            
            for i in range(2, len(lows) - 2):
                if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
                    lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                    
                    # Check if this level has been tested multiple times
                    level = lows[i]
                    touches = sum(1 for low in lows if abs(low - level) / level < 0.02)
                    
                    if touches >= 2:
                        support_levels.append((recent_df.index[i], level, touches))
            
            # Find potential resistance levels
            highs = recent_df['high'].values
            resistance_levels = []
            
            for i in range(2, len(highs) - 2):
                if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
                    highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                    
                    level = highs[i]
                    touches = sum(1 for high in highs if abs(high - level) / level < 0.02)
                    
                    if touches >= 2:
                        resistance_levels.append((recent_df.index[i], level, touches))
            
            # Create patterns for strongest levels
            current_price = recent_df['close'].iloc[-1]
            
            # Support levels
            for date, level, touches in sorted(support_levels, key=lambda x: x[2], reverse=True)[:3]:
                if level < current_price:  # Support should be below current price
                    confidence = min(0.8, 0.3 + touches * 0.1)
                    patterns.append(ChartPattern(
                        pattern_type="support_level",
                        direction="bullish",
                        confidence=confidence,
                        start_date=date,
                        end_date=recent_df.index[-1],
                        target_price=float(level),
                        description=f"Support level at {level:.2f} (tested {touches} times)"
                    ))
            
            # Resistance levels
            for date, level, touches in sorted(resistance_levels, key=lambda x: x[2], reverse=True)[:3]:
                if level > current_price:  # Resistance should be above current price
                    confidence = min(0.8, 0.3 + touches * 0.1)
                    patterns.append(ChartPattern(
                        pattern_type="resistance_level",
                        direction="bearish",
                        confidence=confidence,
                        start_date=date,
                        end_date=recent_df.index[-1],
                        target_price=float(level),
                        description=f"Resistance level at {level:.2f} (tested {touches} times)"
                    ))
        
        except Exception as e:
            logger.error("Error detecting support/resistance", error=str(e))
        
        return patterns
    
    def _detect_flags_pennants(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Flag and Pennant patterns"""
        patterns = []
        
        try:
            # Look for recent strong moves followed by consolidation
            recent_df = df.tail(30)
            
            if len(recent_df) < 20:
                return patterns
            
            # Split into two halves
            first_half = recent_df.iloc[:15]
            second_half = recent_df.iloc[15:]
            
            # Check for strong initial move
            first_move = (first_half['close'].iloc[-1] - first_half['close'].iloc[0]) / first_half['close'].iloc[0]
            
            if abs(first_move) > 0.05:  # At least 5% move
                # Check for consolidation in second half
                second_volatility = second_half['high'].std() / second_half['close'].mean()
                
                if second_volatility < 0.02:  # Low volatility consolidation
                    direction = "bullish" if first_move > 0 else "bearish"
                    pattern_type = "bull_flag" if direction == "bullish" else "bear_flag"
                    
                    # Target is typically equal to the flagpole height
                    flagpole_height = abs(first_half['close'].iloc[-1] - first_half['close'].iloc[0])
                    current_price = recent_df['close'].iloc[-1]
                    
                    if direction == "bullish":
                        target_price = current_price + flagpole_height
                    else:
                        target_price = current_price - flagpole_height
                    
                    patterns.append(ChartPattern(
                        pattern_type=pattern_type,
                        direction=direction,
                        confidence=0.5,
                        start_date=recent_df.index[0],
                        end_date=recent_df.index[-1],
                        target_price=float(target_price),
                        description=f"{pattern_type.replace('_', ' ').title()} pattern after {first_move:.1%} move"
                    ))
        
        except Exception as e:
            logger.error("Error detecting flags/pennants", error=str(e))
        
        return patterns