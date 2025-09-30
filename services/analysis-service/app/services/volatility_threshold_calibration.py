"""
Volatility-Normalized Surprise Threshold Calibration

This module implements sophisticated surprise threshold calibration using:
- Asset-specific volatility normalization (realized & implied volatility)
- Sector/industry-specific threshold calibration
- Event type-specific surprise sensitivity
- Adaptive threshold adjustment based on market regime
- Multi-timeframe volatility analysis (intraday, daily, weekly)

Key Features:
- N-sigma threshold normalization based on rolling volatility
- Sector volatility clustering and threshold adjustment
- Event type sensitivity mapping (earnings vs guidance vs M&A)
- Market regime detection for threshold adaptation
- Cross-asset volatility spillover effects
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import asyncio
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types with different surprise sensitivities"""
    EARNINGS = "earnings"
    GUIDANCE = "guidance" 
    FDA_APPROVAL = "fda_approval"
    MERGER_ACQUISITION = "merger_acquisition"
    ANALYST_UPGRADE = "analyst_upgrade"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    INSIDER_TRADING = "insider_trading"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    MANAGEMENT_CHANGE = "management_change"

class MarketRegime(Enum):
    """Market regimes affecting threshold sensitivity"""
    LOW_VOLATILITY = "low_vol"
    NORMAL_VOLATILITY = "normal_vol" 
    HIGH_VOLATILITY = "high_vol"
    CRISIS = "crisis"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"

class SectorType(Enum):
    """Sector classifications with volatility characteristics"""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    BIOTECH = "biotech"
    FINANCIALS = "financials"
    ENERGY = "energy"
    UTILITIES = "utilities"
    CONSUMER_STAPLES = "consumer_staples"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    TELECOMMUNICATIONS = "telecommunications"

@dataclass
class VolatilityMetrics:
    """Comprehensive volatility metrics for threshold calibration"""
    symbol: str
    realized_vol_1d: float
    realized_vol_5d: float
    realized_vol_30d: float
    implied_vol: Optional[float] = None
    vol_of_vol: float = 0.0
    vol_skew: float = 0.0
    vol_regime: MarketRegime = MarketRegime.NORMAL_VOLATILITY
    sector_vol_percentile: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SectorVolatilityProfile:
    """Sector-specific volatility characteristics"""
    sector: SectorType
    median_vol: float
    vol_range: Tuple[float, float]  # (25th percentile, 75th percentile)
    event_sensitivity: Dict[EventType, float]
    beta_to_market: float
    volatility_clustering: float  # Tendency for vol clustering
    mean_reversion_speed: float

@dataclass
class ThresholdCalibration:
    """Volatility-normalized threshold calibration"""
    symbol: str
    event_type: EventType
    base_threshold: float
    volatility_adjustment: float
    sector_adjustment: float
    regime_adjustment: float
    final_threshold: float
    confidence_interval: Tuple[float, float]
    calibration_quality: float  # 0-1 score
    last_updated: datetime = field(default_factory=datetime.now)

class VolatilityThresholdCalibrator:
    """Advanced volatility-based surprise threshold calibration system"""
    
    def __init__(self):
        self.volatility_cache: Dict[str, VolatilityMetrics] = {}
        self.sector_profiles: Dict[SectorType, SectorVolatilityProfile] = {}
        self.threshold_cache: Dict[str, ThresholdCalibration] = {}
        self.market_regime_detector = MarketRegimeDetector()
        
        # Initialize sector profiles with realistic volatility characteristics
        self._initialize_sector_profiles()
        
        # Event type sensitivity mapping
        self.event_sensitivities = {
            EventType.EARNINGS: 1.0,          # Baseline sensitivity
            EventType.GUIDANCE: 1.2,          # Higher sensitivity to guidance
            EventType.FDA_APPROVAL: 2.5,      # Very high for biotech events
            EventType.MERGER_ACQUISITION: 2.0, # High M&A sensitivity
            EventType.ANALYST_UPGRADE: 0.8,   # Lower for analyst actions
            EventType.ANALYST_DOWNGRADE: 1.1, # Slightly higher for downgrades
            EventType.INSIDER_TRADING: 1.5,   # Moderate insider sensitivity
            EventType.REGULATORY: 1.8,        # High regulatory sensitivity
            EventType.PRODUCT_LAUNCH: 1.3,    # Moderate product sensitivity
            EventType.MANAGEMENT_CHANGE: 1.4  # Moderate management sensitivity
        }
    
    def _initialize_sector_profiles(self):
        """Initialize sector-specific volatility profiles"""
        
        self.sector_profiles = {
            SectorType.TECHNOLOGY: SectorVolatilityProfile(
                sector=SectorType.TECHNOLOGY,
                median_vol=0.25,
                vol_range=(0.18, 0.35),
                event_sensitivity={
                    EventType.EARNINGS: 1.1,
                    EventType.GUIDANCE: 1.3,
                    EventType.PRODUCT_LAUNCH: 1.5
                },
                beta_to_market=1.15,
                volatility_clustering=0.7,
                mean_reversion_speed=0.3
            ),
            
            SectorType.BIOTECH: SectorVolatilityProfile(
                sector=SectorType.BIOTECH,
                median_vol=0.45,
                vol_range=(0.30, 0.65),
                event_sensitivity={
                    EventType.FDA_APPROVAL: 3.0,
                    EventType.EARNINGS: 0.8,
                    EventType.REGULATORY: 2.2
                },
                beta_to_market=1.25,
                volatility_clustering=0.8,
                mean_reversion_speed=0.2
            ),
            
            SectorType.UTILITIES: SectorVolatilityProfile(
                sector=SectorType.UTILITIES,
                median_vol=0.12,
                vol_range=(0.08, 0.18),
                event_sensitivity={
                    EventType.EARNINGS: 0.7,
                    EventType.REGULATORY: 1.5,
                    EventType.GUIDANCE: 0.9
                },
                beta_to_market=0.65,
                volatility_clustering=0.4,
                mean_reversion_speed=0.6
            ),
            
            SectorType.ENERGY: SectorVolatilityProfile(
                sector=SectorType.ENERGY,
                median_vol=0.35,
                vol_range=(0.22, 0.55),
                event_sensitivity={
                    EventType.EARNINGS: 1.2,
                    EventType.GUIDANCE: 1.4,
                    EventType.REGULATORY: 1.8
                },
                beta_to_market=1.35,
                volatility_clustering=0.75,
                mean_reversion_speed=0.25
            ),
            
            SectorType.FINANCIALS: SectorVolatilityProfile(
                sector=SectorType.FINANCIALS,
                median_vol=0.28,
                vol_range=(0.18, 0.42),
                event_sensitivity={
                    EventType.EARNINGS: 1.3,
                    EventType.REGULATORY: 2.0,
                    EventType.GUIDANCE: 1.1
                },
                beta_to_market=1.1,
                volatility_clustering=0.6,
                mean_reversion_speed=0.4
            )
        }
    
    async def calculate_volatility_metrics(
        self, 
        symbol: str, 
        price_data: pd.DataFrame,
        sector: Optional[SectorType] = None
    ) -> VolatilityMetrics:
        """Calculate comprehensive volatility metrics for threshold calibration"""
        
        if len(price_data) < 30:
            logger.warning(f"Insufficient price data for {symbol}: {len(price_data)} days")
            return self._default_volatility_metrics(symbol, sector)
        
        # Ensure we have required columns
        required_cols = ['close', 'timestamp']
        if not all(col in price_data.columns for col in required_cols):
            logger.error(f"Missing required columns for {symbol}: {required_cols}")
            return self._default_volatility_metrics(symbol, sector)
        
        # Sort by timestamp and calculate returns
        price_data = price_data.sort_values('timestamp')
        price_data['returns'] = price_data['close'].pct_change()
        
        # Calculate realized volatilities (annualized)
        returns = price_data['returns'].dropna()
        
        realized_vol_1d = returns.tail(1).std() * np.sqrt(252) if len(returns) >= 1 else 0.2
        realized_vol_5d = returns.tail(5).std() * np.sqrt(252) if len(returns) >= 5 else 0.2
        realized_vol_30d = returns.tail(30).std() * np.sqrt(252) if len(returns) >= 30 else 0.2
        
        # Calculate volatility of volatility (rolling 5-day vol)
        if len(returns) >= 10:
            rolling_vol = returns.rolling(5).std() * np.sqrt(252)
            vol_of_vol = rolling_vol.std()
        else:
            vol_of_vol = 0.05
        
        # Calculate volatility skew (asymmetry in up vs down moves)
        if len(returns) >= 20:
            up_returns = returns[returns > 0]
            down_returns = returns[returns < 0]
            if len(up_returns) > 0 and len(down_returns) > 0:
                vol_skew = up_returns.std() / down_returns.std()
            else:
                vol_skew = 1.0
        else:
            vol_skew = 1.0
        
        # Detect market regime
        vol_regime = self.market_regime_detector.detect_regime(returns)
        
        # Calculate sector volatility percentile
        sector_vol_percentile = 0.5
        if sector and sector in self.sector_profiles:
            sector_profile = self.sector_profiles[sector]
            if realized_vol_30d <= sector_profile.vol_range[0]:
                sector_vol_percentile = 0.25
            elif realized_vol_30d >= sector_profile.vol_range[1]:
                sector_vol_percentile = 0.75
            else:
                # Linear interpolation between 25th and 75th percentiles
                range_size = sector_profile.vol_range[1] - sector_profile.vol_range[0]
                position = (realized_vol_30d - sector_profile.vol_range[0]) / range_size
                sector_vol_percentile = 0.25 + (position * 0.5)
        
        # Try to get implied volatility (would come from options data)
        implied_vol = self._estimate_implied_volatility(symbol, realized_vol_30d)
        
        metrics = VolatilityMetrics(
            symbol=symbol,
            realized_vol_1d=realized_vol_1d,
            realized_vol_5d=realized_vol_5d,
            realized_vol_30d=realized_vol_30d,
            implied_vol=implied_vol,
            vol_of_vol=vol_of_vol,
            vol_skew=vol_skew,
            vol_regime=vol_regime,
            sector_vol_percentile=sector_vol_percentile
        )
        
        # Cache the metrics
        self.volatility_cache[symbol] = metrics
        
        return metrics
    
    def _default_volatility_metrics(self, symbol: str, sector: Optional[SectorType]) -> VolatilityMetrics:
        """Return default volatility metrics when data is insufficient"""
        
        default_vol = 0.25  # 25% default volatility
        if sector and sector in self.sector_profiles:
            default_vol = self.sector_profiles[sector].median_vol
        
        return VolatilityMetrics(
            symbol=symbol,
            realized_vol_1d=default_vol,
            realized_vol_5d=default_vol,
            realized_vol_30d=default_vol,
            implied_vol=default_vol,
            vol_of_vol=0.05,
            vol_skew=1.0,
            vol_regime=MarketRegime.NORMAL_VOLATILITY,
            sector_vol_percentile=0.5
        )
    
    def _estimate_implied_volatility(self, symbol: str, realized_vol: float) -> Optional[float]:
        """Estimate implied volatility (would use options data in production)"""
        
        # Simple estimation: implied vol typically 10-20% higher than realized
        # In production, this would come from options chains
        implied_premium = np.random.uniform(1.1, 1.2)
        return realized_vol * implied_premium
    
    async def calibrate_threshold(
        self,
        symbol: str,
        event_type: EventType,
        base_threshold: float,
        volatility_metrics: Optional[VolatilityMetrics] = None,
        sector: Optional[SectorType] = None,
        target_sigma_level: float = 2.0
    ) -> ThresholdCalibration:
        """Calibrate surprise threshold using volatility normalization"""
        
        # Get or use provided volatility metrics
        if volatility_metrics is None:
            if symbol in self.volatility_cache:
                volatility_metrics = self.volatility_cache[symbol]
            else:
                logger.warning(f"No volatility metrics available for {symbol}")
                volatility_metrics = self._default_volatility_metrics(symbol, sector)
        
        # Volatility adjustment: normalize by N-sigma of realized volatility
        reference_vol = 0.20  # 20% reference volatility
        vol_ratio = volatility_metrics.realized_vol_30d / reference_vol
        
        # Apply log transformation to prevent extreme adjustments
        volatility_adjustment = np.log(vol_ratio) * target_sigma_level
        volatility_adjustment = np.clip(volatility_adjustment, -1.5, 1.5)  # Cap adjustments
        
        # Sector adjustment
        sector_adjustment = 0.0
        if sector and sector in self.sector_profiles:
            sector_profile = self.sector_profiles[sector]
            
            # Adjust based on sector volatility characteristics
            if volatility_metrics.sector_vol_percentile > 0.75:
                sector_adjustment = 0.3  # Higher threshold for high-vol assets in sector
            elif volatility_metrics.sector_vol_percentile < 0.25:
                sector_adjustment = -0.2  # Lower threshold for low-vol assets
            
            # Event-specific sector sensitivity
            if event_type in sector_profile.event_sensitivity:
                event_sensitivity = sector_profile.event_sensitivity[event_type]
                sector_adjustment *= event_sensitivity
        
        # Market regime adjustment
        regime_adjustment = self._calculate_regime_adjustment(
            volatility_metrics.vol_regime, event_type
        )
        
        # Event type sensitivity adjustment
        event_sensitivity = self.event_sensitivities.get(event_type, 1.0)
        
        # Calculate final threshold
        total_adjustment = (
            volatility_adjustment + 
            sector_adjustment + 
            regime_adjustment
        ) * event_sensitivity
        
        final_threshold = base_threshold * (1 + total_adjustment)
        final_threshold = max(final_threshold, base_threshold * 0.3)  # Minimum threshold
        
        # Calculate confidence interval
        vol_uncertainty = volatility_metrics.vol_of_vol / volatility_metrics.realized_vol_30d
        confidence_width = final_threshold * vol_uncertainty * 0.5
        confidence_interval = (
            final_threshold - confidence_width,
            final_threshold + confidence_width
        )
        
        # Calculate calibration quality score
        calibration_quality = self._calculate_calibration_quality(
            volatility_metrics, sector, event_type
        )
        
        calibration = ThresholdCalibration(
            symbol=symbol,
            event_type=event_type,
            base_threshold=base_threshold,
            volatility_adjustment=volatility_adjustment,
            sector_adjustment=sector_adjustment,
            regime_adjustment=regime_adjustment,
            final_threshold=final_threshold,
            confidence_interval=confidence_interval,
            calibration_quality=calibration_quality
        )
        
        # Cache the calibration
        cache_key = f"{symbol}_{event_type.value}"
        self.threshold_cache[cache_key] = calibration
        
        return calibration
    
    def _calculate_regime_adjustment(
        self, 
        regime: MarketRegime, 
        event_type: EventType
    ) -> float:
        """Calculate market regime-based threshold adjustment"""
        
        regime_adjustments = {
            MarketRegime.LOW_VOLATILITY: -0.2,   # Lower thresholds in low vol
            MarketRegime.NORMAL_VOLATILITY: 0.0, # No adjustment
            MarketRegime.HIGH_VOLATILITY: 0.3,   # Higher thresholds in high vol
            MarketRegime.CRISIS: 0.5,            # Much higher in crisis
            MarketRegime.TRENDING: -0.1,         # Slightly lower in trending markets
            MarketRegime.MEAN_REVERTING: 0.1     # Slightly higher in mean reverting
        }
        
        base_adjustment = regime_adjustments.get(regime, 0.0)
        
        # Event-specific regime sensitivity
        if event_type in [EventType.FDA_APPROVAL, EventType.MERGER_ACQUISITION]:
            # These events are less affected by general market regime
            base_adjustment *= 0.5
        elif event_type in [EventType.EARNINGS, EventType.GUIDANCE]:
            # These are more affected by market regime
            base_adjustment *= 1.2
        
        return base_adjustment
    
    def _calculate_calibration_quality(
        self,
        volatility_metrics: VolatilityMetrics,
        sector: Optional[SectorType],
        event_type: EventType
    ) -> float:
        """Calculate quality score for threshold calibration (0-1)"""
        
        quality_score = 0.5  # Base score
        
        # Data quality factors
        if volatility_metrics.implied_vol is not None:
            quality_score += 0.2  # Have implied vol data
        
        if sector is not None:
            quality_score += 0.15  # Have sector information
        
        # Volatility regime clarity
        if volatility_metrics.vol_of_vol < 0.1:
            quality_score += 0.1  # Low vol of vol indicates stable regime
        
        # Volatility level reasonableness
        if 0.1 <= volatility_metrics.realized_vol_30d <= 0.8:
            quality_score += 0.05  # Reasonable volatility level
        
        return min(quality_score, 1.0)
    
    async def get_adaptive_threshold(
        self,
        symbol: str,
        event_type: EventType,
        surprise_value: float,
        price_data: Optional[pd.DataFrame] = None,
        sector: Optional[SectorType] = None
    ) -> Dict[str, Any]:
        """Get adaptive threshold with comprehensive analysis"""
        
        # Calculate volatility metrics if price data provided
        if price_data is not None:
            volatility_metrics = await self.calculate_volatility_metrics(
                symbol, price_data, sector
            )
        else:
            volatility_metrics = self.volatility_cache.get(symbol)
            if volatility_metrics is None:
                volatility_metrics = self._default_volatility_metrics(symbol, sector)
        
        # Base threshold (would be configurable per event type)
        base_thresholds = {
            EventType.EARNINGS: 0.05,           # 5% surprise threshold
            EventType.GUIDANCE: 0.10,           # 10% guidance surprise
            EventType.FDA_APPROVAL: 0.30,       # 30% FDA event threshold
            EventType.MERGER_ACQUISITION: 0.15, # 15% M&A threshold
            EventType.ANALYST_UPGRADE: 0.03,    # 3% analyst action
            EventType.REGULATORY: 0.12          # 12% regulatory threshold
        }
        
        base_threshold = base_thresholds.get(event_type, 0.08)
        
        # Calibrate threshold
        calibration = await self.calibrate_threshold(
            symbol=symbol,
            event_type=event_type,
            base_threshold=base_threshold,
            volatility_metrics=volatility_metrics,
            sector=sector
        )
        
        # Determine if surprise exceeds threshold
        surprise_magnitude = abs(surprise_value)
        exceeds_threshold = surprise_magnitude > calibration.final_threshold
        
        # Calculate normalized surprise (z-score)
        normalized_surprise = surprise_magnitude / calibration.final_threshold
        
        # Calculate confidence in signal
        signal_confidence = min(normalized_surprise / 2.0, 1.0) * calibration.calibration_quality
        
        return {
            "symbol": symbol,
            "event_type": event_type.value,
            "surprise_value": surprise_value,
            "surprise_magnitude": surprise_magnitude,
            "base_threshold": base_threshold,
            "final_threshold": calibration.final_threshold,
            "exceeds_threshold": exceeds_threshold,
            "normalized_surprise": normalized_surprise,
            "signal_confidence": signal_confidence,
            "volatility_metrics": {
                "realized_vol_30d": volatility_metrics.realized_vol_30d,
                "vol_regime": volatility_metrics.vol_regime.value,
                "sector_vol_percentile": volatility_metrics.sector_vol_percentile
            },
            "adjustments": {
                "volatility_adjustment": calibration.volatility_adjustment,
                "sector_adjustment": calibration.sector_adjustment,
                "regime_adjustment": calibration.regime_adjustment
            },
            "confidence_interval": calibration.confidence_interval,
            "calibration_quality": calibration.calibration_quality,
            "timestamp": datetime.now()
        }

class MarketRegimeDetector:
    """Detect market volatility regime for threshold adjustment"""
    
    def detect_regime(self, returns: pd.Series) -> MarketRegime:
        """Detect current market regime based on return characteristics"""
        
        if len(returns) < 20:
            return MarketRegime.NORMAL_VOLATILITY
        
        # Calculate rolling statistics
        recent_returns = returns.tail(20)
        vol = recent_returns.std() * np.sqrt(252)
        
        # Volatility regime detection
        if vol < 0.12:
            return MarketRegime.LOW_VOLATILITY
        elif vol > 0.35:
            return MarketRegime.HIGH_VOLATILITY
        elif vol > 0.50:
            return MarketRegime.CRISIS
        
        # Trend vs mean reversion detection
        if len(returns) >= 30:
            # Calculate autocorrelation for trend detection
            autocorr = returns.tail(30).autocorr(lag=1)
            
            if autocorr > 0.2:
                return MarketRegime.TRENDING
            elif autocorr < -0.2:
                return MarketRegime.MEAN_REVERTING
        
        return MarketRegime.NORMAL_VOLATILITY

# Factory function
async def create_volatility_threshold_calibrator() -> VolatilityThresholdCalibrator:
    """Create volatility threshold calibrator"""
    return VolatilityThresholdCalibrator()