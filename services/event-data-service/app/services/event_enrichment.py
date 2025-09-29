"""Real-time event enrichment with market cap, sector, and volatility context."""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class MarketCapTier(str, Enum):
    """Market capitalization tiers."""
    MEGA_CAP = "mega_cap"        # >$200B
    LARGE_CAP = "large_cap"      # $10B-$200B
    MID_CAP = "mid_cap"          # $2B-$10B
    SMALL_CAP = "small_cap"      # $300M-$2B
    MICRO_CAP = "micro_cap"      # <$300M
    UNKNOWN = "unknown"


class VolatilityLevel(str, Enum):
    """Volatility level classifications."""
    VERY_LOW = "very_low"        # <10%
    LOW = "low"                  # 10-20%
    MODERATE = "moderate"        # 20-35%
    HIGH = "high"                # 35-50%
    VERY_HIGH = "very_high"      # >50%
    UNKNOWN = "unknown"


@dataclass
class MarketContext:
    """Market context data for a symbol."""
    symbol: str
    market_cap: Optional[float] = None
    market_cap_tier: MarketCapTier = MarketCapTier.UNKNOWN
    sector: Optional[str] = None
    industry: Optional[str] = None
    avg_volume: Optional[float] = None
    beta: Optional[float] = None
    volatility_30d: Optional[float] = None
    volatility_level: VolatilityLevel = VolatilityLevel.UNKNOWN
    price: Optional[float] = None
    shares_outstanding: Optional[float] = None
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "market_cap": self.market_cap,
            "market_cap_tier": self.market_cap_tier.value,
            "sector": self.sector,
            "industry": self.industry,
            "avg_volume": self.avg_volume,
            "beta": self.beta,
            "volatility_30d": self.volatility_30d,
            "volatility_level": self.volatility_level.value,
            "price": self.price,
            "shares_outstanding": self.shares_outstanding,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


@dataclass 
class EnrichmentConfig:
    """Configuration for event enrichment."""
    finnhub_api_key: Optional[str] = None
    yahoo_finance_enabled: bool = True
    alpha_vantage_api_key: Optional[str] = None
    cache_duration_minutes: int = 30
    max_retries: int = 3
    timeout_seconds: float = 5.0
    batch_size: int = 10
    
    @classmethod
    def from_env(cls) -> 'EnrichmentConfig':
        """Create config from environment variables."""
        return cls(
            finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
            yahoo_finance_enabled=os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true",
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            cache_duration_minutes=int(os.getenv("ENRICHMENT_CACHE_DURATION_MINUTES", "30")),
            max_retries=int(os.getenv("ENRICHMENT_MAX_RETRIES", "3")),
            timeout_seconds=float(os.getenv("ENRICHMENT_TIMEOUT_SECONDS", "5.0")),
            batch_size=int(os.getenv("ENRICHMENT_BATCH_SIZE", "10"))
        )


class EventEnrichmentService:
    """Service for enriching events with real-time market context."""
    
    def __init__(self, config: Optional[EnrichmentConfig] = None):
        self.config = config or EnrichmentConfig.from_env()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._context_cache: Dict[str, Tuple[MarketContext, datetime]] = {}
        self._sector_mapping = self._load_sector_mapping()
        
    async def start(self):
        """Start the enrichment service."""
        self._http_client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        logger.info("Event enrichment service started")
        
    async def stop(self):
        """Stop the enrichment service."""
        if self._http_client:
            await self._http_client.aclose()
        logger.info("Event enrichment service stopped")
        
    def _load_sector_mapping(self) -> Dict[str, str]:
        """Load sector mapping from environment or defaults."""
        env_mapping = os.getenv("EVENT_SECTOR_MAPPING")
        if env_mapping:
            try:
                return json.loads(env_mapping)
            except json.JSONDecodeError:
                logger.warning("Invalid EVENT_SECTOR_MAPPING JSON, using defaults")
        
        # Default sector mapping for common symbols
        return {
            "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology", "AMZN": "consumer_discretionary",
            "TSLA": "consumer_discretionary", "NVDA": "technology", "META": "technology", "NFLX": "communication_services",
            "JPM": "financials", "BAC": "financials", "WFC": "financials", "GS": "financials", "MS": "financials",
            "JNJ": "healthcare", "PFE": "healthcare", "UNH": "healthcare", "ABBV": "healthcare", "TMO": "healthcare",
            "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy", "EOG": "energy",
            "KO": "consumer_staples", "PEP": "consumer_staples", "WMT": "consumer_staples", "PG": "consumer_staples"
        }
        
    def _get_cached_context(self, symbol: str) -> Optional[MarketContext]:
        """Get cached market context if still valid."""
        if symbol not in self._context_cache:
            return None
            
        context, cached_at = self._context_cache[symbol]
        cache_duration = timedelta(minutes=self.config.cache_duration_minutes)
        
        if datetime.utcnow() - cached_at > cache_duration:
            del self._context_cache[symbol]
            return None
            
        return context
        
    def _cache_context(self, context: MarketContext):
        """Cache market context."""
        self._context_cache[context.symbol] = (context, datetime.utcnow())
        
    async def get_market_context(self, symbol: str) -> MarketContext:
        """Get comprehensive market context for a symbol."""
        # Check cache first
        cached = self._get_cached_context(symbol)
        if cached:
            return cached
            
        # Fetch fresh data
        context = MarketContext(symbol=symbol)
        
        # Try multiple data sources
        await self._enrich_from_finnhub(context)
        await self._enrich_from_yahoo_finance(context)
        await self._enrich_from_sector_mapping(context)
        
        # Calculate derived metrics
        self._calculate_market_cap_tier(context)
        self._calculate_volatility_level(context)
        
        context.last_updated = datetime.utcnow()
        self._cache_context(context)
        
        return context
        
    async def _enrich_from_finnhub(self, context: MarketContext):
        """Enrich context using Finnhub API."""
        if not self.config.finnhub_api_key or not self._http_client:
            return
            
        try:
            # Get company profile
            profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={context.symbol}&token={self.config.finnhub_api_key}"
            response = await self._http_client.get(profile_url)
            
            if response.status_code == 200:
                data = response.json()
                context.market_cap = data.get("marketCapitalization")
                if context.market_cap:
                    context.market_cap *= 1_000_000  # Finnhub returns in millions
                    
                context.sector = data.get("finnhubIndustry")
                context.shares_outstanding = data.get("shareOutstanding")
                if context.shares_outstanding:
                    context.shares_outstanding *= 1_000_000  # Finnhub returns in millions
                    
            # Get basic financials for beta
            financials_url = f"https://finnhub.io/api/v1/stock/metric?symbol={context.symbol}&metric=all&token={self.config.finnhub_api_key}"
            response = await self._http_client.get(financials_url)
            
            if response.status_code == 200:
                data = response.json()
                metrics = data.get("metric", {})
                context.beta = metrics.get("beta")
                
            # Get quote for current price
            quote_url = f"https://finnhub.io/api/v1/quote?symbol={context.symbol}&token={self.config.finnhub_api_key}"
            response = await self._http_client.get(quote_url)
            
            if response.status_code == 200:
                data = response.json()
                context.price = data.get("c")  # Current price
                
        except Exception as e:
            logger.warning(f"Failed to enrich from Finnhub for {context.symbol}: {e}")
            
    async def _enrich_from_yahoo_finance(self, context: MarketContext):
        """Enrich context using Yahoo Finance (via yfinance-like API)."""
        if not self.config.yahoo_finance_enabled or not self._http_client:
            return
            
        try:
            # Use a free Yahoo Finance API alternative
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{context.symbol}"
            
            response = await self._http_client.get(url)
            if response.status_code == 200:
                data = response.json()
                result = data.get("chart", {}).get("result", [])
                
                if result:
                    meta = result[0].get("meta", {})
                    context.price = context.price or meta.get("regularMarketPrice")
                    
                    # Calculate average volume from recent data
                    timestamps = result[0].get("timestamp", [])
                    volumes = result[0].get("indicators", {}).get("quote", [{}])[0].get("volume", [])
                    
                    if volumes:
                        # Get last 30 trading days for average volume
                        recent_volumes = [v for v in volumes[-30:] if v is not None]
                        if recent_volumes:
                            context.avg_volume = sum(recent_volumes) / len(recent_volumes)
                            
                    # Calculate 30-day volatility from price data
                    highs = result[0].get("indicators", {}).get("quote", [{}])[0].get("high", [])
                    lows = result[0].get("indicators", {}).get("quote", [{}])[0].get("low", [])
                    
                    if highs and lows and len(highs) >= 30:
                        # Simple volatility calculation: average of (high-low)/close over 30 days
                        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                        if closes:
                            volatilities = []
                            for i in range(-30, 0):
                                if (highs[i] is not None and lows[i] is not None and 
                                    closes[i] is not None and closes[i] > 0):
                                    daily_vol = (highs[i] - lows[i]) / closes[i]
                                    volatilities.append(daily_vol)
                            
                            if volatilities:
                                context.volatility_30d = (sum(volatilities) / len(volatilities)) * 100  # Convert to percentage
                                
        except Exception as e:
            logger.warning(f"Failed to enrich from Yahoo Finance for {context.symbol}: {e}")
            
    async def _enrich_from_sector_mapping(self, context: MarketContext):
        """Enrich context from predefined sector mapping."""
        if context.symbol in self._sector_mapping:
            context.sector = context.sector or self._sector_mapping[context.symbol]
            
    def _calculate_market_cap_tier(self, context: MarketContext):
        """Calculate market cap tier based on market cap."""
        if not context.market_cap:
            context.market_cap_tier = MarketCapTier.UNKNOWN
            return
            
        market_cap_b = context.market_cap / 1_000_000_000  # Convert to billions
        
        if market_cap_b >= 200:
            context.market_cap_tier = MarketCapTier.MEGA_CAP
        elif market_cap_b >= 10:
            context.market_cap_tier = MarketCapTier.LARGE_CAP
        elif market_cap_b >= 2:
            context.market_cap_tier = MarketCapTier.MID_CAP
        elif market_cap_b >= 0.3:
            context.market_cap_tier = MarketCapTier.SMALL_CAP
        else:
            context.market_cap_tier = MarketCapTier.MICRO_CAP
            
    def _calculate_volatility_level(self, context: MarketContext):
        """Calculate volatility level based on 30-day volatility."""
        if not context.volatility_30d:
            context.volatility_level = VolatilityLevel.UNKNOWN
            return
            
        vol = context.volatility_30d
        
        if vol < 10:
            context.volatility_level = VolatilityLevel.VERY_LOW
        elif vol < 20:
            context.volatility_level = VolatilityLevel.LOW
        elif vol < 35:
            context.volatility_level = VolatilityLevel.MODERATE
        elif vol < 50:
            context.volatility_level = VolatilityLevel.HIGH
        else:
            context.volatility_level = VolatilityLevel.VERY_HIGH
            
    async def enrich_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a single event with market context."""
        symbol = event_data.get("symbol")
        if not symbol:
            return event_data
            
        try:
            context = await self.get_market_context(symbol)
            
            # Add enrichment to metadata
            metadata = event_data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
                
            enrichment = metadata.setdefault("enrichment", {})
            enrichment.update({
                "market_context": context.to_dict(),
                "enrichment_timestamp": datetime.utcnow().isoformat(),
                "enrichment_version": "1.0"
            })
            
            # Add impact modifiers based on context
            impact_modifiers = self._calculate_impact_modifiers(context)
            if impact_modifiers:
                enrichment["impact_modifiers"] = impact_modifiers
                
            event_data["metadata"] = metadata
            
        except Exception as e:
            logger.error(f"Failed to enrich event for {symbol}: {e}")
            
        return event_data
        
    def _calculate_impact_modifiers(self, context: MarketContext) -> Dict[str, Any]:
        """Calculate impact score modifiers based on market context."""
        modifiers = {}
        
        # Market cap tier modifier
        tier_modifiers = {
            MarketCapTier.MEGA_CAP: 2.0,
            MarketCapTier.LARGE_CAP: 1.5,
            MarketCapTier.MID_CAP: 1.0,
            MarketCapTier.SMALL_CAP: 0.5,
            MarketCapTier.MICRO_CAP: -0.5,
        }
        
        if context.market_cap_tier != MarketCapTier.UNKNOWN:
            modifiers["market_cap_modifier"] = tier_modifiers.get(context.market_cap_tier, 0.0)
            
        # Volatility modifier (higher volatility = higher potential impact)
        vol_modifiers = {
            VolatilityLevel.VERY_HIGH: 1.5,
            VolatilityLevel.HIGH: 1.0,
            VolatilityLevel.MODERATE: 0.5,
            VolatilityLevel.LOW: 0.0,
            VolatilityLevel.VERY_LOW: -0.5,
        }
        
        if context.volatility_level != VolatilityLevel.UNKNOWN:
            modifiers["volatility_modifier"] = vol_modifiers.get(context.volatility_level, 0.0)
            
        # Beta modifier (higher beta = more market sensitive)
        if context.beta is not None:
            if context.beta > 1.5:
                modifiers["beta_modifier"] = 1.0
            elif context.beta > 1.2:
                modifiers["beta_modifier"] = 0.5
            elif context.beta < 0.8:
                modifiers["beta_modifier"] = -0.5
            else:
                modifiers["beta_modifier"] = 0.0
                
        # Liquidity modifier based on average volume
        if context.avg_volume is not None:
            if context.avg_volume > 50_000_000:  # Very high volume
                modifiers["liquidity_modifier"] = 0.5
            elif context.avg_volume < 1_000_000:  # Low volume
                modifiers["liquidity_modifier"] = -1.0
            else:
                modifiers["liquidity_modifier"] = 0.0
                
        return modifiers
        
    async def batch_enrich_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich multiple events in batches for efficiency."""
        enriched_events = []
        
        # Process in batches to avoid overwhelming APIs
        for i in range(0, len(events), self.config.batch_size):
            batch = events[i:i + self.config.batch_size]
            
            # Enrich each event in the batch
            batch_tasks = [self.enrich_event(event) for event in batch]
            enriched_batch = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, result in enumerate(enriched_batch):
                if isinstance(result, Exception):
                    logger.error(f"Failed to enrich event in batch: {result}")
                    enriched_events.append(batch[j])  # Use original event
                else:
                    enriched_events.append(result)
                    
            # Small delay between batches to respect rate limits
            if i + self.config.batch_size < len(events):
                await asyncio.sleep(0.1)
                
        return enriched_events
        
    def get_enrichment_stats(self) -> Dict[str, Any]:
        """Get enrichment service statistics."""
        return {
            "cached_symbols": len(self._context_cache),
            "config": {
                "cache_duration_minutes": self.config.cache_duration_minutes,
                "max_retries": self.config.max_retries,
                "batch_size": self.config.batch_size,
                "finnhub_enabled": bool(self.config.finnhub_api_key),
                "yahoo_finance_enabled": self.config.yahoo_finance_enabled,
                "alpha_vantage_enabled": bool(self.config.alpha_vantage_api_key),
            }
        }


def build_enrichment_service(config: Optional[EnrichmentConfig] = None) -> EventEnrichmentService:
    """Build and configure the event enrichment service."""
    return EventEnrichmentService(config)