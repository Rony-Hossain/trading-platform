import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import HTTPException

from ..circuit_breaker import CircuitBreakerManager
from ..core.config import get_settings
from ..providers import FinnhubProvider, YahooFinanceProvider
from ..providers.registry import ProviderRegistry
from .cache import DataCache
from .database import db_service
from .data_collector import DataCollectorService
from .macro_data_service import MacroFactorService
from .options_service import options_service
from .websocket import ConnectionManager

logger = logging.getLogger(__name__)

class MarketDataService:
    SUPPORTED_INTRADAY_INTERVALS = {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
    }

    def __init__(self):
        self.settings = get_settings()
        self.breakers = CircuitBreakerManager()
        self.registry = ProviderRegistry(self.breakers)

        self._init_providers()
        self.providers = [entry.adapter for entry in self.registry.providers.values()]

        # Cache TTL configurable via env/settings
        cache_ttl = int(os.getenv("CACHE_TTL_SECONDS", self.settings.cache_ttl_seconds))
        self.cache = DataCache(ttl_seconds=cache_ttl)
        self.metrics_cache = DataCache(ttl_seconds=int(os.getenv("OPTIONS_METRICS_CACHE_TTL", "300")))
        self.options_metrics_ttl = int(os.getenv("OPTIONS_METRICS_TTL", "900"))
        self.update_interval = int(os.getenv("WEBSOCKET_UPDATE_INTERVAL", "5"))
        self.connection_manager = ConnectionManager()
        self.macro_service = MacroFactorService(
            providers=self.providers,
            cache_ttl_seconds=getattr(self.settings, "macro_cache_ttl_seconds", 300),
            refresh_interval_seconds=getattr(self.settings, "macro_refresh_interval_seconds", 900),
        )
        self.data_collector = DataCollectorService(db=db_service, registry=self.registry)
        self.background_tasks_running = False

    def _init_providers(self) -> None:
        """Register provider adapters with the routing registry."""
        finnhub_key = (
            os.getenv("FINNHUB_API_KEY")
            or self.settings.finnhub.api_key
            or self.settings.finnhub_api_key
        )

        if finnhub_key and finnhub_key not in {"your_finnhub_api_key_here", ""}:
            try:
                finnhub_provider = FinnhubProvider(api_key=finnhub_key)
                self.registry.register("finnhub", finnhub_provider, {"bars_1m", "eod", "quotes_l1"})
                logger.info("Finnhub provider registered")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to initialize Finnhub provider: %s", exc)
        else:
            logger.warning("No valid Finnhub API key provided; Finnhub disabled")

        if self.settings.yfinance_enabled:
            try:
                yahoo_provider = YahooFinanceProvider()
                self.registry.register("yfinance", yahoo_provider, {"bars_1m", "eod", "options_chain"})
                logger.info("Yahoo Finance provider registered")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to initialize Yahoo Finance provider: %s", exc)

    def reload_configuration(self) -> None:
        """Reload providers and settings after a hot reload."""
        self.settings = get_settings()
        self.registry.providers.clear()
        self._init_providers()
        self.providers = [entry.adapter for entry in self.registry.providers.values()]
        if hasattr(self.macro_service, "providers"):
            self.macro_service.providers = self.providers

    async def _try_providers(
        self,
        *,
        capability: str,
        endpoint: str,
        executor,
        provider_hint: Optional[str] = None,
    ):
        ranked = self.registry.rank(capability, provider_hint=provider_hint)
        if not ranked:
            raise HTTPException(status_code=503, detail=f"No providers available for {capability}")

        last_error: Optional[str] = None
        for provider_name in ranked:
            entry = self.registry.providers[provider_name]
            adapter = entry.adapter
            if not adapter.enabled:
                continue

            self.registry.record_selection(capability, provider_name)
            started = time.perf_counter()

            try:
                result = await executor(adapter)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.registry.record_outcome(provider_name, elapsed_ms, error=False, endpoint=endpoint)
                if result:
                    return result, provider_name
                last_error = f"{provider_name}: empty result"
            except HTTPException:
                raise
            except Exception as exc:  # pragma: no cover - provider failure paths
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.registry.record_outcome(provider_name, elapsed_ms, error=True, endpoint=endpoint)
                self.registry.record_error(provider_name, endpoint, exc.__class__.__name__)
                last_error = f"{provider_name}: {exc}"

        raise HTTPException(
            status_code=502,
            detail=f"All providers failed for {endpoint}. last_error={last_error}",
        )
    
    async def get_stock_price(self, symbol: str) -> Dict:
        """Get stock price with caching and health-aware routing."""
        symbol = symbol.upper()
        cache_key = f"price:{symbol}"

        cached_data = self.cache.get(cache_key)
        if cached_data:
            logger.info("Cache hit for %s", symbol)
            return cached_data

        data, provider_used = await self._try_providers(
            capability="quotes_l1",
            endpoint="quotes",
            executor=lambda adapter: adapter.get_price_safe(symbol),
        )

        if isinstance(data, dict):
            data["provider_used"] = provider_used
        self.cache.set(cache_key, data)
        logger.info("Fetched %s from %s", symbol, provider_used)
        return data
    
    async def get_historical_data(self, symbol: str, period: str = "1mo") -> Dict:
        """Get historical data."""
        symbol = symbol.upper()
        data, provider_used = await self._try_providers(
            capability="eod",
            endpoint="historical",
            executor=lambda adapter: adapter.get_historical_safe(symbol, period),
        )

        if isinstance(data, dict):
            data["provider_used"] = provider_used
        return data

    def _normalize_intraday_interval(self, interval: str) -> str:
        interval_normalized = (interval or "1m").strip().lower()

        # Map common aliases to supported values
        alias_map = {
            "1": "1m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "60m",
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "60min": "60m",
            "1hour": "60m",
            "1hr": "60m",
            "1h": "60m",
        }

        resolved = alias_map.get(interval_normalized, interval_normalized)
        if resolved not in self.SUPPORTED_INTRADAY_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported intraday interval '{interval}'"
            )
        return resolved

    async def get_intraday_data(self, symbol: str, interval: str = "1m") -> Dict:
        """Get intraday data with provider fallback and validation."""
        normalized_interval = self._normalize_intraday_interval(interval)

        capability = "bars_1m" if normalized_interval.endswith("m") else "eod"
        data, provider_used = await self._try_providers(
            capability=capability,
            endpoint="intraday",
            executor=lambda adapter: adapter.get_intraday_safe(symbol, normalized_interval),
        )

        if isinstance(data, dict):
            data["provider_used"] = provider_used
        return data
    
    async def start_background_tasks(self):
        """Start background tasks for cache cleanup and real-time updates"""
        if self.background_tasks_running:
            return

        self.background_tasks_running = True

        # Initialize database connections for macro storage
        try:
            await db_service.initialize()
            logger.info("Database service initialized")
        except Exception as exc:
            logger.error(f"Failed to initialize database service: {exc}")

        # Start cache cleanup task
        asyncio.create_task(self._cache_cleanup_task())

        # Start macro refresh task
        asyncio.create_task(self._macro_refresh_task())

        # Start real-time data broadcasting
        asyncio.create_task(self._real_time_broadcast_task())

        # Start data collector (M2/M3: RLC consumer + gap detection + backfill)
        asyncio.create_task(self.data_collector.run())

        logger.info("Background tasks started (including data collector)")
    
    async def _cache_cleanup_task(self):
        """Clean up expired cache entries"""
        while self.background_tasks_running:
            self.cache.clear_expired()
            await asyncio.sleep(60)  # Clean up every minute
    
    async def _real_time_broadcast_task(self):
        """Broadcast real-time data to WebSocket clients"""
        while self.background_tasks_running:
            if self.connection_manager.symbol_subscribers:
                for symbol in list(self.connection_manager.symbol_subscribers.keys()):
                    try:
                        data = await self.get_stock_price(symbol)
                        await self.connection_manager.broadcast_to_symbol(symbol, data)
                    except Exception as e:
                        logger.error(f"Error broadcasting {symbol}: {e}")

            await asyncio.sleep(self.update_interval)  # Update interval configurable

    async def _macro_refresh_task(self):
        """Periodically refresh macro factors from configured providers."""
        interval = getattr(
            self.macro_service,
            "refresh_interval_seconds",
            getattr(self.settings, "macro_refresh_interval_seconds", 900),
        )
        interval = max(60, interval or 900)

        try:
            await self.macro_service.refresh_all()
        except Exception as exc:
            logger.error(f"Initial macro refresh failed: {exc}")

        while self.background_tasks_running:
            await asyncio.sleep(interval)
            try:
                await self.macro_service.refresh_all()
            except Exception as exc:
                logger.error(f"Macro refresh task failed: {exc}")

    async def get_company_profile(self, symbol: str) -> Dict:
        """Get company profile data (Finnhub only)"""
        for provider in self.providers:
            if hasattr(provider, 'get_company_profile'):
                data = await provider.get_company_profile(symbol)
                if data:
                    return data
        
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch company profile for {symbol}"
        )
    
    async def get_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment data (Finnhub only)"""
        for provider in self.providers:
            if hasattr(provider, 'get_news_sentiment'):
                data = await provider.get_news_sentiment(symbol)
                if data:
                    return data
        
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch news sentiment for {symbol}"
        )
    

    async def get_options_metrics(self, symbol: str) -> Dict:
        """Return cached or freshly computed options metrics for a symbol."""
        symbol = symbol.upper()
        cache_key = f"options_metrics:{symbol}"
        cached = self.metrics_cache.get(cache_key)
        if cached:
            return cached

        latest = await db_service.get_latest_options_metrics(symbol)
        if latest:
            try:
                timestamp = datetime.fromisoformat(latest['as_of'])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - timestamp).total_seconds() < self.options_metrics_ttl:
                    self.metrics_cache.set(cache_key, latest)
                    return latest
            except Exception:
                pass

        chain = await options_service.fetch_options_chain(symbol)
        metrics = options_service.calculate_chain_metrics(chain)
        record = metrics.to_db_record()
        await db_service.store_options_metrics(record)
        result = metrics.to_dict()
        result['metadata'].setdefault('source', 'provider')
        self.metrics_cache.set(cache_key, result)
        return result

    async def get_options_metrics_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Return recent history of stored options metrics for a symbol."""
        history = await db_service.get_options_metrics_history(symbol.upper(), limit)
        return history

    async def get_stats(self) -> Dict:
        macro_stats = await self.macro_service.stats()
        return {
            "providers": [
                {
                    "name": provider.name,
                    "available": provider.is_available,
                    "last_error": provider.last_error,
                }
                for provider in self.providers
            ],
            "cache": self.cache.stats(),
            "websocket": self.connection_manager.stats(),
            "database": {
                "connected": db_service.pool is not None,
                "storage_enabled": self.settings.store_historical_data,
            },
            "macro": macro_stats,
            "options_metrics": {
                "cache": self.metrics_cache.stats(),
                "ttl_seconds": self.options_metrics_ttl,
            },
        }

    async def get_macro_snapshot(self, factors: Optional[List[str]] = None) -> Dict:
        """Return the latest macro factor snapshot."""
        return await self.macro_service.get_snapshot(factors)

    async def get_macro_history(self, factor_key: str, lookback_days: int = 30) -> Dict:
        """Return macro factor history for the requested lookback window."""
        return await self.macro_service.get_history(factor_key, lookback_days)

    def list_macro_factors(self) -> List[str]:
        """List available macro factor keys."""
        return self.macro_service.available_factors()

    async def refresh_macro_factors(self, factor_key: Optional[str] = None) -> Dict:
        """Refresh macro factors from providers."""
        if factor_key:
            return await self.macro_service.refresh_factor(factor_key)
        return await self.macro_service.refresh_all()
    
    async def get_unusual_options_activity(self, symbol: str, lookback_days: int = 20) -> Dict:
        """Get unusual options activity for a symbol"""
        try:
            unusual_activities = await options_service.detect_unusual_activity(symbol.upper(), lookback_days)
            
            return {
                "symbol": symbol.upper(),
                "lookback_days": lookback_days,
                "unusual_activities_count": len(unusual_activities),
                "unusual_activities": [
                    {
                        "contract_symbol": activity.contract_symbol,
                        "strike": activity.strike,
                        "expiry": activity.expiry.isoformat(),
                        "option_type": activity.option_type,
                        "volume": activity.volume,
                        "avg_volume_20d": activity.avg_volume_20d,
                        "volume_ratio": activity.volume_ratio,
                        "open_interest": activity.open_interest,
                        "volume_spike": activity.volume_spike,
                        "large_single_trades": activity.large_single_trades,
                        "sweep_activity": activity.sweep_activity,
                        "unusual_volume_vs_oi": activity.unusual_volume_vs_oi,
                        "underlying_price": activity.underlying_price,
                        "strike_distance_pct": activity.strike_distance_pct,
                        "days_to_expiration": activity.days_to_expiration,
                        "unusual_score": activity.unusual_score,
                        "confidence_level": activity.confidence_level,
                        "large_trades": activity.large_trades,
                        "timestamp": activity.timestamp.isoformat()
                    }
                    for activity in unusual_activities
                ],
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting unusual options activity for {symbol}: {e}")
            return {
                "symbol": symbol.upper(),
                "error": str(e),
                "unusual_activities_count": 0,
                "unusual_activities": []
            }
    
    async def get_options_flow_analysis(self, symbol: str) -> Dict:
        """Get comprehensive options flow analysis"""
        try:
            flow_analysis = await options_service.analyze_options_flow(symbol.upper())
            
            return {
                "symbol": symbol.upper(),
                "timestamp": flow_analysis.timestamp.isoformat(),
                "flow_metrics": {
                    "total_call_volume": flow_analysis.total_call_volume,
                    "total_put_volume": flow_analysis.total_put_volume,
                    "call_put_ratio": flow_analysis.call_put_ratio,
                    "large_trades_count": flow_analysis.large_trades_count,
                    "block_trades_value": flow_analysis.block_trades_value,
                    "sweep_trades_count": flow_analysis.sweep_trades_count
                },
                "sentiment_analysis": {
                    "flow_sentiment": flow_analysis.flow_sentiment,
                    "smart_money_score": flow_analysis.smart_money_score,
                    "call_premium_bought": flow_analysis.call_premium_bought,
                    "put_premium_bought": flow_analysis.put_premium_bought,
                    "net_premium_flow": flow_analysis.net_premium_flow
                },
                "unusual_activities_summary": {
                    "count": len(flow_analysis.unusual_activities),
                    "top_activities": [
                        {
                            "contract_symbol": activity.contract_symbol,
                            "unusual_score": activity.unusual_score,
                            "volume_ratio": activity.volume_ratio,
                            "volume": activity.volume,
                            "strike": activity.strike,
                            "option_type": activity.option_type
                        }
                        for activity in flow_analysis.unusual_activities[:5]  # Top 5
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error getting options flow analysis for {symbol}: {e}")
            return {
                "symbol": symbol.upper(),
                "error": str(e),
                "flow_metrics": {},
                "sentiment_analysis": {},
                "unusual_activities_summary": {"count": 0, "top_activities": []}
            }

