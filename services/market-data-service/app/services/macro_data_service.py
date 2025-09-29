import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .cache import DataCache
from .database import db_service

logger = logging.getLogger(__name__)

DEFAULT_MACRO_SERIES: Dict[str, Dict] = {
    "VIX": {
        "display_name": "CBOE Volatility Index",
        "category": "volatility",
        "unit": "index",
        "finnhub_symbols": ["^VIX", "VIX"],
        "yahoo_finance_symbols": ["^VIX"],
        "expected_interval_minutes": 15,
    },
    "US10Y": {
        "display_name": "US 10Y Treasury Yield",
        "category": "rates",
        "unit": "percent",
        "finnhub_symbols": ["US10Y", "^US10Y"],
        "yahoo_finance_symbols": ["^TNX"],
        "expected_interval_minutes": 60,
    },
    "US02Y": {
        "display_name": "US 2Y Treasury Yield",
        "category": "rates",
        "unit": "percent",
        "finnhub_symbols": ["US02Y", "^US02Y"],
        "yahoo_finance_symbols": ["US2Y=RR", "^IRX"],
        "expected_interval_minutes": 60,
    },
    "EURUSD": {
        "display_name": "EUR/USD Spot",
        "category": "fx",
        "unit": "ratio",
        "finnhub_symbols": ["OANDA:EUR_USD", "EURUSD"],
        "yahoo_finance_symbols": ["EURUSD=X"],
        "expected_interval_minutes": 15,
    },
    "WTI": {
        "display_name": "WTI Crude Oil",
        "category": "commodities",
        "unit": "usd",
        "finnhub_symbols": ["OANDA:WTICOUSD", "WTICOUSD"],
        "yahoo_finance_symbols": ["CL=F"],
        "expected_interval_minutes": 30,
    },
}


class MacroFactorService:
    """Collects and stores macro / cross-asset factors for downstream services."""

    def __init__(
        self,
        providers: List,
        cache_ttl_seconds: int = 300,
        refresh_interval_seconds: int = 900,
        series_config: Optional[Dict[str, Dict]] = None,
    ) -> None:
        self.providers = providers
        self.cache = DataCache(ttl_seconds=cache_ttl_seconds)
        self.refresh_interval_seconds = refresh_interval_seconds
        self.series_config = series_config or DEFAULT_MACRO_SERIES
        self.factor_status: Dict[str, Dict] = {
            key: {
                "last_success": None,
                "last_error": None,
                "last_source": None,
                "last_refresh": None,
            }
            for key in self.series_config
        }

    def available_factors(self) -> List[str]:
        return list(self.series_config.keys())

    async def get_snapshot(self, factors: Optional[List[str]] = None) -> Dict:
        keys = [key.upper() for key in (factors or self.available_factors()) if key]
        response = []

        for key in keys:
            config = self.series_config.get(key)
            if not config:
                response.append({
                    "key": key,
                    "error": "unknown_factor",
                })
                continue

            cached = self.cache.get(f"macro:{key}")
            if cached is None:
                cached = await self._load_from_storage(key)

            if cached is None:
                refresh = await self.refresh_factor(key)
                cached = refresh.get("data") if refresh else None

            if cached is None:
                status = self.factor_status.get(key, {})
                response.append({
                    "key": key,
                    "display_name": config.get("display_name"),
                    "category": config.get("category"),
                    "unit": config.get("unit"),
                    "error": status.get("last_error", "no_data"),
                    "last_refresh": status.get("last_refresh"),
                })
                continue

            response.append(cached)

        return {
            "as_of": datetime.utcnow().isoformat(),
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "factors": response,
        }

    async def get_history(
        self,
        factor_key: str,
        lookback_days: int = 30,
        end: Optional[datetime] = None,
    ) -> Dict:
        key = factor_key.upper()
        config = self.series_config.get(key)
        if not config:
            raise ValueError(f"Unknown macro factor: {factor_key}")

        end_time = end or datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)

        try:
            rows = await db_service.get_macro_series(
                key,
                start_date=start_time,
                end_date=end_time,
            )
        except Exception as exc:
            logger.error("Failed to load macro history for %s: %s", key, exc)
            rows = []

        if not rows:
            await self.refresh_factor(key)
            rows = await db_service.get_macro_series(key, start_date=start_time, end_date=end_time)

        return {
            "factor": key,
            "display_name": config.get("display_name"),
            "unit": config.get("unit"),
            "category": config.get("category"),
            "data": rows,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        }

    async def refresh_all(self) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        for key in self.available_factors():
            results[key] = await self.refresh_factor(key)
        return results

    async def refresh_factor(self, factor_key: str) -> Dict:
        key = factor_key.upper()
        config = self.series_config.get(key)
        if not config:
            return {"factor": key, "error": "unknown_factor"}

        for provider in self.providers:
            provider_key = provider.name.lower().replace(" ", "_")
            symbol_candidates = config.get(f"{provider_key}_symbols", [])
            if not symbol_candidates:
                continue

            for candidate in symbol_candidates:
                try:
                    data = await provider.get_price(candidate)
                except Exception as exc:  # pragma: no cover - provider failure path
                    logger.warning("Provider %s failed for %s (%s): %s", provider.name, key, candidate, exc)
                    continue

                if not data or data.get("price") is None:
                    continue

                snapshot = self._build_snapshot(key, config, data, provider_name=provider.name, symbol_used=candidate)
                await self._persist_point(key, snapshot)
                self.cache.set(f"macro:{key}", snapshot)
                self.factor_status[key] = {
                    "last_success": snapshot["timestamp"],
                    "last_error": None,
                    "last_source": snapshot.get("source"),
                    "last_refresh": datetime.utcnow().isoformat(),
                }
                return {"factor": key, "data": snapshot}

        error_msg = "no_provider_data"
        self.factor_status[key] = {
            "last_success": self.factor_status.get(key, {}).get("last_success"),
            "last_error": error_msg,
            "last_source": None,
            "last_refresh": datetime.utcnow().isoformat(),
        }
        return {"factor": key, "error": error_msg}

    async def stats(self) -> Dict:
        response = {
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "factors": {},
        }

        for key, config in self.series_config.items():
            latest = await db_service.get_latest_macro_point(key)
            status = self.factor_status.get(key, {})
            response["factors"][key] = {
                "display_name": config.get("display_name"),
                "category": config.get("category"),
                "unit": config.get("unit"),
                "latest": latest,
                "last_success": status.get("last_success"),
                "last_error": status.get("last_error"),
                "last_source": status.get("last_source"),
            }

        return response

    async def _load_from_storage(self, factor_key: str) -> Optional[Dict]:
        latest = await db_service.get_latest_macro_point(factor_key)
        if not latest:
            return None

        config = self.series_config.get(factor_key, {})
        snapshot = {
            "key": factor_key,
            "display_name": config.get("display_name"),
            "category": config.get("category"),
            "unit": config.get("unit"),
            "value": float(latest["value"]),
            "timestamp": latest["timestamp"],
            "source": latest.get("source"),
            "metadata": latest.get("metadata", {}),
        }
        self.cache.set(f"macro:{factor_key}", snapshot)
        return snapshot

    async def _persist_point(self, factor_key: str, snapshot: Dict) -> None:
        point = {
            "timestamp": datetime.fromisoformat(snapshot["timestamp"]),
            "value": snapshot["value"],
            "source": snapshot.get("source"),
            "metadata": snapshot.get("metadata", {}),
        }
        await db_service.store_macro_points(factor_key, [point])

    def _build_snapshot(
        self,
        factor_key: str,
        config: Dict,
        provider_data: Dict,
        provider_name: str,
        symbol_used: str,
    ) -> Dict:
        timestamp = provider_data.get("timestamp")
        if isinstance(timestamp, datetime):
            ts_iso = timestamp.isoformat()
        elif isinstance(timestamp, str):
            try:
                ts_iso = datetime.fromisoformat(timestamp).isoformat()
            except ValueError:
                ts_iso = datetime.utcnow().isoformat()
        else:
            ts_iso = datetime.utcnow().isoformat()

        value = float(provider_data.get("price"))
        metadata = {
            "change": provider_data.get("change"),
            "change_percent": provider_data.get("change_percent"),
            "high": provider_data.get("high"),
            "low": provider_data.get("low"),
            "open": provider_data.get("open"),
            "previous_close": provider_data.get("previous_close"),
            "provider_symbol": symbol_used,
        }

        return {
            "key": factor_key,
            "display_name": config.get("display_name"),
            "category": config.get("category"),
            "unit": config.get("unit"),
            "value": value,
            "timestamp": ts_iso,
            "source": provider_data.get("source", provider_name),
            "metadata": {k: v for k, v in metadata.items() if v is not None},
        }
