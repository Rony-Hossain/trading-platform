from abc import ABC, abstractmethod
from datetime import datetime
import logging
from typing import Dict, List, Optional

from ..utils.circuit_breaker import (
    CircuitOpenException,
    CircuitTimeoutException,
    circuit_breakers,
)

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Base class for data providers with circuit breaker protection."""

    def __init__(self, name: str):
        self.name = name
        self.is_available = True
        self.enabled = True
        self.last_error = None
        self.last_update = None
        self.circuit_breaker = circuit_breakers.get_breaker(name)

    @abstractmethod
    async def get_price(self, symbol: str) -> Optional[Dict]:
        """Get current stock price."""
        raise NotImplementedError

    @abstractmethod
    async def get_historical(self, symbol: str, period: str) -> Optional[Dict]:
        """Get historical stock data."""
        raise NotImplementedError

    async def get_intraday(self, symbol: str, interval: str = "1m") -> Optional[Dict]:
        """Get intraday stock data at a given interval. Optional to implement."""
        return None

    async def fetch_bars(self, symbol: str, start, end, interval: str) -> List[Dict]:
        """Return normalized bar list for registry consumers."""
        payload = await self.get_intraday(symbol, interval=interval)
        if not payload:
            return []

        data = payload.get("data") or []
        bars: List[Dict] = []
        for bar in data:
            ts = bar.get("timestamp") or bar.get("ts")
            if ts is None:
                continue
            bars.append(
                {
                    "symbol": symbol.upper(),
                    "ts": datetime.fromisoformat(str(ts)),
                    "interval": interval,
                    "o": float(bar.get("open", 0)),
                    "h": float(bar.get("high", 0)),
                    "l": float(bar.get("low", 0)),
                    "c": float(bar.get("close", 0)),
                    "v": float(bar.get("volume", 0)),
                    "status": bar.get("status", "final"),
                    "as_of": datetime.utcnow(),
                }
            )
        return bars

    def mark_unavailable(self, error: str):
        """Mark provider as unavailable."""
        self.is_available = False
        self.last_error = error
        self.last_update = datetime.now()

    def mark_available(self):
        """Mark provider as available."""
        self.is_available = True
        self.last_error = None
        self.last_update = datetime.now()

    async def get_price_safe(self, symbol: str) -> Optional[Dict]:
        """Get price with circuit breaker protection."""
        try:
            result = await self.circuit_breaker.call(self._get_price_impl, symbol)
            self.mark_available()
            return result
        except (CircuitOpenException, CircuitTimeoutException) as exc:
            self.mark_unavailable(str(exc))
            logger.warning("Provider %s failed for %s: %s", self.name, symbol, exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            self.mark_unavailable(str(exc))
            logger.error("Provider %s error for %s: %s", self.name, symbol, exc)
            return None

    async def get_historical_safe(self, symbol: str, period: str) -> Optional[Dict]:
        """Get historical data with circuit breaker protection."""
        try:
            result = await self.circuit_breaker.call(self._get_historical_impl, symbol, period)
            self.mark_available()
            return result
        except (CircuitOpenException, CircuitTimeoutException) as exc:
            self.mark_unavailable(str(exc))
            logger.warning("Provider %s failed for %s: %s", self.name, symbol, exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            self.mark_unavailable(str(exc))
            logger.error("Provider %s error for %s: %s", self.name, symbol, exc)
            return None

    async def get_intraday_safe(self, symbol: str, interval: str = "1m") -> Optional[Dict]:
        """Get intraday data with circuit breaker protection."""
        try:
            result = await self.circuit_breaker.call(self._get_intraday_impl, symbol, interval)
            self.mark_available()
            return result
        except (CircuitOpenException, CircuitTimeoutException) as exc:
            self.mark_unavailable(str(exc))
            logger.warning("Provider %s failed for %s: %s", self.name, symbol, exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive
            self.mark_unavailable(str(exc))
            logger.error("Provider %s error for %s: %s", self.name, symbol, exc)
            return None

    async def _get_price_impl(self, symbol: str) -> Optional[Dict]:
        """Internal implementation - calls the abstract method."""
        return await self.get_price(symbol)

    async def _get_historical_impl(self, symbol: str, period: str) -> Optional[Dict]:
        """Internal implementation - calls the abstract method."""
        return await self.get_historical(symbol, period)

    async def _get_intraday_impl(self, symbol: str, interval: str = "1m") -> Optional[Dict]:
        """Internal implementation - calls the optional method."""
        return await self.get_intraday(symbol, interval)

    def get_provider_stats(self) -> Dict:
        """Get provider statistics including circuit breaker metrics."""
        stats = self.circuit_breaker.get_stats()
        stats.update(
            {
                "is_available": self.is_available,
                "enabled": self.enabled,
                "last_error": self.last_error,
                "last_update": self.last_update.isoformat() if self.last_update else None,
            }
        )
        return stats
