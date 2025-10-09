from __future__ import annotations

from typing import Dict, Optional

from .utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    ProviderCircuitBreakers,
    circuit_breakers,
)


class CircuitBreakerManager:
    """Wrapper used by the provider registry to query breaker state."""

    def __init__(self, manager: Optional[ProviderCircuitBreakers] = None):
        self._manager = manager or circuit_breakers

    def get_state(self, provider_name: str) -> str:
        breaker = self._get_breaker(provider_name)
        return breaker.stats.state.name.upper()

    def get_stats(self) -> Dict[str, Dict]:
        return self._manager.get_all_stats()

    def _get_breaker(self, provider_name: str) -> CircuitBreaker:
        return self._manager.get_breaker(provider_name)


__all__ = ["CircuitBreakerManager", "CircuitState"]
