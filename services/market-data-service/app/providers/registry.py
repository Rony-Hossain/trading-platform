from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from app.circuit_breaker import CircuitBreakerManager
from app.core.config import get_settings
from app.observability.metrics import (
    CIRCUIT_STATE,
    PROVIDER_ERRORS,
    PROVIDER_LATENCY,
    PROVIDER_SELECTED,
)
from app.providers.base import DataProvider


@dataclass
class RollingStats:
    """EWMA approximations for routing inputs."""

    alpha: float = 0.3
    latency_p95_ms: float = 0.0
    error_ewma: float = 0.0
    completeness_deficit: float = 0.0

    def update_latency(self, sample_ms: float) -> None:
        self.latency_p95_ms = (1 - self.alpha) * self.latency_p95_ms + self.alpha * sample_ms

    def update_error(self, is_error: bool) -> None:
        x = 1.0 if is_error else 0.0
        self.error_ewma = (1 - self.alpha) * self.error_ewma + self.alpha * x

    def set_completeness(self, deficit_0_1: float) -> None:
        self.completeness_deficit = max(0.0, min(1.0, deficit_0_1))


@dataclass
class ProviderEntry:
    name: str
    adapter: DataProvider
    capabilities: Set[str]
    stats: RollingStats = field(default_factory=RollingStats)
    h_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=60))


class ProviderRegistry:
    """Runtime registry that ranks providers per capability using deterministic policies."""

    def __init__(self, breakers: CircuitBreakerManager):
        self.breakers = breakers
        self.providers: Dict[str, ProviderEntry] = {}

    def register(self, name: str, adapter: DataProvider, capabilities: Set[str]) -> None:
        adapter.enabled = True
        self.providers[name] = ProviderEntry(name=name, adapter=adapter, capabilities=capabilities)

    def capabilities_map(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for name, entry in self.providers.items():
            for cap in entry.capabilities:
                out.setdefault(cap, []).append(name)
        return out

    def _breaker_score(self, name: str) -> float:
        state = self.breakers.get_state(name)
        if state == "CLOSED":
            return 1.0
        if state == "HALF_OPEN":
            return 0.4
        return 0.0

    def health_score(self, name: str) -> float:
        settings = get_settings()
        entry = self.providers[name]
        breaker_component = self._breaker_score(name)
        latency_cap = float(settings.RECENT_LATENCY_CAP_MS)
        latency_norm = min(entry.stats.latency_p95_ms / latency_cap, 1.0) if latency_cap > 0 else 1.0
        error_component = entry.stats.error_ewma
        completeness_component = entry.stats.completeness_deficit
        score = (
            0.45 * breaker_component
            + 0.25 * (1.0 - latency_norm)
            + 0.20 * (1.0 - error_component)
            + 0.10 * (1.0 - completeness_component)
        )
        return max(0.0, min(1.0, score))

    def rank(self, capability: str, provider_hint: Optional[str] = None) -> List[str]:
        settings = get_settings()
        policy_key = {
            "bars_1m": "POLICY_BARS_1M",
            "eod": "POLICY_EOD",
            "quotes_l1": "POLICY_QUOTES_L1",
            "options_chain": "POLICY_OPTIONS_CHAIN",
        }.get(capability)
        if not policy_key:
            return []

        base_policy = list(getattr(settings, policy_key, []))

        candidates: List[Tuple[str, float]] = []
        for provider_name in base_policy:
            entry = self.providers.get(provider_name)
            if not entry:
                continue
            if capability not in entry.capabilities:
                continue
            if not entry.adapter.enabled:
                continue
            candidates.append((provider_name, self.health_score(provider_name)))

        if provider_hint and settings.ALLOW_SOURCE_OVERRIDE:
            hint = next((c for c in candidates if c[0] == provider_hint), None)
            if hint:
                state = self.breakers.get_state(provider_hint)
                if hint[1] >= settings.BREAKER_DEMOTE_THRESHOLD and state != "OPEN":
                    candidates.sort(key=lambda kv: 0 if kv[0] == provider_hint else 1)

        base_index = {name: idx for idx, name in enumerate(base_policy)}
        candidates.sort(key=lambda kv: (-kv[1], base_index.get(kv[0], len(base_policy))))

        now = time.time()
        ranked = []
        for name, score in candidates:
            ranked.append(name)
            entry = self.providers[name]
            entry.h_history.append((now, score))
        return ranked

    def record_selection(self, capability: str, provider_name: str) -> None:
        PROVIDER_SELECTED.labels(capability=capability, provider=provider_name).inc()
        state = self.breakers.get_state(provider_name)
        CIRCUIT_STATE.labels(provider=provider_name).set(
            1.0 if state == "CLOSED" else 0.5 if state == "HALF_OPEN" else 0.0
        )

    def record_outcome(self, name: str, latency_ms: float, error: bool, endpoint: str = "generic") -> None:
        entry = self.providers.get(name)
        if not entry:
            return
        entry.stats.update_latency(latency_ms)
        entry.stats.update_error(error)
        PROVIDER_LATENCY.labels(provider=name, endpoint=endpoint).observe(latency_ms)
        state = self.breakers.get_state(name)
        CIRCUIT_STATE.labels(provider=name).set(
            1.0 if state == "CLOSED" else 0.5 if state == "HALF_OPEN" else 0.0
        )
        entry.h_history.append((time.time(), self.health_score(name)))

    def record_error(self, provider: str, endpoint: str, code: str) -> None:
        PROVIDER_ERRORS.labels(provider=provider, endpoint=endpoint, code=code).inc()

    def set_completeness(self, name: str, deficit_0_1: float) -> None:
        entry = self.providers.get(name)
        if not entry:
            return
        entry.stats.set_completeness(deficit_0_1)
