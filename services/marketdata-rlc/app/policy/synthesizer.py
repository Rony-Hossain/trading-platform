from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBucketParams:
    refill_rate: float
    burst: int
    jitter_ms: int
    ttl_s: int


@dataclass(frozen=True)
class TierQuotaParams:
    t0_max: int
    t1_max: int
    t2_mode: str


class PolicySynthesizer:
    """Convert target request rates and guardrails into concrete policies."""

    def __init__(
        self,
        min_refill: float = 0.5,
        max_refill: float = 500.0,
        min_burst_sec: float = 0.5,
        max_burst_sec: float = 3.0,
        jitter_cap_ms: int = 2000,
    ) -> None:
        self.min_refill = min_refill
        self.max_refill = max_refill
        self.min_burst_sec = min_burst_sec
        self.max_burst_sec = max_burst_sec
        self.jitter_cap_ms = jitter_cap_ms

    def token_bucket_from_rate(
        self,
        target_rps: float,
        desired_burst_seconds: float = 1.5,
        jitter_fraction: float = 0.1,
        ttl_s: int = 60,
    ) -> TokenBucketParams:
        rps = max(self.min_refill, min(self.max_refill, target_rps))
        burst_seconds = max(self.min_burst_sec, min(self.max_burst_sec, desired_burst_seconds))
        burst_tokens = max(1, math.ceil(rps * burst_seconds))

        if rps <= 0:
            interarrival_ms = 1000.0
        else:
            interarrival_ms = 1000.0 / rps
        jitter_ms = int(min(self.jitter_cap_ms, max(0.0, jitter_fraction * interarrival_ms)))

        return TokenBucketParams(
            refill_rate=rps,
            burst=burst_tokens,
            jitter_ms=jitter_ms,
            ttl_s=max(10, ttl_s),
        )

    def quotas_from_budget(
        self,
        t0_demand: int,
        t1_demand: int,
        budget_breach: bool,
        min_t0_floor: int = 50,
    ) -> TierQuotaParams:
        if budget_breach:
            t0 = max(min_t0_floor, min(t0_demand, min_t0_floor))
            t1 = max(0, math.floor(t1_demand * 0.6))
            t2_mode = "EOD"
        else:
            t0 = t0_demand
            t1 = t1_demand
            t2_mode = "60s"

        return TierQuotaParams(t0_max=t0, t1_max=t1, t2_mode=t2_mode)
