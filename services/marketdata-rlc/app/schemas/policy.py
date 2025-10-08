from typing import Literal, Optional

from pydantic import BaseModel, Field, conint, confloat


class TokenPolicy(BaseModel):
    refill_rate: confloat(ge=0) = Field(..., description="Tokens per second", examples=[120.0])
    burst: conint(ge=0) = Field(0, description="Burst capacity in tokens", examples=[240])
    jitter_ms: conint(ge=0) = Field(0, description="Jitter to randomise refill timing", examples=[8])
    ttl_s: conint(ge=1) = Field(60, description="Policy TTL seconds", examples=[60])


class TierQuota(BaseModel):
    t0_max: conint(ge=0) = Field(..., description="Tier 0 symbol cap", examples=[150])
    t1_max: conint(ge=0) = Field(..., description="Tier 1 symbol cap", examples=[900])
    t2_mode: Literal["60s", "EOD"] = Field(..., description="Tier 2 mode", examples=["60s"])


class BatchHints(BaseModel):
    batch_size: conint(ge=1) = Field(..., description="Recommended batch size", examples=[150])
    inter_batch_delay_ms: conint(ge=0) = Field(
        ..., description="Delay between batches in milliseconds", examples=[120]
    )


class PolicyBundle(BaseModel):
    provider: str = Field(..., examples=["providerA"])
    token_policy: TokenPolicy
    tier_quota: TierQuota
    mode: Literal["shadow", "enforced"] = Field("shadow", examples=["shadow"])
    batch_hints: Optional[BatchHints] = Field(
        default=None, examples=[{"batch_size": 150, "inter_batch_delay_ms": 120}]
    )


class ShadowDiffResponse(BaseModel):
    provider: str
    p95_baseline_ms: int
    p95_shadow_ms: int
    delta_ms: int
    error_rate_baseline: float
    error_rate_shadow: float
    cost_per_min_baseline: float
    cost_per_min_shadow: float
