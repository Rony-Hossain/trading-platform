from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import PlainTextResponse
import yaml

from ..core.config import settings
from ..obs.metrics import rlc_policy_updates, rlc_shadow_delta_ms
from ..schemas.policy import PolicyBundle, ShadowDiffResponse, TierQuota, TokenPolicy

router = APIRouter()


@router.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name, "mode": settings.mode}


@router.get("/readyz", tags=["ops"])
def readyz() -> dict[str, bool]:
    # TODO: add real readiness checks (Kafka, Timescale, models)
    return {"ready": True}


@router.post(
    "/providers/{provider}/token-policy",
    tags=["policy"],
)
def set_token_policy(
    provider: str,
    policy: TokenPolicy = Body(
        ...,
        examples={
            "standard": {
                "summary": "Standard policy",
                "value": {"refill_rate": 120.0, "burst": 240, "jitter_ms": 8, "ttl_s": 60},
            }
        },
    ),
) -> dict[str, PolicyBundle]:
    bundle = PolicyBundle(
        provider=provider,
        token_policy=policy,
        tier_quota=TierQuota(t0_max=100, t1_max=1000, t2_mode="60s"),
        mode=settings.mode,
    )
    rlc_policy_updates.labels(settings.mode, provider).inc()
    return {"applied": bundle}


@router.post(
    "/tiers/quota",
    tags=["policy"],
)
def set_tier_quota(
    quota: TierQuota = Body(
        ...,
        examples={
            "normal": {"summary": "Normal operation", "value": {"t0_max": 150, "t1_max": 900, "t2_mode": "60s"}},
            "degraded": {"summary": "Budget breach", "value": {"t0_max": 50, "t1_max": 540, "t2_mode": "EOD"}},
        },
    ),
) -> dict[str, TierQuota]:
    return {"applied": quota}


@router.get("/shadow/diff", response_model=ShadowDiffResponse, tags=["policy"])
def shadow_diff(provider: str = Query(..., examples=["providerA"])) -> ShadowDiffResponse:
    resp = ShadowDiffResponse(
        provider=provider,
        p95_baseline_ms=180,
        p95_shadow_ms=175,
        delta_ms=-5,
        error_rate_baseline=0.012,
        error_rate_shadow=0.011,
        cost_per_min_baseline=0.45,
        cost_per_min_shadow=0.43,
    )
    rlc_shadow_delta_ms.labels(provider).set(resp.delta_ms)
    return resp


@router.get("/openapi.yaml", response_class=PlainTextResponse, tags=["ops"])
def openapi_yaml(request: Request) -> str:
    """Expose OpenAPI definition in YAML form."""
    openapi_dict = request.app.openapi()
    return yaml.safe_dump(openapi_dict, sort_keys=False)
