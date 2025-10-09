from __future__ import annotations

import time
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ALLOWED_PROVIDERS = {"finnhub", "alphavantage", "yfinance"}


class ProviderCfg(BaseModel):
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_s: int = 8
    retries: int = 3


class Settings(BaseSettings):
    """Central configuration for the market data service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # === Legacy fields kept for backwards compatibility ===
    service_name: str = "market-data-service"
    service_port: int = 8001
    debug: bool = True

    database_url: str = "postgresql://trading_user:trading_pass@localhost:5432/trading_db"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    cache_ttl_seconds: int = 5
    store_historical_data: bool = True
    historical_data_retention_days: int = 365
    alpha_vantage_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    max_requests_per_minute: int = 300
    websocket_heartbeat_interval: int = 30
    websocket_update_interval: int = 5
    macro_refresh_interval_seconds: int = 900
    macro_cache_ttl_seconds: int = 300

    # === New policy/version controls ===
    env: str = Field(default="dev")
    policy_version: str = Field(default_factory=lambda: f"{int(time.time())}")

    # Feature flags
    USE_RLC: bool = False
    ALLOW_SOURCE_OVERRIDE: bool = True
    VIZTRACER_ENABLED: bool = False
    LOCAL_SWEEP_ENABLED: bool = True

    # Providers
    finnhub: ProviderCfg = Field(default_factory=ProviderCfg)
    alphavantage: ProviderCfg = Field(default_factory=ProviderCfg)
    yfinance_enabled: bool = True

    # Routing policy
    POLICY_BARS_1M: List[str] = Field(default_factory=lambda: ["finnhub", "alphavantage", "yfinance"])
    POLICY_EOD: List[str] = Field(default_factory=lambda: ["yfinance", "alphavantage", "finnhub"])
    POLICY_QUOTES_L1: List[str] = Field(default_factory=lambda: ["finnhub", "yfinance"])
    POLICY_OPTIONS_CHAIN: List[str] = Field(default_factory=lambda: ["yfinance", "finnhub"])

    # Circuit breaker / health score tuning
    BREAKER_DEMOTE_THRESHOLD: float = 0.55
    BREAKER_PROMOTE_THRESHOLD: float = 0.70
    BREAKER_PROMOTE_MIN_SAMPLES: int = 3
    BREAKER_COOLDOWN_SEC: int = 60
    RECENT_LATENCY_CAP_MS: int = 1000

    # Tier cadence and sizing
    LOCAL_SWEEP_BATCH: int = 200
    LOCAL_SWEEP_TICK_SEC: int = 30
    CADENCE_T0: Dict[str, int] = Field(default_factory=lambda: {"bars_1m": 60, "quotes_l1": 5})
    CADENCE_T1: Dict[str, int] = Field(default_factory=lambda: {"bars_1m": 300, "quotes_l1": 15})
    CADENCE_T2: Dict[str, int] = Field(default_factory=lambda: {"eod": 1})
    TIER_MAXS: Dict[str, int] = Field(default_factory=lambda: {"T0": 1200, "T1": 3000})

    # RLC / job ingestion
    RLC_BROKER: Literal["redis", "kafka"] = "redis"
    RLC_REDIS_URL: str = "redis://redis:6379/0"
    RLC_REDIS_JOBS_KEY: str = "market:jobs"
    RLC_REDIS_BACKFILL_KEYS: Dict[str, str] = Field(
        default_factory=lambda: {"T0": "market:backfills:T0", "T1": "market:backfills:T1", "T2": "market:backfills:T2"}
    )

    # Backfill controls
    BACKFILL_CHUNK_MINUTES: int = 1440
    BACKFILL_MAX_CONCURRENCY_T0: int = 4
    BACKFILL_MAX_CONCURRENCY_T1: int = 2
    BACKFILL_MAX_CONCURRENCY_T2: int = 1
    BACKFILL_MAX_QUEUE_T0: int = 20000
    BACKFILL_MAX_QUEUE_T1: int = 20000
    BACKFILL_MAX_QUEUE_T2: int = 40000
    BACKFILL_DISPATCH_RATE_PER_SEC: float = 2.0

    # Storage / PIT
    LIVE_BATCH_SIZE: int = 500
    BACKFILL_BATCH_SIZE: int = 5000

    # Websocket replay behaviour
    WS_REPLAY_MAX_BARS: int = 120
    WS_REPLAY_TTL_SEC: int = 4 * 3600

    @field_validator("POLICY_BARS_1M", "POLICY_EOD", "POLICY_QUOTES_L1", "POLICY_OPTIONS_CHAIN", mode="before")
    @classmethod
    def _validate_policy(cls, value: List[str]) -> List[str]:
        if isinstance(value, str):
            # Allow comma separated or JSON-style strings
            split = [v.strip() for v in value.strip("[]").split(",") if v.strip()]
            value = split
        if not isinstance(value, list):
            raise ValueError("policy must be provided as a list")
        invalid = [p for p in value if p not in _ALLOWED_PROVIDERS]
        if invalid:
            raise ValueError(f"unknown provider(s) in policy: {invalid}")
        return value


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def validate_settings(settings: Settings) -> Tuple[bool, str]:
    try:
        if not (0.0 <= settings.BREAKER_DEMOTE_THRESHOLD < settings.BREAKER_PROMOTE_THRESHOLD <= 1.0):
            return False, "invalid breaker thresholds"
        if not all(k in settings.RLC_REDIS_BACKFILL_KEYS for k in ("T0", "T1", "T2")):
            return False, "missing backfill keys for tiers"
        for cadence in (settings.CADENCE_T0, settings.CADENCE_T1):
            if any(v <= 0 for v in cadence.values()):
                return False, "cadence values must be positive"
        return True, "ok"
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)


def hot_reload() -> Dict[str, str | bool]:
    """Reload configuration from the environment in a safe manner."""
    global _settings
    try:
        candidate = Settings()
    except ValidationError as exc:  # pragma: no cover - config errors
        return {"ok": False, "error": f"pydantic validation failed: {exc}"}

    ok, reason = validate_settings(candidate)
    if not ok:
        return {"ok": False, "error": reason}

    _settings = candidate
    _settings.policy_version = f"{int(time.time())}"
    return {"ok": True, "policy_version": _settings.policy_version}


# Backwards compatible global export
settings = get_settings()
