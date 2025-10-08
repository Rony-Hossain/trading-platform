from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration loaded from environment variables."""

    app_name: str = Field(default="marketdata-rlc", description="Service display name")
    env: str = Field(default="dev", description="Deployment environment name")
    mode: str = Field(
        default="shadow",
        description="Operating mode for policies (shadow|enforced)",
    )

    # Kafka
    kafka_bootstrap: str = Field(default="localhost:9092")
    kafka_topic_updates: str = Field(default="policy.updates")
    kafka_topic_shadow: str = Field(default="policy.shadow")

    # Database
    pg_dsn: str = Field(
        default="postgres://postgres:postgres@localhost:5432/marketdata",
        description="Timescale/PostgreSQL connection string",
    )

    # Model artefacts
    model_path: str = Field(default="models/rlc_latency.onnx")
    error_model_path: str = Field(default="models/rlc_error.onnx")

    # Policy loop
    providers: List[str] = Field(
        default_factory=lambda: ["providerA"],
        description="Providers managed by the RLC",
    )
    loop_interval_s: int = Field(default=60, description="Policy loop cadence in seconds")
    p95_slo_ms: int = Field(default=200, description="Latency SLO used for decisions")
    default_rps: float = Field(default=120.0, description="Starting request rate per provider")
    budget_usd_per_min: float = Field(
        default=0.5, description="Budget envelope (USD per minute) per provider"
    )
    min_t0_floor: int = Field(default=50, description="Never degrade T0 below this symbol count")

    # Bandit configuration
    bandit_enabled: bool = True
    bandit_arms: List[tuple[int, int]] = Field(
        default_factory=lambda: [
            (50, 0),
            (100, 100),
            (150, 120),
            (200, 150),
            (250, 200),
        ],
        description="List of (batch_size, inter_batch_delay_ms) arms",
    )
    error_constraint: float = Field(
        default=0.02, description="Maximum acceptable error probability"
    )

    # Feedback job
    feedback_enabled: bool = True
    feedback_interval_s: int = Field(default=60)
    feedback_lookback_min: int = Field(default=2)

    # Observability
    prometheus_enabled: bool = True

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": ("settings_",),
    }


settings = Settings()
