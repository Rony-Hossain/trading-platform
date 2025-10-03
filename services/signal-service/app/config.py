"""
Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Service Info
    SERVICE_NAME: str = "signal-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0

    # Upstream Services
    INFERENCE_SERVICE_URL: str = "http://localhost:8001"
    FORECAST_SERVICE_URL: str = "http://localhost:8002"
    SENTIMENT_SERVICE_URL: str = "http://localhost:8003"
    PORTFOLIO_SERVICE_URL: str = "http://localhost:8004"
    TRADING_ENGINE_URL: str = "http://localhost:8005"
    TRADE_JOURNAL_URL: str = "http://localhost:8008"

    # Alert Configuration
    ALERT_DELIVERY_ENABLED: bool = True
    ALERT_EMAIL_ENABLED: bool = False
    ALERT_EMAIL_TO: Optional[str] = None
    ALERT_SLACK_ENABLED: bool = False
    ALERT_SLACK_WEBHOOK: Optional[str] = None
    ALERT_WEBHOOK_URLS: Optional[str] = None

    # SMTP Configuration (for email alerts)
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM: Optional[str] = None

    # Timeouts (milliseconds)
    INFERENCE_TIMEOUT_MS: int = 60
    FORECAST_TIMEOUT_MS: int = 80
    SENTIMENT_TIMEOUT_MS: int = 80
    PORTFOLIO_TIMEOUT_MS: int = 100

    # Circuit Breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 30
    CIRCUIT_BREAKER_RECOVERY: int = 10

    # Cache
    PLAN_CACHE_TTL: int = 30
    PLAN_CACHE_SWR: int = 10
    ALERT_CACHE_TTL: int = 10
    EXPLAIN_CACHE_TTL: int = 3600

    # Decision Store
    DECISION_SNAPSHOT_TTL_DAYS: int = 30

    # Idempotency
    IDEMPOTENCY_TTL_SECONDS: int = 300

    # Policy
    POLICY_FILE: str = "config/policies.yaml"

    # JWT
    JWT_SECRET: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"

    # Rate Limiting
    RATE_LIMIT_PER_USER: int = 2
    RATE_LIMIT_BURST: int = 6

    # Observability
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
