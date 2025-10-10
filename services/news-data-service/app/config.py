from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    NEWS_PG_DSN: str = Field(..., description="SQLAlchemy async DSN for Postgres")
    REDIS_URL: str = Field(..., description="Redis URL")

    # prefilter / cache
    PREFILTER_TTL_SECONDS: int = 48 * 3600
    IDENTITY_CACHE_TTL_SECONDS: int = 24 * 3600

    # upsert buffer
    UPSERT_BATCH_SIZE: int = 200
    UPSERT_FLUSH_MS: int = 20

    # redis streams
    SSE_REDIS_STREAM: str = "news.events"
    REDIS_STREAM_MAXLEN: int = 10000  # keep last ~10k events

    # circuit breaker
    CIRCUIT_TRIP_ERROR_RATE: float = 0.35
    CIRCUIT_MIN_WINDOW: int = 50
    CIRCUIT_OPEN_SECONDS: int = 60

    # adapter retries
    ADAPTER_RETRY_ATTEMPTS: int = 3
    ADAPTER_RETRY_BASE_DELAY: float = 1.0
    ADAPTER_RETRY_MAX_DELAY: float = 10.0

    # watchlist config
    FINNHUB_COMPANY_WATCHLIST_FILE: str | None = None
    FINNHUB_COMPANY_WATCHLIST_REDIS_SET: str | None = None
    WATCHLIST_REFRESH_SECONDS: int = 60

settings = Settings()
