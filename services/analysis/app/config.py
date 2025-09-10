from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    DEBUG: bool = False
    
    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://trading_user:trading_pass@localhost:5432/trading_db"
    
    # Redis settings  
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_PASSWORD: Optional[str] = None
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://yourdomain.com"]
    
    # Market Data API
    MARKET_DATA_API_URL: str = "http://localhost:8002"
    
    # Cache settings
    CACHE_DEFAULT_TTL: int = 300  # 5 minutes
    TECHNICAL_ANALYSIS_CACHE_TTL: int = 3600    # 1 hour
    PATTERN_CACHE_TTL: int = 7200              # 2 hours
    FORECAST_CACHE_TTL: int = 14400            # 4 hours
    ADVANCED_INDICATORS_CACHE_TTL: int = 1800   # 30 minutes
    
    # Analysis settings
    MIN_DATA_POINTS: int = 50  # Minimum data points for analysis
    MAX_FORECAST_HORIZON: int = 30  # Maximum forecast days
    
    # Model settings
    MODEL_RETRAIN_INTERVAL_HOURS: int = 24
    FEATURE_SELECTION_THRESHOLD: float = 0.01
    CROSS_VALIDATION_FOLDS: int = 5
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # External services
    ENABLE_SENTIMENT_ANALYSIS: bool = False
    NEWS_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def async_database_url(self) -> str:
        """Convert sync DATABASE_URL to async if needed"""
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.DATABASE_URL

settings = Settings()