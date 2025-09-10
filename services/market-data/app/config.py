from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    DEBUG: bool = False
    
    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://trading_user:trading_pass@localhost:5432/trading_db"
    
    # Redis settings  
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_PASSWORD: Optional[str] = None
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "https://yourdomain.com"]
    
    # External API keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    FINNHUB_API_KEY: Optional[str] = None
    IEX_CLOUD_API_KEY: Optional[str] = None
    
    # Cache settings
    CACHE_DEFAULT_TTL: int = 300  # 5 minutes
    PRICE_CACHE_TTL: int = 60     # 1 minute for prices
    PROFILE_CACHE_TTL: int = 3600 # 1 hour for profiles
    SEARCH_CACHE_TTL: int = 1800  # 30 minutes for search
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    EXTERNAL_API_RATE_LIMIT: int = 5  # calls per second to external APIs
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
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