from pydantic_settings import BaseSettings
from typing import List, Optional
from pydantic import ConfigDict

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Service Configuration
    service_name: str = "market-data-service"
    service_port: int = 8001
    debug: bool = True
    
    # Database Configuration
    database_url: str = "postgresql://trading_user:trading_pass@localhost:5432/trading_db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Cache Configuration
    cache_ttl_seconds: int = 5
    
    # Data Storage Configuration
    store_historical_data: bool = True
    historical_data_retention_days: int = 365
    
    # API Keys (for when we add more providers)
    alpha_vantage_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    
    # Rate Limiting
    max_requests_per_minute: int = 300
    
    # WebSocket Configuration
    websocket_heartbeat_interval: int = 30
    websocket_update_interval: int = 5
    
    # Macro / cross-asset configuration
    macro_refresh_interval_seconds: int = 900
    macro_cache_ttl_seconds: int = 300

settings = Settings()
