from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Service Configuration
    service_name: str = "market-data-service"
    service_port: int = 8001
    debug: bool = True
    
    # Cache Configuration
    cache_ttl_seconds: int = 5
    
    # API Keys (for when we add more providers)
    alpha_vantage_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    
    # Rate Limiting
    max_requests_per_minute: int = 300
    
    # WebSocket Configuration
    websocket_heartbeat_interval: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()