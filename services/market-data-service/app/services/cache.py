from datetime import datetime, timedelta
from typing import Dict, Optional

class DataCache:
    """Simple in-memory cache for stock data"""
    def __init__(self, ttl_seconds: int = 5):
        self.cache: Dict[str, Dict] = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: Dict):
        self.cache[key] = (data, datetime.now())
    
    def clear_expired(self):
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= timedelta(seconds=self.ttl)
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def stats(self) -> Dict:
        return {
            "size": len(self.cache),
            "ttl_seconds": self.ttl
        }