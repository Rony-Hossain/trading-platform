from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime

class DataProvider(ABC):
    """Base class for data providers"""
    def __init__(self, name: str):
        self.name = name
        self.is_available = True
        self.last_error = None
        self.last_update = None
    
    @abstractmethod
    async def get_price(self, symbol: str) -> Optional[Dict]:
        """Get current stock price"""
        pass
    
    @abstractmethod
    async def get_historical(self, symbol: str, period: str) -> Optional[Dict]:
        """Get historical stock data"""
        pass
    
    def mark_unavailable(self, error: str):
        """Mark provider as unavailable"""
        self.is_available = False
        self.last_error = error
        self.last_update = datetime.now()
    
    def mark_available(self):
        """Mark provider as available"""
        self.is_available = True
        self.last_error = None
        self.last_update = datetime.now()