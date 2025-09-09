import asyncio
import yfinance as yf
import logging
from datetime import datetime
from typing import Dict, Optional
from .base import DataProvider

logger = logging.getLogger(__name__)

class YahooFinanceProvider(DataProvider):
    def __init__(self):
        super().__init__("Yahoo Finance")
    
    async def get_price(self, symbol: str) -> Optional[Dict]:
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            hist = await asyncio.to_thread(ticker.history, period="2d")
            
            if hist.empty:
                return None
                
            current_price = float(hist['Close'].iloc[-1])
            previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close != 0 else 0
            
            self.mark_available()
            return {
                "symbol": symbol.upper(),
                "price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "volume": int(hist['Volume'].iloc[-1]),
                "timestamp": datetime.now().isoformat(),
                "source": self.name
            }
        except Exception as e:
            error_msg = f"Yahoo Finance error for {symbol}: {e}"
            self.mark_unavailable(error_msg)
            logger.error(error_msg)
            return None
    
    async def get_historical(self, symbol: str, period: str = "1mo") -> Optional[Dict]:
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            hist = await asyncio.to_thread(ticker.history, period=period)
            
            if hist.empty:
                return None
            
            data = []
            for date, row in hist.iterrows():
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(float(row['Open']), 2),
                    "high": round(float(row['High']), 2),
                    "low": round(float(row['Low']), 2),
                    "close": round(float(row['Close']), 2),
                    "volume": int(row['Volume'])
                })
            
            return {
                "symbol": symbol.upper(),
                "period": period,
                "data": data,
                "source": self.name
            }
        except Exception as e:
            logger.error(f"Yahoo Finance historical error for {symbol}: {e}")
            return None
