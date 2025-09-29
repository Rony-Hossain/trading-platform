import finnhub
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
from .base import DataProvider

logger = logging.getLogger(__name__)

class FinnhubProvider(DataProvider):
    def __init__(self, api_key: str = "demo"):
        super().__init__("Finnhub")
        self.api_key = api_key
        self.client = finnhub.Client(api_key=api_key)
        self.rate_limit_delay = 1.0  # 60 calls/minute = 1 call/second for free tier
        
    async def get_price(self, symbol: str) -> Optional[Dict]:
        """Get current stock price from Finnhub"""
        try:
            # Add rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Get current price quote
            quote = await asyncio.to_thread(self.client.quote, symbol)
            
            if not quote or 'c' not in quote:
                return None
            
            current_price = float(quote['c'])  # Current price
            previous_close = float(quote['pc'])  # Previous close
            
            if current_price == 0:
                return None
            
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close != 0 else 0
            
            self.mark_available()
            return {
                "symbol": symbol.upper(),
                "price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "high": round(float(quote.get('h', current_price)), 2),  # Day high
                "low": round(float(quote.get('l', current_price)), 2),   # Day low
                "open": round(float(quote.get('o', current_price)), 2),  # Day open
                "previous_close": round(previous_close, 2),
                "timestamp": datetime.now().isoformat(),
                "source": self.name
            }
            
        except Exception as e:
            error_msg = f"Finnhub error for {symbol}: {e}"
            self.mark_unavailable(error_msg)
            logger.error(error_msg)
            return None
    
    async def get_historical(self, symbol: str, period: str = "1mo") -> Optional[Dict]:
        """Get historical data from Finnhub"""
        try:
            await asyncio.sleep(self.rate_limit_delay)
            
            # Convert period to date range
            end_date = datetime.now()
            
            period_map = {
                "1d": 1,
                "5d": 5,
                "1mo": 30,
                "3mo": 90,
                "6mo": 180,
                "1y": 365
            }
            
            days = period_map.get(period, 30)
            start_date = end_date - timedelta(days=days)
            
            # Convert to Unix timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            # Get candle data (daily resolution)
            candles = await asyncio.to_thread(
                self.client.stock_candles, 
                symbol, 
                'D',  # Daily resolution
                start_ts, 
                end_ts
            )
            
            if not candles or candles['s'] != 'ok':
                return None
            
            data = []
            timestamps = candles['t']
            opens = candles['o']
            highs = candles['h']
            lows = candles['l']
            closes = candles['c']
            volumes = candles['v']
            
            for i in range(len(timestamps)):
                date = datetime.fromtimestamp(timestamps[i])
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(float(opens[i]), 2),
                    "high": round(float(highs[i]), 2),
                    "low": round(float(lows[i]), 2),
                    "close": round(float(closes[i]), 2),
                    "volume": int(volumes[i])
                })
            
            return {
                "symbol": symbol.upper(),
                "period": period,
                "data": data,
                "source": self.name
            }
            
        except Exception as e:
            logger.error(f"Finnhub historical error for {symbol}: {e}")
            return None

    async def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """Get company profile data"""
        try:
            await asyncio.sleep(self.rate_limit_delay)
            
            profile = await asyncio.to_thread(self.client.company_profile2, symbol=symbol)
            
            if not profile:
                return None
                
            return {
                "symbol": symbol.upper(),
                "name": profile.get('name'),
                "country": profile.get('country'),
                "currency": profile.get('currency'),
                "exchange": profile.get('exchange'),
                "ipo": profile.get('ipo'),
                "market_cap": profile.get('marketCapitalization'),
                "shares_outstanding": profile.get('shareOutstanding'),
                "logo": profile.get('logo'),
                "phone": profile.get('phone'),
                "weburl": profile.get('weburl'),
                "industry": profile.get('finnhubIndustry'),
                "source": self.name
            }
            
        except Exception as e:
            logger.error(f"Finnhub company profile error for {symbol}: {e}")
            return None

    async def get_news_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get news sentiment for a symbol"""
        try:
            await asyncio.sleep(self.rate_limit_delay)
            
            sentiment = await asyncio.to_thread(self.client.news_sentiment, symbol)
            
            if not sentiment:
                return None
                
            return {
                "symbol": symbol.upper(),
                "sentiment": {
                    "buzz": sentiment.get('buzz', {}),
                    "sentiment": sentiment.get('sentiment', {}),
                    "company_news_score": sentiment.get('companyNewsScore'),
                    "sector_average_bullishness": sentiment.get('sectorAverageBullishness'),
                    "sector_average_news_score": sentiment.get('sectorAverageNewsScore')
                },
                "source": self.name
            }
            
        except Exception as e:
            logger.error(f"Finnhub news sentiment error for {symbol}: {e}")
            return None

    async def get_intraday(self, symbol: str, interval: str = "1m") -> Optional[Dict]:
        """Get intraday minute data from Finnhub"""
        try:
            await asyncio.sleep(self.rate_limit_delay)

            end_date = datetime.now()
            # fetch last trading day window (~6.5h = 390 minutes)
            start_date = end_date - timedelta(hours=8)
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())

            resolution = '1'  # 1-minute
            candles = await asyncio.to_thread(self.client.stock_candles, symbol, resolution, start_ts, end_ts)
            if not candles or candles.get('s') != 'ok':
                return None

            data = []
            for i, ts in enumerate(candles['t']):
                date = datetime.fromtimestamp(ts)
                data.append({
                    "timestamp": date.isoformat(),
                    "open": round(float(candles['o'][i]), 4),
                    "high": round(float(candles['h'][i]), 4),
                    "low": round(float(candles['l'][i]), 4),
                    "close": round(float(candles['c'][i]), 4),
                    "volume": int(candles['v'][i])
                })

            return {"symbol": symbol.upper(), "interval": interval, "data": data, "source": self.name}
        except Exception as e:
            logger.error(f"Finnhub intraday error for {symbol}: {e}")
            return None
