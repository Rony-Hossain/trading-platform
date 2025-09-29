"""
Headline Capture Service - Real-time news and headline monitoring
Captures breaking news, earnings reports, and other market-moving headlines
"""

import asyncio
import logging
import aiohttp
import feedparser
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from ..core.database import HeadlineCapture, HeadlineData, HeadlineType

logger = logging.getLogger(__name__)


class HeadlineCaptureService:
    """Service for capturing and processing real-time headlines"""
    
    def __init__(self):
        self.running = False
        self.refresh_interval = 300  # 5 minutes
        self.last_refresh = {}
        self.news_sources = {
            "reuters": "https://feeds.reuters.com/reuters/businessNews",
            "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
            "marketwatch": "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
            "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
            "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
        }
        
        # Keywords for relevance scoring
        self.high_relevance_keywords = [
            "earnings", "revenue", "profit", "loss", "guidance", "forecast",
            "merger", "acquisition", "ipo", "split", "dividend", "buyback",
            "fda approval", "clinical trial", "regulatory", "lawsuit",
            "ceo", "cfo", "management", "board", "resignation",
        ]
        
        self.urgency_keywords = [
            "breaking", "urgent", "alert", "halt", "suspended", "emergency",
            "immediate", "flash", "just in", "developing",
        ]
        
        self.sentiment_positive = [
            "beat", "exceed", "strong", "growth", "positive", "up", "gain",
            "success", "approved", "win", "bullish", "optimistic",
        ]
        
        self.sentiment_negative = [
            "miss", "below", "weak", "decline", "negative", "down", "loss",
            "fail", "rejected", "bear", "pessimistic", "warning",
        ]
    
    async def start_background_tasks(self):
        """Start background headline monitoring"""
        self.running = True
        asyncio.create_task(self._background_capture_loop())
        logger.info("Headline capture background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks"""
        self.running = False
        logger.info("Headline capture background tasks stopped")
    
    async def _background_capture_loop(self):
        """Background task to capture headlines"""
        while self.running:
            try:
                await self._capture_from_all_sources()
                await asyncio.sleep(self.refresh_interval)
            except Exception as e:
                logger.error(f"Error in background headline capture: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _capture_from_all_sources(self):
        """Capture headlines from all configured sources"""
        try:
            tasks = []
            for source_name, source_url in self.news_sources.items():
                task = asyncio.create_task(self._capture_from_source(source_name, source_url))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_captured = 0
            for i, result in enumerate(results):
                source_name = list(self.news_sources.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Error capturing from {source_name}: {result}")
                else:
                    total_captured += result
            
            logger.info(f"Captured {total_captured} headlines from {len(self.news_sources)} sources")
            self.last_refresh["all"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in capture cycle: {e}")
    
    async def _capture_from_source(self, source_name: str, source_url: str) -> int:
        """Capture headlines from a specific RSS source"""
        try:
            # For demo purposes, generate synthetic headlines
            # In production, this would actually fetch RSS feeds
            
            headlines_captured = 0
            
            # Generate synthetic headlines for major symbols
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
            
            for symbol in symbols:
                if not self.running:
                    break
                
                # Create synthetic headline
                headline_templates = [
                    f"{symbol} Reports Strong Q4 Earnings, Beats Analyst Expectations",
                    f"{symbol} Stock Surges After Positive Analyst Upgrade",
                    f"{symbol} Announces New Product Launch, Shares React Positively",
                    f"{symbol} Management Provides Optimistic Guidance for Next Quarter",
                    f"Breaking: {symbol} Completes Major Acquisition Deal",
                ]
                
                headline_text = headline_templates[hash(symbol + str(datetime.now().hour)) % len(headline_templates)]
                
                # Calculate scores
                relevance = self._calculate_relevance_score(headline_text, symbol)
                sentiment = self._calculate_sentiment_score(headline_text)
                urgency = self._calculate_urgency_score(headline_text)
                headline_type = self._classify_headline_type(headline_text)
                
                # Only process if relevance is high enough
                if relevance >= 0.6:
                    headline_data = HeadlineData(
                        id=hash(f"{symbol}{headline_text}{datetime.now()}") % 1000000,
                        symbol=symbol,
                        headline=headline_text,
                        source=source_name,
                        source_url=f"{source_url}#{symbol.lower()}",
                        published_at=datetime.now() - timedelta(minutes=hash(symbol) % 60),
                        captured_at=datetime.now(),
                        headline_type=headline_type,
                        relevance_score=relevance,
                        sentiment_score=sentiment,
                        urgency_score=urgency,
                        article_content=None,
                        content_summary=f"Summary of {headline_text[:50]}...",
                        keywords=self._extract_keywords(headline_text),
                        entities={"companies": [symbol], "people": [], "places": []},
                        metadata={
                            "source_type": "rss",
                            "processed_at": datetime.now().isoformat(),
                            "language": "en",
                        }
                    )
                    
                    # In production, this would store to database
                    logger.debug(f"Would store headline: {headline_text[:50]}...")
                    headlines_captured += 1
            
            return headlines_captured
            
        except Exception as e:
            logger.error(f"Error capturing from {source_name}: {e}")
            return 0
    
    def _calculate_relevance_score(self, headline: str, symbol: str) -> float:
        """Calculate relevance score for a headline"""
        score = 0.0
        headline_lower = headline.lower()
        
        # Symbol mention
        if symbol.lower() in headline_lower:
            score += 0.4
        
        # High relevance keywords
        for keyword in self.high_relevance_keywords:
            if keyword in headline_lower:
                score += 0.1
        
        # Financial terms
        financial_terms = ["stock", "share", "price", "market", "trading", "investor"]
        for term in financial_terms:
            if term in headline_lower:
                score += 0.05
        
        return min(1.0, score)
    
    def _calculate_sentiment_score(self, headline: str) -> float:
        """Calculate sentiment score (-1.0 to 1.0)"""
        headline_lower = headline.lower()
        
        positive_count = sum(1 for word in self.sentiment_positive if word in headline_lower)
        negative_count = sum(1 for word in self.sentiment_negative if word in headline_lower)
        
        if positive_count == 0 and negative_count == 0:
            return 0.0
        
        total_sentiment_words = positive_count + negative_count
        sentiment = (positive_count - negative_count) / total_sentiment_words
        
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_urgency_score(self, headline: str) -> float:
        """Calculate urgency/breaking news score"""
        headline_lower = headline.lower()
        
        urgency_score = 0.0
        for keyword in self.urgency_keywords:
            if keyword in headline_lower:
                urgency_score += 0.2
        
        # All caps indicates urgency
        if headline.isupper():
            urgency_score += 0.3
        
        # Exclamation marks
        urgency_score += headline.count("!") * 0.1
        
        return min(1.0, urgency_score)
    
    def _classify_headline_type(self, headline: str) -> str:
        """Classify the type of headline"""
        headline_lower = headline.lower()
        
        if any(word in headline_lower for word in ["breaking", "urgent", "alert"]):
            return HeadlineType.BREAKING.value
        elif any(word in headline_lower for word in ["earnings", "revenue", "profit"]):
            return HeadlineType.EARNINGS.value
        elif any(word in headline_lower for word in ["merger", "acquisition", "deal"]):
            return HeadlineType.MERGER.value
        elif any(word in headline_lower for word in ["fda", "regulatory", "approval"]):
            return HeadlineType.REGULATORY.value
        elif any(word in headline_lower for word in ["analyst", "upgrade", "downgrade"]):
            return HeadlineType.ANALYST.value
        else:
            return HeadlineType.GENERAL.value
    
    def _extract_keywords(self, headline: str) -> List[str]:
        """Extract key terms from headline"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', headline.lower())
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    async def get_symbol_headlines(
        self,
        symbol: str,
        since: datetime,
        min_relevance: float = 0.7,
        include_content: bool = False,
        db: Session = None,
    ) -> List[HeadlineData]:
        """Get headlines for a specific symbol"""
        
        # Generate synthetic headlines for the symbol
        headlines = []
        
        # Create 5-10 synthetic headlines for the symbol
        num_headlines = 5 + (hash(symbol) % 6)
        
        for i in range(num_headlines):
            time_offset = timedelta(hours=i + 1)
            published_time = datetime.now() - time_offset
            
            if published_time < since:
                continue
            
            headline_templates = [
                f"{symbol} Stock Reaches New 52-Week High on Strong Market Sentiment",
                f"Analyst Raises {symbol} Price Target, Cites Strong Fundamentals",
                f"{symbol} Announces Strategic Partnership, Shares Jump in After-Hours",
                f"Breaking: {symbol} Reports Record Quarterly Revenue Growth",
                f"{symbol} CEO Discusses Future Growth Plans in CNBC Interview",
                f"Institutional Investors Increase {symbol} Holdings, SEC Filings Show",
                f"{symbol} Outperforms Sector Peers Amid Market Volatility",
            ]
            
            headline_text = headline_templates[i % len(headline_templates)]
            
            relevance = self._calculate_relevance_score(headline_text, symbol)
            if relevance < min_relevance:
                continue
            
            headline = HeadlineData(
                id=1000 + i,
                symbol=symbol,
                headline=headline_text,
                source="reuters" if i % 2 == 0 else "bloomberg",
                source_url=f"https://example.com/news/{symbol.lower()}-{i}",
                published_at=published_time,
                captured_at=published_time + timedelta(minutes=5),
                headline_type=self._classify_headline_type(headline_text),
                relevance_score=relevance,
                sentiment_score=self._calculate_sentiment_score(headline_text),
                urgency_score=self._calculate_urgency_score(headline_text),
                article_content=f"Full article content for: {headline_text}" if include_content else None,
                content_summary=f"Summary: {headline_text[:60]}...",
                keywords=self._extract_keywords(headline_text),
                entities={"companies": [symbol], "people": [], "places": []},
                metadata={
                    "source_type": "rss",
                    "processed_at": datetime.now().isoformat(),
                    "language": "en",
                }
            )
            
            headlines.append(headline)
        
        # Sort by published time, most recent first
        headlines.sort(key=lambda x: x.published_at, reverse=True)
        
        return headlines
    
    async def get_recent_headlines(
        self,
        since: datetime,
        symbols: Optional[List[str]] = None,
        min_relevance: float = 0.8,
        limit: int = 50,
        db: Session = None,
    ) -> List[HeadlineData]:
        """Get recent high-impact headlines across symbols"""
        
        all_headlines = []
        symbols_to_process = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD"]
        
        for symbol in symbols_to_process:
            symbol_headlines = await self.get_symbol_headlines(
                symbol=symbol,
                since=since,
                min_relevance=min_relevance,
                db=db
            )
            all_headlines.extend(symbol_headlines)
        
        # Sort by urgency and published time
        all_headlines.sort(
            key=lambda x: (x.urgency_score, x.published_at), 
            reverse=True
        )
        
        return all_headlines[:limit]
    
    async def manual_refresh(
        self,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Manually trigger headline refresh"""
        
        try:
            if symbol:
                logger.info(f"Manual headline refresh for symbol: {symbol}")
                # In production, refresh headlines for specific symbol
            else:
                await self._capture_from_all_sources()
            
            return {
                "status": "success",
                "refreshed_at": datetime.now().isoformat(),
                "symbol": symbol,
            }
            
        except Exception as e:
            logger.error(f"Error in manual headline refresh: {e}")
            return {
                "status": "error",
                "error": str(e),
                "refreshed_at": datetime.now().isoformat(),
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of headline capture service"""
        
        last_refresh_time = self.last_refresh.get("all")
        is_healthy = True
        
        if last_refresh_time:
            time_since_refresh = (datetime.now() - last_refresh_time).total_seconds()
            is_healthy = time_since_refresh < (self.refresh_interval * 3)
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "last_refresh": last_refresh_time.isoformat() if last_refresh_time else None,
            "refresh_interval_seconds": self.refresh_interval,
            "running": self.running,
            "sources_count": len(self.news_sources),
        }
    
    async def get_stats(self, db: Session = None) -> Dict[str, Any]:
        """Get headline capture statistics"""
        
        return {
            "total_headlines_captured": 1250,  # Synthetic
            "headlines_last_24h": 180,
            "sources_active": len(self.news_sources),
            "average_relevance_score": 0.73,
            "headline_types": {
                "breaking": 45,
                "earnings": 67,
                "analyst": 58,
                "regulatory": 23,
                "general": 187,
            },
            "top_symbols": ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"],
            "last_refresh": self.last_refresh.get("all", datetime.now()).isoformat(),
        }


# Global service instance
headline_capture_service = HeadlineCaptureService()