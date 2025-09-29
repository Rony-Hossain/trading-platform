"""Event Sentiment Analysis Integration

Integrates with the Sentiment Service to analyze sentiment around events
and their outcomes, providing comprehensive event sentiment tracking.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_

from ..models import EventORM, EventHeadlineORM

logger = logging.getLogger(__name__)


class SentimentTimeframe(str, Enum):
    """Sentiment analysis timeframes."""
    PRE_EVENT = "pre_event"      # Before event occurs
    EVENT_WINDOW = "event_window"  # During event window
    POST_EVENT = "post_event"    # After event occurs
    OVERALL = "overall"          # Complete timeline


class SentimentSource(str, Enum):
    """Sentiment data sources."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    THREADS = "threads"
    TRUTHSOCIAL = "truthsocial"
    ALL = "all"


@dataclass
class SentimentScore:
    """Individual sentiment score result."""
    compound: float  # -1.0 to 1.0
    positive: float
    negative: float
    neutral: float
    label: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float
    source: str
    timeframe: str
    volume: int  # Number of posts/articles analyzed
    metadata: Dict[str, Any]


@dataclass
class EventSentimentAnalysis:
    """Complete sentiment analysis for an event."""
    event_id: str
    symbol: str
    category: str
    analyzed_at: datetime
    timeframes: Dict[str, SentimentScore]
    sources: Dict[str, SentimentScore]
    overall_sentiment: SentimentScore
    sentiment_momentum: float  # Change in sentiment over time
    sentiment_divergence: float  # Difference between sources
    outcome_prediction: Optional[str]  # POSITIVE, NEGATIVE, NEUTRAL
    prediction_confidence: float


class EventSentimentService:
    """Service for analyzing sentiment around events and their outcomes."""
    
    def __init__(self):
        self.sentiment_service_url = os.getenv(
            "SENTIMENT_SERVICE_URL", 
            "http://localhost:8007"
        )
        self.enabled = os.getenv("EVENT_SENTIMENT_ENABLED", "true").lower() == "true"
        self.timeout = float(os.getenv("EVENT_SENTIMENT_TIMEOUT", "30.0"))
        self.pre_event_hours = int(os.getenv("EVENT_SENTIMENT_PRE_HOURS", "24"))
        self.post_event_hours = int(os.getenv("EVENT_SENTIMENT_POST_HOURS", "24"))
        self.event_window_hours = int(os.getenv("EVENT_SENTIMENT_WINDOW_HOURS", "2"))
        
        # Sentiment analysis cache
        self._analysis_cache = {}
        self._cache_ttl = 1800  # 30 minutes
        
        logger.info(f"EventSentimentService initialized (enabled={self.enabled})")
    
    async def analyze_event_sentiment(
        self, 
        event: EventORM,
        session: AsyncSession,
        force_refresh: bool = False
    ) -> Optional[EventSentimentAnalysis]:
        """Analyze sentiment for a specific event across all timeframes."""
        if not self.enabled:
            return None
        
        # Check cache first
        cache_key = f"{event.id}:{event.updated_at.isoformat()}"
        if not force_refresh and cache_key in self._analysis_cache:
            cached_result, cached_time = self._analysis_cache[cache_key]
            if (datetime.utcnow() - cached_time).seconds < self._cache_ttl:
                return cached_result
        
        try:
            # Get sentiment data for all timeframes
            timeframe_sentiments = {}
            source_sentiments = {}
            
            # Analyze different timeframes
            for timeframe in SentimentTimeframe:
                if timeframe == SentimentTimeframe.OVERALL:
                    continue
                    
                start_time, end_time = self._get_timeframe_bounds(event, timeframe)
                sentiment = await self._fetch_sentiment_data(
                    event.symbol, start_time, end_time, SentimentSource.ALL
                )
                if sentiment:
                    timeframe_sentiments[timeframe.value] = sentiment
            
            # Analyze different sources
            for source in [SentimentSource.TWITTER, SentimentSource.REDDIT, SentimentSource.NEWS]:
                start_time = event.scheduled_at - timedelta(hours=self.pre_event_hours)
                end_time = event.scheduled_at + timedelta(hours=self.post_event_hours)
                sentiment = await self._fetch_sentiment_data(
                    event.symbol, start_time, end_time, source
                )
                if sentiment:
                    source_sentiments[source.value] = sentiment
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(
                timeframe_sentiments, source_sentiments
            )
            
            # Calculate momentum and divergence
            momentum = self._calculate_sentiment_momentum(timeframe_sentiments)
            divergence = self._calculate_sentiment_divergence(source_sentiments)
            
            # Predict outcome based on sentiment patterns
            prediction, confidence = self._predict_event_outcome(
                overall_sentiment, momentum, divergence, event
            )
            
            analysis = EventSentimentAnalysis(
                event_id=event.id,
                symbol=event.symbol,
                category=event.category,
                analyzed_at=datetime.utcnow(),
                timeframes=timeframe_sentiments,
                sources=source_sentiments,
                overall_sentiment=overall_sentiment,
                sentiment_momentum=momentum,
                sentiment_divergence=divergence,
                outcome_prediction=prediction,
                prediction_confidence=confidence
            )
            
            # Cache the result
            self._analysis_cache[cache_key] = (analysis, datetime.utcnow())
            
            # Store sentiment analysis in event metadata
            await self._store_sentiment_analysis(event, analysis, session)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for event {event.id}: {str(e)}")
            return None
    
    async def analyze_event_outcome_sentiment(
        self, 
        event: EventORM,
        session: AsyncSession
    ) -> Optional[SentimentScore]:
        """Analyze sentiment specifically around event outcomes."""
        if not self.enabled:
            return None
        
        try:
            # Look for outcome sentiment in headlines first
            headlines = await session.execute(
                select(EventHeadlineORM).where(
                    and_(
                        EventHeadlineORM.event_id == event.id,
                        EventHeadlineORM.published_at >= event.scheduled_at,
                        EventHeadlineORM.published_at <= event.scheduled_at + timedelta(hours=6)
                    )
                )
            )
            headlines = headlines.scalars().all()
            
            if headlines:
                # Analyze sentiment of outcome headlines
                headline_texts = [h.headline for h in headlines]
                outcome_sentiment = await self._analyze_text_sentiment(
                    " ".join(headline_texts), 
                    context=f"event_outcome_{event.category}"
                )
                
                if outcome_sentiment:
                    outcome_sentiment.metadata.update({
                        "source": "headlines",
                        "headline_count": len(headlines),
                        "event_id": event.id
                    })
                    return outcome_sentiment
            
            # Fallback to post-event social sentiment
            start_time = event.scheduled_at
            end_time = event.scheduled_at + timedelta(hours=6)
            
            return await self._fetch_sentiment_data(
                event.symbol, start_time, end_time, SentimentSource.ALL
            )
            
        except Exception as e:
            logger.error(f"Error analyzing outcome sentiment for event {event.id}: {str(e)}")
            return None
    
    async def get_sentiment_trends(
        self, 
        symbol: str, 
        days: int = 7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get sentiment trends for a symbol over time."""
        if not self.enabled:
            return {}
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.sentiment_service_url}/sentiment/{symbol}",
                    params={
                        "timeframe": f"{days}d",
                        "granularity": "1h"
                    }
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Failed to get sentiment trends: {response.status_code}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting sentiment trends for {symbol}: {str(e)}")
            return {}
    
    async def _fetch_sentiment_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        source: SentimentSource
    ) -> Optional[SentimentScore]:
        """Fetch sentiment data from the sentiment service."""
        try:
            timeframe_hours = int((end_time - start_time).total_seconds() / 3600)
            timeframe = f"{timeframe_hours}h" if timeframe_hours < 24 else f"{timeframe_hours//24}d"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                params = {
                    "timeframe": timeframe,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
                
                if source != SentimentSource.ALL:
                    params["sources"] = [source.value]
                
                response = await client.get(
                    f"{self.sentiment_service_url}/analysis/{symbol}",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return SentimentScore(
                        compound=data.get("compound", 0.0),
                        positive=data.get("positive", 0.0),
                        negative=data.get("negative", 0.0),
                        neutral=data.get("neutral", 0.0),
                        label=data.get("label", "NEUTRAL"),
                        confidence=data.get("confidence", 0.0),
                        source=source.value,
                        timeframe=timeframe,
                        volume=data.get("volume", 0),
                        metadata=data.get("metadata", {})
                    )
                
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {str(e)}")
        
        return None
    
    async def _analyze_text_sentiment(
        self, 
        text: str, 
        context: str = "general"
    ) -> Optional[SentimentScore]:
        """Analyze sentiment of specific text using the sentiment service."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.sentiment_service_url}/analyze",
                    json={
                        "text": text,
                        "context": context,
                        "symbol": "GENERAL"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return SentimentScore(
                        compound=data.get("compound", 0.0),
                        positive=data.get("positive", 0.0),
                        negative=data.get("negative", 0.0),
                        neutral=data.get("neutral", 0.0),
                        label=data.get("label", "NEUTRAL"),
                        confidence=data.get("confidence", 0.0),
                        source="text_analysis",
                        timeframe="instant",
                        volume=1,
                        metadata={"context": context, "text_length": len(text)}
                    )
                
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
        
        return None
    
    def _get_timeframe_bounds(
        self, 
        event: EventORM, 
        timeframe: SentimentTimeframe
    ) -> tuple[datetime, datetime]:
        """Get start and end times for a specific timeframe."""
        event_time = event.scheduled_at
        
        if timeframe == SentimentTimeframe.PRE_EVENT:
            start_time = event_time - timedelta(hours=self.pre_event_hours)
            end_time = event_time
        elif timeframe == SentimentTimeframe.EVENT_WINDOW:
            start_time = event_time - timedelta(hours=self.event_window_hours//2)
            end_time = event_time + timedelta(hours=self.event_window_hours//2)
        elif timeframe == SentimentTimeframe.POST_EVENT:
            start_time = event_time
            end_time = event_time + timedelta(hours=self.post_event_hours)
        else:  # OVERALL
            start_time = event_time - timedelta(hours=self.pre_event_hours)
            end_time = event_time + timedelta(hours=self.post_event_hours)
        
        return start_time, end_time
    
    def _calculate_overall_sentiment(
        self,
        timeframe_sentiments: Dict[str, SentimentScore],
        source_sentiments: Dict[str, SentimentScore]
    ) -> SentimentScore:
        """Calculate overall sentiment score from timeframes and sources."""
        all_scores = list(timeframe_sentiments.values()) + list(source_sentiments.values())
        
        if not all_scores:
            return SentimentScore(
                compound=0.0, positive=0.0, negative=0.0, neutral=1.0,
                label="NEUTRAL", confidence=0.0, source="combined",
                timeframe="overall", volume=0, metadata={}
            )
        
        # Weighted average with higher weight for recent timeframes
        weights = {"pre_event": 0.3, "event_window": 0.4, "post_event": 0.3}
        total_weight = sum(weights.get(k, 0.2) for k in timeframe_sentiments.keys())
        
        if total_weight == 0:
            total_weight = len(all_scores)
            weights = {k: 1.0 for k in timeframe_sentiments.keys()}
        
        weighted_compound = sum(
            score.compound * weights.get(timeframe, 0.2) 
            for timeframe, score in timeframe_sentiments.items()
        ) / total_weight
        
        avg_positive = sum(s.positive for s in all_scores) / len(all_scores)
        avg_negative = sum(s.negative for s in all_scores) / len(all_scores)
        avg_neutral = sum(s.neutral for s in all_scores) / len(all_scores)
        avg_confidence = sum(s.confidence for s in all_scores) / len(all_scores)
        total_volume = sum(s.volume for s in all_scores)
        
        # Determine label from compound score
        if weighted_compound >= 0.1:
            label = "BULLISH"
        elif weighted_compound <= -0.1:
            label = "BEARISH"
        else:
            label = "NEUTRAL"
        
        return SentimentScore(
            compound=weighted_compound,
            positive=avg_positive,
            negative=avg_negative,
            neutral=avg_neutral,
            label=label,
            confidence=avg_confidence,
            source="combined",
            timeframe="overall",
            volume=total_volume,
            metadata={
                "timeframe_count": len(timeframe_sentiments),
                "source_count": len(source_sentiments),
                "total_scores": len(all_scores)
            }
        )
    
    def _calculate_sentiment_momentum(
        self, 
        timeframe_sentiments: Dict[str, SentimentScore]
    ) -> float:
        """Calculate sentiment momentum (change from pre-event to post-event)."""
        pre_sentiment = timeframe_sentiments.get("pre_event")
        post_sentiment = timeframe_sentiments.get("post_event")
        
        if not pre_sentiment or not post_sentiment:
            return 0.0
        
        return post_sentiment.compound - pre_sentiment.compound
    
    def _calculate_sentiment_divergence(
        self, 
        source_sentiments: Dict[str, SentimentScore]
    ) -> float:
        """Calculate divergence between different sentiment sources."""
        if len(source_sentiments) < 2:
            return 0.0
        
        compounds = [s.compound for s in source_sentiments.values()]
        mean_compound = sum(compounds) / len(compounds)
        variance = sum((c - mean_compound) ** 2 for c in compounds) / len(compounds)
        
        return variance ** 0.5  # Standard deviation
    
    def _predict_event_outcome(
        self,
        overall_sentiment: SentimentScore,
        momentum: float,
        divergence: float,
        event: EventORM
    ) -> tuple[Optional[str], float]:
        """Predict event outcome based on sentiment patterns."""
        # Base prediction from overall sentiment
        base_confidence = overall_sentiment.confidence
        
        # Adjust for momentum
        if abs(momentum) > 0.2:
            base_confidence += 0.1
        
        # Penalize for high divergence
        if divergence > 0.3:
            base_confidence -= 0.1
        
        # Category-specific adjustments
        category_multipliers = {
            "earnings": 1.2,
            "fda_approval": 1.5,
            "mna": 1.3,
            "regulatory": 1.1,
            "guidance": 1.0
        }
        
        multiplier = category_multipliers.get(event.category, 1.0)
        final_confidence = min(1.0, base_confidence * multiplier)
        
        # Make prediction based on compound score and momentum
        compound_with_momentum = overall_sentiment.compound + (momentum * 0.5)
        
        if compound_with_momentum > 0.15 and final_confidence > 0.6:
            return "POSITIVE", final_confidence
        elif compound_with_momentum < -0.15 and final_confidence > 0.6:
            return "NEGATIVE", final_confidence
        else:
            return "NEUTRAL", final_confidence
    
    async def _store_sentiment_analysis(
        self,
        event: EventORM,
        analysis: EventSentimentAnalysis,
        session: AsyncSession
    ):
        """Store sentiment analysis results in event metadata."""
        try:
            metadata = event.metadata_json or {}
            metadata["sentiment_analysis"] = {
                "analyzed_at": analysis.analyzed_at.isoformat(),
                "overall_sentiment": asdict(analysis.overall_sentiment),
                "momentum": analysis.sentiment_momentum,
                "divergence": analysis.sentiment_divergence,
                "prediction": analysis.outcome_prediction,
                "prediction_confidence": analysis.prediction_confidence,
                "timeframes": {k: asdict(v) for k, v in analysis.timeframes.items()},
                "sources": {k: asdict(v) for k, v in analysis.sources.items()}
            }
            
            await session.execute(
                update(EventORM)
                .where(EventORM.id == event.id)
                .values(metadata_json=metadata)
            )
            await session.commit()
            
        except Exception as e:
            logger.error(f"Error storing sentiment analysis: {str(e)}")


def build_sentiment_service() -> EventSentimentService:
    """Factory function to create sentiment service instance."""
    return EventSentimentService()