"""
Compatibility aggregator shim for sentiment service.
Provides expected methods used by app.main by delegating to SentimentStorage.
"""

from datetime import datetime, timedelta
from typing import Any, Dict
from sqlalchemy.orm import Session

from .sentiment_storage import SentimentStorage
from ..models.schemas import SentimentCreate


class SentimentAggregator:
    def __init__(self):
        self.storage = SentimentStorage()

    def store_sentiment(self, db: Session, data: SentimentCreate) -> Dict[str, Any]:
        """Store an on-demand analyzed text as a social post-like record."""
        # Minimal persistence: map to SentimentPost with source and content
        from ..core.database import SentimentPost
        post = SentimentPost(
            platform=str(data.source),
            platform_post_id=f"manual-{int(datetime.utcnow().timestamp()*1000)}",
            symbol=data.symbol,
            author=data.author,
            content=data.content,
            url=data.url,
            sentiment_score=data.sentiment_score,
            sentiment_label=data.sentiment_label,
            confidence=data.confidence,
            engagement=data.engagement,
            analysis_metadata=data.metadata,
            post_timestamp=datetime.utcnow(),
            analyzed_at=datetime.utcnow(),
        )
        db.add(post)
        db.commit()
        db.refresh(post)
        return {
            "id": str(post.id),
            "symbol": post.symbol,
            "sentiment_score": post.sentiment_score,
            "sentiment_label": post.sentiment_label,
            "confidence": post.confidence,
            "timestamp": post.post_timestamp,
        }

    async def get_sentiment_trend(self, db: Session, symbol: str, start_time: datetime):
        """Return a simple trend summary using storage summary over the timeframe."""
        # Determine timeframe string based on delta
        delta = datetime.utcnow() - start_time
        if delta <= timedelta(hours=1):
            timeframe = "1h"
        elif delta <= timedelta(days=1):
            timeframe = "1d"
        elif delta <= timedelta(days=7):
            timeframe = "1w"
        else:
            timeframe = "1m"
        return self.storage.get_sentiment_summary(db, symbol, timeframe)

    def get_service_stats(self, db: Session) -> Dict[str, Any]:
        return self.storage.get_collection_stats(db)


# Export default instance for easy import
sentiment_aggregator = SentimentAggregator()

