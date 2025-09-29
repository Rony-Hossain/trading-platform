"""
Storage integration utilities for sentiment collectors
Provides helper methods to integrate collectors with storage
"""

from typing import Optional
from sqlalchemy.orm import Session
from .sentiment_storage import sentiment_storage
from ..models.schemas import SocialPost, NewsArticle

async def store_social_post_with_symbol(db: Session, post: SocialPost, symbol: str):
    """Helper to store social post with sentiment analysis"""
    return await sentiment_storage.store_social_post(db, post, symbol)

async def store_news_article_with_symbol(db: Session, article: NewsArticle, symbol: str):
    """Helper to store news article with sentiment analysis"""
    return await sentiment_storage.store_news_article(db, article, symbol)

def update_collection_status(db: Session, platform: str, symbol: str, 
                           posts_collected: int = 0, errors: int = 0,
                           error_message: Optional[str] = None):
    """Helper to update collection status"""
    sentiment_storage.update_collection_status(
        db, platform, symbol, posts_collected, errors, error_message
    )