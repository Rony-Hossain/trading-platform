"""
Enhanced Twitter Collector with Full Storage Integration
Example implementation showing proper data storage
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
import tweepy

from ..models.schemas import SocialPost
from .sentiment_storage import sentiment_storage

logger = logging.getLogger(__name__)

class EnhancedTwitterCollector:
    """Twitter collector with complete storage integration"""
    
    def __init__(self):
        # Twitter API setup (same as before)
        self.client = None  # Initialize with your Twitter client
        
    async def collect_for_symbols(self, symbols: List[str], keywords: Optional[List[str]], db: Session):
        """Collect tweets for symbols with full storage integration"""
        
        for symbol in symbols:
            try:
                logger.info(f"Starting Twitter collection for {symbol}")
                
                posts_collected = 0
                errors = 0
                error_message = None
                
                try:
                    # Collect tweets (simplified example)
                    tweets = await self._fetch_tweets_for_symbol(symbol, keywords)
                    
                    for tweet_data in tweets:
                        try:
                            # Convert to SocialPost
                            post = SocialPost(
                                id=str(tweet_data['id']),
                                platform="twitter",
                                author=tweet_data['author']['username'],
                                content=tweet_data['text'],
                                timestamp=datetime.fromisoformat(tweet_data['created_at'].replace('Z', '+00:00')),
                                url=f"https://twitter.com/{tweet_data['author']['username']}/status/{tweet_data['id']}",
                                engagement={
                                    'likes': tweet_data.get('public_metrics', {}).get('like_count', 0),
                                    'retweets': tweet_data.get('public_metrics', {}).get('retweet_count', 0),
                                    'replies': tweet_data.get('public_metrics', {}).get('reply_count', 0)
                                }
                            )
                            
                            # Store with sentiment analysis
                            stored_post = await sentiment_storage.store_social_post(db, post, symbol)
                            
                            if stored_post:
                                posts_collected += 1
                                logger.debug(f"Stored tweet {post.id} for {symbol}")
                            
                        except Exception as e:
                            errors += 1
                            error_message = str(e)
                            logger.error(f"Error processing tweet: {e}")
                    
                except Exception as e:
                    errors += 1
                    error_message = str(e)
                    logger.error(f"Error fetching tweets for {symbol}: {e}")
                
                # Update collection status
                sentiment_storage.update_collection_status(
                    db, "twitter", symbol, posts_collected, errors, error_message
                )
                
                logger.info(f"Twitter collection for {symbol}: {posts_collected} posts, {errors} errors")
                
            except Exception as e:
                logger.error(f"Critical error in Twitter collection for {symbol}: {e}")
    
    async def _fetch_tweets_for_symbol(self, symbol: str, keywords: Optional[List[str]]) -> List[Dict]:
        """Fetch tweets from Twitter API (simplified)"""
        # This would contain your actual Twitter API calls
        # For now, return mock data to demonstrate structure
        
        mock_tweets = [
            {
                'id': '1234567890',
                'text': f'${symbol} is looking bullish today! 🚀',
                'created_at': '2024-01-01T10:00:00.000Z',
                'author': {'username': 'trader123'},
                'public_metrics': {
                    'like_count': 42,
                    'retweet_count': 15,
                    'reply_count': 8
                }
            },
            {
                'id': '1234567891', 
                'text': f'Thinking about buying more {symbol} shares',
                'created_at': '2024-01-01T10:05:00.000Z',
                'author': {'username': 'investor456'},
                'public_metrics': {
                    'like_count': 23,
                    'retweet_count': 5,
                    'reply_count': 12
                }
            }
        ]
        
        return mock_tweets

# Example usage in main collector
class StorageIntegratedCollectors:
    """Main collector class with proper storage integration"""
    
    def __init__(self):
        self.twitter_collector = EnhancedTwitterCollector()
        # Add other collectors...
    
    async def collect_all_data(self, symbols: List[str], db: Session):
        """Collect data from all sources with storage"""
        
        # Collect from Twitter
        await self.twitter_collector.collect_for_symbols(symbols, None, db)
        
        # Compute aggregates after collection
        for symbol in symbols:
            await sentiment_storage.compute_aggregates(db, symbol, "1h")
        
        # Get collection stats
        stats = sentiment_storage.get_collection_stats(db)
        logger.info(f"Collection completed. Stats: {stats}")
        
        return stats

# Example of retrieving stored data
async def get_sentiment_data_example(db: Session, symbol: str):
    """Example of retrieving stored sentiment data"""
    
    # Get recent posts
    recent_posts = sentiment_storage.get_recent_posts(db, symbol, platform="twitter", hours=24)
    print(f"Found {len(recent_posts)} recent Twitter posts for {symbol}")
    
    # Get sentiment summary
    summary = sentiment_storage.get_sentiment_summary(db, symbol, "1d")
    print(f"Sentiment summary for {symbol}: {summary}")
    
    # Get collection stats
    stats = sentiment_storage.get_collection_stats(db)
    print(f"Overall collection stats: {stats}")
    
    return {
        'recent_posts': len(recent_posts),
        'summary': summary,
        'stats': stats
    }