"""
Social Media Data Collectors
Collects data from Twitter/X, Reddit, StockTwits, and news sources
"""

import logging
import asyncio
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
import os
from sqlalchemy.orm import Session

# Third-party imports
import tweepy
import praw
from bs4 import BeautifulSoup
import yfinance as yf

from ..models.schemas import SocialPost, NewsArticle, SentimentCreate
from .sentiment_storage import sentiment_storage

logger = logging.getLogger(__name__)

@dataclass
class CollectionConfig:
    """Configuration for data collection"""
    max_posts_per_symbol: int = 100
    collection_interval_minutes: int = 15
    rate_limit_delay: float = 1.0
    max_retries: int = 3

class TwitterCollector:
    """Twitter/X data collector using official API"""
    
    def __init__(self):
        self.api_key = os.getenv("TWITTER_API_KEY")
        self.api_secret = os.getenv("TWITTER_API_SECRET")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        
        self.client = None
        self.api = None
        
        if all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            try:
                # Initialize Twitter API v2 client
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_token_secret,
                    wait_on_rate_limit=True
                )
                
                # Initialize v1.1 API for additional features
                auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
                auth.set_access_token(self.access_token, self.access_token_secret)
                self.api = tweepy.API(auth, wait_on_rate_limit=True)
                
                logger.info("Twitter API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter API: {e}")
        else:
            logger.warning("Twitter API credentials not provided")
    
    def is_healthy(self) -> bool:
        """Check if Twitter collector is properly configured"""
        return self.client is not None
    
    async def collect_for_symbols(self, symbols: List[str], keywords: Optional[List[str]], db: Session):
        """Collect tweets for given symbols"""
        if not self.client:
            logger.warning("Twitter API not available")
            return
        
        for symbol in symbols:
            try:
                await self.collect_symbol_tweets(symbol, keywords, db)
                await asyncio.sleep(1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error collecting tweets for {symbol}: {e}")
    
    async def collect_symbol_tweets(self, symbol: str, keywords: Optional[List[str]], db: Session):
        """Collect tweets for a specific symbol"""
        try:
            # Build search query
            query_parts = [f"${symbol}", symbol]
            
            if keywords:
                query_parts.extend(keywords)
            
            # Additional financial terms
            financial_terms = ["stock", "trading", "invest", "earnings", "price"]
            
            # Create search query
            query = " OR ".join([f'"{term}"' for term in query_parts[:5]])  # Limit query complexity
            query += " -is:retweet lang:en"  # Exclude retweets, English only
            
            # Search tweets
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                user_fields=['username', 'name', 'verified'],
                expansions=['author_id'],
                max_results=100
            ).flatten(limit=200)
            
            # Process tweets
            for tweet in tweets:
                try:
                    post = self._process_tweet(tweet, symbol)
                    if post:
                        # Store in database (implement your storage logic)
                        await self._store_social_post(db, post)
                except Exception as e:
                    logger.error(f"Error processing tweet {tweet.id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting tweets for {symbol}: {e}")
    
    def _process_tweet(self, tweet, symbol: str) -> Optional[SocialPost]:
        """Process individual tweet into SocialPost"""
        try:
            # Get user info
            user = None
            if hasattr(tweet, 'includes') and 'users' in tweet.includes:
                user = tweet.includes['users'][0]
            
            username = user.username if user else "unknown"
            
            # Extract engagement metrics
            metrics = tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}
            engagement = {
                'likes': metrics.get('like_count', 0),
                'retweets': metrics.get('retweet_count', 0),
                'replies': metrics.get('reply_count', 0),
                'quotes': metrics.get('quote_count', 0)
            }
            
            # Create post
            return SocialPost(
                id=str(tweet.id),
                platform="twitter",
                author=username,
                content=tweet.text,
                timestamp=tweet.created_at,
                url=f"https://twitter.com/{username}/status/{tweet.id}",
                engagement=engagement
            )
            
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            return None
    
    def get_recent_posts(self, db: Session, symbol: str, start_time: datetime, limit: int) -> List[SocialPost]:
        """Get recent posts from database"""
        # Implement database query
        return []
    
    async def _store_social_post(self, db: Session, post: SocialPost):
        """Store social post in database"""
        # Implement database storage
        pass

class RedditCollector:
    """Reddit data collector using PRAW"""
    
    def __init__(self):
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = os.getenv("REDDIT_USER_AGENT", "TradingPlatform/1.0")
        
        self.reddit = None
        
        if self.client_id and self.client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit API: {e}")
        else:
            logger.warning("Reddit API credentials not provided")
    
    def is_healthy(self) -> bool:
        """Check if Reddit collector is properly configured"""
        return self.reddit is not None
    
    async def collect_for_symbols(self, symbols: List[str], keywords: Optional[List[str]], db: Session):
        """Collect Reddit posts for given symbols"""
        if not self.reddit:
            logger.warning("Reddit API not available")
            return
        
        # Key trading subreddits
        subreddits = [
            'wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis',
            'ValueInvesting', 'pennystocks', 'StockMarket', 'trading'
        ]
        
        for symbol in symbols:
            try:
                await self.collect_symbol_posts(symbol, subreddits, keywords, db)
                await asyncio.sleep(2)  # Rate limiting
            except Exception as e:
                logger.error(f"Error collecting Reddit posts for {symbol}: {e}")
    
    async def collect_symbol_posts(self, symbol: str, subreddits: List[str], 
                                 keywords: Optional[List[str]], db: Session):
        """Collect Reddit posts for a specific symbol"""
        try:
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for symbol mentions
                    search_query = f"{symbol} OR ${symbol}"
                    if keywords:
                        search_query += " OR " + " OR ".join(keywords[:3])
                    
                    # Get recent posts
                    posts = subreddit.search(search_query, sort='new', time_filter='day', limit=50)
                    
                    for post in posts:
                        try:
                            social_post = self._process_reddit_post(post, symbol)
                            if social_post:
                                await self._store_social_post(db, social_post)
                        except Exception as e:
                            logger.error(f"Error processing Reddit post {post.id}: {e}")
                    
                    # Also check hot posts that might mention the symbol
                    hot_posts = subreddit.hot(limit=25)
                    for post in hot_posts:
                        if symbol.lower() in post.title.lower() or symbol.lower() in post.selftext.lower():
                            try:
                                social_post = self._process_reddit_post(post, symbol)
                                if social_post:
                                    await self._store_social_post(db, social_post)
                            except Exception as e:
                                logger.error(f"Error processing hot Reddit post {post.id}: {e}")
                                
                except Exception as e:
                    logger.error(f"Error accessing subreddit {subreddit_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting Reddit posts for {symbol}: {e}")
    
    def _process_reddit_post(self, post, symbol: str) -> Optional[SocialPost]:
        """Process individual Reddit post into SocialPost"""
        try:
            content = f"{post.title}\n{post.selftext}"
            
            engagement = {
                'upvotes': post.score,
                'comments': post.num_comments,
                'upvote_ratio': post.upvote_ratio
            }
            
            return SocialPost(
                id=post.id,
                platform="reddit",
                author=post.author.name if post.author else "unknown",
                content=content,
                timestamp=datetime.fromtimestamp(post.created_utc),
                url=f"https://reddit.com{post.permalink}",
                engagement=engagement
            )
            
        except Exception as e:
            logger.error(f"Error processing Reddit post: {e}")
            return None
    
    def get_recent_posts(self, db: Session, symbol: str, start_time: datetime, limit: int) -> List[SocialPost]:
        """Get recent posts from database"""
        return []
    
    async def _store_social_post(self, db: Session, post: SocialPost):
        """Store social post in database"""
        pass

class StockTwitsCollector:
    """StockTwits data collector using public API"""
    
    def __init__(self):
        self.base_url = "https://api.stocktwits.com/api/2"
        self.session = None
    
    def is_healthy(self) -> bool:
        """Check if StockTwits collector is available"""
        return True  # Public API, no auth required
    
    async def collect_for_symbols(self, symbols: List[str], keywords: Optional[List[str]], db: Session):
        """Collect StockTwits messages for given symbols"""
        async with httpx.AsyncClient() as client:
            self.session = client
            
            for symbol in symbols:
                try:
                    await self.collect_symbol_messages(symbol, db)
                    await asyncio.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error collecting StockTwits for {symbol}: {e}")
    
    async def collect_symbol_messages(self, symbol: str, db: Session):
        """Collect StockTwits messages for a specific symbol"""
        try:
            url = f"{self.base_url}/streams/symbol/{symbol}.json"
            
            response = await self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            messages = data.get('messages', [])
            
            for message in messages:
                try:
                    post = self._process_stocktwits_message(message, symbol)
                    if post:
                        await self._store_social_post(db, post)
                except Exception as e:
                    logger.error(f"Error processing StockTwits message {message.get('id')}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting StockTwits messages for {symbol}: {e}")
    
    def _process_stocktwits_message(self, message: Dict, symbol: str) -> Optional[SocialPost]:
        """Process StockTwits message into SocialPost"""
        try:
            user = message.get('user', {})
            
            # Extract sentiment from StockTwits (they provide it)
            entities = message.get('entities', {})
            sentiment_data = entities.get('sentiment', {})
            
            return SocialPost(
                id=str(message['id']),
                platform="stocktwits",
                author=user.get('username', 'unknown'),
                content=message.get('body', ''),
                timestamp=datetime.strptime(message['created_at'], '%Y-%m-%dT%H:%M:%SZ'),
                url=f"https://stocktwits.com/message/{message['id']}",
                engagement={
                    'likes': message.get('likes', {}).get('total', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing StockTwits message: {e}")
            return None
    
    def get_recent_posts(self, db: Session, symbol: str, start_time: datetime, limit: int) -> List[SocialPost]:
        """Get recent posts from database"""
        return []
    
    async def _store_social_post(self, db: Session, post: SocialPost):
        """Store social post in database"""
        pass

class NewsCollector:
    """News collector from multiple financial news sources"""
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # News sources
        self.news_sources = [
            'reuters', 'bloomberg', 'cnbc', 'marketwatch', 'yahoo-finance',
            'seeking-alpha', 'the-motley-fool', 'benzinga'
        ]
    
    def is_healthy(self) -> bool:
        """Check if news collector has at least one API key"""
        return bool(self.news_api_key or self.finnhub_api_key or self.alpha_vantage_key)
    
    async def collect_for_symbols(self, symbols: List[str], keywords: Optional[List[str]], db: Session):
        """Collect news articles for given symbols"""
        for symbol in symbols:
            try:
                # Collect from multiple sources
                if self.news_api_key:
                    await self._collect_newsapi(symbol, db)
                
                if self.finnhub_api_key:
                    await self._collect_finnhub_news(symbol, db)
                
                # Always try free sources
                await self._collect_yahoo_news(symbol, db)
                
                await asyncio.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting news for {symbol}: {e}")
    
    async def _collect_newsapi(self, symbol: str, db: Session):
        """Collect from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{symbol}" OR "{symbol} stock"',
                'sources': ','.join(self.news_sources),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'apiKey': self.news_api_key
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                articles = data.get('articles', [])
                
                for article in articles:
                    try:
                        news_article = self._process_news_article(article, symbol, 'newsapi')
                        if news_article:
                            await self._store_news_article(db, news_article)
                    except Exception as e:
                        logger.error(f"Error processing NewsAPI article: {e}")
                        
        except Exception as e:
            logger.error(f"Error collecting from NewsAPI for {symbol}: {e}")
    
    async def _collect_finnhub_news(self, symbol: str, db: Session):
        """Collect from Finnhub news API"""
        try:
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'token': self.finnhub_api_key
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                articles = response.json()
                
                for article in articles:
                    try:
                        news_article = self._process_finnhub_article(article, symbol)
                        if news_article:
                            await self._store_news_article(db, news_article)
                    except Exception as e:
                        logger.error(f"Error processing Finnhub article: {e}")
                        
        except Exception as e:
            logger.error(f"Error collecting from Finnhub for {symbol}: {e}")
    
    async def _collect_yahoo_news(self, symbol: str, db: Session):
        """Collect news from Yahoo Finance (free)"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for article in news[:20]:  # Limit to 20 articles
                try:
                    news_article = self._process_yahoo_article(article, symbol)
                    if news_article:
                        await self._store_news_article(db, news_article)
                except Exception as e:
                    logger.error(f"Error processing Yahoo article: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting Yahoo news for {symbol}: {e}")
    
    def _process_news_article(self, article: Dict, symbol: str, source: str) -> Optional[NewsArticle]:
        """Process news article from NewsAPI"""
        try:
            return NewsArticle(
                id=f"{source}_{hash(article['url'])}",
                title=article['title'],
                content=article.get('content', article.get('description', '')),
                author=article.get('author'),
                source=article.get('source', {}).get('name', source),
                url=article['url'],
                published_at=datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
            )
        except Exception as e:
            logger.error(f"Error processing news article: {e}")
            return None
    
    def _process_finnhub_article(self, article: Dict, symbol: str) -> Optional[NewsArticle]:
        """Process article from Finnhub"""
        try:
            return NewsArticle(
                id=f"finnhub_{article['id']}",
                title=article['headline'],
                content=article.get('summary', ''),
                source=article.get('source', 'Finnhub'),
                url=article['url'],
                published_at=datetime.fromtimestamp(article['datetime'])
            )
        except Exception as e:
            logger.error(f"Error processing Finnhub article: {e}")
            return None
    
    def _process_yahoo_article(self, article: Dict, symbol: str) -> Optional[NewsArticle]:
        """Process article from Yahoo Finance"""
        try:
            return NewsArticle(
                id=f"yahoo_{article['uuid']}",
                title=article['title'],
                content=article.get('summary', ''),
                source='Yahoo Finance',
                url=article['link'],
                published_at=datetime.fromtimestamp(article['providerPublishTime'])
            )
        except Exception as e:
            logger.error(f"Error processing Yahoo article: {e}")
            return None
    
    def get_recent_articles(self, db: Session, symbol: str, start_time: datetime, limit: int) -> List[NewsArticle]:
        """Get recent articles from database"""
        return []
    
    async def _store_news_article(self, db: Session, article: NewsArticle):
        """Store news article in database"""
        pass