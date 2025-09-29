"""
Sentiment data storage service
Handles storing and retrieving sentiment data from database
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
import asyncio

from ..core.database import SentimentPost, NewsArticle, SentimentAggregates, CollectionStatus
from ..models.database_models import ContentDeduplication, SourceCredibility
from ..models.schemas import SocialPost, NewsArticle as NewsArticleSchema, SentimentCreate
from ..services.sentiment_analyzer import SentimentAnalyzer
from ..services.novelty_scoring import NoveltyScorer, SourceCredibilityWeights, WeightedSentimentAggregator

logger = logging.getLogger(__name__)

class SentimentStorage:
    """Service for storing and retrieving sentiment data"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.novelty_scorer = NoveltyScorer()
        self.credibility_weights = SourceCredibilityWeights()
        self.weighted_aggregator = WeightedSentimentAggregator()
    
    async def store_social_post(self, db: Session, post: SocialPost, symbol: str) -> Optional[SentimentPost]:
        """Store social media post with sentiment analysis"""
        try:
            # Check if post already exists
            existing = db.query(SentimentPost).filter(
                and_(
                    SentimentPost.platform == post.platform,
                    SentimentPost.platform_post_id == post.id
                )
            ).first()
            
            if existing:
                logger.debug(f"Post {post.id} from {post.platform} already exists")
                return existing
            
            # Analyze sentiment
            sentiment_result = await self.sentiment_analyzer.analyze_text(post.content, symbol)
            
            # Calculate novelty score and credibility weights
            novelty_score = await self.novelty_scorer.calculate_novelty_score(db, post.content, symbol, 'social')
            source_weight = self.credibility_weights.get_source_weight(post.platform, 'social')
            author_weight = self.credibility_weights.get_author_weight(post.author, post.platform, post.engagement)
            engagement_weight = self.credibility_weights.calculate_engagement_weight(post.engagement)
            duplicate_risk = self.novelty_scorer.get_duplicate_risk_level(novelty_score)
            content_hash = self.novelty_scorer.get_content_hash(post.content)
            
            # Create database record
            db_post = SentimentPost(
                platform=post.platform,
                platform_post_id=post.id,
                symbol=symbol,
                author=post.author,
                content=post.content,
                url=post.url,
                sentiment_score=sentiment_result.compound,
                sentiment_label=sentiment_result.label,
                confidence=sentiment_result.confidence,
                engagement=post.engagement,
                analysis_metadata=sentiment_result.metadata,
                novelty_score=novelty_score,
                source_credibility_weight=source_weight,
                author_credibility_weight=author_weight,
                engagement_weight=engagement_weight,
                duplicate_risk=duplicate_risk,
                content_hash=content_hash,
                post_timestamp=post.timestamp,
                analyzed_at=datetime.utcnow()
            )
            
            db.add(db_post)
            db.commit()
            db.refresh(db_post)
            
            logger.info(f"Stored post {post.id} from {post.platform} for {symbol}")
            return db_post
            
        except Exception as e:
            logger.error(f"Error storing social post: {e}")
            db.rollback()
            return None
    
    async def store_news_article(self, db: Session, article: NewsArticleSchema, symbol: str) -> Optional[NewsArticle]:
        """Store news article with sentiment analysis"""
        try:
            # Check if article already exists
            existing = db.query(NewsArticle).filter(
                and_(
                    NewsArticle.source == article.source,
                    NewsArticle.article_id == article.id
                )
            ).first()
            
            if existing:
                logger.debug(f"Article {article.id} from {article.source} already exists")
                return existing
            
            # Analyze sentiment
            content_to_analyze = f"{article.title}\n{article.content}"
            sentiment_result = await self.sentiment_analyzer.analyze_text(content_to_analyze, symbol)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(article.title, article.content, symbol)
            
            # Calculate novelty score and credibility weights
            novelty_score = await self.novelty_scorer.calculate_novelty_score(db, content_to_analyze, symbol, 'news')
            source_weight = self.credibility_weights.get_source_weight(article.source, 'news')
            author_weight = self.credibility_weights.get_author_weight(article.author, 'news', {})
            engagement_weight = 1.0  # News articles don't have social engagement metrics
            duplicate_risk = self.novelty_scorer.get_duplicate_risk_level(novelty_score)
            content_hash = self.novelty_scorer.get_content_hash(content_to_analyze)
            
            # Create database record
            db_article = NewsArticle(
                source=article.source,
                article_id=article.id,
                symbol=symbol,
                title=article.title,
                content=article.content,
                author=article.author,
                url=article.url,
                sentiment_score=sentiment_result.compound,
                sentiment_label=sentiment_result.label,
                confidence=sentiment_result.confidence,
                relevance_score=relevance_score,
                analysis_metadata=sentiment_result.metadata,
                novelty_score=novelty_score,
                source_credibility_weight=source_weight,
                author_credibility_weight=author_weight,
                engagement_weight=engagement_weight,
                duplicate_risk=duplicate_risk,
                content_hash=content_hash,
                published_at=article.published_at,
                analyzed_at=datetime.utcnow()
            )
            
            db.add(db_article)
            db.commit()
            db.refresh(db_article)
            
            logger.info(f"Stored article {article.id} from {article.source} for {symbol}")
            return db_article
            
        except Exception as e:
            logger.error(f"Error storing news article: {e}")
            db.rollback()
            return None
    
    def _calculate_relevance_score(self, title: str, content: str, symbol: str) -> float:
        """Calculate how relevant an article is to a symbol"""
        try:
            title_lower = title.lower()
            content_lower = content.lower() if content else ""
            symbol_lower = symbol.lower()
            
            score = 0.0
            
            # Title mentions get high score
            if symbol_lower in title_lower:
                score += 0.5
            if f"${symbol}" in title:
                score += 0.3
            
            # Content mentions
            if symbol_lower in content_lower:
                score += 0.2
            
            # Financial keywords increase relevance
            financial_keywords = [
                'earnings', 'revenue', 'profit', 'stock', 'shares', 'dividend',
                'merger', 'acquisition', 'IPO', 'SEC', 'analyst', 'rating'
            ]
            
            for keyword in financial_keywords:
                if keyword in title_lower:
                    score += 0.1
                if keyword in content_lower:
                    score += 0.05
            
            return min(1.0, score)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5  # Default relevance
    
    def get_recent_posts(self, db: Session, symbol: str, platform: Optional[str] = None, 
                        hours: int = 24, limit: int = 100) -> List[SentimentPost]:
        """Get recent social media posts"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            query = db.query(SentimentPost).filter(
                and_(
                    SentimentPost.symbol == symbol,
                    SentimentPost.post_timestamp >= start_time
                )
            )
            
            if platform:
                query = query.filter(SentimentPost.platform == platform)
            
            return query.order_by(desc(SentimentPost.post_timestamp)).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Error getting recent posts: {e}")
            return []
    
    def get_recent_news(self, db: Session, symbol: str, hours: int = 24, 
                       limit: int = 50) -> List[NewsArticle]:
        """Get recent news articles"""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            return db.query(NewsArticle).filter(
                and_(
                    NewsArticle.symbol == symbol,
                    NewsArticle.published_at >= start_time
                )
            ).order_by(desc(NewsArticle.published_at)).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Error getting recent news: {e}")
            return []
    
    def get_sentiment_summary(self, db: Session, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Get sentiment summary for a symbol"""
        try:
            # Calculate time range
            time_delta = {
                "1h": timedelta(hours=1),
                "1d": timedelta(days=1),
                "1w": timedelta(weeks=1),
                "1m": timedelta(days=30)
            }.get(timeframe, timedelta(days=1))
            
            start_time = datetime.utcnow() - time_delta
            
            # Get posts
            posts = db.query(SentimentPost).filter(
                and_(
                    SentimentPost.symbol == symbol,
                    SentimentPost.post_timestamp >= start_time,
                    SentimentPost.sentiment_score.isnot(None)
                )
            ).all()
            
            # Get news
            news = db.query(NewsArticle).filter(
                and_(
                    NewsArticle.symbol == symbol,
                    NewsArticle.published_at >= start_time,
                    NewsArticle.sentiment_score.isnot(None)
                )
            ).all()
            
            # Calculate aggregates
            all_scores = [p.sentiment_score for p in posts] + [n.sentiment_score for n in news]
            
            if not all_scores:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "total_mentions": 0,
                    "average_sentiment": 0.0,
                    "sentiment_distribution": {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0},
                    "platform_breakdown": {},
                    "confidence": 0.0
                }
            
            avg_sentiment = sum(all_scores) / len(all_scores)
            
            # Count sentiment labels
            all_labels = [p.sentiment_label for p in posts] + [n.sentiment_label for n in news]
            sentiment_dist = {
                "BULLISH": all_labels.count("BULLISH"),
                "BEARISH": all_labels.count("BEARISH"),
                "NEUTRAL": all_labels.count("NEUTRAL")
            }
            
            # Platform breakdown
            platform_counts = {}
            for post in posts:
                platform_counts[post.platform] = platform_counts.get(post.platform, 0) + 1
            for article in news:
                platform_counts["news"] = platform_counts.get("news", 0) + 1
            
            # Calculate confidence (average of individual confidences)
            all_confidences = [p.confidence for p in posts if p.confidence] + \
                            [n.confidence for n in news if n.confidence]
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "total_mentions": len(all_scores),
                "average_sentiment": avg_sentiment,
                "sentiment_distribution": sentiment_dist,
                "platform_breakdown": platform_counts,
                "confidence": avg_confidence,
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {}
    
    def update_collection_status(self, db: Session, platform: str, symbol: str,
                               posts_collected: int = 0, errors: int = 0,
                               error_message: Optional[str] = None,
                               rate_limit_info: Optional[Dict] = None) -> None:
        """Update collection status tracking"""
        try:
            # Get or create status record
            status = db.query(CollectionStatus).filter(
                and_(
                    CollectionStatus.platform == platform,
                    CollectionStatus.symbol == symbol
                )
            ).first()
            
            if not status:
                status = CollectionStatus(
                    platform=platform,
                    symbol=symbol,
                    posts_collected=0,
                    errors_count=0
                )
                db.add(status)
            
            # Update metrics
            status.last_collection_at = datetime.utcnow()
            status.posts_collected += posts_collected
            status.errors_count += errors
            status.updated_at = datetime.utcnow()
            
            if error_message:
                status.last_error = error_message
                status.is_healthy = False
                status.health_message = f"Last error: {error_message}"
            else:
                status.is_healthy = True
                status.health_message = "OK"
            
            # Update rate limit info
            if rate_limit_info:
                status.rate_limit_remaining = rate_limit_info.get('remaining')
                status.rate_limit_total = rate_limit_info.get('total')
                if rate_limit_info.get('reset_at'):
                    status.rate_limit_reset_at = rate_limit_info['reset_at']
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error updating collection status: {e}")
            db.rollback()
    
    def get_collection_stats(self, db: Session) -> Dict[str, Any]:
        """Get overall collection statistics"""
        try:
            # Get status for all platforms
            statuses = db.query(CollectionStatus).all()
            
            stats = {
                "total_platforms": len(set(s.platform for s in statuses)),
                "total_symbols": len(set(s.symbol for s in statuses)),
                "healthy_platforms": len([s for s in statuses if s.is_healthy]),
                "total_posts_collected": sum(s.posts_collected for s in statuses),
                "total_errors": sum(s.errors_count for s in statuses),
                "platform_stats": {},
                "last_updated": max((s.updated_at for s in statuses), default=datetime.utcnow())
            }
            
            # Per-platform stats
            for status in statuses:
                platform = status.platform
                if platform not in stats["platform_stats"]:
                    stats["platform_stats"][platform] = {
                        "posts_collected": 0,
                        "errors": 0,
                        "symbols": [],
                        "is_healthy": True,
                        "last_collection": None
                    }
                
                platform_stats = stats["platform_stats"][platform]
                platform_stats["posts_collected"] += status.posts_collected
                platform_stats["errors"] += status.errors_count
                platform_stats["symbols"].append(status.symbol)
                platform_stats["is_healthy"] = platform_stats["is_healthy"] and status.is_healthy
                
                if not platform_stats["last_collection"] or status.last_collection_at > platform_stats["last_collection"]:
                    platform_stats["last_collection"] = status.last_collection_at
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    async def compute_aggregates(self, db: Session, symbol: str, timeframe: str = "1h") -> None:
        """Compute and store sentiment aggregates for fast querying"""
        try:
            # Determine bucket size
            bucket_sizes = {
                "1h": timedelta(hours=1),
                "1d": timedelta(days=1),
                "1w": timedelta(weeks=1)
            }
            
            bucket_size = bucket_sizes.get(timeframe, timedelta(hours=1))
            
            # Get the latest aggregate to know where to start
            latest_agg = db.query(SentimentAggregates).filter(
                and_(
                    SentimentAggregates.symbol == symbol,
                    SentimentAggregates.timeframe == timeframe
                )
            ).order_by(desc(SentimentAggregates.bucket_start)).first()
            
            start_time = latest_agg.bucket_start + bucket_size if latest_agg else \
                        datetime.utcnow() - timedelta(days=7)  # Start from 7 days ago
            
            # Generate buckets to compute
            current_time = datetime.utcnow()
            bucket_start = start_time
            
            while bucket_start < current_time:
                bucket_end = bucket_start + bucket_size
                
                # Get posts in this bucket
                posts = db.query(SentimentPost).filter(
                    and_(
                        SentimentPost.symbol == symbol,
                        SentimentPost.post_timestamp >= bucket_start,
                        SentimentPost.post_timestamp < bucket_end,
                        SentimentPost.sentiment_score.isnot(None)
                    )
                ).all()
                
                # Get news in this bucket
                news = db.query(NewsArticle).filter(
                    and_(
                        NewsArticle.symbol == symbol,
                        NewsArticle.published_at >= bucket_start,
                        NewsArticle.published_at < bucket_end,
                        NewsArticle.sentiment_score.isnot(None)
                    )
                ).all()
                
                if posts or news:
                    # Calculate weighted aggregates using the weighted aggregator
                    weighted_metrics = self.weighted_aggregator.calculate_weighted_sentiment_metrics(
                        posts, news
                    )
                    
                    # Platform breakdown
                    platform_breakdown = {}
                    for post in posts:
                        platform_breakdown[post.platform] = platform_breakdown.get(post.platform, 0) + 1
                    if news:
                        platform_breakdown["news"] = len(news)
                    
                    # Total engagement
                    total_engagement = 0
                    for post in posts:
                        if post.engagement:
                            total_engagement += sum(post.engagement.values())
                    
                    # Create aggregate record with quality-weighted metrics
                    aggregate = SentimentAggregates(
                        symbol=symbol,
                        timeframe=timeframe,
                        bucket_start=bucket_start,
                        avg_sentiment=weighted_metrics['traditional_avg_sentiment'],
                        total_mentions=weighted_metrics['total_mentions'],
                        bullish_count=weighted_metrics['bullish_count'],
                        bearish_count=weighted_metrics['bearish_count'],
                        neutral_count=weighted_metrics['neutral_count'],
                        platform_breakdown=platform_breakdown,
                        total_engagement=total_engagement,
                        weighted_avg_sentiment=weighted_metrics['weighted_avg_sentiment'],
                        total_effective_weight=weighted_metrics['total_effective_weight'],
                        quality_score=weighted_metrics['quality_score'],
                        novelty_distribution=weighted_metrics['novelty_distribution'],
                        credibility_distribution=weighted_metrics['credibility_distribution'],
                        duplicate_count=weighted_metrics['duplicate_count']
                    )
                    
                    # Check if aggregate already exists
                    existing = db.query(SentimentAggregates).filter(
                        and_(
                            SentimentAggregates.symbol == symbol,
                            SentimentAggregates.timeframe == timeframe,
                            SentimentAggregates.bucket_start == bucket_start
                        )
                    ).first()
                    
                    if existing:
                        # Update existing
                        for attr in ['avg_sentiment', 'total_mentions', 'bullish_count', 
                                   'bearish_count', 'neutral_count', 'platform_breakdown', 
                                   'total_engagement']:
                            setattr(existing, attr, getattr(aggregate, attr))
                        existing.computed_at = datetime.utcnow()
                    else:
                        # Add new
                        db.add(aggregate)
                
                bucket_start = bucket_end
            
            db.commit()
            logger.info(f"Computed aggregates for {symbol} ({timeframe})")
            
        except Exception as e:
            logger.error(f"Error computing aggregates: {e}")
            db.rollback()

# Global storage instance
sentiment_storage = SentimentStorage()
