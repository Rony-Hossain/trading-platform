"""
Novelty Scoring and Source Credibility Weighting Service
Detects duplicate/similar content and assigns credibility weights to avoid double-counting
"""

import logging
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class NoveltyScorer:
    """
    Detects content novelty to avoid double-counting replicated news and social media posts
    """
    
    def __init__(self):
        # Configure similarity thresholds
        self.similarity_thresholds = {
            'exact_duplicate': 0.98,  # Almost identical content
            'high_similarity': 0.85,  # Very similar content
            'moderate_similarity': 0.70,  # Moderately similar content
            'low_similarity': 0.50    # Somewhat similar content
        }
        
        # Time windows for duplicate detection
        self.time_windows = {
            'news': timedelta(hours=24),      # News articles within 24 hours
            'social': timedelta(hours=6),     # Social posts within 6 hours
            'twitter': timedelta(hours=4),    # Twitter posts within 4 hours
            'reddit': timedelta(hours=8)      # Reddit posts within 8 hours
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by removing formatting, URLs, mentions, etc.
        
        Args:
            text: Raw text content
            
        Returns:
            Normalized text for comparison
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove extra whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity score between two text strings
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize both texts
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        if not norm_text1 or not norm_text2:
            return 0.0
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
        
        return similarity
    
    def generate_content_hash(self, content: str) -> str:
        """
        Generate a hash for content to enable quick duplicate detection
        
        Args:
            content: Text content to hash
            
        Returns:
            Content hash string
        """
        normalized = self.normalize_text(content)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def calculate_novelty_score(self, db: Session, content: str, symbol: str, 
                              platform: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Calculate novelty score for content based on similarity to recent content
        
        Args:
            db: Database session
            content: Content to analyze
            symbol: Stock symbol
            platform: Source platform
            timestamp: Content timestamp
            
        Returns:
            Dictionary with novelty metrics
        """
        from ..models.database_models import SentimentPost, SentimentNews
        
        # Determine time window based on platform
        time_window = self.time_windows.get(platform, self.time_windows['social'])
        cutoff_time = timestamp - time_window
        
        # Get recent content from the same symbol
        recent_posts = []
        recent_news = []
        
        # Query recent social media posts
        posts = db.query(SentimentPost).filter(
            and_(
                SentimentPost.symbol == symbol,
                SentimentPost.post_timestamp >= cutoff_time,
                SentimentPost.post_timestamp <= timestamp
            )
        ).order_by(desc(SentimentPost.post_timestamp)).limit(100).all()
        
        recent_posts = [post.content for post in posts if post.content]
        
        # Query recent news articles
        if platform in ['news', 'reuters', 'bloomberg', 'cnbc', 'marketwatch']:
            news = db.query(SentimentNews).filter(
                and_(
                    SentimentNews.symbol == symbol,
                    SentimentNews.published_at >= cutoff_time,
                    SentimentNews.published_at <= timestamp
                )
            ).order_by(desc(SentimentNews.published_at)).limit(50).all()
            
            recent_news = [f"{article.title}. {article.content}" for article in news 
                          if article.title and article.content]
        
        # Combine all recent content
        all_recent_content = recent_posts + recent_news
        
        if not all_recent_content:
            return {
                'novelty_score': 1.0,
                'similarity_matches': [],
                'duplicate_risk': 'none',
                'unique_content_hash': self.generate_content_hash(content)
            }
        
        # Calculate similarities
        similarities = []
        for recent_content in all_recent_content:
            similarity = self.calculate_text_similarity(content, recent_content)
            if similarity > self.similarity_thresholds['low_similarity']:
                similarities.append({
                    'similarity_score': similarity,
                    'matched_content': recent_content[:100] + "..." if len(recent_content) > 100 else recent_content
                })
        
        # Determine novelty score and duplicate risk
        if not similarities:
            novelty_score = 1.0
            duplicate_risk = 'none'
        else:
            max_similarity = max(sim['similarity_score'] for sim in similarities)
            
            if max_similarity >= self.similarity_thresholds['exact_duplicate']:
                novelty_score = 0.05
                duplicate_risk = 'exact_duplicate'
            elif max_similarity >= self.similarity_thresholds['high_similarity']:
                novelty_score = 0.25
                duplicate_risk = 'high_similarity'
            elif max_similarity >= self.similarity_thresholds['moderate_similarity']:
                novelty_score = 0.50
                duplicate_risk = 'moderate_similarity'
            else:
                novelty_score = 0.75
                duplicate_risk = 'low_similarity'
        
        return {
            'novelty_score': novelty_score,
            'similarity_matches': similarities[:5],  # Top 5 matches
            'duplicate_risk': duplicate_risk,
            'unique_content_hash': self.generate_content_hash(content),
            'comparison_count': len(all_recent_content)
        }

class SourceCredibilityWeights:
    """
    Assigns credibility weights to different sources to prioritize reliable information
    """
    
    def __init__(self):
        # Source credibility weights (0.0 to 1.0)
        self.source_weights = {
            # Tier 1: Highly credible financial news sources
            'reuters': 1.0,
            'bloomberg': 1.0,
            'wall_street_journal': 1.0,
            'financial_times': 0.95,
            'cnbc': 0.9,
            'marketwatch': 0.85,
            'seeking_alpha': 0.8,
            
            # Tier 2: General business news
            'yahoo_finance': 0.8,
            'google_finance': 0.8,
            'cnn_business': 0.75,
            'bbc_business': 0.8,
            'forbes': 0.75,
            
            # Tier 3: Social media platforms
            'twitter': 0.6,
            'reddit': 0.55,
            'stocktwits': 0.65,
            'discord': 0.5,
            
            # Tier 4: Unknown or unverified sources
            'unknown': 0.3,
            'user_generated': 0.4
        }
        
        # Author credibility modifiers
        self.author_modifiers = {
            'verified_journalist': 1.2,
            'financial_analyst': 1.15,
            'company_official': 1.1,
            'verified_account': 1.05,
            'high_follower_count': 1.02,
            'frequent_poster': 0.95,
            'new_account': 0.8,
            'suspicious_activity': 0.3
        }
        
        # Platform-specific engagement weights
        self.engagement_weights = {
            'twitter': {
                'retweets': 0.4,
                'likes': 0.3,
                'replies': 0.3
            },
            'reddit': {
                'upvotes': 0.6,
                'comments': 0.4
            },
            'news': {
                'shares': 0.5,
                'comments': 0.3,
                'views': 0.2
            }
        }
    
    def get_source_weight(self, source: str) -> float:
        """
        Get credibility weight for a source
        
        Args:
            source: Source name/platform
            
        Returns:
            Credibility weight between 0.0 and 1.0
        """
        # Normalize source name
        source_normalized = source.lower().replace(' ', '_').replace('-', '_')
        
        # Check for exact match first
        if source_normalized in self.source_weights:
            return self.source_weights[source_normalized]
        
        # Check for partial matches
        for known_source, weight in self.source_weights.items():
            if known_source in source_normalized or source_normalized in known_source:
                return weight
        
        # Default weight for unknown sources
        return self.source_weights['unknown']
    
    def calculate_author_credibility(self, author: str, platform: str, 
                                   metadata: Dict[str, Any]) -> float:
        """
        Calculate author credibility based on available metadata
        
        Args:
            author: Author name/handle
            platform: Source platform
            metadata: Additional metadata about the author
            
        Returns:
            Author credibility multiplier
        """
        if not metadata:
            return 1.0
        
        credibility_multiplier = 1.0
        
        # Check for verification status
        if metadata.get('verified', False):
            credibility_multiplier *= self.author_modifiers['verified_account']
        
        # Check follower count (for social media)
        if platform in ['twitter', 'reddit'] and 'follower_count' in metadata:
            follower_count = metadata.get('follower_count', 0)
            if follower_count > 100000:
                credibility_multiplier *= self.author_modifiers['high_follower_count']
            elif follower_count < 100:
                credibility_multiplier *= self.author_modifiers['new_account']
        
        # Check account age
        if 'account_age_days' in metadata:
            account_age = metadata.get('account_age_days', 365)
            if account_age < 30:
                credibility_multiplier *= self.author_modifiers['new_account']
        
        # Check posting frequency
        if 'posts_per_day' in metadata:
            posts_per_day = metadata.get('posts_per_day', 1)
            if posts_per_day > 50:  # Potentially spam/bot activity
                credibility_multiplier *= self.author_modifiers['suspicious_activity']
            elif posts_per_day > 10:
                credibility_multiplier *= self.author_modifiers['frequent_poster']
        
        return min(max(credibility_multiplier, 0.1), 2.0)  # Cap between 0.1 and 2.0
    
    def calculate_engagement_weight(self, platform: str, engagement: Dict[str, Any]) -> float:
        """
        Calculate engagement-based credibility weight
        
        Args:
            platform: Source platform
            engagement: Engagement metrics
            
        Returns:
            Engagement weight multiplier
        """
        if not engagement or platform not in self.engagement_weights:
            return 1.0
        
        platform_weights = self.engagement_weights[platform]
        total_engagement_score = 0.0
        max_possible_score = sum(platform_weights.values())
        
        for metric, weight in platform_weights.items():
            if metric in engagement:
                # Normalize engagement metrics using log scale
                raw_value = max(engagement[metric], 1)
                normalized_value = min(np.log10(raw_value) / 5.0, 1.0)  # Cap at log10(100000)
                total_engagement_score += normalized_value * weight
        
        # Convert to multiplier (0.8 to 1.5 range)
        engagement_multiplier = 0.8 + (total_engagement_score / max_possible_score) * 0.7
        
        return engagement_multiplier

class WeightedSentimentAggregator:
    """
    Enhanced sentiment aggregator that uses novelty scores and source credibility weights
    """
    
    def __init__(self):
        self.novelty_scorer = NoveltyScorer()
        self.credibility_weights = SourceCredibilityWeights()
    
    def calculate_weighted_sentiment_metrics(self, db: Session, symbol: str, 
                                           timeframe_hours: int = 24) -> Dict[str, Any]:
        """
        Calculate weighted sentiment metrics that account for novelty and credibility
        
        Args:
            db: Database session
            symbol: Stock symbol
            timeframe_hours: Hours to look back
            
        Returns:
            Weighted sentiment metrics
        """
        from ..models.database_models import SentimentPost, SentimentNews
        
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        # Get all sentiment data for the timeframe
        posts = db.query(SentimentPost).filter(
            and_(
                SentimentPost.symbol == symbol,
                SentimentPost.post_timestamp >= cutoff_time,
                SentimentPost.sentiment_score.isnot(None)
            )
        ).all()
        
        news = db.query(SentimentNews).filter(
            and_(
                SentimentNews.symbol == symbol,
                SentimentNews.published_at >= cutoff_time,
                SentimentNews.sentiment_score.isnot(None)
            )
        ).all()
        
        # Calculate weights for all items
        weighted_items = []
        
        # Process social media posts
        for post in posts:
            # Calculate novelty score
            novelty_result = self.novelty_scorer.calculate_novelty_score(
                db, post.content, symbol, post.platform, post.post_timestamp
            )
            
            # Calculate source credibility weight
            source_weight = self.credibility_weights.get_source_weight(post.platform)
            
            # Calculate author credibility
            author_metadata = post.analysis_metadata or {}
            author_weight = self.credibility_weights.calculate_author_credibility(
                post.author, post.platform, author_metadata
            )
            
            # Calculate engagement weight
            engagement_weight = self.credibility_weights.calculate_engagement_weight(
                post.platform, post.engagement or {}
            )
            
            # Combined weight
            total_weight = (
                novelty_result['novelty_score'] * 
                source_weight * 
                author_weight * 
                engagement_weight
            )
            
            weighted_items.append({
                'type': 'social',
                'sentiment_score': post.sentiment_score,
                'sentiment_label': post.sentiment_label,
                'confidence': post.confidence or 0.5,
                'weight': total_weight,
                'novelty_score': novelty_result['novelty_score'],
                'source_weight': source_weight,
                'author_weight': author_weight,
                'engagement_weight': engagement_weight,
                'duplicate_risk': novelty_result['duplicate_risk'],
                'timestamp': post.post_timestamp,
                'platform': post.platform
            })
        
        # Process news articles
        for article in news:
            # Calculate novelty score
            content = f"{article.title}. {article.content or ''}"
            novelty_result = self.novelty_scorer.calculate_novelty_score(
                db, content, symbol, article.source, article.published_at
            )
            
            # Calculate source credibility weight
            source_weight = self.credibility_weights.get_source_weight(article.source)
            
            # News articles typically have higher base credibility
            author_weight = 1.1 if article.author else 1.0
            engagement_weight = 1.0  # News engagement handled differently
            
            # Combined weight
            total_weight = (
                novelty_result['novelty_score'] * 
                source_weight * 
                author_weight * 
                engagement_weight
            )
            
            weighted_items.append({
                'type': 'news',
                'sentiment_score': article.sentiment_score,
                'sentiment_label': article.sentiment_label,
                'confidence': article.confidence or 0.5,
                'weight': total_weight,
                'novelty_score': novelty_result['novelty_score'],
                'source_weight': source_weight,
                'author_weight': author_weight,
                'engagement_weight': engagement_weight,
                'duplicate_risk': novelty_result['duplicate_risk'],
                'timestamp': article.published_at,
                'platform': article.source
            })
        
        if not weighted_items:
            return {
                'symbol': symbol,
                'timeframe_hours': timeframe_hours,
                'total_items': 0,
                'effective_weight': 0.0,
                'weighted_sentiment': 0.0,
                'confidence': 0.0,
                'sentiment_distribution': {'bullish': 0, 'bearish': 0, 'neutral': 0},
                'quality_metrics': {
                    'avg_novelty_score': 0.0,
                    'avg_source_weight': 0.0,
                    'duplicate_risk_distribution': {}
                }
            }
        
        # Calculate weighted metrics
        total_weight = sum(item['weight'] for item in weighted_items)
        
        if total_weight == 0:
            return {
                'symbol': symbol,
                'timeframe_hours': timeframe_hours,
                'total_items': len(weighted_items),
                'effective_weight': 0.0,
                'weighted_sentiment': 0.0,
                'confidence': 0.0,
                'sentiment_distribution': {'bullish': 0, 'bearish': 0, 'neutral': 0},
                'quality_metrics': {
                    'avg_novelty_score': 0.0,
                    'avg_source_weight': 0.0,
                    'duplicate_risk_distribution': {}
                }
            }
        
        # Calculate weighted average sentiment
        weighted_sentiment = sum(
            item['sentiment_score'] * item['weight'] for item in weighted_items
        ) / total_weight
        
        # Calculate weighted confidence
        weighted_confidence = sum(
            item['confidence'] * item['weight'] for item in weighted_items
        ) / total_weight
        
        # Calculate sentiment distribution
        sentiment_distribution = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        for item in weighted_items:
            if item['sentiment_label']:
                label_key = item['sentiment_label'].lower()
                if label_key in sentiment_distribution:
                    sentiment_distribution[label_key] += item['weight']
        
        # Quality metrics
        avg_novelty_score = np.mean([item['novelty_score'] for item in weighted_items])
        avg_source_weight = np.mean([item['source_weight'] for item in weighted_items])
        
        duplicate_risk_dist = Counter([item['duplicate_risk'] for item in weighted_items])
        
        return {
            'symbol': symbol,
            'timeframe_hours': timeframe_hours,
            'total_items': len(weighted_items),
            'effective_weight': total_weight,
            'weighted_sentiment': round(weighted_sentiment, 4),
            'confidence': round(weighted_confidence, 4),
            'sentiment_distribution': {
                k: round(v, 4) for k, v in sentiment_distribution.items()
            },
            'quality_metrics': {
                'avg_novelty_score': round(avg_novelty_score, 4),
                'avg_source_weight': round(avg_source_weight, 4),
                'duplicate_risk_distribution': dict(duplicate_risk_dist),
                'weight_distribution': {
                    'social_weight': sum(item['weight'] for item in weighted_items if item['type'] == 'social'),
                    'news_weight': sum(item['weight'] for item in weighted_items if item['type'] == 'news')
                }
            },
            'generated_at': datetime.now().isoformat()
        }