"""
BERTopic-based Topic Modeling Service for Financial Sentiment Data
Analyzes themes and topics in social media posts and news articles
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)

class FinancialTopicModeler:
    """
    BERTopic-based topic modeling specialized for financial sentiment analysis
    """
    
    def __init__(self):
        """Initialize the topic modeling components"""
        # Use a finance-tuned sentence transformer if available, otherwise use general one
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configure UMAP for topic modeling
        self.umap_model = UMAP(
            n_neighbors=15, 
            n_components=5, 
            min_dist=0.0, 
            metric='cosine',
            random_state=42
        )
        
        # Configure HDBSCAN for clustering
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=10, 
            metric='euclidean', 
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Initialize BERTopic
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=None,  # Use default
            verbose=False
        )
        
        # Cache for fitted models per symbol
        self.fitted_models = {}
        self.last_fit_time = {}
        
    def _get_sentiment_texts(self, db: Session, symbol: str, hours: int = 72) -> List[Dict[str, Any]]:
        """
        Retrieve sentiment posts and news for topic modeling
        
        Args:
            db: Database session
            symbol: Stock symbol
            hours: Hours to look back
            
        Returns:
            List of documents with metadata
        """
        from ..models.database_models import SentimentPost, SentimentNews
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Get social media posts
        posts = db.query(SentimentPost).filter(
            SentimentPost.symbol == symbol,
            SentimentPost.post_timestamp >= cutoff_time,
            SentimentPost.content.isnot(None)
        ).all()
        
        # Get news articles
        news = db.query(SentimentNews).filter(
            SentimentNews.symbol == symbol,
            SentimentNews.published_at >= cutoff_time,
            SentimentNews.content.isnot(None)
        ).all()
        
        documents = []
        
        # Add social media posts
        for post in posts:
            documents.append({
                'text': post.content,
                'type': 'social',
                'platform': post.platform,
                'timestamp': post.post_timestamp,
                'sentiment_score': post.sentiment_score,
                'sentiment_label': post.sentiment_label,
                'author': post.author,
                'url': post.url,
                'engagement': post.engagement
            })
        
        # Add news articles
        for article in news:
            # Use title + content for richer topic modeling
            content = f"{article.title}. {article.content}" if article.content else article.title
            documents.append({
                'text': content,
                'type': 'news',
                'platform': article.source,
                'timestamp': article.published_at,
                'sentiment_score': article.sentiment_score,
                'sentiment_label': article.sentiment_label,
                'author': article.author,
                'url': article.url,
                'relevance_score': getattr(article, 'relevance_score', None)
            })
        
        return documents
    
    def _preprocess_texts(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Preprocess texts for topic modeling
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Tuple of (processed_texts, metadata)
        """
        texts = []
        metadata = []
        
        for doc in documents:
            text = doc['text']
            if not text or len(text.strip()) < 10:  # Skip very short texts
                continue
                
            # Basic preprocessing - remove URLs, mentions, etc.
            # Keep it minimal for financial context
            processed_text = text.strip()
            
            # Skip if too short after processing
            if len(processed_text) < 10:
                continue
                
            texts.append(processed_text)
            metadata.append(doc)
        
        return texts, metadata
    
    def fit_topics(self, db: Session, symbol: str, hours: int = 72, force_refit: bool = False) -> Dict[str, Any]:
        """
        Fit topic model on recent documents for a symbol
        
        Args:
            db: Database session
            symbol: Stock symbol
            hours: Hours to look back for documents
            force_refit: Force refitting even if model exists
            
        Returns:
            Dictionary with topic fitting results
        """
        try:
            # Check if we need to refit
            model_key = f"{symbol}_{hours}h"
            current_time = datetime.now()
            
            if (not force_refit and 
                model_key in self.fitted_models and 
                model_key in self.last_fit_time and
                (current_time - self.last_fit_time[model_key]).total_seconds() < 3600):  # 1 hour cache
                
                logger.info(f"Using cached topic model for {symbol}")
                return {"status": "cached", "model_key": model_key}
            
            # Get documents
            documents = self._get_sentiment_texts(db, symbol, hours)
            if len(documents) < 20:  # Need minimum documents for meaningful topics
                return {
                    "status": "insufficient_data", 
                    "document_count": len(documents),
                    "minimum_required": 20
                }
            
            # Preprocess
            texts, metadata = self._preprocess_texts(documents)
            if len(texts) < 15:
                return {
                    "status": "insufficient_valid_texts", 
                    "valid_text_count": len(texts),
                    "minimum_required": 15
                }
            
            logger.info(f"Fitting topic model for {symbol} with {len(texts)} documents")
            
            # Fit the model
            topics, probabilities = self.topic_model.fit_transform(texts)
            
            # Store the fitted model
            self.fitted_models[model_key] = {
                "model": self.topic_model,
                "texts": texts,
                "metadata": metadata,
                "topics": topics,
                "probabilities": probabilities,
                "fit_time": current_time
            }
            self.last_fit_time[model_key] = current_time
            
            # Get topic info
            topic_info = self.topic_model.get_topic_info()
            
            return {
                "status": "fitted",
                "model_key": model_key,
                "document_count": len(texts),
                "topic_count": len(topic_info) - 1,  # Exclude outlier topic (-1)
                "topics_overview": topic_info.to_dict('records')[:10]  # Top 10 topics
            }
            
        except Exception as e:
            logger.error(f"Error fitting topics for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_current_topics(self, db: Session, symbol: str, hours: int = 24, top_n: int = 10) -> Dict[str, Any]:
        """
        Get current topics for a symbol
        
        Args:
            db: Database session  
            symbol: Stock symbol
            hours: Hours to look back
            top_n: Number of top topics to return
            
        Returns:
            Current topics analysis
        """
        try:
            # Ensure we have a fitted model (with longer lookback for training)
            training_hours = max(hours * 3, 72)  # Use at least 3x the analysis window for training
            fit_result = self.fit_topics(db, symbol, training_hours)
            
            if fit_result["status"] not in ["fitted", "cached"]:
                return fit_result
            
            model_key = fit_result["model_key"]
            model_data = self.fitted_models.get(model_key)
            
            if not model_data:
                return {"status": "error", "error": "Model data not found"}
            
            # Get recent documents for analysis
            recent_docs = self._get_sentiment_texts(db, symbol, hours)
            if not recent_docs:
                return {"status": "no_recent_data", "hours": hours}
            
            recent_texts, recent_metadata = self._preprocess_texts(recent_docs)
            if not recent_texts:
                return {"status": "no_valid_recent_texts", "hours": hours}
            
            # Transform recent texts with fitted model
            recent_topics, recent_probs = model_data["model"].transform(recent_texts)
            
            # Analyze topic distribution
            topic_counts = Counter(recent_topics)
            total_docs = len(recent_texts)
            
            # Get topic information
            topic_info = model_data["model"].get_topic_info()
            
            # Build results
            topics_analysis = []
            for topic_id, count in topic_counts.most_common(top_n):
                if topic_id == -1:  # Skip outliers
                    continue
                    
                # Get topic details
                topic_words = model_data["model"].get_topic(topic_id)
                topic_row = topic_info[topic_info['Topic'] == topic_id]
                
                # Calculate sentiment for this topic
                topic_docs_indices = [i for i, t in enumerate(recent_topics) if t == topic_id]
                topic_sentiments = [recent_metadata[i]['sentiment_score'] for i in topic_docs_indices 
                                 if recent_metadata[i]['sentiment_score'] is not None]
                
                avg_sentiment = np.mean(topic_sentiments) if topic_sentiments else None
                sentiment_dist = Counter([recent_metadata[i]['sentiment_label'] for i in topic_docs_indices
                                        if recent_metadata[i]['sentiment_label'] is not None])
                
                # Get sample documents
                sample_docs = [recent_texts[i] for i in topic_docs_indices[:3]]
                
                topics_analysis.append({
                    "topic_id": int(topic_id),
                    "topic_words": topic_words[:10],  # Top 10 words
                    "document_count": count,
                    "percentage": round((count / total_docs) * 100, 2),
                    "avg_sentiment": round(avg_sentiment, 3) if avg_sentiment is not None else None,
                    "sentiment_distribution": dict(sentiment_dist),
                    "sample_documents": sample_docs,
                    "representative_docs": model_data["model"].get_representative_docs(topic_id)[:2]
                })
            
            return {
                "status": "success",
                "symbol": symbol,
                "analysis_window_hours": hours,
                "total_documents": total_docs,
                "topics_found": len(topics_analysis),
                "topics": topics_analysis,
                "outliers": topic_counts.get(-1, 0),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing current topics for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_topic_history(self, db: Session, symbol: str, days: int = 7, window_hours: int = 24) -> Dict[str, Any]:
        """
        Get historical topic trends for a symbol
        
        Args:
            db: Database session
            symbol: Stock symbol  
            days: Number of days to analyze
            window_hours: Hours per analysis window
            
        Returns:
            Historical topic analysis
        """
        try:
            # Ensure we have a fitted model (using full period for training)
            training_hours = days * 24
            fit_result = self.fit_topics(db, symbol, training_hours)
            
            if fit_result["status"] not in ["fitted", "cached"]:
                return fit_result
            
            model_key = fit_result["model_key"]
            model_data = self.fitted_models.get(model_key)
            
            if not model_data:
                return {"status": "error", "error": "Model data not found"}
            
            # Create time windows
            end_time = datetime.now()
            windows = []
            
            for i in range(days * 24 // window_hours):
                window_end = end_time - timedelta(hours=i * window_hours)
                window_start = window_end - timedelta(hours=window_hours)
                windows.append((window_start, window_end))
            
            windows.reverse()  # Chronological order
            
            # Analyze each window
            historical_analysis = []
            
            for window_start, window_end in windows:
                # Get documents for this window
                from ..models.database_models import SentimentPost, SentimentNews
                
                posts = db.query(SentimentPost).filter(
                    SentimentPost.symbol == symbol,
                    SentimentPost.post_timestamp >= window_start,
                    SentimentPost.post_timestamp < window_end,
                    SentimentPost.content.isnot(None)
                ).all()
                
                news = db.query(SentimentNews).filter(
                    SentimentNews.symbol == symbol,
                    SentimentNews.published_at >= window_start,
                    SentimentNews.published_at < window_end,
                    SentimentNews.content.isnot(None)
                ).all()
                
                # Build documents list
                window_docs = []
                for post in posts:
                    window_docs.append({
                        'text': post.content,
                        'sentiment_score': post.sentiment_score,
                        'sentiment_label': post.sentiment_label,
                        'platform': post.platform
                    })
                
                for article in news:
                    content = f"{article.title}. {article.content}" if article.content else article.title
                    window_docs.append({
                        'text': content,
                        'sentiment_score': article.sentiment_score,
                        'sentiment_label': article.sentiment_label,
                        'platform': article.source
                    })
                
                if len(window_docs) < 5:  # Need minimum docs
                    historical_analysis.append({
                        "window_start": window_start.isoformat(),
                        "window_end": window_end.isoformat(),
                        "document_count": len(window_docs),
                        "status": "insufficient_data",
                        "topics": []
                    })
                    continue
                
                # Preprocess and transform
                texts, metadata = self._preprocess_texts(window_docs)
                if len(texts) < 3:
                    historical_analysis.append({
                        "window_start": window_start.isoformat(),
                        "window_end": window_end.isoformat(),
                        "document_count": len(window_docs),
                        "valid_texts": len(texts),
                        "status": "insufficient_valid_texts",
                        "topics": []
                    })
                    continue
                
                # Transform with fitted model
                topics, probs = model_data["model"].transform(texts)
                
                # Analyze topic distribution
                topic_counts = Counter(topics)
                total_docs = len(texts)
                
                window_topics = []
                for topic_id, count in topic_counts.most_common(5):  # Top 5 topics per window
                    if topic_id == -1:
                        continue
                        
                    topic_words = model_data["model"].get_topic(topic_id)
                    
                    # Calculate sentiment for this topic in this window
                    topic_docs_indices = [i for i, t in enumerate(topics) if t == topic_id]
                    topic_sentiments = [metadata[i]['sentiment_score'] for i in topic_docs_indices 
                                     if metadata[i]['sentiment_score'] is not None]
                    
                    avg_sentiment = np.mean(topic_sentiments) if topic_sentiments else None
                    
                    window_topics.append({
                        "topic_id": int(topic_id),
                        "topic_words": [word for word, _ in topic_words[:5]],  # Top 5 words
                        "document_count": count,
                        "percentage": round((count / total_docs) * 100, 2),
                        "avg_sentiment": round(avg_sentiment, 3) if avg_sentiment is not None else None
                    })
                
                historical_analysis.append({
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "document_count": total_docs,
                    "status": "success",
                    "topics": window_topics,
                    "outliers": topic_counts.get(-1, 0)
                })
            
            return {
                "status": "success",
                "symbol": symbol,
                "analysis_days": days,
                "window_hours": window_hours,
                "total_windows": len(historical_analysis),
                "windows": historical_analysis,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing topic history for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_topic_sentiment_correlation(self, db: Session, symbol: str, hours: int = 72) -> Dict[str, Any]:
        """
        Analyze correlation between topics and sentiment patterns
        
        Args:
            db: Database session
            symbol: Stock symbol
            hours: Hours to look back
            
        Returns:
            Topic-sentiment correlation analysis
        """
        try:
            # Get current topics
            topics_result = self.get_current_topics(db, symbol, hours)
            
            if topics_result["status"] != "success":
                return topics_result
            
            # Additional correlation analysis
            correlations = []
            
            for topic in topics_result["topics"]:
                topic_id = topic["topic_id"]
                avg_sentiment = topic.get("avg_sentiment")
                
                if avg_sentiment is not None:
                    # Classify topic sentiment tendency
                    if avg_sentiment > 0.2:
                        sentiment_tendency = "bullish"
                    elif avg_sentiment < -0.2:
                        sentiment_tendency = "bearish"  
                    else:
                        sentiment_tendency = "neutral"
                    
                    correlations.append({
                        "topic_id": topic_id,
                        "top_words": [word for word, _ in topic["topic_words"][:5]],
                        "avg_sentiment": avg_sentiment,
                        "sentiment_tendency": sentiment_tendency,
                        "document_count": topic["document_count"],
                        "market_relevance_score": self._calculate_market_relevance(topic["topic_words"])
                    })
            
            return {
                "status": "success", 
                "symbol": symbol,
                "analysis_hours": hours,
                "topic_sentiment_correlations": correlations,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing topic-sentiment correlation for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_market_relevance(self, topic_words: List[Tuple[str, float]]) -> float:
        """Calculate how relevant topic words are to market movements"""
        # Market-relevant keywords
        market_keywords = {
            'earnings', 'revenue', 'profit', 'loss', 'beat', 'miss', 'guidance', 
            'outlook', 'forecast', 'analyst', 'upgrade', 'downgrade', 'target',
            'buy', 'sell', 'hold', 'bullish', 'bearish', 'rally', 'drop',
            'acquisition', 'merger', 'ipo', 'partnership', 'deal', 'contract',
            'fda', 'approval', 'clinical', 'trial', 'drug', 'product', 'launch'
        }
        
        relevance_score = 0.0
        total_weight = 0.0
        
        for word, weight in topic_words[:10]:  # Top 10 words
            if word.lower() in market_keywords:
                relevance_score += weight * 2.0  # Double weight for market terms
            total_weight += weight
        
        return min(relevance_score / total_weight if total_weight > 0 else 0.0, 1.0)