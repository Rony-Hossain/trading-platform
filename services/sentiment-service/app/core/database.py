"""
Database configuration and models for sentiment service
"""

import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, JSON, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://trading_user:trading_pass@localhost:5432/trading_db")

# Create engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class SentimentPost(Base):
    """Social media posts with sentiment analysis"""
    __tablename__ = "sentiment_posts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform = Column(String(50), nullable=False, index=True)
    platform_post_id = Column(String(255), nullable=False)  # Platform's post ID
    symbol = Column(String(10), nullable=False, index=True)
    author = Column(String(255))
    content = Column(Text, nullable=False)
    url = Column(Text)
    
    # Sentiment analysis results
    sentiment_score = Column(Float)  # -1.0 to 1.0
    sentiment_label = Column(String(20))  # BULLISH, BEARISH, NEUTRAL
    confidence = Column(Float)  # 0.0 to 1.0
    
    # Engagement metrics (stored as JSON)
    engagement = Column(JSON)
    
    # Metadata from analysis (avoid reserved attribute name 'metadata')
    analysis_metadata = Column('metadata', JSON)
    
    # Novelty and credibility scoring
    novelty_score = Column(Float, default=1.0)  # 0.0 to 1.0
    source_credibility_weight = Column(Float, default=1.0)  # 0.0 to 2.0
    author_credibility_weight = Column(Float, default=1.0)  # 0.0 to 2.0
    engagement_weight = Column(Float, default=1.0)  # 0.0 to 2.0
    duplicate_risk = Column(String(20))  # none, low_similarity, etc.
    content_hash = Column(String(32))  # MD5 hash of normalized content
    
    # Timestamps
    post_timestamp = Column(DateTime, nullable=False)  # When post was created
    collected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    analyzed_at = Column(DateTime)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_sentiment_symbol_timestamp', 'symbol', 'post_timestamp'),
        Index('idx_sentiment_platform_timestamp', 'platform', 'post_timestamp'),
        Index('idx_sentiment_score_symbol', 'sentiment_score', 'symbol'),
        Index('idx_sentiment_collected_at', 'collected_at'),
        Index('idx_platform_post_unique', 'platform', 'platform_post_id', unique=True),
    )

class NewsArticle(Base):
    """News articles with sentiment analysis"""
    __tablename__ = "sentiment_news"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(100), nullable=False, index=True)
    article_id = Column(String(255), nullable=False)  # Source's article ID
    symbol = Column(String(10), nullable=False, index=True)
    
    title = Column(Text, nullable=False)
    content = Column(Text)
    author = Column(String(255))
    url = Column(Text, nullable=False)
    
    # Sentiment analysis results
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    confidence = Column(Float)
    
    # Article metadata
    relevance_score = Column(Float)  # How relevant to the symbol
    analysis_metadata = Column('metadata', JSON)
    
    # Novelty and credibility scoring
    novelty_score = Column(Float, default=1.0)  # 0.0 to 1.0
    source_credibility_weight = Column(Float, default=1.0)  # 0.0 to 2.0
    author_credibility_weight = Column(Float, default=1.0)  # 0.0 to 2.0
    engagement_weight = Column(Float, default=1.0)  # 0.0 to 2.0
    duplicate_risk = Column(String(20))  # none, low_similarity, etc.
    content_hash = Column(String(32))  # MD5 hash of normalized content
    
    # Timestamps
    published_at = Column(DateTime, nullable=False)
    collected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    analyzed_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_news_symbol_published', 'symbol', 'published_at'),
        Index('idx_news_source_published', 'source', 'published_at'),
        Index('idx_news_sentiment_symbol', 'sentiment_score', 'symbol'),
        Index('idx_source_article_unique', 'source', 'article_id', unique=True),
    )

class SentimentAggregates(Base):
    """Pre-computed sentiment aggregates for fast querying"""
    __tablename__ = "sentiment_aggregates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # 1h, 1d, 1w
    bucket_start = Column(DateTime, nullable=False)
    
    # Aggregate metrics
    avg_sentiment = Column(Float)
    total_mentions = Column(Integer)
    bullish_count = Column(Integer)
    bearish_count = Column(Integer)
    neutral_count = Column(Integer)
    
    # Platform breakdown
    platform_breakdown = Column(JSON)  # {"twitter": 50, "reddit": 30, ...}
    
    # Engagement totals
    total_engagement = Column(Integer)
    
    # Quality-weighted metrics
    weighted_avg_sentiment = Column(Float)
    total_effective_weight = Column(Float, default=0.0)
    quality_score = Column(Float, default=1.0)  # 0.0 to 1.0
    novelty_distribution = Column(JSON)
    credibility_distribution = Column(JSON)
    duplicate_count = Column(Integer, default=0)
    
    # Timestamps
    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_agg_symbol_timeframe_bucket', 'symbol', 'timeframe', 'bucket_start', unique=True),
        Index('idx_agg_bucket_start', 'bucket_start'),
    )

class CollectionStatus(Base):
    """Track collection status and metrics"""
    __tablename__ = "collection_status"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform = Column(String(50), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    
    # Collection metrics
    last_collection_at = Column(DateTime)
    posts_collected = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    last_error = Column(Text)
    
    # API status
    rate_limit_remaining = Column(Integer)
    rate_limit_total = Column(Integer)
    rate_limit_reset_at = Column(DateTime)
    
    # Health status
    is_healthy = Column(Boolean, default=True)
    health_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_collection_platform_symbol', 'platform', 'symbol', unique=True),
        Index('idx_collection_updated_at', 'updated_at'),
    )

def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# TimescaleDB specific functions (if TimescaleDB is available)
def setup_timescaledb_hypertables():
    """Convert tables to TimescaleDB hypertables for better time-series performance"""
    try:
        with engine.connect() as conn:
            # Check if TimescaleDB extension exists
            result = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
            if result.fetchone():
                # Convert tables to hypertables
                hypertables = [
                    ("sentiment_posts", "post_timestamp"),
                    ("sentiment_news", "published_at"),
                    ("sentiment_aggregates", "bucket_start")
                ]
                
                for table_name, time_column in hypertables:
                    try:
                        conn.execute(f"""
                            SELECT create_hypertable(
                                '{table_name}', 
                                '{time_column}',
                                chunk_time_interval => INTERVAL '1 day',
                                if_not_exists => TRUE
                            )
                        """)
                        print(f"Created hypertable for {table_name}")
                    except Exception as e:
                        print(f"Note: {table_name} hypertable creation skipped: {e}")
                
                # Set up retention policies
                retention_policies = [
                    ("sentiment_posts", "90 days"),
                    ("sentiment_news", "180 days"), 
                    ("sentiment_aggregates", "5 years")
                ]
                
                for table_name, retention in retention_policies:
                    try:
                        conn.execute(f"""
                            SELECT add_retention_policy(
                                '{table_name}',
                                INTERVAL '{retention}',
                                if_not_exists => TRUE
                            )
                        """)
                        print(f"Set retention policy for {table_name}: {retention}")
                    except Exception as e:
                        print(f"Note: Retention policy for {table_name} skipped: {e}")
                        
    except Exception as e:
        print(f"TimescaleDB setup skipped: {e}")

if __name__ == "__main__":
    create_tables()
    setup_timescaledb_hypertables()
