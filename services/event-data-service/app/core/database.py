"""
Database models and configuration for Event Data Service
"""

import os
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Date,
    Numeric,
    Integer,
    Boolean,
    Text,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import enum

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://trading_user:trading_pass@localhost:5432/trading_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Database dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)


# Enums
class EventType(enum.Enum):
    EARNINGS = "earnings"
    PRODUCT_LAUNCH = "product_launch"
    ANALYST_DAY = "analyst_day"
    REGULATORY_DECISION = "regulatory_decision"
    MERGER_ACQUISITION = "merger_acquisition"
    IPO = "ipo"
    CONFERENCE_CALL = "conference_call"
    GUIDANCE_UPDATE = "guidance_update"
    DIVIDEND_ANNOUNCEMENT = "dividend_announcement"
    STOCK_SPLIT = "stock_split"
    FDA_APPROVAL = "fda_approval"
    CLINICAL_TRIAL = "clinical_trial"


class ImpactLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EventStatus(enum.Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class HeadlineType(enum.Enum):
    BREAKING = "breaking"
    EARNINGS = "earnings"
    MERGER = "merger"
    REGULATORY = "regulatory"
    ANALYST = "analyst"
    GENERAL = "general"


# Database Models
class ScheduledEvent(Base):
    """Scheduled events/catalysts for companies"""
    __tablename__ = "scheduled_events"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    company_name = Column(String(255))
    event_type = Column(SQLEnum(EventType), nullable=False)
    event_date = Column(Date, nullable=False, index=True)
    event_time = Column(DateTime)  # More specific time if available
    title = Column(String(500), nullable=False)
    description = Column(Text)
    impact_level = Column(SQLEnum(ImpactLevel), default=ImpactLevel.MEDIUM)
    status = Column(SQLEnum(EventStatus), default=EventStatus.SCHEDULED)
    source = Column(String(100))  # Data source
    source_url = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Event-specific data
    metadata = Column(JSON)  # Store additional event-specific information
    
    # Outcome tracking
    actual_outcome = Column(Text)  # What actually happened
    outcome_timestamp = Column(DateTime)  # When outcome was recorded
    surprise_score = Column(Numeric(5, 4))  # Calculated surprise vs expectation


class HeadlineCapture(Base):
    """Real-time headline/news capture"""
    __tablename__ = "headline_captures"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    headline = Column(String(1000), nullable=False)
    source = Column(String(100), nullable=False)
    source_url = Column(String(1000))
    published_at = Column(DateTime, nullable=False, index=True)
    captured_at = Column(DateTime, default=datetime.utcnow)
    
    # Content analysis
    headline_type = Column(SQLEnum(HeadlineType), default=HeadlineType.GENERAL)
    relevance_score = Column(Numeric(3, 2))  # 0.00 to 1.00
    sentiment_score = Column(Numeric(3, 2))  # -1.00 to 1.00
    urgency_score = Column(Numeric(3, 2))   # 0.00 to 1.00
    
    # Full content (optional)
    article_content = Column(Text)
    content_summary = Column(Text)
    
    # Processing metadata
    keywords = Column(JSON)  # Extracted keywords
    entities = Column(JSON)  # Named entities
    metadata = Column(JSON)  # Additional processing data


class EventOutcome(Base):
    """Actual outcomes/results of scheduled events"""
    __tablename__ = "event_outcomes"
    
    id = Column(Integer, primary_key=True, index=True)
    scheduled_event_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    
    # Outcome details
    actual_outcome = Column(Text, nullable=False)
    outcome_timestamp = Column(DateTime, nullable=False)
    outcome_source = Column(String(100))
    outcome_url = Column(String(1000))
    
    # Quantitative results (if applicable)
    actual_value = Column(Numeric(15, 4))  # e.g., actual EPS
    expected_value = Column(Numeric(15, 4))  # e.g., consensus EPS
    surprise_percent = Column(Numeric(8, 4))  # Percentage surprise
    
    # Market reaction
    price_impact_1h = Column(Numeric(8, 4))  # Price change 1 hour after
    price_impact_1d = Column(Numeric(8, 4))  # Price change 1 day after
    volume_impact = Column(Numeric(8, 4))    # Volume surge factor
    
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


class SurpriseScore(Base):
    """Calculated surprise scores for events"""
    __tablename__ = "surprise_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    event_type = Column(SQLEnum(EventType), nullable=False)
    event_date = Column(Date, nullable=False)
    
    # Surprise calculation
    expected_value = Column(Numeric(15, 4))
    actual_value = Column(Numeric(15, 4))
    surprise_magnitude = Column(Numeric(8, 4), nullable=False)  # Raw surprise
    surprise_score = Column(Numeric(5, 4), nullable=False)      # Normalized 0-1
    
    # Context
    historical_volatility = Column(Numeric(8, 4))  # For normalization
    event_importance = Column(Numeric(3, 2))       # Event weight
    market_context = Column(String(50))            # Bull/bear market etc.
    
    # Calculation metadata
    calculation_method = Column(String(100))
    confidence_level = Column(Numeric(3, 2))
    data_sources = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)


# Data Classes for API responses
@dataclass
class EventData:
    """Event data structure for API responses"""
    id: int
    symbol: str
    company_name: Optional[str]
    event_type: str
    event_date: date
    event_time: Optional[datetime]
    title: str
    description: Optional[str]
    impact_level: str
    status: str
    source: Optional[str]
    source_url: Optional[str]
    metadata: Optional[Dict[str, Any]]
    actual_outcome: Optional[str]
    outcome_timestamp: Optional[datetime]
    surprise_score: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "event_type": self.event_type,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "title": self.title,
            "description": self.description,
            "impact_level": self.impact_level,
            "status": self.status,
            "source": self.source,
            "source_url": self.source_url,
            "metadata": self.metadata or {},
            "actual_outcome": self.actual_outcome,
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
            "surprise_score": float(self.surprise_score) if self.surprise_score else None,
        }


@dataclass
class HeadlineData:
    """Headline data structure for API responses"""
    id: int
    symbol: str
    headline: str
    source: str
    source_url: Optional[str]
    published_at: datetime
    captured_at: datetime
    headline_type: str
    relevance_score: Optional[float]
    sentiment_score: Optional[float]
    urgency_score: Optional[float]
    article_content: Optional[str]
    content_summary: Optional[str]
    keywords: Optional[List[str]]
    entities: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "headline": self.headline,
            "source": self.source,
            "source_url": self.source_url,
            "published_at": self.published_at.isoformat(),
            "captured_at": self.captured_at.isoformat(),
            "headline_type": self.headline_type,
            "relevance_score": float(self.relevance_score) if self.relevance_score else None,
            "sentiment_score": float(self.sentiment_score) if self.sentiment_score else None,
            "urgency_score": float(self.urgency_score) if self.urgency_score else None,
            "article_content": self.article_content,
            "content_summary": self.content_summary,
            "keywords": self.keywords or [],
            "entities": self.entities or {},
            "metadata": self.metadata or {},
        }


@dataclass
class SurpriseData:
    """Surprise score data structure for API responses"""
    id: int
    symbol: str
    event_type: str
    event_date: date
    expected_value: Optional[float]
    actual_value: Optional[float]
    surprise_magnitude: float
    surprise_score: float
    historical_volatility: Optional[float]
    event_importance: Optional[float]
    market_context: Optional[str]
    calculation_method: Optional[str]
    confidence_level: Optional[float]
    data_sources: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "event_type": self.event_type,
            "event_date": self.event_date.isoformat(),
            "expected_value": float(self.expected_value) if self.expected_value else None,
            "actual_value": float(self.actual_value) if self.actual_value else None,
            "surprise_magnitude": float(self.surprise_magnitude),
            "surprise_score": float(self.surprise_score),
            "historical_volatility": float(self.historical_volatility) if self.historical_volatility else None,
            "event_importance": float(self.event_importance) if self.event_importance else None,
            "market_context": self.market_context,
            "calculation_method": self.calculation_method,
            "confidence_level": float(self.confidence_level) if self.confidence_level else None,
            "data_sources": self.data_sources or [],
            "metadata": self.metadata or {},
        }