from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Request Models
class BatchAnalysisRequest(BaseModel):
    symbols: List[str]
    analysis_type: str = "comprehensive"  # comprehensive, technical, forecast

# Technical Analysis Models
class MovingAverages(BaseModel):
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

class Oscillators(BaseModel):
    rsi_14: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None

class MACD(BaseModel):
    macd: Optional[float] = None
    signal: Optional[float] = None
    histogram: Optional[float] = None

class Signals(BaseModel):
    overall_signal: Optional[str] = None  # BUY, SELL, HOLD
    strength: Optional[float] = None  # 0.0 to 1.0
    individual_signals: Optional[Dict[str, str]] = None

class TechnicalAnalysisResponse(BaseModel):
    symbol: str
    current_price: Optional[float] = None
    moving_averages: Optional[MovingAverages] = None
    oscillators: Optional[Oscillators] = None
    macd: Optional[MACD] = None
    signals: Optional[Signals] = None
    calculated_at: datetime

# Pattern Recognition Models
class ChartPattern(BaseModel):
    pattern_type: str  # head_and_shoulders, triangle, flag, etc.
    direction: str  # bullish, bearish
    confidence: float  # 0.0 to 1.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    target_price: Optional[float] = None
    description: Optional[str] = None

class ChartPatternsResponse(BaseModel):
    symbol: str
    patterns: List[ChartPattern]
    calculated_at: datetime

# Advanced Indicators Models
class IndicatorValue(BaseModel):
    value: float
    signal: Optional[str] = None  # BUY, SELL, NEUTRAL
    metadata: Optional[Dict[str, float]] = None

class AdvancedIndicatorsResponse(BaseModel):
    symbol: str
    indicators: Dict[str, IndicatorValue]
    calculated_at: datetime

# Forecast Models
class PriceAnalysis(BaseModel):
    expected_return_1d: Optional[float] = None
    expected_return_5d: Optional[float] = None
    max_expected_gain: Optional[float] = None
    max_expected_loss: Optional[float] = None
    volatility_estimate: Optional[float] = None

class TrendForecast(BaseModel):
    direction: str  # UP, DOWN, SIDEWAYS
    confidence: str  # HIGH, MEDIUM, LOW
    probability: Optional[float] = None  # 0.0 to 1.0

class ForecastResponse(BaseModel):
    symbol: str
    predictions: List[float]
    prediction_dates: List[datetime] = Field(alias="dates")
    current_price: Optional[float] = None
    model_type: str
    price_analysis: Optional[PriceAnalysis] = None
    trend_forecast: Optional[TrendForecast] = None
    confidence_score: Optional[float] = None
    calculated_at: datetime

    class Config:
        allow_population_by_field_name = True

# Overall Assessment Models
class OverallAssessment(BaseModel):
    recommendation: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    score: float  # -1.0 to 1.0 (negative = bearish, positive = bullish)
    key_factors: List[str]
    risks: List[str]
    opportunities: List[str]

# Comprehensive Analysis Models
class ComprehensiveAnalysisResponse(BaseModel):
    symbol: str
    technical: Optional[TechnicalAnalysisResponse] = None
    patterns: Optional[ChartPatternsResponse] = None
    forecast: Optional[ForecastResponse] = None
    advanced_indicators: Optional[AdvancedIndicatorsResponse] = None
    overall_assessment: Optional[OverallAssessment] = None

# Database Models
from sqlalchemy import Column, String, Numeric, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class TechnicalAnalysisCache(Base):
    __tablename__ = "technical_analysis_cache"
    
    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    period = Column(String, nullable=False)
    analysis_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)

class ForecastCache(Base):
    __tablename__ = "forecast_cache"
    
    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    horizon = Column(Numeric, nullable=False)
    forecast_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)