from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

class StockPrice(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float = Field(alias="changePercent")
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    day_high: Optional[float] = Field(None, alias="dayHigh")
    day_low: Optional[float] = Field(None, alias="dayLow")
    previous_close: Optional[float] = Field(None, alias="previousClose")

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HistoricalData(BaseModel):
    date: str
    open: float
    high: float  
    low: float
    close: float
    volume: int

class SearchResult(BaseModel):
    symbol: str
    name: str
    exchange: Optional[str] = None
    type: Optional[str] = "stock"  # stock, etf, index, etc.

class CompanyProfile(BaseModel):
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    market_cap: Optional[int] = Field(None, alias="marketCap")
    beta: Optional[float] = None
    pe_ratio: Optional[float] = Field(None, alias="peRatio")
    dividend_yield: Optional[float] = Field(None, alias="dividendYield")
    employees: Optional[int] = None
    headquarters: Optional[str] = None

    class Config:
        allow_population_by_field_name = True

# Database models (SQLAlchemy)
from sqlalchemy import Column, String, Numeric, BigInteger, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Candle(Base):
    __tablename__ = "candles"
    
    symbol = Column(String, primary_key=True)
    ts = Column(DateTime(timezone=True), primary_key=True)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric) 
    close = Column(Numeric)
    volume = Column(BigInteger)

class CompanyProfileDB(Base):
    __tablename__ = "company_profiles"
    
    symbol = Column(String, primary_key=True)
    name = Column(String)
    sector = Column(String)
    industry = Column(String)
    description = Column(Text)
    website = Column(String)
    market_cap = Column(BigInteger)
    beta = Column(Numeric)
    pe_ratio = Column(Numeric)
    dividend_yield = Column(Numeric)
    employees = Column(Integer)
    headquarters = Column(String)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class SymbolDirectory(Base):
    __tablename__ = "symbol_directory"
    
    symbol = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    exchange = Column(String)
    type = Column(String, default="stock")  # stock, etf, index, crypto
    is_active = Column(String, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())