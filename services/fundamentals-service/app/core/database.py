"""
Database models and connection for Fundamentals Service
Handles earnings data, SEC filings, and financial analysis storage
"""

from sqlalchemy import create_engine, Column, String, Integer, Date, DateTime, Boolean, Text, DECIMAL, BigInteger, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from typing import Optional
import os
import logging
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://trading_user:trading_pass@localhost:5432/trading_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Database Models
class EarningsEvent(Base):
    """Earnings events with estimates and actuals"""
    __tablename__ = "earnings_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    
    # Event details
    report_date = Column(Date, nullable=False, index=True)
    period_ending = Column(Date, nullable=False)
    period_type = Column(String(10), nullable=False)  # Q1, Q2, Q3, Q4, FY
    fiscal_year = Column(Integer, nullable=False)
    fiscal_quarter = Column(Integer)
    
    # Estimates and actuals
    estimated_eps = Column(DECIMAL(10, 4))
    actual_eps = Column(DECIMAL(10, 4))
    estimated_revenue = Column(BigInteger)
    actual_revenue = Column(BigInteger)
    surprise_percent = Column(DECIMAL(8, 4))
    
    # Event metadata
    announcement_time = Column(String(10))  # BMO, AMC, TAS
    status = Column(String(20), default='upcoming')
    guidance_updated = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

class QuarterlyPerformance(Base):
    """Quarterly financial performance metrics"""
    __tablename__ = "quarterly_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    quarter = Column(String(5), nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    period_ending = Column(Date, nullable=False)
    
    # Financial metrics
    revenue = Column(BigInteger)
    revenue_growth_yoy = Column(DECIMAL(8, 4))
    revenue_growth_qoq = Column(DECIMAL(8, 4))
    net_income = Column(BigInteger)
    earnings_per_share = Column(DECIMAL(10, 4))
    eps_growth_yoy = Column(DECIMAL(8, 4))
    
    # Margins
    gross_margin = Column(DECIMAL(8, 4))
    operating_margin = Column(DECIMAL(8, 4))
    net_margin = Column(DECIMAL(8, 4))
    
    # Profitability ratios
    roe = Column(DECIMAL(8, 4))
    roa = Column(DECIMAL(8, 4))
    roic = Column(DECIMAL(8, 4))
    
    # Cash flow
    free_cash_flow = Column(BigInteger)
    operating_cash_flow = Column(BigInteger)
    capex = Column(BigInteger)
    
    # Guidance
    guidance_revenue_low = Column(BigInteger)
    guidance_revenue_high = Column(BigInteger)
    guidance_eps_low = Column(DECIMAL(10, 4))
    guidance_eps_high = Column(DECIMAL(10, 4))
    
    # Balance sheet
    total_assets = Column(BigInteger)
    total_debt = Column(BigInteger)
    cash_and_equivalents = Column(BigInteger)
    shareholders_equity = Column(BigInteger)
    
    # Timestamps
    report_date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

class SECFiling(Base):
    """SEC filing documents with parsed data"""
    __tablename__ = "sec_filings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    cik = Column(String(20))
    
    # Filing details
    filing_type = Column(String(10), nullable=False)
    filing_date = Column(Date, nullable=False, index=True)
    period_end_date = Column(Date)
    fiscal_year = Column(Integer)
    fiscal_period = Column(String(10))
    
    # Filing content
    filing_url = Column(Text, nullable=False)
    raw_content = Column(Text)
    processed_data = Column(JSONB)
    
    # Extracted data
    income_statement = Column(JSONB)
    balance_sheet = Column(JSONB)
    cash_flow_statement = Column(JSONB)
    
    # Text analysis
    risk_factors = Column(ARRAY(Text))
    management_discussion = Column(Text)
    business_description = Column(Text)
    key_metrics = Column(JSONB)
    
    # Processing status
    processing_status = Column(String(20), default='pending')
    processing_error = Column(Text)
    
    # Metadata
    document_count = Column(Integer, default=0)
    size_kb = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

class EarningsTrend(Base):
    """Earnings trend analysis and quality scoring"""
    __tablename__ = "earnings_trends"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    analysis_date = Column(Date, nullable=False, server_default=func.current_date())
    
    # Revenue trends
    revenue_trend_direction = Column(String(20))
    revenue_avg_growth = Column(DECIMAL(8, 4))
    revenue_acceleration = Column(DECIMAL(8, 4))
    
    # EPS trends
    eps_trend_direction = Column(String(20))
    eps_avg_growth = Column(DECIMAL(8, 4))
    eps_acceleration = Column(DECIMAL(8, 4))
    
    # Margin trends
    margin_trend_direction = Column(String(20))
    current_margin = Column(DECIMAL(8, 4))
    avg_margin = Column(DECIMAL(8, 4))
    
    # Quality metrics
    earnings_surprise_rate = Column(DECIMAL(6, 4))
    consistency_score = Column(DECIMAL(6, 2))
    growth_quality = Column(String(20))
    growth_quality_score = Column(Integer)
    
    # Guidance analysis
    guidance_accuracy = Column(JSONB)
    
    # Analysis period
    quarters_analyzed = Column(Integer, default=8)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

class SectorEarnings(Base):
    """Sector-wide earnings performance analysis"""
    __tablename__ = "sector_earnings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    sector = Column(String(100), nullable=False, index=True)
    period = Column(String(20), nullable=False)
    analysis_date = Column(Date, nullable=False, server_default=func.current_date())
    
    # Sector metrics
    companies_count = Column(Integer, default=0)
    reporting_complete = Column(Integer, default=0)
    beat_estimates = Column(Integer, default=0)
    missed_estimates = Column(Integer, default=0)
    avg_surprise = Column(DECIMAL(8, 4))
    revenue_growth_avg = Column(DECIMAL(8, 4))
    margin_expansion = Column(Integer, default=0)
    guidance_raises = Column(Integer, default=0)
    guidance_lowers = Column(Integer, default=0)
    
    # Company details
    company_details = Column(JSONB)
    
    # Market impact
    total_market_cap = Column(BigInteger)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

class ConsensusEstimate(Base):
    """Analyst consensus records sourced from market data providers"""
    __tablename__ = "consensus_estimates"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    report_date = Column(Date, nullable=False, index=True)
    fiscal_period = Column(String(10), nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    analyst_count = Column(Integer)
    estimate_eps = Column(DECIMAL(10, 4))
    estimate_eps_high = Column(DECIMAL(10, 4))
    estimate_eps_low = Column(DECIMAL(10, 4))
    actual_eps = Column(DECIMAL(10, 4))
    surprise_percent = Column(DECIMAL(8, 4))
    estimate_revenue = Column(BigInteger)
    estimate_revenue_high = Column(BigInteger)
    estimate_revenue_low = Column(BigInteger)
    actual_revenue = Column(BigInteger)
    guidance_eps = Column(DECIMAL(10, 4))
    guidance_revenue = Column(BigInteger)
    source = Column(String(100))
    retrieved_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AnalystRevision(Base):
    """Analyst rating and price target revisions"""
    __tablename__ = "analyst_revisions"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    revision_date = Column(Date, nullable=False, index=True)
    analyst = Column(String(255))
    firm = Column(String(255))
    action = Column(String(50))
    from_rating = Column(String(50))
    to_rating = Column(String(50))
    old_price_target = Column(DECIMAL(12, 4))
    new_price_target = Column(DECIMAL(12, 4))
    rating_score = Column(DECIMAL(6, 3))
    notes = Column(Text)
    source = Column(String(100))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class InsiderTransaction(Base):
    """Form 4 insider transactions"""
    __tablename__ = "insider_transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    insider = Column(String(255), nullable=False)
    relationship = Column(String(255))
    transaction_date = Column(Date, nullable=False, index=True)
    transaction_type = Column(String(50))
    shares = Column(BigInteger)
    share_change = Column(BigInteger)
    price = Column(DECIMAL(14, 4))
    total_value = Column(BigInteger)
    filing_date = Column(Date)
    link = Column(Text)
    source = Column(String(100))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class InstitutionalHolding(Base):
    """13F institutional holdings"""
    __tablename__ = "institutional_holdings"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    institution_name = Column(String(255), nullable=False)
    institution_cik = Column(String(20))
    filing_date = Column(Date, nullable=False, index=True)
    quarter_end = Column(Date, nullable=False, index=True)
    shares_held = Column(BigInteger)
    market_value = Column(DECIMAL(15, 2))
    percentage_ownership = Column(DECIMAL(8, 4))
    shares_change = Column(BigInteger)
    shares_change_pct = Column(DECIMAL(8, 4))
    form13f_url = Column(Text)
    is_new_position = Column(Boolean, default=False)
    is_sold_out = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class OwnershipFlowAnalysis(Base):
    """Aggregated ownership flow metrics"""
    __tablename__ = "ownership_flow_analysis"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    analysis_date = Column(Date, nullable=False, index=True)
    period_days = Column(Integer, nullable=False)
    insider_buy_transactions = Column(Integer, default=0)
    insider_sell_transactions = Column(Integer, default=0)
    insider_net_shares = Column(BigInteger, default=0)
    insider_net_value = Column(DECIMAL(15, 2), default=0)
    insider_buy_value = Column(DECIMAL(15, 2), default=0)
    insider_sell_value = Column(DECIMAL(15, 2), default=0)
    institutions_increasing = Column(Integer, default=0)
    institutions_decreasing = Column(Integer, default=0)
    institutions_new_positions = Column(Integer, default=0)
    institutions_sold_out = Column(Integer, default=0)
    institutional_net_shares = Column(BigInteger, default=0)
    institutional_net_value = Column(DECIMAL(15, 2), default=0)
    cluster_buying_detected = Column(Boolean, default=False)
    cluster_selling_detected = Column(Boolean, default=False)
    smart_money_score = Column(DECIMAL(5, 4), nullable=False)
    confidence_level = Column(DECIMAL(5, 4), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class RevisionMomentum(Base):
    """Rolling analyst revision momentum metrics"""
    __tablename__ = "revision_momentum"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    analysis_period_start = Column(Date, nullable=False)
    analysis_period_end = Column(Date, nullable=False, index=True)
    days_analyzed = Column(Integer, nullable=False)
    total_revisions = Column(Integer, default=0)
    upgrades = Column(Integer, default=0)
    downgrades = Column(Integer, default=0)
    initiations = Column(Integer, default=0)
    net_rating_changes = Column(Integer, default=0)
    rating_momentum_score = Column(DECIMAL(5, 4), default=0)
    price_target_revisions = Column(Integer, default=0)
    price_target_increases = Column(Integer, default=0)
    price_target_decreases = Column(Integer, default=0)
    average_price_target_change_pct = Column(DECIMAL(8, 4), default=0)
    price_target_momentum_score = Column(DECIMAL(5, 4), default=0)
    consensus_rating_change = Column(DECIMAL(5, 4), default=0)
    consensus_price_target_change_pct = Column(DECIMAL(8, 4))
    consensus_eps_revision_pct = Column(DECIMAL(8, 4))
    revision_intensity = Column(DECIMAL(8, 6), default=0)
    momentum_acceleration = Column(DECIMAL(8, 6), default=0)
    conviction_score = Column(DECIMAL(5, 4), default=0)
    pre_earnings_momentum = Column(Boolean, default=False)
    unusual_activity_detected = Column(Boolean, default=False)
    smart_money_following = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class EarningsAlert(Base):
    """User earnings monitoring alerts"""
    __tablename__ = "earnings_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    symbol = Column(String(10), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True))  # For future user integration
    
    # Alert settings
    days_before_earnings = Column(Integer, default=7)
    surprise_threshold = Column(DECIMAL(6, 2), default=5.0)
    guidance_changes = Column(Boolean, default=True)
    revenue_miss = Column(Boolean, default=True)
    margin_compression_threshold = Column(DECIMAL(6, 2), default=2.0)
    
    # Alert status
    is_active = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

class EarningsCalendar(Base):
    """Earnings calendar for upcoming events"""
    __tablename__ = "earnings_calendar"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    calendar_date = Column(Date, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    market_cap_billions = Column(DECIMAL(10, 2))
    period_type = Column(String(10), nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    announcement_time = Column(String(10))
    estimated_eps = Column(DECIMAL(10, 4))
    estimated_revenue = Column(BigInteger)
    is_high_impact = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

# Database connection functions
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    
def get_db_session() -> Session:
    """Get database session (for direct use)"""
    return SessionLocal()

# Storage utility functions
class FundamentalsStorage:
    """Storage operations for fundamentals data"""
    
    def __init__(self):
        pass
    
    def store_earnings_event(self, db: Session, event_data: dict) -> EarningsEvent:
        """Store earnings event data"""
        try:
            # Check for existing event
            existing = db.query(EarningsEvent).filter(
                EarningsEvent.symbol == event_data['symbol'],
                EarningsEvent.fiscal_year == event_data['fiscal_year'],
                EarningsEvent.fiscal_quarter == event_data.get('fiscal_quarter'),
                EarningsEvent.period_type == event_data['period_type']
            ).first()
            
            if existing:
                # Update existing
                for key, value in event_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.now()
                db.commit()
                return existing
            else:
                # Create new
                event = EarningsEvent(**event_data)
                db.add(event)
                db.commit()
                db.refresh(event)
                return event
                
        except Exception as e:
            db.rollback()
            raise e
    
    def store_quarterly_performance(self, db: Session, performance_data: dict) -> QuarterlyPerformance:
        """Store quarterly performance data"""
        try:
            # Check for existing record
            existing = db.query(QuarterlyPerformance).filter(
                QuarterlyPerformance.symbol == performance_data['symbol'],
                QuarterlyPerformance.fiscal_year == performance_data['fiscal_year'],
                QuarterlyPerformance.quarter == performance_data['quarter']
            ).first()
            
            if existing:
                # Update existing
                for key, value in performance_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.now()
                db.commit()
                return existing
            else:
                # Create new
                performance = QuarterlyPerformance(**performance_data)
                db.add(performance)
                db.commit()
                db.refresh(performance)
                return performance
                
        except Exception as e:
            db.rollback()
            raise e
    
    def store_sec_filing(self, db: Session, filing_data: dict) -> SECFiling:
        """Store SEC filing data"""
        try:
            # Check for existing filing
            existing = db.query(SECFiling).filter(
                SECFiling.symbol == filing_data['symbol'],
                SECFiling.filing_type == filing_data['filing_type'],
                SECFiling.filing_date == filing_data['filing_date']
            ).first()
            
            if existing:
                # Update existing
                for key, value in filing_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.now()
                db.commit()
                return existing
            else:
                # Create new
                filing = SECFiling(**filing_data)
                db.add(filing)
                db.commit()
                db.refresh(filing)
                return filing
                
        except Exception as e:
            db.rollback()
            raise e
    
    def store_earnings_trend(self, db: Session, trend_data: dict) -> EarningsTrend:
        """Store earnings trend analysis"""
        try:
            # Check for existing analysis for today
            existing = db.query(EarningsTrend).filter(
                EarningsTrend.symbol == trend_data['symbol'],
                EarningsTrend.analysis_date == trend_data.get('analysis_date', date.today())
            ).first()
            
            if existing:
                # Update existing
                for key, value in trend_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                db.commit()
                return existing
            else:
                # Create new
                trend = EarningsTrend(**trend_data)
                db.add(trend)
                db.commit()
                db.refresh(trend)
                return trend
                
        except Exception as e:
            db.rollback()
            raise e
    
    def store_consensus_estimate(self, db: Session, estimate_data: dict) -> ConsensusEstimate:
        """Store or update analyst consensus records"""
        try:
            estimate_data['symbol'] = estimate_data['symbol'].upper()
            existing = db.query(ConsensusEstimate).filter(
                ConsensusEstimate.symbol == estimate_data['symbol'],
                ConsensusEstimate.report_date == estimate_data['report_date']
            ).first()

            if existing:
                for key, value in estimate_data.items():
                    if hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)
                existing.updated_at = datetime.now()
                db.commit()
                db.refresh(existing)
                return existing

            record = ConsensusEstimate(**estimate_data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record
        except Exception as exc:
            db.rollback()
            raise exc

    def get_consensus_estimates(self, db: Session, symbol: str, limit: int = 8) -> list[ConsensusEstimate]:
        query = (
            db.query(ConsensusEstimate)
            .filter(ConsensusEstimate.symbol == symbol.upper())
            .order_by(ConsensusEstimate.report_date.desc())
        )
        if limit:
            query = query.limit(limit)
        return query.all()

    def store_insider_transaction(self, db: Session, transaction_data: dict) -> InsiderTransaction:
        """Store or update insider transaction entries"""
        try:
            transaction_data['symbol'] = transaction_data['symbol'].upper()
            existing = db.query(InsiderTransaction).filter(
                InsiderTransaction.symbol == transaction_data['symbol'],
                InsiderTransaction.insider == transaction_data['insider'],
                InsiderTransaction.transaction_date == transaction_data['transaction_date'],
                InsiderTransaction.transaction_type == transaction_data.get('transaction_type')
            ).first()

            if existing:
                for key, value in transaction_data.items():
                    if hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)
                db.commit()
                db.refresh(existing)
                return existing

            record = InsiderTransaction(**transaction_data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record
        except Exception as exc:
            db.rollback()
            raise exc

    def get_insider_transactions(
        self, db: Session, symbol: str, limit: int = 50, start_date: Optional[date] = None
    ) -> list[InsiderTransaction]:
        query = db.query(InsiderTransaction).filter(InsiderTransaction.symbol == symbol.upper())
        if start_date:
            query = query.filter(InsiderTransaction.transaction_date >= start_date)
        query = query.order_by(InsiderTransaction.transaction_date.desc())
        if limit:
            query = query.limit(limit)
        return query.all()

    def store_analyst_revision(self, db: Session, revision_data: dict) -> AnalystRevision:
        """Store or update analyst revision records"""
        try:
            revision_data['symbol'] = revision_data['symbol'].upper()
            existing = db.query(AnalystRevision).filter(
                AnalystRevision.symbol == revision_data['symbol'],
                AnalystRevision.revision_date == revision_data['revision_date'],
                AnalystRevision.firm == revision_data.get('firm'),
                AnalystRevision.action == revision_data.get('action')
            ).first()

            if existing:
                for key, value in revision_data.items():
                    if hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)
                db.commit()
                db.refresh(existing)
                return existing

            record = AnalystRevision(**revision_data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record
        except Exception as exc:
            db.rollback()
            raise exc

    def get_analyst_revisions(
        self, db: Session, symbol: str, limit: int = 50, start_date: Optional[date] = None
    ) -> list[AnalystRevision]:
        query = db.query(AnalystRevision).filter(AnalystRevision.symbol == symbol.upper())
        if start_date:
            query = query.filter(AnalystRevision.revision_date >= start_date)
        query = query.order_by(AnalystRevision.revision_date.desc())
        if limit:
            query = query.limit(limit)
        return query.all()

    def get_earnings_events(self, db: Session, symbol: str, limit: int = 10) -> list:
        """Get earnings events for symbol"""
        return db.query(EarningsEvent).filter(
            EarningsEvent.symbol == symbol
        ).order_by(EarningsEvent.report_date.desc()).limit(limit).all()
    
    def get_quarterly_performance(self, db: Session, symbol: str, quarters: int = 8) -> list:
        """Get quarterly performance data"""
        return db.query(QuarterlyPerformance).filter(
            QuarterlyPerformance.symbol == symbol
        ).order_by(
            QuarterlyPerformance.fiscal_year.desc(),
            QuarterlyPerformance.quarter.desc()
        ).limit(quarters).all()
    
    def get_upcoming_earnings(self, db: Session, days_ahead: int = 30) -> list:
        """Get upcoming earnings from calendar"""
        from datetime import date, timedelta
        
        start_date = date.today()
        end_date = start_date + timedelta(days=days_ahead)
        
        return db.query(EarningsCalendar).filter(
            EarningsCalendar.calendar_date >= start_date,
            EarningsCalendar.calendar_date <= end_date
        ).order_by(EarningsCalendar.calendar_date.asc()).all()
    
    def get_sector_analysis(self, db: Session, sector: str, period: str = "current_quarter") -> Optional[SectorEarnings]:
        """Get latest sector analysis"""
        return db.query(SectorEarnings).filter(
            SectorEarnings.sector == sector,
            SectorEarnings.period == period
        ).order_by(SectorEarnings.analysis_date.desc()).first()
    
    def store_institutional_holding(self, db: Session, holding_data: dict) -> InstitutionalHolding:
        """Store or update institutional holding (13F) data"""
        try:
            holding_data['symbol'] = holding_data['symbol'].upper()
            holding_data['institution_cik'] = holding_data.get('institution_cik') or 'UNKNOWN'

            existing = db.query(InstitutionalHolding).filter(
                InstitutionalHolding.symbol == holding_data['symbol'],
                InstitutionalHolding.institution_cik == holding_data['institution_cik'],
                InstitutionalHolding.quarter_end == holding_data['quarter_end']
            ).first()

            if existing:
                for key, value in holding_data.items():
                    if hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)
                db.commit()
                db.refresh(existing)
                return existing

            record = InstitutionalHolding(**holding_data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record
        except Exception as exc:
            db.rollback()
            raise exc


    def store_ownership_flow_analysis(self, db: Session, flow_data: dict) -> OwnershipFlowAnalysis:

        """Store ownership flow analysis snapshot"""

        try:

            flow_data['symbol'] = flow_data['symbol'].upper()
            record = OwnershipFlowAnalysis(**flow_data)

            db.add(record)

            db.commit()

            db.refresh(record)

            return record

        except Exception as exc:

            db.rollback()

            raise exc





fundamentals_storage = FundamentalsStorage()
