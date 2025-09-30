"""
Fundamentals Data Versioning & Training Constraints

Enforces separation between first-print and latest fundamentals data.
Prevents look-ahead bias in model training by ensuring only historically
available data is used. Part of Phase 3A institutional compliance framework.
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import numpy as np

logger = logging.getLogger(__name__)

class DataVersion(Enum):
    """Fundamentals data version types"""
    FIRST_PRINT = "first_print"
    LATEST = "latest"
    BOTH = "both"

class ConstraintType(Enum):
    """Training data constraint types"""
    FUNDAMENTALS_LAG = "fundamentals_lag"
    EARNINGS_LAG = "earnings_lag"
    REVISION_EXCLUSION = "revision_exclusion"
    DATA_QUALITY_THRESHOLD = "data_quality_threshold"

class ComplianceViolation(Enum):
    """Types of compliance violations"""
    INSUFFICIENT_LAG = "insufficient_lag"
    LOW_QUALITY_DATA = "low_quality_data"
    DATA_NOT_FOUND = "data_not_found"
    AMENDMENT_EXCLUDED = "amendment_excluded"
    REVISION_COUNT_EXCEEDED = "revision_count_exceeded"

@dataclass
class FundamentalsRecord:
    """Represents a fundamentals data record"""
    symbol: str
    report_date: date
    filing_date: date
    first_print_timestamp: datetime
    period_type: str
    fiscal_year: int
    
    # Financial metrics
    revenue: Optional[int] = None
    net_income: Optional[int] = None
    earnings_per_share: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    
    # Data quality
    data_quality_score: float = 1.0
    revision_count: int = 0
    amendment_flag: bool = False
    data_source: str = ""
    
    # Versioning info
    data_version: DataVersion = DataVersion.FIRST_PRINT
    latest_revision_timestamp: Optional[datetime] = None

@dataclass
class TrainingConstraint:
    """Training data constraint definition"""
    constraint_name: str
    constraint_type: ConstraintType
    min_lag_days: Optional[int] = None
    min_quality_score: Optional[float] = None
    max_revision_count: Optional[int] = None
    exclude_amendments: bool = False
    description: str = ""
    is_active: bool = True

@dataclass
class ComplianceResult:
    """Result of compliance checking"""
    compliant: bool
    violations: List[ComplianceViolation]
    constraint_details: Dict[str, Any] = field(default_factory=dict)
    
class FundamentalsVersioningService:
    """Service for managing fundamentals data versioning and training constraints"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self._constraints_cache: Dict[str, TrainingConstraint] = {}
        self._cache_updated: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)
        
    async def get_training_fundamentals(self,
                                      symbols: List[str],
                                      as_of_date: date,
                                      lookback_days: int = 365,
                                      data_version: DataVersion = DataVersion.FIRST_PRINT) -> List[FundamentalsRecord]:
        """
        Get fundamentals data suitable for model training with compliance checks
        """
        await self._ensure_constraints_cached()
        
        # Build query based on data version
        if data_version == DataVersion.FIRST_PRINT:
            table_name = "fundamentals_first_print"
            timestamp_col = "first_print_timestamp"
        elif data_version == DataVersion.LATEST:
            table_name = "fundamentals_latest" 
            timestamp_col = "latest_revision_timestamp"
        else:
            raise ValueError("Use get_training_fundamentals_comparison for BOTH versions")
        
        # Get active constraints
        lag_constraint = self._get_constraint(ConstraintType.FUNDAMENTALS_LAG)
        quality_constraint = self._get_constraint(ConstraintType.DATA_QUALITY_THRESHOLD)
        revision_constraint = self._get_constraint(ConstraintType.REVISION_EXCLUSION)
        
        min_lag_days = lag_constraint.min_lag_days if lag_constraint else 45
        min_quality = quality_constraint.min_quality_score if quality_constraint else 0.8
        exclude_amendments = revision_constraint.exclude_amendments if revision_constraint else False
        
        # Build SQL query
        placeholders = ', '.join([f':symbol_{i}' for i in range(len(symbols))])
        
        query = f"""
        SELECT 
            symbol, report_date, filing_date, {timestamp_col} as data_timestamp,
            period_type, fiscal_year, revenue, net_income, earnings_per_share,
            roe, roa, debt_to_equity, data_quality_score, revision_count,
            amendment_flag, data_source
        FROM {table_name}
        WHERE symbol IN ({placeholders})
        AND {timestamp_col} <= :cutoff_timestamp
        AND report_date >= :lookback_date
        AND data_quality_score >= :min_quality
        """
        
        if exclude_amendments:
            query += " AND amendment_flag = FALSE"
            
        query += f" ORDER BY symbol, report_date DESC"
        
        # Prepare parameters
        params = {
            **{f'symbol_{i}': symbol for i, symbol in enumerate(symbols)},
            'cutoff_timestamp': datetime.combine(as_of_date - timedelta(days=min_lag_days), datetime.min.time()),
            'lookback_date': as_of_date - timedelta(days=lookback_days),
            'min_quality': min_quality
        }
        
        # Execute query
        result = await self.db.execute(text(query), params)
        rows = result.fetchall()
        
        # Convert to FundamentalsRecord objects
        records = []
        for row in rows:
            record = FundamentalsRecord(
                symbol=row.symbol,
                report_date=row.report_date,
                filing_date=row.filing_date,
                first_print_timestamp=row.data_timestamp,
                period_type=row.period_type,
                fiscal_year=row.fiscal_year,
                revenue=row.revenue,
                net_income=row.net_income,
                earnings_per_share=row.earnings_per_share,
                roe=row.roe,
                roa=row.roa,
                debt_to_equity=row.debt_to_equity,
                data_quality_score=row.data_quality_score,
                revision_count=row.revision_count,
                amendment_flag=row.amendment_flag,
                data_source=row.data_source,
                data_version=data_version
            )
            records.append(record)
        
        # Log access for audit
        await self._log_data_access(
            symbols=symbols,
            date_range_start=as_of_date - timedelta(days=lookback_days),
            date_range_end=as_of_date,
            data_version=data_version,
            query_type="training_data",
            records_returned=len(records)
        )
        
        return records
    
    async def check_compliance(self,
                             symbol: str,
                             report_date: date,
                             usage_date: date,
                             data_version: DataVersion = DataVersion.FIRST_PRINT) -> ComplianceResult:
        """
        Check if fundamentals data usage complies with training constraints
        """
        await self._ensure_constraints_cached()
        
        violations = []
        constraint_details = {}
        
        # Get the fundamentals record
        if data_version == DataVersion.FIRST_PRINT:
            query = """
            SELECT first_print_timestamp, filing_date, data_quality_score, 
                   revision_count, amendment_flag
            FROM fundamentals_first_print
            WHERE symbol = :symbol AND report_date = :report_date
            LIMIT 1
            """
        else:
            query = """
            SELECT first_print_timestamp, latest_revision_timestamp, filing_date, 
                   data_quality_score, revision_count, amendment_flag
            FROM fundamentals_latest
            WHERE symbol = :symbol AND report_date = :report_date
            LIMIT 1
            """
        
        result = await self.db.execute(text(query), {
            'symbol': symbol,
            'report_date': report_date
        })
        row = result.fetchone()
        
        if not row:
            violations.append(ComplianceViolation.DATA_NOT_FOUND)
            return ComplianceResult(compliant=False, violations=violations, constraint_details=constraint_details)
        
        # Check lag constraint
        lag_constraint = self._get_constraint(ConstraintType.FUNDAMENTALS_LAG)
        if lag_constraint and lag_constraint.min_lag_days:
            data_timestamp = row.first_print_timestamp
            min_usage_date = data_timestamp.date() + timedelta(days=lag_constraint.min_lag_days)
            
            if usage_date < min_usage_date:
                violations.append(ComplianceViolation.INSUFFICIENT_LAG)
                constraint_details['required_lag_days'] = lag_constraint.min_lag_days
                constraint_details['earliest_usage_date'] = min_usage_date.isoformat()
        
        # Check data quality constraint
        quality_constraint = self._get_constraint(ConstraintType.DATA_QUALITY_THRESHOLD)
        if quality_constraint and quality_constraint.min_quality_score:
            if row.data_quality_score < quality_constraint.min_quality_score:
                violations.append(ComplianceViolation.LOW_QUALITY_DATA)
                constraint_details['actual_quality'] = row.data_quality_score
                constraint_details['required_quality'] = quality_constraint.min_quality_score
        
        # Check revision constraints
        revision_constraint = self._get_constraint(ConstraintType.REVISION_EXCLUSION)
        if revision_constraint:
            if revision_constraint.exclude_amendments and row.amendment_flag:
                violations.append(ComplianceViolation.AMENDMENT_EXCLUDED)
            
            if revision_constraint.max_revision_count is not None:
                if row.revision_count > revision_constraint.max_revision_count:
                    violations.append(ComplianceViolation.REVISION_COUNT_EXCEEDED)
                    constraint_details['actual_revisions'] = row.revision_count
                    constraint_details['max_allowed_revisions'] = revision_constraint.max_revision_count
        
        compliant = len(violations) == 0
        return ComplianceResult(compliant=compliant, violations=violations, constraint_details=constraint_details)
    
    async def get_fundamentals_comparison(self,
                                        symbols: List[str],
                                        start_date: date,
                                        end_date: date) -> pd.DataFrame:
        """
        Compare first-print vs latest fundamentals data for analysis
        """
        query = """
        SELECT 
            fp.symbol,
            fp.report_date,
            fp.period_type,
            fp.fiscal_year,
            fp.earnings_per_share as eps_first_print,
            fl.earnings_per_share as eps_latest,
            fp.revenue as revenue_first_print,
            fl.revenue as revenue_latest,
            fp.roe as roe_first_print,
            fl.roe as roe_latest,
            fp.first_print_timestamp,
            fl.latest_revision_timestamp,
            fl.revision_count,
            CASE 
                WHEN fp.earnings_per_share IS NOT NULL AND fl.earnings_per_share IS NOT NULL 
                THEN ABS(fl.earnings_per_share - fp.earnings_per_share) / ABS(fp.earnings_per_share) * 100
                ELSE NULL 
            END as eps_revision_pct,
            CASE 
                WHEN fp.revenue IS NOT NULL AND fl.revenue IS NOT NULL 
                THEN ABS(fl.revenue - fp.revenue) / ABS(fp.revenue) * 100
                ELSE NULL 
            END as revenue_revision_pct
        FROM fundamentals_first_print fp
        INNER JOIN fundamentals_latest fl ON (
            fp.symbol = fl.symbol 
            AND fp.report_date = fl.report_date 
            AND fp.period_type = fl.period_type
            AND fp.fiscal_year = fl.fiscal_year
        )
        WHERE fp.symbol = ANY(:symbols)
        AND fp.report_date BETWEEN :start_date AND :end_date
        ORDER BY fp.symbol, fp.report_date DESC
        """
        
        result = await self.db.execute(text(query), {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date
        })
        
        # Convert to DataFrame
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # Log access
        await self._log_data_access(
            symbols=symbols,
            date_range_start=start_date,
            date_range_end=end_date,
            data_version=DataVersion.BOTH,
            query_type="research",
            records_returned=len(df)
        )
        
        return df
    
    async def get_earnings_surprises(self,
                                   symbols: List[str],
                                   start_date: date,
                                   end_date: date,
                                   include_revisions: bool = True) -> pd.DataFrame:
        """
        Get earnings surprise data comparing consensus vs actual
        """
        query = """
        SELECT 
            symbol, report_date, period_type, fiscal_year,
            consensus_eps_estimate,
            actual_eps_first_print,
            actual_eps_latest,
            eps_surprise_first_print,
            eps_surprise_percentage_first_print,
            eps_surprise_latest,
            eps_surprise_percentage_latest,
            consensus_revenue_estimate,
            actual_revenue_first_print,
            actual_revenue_latest,
            revenue_surprise_percentage_first_print,
            revenue_surprise_percentage_latest,
            announcement_time,
            stock_price_before,
            stock_price_after_1day,
            stock_price_after_5day
        FROM earnings_surprises
        WHERE symbol = ANY(:symbols)
        AND report_date BETWEEN :start_date AND :end_date
        ORDER BY symbol, report_date DESC
        """
        
        result = await self.db.execute(text(query), {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date
        })
        
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # Calculate additional metrics
        if not df.empty:
            # Market reaction to earnings
            df['price_reaction_1d'] = (df['stock_price_after_1day'] / df['stock_price_before'] - 1) * 100
            df['price_reaction_5d'] = (df['stock_price_after_5day'] / df['stock_price_before'] - 1) * 100
            
            # Surprise magnitude categories
            df['surprise_magnitude'] = pd.cut(
                df['eps_surprise_percentage_first_print'].abs(),
                bins=[0, 5, 15, 30, float('inf')],
                labels=['Small', 'Medium', 'Large', 'Huge']
            )
        
        return df
    
    async def update_constraint(self, constraint: TrainingConstraint) -> bool:
        """
        Update or create a training data constraint
        """
        query = """
        INSERT INTO training_data_constraints (
            constraint_name, constraint_type, min_lag_days, min_quality_score,
            max_revision_count, exclude_amendments, description, is_active
        ) VALUES (
            :constraint_name, :constraint_type, :min_lag_days, :min_quality_score,
            :max_revision_count, :exclude_amendments, :description, :is_active
        )
        ON CONFLICT (constraint_name) DO UPDATE SET
            constraint_type = EXCLUDED.constraint_type,
            min_lag_days = EXCLUDED.min_lag_days,
            min_quality_score = EXCLUDED.min_quality_score,
            max_revision_count = EXCLUDED.max_revision_count,
            exclude_amendments = EXCLUDED.exclude_amendments,
            description = EXCLUDED.description,
            is_active = EXCLUDED.is_active,
            updated_at = NOW()
        """
        
        try:
            await self.db.execute(text(query), {
                'constraint_name': constraint.constraint_name,
                'constraint_type': constraint.constraint_type.value,
                'min_lag_days': constraint.min_lag_days,
                'min_quality_score': constraint.min_quality_score,
                'max_revision_count': constraint.max_revision_count,
                'exclude_amendments': constraint.exclude_amendments,
                'description': constraint.description,
                'is_active': constraint.is_active
            })
            await self.db.commit()
            
            # Clear cache to force reload
            self._constraints_cache.clear()
            self._cache_updated = None
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update constraint {constraint.constraint_name}: {e}")
            await self.db.rollback()
            return False
    
    async def _ensure_constraints_cached(self):
        """Ensure constraints are cached and up to date"""
        if (self._cache_updated is None or 
            datetime.utcnow() - self._cache_updated > self._cache_ttl):
            await self._load_constraints()
    
    async def _load_constraints(self):
        """Load all active constraints from database"""
        query = """
        SELECT constraint_name, constraint_type, min_lag_days, min_quality_score,
               max_revision_count, exclude_amendments, description, is_active
        FROM training_data_constraints
        WHERE is_active = TRUE
        """
        
        result = await self.db.execute(text(query))
        
        self._constraints_cache.clear()
        for row in result.fetchall():
            constraint = TrainingConstraint(
                constraint_name=row.constraint_name,
                constraint_type=ConstraintType(row.constraint_type),
                min_lag_days=row.min_lag_days,
                min_quality_score=row.min_quality_score,
                max_revision_count=row.max_revision_count,
                exclude_amendments=row.exclude_amendments,
                description=row.description,
                is_active=row.is_active
            )
            self._constraints_cache[row.constraint_name] = constraint
        
        self._cache_updated = datetime.utcnow()
        logger.info(f"Loaded {len(self._constraints_cache)} training constraints")
    
    def _get_constraint(self, constraint_type: ConstraintType) -> Optional[TrainingConstraint]:
        """Get the first active constraint of specified type"""
        for constraint in self._constraints_cache.values():
            if constraint.constraint_type == constraint_type and constraint.is_active:
                return constraint
        return None
    
    async def _log_data_access(self,
                             symbols: List[str],
                             date_range_start: date,
                             date_range_end: date,
                             data_version: DataVersion,
                             query_type: str,
                             records_returned: int,
                             user_id: str = "system"):
        """Log data access for audit trail"""
        query = """
        INSERT INTO fundamentals_access_log (
            user_id, query_type, symbols_requested, date_range_start, 
            date_range_end, data_version, records_returned
        ) VALUES (
            :user_id, :query_type, :symbols_requested, :date_range_start,
            :date_range_end, :data_version, :records_returned
        )
        """
        
        try:
            await self.db.execute(text(query), {
                'user_id': user_id,
                'query_type': query_type,
                'symbols_requested': symbols,
                'date_range_start': date_range_start,
                'date_range_end': date_range_end,
                'data_version': data_version.value,
                'records_returned': records_returned
            })
            await self.db.commit()
            
        except Exception as e:
            logger.warning(f"Failed to log data access: {e}")
            # Don't fail the main operation for logging issues
            await self.db.rollback()

# Utility functions for integration with training pipelines

async def get_compliant_fundamentals_dataframe(
    db_session: AsyncSession,
    symbols: List[str],
    as_of_date: date,
    lookback_days: int = 365
) -> pd.DataFrame:
    """
    Convenience function to get compliant fundamentals as DataFrame
    """
    service = FundamentalsVersioningService(db_session)
    records = await service.get_training_fundamentals(
        symbols=symbols,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        data_version=DataVersion.FIRST_PRINT
    )
    
    # Convert to DataFrame
    if not records:
        return pd.DataFrame()
    
    data = []
    for record in records:
        data.append({
            'symbol': record.symbol,
            'report_date': record.report_date,
            'filing_date': record.filing_date,
            'first_print_timestamp': record.first_print_timestamp,
            'period_type': record.period_type,
            'fiscal_year': record.fiscal_year,
            'revenue': record.revenue,
            'net_income': record.net_income,
            'earnings_per_share': record.earnings_per_share,
            'roe': record.roe,
            'roa': record.roa,
            'debt_to_equity': record.debt_to_equity,
            'data_quality_score': record.data_quality_score,
            'revision_count': record.revision_count,
            'amendment_flag': record.amendment_flag,
            'data_source': record.data_source
        })
    
    return pd.DataFrame(data)

async def validate_training_data_compliance(
    db_session: AsyncSession,
    symbol: str,
    report_date: date,
    usage_date: date
) -> Tuple[bool, List[str]]:
    """
    Simple compliance validation function
    Returns (is_compliant, violation_messages)
    """
    service = FundamentalsVersioningService(db_session)
    result = await service.check_compliance(symbol, report_date, usage_date)
    
    violation_messages = []
    for violation in result.violations:
        if violation == ComplianceViolation.INSUFFICIENT_LAG:
            violation_messages.append(f"Insufficient lag: earliest usage {result.constraint_details.get('earliest_usage_date', 'unknown')}")
        elif violation == ComplianceViolation.LOW_QUALITY_DATA:
            violation_messages.append(f"Low quality data: {result.constraint_details.get('actual_quality', 0):.2f} < {result.constraint_details.get('required_quality', 0):.2f}")
        elif violation == ComplianceViolation.DATA_NOT_FOUND:
            violation_messages.append("Fundamentals data not found")
        elif violation == ComplianceViolation.AMENDMENT_EXCLUDED:
            violation_messages.append("Amendment filings excluded from training")
        else:
            violation_messages.append(f"Compliance violation: {violation.value}")
    
    return result.compliant, violation_messages