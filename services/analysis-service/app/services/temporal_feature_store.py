"""
Temporal Feature Store with Point-in-Time Enforcement

This module implements a production-grade temporal feature store with:
1. Point-in-time feature enforcement using temporal tables
2. UTC timestamp normalization with millisecond precision
3. Event/news timestamp synchronization
4. Temporal data integrity validation
5. Historical feature lookups with time travel
6. Data lineage tracking and audit trails
"""

import asyncio
import asyncpg
import logging
import pytz
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import json
import uuid

from ..core.feature_contracts import FeatureContract, FeatureContractValidator, ContractViolation

logger = logging.getLogger(__name__)

class TimestampPrecision(Enum):
    """Timestamp precision levels"""
    SECOND = "second"
    MILLISECOND = "millisecond" 
    MICROSECOND = "microsecond"
    NANOSECOND = "nanosecond"

class TemporalTableType(Enum):
    """Types of temporal tables"""
    SYSTEM_VERSIONED = "system_versioned"
    APPLICATION_TIME = "application_time"
    BITEMPORAL = "bitemporal"

@dataclass
class TemporalMetadata:
    """Metadata for temporal feature records"""
    feature_name: str
    symbol: str
    as_of_timestamp: datetime          # When feature became available (system time)
    effective_timestamp: datetime      # When underlying event occurred (business time)
    ingestion_timestamp: datetime      # When data was ingested into system
    source_timestamp: Optional[datetime] = None  # Original timestamp from source
    revision_number: int = 1
    is_active: bool = True
    lineage_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class FeatureValue:
    """Temporal feature value with full metadata"""
    metadata: TemporalMetadata
    value: Any
    confidence: Optional[float] = None
    quality_score: Optional[float] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PointInTimeQuery:
    """Point-in-time feature query specification"""
    features: List[str]
    symbols: List[str]
    as_of_time: datetime
    effective_time_window: Optional[Tuple[datetime, datetime]] = None
    include_revisions: bool = False
    quality_threshold: Optional[float] = None

class UTCTimestampNormalizer:
    """Handles UTC timestamp normalization with millisecond precision"""
    
    def __init__(self, default_precision: TimestampPrecision = TimestampPrecision.MILLISECOND):
        self.default_precision = default_precision
        self.timezone_mappings = {
            'EST': pytz.timezone('US/Eastern'),
            'PST': pytz.timezone('US/Pacific'),
            'GMT': pytz.UTC,
            'UTC': pytz.UTC,
            'CET': pytz.timezone('Europe/Paris'),
            'JST': pytz.timezone('Asia/Tokyo'),
            'HKT': pytz.timezone('Asia/Hong_Kong'),
            'SGT': pytz.timezone('Asia/Singapore')
        }
    
    def normalize_to_utc(self, 
                        timestamp: Union[datetime, str, int], 
                        source_timezone: Optional[str] = None,
                        precision: Optional[TimestampPrecision] = None) -> datetime:
        """
        Normalize timestamp to UTC with specified precision.
        
        Args:
            timestamp: Input timestamp in various formats
            source_timezone: Source timezone identifier
            precision: Target precision level
            
        Returns:
            UTC timestamp with specified precision
        """
        precision = precision or self.default_precision
        
        # Handle different input formats
        if isinstance(timestamp, str):
            dt = self._parse_string_timestamp(timestamp, source_timezone)
        elif isinstance(timestamp, int):
            dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        elif isinstance(timestamp, datetime):
            dt = self._handle_datetime_timezone(timestamp, source_timezone)
        else:
            raise ValueError(f"Unsupported timestamp format: {type(timestamp)}")
        
        # Ensure UTC timezone
        if dt.tzinfo is None:
            if source_timezone:
                source_tz = self._get_timezone(source_timezone)
                dt = source_tz.localize(dt)
            else:
                # Assume UTC if no timezone specified
                dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to UTC
        utc_dt = dt.astimezone(timezone.utc)
        
        # Apply precision truncation
        return self._apply_precision(utc_dt, precision)
    
    def _parse_string_timestamp(self, timestamp_str: str, source_timezone: Optional[str]) -> datetime:
        """Parse string timestamp with various formats"""
        # Common timestamp formats
        formats = [
            "%Y-%m-%d %H:%M:%S.%f",     # ISO with microseconds
            "%Y-%m-%d %H:%M:%S",        # ISO without microseconds
            "%Y-%m-%dT%H:%M:%S.%fZ",    # ISO 8601 with Z
            "%Y-%m-%dT%H:%M:%SZ",       # ISO 8601 without microseconds
            "%Y-%m-%dT%H:%M:%S.%f",     # ISO 8601 no Z
            "%Y-%m-%dT%H:%M:%S",        # ISO 8601 basic
            "%m/%d/%Y %H:%M:%S",        # US format
            "%d/%m/%Y %H:%M:%S",        # European format
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return dt
            except ValueError:
                continue
        
        # Try pandas for complex parsing
        try:
            return pd.to_datetime(timestamp_str).to_pydatetime()
        except:
            raise ValueError(f"Unable to parse timestamp: {timestamp_str}")
    
    def _handle_datetime_timezone(self, dt: datetime, source_timezone: Optional[str]) -> datetime:
        """Handle datetime timezone assignment"""
        if dt.tzinfo is None and source_timezone:
            source_tz = self._get_timezone(source_timezone)
            return source_tz.localize(dt)
        return dt
    
    def _get_timezone(self, timezone_str: str) -> pytz.BaseTzInfo:
        """Get timezone object from string identifier"""
        if timezone_str in self.timezone_mappings:
            return self.timezone_mappings[timezone_str]
        try:
            return pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone {timezone_str}, using UTC")
            return pytz.UTC
    
    def _apply_precision(self, dt: datetime, precision: TimestampPrecision) -> datetime:
        """Apply timestamp precision truncation"""
        if precision == TimestampPrecision.SECOND:
            return dt.replace(microsecond=0)
        elif precision == TimestampPrecision.MILLISECOND:
            # Round to nearest millisecond
            microseconds = dt.microsecond
            milliseconds = round(microseconds / 1000) * 1000
            return dt.replace(microsecond=min(milliseconds, 999000))
        elif precision == TimestampPrecision.MICROSECOND:
            return dt  # Already at microsecond precision
        elif precision == TimestampPrecision.NANOSECOND:
            # Python datetime doesn't support nanoseconds, return as-is
            return dt
        else:
            return dt

class TemporalFeatureStore:
    """Production temporal feature store with point-in-time enforcement"""
    
    def __init__(self, 
                 database_url: str,
                 contracts_validator: Optional[FeatureContractValidator] = None,
                 table_prefix: str = "temporal_features"):
        self.database_url = database_url
        self.contracts_validator = contracts_validator or FeatureContractValidator()
        self.table_prefix = table_prefix
        self.timestamp_normalizer = UTCTimestampNormalizer()
        self.connection_pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize database connection and create temporal tables"""
        self.connection_pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        await self._create_temporal_tables()
        await self._create_indexes()
        logger.info("Temporal feature store initialized")
    
    async def _create_temporal_tables(self):
        """Create temporal tables with system versioning"""
        async with self.connection_pool.acquire() as conn:
            # Main features table with temporal capabilities
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_prefix}_main (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    feature_name VARCHAR(255) NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    as_of_timestamp TIMESTAMPTZ NOT NULL,
                    effective_timestamp TIMESTAMPTZ NOT NULL,
                    ingestion_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    source_timestamp TIMESTAMPTZ,
                    revision_number INTEGER NOT NULL DEFAULT 1,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    lineage_id UUID NOT NULL,
                    
                    -- Feature value (JSONB for flexibility)
                    feature_value JSONB NOT NULL,
                    confidence DECIMAL(5,4),
                    quality_score DECIMAL(5,4),
                    source_metadata JSONB DEFAULT '{{}}',
                    
                    -- Temporal constraints
                    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    valid_to TIMESTAMPTZ NOT NULL DEFAULT 'infinity',
                    
                    -- Audit fields
                    created_by VARCHAR(255) DEFAULT current_user,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    -- Constraints
                    CONSTRAINT valid_time_range CHECK (valid_from <= valid_to),
                    CONSTRAINT valid_timestamps CHECK (effective_timestamp <= as_of_timestamp),
                    CONSTRAINT confidence_range CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
                    CONSTRAINT quality_range CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1))
                )
            """)
            
            # History table for temporal versioning
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_prefix}_history (
                    LIKE {self.table_prefix}_main INCLUDING ALL,
                    operation_type VARCHAR(10) NOT NULL, -- INSERT, UPDATE, DELETE
                    operation_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    operation_user VARCHAR(255) DEFAULT current_user
                )
            """)
            
            # Metadata table for feature definitions
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_prefix}_metadata (
                    feature_name VARCHAR(255) PRIMARY KEY,
                    feature_type VARCHAR(50) NOT NULL,
                    data_source VARCHAR(255) NOT NULL,
                    contract_version VARCHAR(50) NOT NULL,
                    
                    -- Point-in-time rules
                    as_of_ts_rule TEXT NOT NULL,
                    effective_ts_rule TEXT NOT NULL,
                    arrival_latency_minutes INTEGER NOT NULL,
                    point_in_time_rule TEXT NOT NULL,
                    
                    -- Validation rules
                    validation_rules JSONB DEFAULT '{{}}',
                    
                    -- Audit
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            
            # Lineage tracking table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_prefix}_lineage (
                    lineage_id UUID PRIMARY KEY,
                    parent_lineage_id UUID,
                    feature_name VARCHAR(255) NOT NULL,
                    computation_logic TEXT,
                    dependencies JSONB DEFAULT '[]',
                    execution_metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    FOREIGN KEY (parent_lineage_id) REFERENCES {self.table_prefix}_lineage(lineage_id)
                )
            """)
    
    async def _create_indexes(self):
        """Create optimized indexes for temporal queries"""
        async with self.connection_pool.acquire() as conn:
            # Primary temporal indexes
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_main_temporal
                ON {self.table_prefix}_main (feature_name, symbol, as_of_timestamp DESC, effective_timestamp DESC)
                WHERE is_active = TRUE
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_main_lookup
                ON {self.table_prefix}_main (symbol, as_of_timestamp DESC)
                WHERE is_active = TRUE
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_main_lineage
                ON {self.table_prefix}_main (lineage_id, revision_number)
            """)
            
            # GIN index for JSONB columns
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_main_feature_value
                ON {self.table_prefix}_main USING GIN (feature_value)
            """)
            
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_prefix}_main_metadata
                ON {self.table_prefix}_main USING GIN (source_metadata)
            """)
    
    async def store_feature(self, 
                           feature_name: str,
                           symbol: str,
                           value: Any,
                           effective_timestamp: datetime,
                           source_timestamp: Optional[datetime] = None,
                           confidence: Optional[float] = None,
                           quality_score: Optional[float] = None,
                           source_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a feature value with full temporal metadata.
        
        Args:
            feature_name: Name of the feature
            symbol: Symbol/identifier for the feature
            value: Feature value
            effective_timestamp: When the underlying event occurred
            source_timestamp: Original timestamp from data source
            confidence: Confidence score (0-1)
            quality_score: Data quality score (0-1)
            source_metadata: Additional metadata from source
            
        Returns:
            lineage_id: Unique identifier for this feature record
        """
        # Validate feature contract
        contract = self.contracts_validator.contracts.get(feature_name)
        if contract:
            violations = await self._validate_feature_contract(
                contract, symbol, value, effective_timestamp, source_timestamp
            )
            if violations:
                logger.warning(f"Contract violations for {feature_name}: {violations}")
        
        # Normalize timestamps to UTC with millisecond precision
        effective_timestamp_utc = self.timestamp_normalizer.normalize_to_utc(
            effective_timestamp, precision=TimestampPrecision.MILLISECOND
        )
        
        if source_timestamp:
            source_timestamp_utc = self.timestamp_normalizer.normalize_to_utc(
                source_timestamp, precision=TimestampPrecision.MILLISECOND
            )
        else:
            source_timestamp_utc = None
        
        # Calculate as_of_timestamp based on contract rules
        as_of_timestamp_utc = await self._calculate_as_of_timestamp(
            feature_name, effective_timestamp_utc, contract
        )
        
        ingestion_timestamp_utc = datetime.now(timezone.utc)
        lineage_id = str(uuid.uuid4())
        
        # Create temporal metadata
        metadata = TemporalMetadata(
            feature_name=feature_name,
            symbol=symbol,
            as_of_timestamp=as_of_timestamp_utc,
            effective_timestamp=effective_timestamp_utc,
            ingestion_timestamp=ingestion_timestamp_utc,
            source_timestamp=source_timestamp_utc,
            lineage_id=lineage_id
        )
        
        # Store in database
        async with self.connection_pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {self.table_prefix}_main (
                    feature_name, symbol, as_of_timestamp, effective_timestamp,
                    ingestion_timestamp, source_timestamp, lineage_id,
                    feature_value, confidence, quality_score, source_metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, 
                feature_name, symbol, as_of_timestamp_utc, effective_timestamp_utc,
                ingestion_timestamp_utc, source_timestamp_utc, lineage_id,
                json.dumps(value), confidence, quality_score, 
                json.dumps(source_metadata or {})
            )
        
        logger.debug(f"Stored feature {feature_name} for {symbol} with lineage_id {lineage_id}")
        return lineage_id
    
    async def point_in_time_lookup(self, query: PointInTimeQuery) -> Dict[str, Dict[str, FeatureValue]]:
        """
        Perform point-in-time feature lookup.
        
        Args:
            query: Point-in-time query specification
            
        Returns:
            Dictionary mapping symbol -> feature_name -> FeatureValue
        """
        as_of_time_utc = self.timestamp_normalizer.normalize_to_utc(
            query.as_of_time, precision=TimestampPrecision.MILLISECOND
        )
        
        results = {}
        
        async with self.connection_pool.acquire() as conn:
            # Build query conditions
            feature_conditions = " AND ".join([f"${i+3}" for i in range(len(query.features))])
            symbol_conditions = " AND ".join([f"${i+3+len(query.features)}" for i in range(len(query.symbols))])
            
            # Quality filter
            quality_filter = ""
            if query.quality_threshold:
                quality_filter = f" AND (quality_score IS NULL OR quality_score >= ${len(query.features) + len(query.symbols) + 3})"
            
            # Effective time window filter
            time_window_filter = ""
            params = [as_of_time_utc, as_of_time_utc] + query.features + query.symbols
            if query.effective_time_window:
                start_time, end_time = query.effective_time_window
                start_time_utc = self.timestamp_normalizer.normalize_to_utc(start_time)
                end_time_utc = self.timestamp_normalizer.normalize_to_utc(end_time)
                time_window_filter = f" AND effective_timestamp BETWEEN ${len(params)+1} AND ${len(params)+2}"
                params.extend([start_time_utc, end_time_utc])
            
            if query.quality_threshold:
                params.append(query.quality_threshold)
            
            sql = f"""
                WITH ranked_features AS (
                    SELECT 
                        feature_name, symbol, as_of_timestamp, effective_timestamp,
                        ingestion_timestamp, source_timestamp, revision_number,
                        lineage_id, feature_value, confidence, quality_score,
                        source_metadata,
                        ROW_NUMBER() OVER (
                            PARTITION BY feature_name, symbol 
                            ORDER BY as_of_timestamp DESC, revision_number DESC
                        ) as rn
                    FROM {self.table_prefix}_main
                    WHERE as_of_timestamp <= $1
                      AND valid_from <= $2
                      AND valid_to > $2
                      AND is_active = TRUE
                      AND feature_name = ANY($3)
                      AND symbol = ANY($4)
                      {quality_filter}
                      {time_window_filter}
                )
                SELECT * FROM ranked_features WHERE rn = 1
                ORDER BY symbol, feature_name
            """
            
            rows = await conn.fetch(sql, *params)
            
            for row in rows:
                symbol = row['symbol']
                feature_name = row['feature_name']
                
                if symbol not in results:
                    results[symbol] = {}
                
                # Create temporal metadata
                metadata = TemporalMetadata(
                    feature_name=feature_name,
                    symbol=symbol,
                    as_of_timestamp=row['as_of_timestamp'],
                    effective_timestamp=row['effective_timestamp'],
                    ingestion_timestamp=row['ingestion_timestamp'],
                    source_timestamp=row['source_timestamp'],
                    revision_number=row['revision_number'],
                    lineage_id=row['lineage_id']
                )
                
                # Create feature value
                feature_value = FeatureValue(
                    metadata=metadata,
                    value=json.loads(row['feature_value']),
                    confidence=float(row['confidence']) if row['confidence'] else None,
                    quality_score=float(row['quality_score']) if row['quality_score'] else None,
                    source_metadata=json.loads(row['source_metadata'] or '{}')
                )
                
                results[symbol][feature_name] = feature_value
        
        return results
    
    async def get_feature_history(self, 
                                 feature_name: str,
                                 symbol: str,
                                 start_time: datetime,
                                 end_time: datetime,
                                 include_revisions: bool = False) -> List[FeatureValue]:
        """Get historical feature values for a time range"""
        start_time_utc = self.timestamp_normalizer.normalize_to_utc(start_time)
        end_time_utc = self.timestamp_normalizer.normalize_to_utc(end_time)
        
        revision_filter = ""
        if not include_revisions:
            revision_filter = " AND revision_number = 1"
        
        async with self.connection_pool.acquire() as conn:
            sql = f"""
                SELECT * FROM {self.table_prefix}_main
                WHERE feature_name = $1
                  AND symbol = $2
                  AND effective_timestamp BETWEEN $3 AND $4
                  AND is_active = TRUE
                  {revision_filter}
                ORDER BY effective_timestamp ASC, revision_number ASC
            """
            
            rows = await conn.fetch(sql, feature_name, symbol, start_time_utc, end_time_utc)
            
            results = []
            for row in rows:
                metadata = TemporalMetadata(
                    feature_name=row['feature_name'],
                    symbol=row['symbol'],
                    as_of_timestamp=row['as_of_timestamp'],
                    effective_timestamp=row['effective_timestamp'],
                    ingestion_timestamp=row['ingestion_timestamp'],
                    source_timestamp=row['source_timestamp'],
                    revision_number=row['revision_number'],
                    lineage_id=row['lineage_id']
                )
                
                feature_value = FeatureValue(
                    metadata=metadata,
                    value=json.loads(row['feature_value']),
                    confidence=float(row['confidence']) if row['confidence'] else None,
                    quality_score=float(row['quality_score']) if row['quality_score'] else None,
                    source_metadata=json.loads(row['source_metadata'] or '{}')
                )
                
                results.append(feature_value)
        
        return results
    
    async def _calculate_as_of_timestamp(self, 
                                       feature_name: str,
                                       effective_timestamp: datetime,
                                       contract: Optional[FeatureContract]) -> datetime:
        """Calculate when a feature becomes available based on contract rules"""
        if not contract:
            # Default: feature available immediately
            return effective_timestamp
        
        # Apply arrival latency from contract
        latency_minutes = contract.arrival_latency_minutes
        as_of_time = effective_timestamp + timedelta(minutes=latency_minutes)
        
        # Apply specific as_of_ts_rule logic
        as_of_rule = contract.as_of_ts_rule.lower()
        
        if "market_close" in as_of_rule:
            # Feature available at next market close after effective time
            # For simplicity, assume 4 PM ET market close
            market_close_hour = 16
            as_of_date = as_of_time.date()
            if as_of_time.hour >= market_close_hour:
                as_of_date = as_of_date + timedelta(days=1)
            as_of_time = datetime.combine(as_of_date, datetime.min.time().replace(hour=market_close_hour))
            as_of_time = as_of_time.replace(tzinfo=timezone.utc)
        
        elif "next_day" in as_of_rule:
            # Feature available next day
            next_day = as_of_time.date() + timedelta(days=1)
            as_of_time = datetime.combine(next_day, datetime.min.time())
            as_of_time = as_of_time.replace(tzinfo=timezone.utc)
        
        elif "immediate" in as_of_rule:
            # Feature available immediately after latency
            pass  # as_of_time already calculated above
        
        return as_of_time
    
    async def _validate_feature_contract(self, 
                                       contract: FeatureContract,
                                       symbol: str,
                                       value: Any,
                                       effective_timestamp: datetime,
                                       source_timestamp: Optional[datetime]) -> List[ContractViolation]:
        """Validate feature against its contract"""
        violations = []
        
        # Validate value range
        if contract.validation_rules.valid_range:
            min_val, max_val = contract.validation_rules.valid_range
            if isinstance(value, (int, float)) and not (min_val <= value <= max_val):
                violations.append(ContractViolation(
                    feature_name=contract.feature_name,
                    violation_type="value_range",
                    severity="HIGH",
                    message=f"Value {value} outside valid range [{min_val}, {max_val}]",
                    actual_value=value,
                    expected_value=f"[{min_val}, {max_val}]"
                ))
        
        # Validate null handling
        if value is None and contract.validation_rules.null_handling == "reject":
            violations.append(ContractViolation(
                feature_name=contract.feature_name,
                violation_type="null_value",
                severity="CRITICAL",
                message="Null values not allowed for this feature",
                actual_value=None,
                expected_value="non-null"
            ))
        
        # Validate timestamp constraints
        now_utc = datetime.now(timezone.utc)
        if effective_timestamp > now_utc:
            violations.append(ContractViolation(
                feature_name=contract.feature_name,
                violation_type="future_timestamp",
                severity="CRITICAL",
                message="Effective timestamp cannot be in the future",
                actual_value=effective_timestamp,
                expected_value=f"<= {now_utc}"
            ))
        
        return violations
    
    async def close(self):
        """Close database connections"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Temporal feature store connections closed")

class EventNewsTimestampSynchronizer:
    """Synchronizes event and news timestamps with market data timestamps"""
    
    def __init__(self, temporal_store: TemporalFeatureStore):
        self.temporal_store = temporal_store
        self.timestamp_normalizer = UTCTimestampNormalizer()
    
    async def synchronize_event_timestamps(self, 
                                         events: List[Dict[str, Any]],
                                         reference_market_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Synchronize event timestamps with market data timeline.
        
        Args:
            events: List of event data with timestamps
            reference_market_data: Optional market data for synchronization
            
        Returns:
            Events with synchronized UTC timestamps
        """
        synchronized_events = []
        
        for event in events:
            synchronized_event = event.copy()
            
            # Extract and normalize event timestamp
            event_timestamp = event.get('timestamp') or event.get('event_time') or event.get('published_at')
            if not event_timestamp:
                logger.warning(f"No timestamp found in event: {event}")
                continue
            
            # Normalize to UTC with millisecond precision
            try:
                utc_timestamp = self.timestamp_normalizer.normalize_to_utc(
                    event_timestamp, 
                    source_timezone=event.get('timezone'),
                    precision=TimestampPrecision.MILLISECOND
                )
                
                synchronized_event['normalized_timestamp_utc'] = utc_timestamp
                synchronized_event['original_timestamp'] = event_timestamp
                synchronized_event['timezone_applied'] = event.get('timezone', 'UTC')
                
                # Calculate market alignment
                if reference_market_data:
                    market_alignment = await self._calculate_market_alignment(
                        utc_timestamp, reference_market_data
                    )
                    synchronized_event['market_alignment'] = market_alignment
                
                synchronized_events.append(synchronized_event)
                
            except Exception as e:
                logger.error(f"Failed to normalize timestamp for event {event}: {e}")
                continue
        
        return synchronized_events
    
    async def _calculate_market_alignment(self, 
                                        event_timestamp: datetime,
                                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how event timestamp aligns with market sessions"""
        # This would integrate with market calendar data
        # For now, provide basic alignment info
        return {
            "is_market_hours": self._is_market_hours(event_timestamp),
            "next_market_open": self._next_market_open(event_timestamp),
            "prev_market_close": self._prev_market_close(event_timestamp),
            "trading_session": self._get_trading_session(event_timestamp)
        }
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within market hours (simplified)"""
        # US market hours: 9:30 AM - 4:00 PM ET
        et_tz = pytz.timezone('US/Eastern')
        et_time = timestamp.astimezone(et_tz)
        
        # Check if weekday
        if et_time.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check if within market hours
        market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= et_time <= market_close
    
    def _next_market_open(self, timestamp: datetime) -> datetime:
        """Calculate next market open time"""
        et_tz = pytz.timezone('US/Eastern')
        et_time = timestamp.astimezone(et_tz)
        
        # Start with current day at 9:30 AM
        next_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # If already past today's open, move to next trading day
        if et_time >= next_open or et_time.weekday() >= 5:
            days_ahead = 1
            while True:
                next_date = et_time.date() + timedelta(days=days_ahead)
                if next_date.weekday() < 5:  # Monday=0, Friday=4
                    next_open = datetime.combine(next_date, next_open.time())
                    next_open = et_tz.localize(next_open)
                    break
                days_ahead += 1
        
        return next_open.astimezone(timezone.utc)
    
    def _prev_market_close(self, timestamp: datetime) -> datetime:
        """Calculate previous market close time"""
        et_tz = pytz.timezone('US/Eastern')
        et_time = timestamp.astimezone(et_tz)
        
        # Start with current day at 4:00 PM
        prev_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # If before today's close, move to previous trading day
        if et_time < prev_close or et_time.weekday() >= 5:
            days_back = 1
            while True:
                prev_date = et_time.date() - timedelta(days=days_back)
                if prev_date.weekday() < 5:  # Monday=0, Friday=4
                    prev_close = datetime.combine(prev_date, prev_close.time())
                    prev_close = et_tz.localize(prev_close)
                    break
                days_back += 1
        
        return prev_close.astimezone(timezone.utc)
    
    def _get_trading_session(self, timestamp: datetime) -> str:
        """Determine trading session for timestamp"""
        et_tz = pytz.timezone('US/Eastern')
        et_time = timestamp.astimezone(et_tz)
        
        if et_time.weekday() >= 5:
            return "weekend"
        
        hour = et_time.hour
        minute = et_time.minute
        
        if hour < 4:
            return "overnight"
        elif hour < 9 or (hour == 9 and minute < 30):
            return "pre_market"
        elif hour < 16:
            return "regular_hours"
        elif hour < 20:
            return "after_hours"
        else:
            return "overnight"