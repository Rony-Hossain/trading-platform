"""
Point-in-Time Portfolio Universe Loader
Handles loading of portfolio constituents with proper survivorship bias prevention
Includes delisted securities to avoid survivorship bias in backtesting
"""

import logging
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncpg
import pandas as pd
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)


class UniverseType(Enum):
    SP500 = "sp500"
    SP400 = "sp400"  # Mid-cap
    SP600 = "sp600"  # Small-cap
    RUSSELL1000 = "russell1000"
    RUSSELL2000 = "russell2000"
    NASDAQ100 = "nasdaq100"
    CUSTOM = "custom"


class ListingStatus(Enum):
    ACTIVE = "active"
    DELISTED = "delisted"
    SUSPENDED = "suspended"
    MERGED = "merged"
    ACQUIRED = "acquired"
    BANKRUPT = "bankrupt"


@dataclass
class UniverseConstituent:
    """Represents a constituent of a portfolio universe at a point in time"""
    symbol: str
    universe_type: UniverseType
    effective_date: date
    expiration_date: Optional[date]
    listing_status: ListingStatus
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    weight: Optional[float] = None
    gics_sector: Optional[str] = None
    gics_industry: Optional[str] = None
    delisting_reason: Optional[str] = None
    successor_symbol: Optional[str] = None
    data_source: str = "manual"
    
    def is_active_on(self, check_date: date) -> bool:
        """Check if constituent was active on a given date"""
        if check_date < self.effective_date:
            return False
        if self.expiration_date and check_date > self.expiration_date:
            return False
        return True


@dataclass 
class SurvivorshipStats:
    """Statistics about survivorship bias in a universe"""
    total_symbols: int
    active_symbols: int
    delisted_symbols: int
    delisted_rate: float
    avg_listing_duration_years: float
    delisting_reasons: Dict[str, int]


class PortfolioUniverseLoader:
    """Loads and manages portfolio universe data with survivorship bias handling"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url
        self.data_sources = {
            UniverseType.SP500: self._load_sp500_constituents,
            UniverseType.NASDAQ100: self._load_nasdaq100_constituents,
            UniverseType.RUSSELL1000: self._load_russell_constituents,
        }
    
    async def get_connection(self) -> asyncpg.Connection:
        """Get database connection"""
        if not self.database_url:
            import os
            self.database_url = os.getenv('DATABASE_URL', 
                'postgresql://trading_user:trading_pass@localhost:5432/trading_db')
        return await asyncpg.connect(self.database_url)
    
    async def initialize_universe_tables(self):
        """Create universe tracking tables if they don't exist"""
        create_tables_sql = """
        -- Portfolio universe constituents table
        CREATE TABLE IF NOT EXISTS universe_constituents (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            universe_type VARCHAR(50) NOT NULL,
            effective_date DATE NOT NULL,
            expiration_date DATE,
            listing_status VARCHAR(20) NOT NULL DEFAULT 'active',
            sector VARCHAR(100),
            industry VARCHAR(100),
            market_cap DECIMAL(15,2),
            weight DECIMAL(8,6),
            gics_sector VARCHAR(100),
            gics_industry VARCHAR(100),
            delisting_reason VARCHAR(200),
            successor_symbol VARCHAR(20),
            data_source VARCHAR(50) NOT NULL DEFAULT 'manual',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            UNIQUE(symbol, universe_type, effective_date)
        );
        
        -- Index for efficient point-in-time lookups
        CREATE INDEX IF NOT EXISTS idx_universe_constituents_pit
        ON universe_constituents(universe_type, effective_date, expiration_date, symbol);
        
        CREATE INDEX IF NOT EXISTS idx_universe_constituents_symbol
        ON universe_constituents(symbol, effective_date);
        
        CREATE INDEX IF NOT EXISTS idx_universe_constituents_status
        ON universe_constituents(listing_status, universe_type);
        
        -- Universe snapshots for faster lookups
        CREATE TABLE IF NOT EXISTS universe_snapshots (
            id SERIAL PRIMARY KEY,
            universe_type VARCHAR(50) NOT NULL,
            snapshot_date DATE NOT NULL,
            constituents_count INTEGER NOT NULL,
            active_count INTEGER NOT NULL,
            delisted_count INTEGER NOT NULL,
            constituents_json JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            UNIQUE(universe_type, snapshot_date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_universe_snapshots_type_date
        ON universe_snapshots(universe_type, snapshot_date DESC);
        
        -- Delisting events tracking
        CREATE TABLE IF NOT EXISTS delisting_events (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            delisting_date DATE NOT NULL,
            delisting_reason VARCHAR(200),
            last_trading_date DATE,
            final_price DECIMAL(10,4),
            successor_symbol VARCHAR(20),
            event_type VARCHAR(50), -- 'merger', 'acquisition', 'bankruptcy', 'voluntary'
            data_source VARCHAR(50) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            UNIQUE(symbol, delisting_date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_delisting_events_date
        ON delisting_events(delisting_date);
        
        CREATE INDEX IF NOT EXISTS idx_delisting_events_symbol
        ON delisting_events(symbol);
        """
        
        conn = await self.get_connection()
        try:
            await conn.execute(create_tables_sql)
            logger.info("Universe tables created/verified")
        finally:
            await conn.close()
    
    async def load_universe_constituents(self, 
                                       universe_type: UniverseType,
                                       as_of_date: date = None,
                                       include_delisted: bool = True,
                                       lookback_years: int = 10) -> List[UniverseConstituent]:
        """Load universe constituents for a specific date with survivorship handling"""
        
        if as_of_date is None:
            as_of_date = date.today()
        
        # Define lookback period for delisted securities
        start_date = as_of_date - timedelta(days=lookback_years * 365)
        
        query = """
        SELECT 
            symbol, universe_type, effective_date, expiration_date,
            listing_status, sector, industry, market_cap, weight,
            gics_sector, gics_industry, delisting_reason, successor_symbol,
            data_source
        FROM universe_constituents 
        WHERE universe_type = $1
        AND effective_date <= $2
        AND (expiration_date IS NULL OR expiration_date >= $3)
        """
        
        params = [universe_type.value, as_of_date]
        
        if include_delisted:
            # Include delisted securities within lookback period
            params.append(start_date)
        else:
            # Only active securities
            query += " AND listing_status = 'active'"
            params.append(as_of_date)
        
        query += " ORDER BY effective_date DESC, symbol"
        
        conn = await self.get_connection()
        try:
            rows = await conn.fetch(query, *params)
            constituents = []
            
            for row in rows:
                constituent = UniverseConstituent(
                    symbol=row['symbol'],
                    universe_type=UniverseType(row['universe_type']),
                    effective_date=row['effective_date'],
                    expiration_date=row['expiration_date'],
                    listing_status=ListingStatus(row['listing_status']),
                    sector=row['sector'],
                    industry=row['industry'],
                    market_cap=float(row['market_cap']) if row['market_cap'] else None,
                    weight=float(row['weight']) if row['weight'] else None,
                    gics_sector=row['gics_sector'],
                    gics_industry=row['gics_industry'],
                    delisting_reason=row['delisting_reason'],
                    successor_symbol=row['successor_symbol'],
                    data_source=row['data_source']
                )
                constituents.append(constituent)
            
            logger.info(f"Loaded {len(constituents)} constituents for {universe_type.value} "
                       f"as of {as_of_date} (delisted: {include_delisted})")
            
            return constituents
            
        finally:
            await conn.close()
    
    async def get_active_universe(self, 
                                universe_type: UniverseType,
                                as_of_date: date) -> Set[str]:
        """Get set of active symbols in universe on specific date"""
        constituents = await self.load_universe_constituents(
            universe_type, as_of_date, include_delisted=False
        )
        
        active_symbols = set()
        for constituent in constituents:
            if constituent.is_active_on(as_of_date):
                active_symbols.add(constituent.symbol)
        
        return active_symbols
    
    async def get_survivorship_stats(self, 
                                   universe_type: UniverseType,
                                   start_date: date,
                                   end_date: date) -> SurvivorshipStats:
        """Calculate survivorship bias statistics for a universe over a period"""
        
        query = """
        SELECT 
            COUNT(*) as total_symbols,
            COUNT(CASE WHEN listing_status = 'active' THEN 1 END) as active_symbols,
            COUNT(CASE WHEN listing_status != 'active' THEN 1 END) as delisted_symbols,
            delisting_reason,
            AVG(EXTRACT(DAYS FROM COALESCE(expiration_date, CURRENT_DATE) - effective_date) / 365.25) as avg_duration
        FROM universe_constituents
        WHERE universe_type = $1
        AND effective_date >= $2
        AND effective_date <= $3
        GROUP BY delisting_reason
        """
        
        conn = await self.get_connection()
        try:
            rows = await conn.fetch(query, universe_type.value, start_date, end_date)
            
            total_symbols = sum(row['total_symbols'] for row in rows)
            active_symbols = sum(row['active_symbols'] for row in rows if row['active_symbols'])
            delisted_symbols = sum(row['delisted_symbols'] for row in rows if row['delisted_symbols'])
            
            delisting_reasons = {}
            total_duration = 0
            duration_count = 0
            
            for row in rows:
                if row['delisting_reason']:
                    delisting_reasons[row['delisting_reason']] = row['delisted_symbols'] or 0
                if row['avg_duration']:
                    total_duration += row['avg_duration'] * (row['total_symbols'] or 0)
                    duration_count += (row['total_symbols'] or 0)
            
            avg_duration = total_duration / duration_count if duration_count > 0 else 0
            delisted_rate = delisted_symbols / total_symbols if total_symbols > 0 else 0
            
            return SurvivorshipStats(
                total_symbols=total_symbols,
                active_symbols=active_symbols,
                delisted_symbols=delisted_symbols,
                delisted_rate=delisted_rate,
                avg_listing_duration_years=avg_duration,
                delisting_reasons=delisting_reasons
            )
            
        finally:
            await conn.close()
    
    async def create_universe_snapshot(self, 
                                     universe_type: UniverseType,
                                     snapshot_date: date) -> Dict:
        """Create a point-in-time snapshot of universe constituents"""
        
        constituents = await self.load_universe_constituents(
            universe_type, snapshot_date, include_delisted=True
        )
        
        active_constituents = [c for c in constituents if c.is_active_on(snapshot_date)]
        delisted_constituents = [c for c in constituents if not c.is_active_on(snapshot_date)]
        
        snapshot_data = {
            'active_symbols': [c.symbol for c in active_constituents],
            'delisted_symbols': [c.symbol for c in delisted_constituents],
            'metadata': {
                'snapshot_date': snapshot_date.isoformat(),
                'universe_type': universe_type.value,
                'active_count': len(active_constituents),
                'delisted_count': len(delisted_constituents),
                'total_count': len(constituents)
            }
        }
        
        # Store snapshot in database
        conn = await self.get_connection()
        try:
            await conn.execute("""
                INSERT INTO universe_snapshots 
                (universe_type, snapshot_date, constituents_count, active_count, delisted_count, constituents_json)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (universe_type, snapshot_date) 
                DO UPDATE SET 
                    constituents_count = EXCLUDED.constituents_count,
                    active_count = EXCLUDED.active_count,
                    delisted_count = EXCLUDED.delisted_count,
                    constituents_json = EXCLUDED.constituents_json
            """, 
            universe_type.value, snapshot_date,
            len(constituents), len(active_constituents), len(delisted_constituents),
            snapshot_data)
            
        finally:
            await conn.close()
        
        return snapshot_data
    
    async def load_delisting_events(self, 
                                  start_date: date = None,
                                  end_date: date = None,
                                  symbol: str = None) -> List[Dict]:
        """Load delisting events for analysis"""
        
        conditions = []
        params = []
        param_count = 0
        
        if start_date:
            param_count += 1
            conditions.append(f"delisting_date >= ${param_count}")
            params.append(start_date)
            
        if end_date:
            param_count += 1
            conditions.append(f"delisting_date <= ${param_count}")
            params.append(end_date)
            
        if symbol:
            param_count += 1
            conditions.append(f"symbol = ${param_count}")
            params.append(symbol)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT 
            symbol, delisting_date, delisting_reason, last_trading_date,
            final_price, successor_symbol, event_type, data_source
        FROM delisting_events
        {where_clause}
        ORDER BY delisting_date DESC
        """
        
        conn = await self.get_connection()
        try:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
        finally:
            await conn.close()
    
    async def _load_sp500_constituents(self) -> List[UniverseConstituent]:
        """Load S&P 500 constituents (placeholder for external data source)"""
        # This would typically fetch from external APIs or data files
        logger.info("Loading S&P 500 constituents (placeholder implementation)")
        return []
    
    async def _load_nasdaq100_constituents(self) -> List[UniverseConstituent]:
        """Load NASDAQ 100 constituents (placeholder for external data source)"""
        logger.info("Loading NASDAQ 100 constituents (placeholder implementation)")
        return []
    
    async def _load_russell_constituents(self) -> List[UniverseConstituent]:
        """Load Russell constituents (placeholder for external data source)"""
        logger.info("Loading Russell constituents (placeholder implementation)")
        return []
    
    async def validate_survivorship_bias_prevention(self, 
                                                  universe_type: UniverseType,
                                                  backtest_start: date,
                                                  backtest_end: date) -> Dict:
        """Validate that backtest includes appropriate delisted securities"""
        
        # Get all constituents that were ever active during backtest period
        all_constituents = await self.load_universe_constituents(
            universe_type, backtest_end, include_delisted=True, lookback_years=20
        )
        
        # Filter to those relevant for backtest period
        relevant_constituents = []
        for constituent in all_constituents:
            # Include if active during any part of backtest period
            if (constituent.effective_date <= backtest_end and 
                (constituent.expiration_date is None or constituent.expiration_date >= backtest_start)):
                relevant_constituents.append(constituent)
        
        active_during_backtest = [c for c in relevant_constituents 
                                if c.listing_status == ListingStatus.ACTIVE]
        delisted_during_backtest = [c for c in relevant_constituents 
                                  if c.listing_status != ListingStatus.ACTIVE]
        
        validation_result = {
            'backtest_period': {
                'start_date': backtest_start.isoformat(),
                'end_date': backtest_end.isoformat()
            },
            'universe_composition': {
                'total_constituents': len(relevant_constituents),
                'active_constituents': len(active_during_backtest),
                'delisted_constituents': len(delisted_during_backtest),
                'delisted_percentage': len(delisted_during_backtest) / len(relevant_constituents) * 100
            },
            'survivorship_bias_assessment': {
                'includes_delisted': len(delisted_during_backtest) > 0,
                'delisted_ratio_adequate': len(delisted_during_backtest) / len(relevant_constituents) > 0.05,
                'bias_risk': 'low' if len(delisted_during_backtest) > 0 else 'high'
            },
            'delisted_symbols': [c.symbol for c in delisted_during_backtest],
            'delisting_reasons': {}
        }
        
        # Aggregate delisting reasons
        for constituent in delisted_during_backtest:
            reason = constituent.delisting_reason or 'unknown'
            validation_result['delisting_reasons'][reason] = \
                validation_result['delisting_reasons'].get(reason, 0) + 1
        
        return validation_result


# Convenience functions

async def get_universe_for_backtest(universe_type: UniverseType,
                                  start_date: date,
                                  end_date: date,
                                  include_delisted: bool = True) -> Set[str]:
    """Get universe symbols for backtesting with survivorship bias prevention"""
    loader = PortfolioUniverseLoader()
    
    constituents = await loader.load_universe_constituents(
        universe_type, end_date, include_delisted, lookback_years=20
    )
    
    # Filter to symbols that were active during any part of the backtest period
    relevant_symbols = set()
    for constituent in constituents:
        if (constituent.effective_date <= end_date and 
            (constituent.expiration_date is None or constituent.expiration_date >= start_date)):
            relevant_symbols.add(constituent.symbol)
    
    return relevant_symbols


async def validate_universe_completeness(universe_type: UniverseType,
                                       backtest_period_years: int = 10) -> Dict:
    """Validate universe data completeness and survivorship bias handling"""
    loader = PortfolioUniverseLoader()
    end_date = date.today()
    start_date = end_date - timedelta(days=backtest_period_years * 365)
    
    return await loader.validate_survivorship_bias_prevention(
        universe_type, start_date, end_date
    )