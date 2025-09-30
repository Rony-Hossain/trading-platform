"""
Feature Snapshot Manager

Manages snapshots of feature values for offline/online skew monitoring.
Tracks feature values across different environments and time periods.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncpg
import json

logger = logging.getLogger(__name__)


class Environment(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    BACKTEST = "backtest"


class FeatureType(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    DERIVED = "derived"


@dataclass
class FeatureSnapshot:
    """Single feature value snapshot."""
    feature_name: str
    symbol: str
    value: float
    timestamp: datetime
    environment: Environment
    feature_type: FeatureType
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'symbol': self.symbol,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'environment': self.environment.value,
            'feature_type': self.feature_type.value,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSnapshot':
        return cls(
            feature_name=data['feature_name'],
            symbol=data['symbol'],
            value=data['value'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            environment=Environment(data['environment']),
            feature_type=FeatureType(data['feature_type']),
            metadata=data.get('metadata', {})
        )


@dataclass
class SkewMetrics:
    """Metrics for comparing offline vs online features."""
    feature_name: str
    symbol: str
    offline_value: float
    online_value: float
    ratio: float
    absolute_diff: float
    relative_diff_pct: float
    timestamp: datetime
    
    def is_significant_skew(self, ratio_threshold: float = 0.1, 
                           absolute_threshold: float = 1.0) -> bool:
        """Check if skew exceeds significance thresholds."""
        ratio_skew = abs(self.ratio - 1.0) > ratio_threshold
        absolute_skew = self.absolute_diff > absolute_threshold
        return ratio_skew or absolute_skew


class FeatureSnapshotManager:
    """Manages feature snapshots for skew monitoring."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url
        
    async def get_connection(self) -> asyncpg.Connection:
        """Get database connection."""
        if not self.database_url:
            from core.config import get_settings
            settings = get_settings()
            self.database_url = settings.database_url
            
        return await asyncpg.connect(self.database_url)
    
    async def store_snapshot(self, snapshot: FeatureSnapshot) -> bool:
        """Store a single feature snapshot."""
        query = """
        INSERT INTO feature_snapshots 
        (feature_name, symbol, value, timestamp, environment, feature_type, metadata, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        ON CONFLICT (feature_name, symbol, timestamp, environment) 
        DO UPDATE SET 
            value = EXCLUDED.value,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """
        
        conn = await self.get_connection()
        try:
            await conn.execute(
                query,
                snapshot.feature_name,
                snapshot.symbol,
                snapshot.value,
                snapshot.timestamp,
                snapshot.environment.value,
                snapshot.feature_type.value,
                json.dumps(snapshot.metadata or {})
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store snapshot {snapshot.feature_name}: {e}")
            return False
        finally:
            await conn.close()
    
    async def store_snapshots_batch(self, snapshots: List[FeatureSnapshot]) -> int:
        """Store multiple snapshots in batch."""
        if not snapshots:
            return 0
            
        query = """
        INSERT INTO feature_snapshots 
        (feature_name, symbol, value, timestamp, environment, feature_type, metadata, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        ON CONFLICT (feature_name, symbol, timestamp, environment) 
        DO UPDATE SET 
            value = EXCLUDED.value,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """
        
        conn = await self.get_connection()
        try:
            async with conn.transaction():
                successful = 0
                for snapshot in snapshots:
                    try:
                        await conn.execute(
                            query,
                            snapshot.feature_name,
                            snapshot.symbol,
                            snapshot.value,
                            snapshot.timestamp,
                            snapshot.environment.value,
                            snapshot.feature_type.value,
                            json.dumps(snapshot.metadata or {})
                        )
                        successful += 1
                    except Exception as e:
                        logger.error(f"Failed to store snapshot {snapshot.feature_name}: {e}")
                        
                return successful
        finally:
            await conn.close()
    
    async def get_snapshots(self, 
                           feature_name: str = None,
                           symbol: str = None,
                           environment: Environment = None,
                           start_time: datetime = None,
                           end_time: datetime = None,
                           limit: int = 1000) -> List[FeatureSnapshot]:
        """Retrieve feature snapshots with filtering."""
        conditions = []
        params = []
        param_count = 0
        
        if feature_name:
            param_count += 1
            conditions.append(f"feature_name = ${param_count}")
            params.append(feature_name)
            
        if symbol:
            param_count += 1
            conditions.append(f"symbol = ${param_count}")
            params.append(symbol)
            
        if environment:
            param_count += 1
            conditions.append(f"environment = ${param_count}")
            params.append(environment.value)
            
        if start_time:
            param_count += 1
            conditions.append(f"timestamp >= ${param_count}")
            params.append(start_time)
            
        if end_time:
            param_count += 1
            conditions.append(f"timestamp <= ${param_count}")
            params.append(end_time)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT feature_name, symbol, value, timestamp, environment, feature_type, metadata
        FROM feature_snapshots
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        conn = await self.get_connection()
        try:
            rows = await conn.fetch(query, *params)
            snapshots = []
            
            for row in rows:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                snapshot = FeatureSnapshot(
                    feature_name=row['feature_name'],
                    symbol=row['symbol'],
                    value=row['value'],
                    timestamp=row['timestamp'],
                    environment=Environment(row['environment']),
                    feature_type=FeatureType(row['feature_type']),
                    metadata=metadata
                )
                snapshots.append(snapshot)
                
            return snapshots
        finally:
            await conn.close()
    
    async def get_latest_snapshots(self, 
                                  symbol: str,
                                  environment: Environment,
                                  feature_names: List[str] = None) -> Dict[str, FeatureSnapshot]:
        """Get the latest snapshot for each feature."""
        conditions = ["symbol = $1", "environment = $2"]
        params = [symbol, environment.value]
        
        if feature_names:
            placeholders = ",".join([f"${i+3}" for i in range(len(feature_names))])
            conditions.append(f"feature_name IN ({placeholders})")
            params.extend(feature_names)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT DISTINCT ON (feature_name) 
            feature_name, symbol, value, timestamp, environment, feature_type, metadata
        FROM feature_snapshots
        WHERE {where_clause}
        ORDER BY feature_name, timestamp DESC
        """
        
        conn = await self.get_connection()
        try:
            rows = await conn.fetch(query, *params)
            snapshots = {}
            
            for row in rows:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                snapshot = FeatureSnapshot(
                    feature_name=row['feature_name'],
                    symbol=row['symbol'],
                    value=row['value'],
                    timestamp=row['timestamp'],
                    environment=Environment(row['environment']),
                    feature_type=FeatureType(row['feature_type']),
                    metadata=metadata
                )
                snapshots[row['feature_name']] = snapshot
                
            return snapshots
        finally:
            await conn.close()
    
    async def calculate_skew_metrics(self, 
                                   symbol: str,
                                   feature_names: List[str] = None,
                                   comparison_time: datetime = None) -> List[SkewMetrics]:
        """Calculate skew metrics between offline and online features."""
        if comparison_time is None:
            comparison_time = datetime.now()
        
        offline_snapshots = await self.get_latest_snapshots(
            symbol, Environment.OFFLINE, feature_names
        )
        online_snapshots = await self.get_latest_snapshots(
            symbol, Environment.ONLINE, feature_names
        )
        
        skew_metrics = []
        common_features = set(offline_snapshots.keys()) & set(online_snapshots.keys())
        
        for feature_name in common_features:
            offline_snap = offline_snapshots[feature_name]
            online_snap = online_snapshots[feature_name]
            
            # Calculate metrics
            ratio = offline_snap.value / online_snap.value if online_snap.value != 0 else float('inf')
            absolute_diff = abs(offline_snap.value - online_snap.value)
            
            if online_snap.value != 0:
                relative_diff_pct = (absolute_diff / abs(online_snap.value)) * 100
            else:
                relative_diff_pct = float('inf')
            
            metrics = SkewMetrics(
                feature_name=feature_name,
                symbol=symbol,
                offline_value=offline_snap.value,
                online_value=online_snap.value,
                ratio=ratio,
                absolute_diff=absolute_diff,
                relative_diff_pct=relative_diff_pct,
                timestamp=comparison_time
            )
            
            skew_metrics.append(metrics)
        
        return skew_metrics
    
    async def get_symbols_with_snapshots(self, 
                                       environment: Environment = None,
                                       days_back: int = 7) -> List[str]:
        """Get list of symbols that have snapshots."""
        conditions = []
        params = []
        param_count = 0
        
        if environment:
            param_count += 1
            conditions.append(f"environment = ${param_count}")
            params.append(environment.value)
        
        if days_back:
            param_count += 1
            conditions.append(f"timestamp >= NOW() - INTERVAL '{days_back} days'")
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT DISTINCT symbol
        FROM feature_snapshots
        {where_clause}
        ORDER BY symbol
        """
        
        conn = await self.get_connection()
        try:
            rows = await conn.fetch(query, *params)
            return [row['symbol'] for row in rows]
        finally:
            await conn.close()
    
    async def cleanup_old_snapshots(self, days_to_keep: int = 30) -> int:
        """Clean up old snapshots beyond retention period."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        query = """
        DELETE FROM feature_snapshots 
        WHERE timestamp < $1
        """
        
        conn = await self.get_connection()
        try:
            result = await conn.execute(query, cutoff_date)
            deleted_count = int(result.split()[-1])
            logger.info(f"Cleaned up {deleted_count} old feature snapshots")
            return deleted_count
        finally:
            await conn.close()
    
    async def create_tables(self):
        """Create required database tables if they don't exist."""
        create_snapshots_table = """
        CREATE TABLE IF NOT EXISTS feature_snapshots (
            id SERIAL PRIMARY KEY,
            feature_name VARCHAR(255) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            environment VARCHAR(20) NOT NULL,
            feature_type VARCHAR(20) NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            UNIQUE(feature_name, symbol, timestamp, environment)
        );
        
        CREATE INDEX IF NOT EXISTS idx_feature_snapshots_lookup 
        ON feature_snapshots(feature_name, symbol, environment, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_feature_snapshots_timestamp 
        ON feature_snapshots(timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_feature_snapshots_symbol_env 
        ON feature_snapshots(symbol, environment);
        """
        
        create_skew_results_table = """
        CREATE TABLE IF NOT EXISTS skew_monitoring_results (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            monitoring_date DATE NOT NULL,
            features_compared INTEGER NOT NULL DEFAULT 0,
            violations_count INTEGER NOT NULL DEFAULT 0,
            results_json JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            UNIQUE(symbol, monitoring_date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_skew_results_date 
        ON skew_monitoring_results(monitoring_date DESC);
        """
        
        conn = await self.get_connection()
        try:
            await conn.execute(create_snapshots_table)
            await conn.execute(create_skew_results_table)
            logger.info("Feature snapshot tables created/verified")
        finally:
            await conn.close()


# Convenience functions for common operations

async def record_feature_snapshot(feature_name: str, 
                                symbol: str,
                                value: float,
                                environment: Environment,
                                feature_type: FeatureType,
                                timestamp: datetime = None,
                                metadata: Dict[str, Any] = None) -> bool:
    """Record a single feature snapshot."""
    if timestamp is None:
        timestamp = datetime.now()
    
    snapshot = FeatureSnapshot(
        feature_name=feature_name,
        symbol=symbol,
        value=value,
        timestamp=timestamp,
        environment=environment,
        feature_type=feature_type,
        metadata=metadata
    )
    
    manager = FeatureSnapshotManager()
    return await manager.store_snapshot(snapshot)


async def get_skew_for_symbol(symbol: str, feature_names: List[str] = None) -> List[SkewMetrics]:
    """Get current skew metrics for a symbol."""
    manager = FeatureSnapshotManager()
    return await manager.calculate_skew_metrics(symbol, feature_names)