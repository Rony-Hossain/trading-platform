#!/usr/bin/env python3
"""
Offline/Online Skew Guardrail

Nightly monitoring job to detect feature drift between offline and online environments.
Tracks feature value differences and publishes Prometheus metrics for alerting.
"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import asyncpg
import httpx
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "analysis-service" / "app"))

from core.feature_snapshots import FeatureSnapshotManager
from core.config import get_settings

# Prometheus metrics
REGISTRY = CollectorRegistry()
FEATURE_SKEW_RATIO = Gauge(
    'feature_skew_ratio',
    'Ratio of offline vs online feature values',
    ['feature', 'symbol', 'environment'],
    registry=REGISTRY
)

FEATURE_SKEW_ABSOLUTE = Gauge(
    'feature_skew_absolute',
    'Absolute difference between offline and online feature values',
    ['feature', 'symbol', 'environment'],
    registry=REGISTRY
)

SKEW_VIOLATIONS = Gauge(
    'skew_violations_total',
    'Total number of skew tolerance violations',
    ['severity'],
    registry=REGISTRY
)

logger = logging.getLogger(__name__)


class SkewMonitor:
    def __init__(self):
        self.settings = get_settings()
        self.snapshot_manager = FeatureSnapshotManager()
        self.skew_tolerances = self._load_skew_tolerances()
        
    def _load_skew_tolerances(self) -> Dict:
        """Load skew tolerance configuration from environment."""
        tolerances_json = os.getenv('SKEW_TOLERANCES_JSON', '{}')
        try:
            tolerances = json.loads(tolerances_json)
            logger.info(f"Loaded skew tolerances for {len(tolerances)} features")
            return tolerances
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SKEW_TOLERANCES_JSON: {e}")
            return {}
    
    async def get_offline_features(self, symbol: str, date: datetime) -> Dict[str, float]:
        """Get offline feature values from batch processing results."""
        query = """
        SELECT feature_name, feature_value, computed_at
        FROM feature_store 
        WHERE symbol = $1 
        AND DATE(computed_at) = $2
        AND environment = 'offline'
        ORDER BY computed_at DESC
        """
        
        conn = await asyncpg.connect(self.settings.database_url)
        try:
            rows = await conn.fetch(query, symbol, date.date())
            features = {}
            
            for row in rows:
                feature_name = row['feature_name']
                if feature_name not in features:  # Take most recent
                    features[feature_name] = row['feature_value']
                    
            return features
        finally:
            await conn.close()
    
    async def get_online_features(self, symbol: str, date: datetime) -> Dict[str, float]:
        """Get online feature values from real-time processing."""
        query = """
        SELECT feature_name, feature_value, computed_at
        FROM feature_store 
        WHERE symbol = $1 
        AND DATE(computed_at) = $2
        AND environment = 'online'
        ORDER BY computed_at DESC
        """
        
        conn = await asyncpg.connect(self.settings.database_url)
        try:
            rows = await conn.fetch(query, symbol, date.date())
            features = {}
            
            for row in rows:
                feature_name = row['feature_name']
                if feature_name not in features:  # Take most recent
                    features[feature_name] = row['feature_value']
                    
            return features
        finally:
            await conn.close()
    
    def calculate_skew_metrics(self, offline_value: float, online_value: float) -> Tuple[float, float]:
        """Calculate skew ratio and absolute difference."""
        if online_value == 0:
            ratio = float('inf') if offline_value != 0 else 1.0
        else:
            ratio = offline_value / online_value
            
        absolute_diff = abs(offline_value - online_value)
        return ratio, absolute_diff
    
    def check_tolerance_violation(self, feature_name: str, ratio: float, absolute_diff: float) -> Optional[str]:
        """Check if skew exceeds configured tolerances."""
        if feature_name not in self.skew_tolerances:
            return None
            
        tolerance = self.skew_tolerances[feature_name]
        
        # Check ratio tolerance
        if 'ratio_tolerance' in tolerance:
            min_ratio = tolerance['ratio_tolerance'].get('min', 0.8)
            max_ratio = tolerance['ratio_tolerance'].get('max', 1.2)
            
            if ratio < min_ratio or ratio > max_ratio:
                return 'critical'
                
        # Check absolute difference tolerance
        if 'absolute_tolerance' in tolerance:
            max_diff = tolerance['absolute_tolerance'].get('max', 1.0)
            
            if absolute_diff > max_diff:
                return 'high'
                
        return None
    
    async def monitor_symbol_skew(self, symbol: str, date: datetime) -> Dict:
        """Monitor skew for a single symbol."""
        logger.info(f"Monitoring skew for {symbol} on {date.date()}")
        
        offline_features = await self.get_offline_features(symbol, date)
        online_features = await self.get_online_features(symbol, date)
        
        results = {
            'symbol': symbol,
            'date': date.isoformat(),
            'features_compared': 0,
            'violations': []
        }
        
        # Compare common features
        common_features = set(offline_features.keys()) & set(online_features.keys())
        
        for feature_name in common_features:
            offline_val = offline_features[feature_name]
            online_val = online_features[feature_name]
            
            ratio, absolute_diff = self.calculate_skew_metrics(offline_val, online_val)
            
            # Update Prometheus metrics
            FEATURE_SKEW_RATIO.labels(
                feature=feature_name,
                symbol=symbol,
                environment='comparison'
            ).set(ratio)
            
            FEATURE_SKEW_ABSOLUTE.labels(
                feature=feature_name,
                symbol=symbol,
                environment='comparison'
            ).set(absolute_diff)
            
            # Check for violations
            violation_severity = self.check_tolerance_violation(feature_name, ratio, absolute_diff)
            
            if violation_severity:
                violation = {
                    'feature': feature_name,
                    'offline_value': offline_val,
                    'online_value': online_val,
                    'ratio': ratio,
                    'absolute_diff': absolute_diff,
                    'severity': violation_severity
                }
                results['violations'].append(violation)
                
                logger.warning(f"Skew violation for {symbol}.{feature_name}: "
                             f"offline={offline_val}, online={online_val}, "
                             f"ratio={ratio:.3f}, diff={absolute_diff:.3f}, "
                             f"severity={violation_severity}")
            
            results['features_compared'] += 1
        
        return results
    
    async def get_active_symbols(self) -> List[str]:
        """Get list of actively traded symbols to monitor."""
        query = """
        SELECT DISTINCT symbol 
        FROM market_data 
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        AND volume > 0
        ORDER BY symbol
        """
        
        conn = await asyncpg.connect(self.settings.database_url)
        try:
            rows = await conn.fetch(query)
            return [row['symbol'] for row in rows]
        finally:
            await conn.close()
    
    async def store_skew_results(self, results: List[Dict]):
        """Store skew monitoring results in database."""
        query = """
        INSERT INTO skew_monitoring_results 
        (symbol, monitoring_date, features_compared, violations_count, results_json, created_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        """
        
        conn = await asyncpg.connect(self.settings.database_url)
        try:
            for result in results:
                await conn.execute(
                    query,
                    result['symbol'],
                    datetime.fromisoformat(result['date']).date(),
                    result['features_compared'],
                    len(result['violations']),
                    json.dumps(result)
                )
        finally:
            await conn.close()
    
    async def run_monitoring(self, target_date: Optional[datetime] = None):
        """Run complete skew monitoring process."""
        if target_date is None:
            target_date = datetime.now() - timedelta(days=1)  # Yesterday
            
        logger.info(f"Starting skew monitoring for {target_date.date()}")
        
        symbols = await self.get_active_symbols()
        logger.info(f"Monitoring {len(symbols)} symbols")
        
        all_results = []
        total_violations = {'critical': 0, 'high': 0, 'medium': 0}
        
        for symbol in symbols:
            try:
                result = await self.monitor_symbol_skew(symbol, target_date)
                all_results.append(result)
                
                # Count violations by severity
                for violation in result['violations']:
                    severity = violation['severity']
                    total_violations[severity] = total_violations.get(severity, 0) + 1
                    
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
                continue
        
        # Update violation metrics
        for severity, count in total_violations.items():
            SKEW_VIOLATIONS.labels(severity=severity).set(count)
        
        # Store results
        await self.store_skew_results(all_results)
        
        # Push metrics to Prometheus Gateway
        prometheus_gateway = os.getenv('PROMETHEUS_PUSHGATEWAY_URL')
        if prometheus_gateway:
            try:
                push_to_gateway(prometheus_gateway, job='skew_monitoring', registry=REGISTRY)
                logger.info("Pushed metrics to Prometheus Gateway")
            except Exception as e:
                logger.error(f"Failed to push metrics: {e}")
        
        # Summary
        total_features = sum(r['features_compared'] for r in all_results)
        total_violation_count = sum(total_violations.values())
        
        logger.info(f"Skew monitoring complete: "
                   f"{len(symbols)} symbols, {total_features} features compared, "
                   f"{total_violation_count} violations found")
        
        return all_results


async def main():
    """Main entry point for skew monitoring job."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    monitor = SkewMonitor()
    results = await monitor.run_monitoring()
    
    # Exit with error code if critical violations found
    critical_violations = sum(
        len([v for v in r['violations'] if v['severity'] == 'critical'])
        for r in results
    )
    
    if critical_violations > 0:
        logger.error(f"Found {critical_violations} critical skew violations")
        sys.exit(1)
    
    logger.info("Skew monitoring completed successfully")


if __name__ == "__main__":
    asyncio.run(main())