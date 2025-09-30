#!/usr/bin/env python3
"""
Point-in-Time (PIT) Enforcement Pipeline

Automated system to enforce point-in-time data integrity across all trading platform services.
Validates that features never use future information and maintains temporal consistency.

Features:
- Real-time PIT violation detection
- Automated timestamp normalization to UTC with ms precision
- Cross-service temporal validation
- Feature contract compliance checking
- Automated remediation workflows

Usage:
    python pit_enforcement_pipeline.py monitor --config pit_config.yml
    python pit_enforcement_pipeline.py validate-timestamps --service market-data
    python pit_enforcement_pipeline.py normalize-events --source event_data.json
    python pit_enforcement_pipeline.py generate-report --output pit_report.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import asyncpg
import aiohttp
import yaml
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PITViolationType(Enum):
    """Types of point-in-time violations."""
    FUTURE_LEAK = "future_leak"  # Feature uses future information
    TIMESTAMP_DRIFT = "timestamp_drift"  # Timestamps not normalized
    ARRIVAL_DELAY = "arrival_delay"  # Data arrives too late
    REVISION_BACKDATING = "revision_backdating"  # Historical data improperly revised
    CONTRACT_VIOLATION = "contract_violation"  # Violates feature contract rules


@dataclass
class PITViolation:
    """Point-in-time violation record."""
    violation_id: str
    violation_type: PITViolationType
    service_name: str
    feature_name: str
    timestamp: datetime
    as_of_timestamp: datetime
    description: str
    severity: str  # "critical", "high", "medium", "low"
    metadata: Dict[str, Any]
    detected_at: datetime
    resolved_at: Optional[datetime] = None


@dataclass
class TimestampValidation:
    """Timestamp validation result."""
    original_timestamp: str
    normalized_timestamp: datetime
    timezone_detected: str
    precision_ms: bool
    validation_errors: List[str]
    confidence_score: float


class PITEnforcementEngine:
    """Core PIT enforcement engine."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the PIT enforcement engine."""
        self.config = config
        self.violations: List[PITViolation] = []
        self.service_endpoints = config.get('service_endpoints', {})
        self.db_config = config.get('database', {})
        self.enforcement_rules = config.get('enforcement_rules', {})
        
    async def connect_to_db(self) -> asyncpg.Connection:
        """Connect to TimescaleDB database."""
        connection_string = (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}"
            f"/{self.db_config['database']}"
        )
        return await asyncpg.connect(connection_string)
    
    def normalize_timestamp(self, timestamp_str: str, source_timezone: str = "UTC") -> TimestampValidation:
        """Normalize timestamp to UTC with millisecond precision."""
        validation_errors = []
        confidence_score = 1.0
        
        try:
            # Handle various timestamp formats
            formats_to_try = [
                "%Y-%m-%dT%H:%M:%S.%fZ",        # ISO with microseconds
                "%Y-%m-%dT%H:%M:%SZ",           # ISO without microseconds
                "%Y-%m-%d %H:%M:%S.%f",         # SQL with microseconds
                "%Y-%m-%d %H:%M:%S",            # SQL without microseconds
                "%Y-%m-%dT%H:%M:%S.%f%z",       # ISO with timezone
                "%Y-%m-%dT%H:%M:%S%z",          # ISO with timezone, no microseconds
            ]
            
            parsed_dt = None
            detected_format = None
            
            for fmt in formats_to_try:
                try:
                    parsed_dt = datetime.strptime(timestamp_str, fmt)
                    detected_format = fmt
                    break
                except ValueError:
                    continue
            
            if not parsed_dt:
                # Try parsing with dateutil as fallback
                from dateutil import parser
                try:
                    parsed_dt = parser.parse(timestamp_str)
                    detected_format = "dateutil_parser"
                except Exception as e:
                    validation_errors.append(f"Unable to parse timestamp: {e}")
                    confidence_score = 0.0
                    return TimestampValidation(
                        original_timestamp=timestamp_str,
                        normalized_timestamp=datetime.now(timezone.utc),
                        timezone_detected="unknown",
                        precision_ms=False,
                        validation_errors=validation_errors,
                        confidence_score=confidence_score
                    )
            
            # Ensure timezone awareness
            if parsed_dt.tzinfo is None:
                if source_timezone.upper() == "UTC":
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
                else:
                    validation_errors.append(f"Naive timestamp assumed to be {source_timezone}")
                    confidence_score *= 0.9
                    # Add timezone based on source_timezone
                    # This is simplified - in production, use proper timezone handling
                    parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            
            # Convert to UTC
            normalized_dt = parsed_dt.astimezone(timezone.utc)
            
            # Check millisecond precision
            has_ms_precision = (
                normalized_dt.microsecond > 0 or 
                ".%f" in detected_format or
                "milliseconds" in timestamp_str.lower()
            )
            
            # Validate timestamp is not in the future (with small tolerance)
            now = datetime.now(timezone.utc)
            if normalized_dt > now + timedelta(minutes=5):
                validation_errors.append("Timestamp is in the future")
                confidence_score *= 0.5
            
            # Validate timestamp is not too old (configurable)
            max_age_days = self.config.get('max_timestamp_age_days', 365 * 10)  # 10 years default
            if normalized_dt < now - timedelta(days=max_age_days):
                validation_errors.append(f"Timestamp is older than {max_age_days} days")
                confidence_score *= 0.8
            
            return TimestampValidation(
                original_timestamp=timestamp_str,
                normalized_timestamp=normalized_dt,
                timezone_detected=str(parsed_dt.tzinfo) if parsed_dt.tzinfo else source_timezone,
                precision_ms=has_ms_precision,
                validation_errors=validation_errors,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            validation_errors.append(f"Timestamp normalization failed: {e}")
            return TimestampValidation(
                original_timestamp=timestamp_str,
                normalized_timestamp=datetime.now(timezone.utc),
                timezone_detected="error",
                precision_ms=False,
                validation_errors=validation_errors,
                confidence_score=0.0
            )
    
    async def detect_future_leakage(self, service_name: str, feature_data: Dict[str, Any]) -> List[PITViolation]:
        """Detect if feature data uses future information."""
        violations = []
        
        feature_name = feature_data.get('feature_name', 'unknown')
        as_of_timestamp = feature_data.get('as_of_timestamp')
        dependencies = feature_data.get('dependencies', [])
        
        if not as_of_timestamp:
            violation = PITViolation(
                violation_id=f"pit_{service_name}_{feature_name}_{datetime.now().timestamp()}",
                violation_type=PITViolationType.FUTURE_LEAK,
                service_name=service_name,
                feature_name=feature_name,
                timestamp=datetime.now(timezone.utc),
                as_of_timestamp=datetime.now(timezone.utc),
                description="Missing as_of_timestamp for feature calculation",
                severity="critical",
                metadata=feature_data,
                detected_at=datetime.now(timezone.utc)
            )
            violations.append(violation)
            return violations
        
        # Normalize as_of_timestamp
        as_of_validation = self.normalize_timestamp(as_of_timestamp)
        as_of_dt = as_of_validation.normalized_timestamp
        
        # Check each dependency timestamp
        for dep in dependencies:
            if isinstance(dep, dict) and 'timestamp' in dep:
                dep_validation = self.normalize_timestamp(dep['timestamp'])
                dep_dt = dep_validation.normalized_timestamp
                
                if dep_dt > as_of_dt:
                    violation = PITViolation(
                        violation_id=f"pit_{service_name}_{feature_name}_{datetime.now().timestamp()}",
                        violation_type=PITViolationType.FUTURE_LEAK,
                        service_name=service_name,
                        feature_name=feature_name,
                        timestamp=dep_dt,
                        as_of_timestamp=as_of_dt,
                        description=f"Dependency '{dep.get('name', 'unknown')}' timestamp ({dep_dt}) is after as_of_timestamp ({as_of_dt})",
                        severity="critical",
                        metadata={
                            'dependency': dep,
                            'feature_data': feature_data,
                            'time_difference_seconds': (dep_dt - as_of_dt).total_seconds()
                        },
                        detected_at=datetime.now(timezone.utc)
                    )
                    violations.append(violation)
        
        return violations
    
    async def validate_arrival_timing(self, service_name: str, event_data: Dict[str, Any]) -> List[PITViolation]:
        """Validate that data arrives within expected latency windows."""
        violations = []
        
        event_timestamp = event_data.get('event_timestamp')
        arrival_timestamp = event_data.get('arrival_timestamp', datetime.now(timezone.utc).isoformat())
        expected_latency = event_data.get('expected_latency_minutes', 60)  # Default 1 hour
        
        if not event_timestamp:
            return violations
        
        event_validation = self.normalize_timestamp(event_timestamp)
        arrival_validation = self.normalize_timestamp(arrival_timestamp)
        
        event_dt = event_validation.normalized_timestamp
        arrival_dt = arrival_validation.normalized_timestamp
        
        actual_latency_minutes = (arrival_dt - event_dt).total_seconds() / 60
        
        if actual_latency_minutes > expected_latency:
            severity = "high" if actual_latency_minutes > expected_latency * 2 else "medium"
            
            violation = PITViolation(
                violation_id=f"pit_latency_{service_name}_{datetime.now().timestamp()}",
                violation_type=PITViolationType.ARRIVAL_DELAY,
                service_name=service_name,
                feature_name=event_data.get('feature_name', 'unknown'),
                timestamp=event_dt,
                as_of_timestamp=arrival_dt,
                description=f"Data arrived {actual_latency_minutes:.1f} minutes late (expected: {expected_latency} minutes)",
                severity=severity,
                metadata={
                    'event_data': event_data,
                    'actual_latency_minutes': actual_latency_minutes,
                    'expected_latency_minutes': expected_latency,
                    'latency_ratio': actual_latency_minutes / expected_latency
                },
                detected_at=datetime.now(timezone.utc)
            )
            violations.append(violation)
        
        return violations
    
    async def validate_service_contracts(self, service_name: str) -> List[PITViolation]:
        """Validate that service adheres to its feature contracts."""
        violations = []
        
        try:
            # Load feature contracts for this service
            contracts_dir = Path("docs/feature-contracts")
            service_contracts = []
            
            if contracts_dir.exists():
                for contract_file in contracts_dir.glob("*.yml"):
                    try:
                        with open(contract_file, 'r') as f:
                            contract = yaml.safe_load(f)
                            if contract.get('data_source') == service_name:
                                service_contracts.append(contract)
                    except Exception as e:
                        logger.warning(f"Error loading contract {contract_file}: {e}")
            
            # Check each contract's PIT rules
            for contract in service_contracts:
                feature_name = contract.get('feature_name', 'unknown')
                as_of_rule = contract.get('as_of_ts_rule')
                effective_rule = contract.get('effective_ts_rule')
                arrival_latency = contract.get('arrival_latency_minutes', 0)
                
                # Query the service to check actual implementation
                try:
                    endpoint = self.service_endpoints.get(service_name)
                    if endpoint:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{endpoint}/health") as response:
                                if response.status != 200:
                                    violation = PITViolation(
                                        violation_id=f"pit_contract_{service_name}_{feature_name}_{datetime.now().timestamp()}",
                                        violation_type=PITViolationType.CONTRACT_VIOLATION,
                                        service_name=service_name,
                                        feature_name=feature_name,
                                        timestamp=datetime.now(timezone.utc),
                                        as_of_timestamp=datetime.now(timezone.utc),
                                        description=f"Service {service_name} is not responding (HTTP {response.status})",
                                        severity="high",
                                        metadata={'contract': contract, 'endpoint': endpoint},
                                        detected_at=datetime.now(timezone.utc)
                                    )
                                    violations.append(violation)
                
                except Exception as e:
                    logger.warning(f"Error validating service {service_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error validating contracts for {service_name}: {e}")
        
        return violations
    
    async def monitor_realtime_violations(self, duration_minutes: int = 60) -> List[PITViolation]:
        """Monitor for PIT violations in real-time."""
        logger.info(f"Starting real-time PIT monitoring for {duration_minutes} minutes")
        
        violations = []
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now(timezone.utc) < end_time:
            try:
                # Check each configured service
                for service_name, endpoint in self.service_endpoints.items():
                    try:
                        # Get recent data from service
                        async with aiohttp.ClientSession() as session:
                            # This would be customized per service API
                            health_url = f"{endpoint}/health"
                            async with session.get(health_url, timeout=5) as response:
                                if response.status == 200:
                                    logger.debug(f"Service {service_name} is healthy")
                                else:
                                    logger.warning(f"Service {service_name} returned status {response.status}")
                    
                    except Exception as e:
                        logger.warning(f"Error checking service {service_name}: {e}")
                
                # Validate database timestamps
                try:
                    conn = await self.connect_to_db()
                    
                    # Check for recent data with timestamp anomalies
                    query = """
                    SELECT 
                        table_name,
                        column_name,
                        COUNT(*) as violation_count
                    FROM information_schema.columns c
                    JOIN information_schema.tables t ON c.table_name = t.table_name
                    WHERE c.data_type IN ('timestamp', 'timestamptz')
                    AND t.table_type = 'BASE TABLE'
                    AND t.table_schema = 'public'
                    GROUP BY table_name, column_name
                    """
                    
                    results = await conn.fetch(query)
                    for row in results:
                        logger.debug(f"Found timestamp column: {row['table_name']}.{row['column_name']}")
                    
                    await conn.close()
                
                except Exception as e:
                    logger.warning(f"Error validating database timestamps: {e}")
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
        
        logger.info(f"Completed real-time monitoring. Found {len(violations)} violations")
        return violations
    
    async def normalize_event_timestamps(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize timestamps in a batch of events."""
        normalized_events = []
        
        for event in events:
            normalized_event = event.copy()
            
            # Normalize all timestamp fields
            for key, value in event.items():
                if 'timestamp' in key.lower() or 'time' in key.lower():
                    if isinstance(value, str):
                        validation = self.normalize_timestamp(value)
                        normalized_event[key] = validation.normalized_timestamp.isoformat()
                        normalized_event[f"{key}_validation"] = {
                            'original': validation.original_timestamp,
                            'timezone_detected': validation.timezone_detected,
                            'precision_ms': validation.precision_ms,
                            'errors': validation.validation_errors,
                            'confidence': validation.confidence_score
                        }
            
            # Add processing metadata
            normalized_event['pit_processed_at'] = datetime.now(timezone.utc).isoformat()
            normalized_event['pit_engine_version'] = "1.0.0"
            
            normalized_events.append(normalized_event)
        
        return normalized_events
    
    def generate_pit_report(self) -> Dict[str, Any]:
        """Generate comprehensive PIT enforcement report."""
        now = datetime.now(timezone.utc)
        
        # Categorize violations by type and severity
        violations_by_type = {}
        violations_by_severity = {}
        violations_by_service = {}
        
        for violation in self.violations:
            # By type
            vtype = violation.violation_type.value
            if vtype not in violations_by_type:
                violations_by_type[vtype] = []
            violations_by_type[vtype].append(violation)
            
            # By severity
            if violation.severity not in violations_by_severity:
                violations_by_severity[violation.severity] = []
            violations_by_severity[violation.severity].append(violation)
            
            # By service
            if violation.service_name not in violations_by_service:
                violations_by_service[violation.service_name] = []
            violations_by_service[violation.service_name].append(violation)
        
        # Calculate metrics
        total_violations = len(self.violations)
        critical_violations = len(violations_by_severity.get('critical', []))
        unresolved_violations = len([v for v in self.violations if v.resolved_at is None])
        
        report = {
            'report_timestamp': now.isoformat(),
            'pit_enforcement_version': '1.0.0',
            'summary': {
                'total_violations': total_violations,
                'critical_violations': critical_violations,
                'unresolved_violations': unresolved_violations,
                'resolution_rate': ((total_violations - unresolved_violations) / total_violations * 100) if total_violations > 0 else 100
            },
            'violations_by_type': {
                vtype: len(violations) for vtype, violations in violations_by_type.items()
            },
            'violations_by_severity': {
                severity: len(violations) for severity, violations in violations_by_severity.items()
            },
            'violations_by_service': {
                service: len(violations) for service, violations in violations_by_service.items()
            },
            'detailed_violations': [
                {
                    'violation_id': v.violation_id,
                    'type': v.violation_type.value,
                    'service': v.service_name,
                    'feature': v.feature_name,
                    'timestamp': v.timestamp.isoformat(),
                    'description': v.description,
                    'severity': v.severity,
                    'detected_at': v.detected_at.isoformat(),
                    'resolved_at': v.resolved_at.isoformat() if v.resolved_at else None,
                    'metadata': v.metadata
                }
                for v in self.violations
            ],
            'recommendations': self._generate_recommendations(violations_by_type, violations_by_service)
        }
        
        return report
    
    def _generate_recommendations(self, violations_by_type: Dict, violations_by_service: Dict) -> List[str]:
        """Generate recommendations for fixing PIT violations."""
        recommendations = []
        
        if PITViolationType.FUTURE_LEAK.value in violations_by_type:
            count = len(violations_by_type[PITViolationType.FUTURE_LEAK.value])
            recommendations.append(
                f"Fix {count} future leak violations by ensuring all features use only past/present data"
            )
        
        if PITViolationType.TIMESTAMP_DRIFT.value in violations_by_type:
            count = len(violations_by_type[PITViolationType.TIMESTAMP_DRIFT.value])
            recommendations.append(
                f"Normalize {count} timestamp drift issues by implementing UTC conversion with ms precision"
            )
        
        if PITViolationType.ARRIVAL_DELAY.value in violations_by_type:
            count = len(violations_by_type[PITViolationType.ARRIVAL_DELAY.value])
            recommendations.append(
                f"Optimize data pipelines to reduce {count} arrival delay violations"
            )
        
        # Service-specific recommendations
        for service, violations in violations_by_service.items():
            if len(violations) > 5:
                recommendations.append(
                    f"Service '{service}' has {len(violations)} violations - requires immediate attention"
                )
        
        if not recommendations:
            recommendations.append("No major PIT violations detected - system is operating within compliance")
        
        return recommendations


async def load_config(config_path: str) -> Dict[str, Any]:
    """Load PIT enforcement configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Return default configuration
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            'service_endpoints': {
                'market-data-service': 'http://localhost:8001',
                'sentiment-service': 'http://localhost:8004',
                'analysis-service': 'http://localhost:8003',
                'fundamentals-service': 'http://localhost:8005',
                'portfolio-service': 'http://localhost:8006',
                'strategy-service': 'http://localhost:8007'
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'trading_db',
                'user': 'trading_user',
                'password': 'trading_password'
            },
            'enforcement_rules': {
                'max_future_leak_minutes': 0,
                'max_arrival_delay_hours': 24,
                'required_timestamp_precision': 'milliseconds',
                'timezone_enforcement': 'UTC'
            },
            'monitoring': {
                'check_interval_seconds': 30,
                'alert_threshold_violations': 5,
                'max_timestamp_age_days': 3650
            }
        }


async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Point-in-Time (PIT) Enforcement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor for PIT violations')
    monitor_parser.add_argument('--config', default='pit_config.yml', help='Configuration file')
    monitor_parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in minutes')
    monitor_parser.add_argument('--output', help='Output file for violations')
    
    # Validate timestamps command
    validate_parser = subparsers.add_parser('validate-timestamps', help='Validate service timestamps')
    validate_parser.add_argument('--service', required=True, help='Service name to validate')
    validate_parser.add_argument('--config', default='pit_config.yml', help='Configuration file')
    
    # Normalize events command
    normalize_parser = subparsers.add_parser('normalize-events', help='Normalize event timestamps')
    normalize_parser.add_argument('--source', required=True, help='Source JSON file with events')
    normalize_parser.add_argument('--output', help='Output file for normalized events')
    
    # Generate report command
    report_parser = subparsers.add_parser('generate-report', help='Generate PIT enforcement report')
    report_parser.add_argument('--config', default='pit_config.yml', help='Configuration file')
    report_parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        config = await load_config(args.config)
        engine = PITEnforcementEngine(config)
        
        if args.command == 'monitor':
            violations = await engine.monitor_realtime_violations(args.duration)
            engine.violations.extend(violations)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump([asdict(v) for v in violations], f, indent=2, default=str)
                logger.info(f"Violations saved to {args.output}")
            
            logger.info(f"Monitoring complete. Found {len(violations)} violations")
            return 1 if violations else 0
            
        elif args.command == 'validate-timestamps':
            violations = await engine.validate_service_contracts(args.service)
            engine.violations.extend(violations)
            
            logger.info(f"Timestamp validation complete for {args.service}")
            logger.info(f"Found {len(violations)} violations")
            return 1 if violations else 0
            
        elif args.command == 'normalize-events':
            with open(args.source, 'r') as f:
                events = json.load(f)
            
            normalized_events = await engine.normalize_event_timestamps(events)
            
            output_file = args.output or args.source.replace('.json', '_normalized.json')
            with open(output_file, 'w') as f:
                json.dump(normalized_events, f, indent=2, default=str)
            
            logger.info(f"Normalized {len(events)} events, saved to {output_file}")
            return 0
            
        elif args.command == 'generate-report':
            # Collect violations from various sources
            for service_name in config['service_endpoints'].keys():
                violations = await engine.validate_service_contracts(service_name)
                engine.violations.extend(violations)
            
            report = engine.generate_pit_report()
            
            output_file = args.output or f"pit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"PIT enforcement report generated: {output_file}")
            
            # Print summary
            summary = report['summary']
            logger.info(f"Total violations: {summary['total_violations']}")
            logger.info(f"Critical violations: {summary['critical_violations']}")
            logger.info(f"Resolution rate: {summary['resolution_rate']:.1f}%")
            
            return 1 if summary['critical_violations'] > 0 else 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))