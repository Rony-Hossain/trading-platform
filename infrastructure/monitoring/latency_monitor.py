"""
Data Latency Monitoring System
Comprehensive end-to-end latency tracking for trading platform
"""

import asyncio
import logging
import json
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Types of data sources being monitored"""
    MARKET_DATA = "market_data"
    SENTIMENT = "sentiment"  
    FUNDAMENTALS = "fundamentals"
    MACRO_FACTORS = "macro_factors"
    OPTIONS = "options"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    EARNINGS = "earnings"
    EVENT_DATA = "event_data"

class LatencyStage(Enum):
    """Stages in the data pipeline"""
    SOURCE_INGESTION = "source_ingestion"
    DATA_VALIDATION = "data_validation"
    DATA_PROCESSING = "data_processing"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_STORAGE = "feature_storage"
    FEATURE_AVAILABILITY = "feature_availability"
    MODEL_INFERENCE = "model_inference"
    SIGNAL_GENERATION = "signal_generation"
    STRATEGY_EXECUTION = "strategy_execution"

@dataclass
class LatencyMeasurement:
    """Individual latency measurement"""
    measurement_id: str
    data_source: DataSourceType
    stage: LatencyStage
    timestamp: datetime
    latency_ms: float
    metadata: Dict[str, Any]
    success: bool = True
    error_message: str = ""

@dataclass
class EndToEndLatency:
    """End-to-end latency measurement"""
    trace_id: str
    data_source: DataSourceType
    symbol: str
    start_timestamp: datetime
    end_timestamp: datetime
    total_latency_ms: float
    stage_latencies: Dict[LatencyStage, float]
    critical_path: List[LatencyStage]
    sla_compliance: bool
    sla_target_ms: float

@dataclass
class LatencyAlert:
    """Latency degradation alert"""
    alert_id: str
    alert_type: str
    data_source: DataSourceType
    stage: LatencyStage
    current_latency_ms: float
    threshold_ms: float
    degradation_percentage: float
    impact_assessment: str
    timestamp: datetime
    priority: str

class LatencyMonitor:
    """
    Comprehensive data latency monitoring system
    
    Features:
    - End-to-end latency tracking from source to feature availability
    - Stage-by-stage latency decomposition
    - SLA compliance monitoring
    - Critical path analysis for latency-sensitive strategies
    - Real-time alerting for latency degradation
    - Prometheus metrics export
    - Grafana dashboard data
    """
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 postgres_url: str = "postgresql://trading_user:trading_pass@localhost:5432/trading_db",
                 prometheus_gateway: str = "localhost:9091"):
        
        self.redis_url = redis_url
        self.postgres_url = postgres_url
        self.prometheus_gateway = prometheus_gateway
        self.redis_client = None
        
        # Prometheus metrics
        self.metrics_registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # SLA targets (milliseconds)
        self.sla_targets = {
            DataSourceType.MARKET_DATA: {
                LatencyStage.SOURCE_INGESTION: 100,
                LatencyStage.FEATURE_AVAILABILITY: 500,
                "end_to_end": 1000
            },
            DataSourceType.SENTIMENT: {
                LatencyStage.SOURCE_INGESTION: 1000,
                LatencyStage.FEATURE_AVAILABILITY: 5000,
                "end_to_end": 10000
            },
            DataSourceType.FUNDAMENTALS: {
                LatencyStage.SOURCE_INGESTION: 5000,
                LatencyStage.FEATURE_AVAILABILITY: 30000,
                "end_to_end": 60000
            },
            DataSourceType.MACRO_FACTORS: {
                LatencyStage.SOURCE_INGESTION: 2000,
                LatencyStage.FEATURE_AVAILABILITY: 10000,
                "end_to_end": 15000
            },
            DataSourceType.OPTIONS: {
                LatencyStage.SOURCE_INGESTION: 1000,
                LatencyStage.FEATURE_AVAILABILITY: 3000,
                "end_to_end": 5000
            }
        }
        
        # Alert thresholds (percentage over SLA)
        self.alert_thresholds = {
            "warning": 1.5,   # 50% over SLA
            "critical": 2.0,  # 100% over SLA
            "severe": 3.0     # 200% over SLA
        }
        
        # Active traces for end-to-end measurement
        self.active_traces: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize latency monitor"""
        self.redis_client = await aioredis.from_url(self.redis_url)
        logger.info("Latency monitor initialized")
        
        # Start background tasks
        asyncio.create_task(self._process_latency_measurements())
        asyncio.create_task(self._check_sla_compliance())
        asyncio.create_task(self._export_prometheus_metrics())
    
    async def close(self):
        """Close connections"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def start_trace(self, 
                         data_source: DataSourceType, 
                         symbol: str, 
                         metadata: Dict[str, Any] = None) -> str:
        """Start end-to-end latency trace"""
        trace_id = f"{data_source.value}_{symbol}_{int(time.time() * 1000000)}"
        
        trace_data = {
            "trace_id": trace_id,
            "data_source": data_source.value,
            "symbol": symbol,
            "start_timestamp": datetime.utcnow().isoformat(),
            "stages": {},
            "metadata": metadata or {}
        }
        
        self.active_traces[trace_id] = trace_data
        
        # Store in Redis with expiration
        await self.redis_client.setex(
            f"latency_trace:{trace_id}",
            3600,  # 1 hour expiration
            json.dumps(trace_data)
        )
        
        logger.debug(f"Started latency trace {trace_id} for {data_source.value} {symbol}")
        return trace_id
    
    async def record_stage_latency(self,
                                  trace_id: str,
                                  stage: LatencyStage,
                                  latency_ms: float,
                                  success: bool = True,
                                  metadata: Dict[str, Any] = None):
        """Record latency for a specific stage"""
        measurement = LatencyMeasurement(
            measurement_id=f"{trace_id}_{stage.value}_{int(time.time() * 1000000)}",
            data_source=DataSourceType.MARKET_DATA,  # Will be updated from trace
            stage=stage,
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            metadata=metadata or {},
            success=success
        )
        
        # Update active trace
        if trace_id in self.active_traces:
            self.active_traces[trace_id]["stages"][stage.value] = {
                "latency_ms": latency_ms,
                "timestamp": measurement.timestamp.isoformat(),
                "success": success
            }
            
            # Update Redis
            await self.redis_client.setex(
                f"latency_trace:{trace_id}",
                3600,
                json.dumps(self.active_traces[trace_id])
            )
        
        # Store individual measurement
        await self._store_measurement(measurement)
        
        # Update Prometheus metrics
        data_source = self.active_traces.get(trace_id, {}).get("data_source", "unknown")
        self.stage_latency_histogram.labels(
            data_source=data_source,
            stage=stage.value
        ).observe(latency_ms / 1000.0)  # Convert to seconds
        
        logger.debug(f"Recorded {stage.value} latency: {latency_ms}ms for trace {trace_id}")
    
    async def complete_trace(self, trace_id: str) -> Optional[EndToEndLatency]:
        """Complete end-to-end latency trace"""
        if trace_id not in self.active_traces:
            logger.warning(f"Trace {trace_id} not found in active traces")
            return None
        
        trace_data = self.active_traces[trace_id]
        
        # Calculate end-to-end latency
        start_time = datetime.fromisoformat(trace_data["start_timestamp"])
        end_time = datetime.utcnow()
        total_latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Extract stage latencies
        stage_latencies = {}
        for stage_name, stage_data in trace_data["stages"].items():
            stage_latencies[LatencyStage(stage_name)] = stage_data["latency_ms"]
        
        # Determine critical path
        critical_path = self._calculate_critical_path(stage_latencies)
        
        # Check SLA compliance
        data_source = DataSourceType(trace_data["data_source"])
        sla_target = self.sla_targets.get(data_source, {}).get("end_to_end", 10000)
        sla_compliance = total_latency_ms <= sla_target
        
        end_to_end = EndToEndLatency(
            trace_id=trace_id,
            data_source=data_source,
            symbol=trace_data["symbol"],
            start_timestamp=start_time,
            end_timestamp=end_time,
            total_latency_ms=total_latency_ms,
            stage_latencies=stage_latencies,
            critical_path=critical_path,
            sla_compliance=sla_compliance,
            sla_target_ms=sla_target
        )
        
        # Store end-to-end measurement
        await self._store_end_to_end_measurement(end_to_end)
        
        # Update Prometheus metrics
        self.end_to_end_latency_histogram.labels(
            data_source=data_source.value
        ).observe(total_latency_ms / 1000.0)
        
        self.sla_compliance_gauge.labels(
            data_source=data_source.value
        ).set(1 if sla_compliance else 0)
        
        # Check for alerts
        await self._check_latency_alerts(end_to_end)
        
        # Clean up active trace
        del self.active_traces[trace_id]
        
        logger.info(f"Completed trace {trace_id}: {total_latency_ms:.2f}ms (SLA: {sla_compliance})")
        return end_to_end
    
    async def get_latency_statistics(self,
                                   data_source: DataSourceType = None,
                                   stage: LatencyStage = None,
                                   time_window_hours: int = 24) -> Dict[str, Any]:
        """Get latency statistics for analysis"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build query conditions
            conditions = ["timestamp >= %s"]
            params = [datetime.utcnow() - timedelta(hours=time_window_hours)]
            
            if data_source:
                conditions.append("data_source = %s")
                params.append(data_source.value)
                
            if stage:
                conditions.append("stage = %s") 
                params.append(stage.value)
            
            where_clause = " AND ".join(conditions)
            
            # Query latency measurements
            query = f"""
                SELECT 
                    data_source,
                    stage,
                    AVG(latency_ms) as avg_latency,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as median_latency,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency,
                    MAX(latency_ms) as max_latency,
                    MIN(latency_ms) as min_latency,
                    COUNT(*) as measurement_count,
                    SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as failure_count
                FROM latency_measurements 
                WHERE {where_clause}
                GROUP BY data_source, stage
                ORDER BY data_source, stage
            """
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Query end-to-end statistics
            e2e_query = f"""
                SELECT 
                    data_source,
                    AVG(total_latency_ms) as avg_e2e_latency,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms) as p95_e2e_latency,
                    COUNT(*) as e2e_count,
                    SUM(CASE WHEN sla_compliance = true THEN 1 ELSE 0 END) as sla_compliant_count
                FROM end_to_end_latency
                WHERE timestamp >= %s
                {f"AND data_source = '{data_source.value}'" if data_source else ""}
                GROUP BY data_source
            """
            
            cursor.execute(e2e_query, [datetime.utcnow() - timedelta(hours=time_window_hours)])
            e2e_results = cursor.fetchall()
            
            conn.close()
            
            return {
                "stage_statistics": [dict(row) for row in results],
                "end_to_end_statistics": [dict(row) for row in e2e_results],
                "time_window_hours": time_window_hours,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting latency statistics: {e}")
            return {"error": str(e)}
    
    async def get_critical_path_analysis(self, symbol: str, time_window_hours: int = 1) -> Dict[str, Any]:
        """Analyze critical path for latency optimization"""
        try:
            # Get recent end-to-end measurements for symbol
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT * FROM end_to_end_latency 
                WHERE symbol = %s AND timestamp >= %s
                ORDER BY timestamp DESC
                LIMIT 100
            """
            
            cursor.execute(query, [
                symbol, 
                datetime.utcnow() - timedelta(hours=time_window_hours)
            ])
            measurements = cursor.fetchall()
            conn.close()
            
            if not measurements:
                return {"error": f"No measurements found for {symbol}"}
            
            # Analyze stage contributions
            stage_analysis = {}
            for measurement in measurements:
                stage_latencies = json.loads(measurement["stage_latencies"]) if isinstance(measurement["stage_latencies"], str) else measurement["stage_latencies"]
                
                for stage, latency in stage_latencies.items():
                    if stage not in stage_analysis:
                        stage_analysis[stage] = []
                    stage_analysis[stage].append(latency)
            
            # Calculate statistics for each stage
            stage_stats = {}
            total_avg_latency = 0
            
            for stage, latencies in stage_analysis.items():
                avg_latency = statistics.mean(latencies)
                total_avg_latency += avg_latency
                
                stage_stats[stage] = {
                    "avg_latency_ms": avg_latency,
                    "median_latency_ms": statistics.median(latencies),
                    "max_latency_ms": max(latencies),
                    "contribution_percentage": 0  # Will calculate after
                }
            
            # Calculate contribution percentages
            for stage in stage_stats:
                stage_stats[stage]["contribution_percentage"] = (
                    stage_stats[stage]["avg_latency_ms"] / total_avg_latency * 100
                )
            
            # Identify bottleneck stages
            sorted_stages = sorted(
                stage_stats.items(), 
                key=lambda x: x[1]["contribution_percentage"], 
                reverse=True
            )
            
            recommendations = []
            for i, (stage, stats) in enumerate(sorted_stages[:3]):
                if stats["contribution_percentage"] > 20:
                    recommendations.append(f"Optimize {stage}: {stats['contribution_percentage']:.1f}% of total latency")
            
            return {
                "symbol": symbol,
                "analysis_period_hours": time_window_hours,
                "measurements_analyzed": len(measurements),
                "stage_analysis": stage_stats,
                "bottleneck_stages": sorted_stages[:5],
                "optimization_recommendations": recommendations,
                "average_end_to_end_ms": statistics.mean([m["total_latency_ms"] for m in measurements]),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in critical path analysis: {e}")
            return {"error": str(e)}
    
    async def get_sla_compliance_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate SLA compliance report"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT 
                    data_source,
                    COUNT(*) as total_measurements,
                    SUM(CASE WHEN sla_compliance = true THEN 1 ELSE 0 END) as compliant_measurements,
                    AVG(total_latency_ms) as avg_latency_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms) as p95_latency_ms,
                    MAX(total_latency_ms) as max_latency_ms
                FROM end_to_end_latency 
                WHERE timestamp >= %s
                GROUP BY data_source
                ORDER BY data_source
            """
            
            cursor.execute(query, [datetime.utcnow() - timedelta(hours=time_window_hours)])
            results = cursor.fetchall()
            conn.close()
            
            compliance_report = {}
            overall_compliance = 0
            total_measurements = 0
            
            for row in results:
                data_source = row["data_source"]
                compliance_rate = (row["compliant_measurements"] / row["total_measurements"]) * 100
                sla_target = self.sla_targets.get(DataSourceType(data_source), {}).get("end_to_end", 10000)
                
                compliance_report[data_source] = {
                    "compliance_rate_percentage": compliance_rate,
                    "total_measurements": row["total_measurements"],
                    "compliant_measurements": row["compliant_measurements"],
                    "avg_latency_ms": row["avg_latency_ms"],
                    "p95_latency_ms": row["p95_latency_ms"],
                    "max_latency_ms": row["max_latency_ms"],
                    "sla_target_ms": sla_target,
                    "sla_breach_percentage": ((row["avg_latency_ms"] - sla_target) / sla_target * 100) if row["avg_latency_ms"] > sla_target else 0
                }
                
                overall_compliance += row["compliant_measurements"]
                total_measurements += row["total_measurements"]
            
            overall_compliance_rate = (overall_compliance / total_measurements * 100) if total_measurements > 0 else 0
            
            return {
                "report_period_hours": time_window_hours,
                "overall_compliance_rate": overall_compliance_rate,
                "data_source_compliance": compliance_report,
                "total_measurements": total_measurements,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating SLA compliance report: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.stage_latency_histogram = Histogram(
            'trading_stage_latency_seconds',
            'Latency of individual pipeline stages',
            ['data_source', 'stage'],
            registry=self.metrics_registry,
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.end_to_end_latency_histogram = Histogram(
            'trading_end_to_end_latency_seconds', 
            'End-to-end data pipeline latency',
            ['data_source'],
            registry=self.metrics_registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        )
        
        self.sla_compliance_gauge = Gauge(
            'trading_sla_compliance',
            'SLA compliance status (1=compliant, 0=breach)',
            ['data_source'],
            registry=self.metrics_registry
        )
        
        self.latency_alerts_counter = Counter(
            'trading_latency_alerts_total',
            'Total latency alerts triggered',
            ['data_source', 'stage', 'severity'],
            registry=self.metrics_registry
        )
    
    async def _store_measurement(self, measurement: LatencyMeasurement):
        """Store individual latency measurement"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            # Create table if not exists
            create_table_query = """
                CREATE TABLE IF NOT EXISTS latency_measurements (
                    measurement_id VARCHAR PRIMARY KEY,
                    data_source VARCHAR NOT NULL,
                    stage VARCHAR NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    latency_ms FLOAT NOT NULL,
                    success BOOLEAN NOT NULL,
                    metadata JSONB,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_latency_measurements_timestamp ON latency_measurements(timestamp);
                CREATE INDEX IF NOT EXISTS idx_latency_measurements_data_source ON latency_measurements(data_source);
                CREATE INDEX IF NOT EXISTS idx_latency_measurements_stage ON latency_measurements(stage);
            """
            cursor.execute(create_table_query)
            
            # Insert measurement
            insert_query = """
                INSERT INTO latency_measurements 
                (measurement_id, data_source, stage, timestamp, latency_ms, success, metadata, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (measurement_id) DO NOTHING
            """
            
            cursor.execute(insert_query, [
                measurement.measurement_id,
                measurement.data_source.value,
                measurement.stage.value,
                measurement.timestamp,
                measurement.latency_ms,
                measurement.success,
                json.dumps(measurement.metadata),
                measurement.error_message
            ])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing latency measurement: {e}")
    
    async def _store_end_to_end_measurement(self, e2e: EndToEndLatency):
        """Store end-to-end latency measurement"""
        try:
            conn = psycopg2.connect(self.postgres_url)
            cursor = conn.cursor()
            
            # Create table if not exists
            create_table_query = """
                CREATE TABLE IF NOT EXISTS end_to_end_latency (
                    trace_id VARCHAR PRIMARY KEY,
                    data_source VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    total_latency_ms FLOAT NOT NULL,
                    stage_latencies JSONB,
                    critical_path JSONB,
                    sla_compliance BOOLEAN NOT NULL,
                    sla_target_ms FLOAT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_e2e_latency_timestamp ON end_to_end_latency(timestamp);
                CREATE INDEX IF NOT EXISTS idx_e2e_latency_symbol ON end_to_end_latency(symbol);
                CREATE INDEX IF NOT EXISTS idx_e2e_latency_data_source ON end_to_end_latency(data_source);
            """
            cursor.execute(create_table_query)
            
            # Insert measurement
            insert_query = """
                INSERT INTO end_to_end_latency 
                (trace_id, data_source, symbol, timestamp, total_latency_ms, stage_latencies, critical_path, sla_compliance, sla_target_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trace_id) DO NOTHING
            """
            
            stage_latencies_json = {k.value: v for k, v in e2e.stage_latencies.items()}
            critical_path_json = [stage.value for stage in e2e.critical_path]
            
            cursor.execute(insert_query, [
                e2e.trace_id,
                e2e.data_source.value,
                e2e.symbol,
                e2e.end_timestamp,
                e2e.total_latency_ms,
                json.dumps(stage_latencies_json),
                json.dumps(critical_path_json),
                e2e.sla_compliance,
                e2e.sla_target_ms
            ])
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing end-to-end measurement: {e}")
    
    def _calculate_critical_path(self, stage_latencies: Dict[LatencyStage, float]) -> List[LatencyStage]:
        """Calculate critical path based on stage latencies"""
        # Sort stages by latency (descending) to identify bottlenecks
        sorted_stages = sorted(
            stage_latencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top 3 stages as critical path
        return [stage for stage, _ in sorted_stages[:3]]
    
    async def _check_latency_alerts(self, e2e: EndToEndLatency):
        """Check if latency alerts should be triggered"""
        sla_target = e2e.sla_target_ms
        current_latency = e2e.total_latency_ms
        
        # Check if thresholds are breached
        for alert_type, multiplier in self.alert_thresholds.items():
            threshold = sla_target * multiplier
            
            if current_latency > threshold:
                degradation_pct = ((current_latency - sla_target) / sla_target) * 100
                
                alert = LatencyAlert(
                    alert_id=f"latency_{e2e.data_source.value}_{alert_type}_{int(time.time())}",
                    alert_type=alert_type,
                    data_source=e2e.data_source,
                    stage=LatencyStage.FEATURE_AVAILABILITY,  # End-to-end stage
                    current_latency_ms=current_latency,
                    threshold_ms=threshold,
                    degradation_percentage=degradation_pct,
                    impact_assessment=self._assess_latency_impact(e2e.data_source, current_latency),
                    timestamp=datetime.utcnow(),
                    priority=alert_type
                )
                
                await self._send_latency_alert(alert)
                
                # Update Prometheus counter
                self.latency_alerts_counter.labels(
                    data_source=e2e.data_source.value,
                    stage="end_to_end",
                    severity=alert_type
                ).inc()
                
                break  # Only send highest severity alert
    
    def _assess_latency_impact(self, data_source: DataSourceType, latency_ms: float) -> str:
        """Assess impact of latency degradation on alpha generation"""
        if data_source == DataSourceType.MARKET_DATA:
            if latency_ms > 2000:
                return "SEVERE: High-frequency strategies significantly impacted"
            elif latency_ms > 1000:
                return "HIGH: Market data latency affecting real-time signals"
            else:
                return "MODERATE: Some impact on time-sensitive strategies"
                
        elif data_source == DataSourceType.SENTIMENT:
            if latency_ms > 30000:
                return "HIGH: Sentiment signals significantly delayed"
            elif latency_ms > 15000:
                return "MODERATE: Event-driven strategies may be impacted" 
            else:
                return "LOW: Minimal impact on sentiment-based strategies"
                
        else:
            return "MODERATE: Monitor for strategy performance impact"
    
    async def _send_latency_alert(self, alert: LatencyAlert):
        """Send latency alert"""
        # Store alert in Redis
        await self.redis_client.setex(
            f"latency_alert:{alert.alert_id}",
            86400,  # 24 hours
            json.dumps(asdict(alert), default=str)
        )
        
        logger.warning(f"LATENCY ALERT [{alert.priority.upper()}]: {alert.data_source.value} - {alert.current_latency_ms:.2f}ms (>{alert.threshold_ms:.2f}ms)")
    
    async def _process_latency_measurements(self):
        """Background task to process latency measurements"""
        while True:
            try:
                # Process any queued measurements
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error processing latency measurements: {e}")
                await asyncio.sleep(60)
    
    async def _check_sla_compliance(self):
        """Background task to check SLA compliance"""
        while True:
            try:
                # Check compliance every 5 minutes
                await asyncio.sleep(300)
                
                # Generate compliance report
                report = await self.get_sla_compliance_report(time_window_hours=1)
                
                # Check for severe SLA breaches
                if "data_source_compliance" in report:
                    for data_source, compliance in report["data_source_compliance"].items():
                        if compliance["compliance_rate_percentage"] < 80:  # Less than 80% compliance
                            logger.warning(f"SLA BREACH: {data_source} compliance at {compliance['compliance_rate_percentage']:.1f}%")
                
            except Exception as e:
                logger.error(f"Error checking SLA compliance: {e}")
                await asyncio.sleep(300)
    
    async def _export_prometheus_metrics(self):
        """Background task to export metrics to Prometheus"""
        while True:
            try:
                # Export metrics every 30 seconds
                if self.prometheus_gateway:
                    push_to_gateway(
                        self.prometheus_gateway,
                        job='trading-latency-monitor',
                        registry=self.metrics_registry
                    )
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error exporting Prometheus metrics: {e}")
                await asyncio.sleep(60)