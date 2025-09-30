"""
Offline/Online Skew Detection & Guardrail System

Monitors for performance drift between offline backtesting and online production.
Triggers alerts when model performance degrades beyond acceptable thresholds.
Part of Phase 3A institutional-grade monitoring framework.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import asyncio
from enum import Enum

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

class SkewSeverity(Enum):
    """Skew detection severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PerformanceMetric(Enum):
    """Performance metrics to monitor"""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    DAILY_RETURNS_MEAN = "daily_returns_mean"
    DAILY_RETURNS_STD = "daily_returns_std"

@dataclass
class SkewAlert:
    """Represents a detected performance skew"""
    timestamp: datetime
    metric: PerformanceMetric
    offline_value: float
    online_value: float
    skew_magnitude: float
    severity: SkewSeverity
    confidence: float
    strategy_name: str
    lookback_days: int
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceWindow:
    """Performance metrics over a time window"""
    start_date: datetime
    end_date: datetime
    strategy_name: str
    metrics: Dict[PerformanceMetric, float]
    trade_count: int
    data_source: str  # 'offline' or 'online'

class PrometheusMetrics:
    """Prometheus metrics for skew monitoring"""
    
    def __init__(self):
        self.skew_alerts_total = Counter(
            'skew_alerts_total',
            'Total number of skew alerts generated',
            ['strategy', 'metric', 'severity']
        )
        
        self.performance_skew_magnitude = Histogram(
            'performance_skew_magnitude',
            'Magnitude of performance skew detected',
            ['strategy', 'metric'],
            buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.offline_performance = Gauge(
            'offline_performance_value',
            'Offline backtest performance metric value',
            ['strategy', 'metric']
        )
        
        self.online_performance = Gauge(
            'online_performance_value', 
            'Online production performance metric value',
            ['strategy', 'metric']
        )
        
        self.skew_monitoring_runs = Counter(
            'skew_monitoring_runs_total',
            'Total number of skew monitoring runs',
            ['status']
        )

class SkewDetector:
    """Offline/Online Performance Skew Detection System"""
    
    def __init__(self, 
                 db_session: AsyncSession,
                 redis_client: aioredis.Redis,
                 config: Dict[str, Any] = None):
        self.db = db_session
        self.redis = redis_client
        self.config = config or self._default_config()
        self.metrics = PrometheusMetrics()
        
        # Skew detection thresholds
        self.thresholds = {
            SkewSeverity.LOW: 0.15,      # 15% deviation
            SkewSeverity.MEDIUM: 0.30,   # 30% deviation  
            SkewSeverity.HIGH: 0.50,     # 50% deviation
            SkewSeverity.CRITICAL: 1.0,  # 100% deviation
        }
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for skew detection"""
        return {
            "lookback_days": 30,
            "min_trades_required": 20,
            "statistical_significance_threshold": 0.05,
            "alert_cooldown_hours": 6,
            "performance_metrics": [
                PerformanceMetric.SHARPE_RATIO,
                PerformanceMetric.MAX_DRAWDOWN,
                PerformanceMetric.WIN_RATE,
                PerformanceMetric.PROFIT_FACTOR
            ],
            "ignore_weekends": True,
            "require_overlapping_periods": True
        }
    
    async def detect_skew(self, strategy_name: str, 
                         lookback_days: Optional[int] = None) -> List[SkewAlert]:
        """
        Main skew detection logic comparing offline vs online performance
        """
        lookback_days = lookback_days or self.config["lookback_days"]
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            # Get performance data for both offline and online
            offline_perf = await self._get_offline_performance(
                strategy_name, start_date, end_date
            )
            online_perf = await self._get_online_performance(
                strategy_name, start_date, end_date
            )
            
            if not offline_perf or not online_perf:
                logger.warning(f"Insufficient data for skew detection: {strategy_name}")
                return []
            
            # Validate data quality
            if not self._validate_performance_data(offline_perf, online_perf):
                logger.warning(f"Performance data validation failed: {strategy_name}")
                return []
            
            # Detect skews across all metrics
            skew_alerts = []
            for metric in self.config["performance_metrics"]:
                if metric in offline_perf.metrics and metric in online_perf.metrics:
                    alert = await self._detect_metric_skew(
                        metric, offline_perf, online_perf, strategy_name, lookback_days
                    )
                    if alert:
                        skew_alerts.append(alert)
            
            # Update Prometheus metrics
            for alert in skew_alerts:
                self.metrics.skew_alerts_total.labels(
                    strategy=strategy_name,
                    metric=alert.metric.value,
                    severity=alert.severity.value
                ).inc()
                
                self.metrics.performance_skew_magnitude.labels(
                    strategy=strategy_name,
                    metric=alert.metric.value
                ).observe(alert.skew_magnitude)
            
            # Store alerts
            await self._store_skew_alerts(skew_alerts)
            
            self.metrics.skew_monitoring_runs.labels(status='success').inc()
            return skew_alerts
            
        except Exception as e:
            logger.error(f"Error in skew detection for {strategy_name}: {e}")
            self.metrics.skew_monitoring_runs.labels(status='error').inc()
            return []
    
    async def _detect_metric_skew(self, 
                                metric: PerformanceMetric,
                                offline_perf: PerformanceWindow,
                                online_perf: PerformanceWindow,
                                strategy_name: str,
                                lookback_days: int) -> Optional[SkewAlert]:
        """Detect skew for a specific performance metric"""
        
        offline_value = offline_perf.metrics[metric]
        online_value = online_perf.metrics[metric]
        
        # Update Prometheus gauges
        self.metrics.offline_performance.labels(
            strategy=strategy_name, metric=metric.value
        ).set(offline_value)
        
        self.metrics.online_performance.labels(
            strategy=strategy_name, metric=metric.value
        ).set(online_value)
        
        # Calculate skew magnitude
        if offline_value == 0:
            if online_value == 0:
                return None  # Both zero, no skew
            skew_magnitude = float('inf')
        else:
            skew_magnitude = abs(online_value - offline_value) / abs(offline_value)
        
        # Determine severity
        severity = self._classify_skew_severity(skew_magnitude)
        if severity is None:
            return None  # Below threshold
        
        # Check alert cooldown
        if await self._is_alert_on_cooldown(strategy_name, metric):
            return None
        
        # Calculate statistical confidence
        confidence = await self._calculate_statistical_confidence(
            metric, offline_perf, online_perf
        )
        
        # Generate alert message
        message = self._generate_alert_message(
            metric, offline_value, online_value, skew_magnitude, severity
        )
        
        alert = SkewAlert(
            timestamp=datetime.utcnow(),
            metric=metric,
            offline_value=offline_value,
            online_value=online_value,
            skew_magnitude=skew_magnitude,
            severity=severity,
            confidence=confidence,
            strategy_name=strategy_name,
            lookback_days=lookback_days,
            message=message,
            metadata={
                'offline_trades': offline_perf.trade_count,
                'online_trades': online_perf.trade_count,
                'offline_period': f"{offline_perf.start_date} to {offline_perf.end_date}",
                'online_period': f"{online_perf.start_date} to {online_perf.end_date}"
            }
        )
        
        return alert
    
    def _classify_skew_severity(self, skew_magnitude: float) -> Optional[SkewSeverity]:
        """Classify skew magnitude into severity levels"""
        if skew_magnitude >= self.thresholds[SkewSeverity.CRITICAL]:
            return SkewSeverity.CRITICAL
        elif skew_magnitude >= self.thresholds[SkewSeverity.HIGH]:
            return SkewSeverity.HIGH
        elif skew_magnitude >= self.thresholds[SkewSeverity.MEDIUM]:
            return SkewSeverity.MEDIUM
        elif skew_magnitude >= self.thresholds[SkewSeverity.LOW]:
            return SkewSeverity.LOW
        else:
            return None
    
    async def _get_offline_performance(self, 
                                     strategy_name: str,
                                     start_date: datetime,
                                     end_date: datetime) -> Optional[PerformanceWindow]:
        """Get offline backtest performance metrics"""
        # Query backtest results from database
        query = """
        SELECT 
            strategy_name,
            backtest_start_date,
            backtest_end_date,
            total_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            calmar_ratio,
            information_ratio,
            daily_returns_mean,
            daily_returns_std,
            total_trades,
            created_at
        FROM backtest_results 
        WHERE strategy_name = :strategy_name
        AND backtest_end_date >= :start_date
        AND backtest_start_date <= :end_date
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        result = await self.db.execute(query, {
            'strategy_name': strategy_name,
            'start_date': start_date,
            'end_date': end_date
        })
        
        row = result.fetchone()
        if not row:
            return None
        
        metrics = {
            PerformanceMetric.SHARPE_RATIO: row.sharpe_ratio or 0.0,
            PerformanceMetric.MAX_DRAWDOWN: row.max_drawdown or 0.0,
            PerformanceMetric.WIN_RATE: row.win_rate or 0.0,
            PerformanceMetric.PROFIT_FACTOR: row.profit_factor or 0.0,
            PerformanceMetric.CALMAR_RATIO: row.calmar_ratio or 0.0,
            PerformanceMetric.INFORMATION_RATIO: row.information_ratio or 0.0,
            PerformanceMetric.DAILY_RETURNS_MEAN: row.daily_returns_mean or 0.0,
            PerformanceMetric.DAILY_RETURNS_STD: row.daily_returns_std or 0.0,
        }
        
        return PerformanceWindow(
            start_date=row.backtest_start_date,
            end_date=row.backtest_end_date,
            strategy_name=strategy_name,
            metrics=metrics,
            trade_count=row.total_trades or 0,
            data_source='offline'
        )
    
    async def _get_online_performance(self,
                                    strategy_name: str, 
                                    start_date: datetime,
                                    end_date: datetime) -> Optional[PerformanceWindow]:
        """Get online production performance metrics"""
        # Query live trading results
        query = """
        SELECT 
            strategy_name,
            DATE_TRUNC('day', execution_time) as trade_date,
            SUM(realized_pnl) as daily_pnl,
            COUNT(*) as trade_count
        FROM trade_executions
        WHERE strategy_name = :strategy_name
        AND execution_time >= :start_date
        AND execution_time <= :end_date
        AND status = 'filled'
        GROUP BY strategy_name, DATE_TRUNC('day', execution_time)
        ORDER BY trade_date
        """
        
        result = await self.db.execute(query, {
            'strategy_name': strategy_name,
            'start_date': start_date,
            'end_date': end_date
        })
        
        daily_pnl_data = result.fetchall()
        if not daily_pnl_data:
            return None
        
        # Calculate performance metrics from daily P&L
        daily_returns = [row.daily_pnl for row in daily_pnl_data]
        total_trades = sum(row.trade_count for row in daily_pnl_data)
        
        metrics = self._calculate_performance_metrics(daily_returns)
        
        return PerformanceWindow(
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy_name,
            metrics=metrics,
            trade_count=total_trades,
            data_source='online'
        )
    
    def _calculate_performance_metrics(self, daily_returns: List[float]) -> Dict[PerformanceMetric, float]:
        """Calculate performance metrics from daily returns"""
        if not daily_returns:
            return {}
        
        returns_array = np.array(daily_returns)
        
        # Basic statistics
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1) if len(returns_array) > 1 else 0.0
        
        # Sharpe ratio (annualized)
        sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0.0
        
        # Max drawdown
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate
        winning_days = np.sum(returns_array > 0)
        win_rate = winning_days / len(returns_array) if len(returns_array) > 0 else 0.0
        
        # Profit factor
        gross_profit = np.sum(returns_array[returns_array > 0])
        gross_loss = abs(np.sum(returns_array[returns_array < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Calmar ratio
        calmar_ratio = (mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            PerformanceMetric.SHARPE_RATIO: sharpe_ratio,
            PerformanceMetric.MAX_DRAWDOWN: max_drawdown,
            PerformanceMetric.WIN_RATE: win_rate,
            PerformanceMetric.PROFIT_FACTOR: profit_factor,
            PerformanceMetric.CALMAR_RATIO: calmar_ratio,
            PerformanceMetric.DAILY_RETURNS_MEAN: mean_return,
            PerformanceMetric.DAILY_RETURNS_STD: std_return,
        }
    
    def _validate_performance_data(self, 
                                 offline_perf: PerformanceWindow,
                                 online_perf: PerformanceWindow) -> bool:
        """Validate that performance data is suitable for comparison"""
        # Check minimum trades
        min_trades = self.config["min_trades_required"]
        if offline_perf.trade_count < min_trades or online_perf.trade_count < min_trades:
            return False
        
        # Check data quality
        if not offline_perf.metrics or not online_perf.metrics:
            return False
        
        # Check for required metrics
        required_metrics = set(self.config["performance_metrics"])
        offline_metrics = set(offline_perf.metrics.keys())
        online_metrics = set(online_perf.metrics.keys())
        
        if not required_metrics.issubset(offline_metrics) or not required_metrics.issubset(online_metrics):
            return False
        
        return True
    
    async def _calculate_statistical_confidence(self,
                                              metric: PerformanceMetric,
                                              offline_perf: PerformanceWindow,
                                              online_perf: PerformanceWindow) -> float:
        """Calculate statistical confidence of the observed skew"""
        # Simplified confidence calculation - can be enhanced with proper statistical tests
        offline_trades = offline_perf.trade_count
        online_trades = online_perf.trade_count
        
        # Higher confidence with more trades
        min_sample_size = max(offline_trades, online_trades)
        confidence = min(min_sample_size / 100.0, 1.0)  # Cap at 1.0
        
        return confidence
    
    async def _is_alert_on_cooldown(self, strategy_name: str, metric: PerformanceMetric) -> bool:
        """Check if alert is on cooldown to prevent spam"""
        cooldown_key = f"skew_alert_cooldown:{strategy_name}:{metric.value}"
        cooldown_hours = self.config["alert_cooldown_hours"]
        
        last_alert = await self.redis.get(cooldown_key)
        if last_alert:
            last_alert_time = datetime.fromisoformat(last_alert.decode())
            if datetime.utcnow() - last_alert_time < timedelta(hours=cooldown_hours):
                return True
        
        # Set cooldown
        await self.redis.setex(
            cooldown_key, 
            timedelta(hours=cooldown_hours),
            datetime.utcnow().isoformat()
        )
        return False
    
    def _generate_alert_message(self,
                              metric: PerformanceMetric,
                              offline_value: float,
                              online_value: float,
                              skew_magnitude: float,
                              severity: SkewSeverity) -> str:
        """Generate human-readable alert message"""
        direction = "deteriorated" if online_value < offline_value else "improved"
        
        message = (
            f"{severity.value.upper()} skew detected in {metric.value}: "
            f"Offline: {offline_value:.4f}, Online: {online_value:.4f} "
            f"({skew_magnitude:.1%} {direction})"
        )
        
        return message
    
    async def _store_skew_alerts(self, alerts: List[SkewAlert]) -> None:
        """Store skew alerts in database and cache"""
        for alert in alerts:
            # Store in database
            query = """
            INSERT INTO skew_alerts (
                timestamp, strategy_name, metric, offline_value, online_value,
                skew_magnitude, severity, confidence, lookback_days, message, metadata
            ) VALUES (
                :timestamp, :strategy_name, :metric, :offline_value, :online_value,
                :skew_magnitude, :severity, :confidence, :lookback_days, :message, :metadata
            )
            """
            
            await self.db.execute(query, {
                'timestamp': alert.timestamp,
                'strategy_name': alert.strategy_name,
                'metric': alert.metric.value,
                'offline_value': alert.offline_value,
                'online_value': alert.online_value,
                'skew_magnitude': alert.skew_magnitude,
                'severity': alert.severity.value,
                'confidence': alert.confidence,
                'lookback_days': alert.lookback_days,
                'message': alert.message,
                'metadata': json.dumps(alert.metadata)
            })
            
            # Store in Redis for fast access
            alert_key = f"skew_alert:{alert.strategy_name}:{alert.timestamp.isoformat()}"
            alert_data = {
                'metric': alert.metric.value,
                'severity': alert.severity.value,
                'message': alert.message,
                'skew_magnitude': alert.skew_magnitude
            }
            await self.redis.setex(alert_key, timedelta(days=7), json.dumps(alert_data))
        
        await self.db.commit()
    
    async def run_nightly_monitoring(self) -> Dict[str, List[SkewAlert]]:
        """Run nightly skew monitoring for all active strategies"""
        logger.info("Starting nightly skew monitoring...")
        
        # Get list of active strategies
        query = "SELECT DISTINCT strategy_name FROM strategy_configs WHERE is_active = true"
        result = await self.db.execute(query)
        active_strategies = [row.strategy_name for row in result.fetchall()]
        
        all_alerts = {}
        for strategy_name in active_strategies:
            try:
                alerts = await self.detect_skew(strategy_name)
                if alerts:
                    all_alerts[strategy_name] = alerts
                    logger.info(f"Detected {len(alerts)} skew alerts for {strategy_name}")
            except Exception as e:
                logger.error(f"Error monitoring {strategy_name}: {e}")
        
        logger.info(f"Nightly skew monitoring completed. Total strategies: {len(active_strategies)}, "
                   f"Strategies with alerts: {len(all_alerts)}")
        
        return all_alerts

async def start_skew_monitoring_service(port: int = 8090):
    """Start the skew monitoring service with Prometheus metrics endpoint"""
    # Start Prometheus metrics server
    start_http_server(port)
    logger.info(f"Skew monitoring Prometheus metrics available at http://localhost:{port}")
    
    # Additional service initialization would go here
    # (database connections, Redis connections, etc.)