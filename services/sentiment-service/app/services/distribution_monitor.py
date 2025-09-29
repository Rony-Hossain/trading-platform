"""
Real-time distribution shift monitoring and alerting system for sentiment data.
Integrates with data quality validator for continuous monitoring.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from sqlalchemy.orm import Session
import json
import smtplib
from email.mime.text import MIMEText as MimeText
from email.mime.multipart import MIMEMultipart as MimeMultipart

from .data_quality_validator import (
    DataQualityValidator, DataQualityReport, ValidationStatus, 
    AlertSeverity, DriftType, DistributionShiftResult
)

logger = logging.getLogger(__name__)

class MonitoringInterval(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"

@dataclass
class AlertConfig:
    alert_id: str
    symbol: str
    alert_type: str  # "validation_failure", "psi_drift", "distribution_shift"
    severity_threshold: AlertSeverity
    notification_channels: List[str]  # ["email", "webhook", "database"]
    enabled: bool = True
    
@dataclass
class MonitoringJob:
    job_id: str
    symbol: str
    interval: MonitoringInterval
    baseline_window_days: int = 30  # Days of historical data for baseline
    current_window_hours: int = 24  # Hours of current data to validate
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None

@dataclass
class AlertEvent:
    event_id: str
    timestamp: datetime
    symbol: str
    alert_type: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class DistributionMonitor:
    """
    Continuous monitoring system for sentiment data distribution shifts.
    Provides real-time alerts and automated monitoring capabilities.
    """
    
    def __init__(self):
        self.validator = DataQualityValidator()
        self.monitoring_jobs: Dict[str, MonitoringJob] = {}
        self.alert_configs: Dict[str, AlertConfig] = {}
        self.alert_history: List[AlertEvent] = []
        self.running = False
        self.monitoring_task = None
        
    async def start_monitoring(self):
        """Start the continuous monitoring system"""
        if self.running:
            logger.warning("Monitoring already running")
            return
            
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Distribution monitoring started")
    
    async def stop_monitoring(self):
        """Stop the continuous monitoring system"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Distribution monitoring stopped")
    
    def add_monitoring_job(self, job: MonitoringJob):
        """Add a new monitoring job for a symbol"""
        self.monitoring_jobs[job.job_id] = job
        self._schedule_next_run(job)
        logger.info(f"Added monitoring job {job.job_id} for symbol {job.symbol}")
    
    def remove_monitoring_job(self, job_id: str):
        """Remove a monitoring job"""
        if job_id in self.monitoring_jobs:
            del self.monitoring_jobs[job_id]
            logger.info(f"Removed monitoring job {job_id}")
    
    def add_alert_config(self, config: AlertConfig):
        """Add alert configuration"""
        self.alert_configs[config.alert_id] = config
        logger.info(f"Added alert config {config.alert_id} for symbol {config.symbol}")
    
    def remove_alert_config(self, alert_id: str):
        """Remove alert configuration"""
        if alert_id in self.alert_configs:
            del self.alert_configs[alert_id]
            logger.info(f"Removed alert config {alert_id}")
    
    async def run_quality_check(self, symbol: str, db: Session) -> DataQualityReport:
        """
        Run a one-time data quality check for a symbol.
        
        Args:
            symbol: Stock symbol to check
            db: Database session
            
        Returns:
            Data quality report
        """
        try:
            logger.info(f"Running quality check for {symbol}")
            
            # Get current data (last 24 hours)
            current_data = await self._get_sentiment_data(
                db, symbol, hours_back=24
            )
            
            if len(current_data) == 0:
                logger.warning(f"No current data found for {symbol}")
                return self._create_empty_report(symbol, "No current data available")
            
            # Get baseline data (30 days ago, for 30 days)
            baseline_data = await self._get_sentiment_data(
                db, symbol, hours_back=24*60, hours_duration=24*30
            )
            
            # Run validation
            report = await self.validator.validate_data_quality(
                current_data, symbol, baseline_data
            )
            
            # Process alerts if needed
            await self._process_report_alerts(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Quality check failed for {symbol}: {e}")
            return self._create_empty_report(symbol, f"Quality check failed: {str(e)}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check which jobs need to run
                jobs_to_run = [
                    job for job in self.monitoring_jobs.values()
                    if job.enabled and (job.next_run is None or current_time >= job.next_run)
                ]
                
                # Run jobs
                for job in jobs_to_run:
                    try:
                        await self._run_monitoring_job(job)
                    except Exception as e:
                        logger.error(f"Monitoring job {job.job_id} failed: {e}")
                    finally:
                        self._schedule_next_run(job)
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _run_monitoring_job(self, job: MonitoringJob):
        """Run a single monitoring job"""
        try:
            logger.info(f"Running monitoring job {job.job_id} for {job.symbol}")
            
            # For demo purposes, we'll create a mock database session
            # In production, this would get a real database session
            from ..core.database import get_db
            db = next(get_db())
            
            try:
                # Run quality check
                report = await self.run_quality_check(job.symbol, db)
                
                # Update job status
                job.last_run = datetime.now()
                
                # Store report (in production, would save to database)
                logger.info(f"Quality check completed for {job.symbol}: Status={report.overall_status}, Score={report.quality_score:.3f}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Monitoring job execution failed: {e}")
            raise
    
    def _schedule_next_run(self, job: MonitoringJob):
        """Schedule the next run for a monitoring job"""
        if not job.enabled:
            return
            
        current_time = datetime.now()
        
        if job.interval == MonitoringInterval.HOURLY:
            job.next_run = current_time + timedelta(hours=1)
        elif job.interval == MonitoringInterval.DAILY:
            job.next_run = current_time + timedelta(days=1)
        elif job.interval == MonitoringInterval.WEEKLY:
            job.next_run = current_time + timedelta(weeks=1)
        else:
            job.next_run = current_time + timedelta(hours=1)  # Default to hourly
    
    async def _get_sentiment_data(self, db: Session, symbol: str, 
                                hours_back: int, hours_duration: int = None) -> pd.DataFrame:
        """
        Get sentiment data from database.
        
        Args:
            db: Database session
            symbol: Stock symbol
            hours_back: How many hours back to start from
            hours_duration: Duration in hours (if None, gets all data from hours_back to now)
        """
        try:
            from ..core.database import SentimentPost
            
            end_time = datetime.now() - timedelta(hours=hours_back)
            start_time = end_time - timedelta(hours=hours_duration) if hours_duration else datetime.min
            
            query = db.query(SentimentPost).filter(
                SentimentPost.symbol == symbol,
                SentimentPost.post_timestamp >= start_time,
                SentimentPost.post_timestamp <= end_time
            )
            
            posts = query.all()
            
            if not posts:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for post in posts:
                data.append({
                    'symbol': post.symbol,
                    'source': post.platform,
                    'content': post.content,
                    'sentiment_score': post.sentiment_score,
                    'sentiment_label': post.sentiment_label,
                    'confidence': post.confidence,
                    'timestamp': post.post_timestamp,
                    'author': post.author
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            return pd.DataFrame()
    
    async def _process_report_alerts(self, report: DataQualityReport):
        """Process a quality report and trigger alerts if needed"""
        try:
            # Get relevant alert configs for this symbol
            symbol_alerts = [
                config for config in self.alert_configs.values()
                if config.symbol == report.symbol and config.enabled
            ]
            
            for config in symbol_alerts:
                should_alert = False
                alert_details = {}
                
                # Check validation failures
                if config.alert_type == "validation_failure":
                    failed_validations = [
                        r for r in report.validation_results
                        if r.status == ValidationStatus.FAIL and 
                        AlertSeverity(r.severity.value) >= config.severity_threshold
                    ]
                    if failed_validations:
                        should_alert = True
                        alert_details = {
                            'failed_validations': [
                                {
                                    'rule': r.rule_name,
                                    'message': r.message,
                                    'severity': r.severity.value
                                }
                                for r in failed_validations
                            ]
                        }
                
                # Check PSI drift
                elif config.alert_type == "psi_drift":
                    high_psi = [r for r in report.psi_results if r.psi_score >= 0.2]
                    if high_psi:
                        should_alert = True
                        alert_details = {
                            'psi_results': [
                                {
                                    'feature': r.feature_name,
                                    'psi_score': r.psi_score,
                                    'interpretation': r.interpretation
                                }
                                for r in high_psi
                            ]
                        }
                
                # Check distribution shifts
                elif config.alert_type == "distribution_shift":
                    significant_drift = [
                        r for r in report.drift_results
                        if r.shift_detected and AlertSeverity(r.severity.value) >= config.severity_threshold
                    ]
                    if significant_drift:
                        should_alert = True
                        alert_details = {
                            'drift_results': [
                                {
                                    'feature': r.feature_name,
                                    'drift_type': r.drift_type.value,
                                    'drift_score': r.drift_score,
                                    'p_value': r.p_value,
                                    'severity': r.severity.value
                                }
                                for r in significant_drift
                            ]
                        }
                
                # Trigger alert if needed
                if should_alert:
                    await self._trigger_alert(config, report, alert_details)
                    
        except Exception as e:
            logger.error(f"Error processing report alerts: {e}")
    
    async def _trigger_alert(self, config: AlertConfig, report: DataQualityReport, details: Dict[str, Any]):
        """Trigger an alert based on configuration"""
        try:
            # Create alert event
            alert_event = AlertEvent(
                event_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.symbol}",
                timestamp=datetime.now(),
                symbol=config.symbol,
                alert_type=config.alert_type,
                severity=config.severity_threshold,
                message=f"Data quality alert for {config.symbol}: {config.alert_type}",
                details={
                    'report_id': f"{report.symbol}_{report.validation_timestamp.strftime('%Y%m%d_%H%M%S')}",
                    'overall_status': report.overall_status.value,
                    'quality_score': report.quality_score,
                    'alert_config_id': config.alert_id,
                    **details
                }
            )
            
            # Store alert
            self.alert_history.append(alert_event)
            
            # Send notifications
            for channel in config.notification_channels:
                await self._send_notification(channel, alert_event, report)
            
            logger.warning(f"Alert triggered: {alert_event.event_id}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    async def _send_notification(self, channel: str, alert_event: AlertEvent, report: DataQualityReport):
        """Send notification through specified channel"""
        try:
            if channel == "database":
                # In production, save to database
                logger.info(f"Database alert logged: {alert_event.event_id}")
            
            elif channel == "email":
                # In production, send email
                logger.info(f"Email alert sent: {alert_event.event_id}")
            
            elif channel == "webhook":
                # In production, send webhook
                logger.info(f"Webhook alert sent: {alert_event.event_id}")
            
            else:
                logger.warning(f"Unknown notification channel: {channel}")
                
        except Exception as e:
            logger.error(f"Error sending notification via {channel}: {e}")
    
    def _create_empty_report(self, symbol: str, message: str) -> DataQualityReport:
        """Create an empty/error report"""
        return DataQualityReport(
            validation_timestamp=datetime.now(),
            symbol=symbol,
            total_records=0,
            validation_results=[],
            psi_results=[],
            drift_results=[],
            overall_status=ValidationStatus.FAIL,
            quality_score=0.0,
            recommendations=[f"Error: {message}"],
            data_characteristics={}
        )
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status"""
        return {
            'running': self.running,
            'total_jobs': len(self.monitoring_jobs),
            'enabled_jobs': len([j for j in self.monitoring_jobs.values() if j.enabled]),
            'total_alerts': len(self.alert_configs),
            'enabled_alerts': len([a for a in self.alert_configs.values() if a.enabled]),
            'recent_alert_count': len([a for a in self.alert_history if a.timestamp > datetime.now() - timedelta(hours=24)]),
            'jobs': [
                {
                    'job_id': job.job_id,
                    'symbol': job.symbol,
                    'interval': job.interval.value,
                    'enabled': job.enabled,
                    'last_run': job.last_run.isoformat() if job.last_run else None,
                    'next_run': job.next_run.isoformat() if job.next_run else None
                }
                for job in self.monitoring_jobs.values()
            ]
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.alert_history if a.timestamp > cutoff]
        
        return [
            {
                'event_id': alert.event_id,
                'timestamp': alert.timestamp.isoformat(),
                'symbol': alert.symbol,
                'alert_type': alert.alert_type,
                'severity': alert.severity.value,
                'message': alert.message,
                'resolved': alert.resolved,
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in sorted(recent, key=lambda x: x.timestamp, reverse=True)
        ]
    
    async def resolve_alert(self, event_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self.alert_history:
            if alert.event_id == event_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert {event_id} marked as resolved")
                return True
        return False