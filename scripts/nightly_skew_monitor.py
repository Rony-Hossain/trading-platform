#!/usr/bin/env python3
"""
Nightly Skew Monitoring Script

Runs offline/online performance skew detection and sends alerts.
Integrates with Prometheus metrics and notification systems.
Scheduled to run via cron at 2 AM daily.

Usage:
    python scripts/nightly_skew_monitor.py
    python scripts/nightly_skew_monitor.py --strategies SPY_MOMENTUM,VIX_MEAN_REVERT
    python scripts/nightly_skew_monitor.py --dry-run --verbose
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add services/analysis-service to path
analysis_service_path = project_root / 'services' / 'analysis-service'
sys.path.insert(0, str(analysis_service_path))

try:
    from app.monitoring.skew_detector import SkewDetector, SkewAlert, start_skew_monitoring_service
    from app.core.database import get_database_session
    import aioredis
    import asyncpg
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Ensure you're running from project root and dependencies are installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'logs' / 'skew_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SkewMonitoringOrchestrator:
    """Orchestrates nightly skew monitoring across all strategies"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.db_session = None
        self.redis_client = None
        self.skew_detector = None
        
    async def initialize(self):
        """Initialize database and Redis connections"""
        try:
            # Database connection
            database_url = os.getenv(
                'DATABASE_URL', 
                'postgresql+asyncpg://trading_user:trading_pass@localhost:5432/trading_db'
            )
            self.db_session = await get_database_session(database_url)
            
            # Redis connection
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = await aioredis.from_url(redis_url)
            
            # Initialize skew detector
            self.skew_detector = SkewDetector(
                db_session=self.db_session,
                redis_client=self.redis_client
            )
            
            logger.info("Skew monitoring orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize skew monitoring: {e}")
            raise
    
    async def run_monitoring(self, specific_strategies: List[str] = None) -> Dict[str, List[SkewAlert]]:
        """Run skew monitoring for specified strategies or all active strategies"""
        if self.dry_run:
            logger.info("DRY RUN MODE - No alerts will be sent or stored")
        
        logger.info("Starting nightly skew monitoring...")
        start_time = datetime.utcnow()
        
        try:
            if specific_strategies:
                # Monitor specific strategies
                all_alerts = {}
                for strategy_name in specific_strategies:
                    logger.info(f"Monitoring strategy: {strategy_name}")
                    alerts = await self.skew_detector.detect_skew(strategy_name)
                    if alerts:
                        all_alerts[strategy_name] = alerts
            else:
                # Monitor all active strategies
                all_alerts = await self.skew_detector.run_nightly_monitoring()
            
            # Process alerts
            await self._process_alerts(all_alerts)
            
            # Generate summary report
            summary = self._generate_summary_report(all_alerts, start_time)
            logger.info(f"Monitoring completed: {summary}")
            
            return all_alerts
            
        except Exception as e:
            logger.error(f"Error during skew monitoring: {e}")
            raise
    
    async def _process_alerts(self, all_alerts: Dict[str, List[SkewAlert]]):
        """Process and send notifications for detected skew alerts"""
        if not all_alerts:
            logger.info("No skew alerts detected")
            return
        
        # Count alerts by severity
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        critical_alerts = []
        
        for strategy_name, alerts in all_alerts.items():
            for alert in alerts:
                severity_counts[alert.severity.value] += 1
                if alert.severity.value in ['critical', 'high']:
                    critical_alerts.append(alert)
        
        # Send notifications
        if not self.dry_run:
            await self._send_notifications(all_alerts, critical_alerts)
        
        # Log alert summary
        logger.info(f"Alert summary: {severity_counts}")
        
        if critical_alerts:
            logger.warning(f"CRITICAL/HIGH alerts detected: {len(critical_alerts)}")
            for alert in critical_alerts:
                logger.warning(f"  {alert.strategy_name}: {alert.message}")
    
    async def _send_notifications(self, 
                                all_alerts: Dict[str, List[SkewAlert]], 
                                critical_alerts: List[SkewAlert]):
        """Send notifications via multiple channels"""
        
        # Send Slack notification for critical alerts
        if critical_alerts:
            await self._send_slack_notification(critical_alerts)
        
        # Send email summary
        await self._send_email_summary(all_alerts)
        
        # Update monitoring dashboard
        await self._update_dashboard(all_alerts)
    
    async def _send_slack_notification(self, critical_alerts: List[SkewAlert]):
        """Send Slack notification for critical alerts"""
        try:
            slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
            if not slack_webhook:
                logger.warning("No Slack webhook configured")
                return
            
            # Format Slack message
            message = {
                "text": f"🚨 CRITICAL Performance Skew Detected ({len(critical_alerts)} alerts)",
                "attachments": []
            }
            
            for alert in critical_alerts:
                attachment = {
                    "color": "danger" if alert.severity.value == 'critical' else "warning",
                    "fields": [
                        {"title": "Strategy", "value": alert.strategy_name, "short": True},
                        {"title": "Metric", "value": alert.metric.value, "short": True},
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Skew", "value": f"{alert.skew_magnitude:.1%}", "short": True},
                        {"title": "Message", "value": alert.message, "short": False}
                    ],
                    "timestamp": int(alert.timestamp.timestamp())
                }
                message["attachments"].append(attachment)
            
            # Send to Slack (implementation would use requests or aiohttp)
            logger.info(f"Would send Slack notification for {len(critical_alerts)} critical alerts")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    async def _send_email_summary(self, all_alerts: Dict[str, List[SkewAlert]]):
        """Send email summary of all alerts"""
        try:
            # Generate email content
            total_alerts = sum(len(alerts) for alerts in all_alerts.values())
            
            if total_alerts == 0:
                subject = "Nightly Skew Monitoring - No Issues Detected"
                body = "All strategy performance metrics are within acceptable ranges."
            else:
                subject = f"Nightly Skew Monitoring - {total_alerts} Alerts Detected"
                body = self._generate_email_body(all_alerts)
            
            # Send email (implementation would use SMTP or email service)
            logger.info(f"Would send email summary: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email summary: {e}")
    
    def _generate_email_body(self, all_alerts: Dict[str, List[SkewAlert]]) -> str:
        """Generate detailed email body with alert information"""
        body_lines = ["Performance Skew Monitoring Report", "=" * 40, ""]
        
        for strategy_name, alerts in all_alerts.items():
            body_lines.append(f"Strategy: {strategy_name}")
            body_lines.append("-" * 20)
            
            for alert in alerts:
                body_lines.extend([
                    f"  • {alert.metric.value}: {alert.severity.value.upper()}",
                    f"    Offline: {alert.offline_value:.4f}, Online: {alert.online_value:.4f}",
                    f"    Skew: {alert.skew_magnitude:.1%}",
                    f"    Message: {alert.message}",
                    ""
                ])
        
        return "\n".join(body_lines)
    
    async def _update_dashboard(self, all_alerts: Dict[str, List[SkewAlert]]):
        """Update monitoring dashboard with latest alerts"""
        try:
            # Store dashboard data in Redis
            dashboard_data = {
                "last_update": datetime.utcnow().isoformat(),
                "total_strategies_monitored": len(all_alerts),
                "total_alerts": sum(len(alerts) for alerts in all_alerts.values()),
                "alerts_by_strategy": {
                    strategy: len(alerts) for strategy, alerts in all_alerts.items()
                }
            }
            
            await self.redis_client.setex(
                "skew_monitoring_dashboard",
                timedelta(hours=25),  # Keep for more than 24h
                json.dumps(dashboard_data)
            )
            
            logger.info("Updated monitoring dashboard")
            
        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")
    
    def _generate_summary_report(self, 
                               all_alerts: Dict[str, List[SkewAlert]], 
                               start_time: datetime) -> str:
        """Generate summary report of monitoring run"""
        duration = datetime.utcnow() - start_time
        total_strategies = len(all_alerts) if all_alerts else 0
        total_alerts = sum(len(alerts) for alerts in all_alerts.values()) if all_alerts else 0
        
        return (
            f"Strategies monitored: {total_strategies}, "
            f"Alerts generated: {total_alerts}, "
            f"Duration: {duration.total_seconds():.1f}s"
        )
    
    async def cleanup(self):
        """Clean up database and Redis connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.db_session:
            await self.db_session.close()

async def main():
    """Main entry point for nightly skew monitoring"""
    parser = argparse.ArgumentParser(description="Nightly Skew Monitoring")
    parser.add_argument(
        '--strategies',
        help='Comma-separated list of strategies to monitor (default: all active)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without sending alerts or storing results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--start-prometheus',
        action='store_true',
        help='Start Prometheus metrics server'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse strategies list
    specific_strategies = None
    if args.strategies:
        specific_strategies = [s.strip() for s in args.strategies.split(',')]
    
    orchestrator = SkewMonitoringOrchestrator(dry_run=args.dry_run)
    
    try:
        # Start Prometheus metrics server if requested
        if args.start_prometheus:
            await start_skew_monitoring_service(port=8090)
        
        # Initialize and run monitoring
        await orchestrator.initialize()
        all_alerts = await orchestrator.run_monitoring(specific_strategies)
        
        # Exit with appropriate code
        critical_alerts = []
        if all_alerts:
            for alerts in all_alerts.values():
                critical_alerts.extend([a for a in alerts if a.severity.value in ['critical', 'high']])
        
        if critical_alerts and not args.dry_run:
            logger.error(f"Exiting with error code due to {len(critical_alerts)} critical/high alerts")
            sys.exit(1)
        else:
            logger.info("Monitoring completed successfully")
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(2)
    
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    # Ensure logs directory exists
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Run main
    asyncio.run(main())