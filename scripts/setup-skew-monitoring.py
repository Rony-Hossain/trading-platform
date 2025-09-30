#!/usr/bin/env python3
"""
Setup script for Offline/Online Skew Monitoring

This script initializes the skew monitoring system:
- Creates database tables
- Sets up cron jobs
- Validates configuration
- Tests the monitoring pipeline
"""

import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add services to path
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "analysis-service" / "app"))

from core.feature_snapshots import FeatureSnapshotManager
from core.config import get_settings

logger = logging.getLogger(__name__)


async def create_database_tables():
    """Create required database tables for skew monitoring."""
    logger.info("Creating database tables for skew monitoring...")
    
    manager = FeatureSnapshotManager()
    await manager.create_tables()
    
    logger.info("Database tables created successfully")


def validate_configuration():
    """Validate skew monitoring configuration."""
    logger.info("Validating skew monitoring configuration...")
    
    # Check for required environment variables
    required_vars = [
        'SKEW_TOLERANCES_JSON',
        'PROMETHEUS_PUSHGATEWAY_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    # Validate JSON configuration
    try:
        tolerances_json = os.getenv('SKEW_TOLERANCES_JSON', '{}')
        tolerances = json.loads(tolerances_json)
        
        if not tolerances:
            logger.warning("SKEW_TOLERANCES_JSON is empty - no tolerance checking will be performed")
        else:
            logger.info(f"Loaded {len(tolerances)} feature tolerance configurations")
            
        # Validate structure of tolerance config
        for feature, config in tolerances.items():
            if 'ratio_tolerance' not in config and 'absolute_tolerance' not in config:
                logger.error(f"Feature {feature} has no tolerance configuration")
                return False
                
    except json.JSONDecodeError as e:
        logger.error(f"Invalid SKEW_TOLERANCES_JSON format: {e}")
        return False
    
    logger.info("Configuration validation successful")
    return True


def setup_cron_job():
    """Setup cron job for nightly skew monitoring."""
    logger.info("Setting up cron job for skew monitoring...")
    
    script_path = Path(__file__).parent.parent / "jobs" / "guardrails" / "offline_online_skew.py"
    schedule = os.getenv('SKEW_MONITORING_SCHEDULE', '0 2 * * *')  # Default: 2 AM daily
    
    cron_entry = f"{schedule} cd {script_path.parent.parent} && python {script_path}"
    
    logger.info(f"Suggested cron entry:")
    logger.info(f"  {cron_entry}")
    logger.info("Add this to your crontab with: crontab -e")
    
    # Try to add automatically (requires crontab command)
    try:
        import subprocess
        
        # Get current crontab
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        current_crontab = result.stdout if result.returncode == 0 else ""
        
        # Check if entry already exists
        if str(script_path) in current_crontab:
            logger.info("Skew monitoring cron job already exists")
            return True
        
        # Add new entry
        new_crontab = current_crontab + f"\n{cron_entry}\n"
        
        # Write back to crontab
        proc = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, text=True)
        proc.communicate(new_crontab)
        
        if proc.returncode == 0:
            logger.info("Cron job added successfully")
            return True
        else:
            logger.error("Failed to add cron job automatically")
            return False
            
    except Exception as e:
        logger.warning(f"Could not set up cron job automatically: {e}")
        logger.info("Please add the cron job manually")
        return False


async def test_monitoring_pipeline():
    """Test the monitoring pipeline with sample data."""
    logger.info("Testing skew monitoring pipeline...")
    
    try:
        # Import the monitoring module
        sys.path.insert(0, str(Path(__file__).parent.parent / "jobs" / "guardrails"))
        from offline_online_skew import SkewMonitor
        
        # Create test monitor
        monitor = SkewMonitor()
        
        # Test configuration loading
        if not monitor.skew_tolerances:
            logger.warning("No skew tolerances loaded - monitoring will work but no violations will be detected")
        else:
            logger.info(f"Loaded {len(monitor.skew_tolerances)} tolerance configurations")
        
        # Test database connectivity
        manager = FeatureSnapshotManager()
        symbols = await manager.get_symbols_with_snapshots(days_back=1)
        
        if not symbols:
            logger.warning("No symbols with recent snapshots found")
            logger.info("The monitoring system is configured but requires feature data to monitor")
        else:
            logger.info(f"Found {len(symbols)} symbols with recent snapshot data")
        
        logger.info("Pipeline test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False


def create_example_snapshot_data():
    """Create example snapshot data for testing."""
    logger.info("Would you like to create example snapshot data for testing? (y/n)")
    response = input().lower().strip()
    
    if response != 'y':
        return
    
    logger.info("Creating example snapshot data...")
    
    # This would create sample data - implementation depends on your data model
    logger.info("Example data creation not implemented - please populate with real feature data")


async def main():
    """Main setup function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting skew monitoring setup...")
    
    # Validate configuration
    if not validate_configuration():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Create database tables
    try:
        await create_database_tables()
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        sys.exit(1)
    
    # Setup cron job
    setup_cron_job()
    
    # Test pipeline
    pipeline_ok = await test_monitoring_pipeline()
    if not pipeline_ok:
        logger.warning("Pipeline test failed - check logs above")
    
    # Offer to create example data
    create_example_snapshot_data()
    
    logger.info("Skew monitoring setup completed!")
    logger.info("\nNext steps:")
    logger.info("1. Ensure feature data is being populated in both 'offline' and 'online' environments")
    logger.info("2. Run the monitoring job manually to test: python jobs/guardrails/offline_online_skew.py")
    logger.info("3. Check Prometheus metrics at your pushgateway endpoint")
    logger.info("4. Configure Grafana dashboards for visualization")
    logger.info("5. Set up alerting based on the Prometheus rules")


if __name__ == "__main__":
    asyncio.run(main())