"""
Feed Outage Chaos Drill

Simulates data feed outages to test system resilience and recovery procedures.
Runs monthly drills for macro, options, and headlines feeds.

Acceptance: MTTR < 30 min; chaos pass-rate ≥ 95%
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedType(Enum):
    """Data feed types"""
    MARKET_DATA = "market_data"
    OPTIONS_CHAIN = "options_chain"
    MACRO_INDICATORS = "macro_indicators"
    NEWS_HEADLINES = "news_headlines"
    SOCIAL_SENTIMENT = "social_sentiment"
    CORPORATE_ACTIONS = "corporate_actions"


class OutageType(Enum):
    """Types of feed outages to simulate"""
    COMPLETE_FAILURE = "complete_failure"
    DEGRADED_PERFORMANCE = "degraded_performance"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    SLOW_RESPONSE = "slow_response"


class DrillStatus(Enum):
    """Drill execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FeedOutageScenario:
    """Chaos scenario configuration"""
    feed_type: FeedType
    outage_type: OutageType
    duration_seconds: int
    expected_mttr_seconds: int = 1800  # 30 minutes default
    severity: str = "P2"

    # Scenario-specific parameters
    degradation_percent: Optional[int] = None  # For DEGRADED_PERFORMANCE
    latency_multiplier: Optional[float] = None  # For SLOW_RESPONSE
    corruption_rate: Optional[float] = None  # For DATA_CORRUPTION


@dataclass
class DrillMetrics:
    """Metrics collected during drill"""
    drill_id: str
    scenario: FeedOutageScenario
    start_time: datetime
    end_time: Optional[datetime] = None

    # Recovery metrics
    outage_detected_at: Optional[datetime] = None
    failover_activated_at: Optional[datetime] = None
    recovery_completed_at: Optional[datetime] = None

    # Calculated metrics
    detection_time_seconds: float = 0.0
    failover_time_seconds: float = 0.0
    recovery_time_seconds: float = 0.0
    total_mttr_seconds: float = 0.0

    # Alert tracking
    alerts_triggered: List[Dict] = field(default_factory=list)
    actions_taken: List[Dict] = field(default_factory=list)

    # Status
    status: DrillStatus = DrillStatus.PENDING
    passed: bool = False
    notes: str = ""

    def calculate_metrics(self):
        """Calculate time-based metrics"""
        if self.outage_detected_at:
            self.detection_time_seconds = (
                self.outage_detected_at - self.start_time
            ).total_seconds()

        if self.failover_activated_at and self.outage_detected_at:
            self.failover_time_seconds = (
                self.failover_activated_at - self.outage_detected_at
            ).total_seconds()

        if self.recovery_completed_at and self.start_time:
            self.total_mttr_seconds = (
                self.recovery_completed_at - self.start_time
            ).total_seconds()

            # Check if passed MTTR requirement
            self.passed = self.total_mttr_seconds <= self.scenario.expected_mttr_seconds

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "drill_id": self.drill_id,
            "scenario": {
                "feed_type": self.scenario.feed_type.value,
                "outage_type": self.scenario.outage_type.value,
                "duration_seconds": self.scenario.duration_seconds,
                "expected_mttr_seconds": self.scenario.expected_mttr_seconds,
                "severity": self.scenario.severity
            },
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": {
                "detection_time_seconds": self.detection_time_seconds,
                "failover_time_seconds": self.failover_time_seconds,
                "recovery_time_seconds": self.recovery_time_seconds,
                "total_mttr_seconds": self.total_mttr_seconds
            },
            "alerts_triggered": self.alerts_triggered,
            "actions_taken": self.actions_taken,
            "status": self.status.value,
            "passed": self.passed,
            "notes": self.notes
        }


class FeedOutageDrill:
    """
    Feed Outage Chaos Engineering Drill

    Simulates various feed outage scenarios and measures system response.
    """

    def __init__(self,
                 results_dir: str = "artifacts/chaos/feed_drills",
                 announce_drills: bool = True):
        """
        Initialize chaos drill runner.

        Args:
            results_dir: Directory to store drill results
            announce_drills: Whether to announce drills beforehand
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.announce_drills = announce_drills

        # Drill history
        self.drill_history: List[DrillMetrics] = []
        self.current_drill: Optional[DrillMetrics] = None

    def create_scenario(self,
                       feed_type: FeedType,
                       outage_type: OutageType,
                       duration_seconds: int = 600) -> FeedOutageScenario:
        """Create a chaos scenario"""
        # Set expected MTTR based on feed type
        mttr_targets = {
            FeedType.MARKET_DATA: 120,  # 2 minutes
            FeedType.OPTIONS_CHAIN: 300,  # 5 minutes
            FeedType.MACRO_INDICATORS: 600,  # 10 minutes
            FeedType.NEWS_HEADLINES: 180,  # 3 minutes
            FeedType.SOCIAL_SENTIMENT: 900,  # 15 minutes
            FeedType.CORPORATE_ACTIONS: 1200  # 20 minutes
        }

        return FeedOutageScenario(
            feed_type=feed_type,
            outage_type=outage_type,
            duration_seconds=duration_seconds,
            expected_mttr_seconds=mttr_targets.get(feed_type, 1800)
        )

    async def announce_drill(self, scenario: FeedOutageScenario):
        """Announce drill 24 hours in advance"""
        announcement = {
            "type": "chaos_drill_announcement",
            "timestamp": datetime.now().isoformat(),
            "drill_scheduled_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            "scenario": {
                "feed_type": scenario.feed_type.value,
                "outage_type": scenario.outage_type.value,
                "duration_seconds": scenario.duration_seconds,
                "expected_mttr": scenario.expected_mttr_seconds
            },
            "message": (
                f"Scheduled chaos drill: {scenario.feed_type.value} "
                f"will experience {scenario.outage_type.value} "
                f"for {scenario.duration_seconds}s. "
                f"Expected recovery time: {scenario.expected_mttr_seconds}s"
            )
        }

        logger.info(f"DRILL ANNOUNCEMENT: {announcement['message']}")

        # In production, this would send to Slack/email
        return announcement

    async def simulate_outage(self, scenario: FeedOutageScenario) -> DrillMetrics:
        """
        Simulate feed outage and measure response.

        Args:
            scenario: Outage scenario to simulate

        Returns:
            DrillMetrics with results
        """
        drill_id = f"drill_{scenario.feed_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        metrics = DrillMetrics(
            drill_id=drill_id,
            scenario=scenario,
            start_time=datetime.now(),
            status=DrillStatus.IN_PROGRESS
        )

        self.current_drill = metrics

        logger.info(f"Starting chaos drill: {drill_id}")
        logger.info(f"Scenario: {scenario.outage_type.value} on {scenario.feed_type.value}")

        try:
            # Phase 1: Inject failure
            logger.info(f"Phase 1: Injecting {scenario.outage_type.value}...")
            await self._inject_failure(scenario, metrics)

            # Phase 2: Wait for detection
            logger.info("Phase 2: Waiting for outage detection...")
            await self._wait_for_detection(scenario, metrics)

            # Phase 3: Monitor failover
            logger.info("Phase 3: Monitoring failover activation...")
            await self._monitor_failover(scenario, metrics)

            # Phase 4: Wait for recovery
            logger.info("Phase 4: Waiting for full recovery...")
            await self._wait_for_recovery(scenario, metrics)

            # Phase 5: Validate recovery
            logger.info("Phase 5: Validating recovery...")
            await self._validate_recovery(scenario, metrics)

            metrics.end_time = datetime.now()
            metrics.status = DrillStatus.COMPLETED
            metrics.calculate_metrics()

            logger.info(f"Drill completed. MTTR: {metrics.total_mttr_seconds:.1f}s")
            logger.info(f"Passed: {metrics.passed}")

        except Exception as e:
            logger.error(f"Drill failed with error: {e}")
            metrics.status = DrillStatus.FAILED
            metrics.notes = f"Error: {str(e)}"
            metrics.passed = False

        finally:
            self.drill_history.append(metrics)
            self.current_drill = None

            # Save results
            await self._save_drill_results(metrics)

        return metrics

    async def _inject_failure(self,
                             scenario: FeedOutageScenario,
                             metrics: DrillMetrics):
        """Inject the failure into the feed"""

        if scenario.outage_type == OutageType.COMPLETE_FAILURE:
            # Simulate complete feed shutdown
            logger.info(f"Shutting down {scenario.feed_type.value} feed...")
            # In production: actually disable feed connector
            await asyncio.sleep(1)

        elif scenario.outage_type == OutageType.DEGRADED_PERFORMANCE:
            # Simulate degraded performance
            degradation = scenario.degradation_percent or 70
            logger.info(f"Degrading {scenario.feed_type.value} performance by {degradation}%...")
            # In production: throttle feed, drop messages
            await asyncio.sleep(1)

        elif scenario.outage_type == OutageType.NETWORK_PARTITION:
            # Simulate network partition
            logger.info(f"Creating network partition for {scenario.feed_type.value}...")
            # In production: firewall rules, network chaos
            await asyncio.sleep(1)

        elif scenario.outage_type == OutageType.DATA_CORRUPTION:
            # Simulate data corruption
            corruption_rate = scenario.corruption_rate or 0.5
            logger.info(f"Injecting data corruption ({corruption_rate*100}% rate)...")
            # In production: corrupt message payloads
            await asyncio.sleep(1)

        elif scenario.outage_type == OutageType.SLOW_RESPONSE:
            # Simulate slow response times
            multiplier = scenario.latency_multiplier or 10.0
            logger.info(f"Increasing feed latency by {multiplier}x...")
            # In production: add artificial delays
            await asyncio.sleep(1)

        metrics.actions_taken.append({
            "timestamp": datetime.now().isoformat(),
            "action": "failure_injected",
            "details": f"{scenario.outage_type.value} injected"
        })

    async def _wait_for_detection(self,
                                 scenario: FeedOutageScenario,
                                 metrics: DrillMetrics):
        """Wait for monitoring system to detect the outage"""

        # Simulate monitoring detection
        # In production: wait for actual Prometheus alerts
        detection_delay = random.uniform(5, 30)  # 5-30 seconds
        await asyncio.sleep(detection_delay)

        metrics.outage_detected_at = datetime.now()

        # Simulate alert triggering
        alert = {
            "timestamp": metrics.outage_detected_at.isoformat(),
            "severity": scenario.severity,
            "feed": scenario.feed_type.value,
            "message": f"{scenario.outage_type.value} detected on {scenario.feed_type.value}",
            "alert_name": f"feed_outage_{scenario.feed_type.value}"
        }

        metrics.alerts_triggered.append(alert)
        logger.warning(f"ALERT: {alert['message']}")

    async def _monitor_failover(self,
                               scenario: FeedOutageScenario,
                               metrics: DrillMetrics):
        """Monitor failover system activation"""

        # Simulate failover activation
        # In production: monitor actual failover systems
        failover_delay = random.uniform(10, 60)  # 10-60 seconds
        await asyncio.sleep(failover_delay)

        metrics.failover_activated_at = datetime.now()

        metrics.actions_taken.append({
            "timestamp": metrics.failover_activated_at.isoformat(),
            "action": "failover_activated",
            "details": f"Backup feed activated for {scenario.feed_type.value}"
        })

        logger.info(f"Failover activated at {metrics.failover_activated_at}")

    async def _wait_for_recovery(self,
                                scenario: FeedOutageScenario,
                                metrics: DrillMetrics):
        """Wait for system to fully recover"""

        # Simulate recovery process
        # In production: wait for actual system recovery
        recovery_delay = random.uniform(30, 120)  # 30-120 seconds
        await asyncio.sleep(recovery_delay)

        metrics.recovery_completed_at = datetime.now()

        metrics.actions_taken.append({
            "timestamp": metrics.recovery_completed_at.isoformat(),
            "action": "recovery_completed",
            "details": f"{scenario.feed_type.value} fully recovered"
        })

        logger.info(f"Recovery completed at {metrics.recovery_completed_at}")

    async def _validate_recovery(self,
                                scenario: FeedOutageScenario,
                                metrics: DrillMetrics):
        """Validate that system has fully recovered"""

        # Simulate validation checks
        # In production: run actual health checks
        validations = [
            "Feed connectivity restored",
            "Data freshness within SLO",
            "Message throughput normal",
            "Error rate below threshold",
            "Latency within acceptable range"
        ]

        for validation in validations:
            logger.info(f"✓ {validation}")
            await asyncio.sleep(0.5)

        metrics.actions_taken.append({
            "timestamp": datetime.now().isoformat(),
            "action": "recovery_validated",
            "details": "All health checks passed"
        })

    async def _save_drill_results(self, metrics: DrillMetrics):
        """Save drill results to file"""

        # Create date-based directory
        date_dir = self.results_dir / datetime.now().strftime("%Y-%m")
        date_dir.mkdir(parents=True, exist_ok=True)

        # Save drill results
        filename = f"{metrics.drill_id}.json"
        filepath = date_dir / filename

        with open(filepath, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Drill results saved to {filepath}")

    def generate_drill_report(self,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict:
        """
        Generate comprehensive drill report.

        Args:
            start_date: Report start date (defaults to 30 days ago)
            end_date: Report end date (defaults to now)

        Returns:
            Drill report dictionary
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Filter drills in period
        period_drills = [
            d for d in self.drill_history
            if start_date <= d.start_time <= end_date
        ]

        if not period_drills:
            return {"error": "No drills in specified period"}

        # Calculate statistics
        total_drills = len(period_drills)
        passed_drills = sum(1 for d in period_drills if d.passed)
        pass_rate = (passed_drills / total_drills) * 100

        avg_mttr = sum(d.total_mttr_seconds for d in period_drills) / total_drills

        # Group by feed type
        by_feed = {}
        for drill in period_drills:
            feed = drill.scenario.feed_type.value
            if feed not in by_feed:
                by_feed[feed] = {
                    "total_drills": 0,
                    "passed": 0,
                    "avg_mttr": 0,
                    "min_mttr": float('inf'),
                    "max_mttr": 0
                }

            by_feed[feed]["total_drills"] += 1
            if drill.passed:
                by_feed[feed]["passed"] += 1

            by_feed[feed]["avg_mttr"] += drill.total_mttr_seconds
            by_feed[feed]["min_mttr"] = min(by_feed[feed]["min_mttr"], drill.total_mttr_seconds)
            by_feed[feed]["max_mttr"] = max(by_feed[feed]["max_mttr"], drill.total_mttr_seconds)

        # Calculate averages
        for feed_data in by_feed.values():
            feed_data["avg_mttr"] /= feed_data["total_drills"]
            feed_data["pass_rate"] = (feed_data["passed"] / feed_data["total_drills"]) * 100

        report = {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_drills": total_drills,
                "passed_drills": passed_drills,
                "failed_drills": total_drills - passed_drills,
                "pass_rate_percent": pass_rate,
                "meets_target": pass_rate >= 95.0,
                "avg_mttr_seconds": avg_mttr,
                "target_mttr_seconds": 1800
            },
            "by_feed_type": by_feed,
            "drills": [d.to_dict() for d in period_drills],
            "generated_at": datetime.now().isoformat()
        }

        return report

    async def run_monthly_drills(self):
        """
        Run monthly chaos drills for all feed types.

        Returns:
            List of drill results
        """
        logger.info("Starting monthly chaos drill suite...")

        scenarios = [
            # Market data drills
            self.create_scenario(FeedType.MARKET_DATA, OutageType.COMPLETE_FAILURE, 300),

            # Options drills
            self.create_scenario(FeedType.OPTIONS_CHAIN, OutageType.DEGRADED_PERFORMANCE, 600),

            # Macro drills
            self.create_scenario(FeedType.MACRO_INDICATORS, OutageType.NETWORK_PARTITION, 900),

            # Headlines drills
            self.create_scenario(FeedType.NEWS_HEADLINES, OutageType.SLOW_RESPONSE, 300),
        ]

        results = []

        for scenario in scenarios:
            if self.announce_drills:
                await self.announce_drill(scenario)

            # Run drill
            metrics = await self.simulate_outage(scenario)
            results.append(metrics)

            # Wait between drills
            await asyncio.sleep(60)

        # Generate report
        report = self.generate_drill_report()

        logger.info(f"\n=== Monthly Drill Report ===")
        logger.info(f"Total Drills: {report['summary']['total_drills']}")
        logger.info(f"Pass Rate: {report['summary']['pass_rate_percent']:.1f}%")
        logger.info(f"Avg MTTR: {report['summary']['avg_mttr_seconds']:.1f}s")
        logger.info(f"Target Met: {report['summary']['meets_target']}")

        return results


async def main():
    """Run chaos drill demonstration"""

    drill_runner = FeedOutageDrill(announce_drills=False)

    # Run a single drill
    logger.info("Running single chaos drill...")

    scenario = drill_runner.create_scenario(
        FeedType.MARKET_DATA,
        OutageType.COMPLETE_FAILURE,
        duration_seconds=300
    )

    metrics = await drill_runner.simulate_outage(scenario)

    print(f"\n=== Drill Results ===")
    print(f"Drill ID: {metrics.drill_id}")
    print(f"Status: {metrics.status.value}")
    print(f"Passed: {metrics.passed}")
    print(f"Detection Time: {metrics.detection_time_seconds:.1f}s")
    print(f"Failover Time: {metrics.failover_time_seconds:.1f}s")
    print(f"Total MTTR: {metrics.total_mttr_seconds:.1f}s")
    print(f"Target MTTR: {scenario.expected_mttr_seconds}s")
    print(f"Alerts Triggered: {len(metrics.alerts_triggered)}")
    print(f"Actions Taken: {len(metrics.actions_taken)}")

    # Run monthly drills
    # logger.info("\nRunning monthly drill suite...")
    # await drill_runner.run_monthly_drills()


if __name__ == "__main__":
    asyncio.run(main())
