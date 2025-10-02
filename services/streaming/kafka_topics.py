"""
Kafka Topic Definitions and Management
Defines all topics with appropriate partitioning and retention
"""
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import logging

logger = logging.getLogger(__name__)

class TopicTier(Enum):
    """Topic tier for different latency requirements"""
    CRITICAL = "CRITICAL"   # <10ms, highest priority
    HIGH = "HIGH"           # <50ms
    NORMAL = "NORMAL"       # <500ms
    BATCH = "BATCH"         # >1s, batch processing

@dataclass
class TopicConfig:
    """Kafka topic configuration"""
    name: str
    partitions: int
    replication_factor: int
    retention_ms: int  # -1 for infinite
    tier: TopicTier
    compression_type: str = "lz4"
    min_in_sync_replicas: int = 1
    cleanup_policy: str = "delete"  # or "compact"

# ==============================================================================
# TOPIC DEFINITIONS
# ==============================================================================

TOPICS = {
    # Market Data Topics (CRITICAL tier - <10ms)
    "market_data.level1_quotes": TopicConfig(
        name="market_data.level1_quotes",
        partitions=16,  # High parallelism for throughput
        replication_factor=3,
        retention_ms=7 * 24 * 60 * 60 * 1000,  # 7 days
        tier=TopicTier.CRITICAL,
        compression_type="lz4",
        min_in_sync_replicas=2
    ),

    "market_data.trades": TopicConfig(
        name="market_data.trades",
        partitions=16,
        replication_factor=3,
        retention_ms=7 * 24 * 60 * 60 * 1000,
        tier=TopicTier.CRITICAL,
        compression_type="lz4",
        min_in_sync_replicas=2
    ),

    "market_data.level2_depth": TopicConfig(
        name="market_data.level2_depth",
        partitions=16,
        replication_factor=3,
        retention_ms=24 * 60 * 60 * 1000,  # 1 day
        tier=TopicTier.CRITICAL,
        compression_type="lz4",
        min_in_sync_replicas=2
    ),

    # Signals and Predictions (HIGH tier - <50ms)
    "signals.alpha_predictions": TopicConfig(
        name="signals.alpha_predictions",
        partitions=8,
        replication_factor=3,
        retention_ms=30 * 24 * 60 * 60 * 1000,  # 30 days
        tier=TopicTier.HIGH,
        compression_type="lz4",
        min_in_sync_replicas=2
    ),

    "signals.risk_alerts": TopicConfig(
        name="signals.risk_alerts",
        partitions=4,
        replication_factor=3,
        retention_ms=90 * 24 * 60 * 60 * 1000,  # 90 days
        tier=TopicTier.HIGH,
        compression_type="lz4",
        min_in_sync_replicas=2
    ),

    # Orders and Executions (CRITICAL tier)
    "orders.new": TopicConfig(
        name="orders.new",
        partitions=8,
        replication_factor=3,
        retention_ms=365 * 24 * 60 * 60 * 1000,  # 1 year
        tier=TopicTier.CRITICAL,
        compression_type="lz4",
        min_in_sync_replicas=2,
        cleanup_policy="compact"  # Keep latest state per order
    ),

    "orders.fills": TopicConfig(
        name="orders.fills",
        partitions=8,
        replication_factor=3,
        retention_ms=365 * 24 * 60 * 60 * 1000,
        tier=TopicTier.CRITICAL,
        compression_type="lz4",
        min_in_sync_replicas=2
    ),

    "orders.cancels": TopicConfig(
        name="orders.cancels",
        partitions=8,
        replication_factor=3,
        retention_ms=365 * 24 * 60 * 60 * 1000,
        tier=TopicTier.CRITICAL,
        compression_type="lz4",
        min_in_sync_replicas=2
    ),

    # Portfolio and Risk (NORMAL tier - <500ms)
    "portfolio.positions": TopicConfig(
        name="portfolio.positions",
        partitions=4,
        replication_factor=3,
        retention_ms=-1,  # Infinite (compacted)
        tier=TopicTier.NORMAL,
        compression_type="snappy",
        cleanup_policy="compact"
    ),

    "portfolio.pnl_updates": TopicConfig(
        name="portfolio.pnl_updates",
        partitions=4,
        replication_factor=3,
        retention_ms=365 * 24 * 60 * 60 * 1000,
        tier=TopicTier.NORMAL,
        compression_type="snappy"
    ),

    "risk.var_breaches": TopicConfig(
        name="risk.var_breaches",
        partitions=2,
        replication_factor=3,
        retention_ms=365 * 24 * 60 * 60 * 1000,
        tier=TopicTier.HIGH,
        compression_type="snappy",
        min_in_sync_replicas=2
    ),

    # Feature Updates (NORMAL tier)
    "features.realtime_updates": TopicConfig(
        name="features.realtime_updates",
        partitions=16,
        replication_factor=2,
        retention_ms=7 * 24 * 60 * 60 * 1000,
        tier=TopicTier.NORMAL,
        compression_type="lz4"
    ),

    # Model Updates (BATCH tier)
    "models.retraining_complete": TopicConfig(
        name="models.retraining_complete",
        partitions=2,
        replication_factor=2,
        retention_ms=90 * 24 * 60 * 60 * 1000,
        tier=TopicTier.BATCH,
        compression_type="snappy"
    ),

    "models.deployment_events": TopicConfig(
        name="models.deployment_events",
        partitions=2,
        replication_factor=3,
        retention_ms=365 * 24 * 60 * 60 * 1000,
        tier=TopicTier.BATCH,
        compression_type="snappy",
        min_in_sync_replicas=2
    ),

    # Monitoring and Metrics (NORMAL tier)
    "monitoring.latency_metrics": TopicConfig(
        name="monitoring.latency_metrics",
        partitions=4,
        replication_factor=2,
        retention_ms=30 * 24 * 60 * 60 * 1000,
        tier=TopicTier.NORMAL,
        compression_type="snappy"
    ),

    "monitoring.data_quality": TopicConfig(
        name="monitoring.data_quality",
        partitions=2,
        replication_factor=2,
        retention_ms=30 * 24 * 60 * 60 * 1000,
        tier=TopicTier.NORMAL,
        compression_type="snappy"
    ),

    # Dead Letter Queue (BATCH tier)
    "dlq.failed_events": TopicConfig(
        name="dlq.failed_events",
        partitions=4,
        replication_factor=3,
        retention_ms=90 * 24 * 60 * 60 * 1000,
        tier=TopicTier.BATCH,
        compression_type="snappy",
        min_in_sync_replicas=2
    ),
}

class KafkaTopicManager:
    """Manages Kafka topic lifecycle"""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id="topic_manager"
        )
        logger.info(f"Kafka Topic Manager connected to {bootstrap_servers}")

    def create_all_topics(self):
        """Create all defined topics"""
        topics_to_create = []

        for topic_name, config in TOPICS.items():
            new_topic = NewTopic(
                name=config.name,
                num_partitions=config.partitions,
                replication_factor=config.replication_factor,
                topic_configs={
                    "retention.ms": str(config.retention_ms),
                    "compression.type": config.compression_type,
                    "min.insync.replicas": str(config.min_in_sync_replicas),
                    "cleanup.policy": config.cleanup_policy,
                    # Performance tuning
                    "segment.ms": "600000",  # 10 min segments
                    "segment.bytes": "1073741824",  # 1GB segments
                }
            )
            topics_to_create.append(new_topic)

        try:
            result = self.admin_client.create_topics(
                new_topics=topics_to_create,
                validate_only=False
            )

            for topic, future in result.items():
                try:
                    future.result()
                    logger.info(f"Created topic: {topic}")
                except TopicAlreadyExistsError:
                    logger.info(f"Topic already exists: {topic}")
                except Exception as e:
                    logger.error(f"Failed to create topic {topic}: {e}")

        except Exception as e:
            logger.error(f"Error creating topics: {e}")

    def delete_all_topics(self):
        """Delete all managed topics (USE WITH CAUTION)"""
        topic_names = list(TOPICS.keys())

        try:
            result = self.admin_client.delete_topics(topics=topic_names)

            for topic, future in result.items():
                try:
                    future.result()
                    logger.info(f"Deleted topic: {topic}")
                except Exception as e:
                    logger.error(f"Failed to delete topic {topic}: {e}")

        except Exception as e:
            logger.error(f"Error deleting topics: {e}")

    def list_topics(self) -> List[str]:
        """List all topics"""
        metadata = self.admin_client.list_topics()
        return list(metadata)

    def get_topic_config(self, topic_name: str) -> Dict:
        """Get topic configuration"""
        if topic_name in TOPICS:
            return TOPICS[topic_name].__dict__
        return None

    def close(self):
        """Close admin client"""
        self.admin_client.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage Kafka topics")
    parser.add_argument("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--create", action="store_true", help="Create all topics")
    parser.add_argument("--list", action="store_true", help="List all topics")
    parser.add_argument("--delete", action="store_true", help="Delete all topics (USE WITH CAUTION)")

    args = parser.parse_args()

    manager = KafkaTopicManager(bootstrap_servers=args.bootstrap_servers)

    try:
        if args.create:
            print("Creating topics...")
            manager.create_all_topics()
            print("\nTopics created successfully!")

        if args.list:
            print("\nExisting topics:")
            topics = manager.list_topics()
            for topic in sorted(topics):
                print(f"  - {topic}")

        if args.delete:
            confirm = input("Are you sure you want to delete ALL topics? (yes/no): ")
            if confirm.lower() == "yes":
                print("Deleting topics...")
                manager.delete_all_topics()
                print("\nTopics deleted!")
            else:
                print("Deletion cancelled")

    finally:
        manager.close()
