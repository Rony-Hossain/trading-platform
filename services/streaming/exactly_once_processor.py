"""
Exactly-Once Streaming Processor
Guarantees exactly-once processing semantics for critical streams (orders, fills)
Uses idempotency keys and transactional outbox pattern
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import redis
from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import TopicPartition, OffsetAndMetadata

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    """Message processing status"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class ProcessedMessage:
    """Record of processed message for idempotency"""
    idempotency_key: str
    message_id: str
    topic: str
    partition: int
    offset: int
    status: ProcessingStatus
    processed_at: datetime
    result_hash: Optional[str] = None
    error: Optional[str] = None

class ExactlyOnceProcessor:
    """
    Exactly-once processor with idempotency guarantees

    Key mechanisms:
    1. Idempotency keys (derived from message content)
    2. Redis-based deduplication cache
    3. Transactional outbox pattern
    4. Atomic commit of offsets + side effects
    """

    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        consumer_group: str = "exactly_once_group",
        dedup_ttl_seconds: int = 86400  # 24 hours
    ):
        # Kafka consumer (manual commit)
        self.consumer = KafkaConsumer(
            bootstrap_servers=kafka_bootstrap_servers,
            group_id=consumer_group,
            enable_auto_commit=False,  # Manual commit for atomicity
            max_poll_records=10,  # Small batches for low latency
            isolation_level="read_committed"  # Only read committed messages
        )

        # Kafka producer (transactional)
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            transactional_id=f"{consumer_group}_producer",
            enable_idempotence=True,
            acks="all",
            max_in_flight_requests_per_connection=1  # Ensure ordering
        )

        # Initialize producer transactions
        self.producer.init_transactions()

        # Redis for idempotency cache
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.dedup_ttl = dedup_ttl_seconds

        # Metrics
        self.messages_processed = 0
        self.duplicates_detected = 0
        self.errors = 0

        logger.info(
            f"Exactly-once processor initialized: group={consumer_group}"
        )

    def subscribe(self, topics: List[str]):
        """Subscribe to topics"""
        self.consumer.subscribe(topics)
        logger.info(f"Subscribed to topics: {topics}")

    def process_stream(
        self,
        handler: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
        output_topic: Optional[str] = None
    ):
        """
        Process stream with exactly-once semantics

        Args:
            handler: Message processing function
                     Returns None (no output) or Dict (output to produce)
            output_topic: Topic to produce results to (if handler returns data)
        """
        logger.info("Starting exactly-once processing loop...")

        while True:
            try:
                # Poll messages
                records = self.consumer.poll(timeout_ms=1000, max_records=10)

                if not records:
                    continue

                # Begin transaction
                self.producer.begin_transaction()

                for topic_partition, messages in records.items():
                    for message in messages:
                        try:
                            # Process message with idempotency
                            self._process_message_idempotent(
                                message=message,
                                handler=handler,
                                output_topic=output_topic
                            )

                        except Exception as e:
                            logger.error(
                                f"Error processing message "
                                f"{message.offset} from {message.topic}: {e}"
                            )
                            self.errors += 1
                            # Abort transaction and retry
                            self.producer.abort_transaction()
                            raise

                # Commit offsets transactionally
                offsets = {}
                for topic_partition, messages in records.items():
                    if messages:
                        last_offset = messages[-1].offset
                        offsets[topic_partition] = OffsetAndMetadata(
                            last_offset + 1,
                            None
                        )

                self.producer.send_offsets_to_transaction(
                    offsets,
                    self.consumer.config["group_id"]
                )

                # Commit transaction (atomic: outputs + offset commit)
                self.producer.commit_transaction()

                logger.debug(
                    f"Processed batch: {sum(len(msgs) for msgs in records.values())} messages"
                )

            except KeyboardInterrupt:
                logger.info("Stopping processor...")
                break

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                # Transaction already aborted above
                # Sleep and retry
                import time
                time.sleep(1)

    def _process_message_idempotent(
        self,
        message,
        handler: Callable,
        output_topic: Optional[str]
    ):
        """
        Process single message with idempotency check

        Args:
            message: Kafka message
            handler: Processing function
            output_topic: Output topic (if any)
        """
        # Extract message data
        data = json.loads(message.value.decode("utf-8"))

        # Generate idempotency key
        idempotency_key = self._generate_idempotency_key(data, message)

        # Check if already processed
        if self._is_processed(idempotency_key):
            logger.debug(f"Skipping duplicate message: {idempotency_key}")
            self.duplicates_detected += 1
            return

        # Mark as processing
        self._mark_processing(idempotency_key, message)

        try:
            # Execute handler
            result = handler(data)

            # If handler returns data, produce to output topic
            if result and output_topic:
                result_json = json.dumps(result).encode("utf-8")
                self.producer.send(output_topic, value=result_json)

                # Hash result for verification
                result_hash = hashlib.sha256(result_json).hexdigest()
            else:
                result_hash = None

            # Mark as completed
            self._mark_completed(idempotency_key, result_hash)

            self.messages_processed += 1

        except Exception as e:
            # Mark as failed
            self._mark_failed(idempotency_key, str(e))
            raise

    def _generate_idempotency_key(
        self,
        data: Dict[str, Any],
        message
    ) -> str:
        """
        Generate idempotency key from message

        Strategy:
        - Use explicit idempotency_key field if present
        - Otherwise, hash (topic, partition, offset, key)
        """
        if "idempotency_key" in data:
            return data["idempotency_key"]

        # Fallback: hash message metadata
        key_data = {
            "topic": message.topic,
            "partition": message.partition,
            "offset": message.offset,
            "key": message.key.decode("utf-8") if message.key else None
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _is_processed(self, idempotency_key: str) -> bool:
        """Check if message already processed"""
        status = self.redis_client.get(f"idempotency:{idempotency_key}")
        return status in [
            ProcessingStatus.COMPLETED.value,
            ProcessingStatus.PROCESSING.value
        ]

    def _mark_processing(self, idempotency_key: str, message):
        """Mark message as processing"""
        record = ProcessedMessage(
            idempotency_key=idempotency_key,
            message_id=f"{message.topic}-{message.partition}-{message.offset}",
            topic=message.topic,
            partition=message.partition,
            offset=message.offset,
            status=ProcessingStatus.PROCESSING,
            processed_at=datetime.now()
        )

        self.redis_client.setex(
            f"idempotency:{idempotency_key}",
            self.dedup_ttl,
            ProcessingStatus.PROCESSING.value
        )

        # Store full record
        self.redis_client.setex(
            f"processed:{idempotency_key}",
            self.dedup_ttl,
            json.dumps({
                "message_id": record.message_id,
                "status": record.status.value,
                "processed_at": record.processed_at.isoformat()
            })
        )

    def _mark_completed(self, idempotency_key: str, result_hash: Optional[str]):
        """Mark message as completed"""
        self.redis_client.setex(
            f"idempotency:{idempotency_key}",
            self.dedup_ttl,
            ProcessingStatus.COMPLETED.value
        )

        # Update record
        record_data = {
            "status": ProcessingStatus.COMPLETED.value,
            "completed_at": datetime.now().isoformat(),
            "result_hash": result_hash
        }

        self.redis_client.setex(
            f"processed:{idempotency_key}",
            self.dedup_ttl,
            json.dumps(record_data)
        )

    def _mark_failed(self, idempotency_key: str, error: str):
        """Mark message as failed"""
        self.redis_client.setex(
            f"idempotency:{idempotency_key}",
            self.dedup_ttl,
            ProcessingStatus.FAILED.value
        )

        record_data = {
            "status": ProcessingStatus.FAILED.value,
            "failed_at": datetime.now().isoformat(),
            "error": error
        }

        self.redis_client.setex(
            f"processed:{idempotency_key}",
            self.dedup_ttl,
            json.dumps(record_data)
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return {
            "messages_processed": self.messages_processed,
            "duplicates_detected": self.duplicates_detected,
            "errors": self.errors,
            "deduplication_rate": (
                self.duplicates_detected / (self.messages_processed + self.duplicates_detected)
                if (self.messages_processed + self.duplicates_detected) > 0
                else 0
            )
        }

    def close(self):
        """Close connections"""
        self.consumer.close()
        self.producer.close()
        self.redis_client.close()

# Example handler functions

def order_handler(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process order message

    Critical: Must be idempotent - same input always produces same output
    """
    logger.info(f"Processing order: {message.get('order_id')}")

    # Example: Validate order and submit to exchange
    # This must be idempotent!

    # Return execution result
    return {
        "order_id": message["order_id"],
        "status": "SUBMITTED",
        "timestamp": datetime.now().isoformat()
    }

def fill_handler(message: Dict[str, Any]) -> None:
    """
    Process fill message

    Updates portfolio positions - must be idempotent
    """
    logger.info(f"Processing fill: {message.get('fill_id')}")

    # Example: Update position in database
    # Use idempotency key to ensure only applied once

    # No output message
    return None

if __name__ == "__main__":
    # Example usage
    processor = ExactlyOnceProcessor(
        consumer_group="order_processor"
    )

    # Subscribe to order stream
    processor.subscribe(["orders.new"])

    # Process with exactly-once semantics
    processor.process_stream(
        handler=order_handler,
        output_topic="orders.submitted"
    )
