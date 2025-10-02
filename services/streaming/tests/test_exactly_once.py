"""
Tests for Exactly-Once Streaming Processor
Validates idempotency and exactly-once guarantees
"""
import pytest
import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from exactly_once_processor import (
    ExactlyOnceProcessor,
    ProcessingStatus
)

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch('redis.Redis') as mock:
        yield mock.return_value

@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer"""
    with patch('kafka.KafkaConsumer') as mock:
        yield mock.return_value

@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer"""
    with patch('kafka.KafkaProducer') as mock:
        yield mock.return_value

def test_idempotency_key_generation():
    """Test that idempotency keys are generated correctly"""
    processor = ExactlyOnceProcessor()

    # Test explicit idempotency key
    data1 = {"idempotency_key": "ORDER-12345", "order_id": "12345"}
    message1 = MagicMock(topic="orders", partition=0, offset=100, key=None)

    key1 = processor._generate_idempotency_key(data1, message1)
    assert key1 == "ORDER-12345"

    # Test implicit key (hash of metadata)
    data2 = {"order_id": "67890"}
    message2 = MagicMock(topic="orders", partition=0, offset=101, key=b"key1")

    key2 = processor._generate_idempotency_key(data2, message2)
    assert len(key2) == 64  # SHA256 hex digest

    # Same message -> same key
    key2_repeat = processor._generate_idempotency_key(data2, message2)
    assert key2 == key2_repeat

def test_duplicate_detection(mock_redis):
    """Test that duplicate messages are detected and skipped"""
    processor = ExactlyOnceProcessor()
    processor.redis_client = mock_redis

    # Mock: message already processed
    mock_redis.get.return_value = ProcessingStatus.COMPLETED.value

    idempotency_key = "ORDER-12345"

    # Should return True (already processed)
    assert processor._is_processed(idempotency_key) is True

    # Mock: message not processed
    mock_redis.get.return_value = None

    # Should return False
    assert processor._is_processed(idempotency_key) is False

def test_processing_workflow(mock_redis):
    """Test complete processing workflow"""
    processor = ExactlyOnceProcessor()
    processor.redis_client = mock_redis

    idempotency_key = "ORDER-12345"
    message = MagicMock(
        topic="orders",
        partition=0,
        offset=100,
        key=None
    )

    # 1. Mark as processing
    processor._mark_processing(idempotency_key, message)

    # Verify Redis calls
    assert mock_redis.setex.called

    # 2. Mark as completed
    processor._mark_completed(idempotency_key, result_hash="abc123")

    # Verify completed status stored
    calls = mock_redis.setex.call_args_list
    assert any(ProcessingStatus.COMPLETED.value in str(call) for call in calls)

def test_handler_invocation():
    """Test that handler is invoked correctly"""
    processor = ExactlyOnceProcessor()

    # Mock handler
    handler = MagicMock(return_value={"status": "success"})

    message = MagicMock(
        topic="orders",
        partition=0,
        offset=100,
        value=json.dumps({"order_id": "12345"}).encode("utf-8"),
        key=None
    )

    # Mock Redis (not processed yet)
    with patch.object(processor, '_is_processed', return_value=False):
        with patch.object(processor, '_mark_processing'):
            with patch.object(processor, '_mark_completed'):
                processor._process_message_idempotent(
                    message=message,
                    handler=handler,
                    output_topic=None
                )

    # Handler should be called once
    handler.assert_called_once()

def test_duplicate_skipped():
    """Test that duplicate messages skip handler execution"""
    processor = ExactlyOnceProcessor()

    # Mock handler
    handler = MagicMock()

    message = MagicMock(
        topic="orders",
        partition=0,
        offset=100,
        value=json.dumps({"order_id": "12345"}).encode("utf-8"),
        key=None
    )

    # Mock Redis (already processed)
    with patch.object(processor, '_is_processed', return_value=True):
        processor._process_message_idempotent(
            message=message,
            handler=handler,
            output_topic=None
        )

    # Handler should NOT be called
    handler.assert_not_called()

    # Duplicate counter should increment
    assert processor.duplicates_detected == 1

def test_error_handling():
    """Test that errors are handled correctly"""
    processor = ExactlyOnceProcessor()

    # Mock handler that raises error
    handler = MagicMock(side_effect=ValueError("Test error"))

    message = MagicMock(
        topic="orders",
        partition=0,
        offset=100,
        value=json.dumps({"order_id": "12345"}).encode("utf-8"),
        key=None
    )

    # Mock Redis
    with patch.object(processor, '_is_processed', return_value=False):
        with patch.object(processor, '_mark_processing'):
            with patch.object(processor, '_mark_failed') as mock_mark_failed:
                with pytest.raises(ValueError):
                    processor._process_message_idempotent(
                        message=message,
                        handler=handler,
                        output_topic=None
                    )

                # Should mark as failed
                mock_mark_failed.assert_called()

# ==============================================================================
# ACCEPTANCE CRITERIA TESTS
# ==============================================================================

def test_acceptance_exactly_once_delivery():
    """
    ACCEPTANCE: Exactly-once delivery for critical streams

    Same message processed multiple times should produce same side effect only once
    """
    processor = ExactlyOnceProcessor()

    # Simulate processing same message twice
    handler = MagicMock(return_value={"status": "success"})

    message = MagicMock(
        topic="orders",
        partition=0,
        offset=100,
        value=json.dumps({"idempotency_key": "ORDER-12345", "order_id": "12345"}).encode("utf-8"),
        key=None
    )

    # First processing
    with patch.object(processor.redis_client, 'get', return_value=None):
        with patch.object(processor.redis_client, 'setex'):
            processor._process_message_idempotent(message, handler, None)

    # Handler called once
    assert handler.call_count == 1

    # Second processing (duplicate)
    with patch.object(processor.redis_client, 'get', return_value=ProcessingStatus.COMPLETED.value):
        processor._process_message_idempotent(message, handler, None)

    # Handler still called only once (not twice)
    assert handler.call_count == 1

def test_acceptance_message_ordering():
    """
    ACCEPTANCE: Message ordering preserved within partition

    Messages from same partition should be processed in order
    """
    # This is guaranteed by Kafka's partition ordering
    # and our max_in_flight_requests_per_connection=1

    processor = ExactlyOnceProcessor()

    # Verify producer config
    assert processor.producer.config.get("max_in_flight_requests_per_connection") == 1

def test_acceptance_zero_message_loss():
    """
    ACCEPTANCE: Zero message loss under normal operations

    All messages must be processed (no silent drops)
    """
    processor = ExactlyOnceProcessor()

    # Track all messages
    processed_ids = []

    def tracking_handler(data):
        processed_ids.append(data["order_id"])
        return {"status": "success"}

    # Simulate batch of messages
    messages = [
        MagicMock(
            topic="orders",
            partition=0,
            offset=i,
            value=json.dumps({"order_id": f"ORDER-{i}"}).encode("utf-8"),
            key=None
        )
        for i in range(100)
    ]

    # Process all
    with patch.object(processor.redis_client, 'get', return_value=None):
        with patch.object(processor.redis_client, 'setex'):
            for msg in messages:
                processor._process_message_idempotent(msg, tracking_handler, None)

    # All should be processed
    assert len(processed_ids) == 100
    assert processed_ids == [f"ORDER-{i}" for i in range(100)]

def test_acceptance_atomic_commit():
    """
    ACCEPTANCE: Atomic commit of offsets + side effects

    Either both offset and side effect are committed, or neither
    (Transaction semantics)
    """
    processor = ExactlyOnceProcessor()

    # This is tested via Kafka's transaction API
    # Verify producer is initialized with transactions
    assert processor.producer.config.get("transactional_id") is not None
    assert processor.producer.config.get("enable_idempotence") is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
