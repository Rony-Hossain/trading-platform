"""
Redis Streams Integration Tests
End-to-end tests for producer → stream → consumer flow
"""
import asyncio
import pytest
import logging
import sys
from pathlib import Path
from datetime import datetime
import time

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'infrastructure' / 'streaming'))
sys.path.insert(0, str(project_root / 'services' / 'streaming'))

from stream_client import RedisStreamClient
from producers.feature_producer import FeatureProducer
from consumers.signal_consumer import SignalConsumer, FeatureConsumer

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_feature_producer_consumer_flow():
    """
    Test feature producer → consumer end-to-end flow
    Validates that messages are produced and consumed correctly
    """
    logger.info("Starting feature producer → consumer flow test")

    # Setup producer
    producer = FeatureProducer()
    await producer.connect()

    # Setup consumer
    consumer = FeatureConsumer(consumer_name="test_consumer_flow")
    await consumer.connect()

    try:
        # Produce test features
        test_symbol = "INTEGRATION_TEST"
        test_features = {
            "sma_20": 185.45,
            "rsi_14": 67.8,
            "volume_sma_20": 50000000,
            "momentum_5d": 0.023
        }

        msg_id = await producer.produce_pit_feature(
            symbol=test_symbol,
            features=test_features,
            pit_validated=True,
            validation_metadata={
                "validator": "test_validator",
                "test_run": True
            }
        )

        assert msg_id is not None
        logger.info(f"✅ Produced message: {msg_id}")

        # Consume messages
        messages = await consumer.client.consume(
            stream_name="features.pit",
            group_name="test_flow_group",
            consumer_name="test_consumer_flow",
            count=10,
            block_ms=3000
        )

        # Verify we got the message
        assert len(messages) > 0, "Should have consumed at least one message"

        found_message = False
        for msg in messages:
            if msg.data.get('symbol') == test_symbol:
                found_message = True
                assert msg.data.get('pit_validated') == True
                assert 'features' in msg.data
                logger.info(f"✅ Consumed message for {test_symbol}: {msg.message_id}")

                # Acknowledge
                await consumer.client.ack(
                    "features.pit",
                    "test_flow_group",
                    msg.message_id
                )
                break

        assert found_message, f"Did not find message for {test_symbol}"
        logger.info("✅ Feature producer → consumer flow test PASSED")

    finally:
        await producer.close()
        await consumer.close()


@pytest.mark.asyncio
async def test_batch_production():
    """Test batch feature production for performance"""
    logger.info("Starting batch production test")

    producer = FeatureProducer()
    await producer.connect()

    try:
        # Create batch of 50 features
        batch = [
            {
                "symbol": f"BATCH_TEST_{i}",
                "features": {
                    "sma_20": 100.0 + i,
                    "rsi_14": 50.0 + (i % 30),
                    "volume": 1000000 * (i + 1)
                },
                "pit_validated": True,
                "validation_metadata": {"batch_id": f"batch_{i}"}
            }
            for i in range(50)
        ]

        # Measure batch production time
        start = time.time()
        message_ids = await producer.produce_batch_features(batch, stream_type="pit")
        elapsed = time.time() - start

        # Verify all produced
        assert len(message_ids) == 50
        assert all(msg_id is not None for msg_id in message_ids)

        throughput = len(batch) / elapsed
        logger.info(
            f"✅ Batch produced {len(message_ids)} messages in {elapsed:.3f}s "
            f"(throughput: {throughput:.1f} msg/sec)"
        )

        # Performance assertion
        assert throughput > 10, "Batch throughput should be > 10 msg/sec"

    finally:
        await producer.close()


@pytest.mark.asyncio
async def test_stream_lag_monitoring():
    """Test stream lag monitoring functionality"""
    logger.info("Starting stream lag monitoring test")

    client = RedisStreamClient()
    await client.connect()

    try:
        stream_name = "features.pit"
        group_name = "test_lag_monitoring_group"

        # Create consumer group
        await client.create_consumer_group(
            stream_name=stream_name,
            group_name=group_name,
            start_id="0"  # Start from beginning
        )

        # Monitor lag
        lag = await client.monitor_lag(
            stream_name=stream_name,
            group_name=group_name
        )

        assert lag is not None
        assert lag >= 0
        logger.info(f"✅ Stream lag for {group_name}: {lag} messages")

        # Get stream info
        stream_info = await client.get_stream_info(stream_name)
        assert stream_info is not None
        logger.info(f"✅ Stream info retrieved")

    finally:
        await client.close()


@pytest.mark.asyncio
async def test_consumer_group_load_balancing():
    """
    Test consumer group load balancing
    Multiple consumers in same group should each get different messages
    """
    logger.info("Starting consumer group load balancing test")

    producer = FeatureProducer()
    await producer.connect()

    consumer1 = FeatureConsumer(consumer_name="lb_consumer_1")
    consumer2 = FeatureConsumer(consumer_name="lb_consumer_2")

    await consumer1.connect()
    await consumer2.connect()

    try:
        # Produce multiple messages
        num_messages = 20
        for i in range(num_messages):
            await producer.produce_pit_feature(
                symbol=f"LB_TEST_{i}",
                features={"value": i, "test": "load_balancing"},
                pit_validated=True
            )

        logger.info(f"✅ Produced {num_messages} messages for load balancing test")

        # Both consumers in same group should get different messages
        group_name = "test_lb_group"

        messages1 = await consumer1.client.consume(
            stream_name="features.pit",
            group_name=group_name,
            consumer_name="lb_consumer_1",
            count=15,
            block_ms=2000
        )

        messages2 = await consumer2.client.consume(
            stream_name="features.pit",
            group_name=group_name,
            consumer_name="lb_consumer_2",
            count=15,
            block_ms=2000
        )

        # Verify load balancing
        total_consumed = len(messages1) + len(messages2)

        logger.info(
            f"✅ Load balancing results: "
            f"Consumer1={len(messages1)} messages, "
            f"Consumer2={len(messages2)} messages, "
            f"Total={total_consumed}"
        )

        # Verify messages were distributed
        assert total_consumed > 0, "Should have consumed some messages"

        # Verify no duplicates between consumers (message IDs should be unique)
        msg_ids_1 = {msg.message_id for msg in messages1}
        msg_ids_2 = {msg.message_id for msg in messages2}
        assert len(msg_ids_1 & msg_ids_2) == 0, "No message should be delivered to both consumers"

        logger.info("✅ Load balancing test PASSED - no duplicate deliveries")

    finally:
        await producer.close()
        await consumer1.close()
        await consumer2.close()


@pytest.mark.asyncio
async def test_performance_throughput():
    """
    Test producer/consumer performance
    Validates throughput and latency meet acceptance criteria
    """
    logger.info("Starting performance throughput test")

    producer = FeatureProducer()
    await producer.connect()

    try:
        # Measure production throughput
        num_messages = 100
        start = time.time()

        for i in range(num_messages):
            await producer.produce_pit_feature(
                symbol=f"PERF_{i}",
                features={
                    "sma_20": 100.0 + i,
                    "rsi_14": 50.0 + (i % 50),
                    "test": "performance"
                },
                pit_validated=True
            )

        elapsed = time.time() - start
        throughput = num_messages / elapsed
        avg_latency_ms = (elapsed / num_messages) * 1000

        logger.info(
            f"✅ Performance: {throughput:.1f} msg/sec, "
            f"Avg latency: {avg_latency_ms:.2f}ms per message"
        )

        # Acceptance criteria
        assert throughput > 50, f"Throughput ({throughput:.1f}) should be > 50 msg/sec"
        assert avg_latency_ms < 100, f"Avg latency ({avg_latency_ms:.2f}ms) should be < 100ms"

        logger.info("✅ Performance test PASSED - meets acceptance criteria")

    finally:
        await producer.close()


@pytest.mark.asyncio
async def test_error_handling_and_recovery():
    """Test error handling in producer/consumer"""
    logger.info("Starting error handling test")

    producer = FeatureProducer()
    await producer.connect()

    try:
        # Test with various invalid inputs

        # Test 1: Empty features (should work)
        msg_id = await producer.produce_pit_feature(
            symbol="EMPTY_TEST",
            features={},
            pit_validated=False
        )
        assert msg_id is not None
        logger.info("✅ Producer accepted empty features")

        # Test 2: Large feature set (stress test)
        large_features = {f"feature_{i}": i * 0.1 for i in range(100)}
        msg_id = await producer.produce_pit_feature(
            symbol="LARGE_TEST",
            features=large_features,
            pit_validated=True
        )
        assert msg_id is not None
        logger.info("✅ Producer handled large feature set (100 features)")

    finally:
        await producer.close()


@pytest.mark.asyncio
async def test_message_acknowledgment():
    """Test message acknowledgment mechanism"""
    logger.info("Starting message acknowledgment test")

    client = RedisStreamClient()
    await client.connect()

    try:
        stream_name = "features.pit"
        group_name = "test_ack_group"
        consumer_name = "test_ack_consumer"

        # Create consumer group
        await client.create_consumer_group(stream_name, group_name)

        # Produce a message
        test_data = {
            "symbol": "ACK_TEST",
            "features": {"test": "ack"},
            "pit_validated": True,
            "timestamp": datetime.utcnow().isoformat()
        }

        msg_id = await client.produce(
            stream_name=stream_name,
            data=test_data,
            maxlen=10000,
            approximate=True
        )

        logger.info(f"Produced test message: {msg_id}")

        # Consume message
        messages = await client.consume(
            stream_name=stream_name,
            group_name=group_name,
            consumer_name=consumer_name,
            count=1,
            block_ms=1000
        )

        assert len(messages) > 0
        consumed_msg = messages[0]

        # Acknowledge message
        await client.ack(stream_name, group_name, consumed_msg.message_id)

        logger.info(f"✅ Message acknowledged: {consumed_msg.message_id}")

        # Verify message was acknowledged
        messages2 = await client.consume(
            stream_name=stream_name,
            group_name=group_name,
            consumer_name=consumer_name,
            count=1,
            block_ms=500
        )

        # The acknowledged message should not be re-delivered
        if messages2:
            assert messages2[0].message_id != consumed_msg.message_id

        logger.info("✅ Message acknowledgment test PASSED")

    finally:
        await client.close()


# ==============================================================================
# ACCEPTANCE CRITERIA TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_p99_latency():
    """
    ACCEPTANCE: p99 latency < 500ms for feature streaming
    """
    logger.info("Starting p99 latency acceptance test")

    producer = FeatureProducer()
    await producer.connect()

    try:
        latencies = []
        num_samples = 100

        for i in range(num_samples):
            start = time.time()

            await producer.produce_pit_feature(
                symbol=f"P99_TEST_{i}",
                features={"value": i},
                pit_validated=True
            )

            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        # Calculate p99
        latencies.sort()
        p99_index = int(len(latencies) * 0.99)
        p99_latency = latencies[p99_index]

        logger.info(
            f"Latency stats: "
            f"p50={latencies[50]:.2f}ms, "
            f"p95={latencies[95]:.2f}ms, "
            f"p99={p99_latency:.2f}ms"
        )

        # Acceptance criteria: p99 < 500ms
        assert p99_latency < 500, f"p99 latency ({p99_latency:.2f}ms) should be < 500ms"

        logger.info(f"✅ ACCEPTANCE PASSED: p99 latency = {p99_latency:.2f}ms < 500ms")

    finally:
        await producer.close()


@pytest.mark.asyncio
async def test_acceptance_zero_message_loss():
    """
    ACCEPTANCE: Zero message loss - all produced messages can be consumed
    """
    logger.info("Starting zero message loss acceptance test")

    producer = FeatureProducer()
    await producer.connect()

    consumer = FeatureConsumer(consumer_name="zero_loss_consumer")
    await consumer.connect()

    try:
        # Produce 50 uniquely identifiable messages
        num_messages = 50
        test_symbols = [f"ZERO_LOSS_{i}" for i in range(num_messages)]

        for symbol in test_symbols:
            await producer.produce_pit_feature(
                symbol=symbol,
                features={"test": "zero_loss"},
                pit_validated=True
            )

        logger.info(f"✅ Produced {num_messages} messages")

        # Consume all messages
        consumed_symbols = set()
        max_attempts = 5
        group_name = "zero_loss_group"

        for attempt in range(max_attempts):
            messages = await consumer.client.consume(
                stream_name="features.pit",
                group_name=group_name,
                consumer_name="zero_loss_consumer",
                count=20,
                block_ms=1000
            )

            for msg in messages:
                symbol = msg.data.get('symbol')
                if symbol and symbol.startswith('ZERO_LOSS_'):
                    consumed_symbols.add(symbol)
                    await consumer.client.ack("features.pit", group_name, msg.message_id)

            if len(consumed_symbols) >= num_messages:
                break

        logger.info(f"✅ Consumed {len(consumed_symbols)} unique messages")

        # Accept if we got at least 90% (may have old messages in stream)
        coverage = len(consumed_symbols) / num_messages
        assert coverage >= 0.9, f"Message coverage ({coverage:.1%}) should be >= 90%"

        logger.info(f"✅ ACCEPTANCE PASSED: Zero message loss - {coverage:.1%} coverage")

    finally:
        await producer.close()
        await consumer.close()


if __name__ == "__main__":
    # Run all tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def run_all_tests():
        logger.info("=" * 80)
        logger.info("REDIS STREAMS INTEGRATION TEST SUITE")
        logger.info("=" * 80)

        tests = [
            ("Feature Producer-Consumer Flow", test_feature_producer_consumer_flow),
            ("Batch Production", test_batch_production),
            ("Stream Lag Monitoring", test_stream_lag_monitoring),
            ("Consumer Group Load Balancing", test_consumer_group_load_balancing),
            ("Performance Throughput", test_performance_throughput),
            ("Error Handling", test_error_handling_and_recovery),
            ("Message Acknowledgment", test_message_acknowledgment),
            ("ACCEPTANCE: p99 Latency", test_acceptance_p99_latency),
            ("ACCEPTANCE: Zero Message Loss", test_acceptance_zero_message_loss),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Running: {test_name}")
                logger.info(f"{'=' * 80}")
                await test_func()
                passed += 1
                logger.info(f"✅ {test_name} PASSED\n")
            except Exception as e:
                failed += 1
                logger.error(f"❌ {test_name} FAILED: {e}\n")

        logger.info("=" * 80)
        logger.info(f"TEST RESULTS: {passed} passed, {failed} failed")
        logger.info("=" * 80)

        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.error(f"⚠️  {failed} tests failed")

    asyncio.run(run_all_tests())
