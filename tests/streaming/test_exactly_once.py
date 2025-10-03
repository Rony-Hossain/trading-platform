"""
Tests for Redis Streams exactly-once semantics and reliability
"""
import pytest
import asyncio
import time
from pathlib import Path
import sys

# Add infrastructure to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'infrastructure' / 'streaming'))

from stream_client import RedisStreamClient


@pytest.fixture
async def stream_client():
    """Fixture for stream client"""
    client = RedisStreamClient("redis://localhost:6379/15")  # Use test DB
    await client.connect()
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_produce_consume_message(stream_client):
    """Test basic message production and consumption"""
    stream_name = "test.basic"
    group_name = "test_group"
    consumer_name = "test_consumer"

    # Create consumer group
    await stream_client.create_consumer_group(stream_name, group_name)

    # Produce a message
    test_data = {
        "symbol": "AAPL",
        "price": 185.45,
        "volume": 50000000
    }

    message_id = await stream_client.produce(stream_name, test_data, maxlen=100)
    assert message_id is not None

    # Consume the message
    messages = await stream_client.consume(
        stream_name,
        group_name,
        consumer_name,
        count=1,
        block_ms=1000
    )

    assert len(messages) == 1
    assert messages[0].data["symbol"] == "AAPL"
    assert float(messages[0].data["price"]) == 185.45

    # Acknowledge the message
    await stream_client.ack(stream_name, group_name, messages[0].message_id)


@pytest.mark.asyncio
async def test_exactly_once_semantics(stream_client):
    """Test that messages are delivered exactly once"""
    stream_name = "test.exactly_once"
    group_name = "test_group"
    consumer1_name = "consumer1"
    consumer2_name = "consumer2"

    # Create consumer group
    await stream_client.create_consumer_group(stream_name, group_name)

    # Produce 10 messages
    for i in range(10):
        await stream_client.produce(stream_name, {"id": i}, maxlen=100)

    # Consume with consumer 1
    messages_c1 = await stream_client.consume(
        stream_name,
        group_name,
        consumer1_name,
        count=10,
        block_ms=1000
    )

    # Consume with consumer 2 (should get no messages since consumer1 hasn't acked)
    messages_c2 = await stream_client.consume(
        stream_name,
        group_name,
        consumer2_name,
        count=10,
        block_ms=100
    )

    # Consumer 1 should get all messages, consumer 2 should get none
    assert len(messages_c1) == 10
    assert len(messages_c2) == 0

    # Acknowledge messages from consumer 1
    for msg in messages_c1:
        await stream_client.ack(stream_name, group_name, msg.message_id)


@pytest.mark.asyncio
async def test_message_ordering(stream_client):
    """Test that messages maintain order"""
    stream_name = "test.ordering"
    group_name = "test_group"
    consumer_name = "test_consumer"

    # Create consumer group
    await stream_client.create_consumer_group(stream_name, group_name)

    # Produce messages in order
    expected_order = []
    for i in range(20):
        await stream_client.produce(stream_name, {"sequence": i}, maxlen=100)
        expected_order.append(i)

    # Consume all messages
    all_messages = []
    for _ in range(2):  # Read in two batches
        messages = await stream_client.consume(
            stream_name,
            group_name,
            consumer_name,
            count=10,
            block_ms=1000
        )
        all_messages.extend(messages)

        # Acknowledge
        for msg in messages:
            await stream_client.ack(stream_name, group_name, msg.message_id)

    # Verify order
    actual_order = [int(msg.data["sequence"]) for msg in all_messages]
    assert actual_order == expected_order


@pytest.mark.asyncio
async def test_consumer_group_rebalancing(stream_client):
    """Test consumer group behavior with multiple consumers"""
    stream_name = "test.rebalance"
    group_name = "test_group"

    # Create consumer group
    await stream_client.create_consumer_group(stream_name, group_name)

    # Produce 30 messages
    for i in range(30):
        await stream_client.produce(stream_name, {"id": i}, maxlen=100)

    # Start 3 consumers
    consumers = [f"consumer_{i}" for i in range(3)]
    messages_per_consumer = {}

    for consumer_name in consumers:
        messages = await stream_client.consume(
            stream_name,
            group_name,
            consumer_name,
            count=10,
            block_ms=1000
        )
        messages_per_consumer[consumer_name] = messages

        # Acknowledge
        for msg in messages:
            await stream_client.ack(stream_name, group_name, msg.message_id)

    # Each consumer should have gotten exactly 10 messages
    for consumer_name, messages in messages_per_consumer.items():
        assert len(messages) == 10

    # Total messages consumed should be 30
    total_consumed = sum(len(msgs) for msgs in messages_per_consumer.values())
    assert total_consumed == 30


@pytest.mark.asyncio
async def test_backpressure_handling(stream_client):
    """Test handling of backpressure (stream at maxlen)"""
    stream_name = "test.backpressure"
    maxlen = 100

    # Fill the stream to maxlen
    for i in range(150):  # Produce more than maxlen
        await stream_client.produce(stream_name, {"id": i}, maxlen=maxlen, approximate=False)

    # Check stream info
    stream_info = await stream_client.get_stream_info(stream_name)
    stream_length = stream_info.get('length', 0)

    # Stream should be at or near maxlen
    assert stream_length <= maxlen + 5  # Allow small buffer for exact trimming


@pytest.mark.asyncio
async def test_latency_measurement(stream_client):
    """Test end-to-end latency"""
    stream_name = "test.latency"
    group_name = "test_group"
    consumer_name = "test_consumer"

    # Create consumer group
    await stream_client.create_consumer_group(stream_name, group_name)

    # Produce message with timestamp
    produce_time = time.time()
    await stream_client.produce(stream_name, {"test": "latency"}, maxlen=100)

    # Consume immediately
    messages = await stream_client.consume(
        stream_name,
        group_name,
        consumer_name,
        count=1,
        block_ms=1000
    )

    consume_time = time.time()

    # Calculate latency
    latency_ms = (consume_time - produce_time) * 1000

    # Assert latency is reasonable (< 100ms for local Redis)
    assert latency_ms < 100

    print(f"End-to-end latency: {latency_ms:.2f}ms")

    # Cleanup
    await stream_client.ack(stream_name, group_name, messages[0].message_id)


@pytest.mark.asyncio
async def test_stream_trimming(stream_client):
    """Test approximate vs exact trimming"""
    stream_exact = "test.trim_exact"
    stream_approx = "test.trim_approx"
    maxlen = 50

    # Produce to exact trimming stream
    for i in range(100):
        await stream_client.produce(stream_exact, {"id": i}, maxlen=maxlen, approximate=False)

    # Produce to approximate trimming stream
    for i in range(100):
        await stream_client.produce(stream_approx, {"id": i}, maxlen=maxlen, approximate=True)

    # Check lengths
    exact_info = await stream_client.get_stream_info(stream_exact)
    approx_info = await stream_client.get_stream_info(stream_approx)

    exact_length = exact_info.get('length', 0)
    approx_length = approx_info.get('length', 0)

    # Exact should be at maxlen, approximate might be slightly more
    assert exact_length == maxlen
    assert approx_length >= maxlen


@pytest.mark.asyncio
async def test_error_handling(stream_client):
    """Test error handling for invalid operations"""
    # Test consuming from non-existent group
    with pytest.raises(Exception):
        await stream_client.consume(
            "test.errors",
            "nonexistent_group",
            "consumer",
            count=1,
            block_ms=100
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
