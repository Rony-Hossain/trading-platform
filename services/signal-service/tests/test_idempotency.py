"""
Tests for Idempotency Manager
"""
import pytest
import json
from datetime import datetime
from unittest.mock import MagicMock
from app.core.idempotency import IdempotencyManager, ActionRecord


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = MagicMock()
    redis.get = MagicMock(return_value=None)
    redis.set = MagicMock(return_value=True)
    redis.setex = MagicMock()
    redis.delete = MagicMock(return_value=1)
    redis.scan = MagicMock(return_value=(0, []))
    return redis


@pytest.fixture
def idempotency_manager(mock_redis):
    """Idempotency manager instance"""
    return IdempotencyManager(mock_redis, ttl_seconds=300)


@pytest.fixture
def sample_action_data():
    """Sample action data"""
    return {
        "action_type": "buy",
        "symbol": "AAPL",
        "shares": 5
    }


def test_action_record_creation():
    """Test ActionRecord model creation"""
    record = ActionRecord(
        idempotency_key="01JC123",
        user_id="user456",
        action_type="buy",
        symbol="AAPL",
        shares=5,
        status="pending",
        created_at=datetime.utcnow().isoformat()
    )

    assert record.idempotency_key == "01JC123"
    assert record.user_id == "user456"
    assert record.action_type == "buy"
    assert record.status == "pending"


def test_check_or_create_new_action(idempotency_manager, mock_redis, sample_action_data):
    """Test creating new action (not duplicate)"""
    mock_redis.get.return_value = None  # No existing record
    mock_redis.set.return_value = True  # Successfully stored

    is_duplicate, record = idempotency_manager.check_or_create(
        idempotency_key="01JC123",
        user_id="user456",
        action_data=sample_action_data
    )

    assert is_duplicate is False
    assert record is not None
    assert record.status == "pending"
    assert record.symbol == "AAPL"
    assert record.shares == 5

    # Verify Redis SET called with NX flag
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args
    assert call_args[1]['nx'] is True  # nx=True for atomic check


def test_check_or_create_duplicate_action(idempotency_manager, mock_redis, sample_action_data):
    """Test detecting duplicate action"""
    # Mock existing record
    existing_record = ActionRecord(
        idempotency_key="01JC123",
        user_id="user456",
        action_type="buy",
        symbol="AAPL",
        shares=5,
        status="executed",
        created_at=datetime.utcnow().isoformat(),
        executed_at=datetime.utcnow().isoformat(),
        result={"filled_price": 185.50}
    )
    mock_redis.get.return_value = existing_record.json()

    is_duplicate, record = idempotency_manager.check_or_create(
        idempotency_key="01JC123",
        user_id="user456",
        action_data=sample_action_data
    )

    assert is_duplicate is True
    assert record is not None
    assert record.status == "executed"
    assert record.result == {"filled_price": 185.50}

    # Should not call SET for duplicate
    mock_redis.set.assert_not_called()


def test_check_or_create_uses_correct_key(idempotency_manager, mock_redis, sample_action_data):
    """Test correct Redis key format"""
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True

    idempotency_manager.check_or_create(
        idempotency_key="01JC123",
        user_id="user456",
        action_data=sample_action_data
    )

    # Check GET called with correct key
    get_call = mock_redis.get.call_args
    assert get_call[0][0] == "action:idem:01JC123"

    # Check SET called with correct key
    set_call = mock_redis.set.call_args
    assert set_call[0][0] == "action:idem:01JC123"


def test_update_status_success(idempotency_manager, mock_redis):
    """Test updating action status to executed"""
    # Mock existing record
    existing_record = ActionRecord(
        idempotency_key="01JC123",
        user_id="user456",
        action_type="buy",
        symbol="AAPL",
        shares=5,
        status="pending",
        created_at=datetime.utcnow().isoformat()
    )
    mock_redis.get.return_value = existing_record.json()

    result = idempotency_manager.update_status(
        idempotency_key="01JC123",
        status="executed",
        result={"filled_price": 185.50, "filled_shares": 5}
    )

    assert result is True
    assert mock_redis.setex.called

    # Check TTL is extended for successful actions
    setex_call = mock_redis.setex.call_args
    ttl = setex_call[0][1]
    assert ttl == 600  # 2x default TTL for executed actions


def test_update_status_failed(idempotency_manager, mock_redis):
    """Test updating action status to failed"""
    existing_record = ActionRecord(
        idempotency_key="01JC123",
        user_id="user456",
        action_type="buy",
        symbol="AAPL",
        shares=5,
        status="pending",
        created_at=datetime.utcnow().isoformat()
    )
    mock_redis.get.return_value = existing_record.json()

    result = idempotency_manager.update_status(
        idempotency_key="01JC123",
        status="failed",
        error_message="Insufficient funds"
    )

    assert result is True
    assert mock_redis.setex.called

    # Check TTL is normal for failed actions
    setex_call = mock_redis.setex.call_args
    ttl = setex_call[0][1]
    assert ttl == 300  # Default TTL


def test_update_status_not_found(idempotency_manager, mock_redis):
    """Test updating non-existent action"""
    mock_redis.get.return_value = None

    result = idempotency_manager.update_status(
        idempotency_key="nonexistent",
        status="executed"
    )

    assert result is False


def test_get_action(idempotency_manager, mock_redis):
    """Test retrieving action by key"""
    existing_record = ActionRecord(
        idempotency_key="01JC123",
        user_id="user456",
        action_type="buy",
        symbol="AAPL",
        shares=5,
        status="executed",
        created_at=datetime.utcnow().isoformat()
    )
    mock_redis.get.return_value = existing_record.json()

    record = idempotency_manager.get_action("01JC123")

    assert record is not None
    assert record.idempotency_key == "01JC123"
    assert record.status == "executed"


def test_get_action_not_found(idempotency_manager, mock_redis):
    """Test retrieving non-existent action"""
    mock_redis.get.return_value = None

    record = idempotency_manager.get_action("nonexistent")

    assert record is None


def test_delete_action(idempotency_manager, mock_redis):
    """Test deleting action"""
    result = idempotency_manager.delete_action("01JC123")

    assert result is True
    assert mock_redis.delete.called

    delete_call = mock_redis.delete.call_args
    assert delete_call[0][0] == "action:idem:01JC123"


def test_ttl_configuration(mock_redis):
    """Test TTL configuration"""
    manager = IdempotencyManager(mock_redis, ttl_seconds=600)

    assert manager.ttl_seconds == 600
    assert manager.success_ttl_seconds == 1200  # 2x


def test_race_condition_handling(idempotency_manager, mock_redis, sample_action_data):
    """Test handling race condition (SET NX fails)"""
    mock_redis.get.return_value = None
    mock_redis.set.return_value = False  # SET NX failed (race condition)

    # Mock second GET returns existing record
    existing_record = ActionRecord(
        idempotency_key="01JC123",
        user_id="user456",
        action_type="buy",
        symbol="AAPL",
        shares=5,
        status="pending",
        created_at=datetime.utcnow().isoformat()
    )

    def get_side_effect(key):
        # First call returns None, second call returns existing
        if mock_redis.get.call_count > 1:
            return existing_record.json()
        return None

    mock_redis.get.side_effect = get_side_effect

    is_duplicate, record = idempotency_manager.check_or_create(
        idempotency_key="01JC123",
        user_id="user456",
        action_data=sample_action_data
    )

    # Should detect as duplicate due to race condition
    assert is_duplicate is True
    assert record is not None


def test_get_user_actions(idempotency_manager, mock_redis):
    """Test retrieving user actions"""
    # Mock scan results
    record1 = ActionRecord(
        idempotency_key="01JC123",
        user_id="user456",
        action_type="buy",
        symbol="AAPL",
        shares=5,
        status="executed",
        created_at="2025-10-02T14:00:00"
    )
    record2 = ActionRecord(
        idempotency_key="01JC124",
        user_id="user456",
        action_type="sell",
        symbol="MSFT",
        shares=3,
        status="executed",
        created_at="2025-10-02T15:00:00"
    )

    mock_redis.scan.return_value = (0, [b"action:idem:01JC123", b"action:idem:01JC124"])
    mock_redis.get.side_effect = [record1.json(), record2.json()]

    actions = idempotency_manager.get_user_actions("user456", limit=10)

    assert len(actions) == 2
    # Should be sorted by created_at (newest first)
    assert actions[0].idempotency_key == "01JC124"
    assert actions[1].idempotency_key == "01JC123"


def test_error_handling_on_check_or_create(idempotency_manager, mock_redis, sample_action_data):
    """Test error handling during check_or_create"""
    mock_redis.get.side_effect = Exception("Redis connection error")

    is_duplicate, record = idempotency_manager.check_or_create(
        idempotency_key="01JC123",
        user_id="user456",
        action_data=sample_action_data
    )

    # On error, should assume not duplicate to allow action
    assert is_duplicate is False
    assert record is None


def test_action_record_serialization():
    """Test ActionRecord JSON serialization"""
    record = ActionRecord(
        idempotency_key="01JC123",
        user_id="user456",
        action_type="buy",
        symbol="AAPL",
        shares=5,
        status="executed",
        created_at=datetime.utcnow().isoformat(),
        result={"filled_price": 185.50}
    )

    json_str = record.json()
    parsed = json.loads(json_str)

    assert parsed["idempotency_key"] == "01JC123"
    assert parsed["status"] == "executed"
    assert parsed["result"]["filled_price"] == 185.50


def test_cleanup_expired_noop(idempotency_manager):
    """Test cleanup_expired is no-op (Redis TTL handles it)"""
    count = idempotency_manager.cleanup_expired()
    assert count == 0


def test_get_stats(idempotency_manager, mock_redis):
    """Test statistics retrieval"""
    # Mock scan with action keys
    record1 = ActionRecord(
        idempotency_key="01JC123",
        user_id="user456",
        action_type="buy",
        symbol="AAPL",
        shares=5,
        status="executed",
        created_at=datetime.utcnow().isoformat()
    )
    record2 = ActionRecord(
        idempotency_key="01JC124",
        user_id="user457",
        action_type="buy",
        symbol="MSFT",
        shares=3,
        status="pending",
        created_at=datetime.utcnow().isoformat()
    )

    mock_redis.scan.return_value = (0, [b"action:idem:01JC123", b"action:idem:01JC124"])
    mock_redis.get.side_effect = [record1.json(), record2.json()]

    stats = idempotency_manager.get_stats()

    assert stats["total_actions"] == 2
    assert stats["executed"] == 1
    assert stats["pending"] == 1
