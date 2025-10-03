"""
Tests for Decision Store
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from app.core.decision_store import DecisionStore, DecisionSnapshot
from app.core.contracts import (
    Pick, ReasonCode, Constraints, LimitsApplied, Compliance,
    ResponseMetadata, SourceModel
)


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = MagicMock()
    redis.setex = MagicMock()
    redis.lpush = MagicMock()
    redis.ltrim = MagicMock()
    redis.expire = MagicMock()
    redis.get = MagicMock(return_value=None)
    redis.lrange = MagicMock(return_value=[])
    redis.delete = MagicMock(return_value=1)
    return redis


@pytest.fixture
def decision_store(mock_redis):
    """Decision store instance"""
    return DecisionStore(mock_redis, ttl_days=30)


@pytest.fixture
def sample_pick():
    """Sample pick for testing"""
    return Pick(
        symbol="AAPL",
        action="BUY",
        shares=5,
        entry_hint=185.50,
        safety_line=182.00,
        target=192.00,
        confidence="medium",
        reason="Price holding above recent floor and buyers are active.",
        reason_codes=[ReasonCode.SUPPORT_BOUNCE, ReasonCode.BUYER_PRESSURE],
        max_risk_usd=17.50,
        budget_impact={"cash_left": 982.50},
        constraints=Constraints(
            stop_loss=182.00,
            max_position_value_usd=927.50,
            min_holding_period_sec=300
        ),
        limits_applied=LimitsApplied(),
        compliance=Compliance(),
        decision_path="ALPHA>THRESH>RISK_OK>SENTIMENT_OK",
        reason_score=0.75
    )


@pytest.fixture
def sample_metadata():
    """Sample response metadata"""
    return ResponseMetadata(
        request_id="01JC123",
        generated_at=datetime.utcnow(),
        version="plan.v1",
        latency_ms=142.5,
        source_models=[
            SourceModel(
                name="lgbm-alpha",
                version="1.12.3",
                sha="abc123",
                confidence=0.85
            )
        ]
    )


def test_decision_snapshot_creation():
    """Test DecisionSnapshot creation"""
    snapshot = DecisionSnapshot(
        request_id="01JC123",
        user_id="user456",
        inputs={"watchlist": ["AAPL"], "mode": "beginner"},
        picks=[],
        metadata={"version": "plan.v1"},
        degraded_fields=["sentiment"]
    )

    assert snapshot.request_id == "01JC123"
    assert snapshot.user_id == "user456"
    assert snapshot.snapshot_version == "1.0"
    assert len(snapshot.snapshot_hash) == 16  # SHA-256 truncated


def test_decision_snapshot_to_dict():
    """Test snapshot serialization"""
    snapshot = DecisionSnapshot(
        request_id="01JC123",
        user_id="user456",
        inputs={"watchlist": ["AAPL"]},
        picks=[],
        metadata={}
    )

    snapshot_dict = snapshot.to_dict()

    assert "request_id" in snapshot_dict
    assert "user_id" in snapshot_dict
    assert "timestamp" in snapshot_dict
    assert "snapshot_hash" in snapshot_dict
    assert snapshot_dict["snapshot_version"] == "1.0"


def test_decision_snapshot_from_dict():
    """Test snapshot deserialization"""
    data = {
        "request_id": "01JC123",
        "user_id": "user456",
        "timestamp": "2025-10-02T14:00:00",
        "inputs": {"watchlist": ["AAPL"]},
        "picks": [],
        "metadata": {},
        "degraded_fields": [],
        "snapshot_version": "1.0",
        "snapshot_hash": "abc123def456"
    }

    snapshot = DecisionSnapshot.from_dict(data)

    assert snapshot.request_id == "01JC123"
    assert snapshot.user_id == "user456"
    assert snapshot.snapshot_hash == "abc123def456"


def test_save_snapshot(decision_store, mock_redis, sample_pick, sample_metadata):
    """Test saving decision snapshot"""
    snapshot_hash = decision_store.save_snapshot(
        request_id="01JC123",
        user_id="user456",
        inputs={"watchlist": ["AAPL"], "mode": "beginner"},
        picks=[sample_pick],
        metadata=sample_metadata,
        degraded_fields=["sentiment"]
    )

    # Verify Redis calls
    assert mock_redis.setex.called
    assert mock_redis.lpush.called
    assert mock_redis.ltrim.called
    assert mock_redis.expire.called

    # Verify snapshot hash
    assert len(snapshot_hash) == 16


def test_save_snapshot_stores_with_correct_key(decision_store, mock_redis, sample_pick, sample_metadata):
    """Test snapshot stored with correct Redis key"""
    decision_store.save_snapshot(
        request_id="01JC123",
        user_id="user456",
        inputs={},
        picks=[sample_pick],
        metadata=sample_metadata
    )

    # Check setex was called with correct key
    call_args = mock_redis.setex.call_args
    key = call_args[0][0]
    assert key == "decision:snapshot:01JC123"


def test_save_snapshot_updates_user_index(decision_store, mock_redis, sample_pick, sample_metadata):
    """Test user index updated on save"""
    decision_store.save_snapshot(
        request_id="01JC123",
        user_id="user456",
        inputs={},
        picks=[sample_pick],
        metadata=sample_metadata
    )

    # Verify user index calls
    assert mock_redis.lpush.called
    call_args = mock_redis.lpush.call_args
    user_key = call_args[0][0]
    request_id = call_args[0][1]

    assert user_key == "decision:user:user456:recent"
    assert request_id == "01JC123"

    # Verify trim to 100
    assert mock_redis.ltrim.called
    trim_args = mock_redis.ltrim.call_args
    assert trim_args[0][1] == 0
    assert trim_args[0][2] == 99


def test_get_snapshot(decision_store, mock_redis):
    """Test retrieving snapshot"""
    import json

    # Mock Redis response
    snapshot_data = {
        "request_id": "01JC123",
        "user_id": "user456",
        "timestamp": "2025-10-02T14:00:00",
        "inputs": {"watchlist": ["AAPL"]},
        "picks": [],
        "metadata": {},
        "degraded_fields": [],
        "snapshot_version": "1.0",
        "snapshot_hash": "abc123"
    }
    mock_redis.get.return_value = json.dumps(snapshot_data)

    snapshot = decision_store.get_snapshot("01JC123")

    assert snapshot is not None
    assert snapshot.request_id == "01JC123"
    assert snapshot.user_id == "user456"
    assert mock_redis.get.called


def test_get_snapshot_not_found(decision_store, mock_redis):
    """Test retrieving non-existent snapshot"""
    mock_redis.get.return_value = None

    snapshot = decision_store.get_snapshot("nonexistent")

    assert snapshot is None


def test_get_user_decisions(decision_store, mock_redis):
    """Test retrieving user decisions"""
    import json

    # Mock user index
    mock_redis.lrange.return_value = [b"01JC123", b"01JC124"]

    # Mock snapshots
    snapshot_data = {
        "request_id": "01JC123",
        "user_id": "user456",
        "timestamp": "2025-10-02T14:00:00",
        "inputs": {},
        "picks": [],
        "metadata": {},
        "degraded_fields": [],
        "snapshot_version": "1.0",
        "snapshot_hash": "abc123"
    }
    mock_redis.get.return_value = json.dumps(snapshot_data)

    decisions = decision_store.get_user_decisions("user456", limit=20)

    # Should have retrieved 2 decisions
    assert len(decisions) == 2
    assert mock_redis.lrange.called


def test_get_user_decisions_no_history(decision_store, mock_redis):
    """Test retrieving decisions for user with no history"""
    mock_redis.lrange.return_value = []

    decisions = decision_store.get_user_decisions("newuser", limit=20)

    assert len(decisions) == 0


def test_get_user_decisions_limit_capped(decision_store, mock_redis):
    """Test user decisions limit capped at 100"""
    decision_store.get_user_decisions("user456", limit=200)

    # Check lrange was called with max 99 (0-99 = 100 items)
    call_args = mock_redis.lrange.call_args
    assert call_args[0][2] == 99  # limit - 1


def test_verify_integrity():
    """Test snapshot integrity verification"""
    snapshot = DecisionSnapshot(
        request_id="01JC123",
        user_id="user456",
        inputs={"watchlist": ["AAPL"]},
        picks=[],
        metadata={}
    )

    store = DecisionStore(MagicMock(), ttl_days=30)

    # Integrity should pass
    assert store.verify_integrity(snapshot) is True


def test_verify_integrity_tampered():
    """Test integrity verification fails on tampering"""
    snapshot = DecisionSnapshot(
        request_id="01JC123",
        user_id="user456",
        inputs={"watchlist": ["AAPL"]},
        picks=[],
        metadata={}
    )

    # Tamper with data
    snapshot.inputs["watchlist"].append("MSFT")

    store = DecisionStore(MagicMock(), ttl_days=30)

    # Integrity should fail
    assert store.verify_integrity(snapshot) is False


def test_delete_snapshot(decision_store, mock_redis):
    """Test snapshot deletion"""
    result = decision_store.delete_snapshot("01JC123")

    assert result is True
    assert mock_redis.delete.called
    call_args = mock_redis.delete.call_args
    assert call_args[0][0] == "decision:snapshot:01JC123"


def test_ttl_configuration(mock_redis):
    """Test TTL configuration"""
    store = DecisionStore(mock_redis, ttl_days=7)

    assert store.ttl_days == 7
    assert store.ttl_seconds == 7 * 24 * 60 * 60


def test_serialize_pick(decision_store, sample_pick):
    """Test pick serialization"""
    serialized = decision_store._serialize_pick(sample_pick)

    assert serialized["symbol"] == "AAPL"
    assert serialized["action"] == "BUY"
    assert serialized["shares"] == 5
    assert serialized["confidence"] == "medium"
    assert "reason_codes" in serialized
    assert isinstance(serialized["reason_codes"], list)
