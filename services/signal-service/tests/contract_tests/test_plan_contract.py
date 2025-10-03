"""
Contract tests for /api/v1/plan endpoint
Validates response structure matches contract
"""
import pytest
from pydantic import ValidationError

from app.core.contracts import PlanResponse, Pick, ReasonCode


class TestPlanContract:
    """Test plan endpoint contract"""

    def test_plan_response_valid_structure(self):
        """Test valid plan response structure"""
        response_data = {
            "request_id": "req_abc123",
            "picks": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "shares": 10,
                    "confidence": "high",
                    "reason": "Strong technical and fundamental signals",
                    "reason_codes": ["HIGH_CONFIDENCE_PREDICTION", "UPTREND_CONFIRMED"],
                    "decision_path": "inference:v1.2.3",
                    "limit_price": 175.50,
                    "stop_loss_price": 170.00,
                    "metadata": {"sector": "TECH"}
                }
            ],
            "daily_cap_reached": False,
            "degraded_fields": [],
            "metadata": {
                "model_version": "v1.2.3",
                "generated_at": "2024-01-20T15:30:00Z"
            }
        }

        # Should not raise validation error
        plan = PlanResponse(**response_data)
        assert plan.request_id == "req_abc123"
        assert len(plan.picks) == 1
        assert plan.picks[0].symbol == "AAPL"

    def test_plan_response_missing_required_field(self):
        """Test plan response with missing required field"""
        response_data = {
            # Missing request_id
            "picks": [],
            "daily_cap_reached": False,
            "degraded_fields": [],
            "metadata": {}
        }

        with pytest.raises(ValidationError):
            PlanResponse(**response_data)

    def test_pick_invalid_action(self):
        """Test pick with invalid action"""
        pick_data = {
            "symbol": "AAPL",
            "action": "INVALID",  # Not in ["BUY", "SELL", "HOLD", "AVOID"]
            "shares": 10,
            "confidence": "high",
            "reason": "Test",
            "reason_codes": [],
            "decision_path": "test"
        }

        with pytest.raises(ValidationError):
            Pick(**pick_data)

    def test_pick_invalid_confidence(self):
        """Test pick with invalid confidence"""
        pick_data = {
            "symbol": "AAPL",
            "action": "BUY",
            "shares": 10,
            "confidence": "very_high",  # Not in ["low", "medium", "high"]
            "reason": "Test",
            "reason_codes": [],
            "decision_path": "test"
        }

        with pytest.raises(ValidationError):
            Pick(**pick_data)

    def test_pick_negative_shares(self):
        """Test pick with negative shares"""
        pick_data = {
            "symbol": "AAPL",
            "action": "BUY",
            "shares": -10,  # Must be positive
            "confidence": "high",
            "reason": "Test",
            "reason_codes": [],
            "decision_path": "test"
        }

        with pytest.raises(ValidationError):
            Pick(**pick_data)

    def test_degraded_fields_list(self):
        """Test degraded_fields is list of strings"""
        response_data = {
            "request_id": "req_abc123",
            "picks": [],
            "daily_cap_reached": False,
            "degraded_fields": ["inference", "sentiment"],
            "metadata": {}
        }

        plan = PlanResponse(**response_data)
        assert "inference" in plan.degraded_fields
        assert "sentiment" in plan.degraded_fields

    def test_metadata_optional_fields(self):
        """Test metadata with optional fields"""
        response_data = {
            "request_id": "req_abc123",
            "picks": [],
            "daily_cap_reached": False,
            "degraded_fields": [],
            "metadata": {
                "model_version": "v1.2.3",
                "generated_at": "2024-01-20T15:30:00Z",
                "custom_field": "value"  # Additional fields allowed
            }
        }

        plan = PlanResponse(**response_data)
        assert plan.metadata["model_version"] == "v1.2.3"

    def test_empty_picks_list(self):
        """Test plan response with empty picks"""
        response_data = {
            "request_id": "req_abc123",
            "picks": [],
            "daily_cap_reached": True,
            "degraded_fields": [],
            "metadata": {}
        }

        plan = PlanResponse(**response_data)
        assert len(plan.picks) == 0
        assert plan.daily_cap_reached is True

    def test_multiple_picks(self):
        """Test plan response with multiple picks"""
        response_data = {
            "request_id": "req_abc123",
            "picks": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "shares": 10,
                    "confidence": "high",
                    "reason": "Test 1",
                    "reason_codes": [],
                    "decision_path": "test"
                },
                {
                    "symbol": "MSFT",
                    "action": "SELL",
                    "shares": 5,
                    "confidence": "medium",
                    "reason": "Test 2",
                    "reason_codes": [],
                    "decision_path": "test"
                }
            ],
            "daily_cap_reached": False,
            "degraded_fields": [],
            "metadata": {}
        }

        plan = PlanResponse(**response_data)
        assert len(plan.picks) == 2
        assert plan.picks[0].symbol == "AAPL"
        assert plan.picks[1].symbol == "MSFT"
