"""
Contract tests for /api/v1/alerts endpoint
"""
import pytest
from pydantic import ValidationError

from app.core.contracts import Alert, DailyCap


class TestAlertsContract:
    """Test alerts endpoint contract"""

    def test_alert_valid_structure(self):
        """Test valid alert structure"""
        alert_data = {
            "severity": "warning",
            "title": "Daily Trade Limit",
            "message": "You have 1 trade remaining",
            "action_required": False
        }

        alert = Alert(**alert_data)
        assert alert.severity == "warning"
        assert alert.title == "Daily Trade Limit"

    def test_alert_invalid_severity(self):
        """Test alert with invalid severity"""
        alert_data = {
            "severity": "critical",  # Not in ["info", "warning", "error"]
            "title": "Test",
            "message": "Test message",
            "action_required": False
        }

        with pytest.raises(ValidationError):
            Alert(**alert_data)

    def test_daily_cap_valid_structure(self):
        """Test valid daily cap structure"""
        cap_data = {
            "trades_today": 2,
            "trades_remaining": 1,
            "cap_reason": None
        }

        cap = DailyCap(**cap_data)
        assert cap.trades_today == 2
        assert cap.trades_remaining == 1

    def test_daily_cap_with_reason(self):
        """Test daily cap with reason"""
        cap_data = {
            "trades_today": 3,
            "trades_remaining": 0,
            "cap_reason": "Daily trade limit reached"
        }

        cap = DailyCap(**cap_data)
        assert cap.cap_reason == "Daily trade limit reached"

    def test_alert_all_severities(self):
        """Test all valid severity levels"""
        for severity in ["info", "warning", "error"]:
            alert_data = {
                "severity": severity,
                "title": "Test",
                "message": "Test message",
                "action_required": False
            }
            alert = Alert(**alert_data)
            assert alert.severity == severity
