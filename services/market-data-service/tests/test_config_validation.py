"""
Tests for configuration validation and hot-reload safety.

Ensures that invalid configs are rejected before they can break the system.
"""
import importlib
import os
import pytest


def test_policy_validation_rejects_unknown_provider(monkeypatch):
    """Test that unknown providers in policy are rejected."""
    monkeypatch.setenv("POLICY_BARS_1M", '["finnhub", "unknown_provider"]')

    # Reload config module to pick up new env vars
    import app.core.config as cfg
    importlib.reload(cfg)

    # Attempt hot reload with invalid provider
    result = cfg.hot_reload()

    assert result.get("ok") is False
    assert "unknown provider" in result.get("error", "").lower()


def test_breaker_thresholds_invalid_order(monkeypatch):
    """Test that demote > promote threshold is rejected."""
    monkeypatch.setenv("BREAKER_DEMOTE_THRESHOLD", "0.9")
    monkeypatch.setenv("BREAKER_PROMOTE_THRESHOLD", "0.8")

    import app.core.config as cfg
    importlib.reload(cfg)

    result = cfg.hot_reload()

    assert result.get("ok") is False
    assert "breaker threshold" in result.get("error", "").lower()


def test_breaker_thresholds_out_of_range(monkeypatch):
    """Test that thresholds outside [0, 1] are rejected."""
    monkeypatch.setenv("BREAKER_DEMOTE_THRESHOLD", "-0.1")
    monkeypatch.setenv("BREAKER_PROMOTE_THRESHOLD", "1.5")

    import app.core.config as cfg
    importlib.reload(cfg)

    result = cfg.hot_reload()

    assert result.get("ok") is False


def test_valid_config_accepts(monkeypatch):
    """Test that valid config passes validation."""
    monkeypatch.setenv("POLICY_BARS_1M", '["polygon", "finnhub"]')
    monkeypatch.setenv("BREAKER_DEMOTE_THRESHOLD", "0.55")
    monkeypatch.setenv("BREAKER_PROMOTE_THRESHOLD", "0.70")

    import app.core.config as cfg
    importlib.reload(cfg)

    result = cfg.hot_reload()

    assert result.get("ok") is True
    assert "policy_version" in result


def test_empty_policy_rejected(monkeypatch):
    """Test that empty provider policy is rejected."""
    monkeypatch.setenv("POLICY_BARS_1M", "[]")

    import app.core.config as cfg
    importlib.reload(cfg)

    result = cfg.hot_reload()

    assert result.get("ok") is False
    assert "empty" in result.get("error", "").lower() or "no providers" in result.get("error", "").lower()


def test_duplicate_providers_in_policy(monkeypatch):
    """Test that duplicate providers in policy are handled."""
    monkeypatch.setenv("POLICY_BARS_1M", '["polygon", "polygon", "finnhub"]')

    import app.core.config as cfg
    importlib.reload(cfg)

    # This may be allowed (deduplicated) or rejected depending on implementation
    result = cfg.hot_reload()

    # At minimum, should not crash
    assert "ok" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
