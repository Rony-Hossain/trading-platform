"""
Tests for Policy Manager
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from datetime import datetime
from app.core.policy_manager import PolicyManager


@pytest.fixture
def sample_policies():
    """Sample policies dict"""
    return {
        'version': '1.0',
        'last_updated': '2025-10-02T10:00:00Z',
        'beginner_mode': {
            'max_stop_distance_pct': 4.0,
            'min_liquidity_adv': 500000,
            'max_spread_bps': 20
        },
        'volatility_brakes': {
            'sectors': {
                'TECH': 0.035,
                'FINANCE': 0.025,
                'DEFAULT': 0.030
            },
            'fed_meeting_days': ['2025-10-15', '2025-11-05']
        },
        'reason_scoring': {
            'weights': {
                'support_bounce': 0.40,
                'buyer_pressure': 0.30
            },
            'min_score_beginner': 0.50,
            'min_score_expert': 0.30
        },
        'fitness_checks': {
            'min_adv_shares': 100000,
            'max_spread_bps': 30
        },
        'quiet_hours': {
            'enabled': True,
            'windows': ['22:00-07:00', '16:30-17:00']
        },
        'conservative_mode': {
            'enabled': False
        }
    }


@pytest.fixture
def policy_file(sample_policies):
    """Create temporary policy file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_policies, f)
        return Path(f.name)


def test_policy_manager_init(policy_file):
    """Test PolicyManager initialization"""
    manager = PolicyManager(policy_file)
    assert manager.policies is not None
    assert manager.last_loaded is not None
    assert manager.get('version') == '1.0'


def test_policy_get_simple(policy_file):
    """Test simple policy get"""
    manager = PolicyManager(policy_file)
    assert manager.get('version') == '1.0'
    assert manager.get('beginner_mode.max_stop_distance_pct') == 4.0


def test_policy_get_nested(policy_file):
    """Test nested policy get"""
    manager = PolicyManager(policy_file)
    assert manager.get('volatility_brakes.sectors.TECH') == 0.035
    assert manager.get('reason_scoring.weights.support_bounce') == 0.40


def test_policy_get_default(policy_file):
    """Test get with default value"""
    manager = PolicyManager(policy_file)
    assert manager.get('nonexistent.key', 'default') == 'default'
    assert manager.get('another.missing.key', 42) == 42


def test_is_fed_day(policy_file):
    """Test Fed day check"""
    manager = PolicyManager(policy_file)
    fed_day = datetime(2025, 10, 15)
    normal_day = datetime(2025, 10, 16)

    assert manager.is_fed_day(fed_day) is True
    assert manager.is_fed_day(normal_day) is False


def test_get_sector_volatility_threshold(policy_file):
    """Test sector volatility threshold"""
    manager = PolicyManager(policy_file)
    assert manager.get_sector_volatility_threshold('TECH') == 0.035
    assert manager.get_sector_volatility_threshold('FINANCE') == 0.025
    assert manager.get_sector_volatility_threshold('UNKNOWN') == 0.030  # DEFAULT


def test_get_reason_weight(policy_file):
    """Test reason weight lookup"""
    manager = PolicyManager(policy_file)
    assert manager.get_reason_weight('support_bounce') == 0.40
    assert manager.get_reason_weight('buyer_pressure') == 0.30
    assert manager.get_reason_weight('unknown_reason') == 0.1  # default


def test_is_conservative_mode(policy_file):
    """Test conservative mode check"""
    manager = PolicyManager(policy_file)
    assert manager.is_conservative_mode() is False


def test_is_quiet_hours(policy_file):
    """Test quiet hours check"""
    manager = PolicyManager(policy_file)

    # Night time (22:00-07:00)
    night_time = datetime(2025, 10, 2, 23, 0)  # 23:00
    assert manager.is_quiet_hours(night_time) is True

    # Day time
    day_time = datetime(2025, 10, 2, 14, 0)  # 14:00
    assert manager.is_quiet_hours(day_time) is False

    # Market close (16:30-17:00)
    close_time = datetime(2025, 10, 2, 16, 45)  # 16:45
    assert manager.is_quiet_hours(close_time) is True


def test_reload_policies(policy_file, sample_policies):
    """Test policy reload"""
    manager = PolicyManager(policy_file)
    original_version = manager.get('version')

    # Modify policy file
    sample_policies['version'] = '2.0'
    with open(policy_file, 'w') as f:
        yaml.dump(sample_policies, f)

    # Reload
    assert manager.reload() is True
    assert manager.get('version') == '2.0'


def test_reload_invalid_yaml(policy_file):
    """Test reload with invalid YAML"""
    manager = PolicyManager(policy_file)
    original_version = manager.get('version')

    # Write invalid YAML
    with open(policy_file, 'w') as f:
        f.write('invalid: yaml: content: [')

    # Reload should fail but keep old policies
    assert manager.reload() is False
    assert manager.get('version') == original_version  # Old version preserved


def test_validation_missing_required_key(policy_file, sample_policies):
    """Test validation catches missing required keys"""
    manager = PolicyManager(policy_file)

    # Remove required key
    del sample_policies['beginner_mode']
    with open(policy_file, 'w') as f:
        yaml.dump(sample_policies, f)

    # Reload should fail
    assert manager.reload() is False


def test_get_metadata(policy_file):
    """Test metadata retrieval"""
    manager = PolicyManager(policy_file)
    metadata = manager.get_metadata()

    assert metadata['version'] == '1.0'
    assert metadata['last_updated'] is not None
    assert metadata['last_loaded'] is not None
    assert 'config_path' in metadata


def test_get_all_policies(policy_file):
    """Test getting all policies"""
    manager = PolicyManager(policy_file)
    all_policies = manager.get_all()

    assert isinstance(all_policies, dict)
    assert 'version' in all_policies
    assert 'beginner_mode' in all_policies
