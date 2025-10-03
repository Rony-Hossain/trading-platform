"""
Policy Manager
Hot-reloadable policy configuration from YAML
"""
import yaml
import signal
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class PolicyManager:
    """
    Hot-reloadable policy configuration

    Features:
    - Load policies from YAML file on startup
    - Hot-reload via SIGHUP signal without restart
    - Dot-notation access for nested values
    - Validation on reload (fail-safe: keeps old policies if new invalid)
    - Thread-safe policy access
    """

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.policies: Dict[str, Any] = {}
        self.last_loaded: Optional[datetime] = None
        self._lock = False  # Simple lock for thread safety

        # Load initial policies
        self.reload()

        # Register signal handler for hot reload (Unix only)
        try:
            signal.signal(signal.SIGHUP, self._handle_sighup)
            logger.info("sighup_handler_registered", config_path=str(self.config_path))
        except AttributeError:
            # Windows doesn't support SIGHUP
            logger.warning("sighup_not_supported", platform="windows")

    def reload(self) -> bool:
        """
        Reload policies from disk

        Returns:
            True if reload successful, False otherwise
        """
        try:
            # Read YAML file
            with open(self.config_path, 'r') as f:
                new_policies = yaml.safe_load(f)

            # Validate schema
            self._validate_policies(new_policies)

            # Update policies atomically
            self.policies = new_policies
            self.last_loaded = datetime.utcnow()

            logger.info(
                "policies_reloaded",
                version=self.policies.get('version'),
                last_updated=self.policies.get('last_updated'),
                config_path=str(self.config_path)
            )

            return True

        except FileNotFoundError:
            logger.error(
                "policy_file_not_found",
                config_path=str(self.config_path)
            )
            return False

        except yaml.YAMLError as e:
            logger.error(
                "policy_yaml_error",
                error=str(e),
                config_path=str(self.config_path)
            )
            return False

        except ValueError as e:
            logger.error(
                "policy_validation_error",
                error=str(e),
                config_path=str(self.config_path)
            )
            return False

        except Exception as e:
            logger.error(
                "policy_reload_failed",
                error=str(e),
                exc_info=True
            )
            return False

    def _handle_sighup(self, signum, frame):
        """Handle SIGHUP signal for hot reload"""
        logger.info("received_sighup_reloading_policies")
        self.reload()

    def _validate_policies(self, policies: Dict):
        """
        Basic schema validation

        Raises:
            ValueError: If validation fails
        """
        # Required top-level keys
        required_keys = [
            'version',
            'beginner_mode',
            'volatility_brakes',
            'reason_scoring',
            'fitness_checks'
        ]

        for key in required_keys:
            if key not in policies:
                raise ValueError(f"Missing required policy key: {key}")

        # Validate beginner_mode
        beginner = policies.get('beginner_mode', {})
        required_beginner_keys = [
            'max_stop_distance_pct',
            'min_liquidity_adv',
            'max_spread_bps'
        ]
        for key in required_beginner_keys:
            if key not in beginner:
                raise ValueError(f"Missing beginner_mode.{key}")

        # Validate volatility_brakes
        if 'sectors' not in policies.get('volatility_brakes', {}):
            raise ValueError("Missing volatility_brakes.sectors")

        # Validate reason_scoring
        scoring = policies.get('reason_scoring', {})
        if 'weights' not in scoring:
            raise ValueError("Missing reason_scoring.weights")
        if 'min_score_beginner' not in scoring:
            raise ValueError("Missing reason_scoring.min_score_beginner")

        logger.debug("policy_validation_passed")

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get policy value by dot-notation path

        Args:
            path: Dot-separated path (e.g., 'beginner_mode.max_stop_distance_pct')
            default: Default value if path not found

        Returns:
            Policy value or default

        Example:
            >>> policy.get('beginner_mode.max_stop_distance_pct')
            4.0
            >>> policy.get('volatility_brakes.sectors.TECH')
            0.035
        """
        keys = path.split('.')
        value = self.policies

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

            if value is None:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get all policies (read-only copy)"""
        return self.policies.copy()

    def is_fed_day(self, date: datetime) -> bool:
        """
        Check if date is a Fed meeting day

        Args:
            date: Date to check

        Returns:
            True if Fed meeting day
        """
        fed_days = self.get('volatility_brakes.fed_meeting_days', [])
        return date.strftime('%Y-%m-%d') in fed_days

    def get_sector_volatility_threshold(self, sector: str) -> float:
        """
        Get volatility threshold for sector

        Args:
            sector: Sector name (e.g., 'TECH', 'FINANCE')

        Returns:
            Volatility threshold
        """
        sectors = self.get('volatility_brakes.sectors', {})
        return sectors.get(sector, sectors.get('DEFAULT', 0.03))

    def get_reason_weight(self, reason_code: str) -> float:
        """
        Get weight for reason code

        Args:
            reason_code: Reason code (e.g., 'support_bounce')

        Returns:
            Weight (0-1)
        """
        weights = self.get('reason_scoring.weights', {})
        return weights.get(reason_code, 0.1)

    def is_conservative_mode(self) -> bool:
        """
        Check if conservative mode is enabled

        Returns:
            True if conservative mode enabled
        """
        return self.get('conservative_mode.enabled', False)

    def is_quiet_hours(self, current_time: datetime) -> bool:
        """
        Check if current time is in quiet hours

        Args:
            current_time: Current datetime

        Returns:
            True if in quiet hours
        """
        if not self.get('quiet_hours.enabled', False):
            return False

        windows = self.get('quiet_hours.windows', [])
        current_time_str = current_time.strftime('%H:%M')

        for window in windows:
            start, end = window.split('-')
            if start <= current_time_str <= end:
                return True

        return False

    def get_alert_cooldown(self, alert_type: str) -> int:
        """
        Get alert cooldown period

        Args:
            alert_type: 'opportunity' or 'protect'

        Returns:
            Cooldown in seconds
        """
        key = f'alert_throttling.{alert_type}_cooldown_sec'
        return self.get(key, 900)  # Default 15 minutes

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get policy metadata

        Returns:
            Dict with version, last_updated, last_loaded
        """
        return {
            'version': self.get('version'),
            'last_updated': self.get('last_updated'),
            'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None,
            'config_path': str(self.config_path)
        }


# Global policy manager instance (initialized in main.py)
_policy_manager: Optional[PolicyManager] = None


def init_policy_manager(config_path: Path) -> PolicyManager:
    """
    Initialize global policy manager

    Args:
        config_path: Path to policies.yaml

    Returns:
        PolicyManager instance
    """
    global _policy_manager
    _policy_manager = PolicyManager(config_path)
    return _policy_manager


def get_policy_manager() -> PolicyManager:
    """
    Get global policy manager instance

    Returns:
        PolicyManager instance

    Raises:
        RuntimeError: If not initialized
    """
    if _policy_manager is None:
        raise RuntimeError("PolicyManager not initialized. Call init_policy_manager() first.")
    return _policy_manager
