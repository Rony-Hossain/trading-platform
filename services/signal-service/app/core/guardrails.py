"""
Guardrail Engine
Enforces beginner safety rules and trading constraints
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
import structlog

from .contracts import Pick, ReasonCode
from .policy_manager import get_policy_manager
from ..upstream.halt_client import get_halt_client

logger = structlog.get_logger(__name__)


class GuardrailViolation:
    """Represents a guardrail violation"""
    def __init__(self, code: str, message: str, severity: str = "warning"):
        self.code = code
        self.message = message
        self.severity = severity  # "warning" or "blocking"

    def to_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity
        }


class GuardrailEngine:
    """
    Enforces safety guardrails for beginner traders

    Checks:
    - Stop loss requirements
    - Position size limits
    - Volatility brakes
    - Liquidity requirements
    - Sector concentration
    - Time-based restrictions (quiet hours, Fed days)
    - Daily trade caps
    """

    def __init__(self, enable_halt_detection: bool = True):
        self.policy_manager = get_policy_manager()
        self.enable_halt_detection = enable_halt_detection
        self.halt_client = get_halt_client() if enable_halt_detection else None
        logger.info("guardrail_engine_initialized", halt_detection=enable_halt_detection)

    def check_pick(
        self,
        pick: Pick,
        user_context: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Tuple[bool, List[GuardrailViolation]]:
        """
        Check if pick passes all guardrails

        Args:
            pick: Trading pick to validate
            user_context: User portfolio and settings
            market_context: Current market conditions

        Returns:
            (is_allowed, violations) - True if allowed, list of violations
        """
        violations = []
        is_beginner = user_context.get("beginner_mode", False)

        # Check halt status (async check run synchronously)
        if self.enable_halt_detection:
            halt_violations = asyncio.run(self._check_halt_status(pick))
            violations.extend(halt_violations)

        # Run all guardrail checks
        violations.extend(self._check_stop_loss(pick, is_beginner))
        violations.extend(self._check_position_size(pick, user_context, is_beginner))
        violations.extend(self._check_volatility(pick, market_context, is_beginner))
        violations.extend(self._check_liquidity(pick, market_context, is_beginner))
        violations.extend(self._check_sector_concentration(pick, user_context, is_beginner))
        violations.extend(self._check_time_restrictions(market_context, is_beginner))
        violations.extend(self._check_daily_caps(user_context, is_beginner))

        # Determine if pick is allowed
        blocking_violations = [v for v in violations if v.severity == "blocking"]
        is_allowed = len(blocking_violations) == 0

        if not is_allowed:
            logger.warning(
                "guardrail_check_failed",
                symbol=pick.symbol,
                action=pick.action,
                blocking_count=len(blocking_violations),
                warning_count=len([v for v in violations if v.severity == "warning"])
            )
        elif violations:
            logger.info(
                "guardrail_check_passed_with_warnings",
                symbol=pick.symbol,
                action=pick.action,
                warning_count=len(violations)
            )

        return is_allowed, violations

    def _check_stop_loss(self, pick: Pick, is_beginner: bool) -> List[GuardrailViolation]:
        """Check stop loss requirements"""
        violations = []

        if is_beginner and pick.action == "BUY":
            # Beginners must have stop loss
            if not pick.stop_loss_price:
                violations.append(GuardrailViolation(
                    code="STOP_LOSS_REQUIRED",
                    message="A stop loss is required for all buy orders in beginner mode",
                    severity="blocking"
                ))
            else:
                # Check stop loss distance
                max_distance_pct = self.policy_manager.get("beginner_mode.max_stop_distance_pct", 4.0)
                stop_distance_pct = abs(pick.stop_loss_price - pick.limit_price) / pick.limit_price * 100

                if stop_distance_pct > max_distance_pct:
                    violations.append(GuardrailViolation(
                        code="STOP_LOSS_TOO_WIDE",
                        message=f"Stop loss is too far ({stop_distance_pct:.1f}%). Maximum allowed: {max_distance_pct}%",
                        severity="blocking"
                    ))

        return violations

    def _check_position_size(
        self,
        pick: Pick,
        user_context: Dict[str, Any],
        is_beginner: bool
    ) -> List[GuardrailViolation]:
        """Check position size limits"""
        violations = []

        portfolio_value = user_context.get("total_value", 0)
        if portfolio_value == 0:
            return violations

        position_value = pick.shares * (pick.limit_price or 0)
        position_pct = position_value / portfolio_value * 100

        if is_beginner:
            max_position_pct = self.policy_manager.get("beginner_mode.max_position_size_pct", 10.0)
            if position_pct > max_position_pct:
                violations.append(GuardrailViolation(
                    code="POSITION_SIZE_EXCEEDED",
                    message=f"Position size ({position_pct:.1f}%) exceeds beginner limit of {max_position_pct}%",
                    severity="blocking"
                ))

        # Check minimum trade size
        if is_beginner:
            min_trade_value = self.policy_manager.get("beginner_mode.min_trade_value", 100)
            if position_value < min_trade_value:
                violations.append(GuardrailViolation(
                    code="POSITION_TOO_SMALL",
                    message=f"Trade value (${position_value:.0f}) is below minimum ${min_trade_value}",
                    severity="warning"
                ))

        return violations

    def _check_volatility(
        self,
        pick: Pick,
        market_context: Dict[str, Any],
        is_beginner: bool
    ) -> List[GuardrailViolation]:
        """Check volatility brakes"""
        violations = []

        if not is_beginner:
            return violations

        # Get symbol volatility
        symbol_data = market_context.get("symbols", {}).get(pick.symbol, {})
        volatility = symbol_data.get("volatility", 0)
        sector = symbol_data.get("sector", "UNKNOWN")

        # Check sector-specific volatility threshold
        volatility_thresholds = self.policy_manager.get("volatility_brakes.sectors", {})
        max_volatility = volatility_thresholds.get(sector, 0.05)  # Default 5%

        if volatility > max_volatility:
            violations.append(GuardrailViolation(
                code="HIGH_VOLATILITY",
                message=f"Stock volatility ({volatility*100:.1f}%) exceeds beginner threshold for {sector} sector ({max_volatility*100:.1f}%)",
                severity="blocking"
            ))

        return violations

    def _check_liquidity(
        self,
        pick: Pick,
        market_context: Dict[str, Any],
        is_beginner: bool
    ) -> List[GuardrailViolation]:
        """Check liquidity requirements"""
        violations = []

        if not is_beginner:
            return violations

        # Get symbol liquidity
        symbol_data = market_context.get("symbols", {}).get(pick.symbol, {})
        avg_daily_volume = symbol_data.get("avg_daily_volume", 0)

        # Minimum liquidity requirement
        min_liquidity = self.policy_manager.get("beginner_mode.min_liquidity_adv", 500000)

        if avg_daily_volume < min_liquidity:
            violations.append(GuardrailViolation(
                code="LOW_LIQUIDITY",
                message=f"Stock has low trading volume ({avg_daily_volume:,} shares). Minimum required: {min_liquidity:,}",
                severity="blocking"
            ))

        return violations

    def _check_sector_concentration(
        self,
        pick: Pick,
        user_context: Dict[str, Any],
        is_beginner: bool
    ) -> List[GuardrailViolation]:
        """Check sector concentration limits"""
        violations = []

        if not is_beginner or pick.action != "BUY":
            return violations

        # Get current sector allocation
        sector_allocations = user_context.get("sector_allocations", {})
        symbol_sector = pick.metadata.get("sector", "UNKNOWN") if pick.metadata else "UNKNOWN"
        current_sector_pct = sector_allocations.get(symbol_sector, 0)

        # Calculate new sector allocation
        portfolio_value = user_context.get("total_value", 0)
        if portfolio_value > 0:
            position_value = pick.shares * (pick.limit_price or 0)
            new_sector_pct = (current_sector_pct * portfolio_value + position_value) / portfolio_value

            max_sector_pct = self.policy_manager.get("beginner_mode.max_sector_concentration_pct", 30.0)
            if new_sector_pct > max_sector_pct:
                violations.append(GuardrailViolation(
                    code="SECTOR_CONCENTRATION",
                    message=f"This would increase {symbol_sector} allocation to {new_sector_pct:.1f}% (limit: {max_sector_pct}%)",
                    severity="warning"
                ))

        return violations

    def _check_time_restrictions(
        self,
        market_context: Dict[str, Any],
        is_beginner: bool
    ) -> List[GuardrailViolation]:
        """Check time-based restrictions"""
        violations = []

        if not is_beginner:
            return violations

        # Check quiet hours
        is_quiet_hour = self.policy_manager.is_quiet_hours()
        if is_quiet_hour:
            violations.append(GuardrailViolation(
                code="QUIET_HOURS",
                message="Trading during market open/close hours has higher volatility. Consider waiting 30 minutes",
                severity="warning"
            ))

        # Check Fed announcement days
        is_fed_day = self.policy_manager.is_fed_day(market_context.get("date"))
        if is_fed_day:
            violations.append(GuardrailViolation(
                code="FED_DAY",
                message="Federal Reserve announcement today may cause market volatility",
                severity="warning"
            ))

        return violations

    def _check_daily_caps(
        self,
        user_context: Dict[str, Any],
        is_beginner: bool
    ) -> List[GuardrailViolation]:
        """Check daily trading caps"""
        violations = []

        if not is_beginner:
            return violations

        # Check daily trade count
        daily_trades = user_context.get("daily_trades", 0)
        max_daily_trades = self.policy_manager.get("beginner_mode.max_daily_trades", 3)

        if daily_trades >= max_daily_trades:
            violations.append(GuardrailViolation(
                code="DAILY_TRADE_CAP",
                message=f"You've reached your daily trade limit ({max_daily_trades} trades)",
                severity="blocking"
            ))

        # Check daily loss limit
        daily_pnl_pct = user_context.get("daily_pnl_pct", 0)
        max_daily_loss_pct = self.policy_manager.get("beginner_mode.max_daily_loss_pct", 5.0)

        if daily_pnl_pct < -max_daily_loss_pct:
            violations.append(GuardrailViolation(
                code="DAILY_LOSS_LIMIT",
                message=f"Daily loss limit reached ({abs(daily_pnl_pct):.1f}% loss). Trading paused for today",
                severity="blocking"
            ))

        return violations

    def check_action_allowed(
        self,
        action: str,
        symbol: str,
        shares: int,
        user_context: Dict[str, Any]
    ) -> Tuple[bool, List[GuardrailViolation]]:
        """
        Quick check if action is allowed (for action execution endpoint)

        Args:
            action: "BUY" or "SELL"
            symbol: Ticker symbol
            shares: Number of shares
            user_context: User context

        Returns:
            (is_allowed, violations)
        """
        violations = []
        is_beginner = user_context.get("beginner_mode", False)

        # Check daily caps
        violations.extend(self._check_daily_caps(user_context, is_beginner))

        # Check if selling shares user doesn't own
        if action == "SELL":
            positions = user_context.get("positions", [])
            owned_shares = next(
                (p["shares"] for p in positions if p["symbol"] == symbol),
                0
            )
            if shares > owned_shares:
                violations.append(GuardrailViolation(
                    code="INSUFFICIENT_SHARES",
                    message=f"Cannot sell {shares} shares. You only own {owned_shares} shares of {symbol}",
                    severity="blocking"
                ))

        blocking_violations = [v for v in violations if v.severity == "blocking"]
        return len(blocking_violations) == 0, violations

    async def _check_halt_status(self, pick: Pick) -> List[GuardrailViolation]:
        """
        Check if symbol is currently halted

        Blocks picks for symbols that are:
        - LULD halted
        - Volatility halted
        - News pending halt
        - Circuit breaker active
        """
        violations = []

        if not self.halt_client:
            return violations

        try:
            # Check halt status for the symbol
            halt_status = await self.halt_client.check_halt_status(pick.symbol)

            if halt_status.get("is_halted"):
                halt_type = halt_status.get("halt_type", "UNKNOWN")
                message = halt_status.get("message", f"{pick.symbol} is currently halted")

                violations.append(GuardrailViolation(
                    code="SYMBOL_HALTED",
                    message=f"Cannot trade {pick.symbol}: {message} ({halt_type})",
                    severity="blocking"
                ))

                logger.warning(
                    "halt_detected_blocking_pick",
                    symbol=pick.symbol,
                    halt_type=halt_type
                )

            # Also check market-wide circuit breaker
            circuit_breaker = await self.halt_client.check_circuit_breaker()

            if circuit_breaker.get("circuit_breaker_active"):
                cb_level = circuit_breaker.get("circuit_breaker_level", "UNKNOWN")

                violations.append(GuardrailViolation(
                    code="CIRCUIT_BREAKER_ACTIVE",
                    message=f"Market-wide circuit breaker active ({cb_level}). Trading restricted.",
                    severity="blocking"
                ))

                logger.warning(
                    "circuit_breaker_blocking_pick",
                    circuit_breaker_level=cb_level
                )

        except Exception as e:
            logger.error(
                "halt_check_failed",
                symbol=pick.symbol,
                error=str(e)
            )
            # Don't block on error - fail open for halt detection

        return violations


# Global guardrail engine instance
_guardrail_engine: Optional[GuardrailEngine] = None


def init_guardrail_engine() -> GuardrailEngine:
    """Initialize global guardrail engine"""
    global _guardrail_engine
    _guardrail_engine = GuardrailEngine()
    logger.info("guardrail_engine_initialized")
    return _guardrail_engine


def get_guardrail_engine() -> GuardrailEngine:
    """Get global guardrail engine instance"""
    if _guardrail_engine is None:
        raise RuntimeError("GuardrailEngine not initialized. Call init_guardrail_engine() first.")
    return _guardrail_engine
