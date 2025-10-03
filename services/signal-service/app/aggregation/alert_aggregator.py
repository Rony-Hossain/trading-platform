"""
Alert Aggregator
Generates alerts for daily caps, risk warnings, and important notifications
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from ..core.contracts import Alert, DailyCap
from ..core.policy_manager import get_policy_manager
from ..core.alert_delivery import get_alert_delivery_service
from ..upstream.portfolio_client import PortfolioClient

logger = structlog.get_logger(__name__)


class AlertAggregator:
    """
    Aggregates alerts from various sources

    Alert types:
    - Daily trade cap warnings
    - Daily loss limit warnings
    - Position concentration warnings
    - Volatility warnings
    - Stop loss triggers
    - Profit target hits
    """

    def __init__(self, portfolio_client: PortfolioClient):
        self.portfolio_client = portfolio_client
        self.policy_manager = get_policy_manager()
        self.alert_delivery = get_alert_delivery_service()
        logger.info("alert_aggregator_initialized")

    async def get_alerts(
        self,
        user_id: str,
        mode: str = "beginner"
    ) -> tuple[List[Alert], DailyCap]:
        """
        Get all alerts for user

        Args:
            user_id: User ID
            mode: "beginner" or "expert"

        Returns:
            (alerts, daily_cap) - List of alerts and daily cap info
        """
        is_beginner = mode == "beginner"
        alerts = []

        try:
            # Fetch portfolio data
            portfolio_data = await self.portfolio_client.get_portfolio(user_id)
            positions_data = await self.portfolio_client.get_positions(user_id)

            # Generate alerts
            alerts.extend(self._check_daily_caps(portfolio_data, is_beginner))
            alerts.extend(self._check_position_alerts(positions_data, is_beginner))
            alerts.extend(self._check_portfolio_risk(portfolio_data, is_beginner))
            alerts.extend(self._check_market_conditions(is_beginner))

            # Build daily cap info
            daily_cap = self._build_daily_cap(portfolio_data, is_beginner)

            logger.info(
                "alerts_generated",
                user_id=user_id,
                alert_count=len(alerts),
                daily_trades=daily_cap.trades_today
            )

            # Send high-priority alerts via configured channels
            await self._deliver_alerts(alerts, user_id)

            return alerts, daily_cap

        except Exception as e:
            logger.error(
                "alert_generation_failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            # Return empty alerts on failure
            return [], self._build_empty_daily_cap(is_beginner)

    def _check_daily_caps(
        self,
        portfolio_data: Dict[str, Any],
        is_beginner: bool
    ) -> List[Alert]:
        """Check daily trading caps"""
        alerts = []

        if not is_beginner:
            return alerts

        # Trade count cap
        daily_trades = portfolio_data.get("daily_trades", 0)
        max_trades = self.policy_manager.get("beginner_mode.max_daily_trades", 3)

        if daily_trades >= max_trades:
            alerts.append(Alert(
                severity="error",
                title="Daily Trade Limit Reached",
                message=f"You've reached your daily limit of {max_trades} trades. Trading will resume tomorrow.",
                action_required=False
            ))
        elif daily_trades == max_trades - 1:
            alerts.append(Alert(
                severity="warning",
                title="Approaching Daily Trade Limit",
                message=f"You have 1 trade remaining today (limit: {max_trades} trades).",
                action_required=False
            ))

        # Daily loss limit
        daily_pnl_pct = portfolio_data.get("daily_pnl_pct", 0)
        max_loss_pct = self.policy_manager.get("beginner_mode.max_daily_loss_pct", 5.0)

        if daily_pnl_pct < -max_loss_pct:
            alerts.append(Alert(
                severity="error",
                title="Daily Loss Limit Reached",
                message=f"Daily loss limit reached ({abs(daily_pnl_pct):.1f}%). Trading paused for today to protect your portfolio.",
                action_required=False
            ))
        elif daily_pnl_pct < -(max_loss_pct * 0.75):  # 75% of limit
            alerts.append(Alert(
                severity="warning",
                title="Approaching Daily Loss Limit",
                message=f"You're approaching the daily loss limit ({abs(daily_pnl_pct):.1f}% of {max_loss_pct}%). Consider pausing trading.",
                action_required=False
            ))

        return alerts

    def _check_position_alerts(
        self,
        positions_data: Dict[str, Any],
        is_beginner: bool
    ) -> List[Alert]:
        """Check position-specific alerts"""
        alerts = []

        positions = positions_data.get("positions", [])

        for position in positions:
            symbol = position["symbol"]
            unrealized_pnl_pct = position.get("unrealized_pnl_pct", 0)

            # Large gain alert
            if unrealized_pnl_pct > 20:
                alerts.append(Alert(
                    severity="info",
                    title=f"{symbol}: Large Unrealized Gain",
                    message=f"{symbol} is up {unrealized_pnl_pct:.1f}%. Consider taking some profit.",
                    action_required=False
                ))

            # Large loss alert (approaching stop loss)
            if unrealized_pnl_pct < -8:
                alerts.append(Alert(
                    severity="warning",
                    title=f"{symbol}: Significant Loss",
                    message=f"{symbol} is down {abs(unrealized_pnl_pct):.1f}%. Review your position.",
                    action_required=True
                ))

        # Concentration risk
        concentration_risk = positions_data.get("concentration_risk", "low")
        if concentration_risk == "high" and is_beginner:
            alerts.append(Alert(
                severity="warning",
                title="Portfolio Concentration Risk",
                message="Your portfolio is heavily concentrated in a few positions. Consider diversifying.",
                action_required=False
            ))

        return alerts

    def _check_portfolio_risk(
        self,
        portfolio_data: Dict[str, Any],
        is_beginner: bool
    ) -> List[Alert]:
        """Check portfolio-level risk alerts"""
        alerts = []

        # Low buying power
        buying_power = portfolio_data.get("buying_power", 0)
        total_value = portfolio_data.get("total_value", 1)
        buying_power_pct = (buying_power / total_value) * 100

        if buying_power_pct < 10 and is_beginner:
            alerts.append(Alert(
                severity="info",
                title="Low Buying Power",
                message=f"You have {buying_power_pct:.0f}% buying power remaining. You may not be able to make new purchases.",
                action_required=False
            ))

        return alerts

    def _check_market_conditions(self, is_beginner: bool) -> List[Alert]:
        """Check market-wide conditions"""
        alerts = []

        # Check if it's a Fed day
        if self.policy_manager.is_fed_day():
            alerts.append(Alert(
                severity="warning",
                title="Federal Reserve Announcement Today",
                message="The Fed is making an announcement today, which may cause market volatility. Trade with caution.",
                action_required=False
            ))

        # Check quiet hours
        if is_beginner and self.policy_manager.is_quiet_hours():
            alerts.append(Alert(
                severity="info",
                title="High Volatility Period",
                message="Market open and close hours tend to have higher volatility. Consider waiting 30 minutes.",
                action_required=False
            ))

        return alerts

    def _build_daily_cap(
        self,
        portfolio_data: Dict[str, Any],
        is_beginner: bool
    ) -> DailyCap:
        """Build daily cap information"""
        if not is_beginner:
            return DailyCap(
                trades_today=portfolio_data.get("daily_trades", 0),
                trades_remaining=999,  # No limit for experts
                cap_reason=None
            )

        max_trades = self.policy_manager.get("beginner_mode.max_daily_trades", 3)
        trades_today = portfolio_data.get("daily_trades", 0)
        trades_remaining = max(0, max_trades - trades_today)

        cap_reason = None
        if trades_remaining == 0:
            cap_reason = "Daily trade limit reached"

        # Check loss limit
        daily_pnl_pct = portfolio_data.get("daily_pnl_pct", 0)
        max_loss_pct = self.policy_manager.get("beginner_mode.max_daily_loss_pct", 5.0)

        if daily_pnl_pct < -max_loss_pct:
            trades_remaining = 0
            cap_reason = "Daily loss limit reached"

        return DailyCap(
            trades_today=trades_today,
            trades_remaining=trades_remaining,
            cap_reason=cap_reason
        )

    def _build_empty_daily_cap(self, is_beginner: bool) -> DailyCap:
        """Build empty daily cap for error cases"""
        max_trades = self.policy_manager.get("beginner_mode.max_daily_trades", 3) if is_beginner else 999
        return DailyCap(
            trades_today=0,
            trades_remaining=max_trades,
            cap_reason=None
        )

    async def _deliver_alerts(self, alerts: List[Alert], user_id: str) -> None:
        """
        Deliver high-priority alerts via configured channels

        Only sends alerts that are:
        - action_required = True, OR
        - severity = "warning" or "error"

        Args:
            alerts: List of alerts
            user_id: User ID
        """
        if not alerts:
            return

        # Filter to high-priority alerts
        high_priority = [
            alert for alert in alerts
            if alert.action_required or alert.severity in ["warning", "error"]
        ]

        if not high_priority:
            logger.debug("no_high_priority_alerts", total=len(alerts))
            return

        try:
            # Send alerts asynchronously
            delivery_results = await self.alert_delivery.send_batch_alerts(
                high_priority,
                user_id
            )

            logger.info(
                "alerts_delivered",
                user_id=user_id,
                alert_count=len(high_priority),
                results=delivery_results
            )

        except Exception as e:
            logger.error(
                "alert_delivery_failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
