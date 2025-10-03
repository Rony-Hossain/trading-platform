"""
Alert Delivery System
Integrates Signal Service alerts with multi-channel notification delivery
"""
import asyncio
import smtplib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import structlog
import httpx

from ..config import settings
from ..core.contracts import Alert

logger = structlog.get_logger(__name__)


class AlertDeliveryService:
    """
    Multi-channel alert delivery service

    Supports:
    - Email (SMTP)
    - Slack (webhook)
    - Generic webhooks

    Integrates with Signal Service alerts for:
    - Daily trade cap warnings
    - Daily loss limit warnings
    - Position risk alerts
    - Guardrail violations
    - Market condition warnings
    """

    def __init__(self):
        self.enabled = settings.ALERT_DELIVERY_ENABLED
        self.email_enabled = settings.ALERT_EMAIL_ENABLED
        self.slack_enabled = settings.ALERT_SLACK_ENABLED

        logger.info(
            "alert_delivery_initialized",
            enabled=self.enabled,
            email_enabled=self.email_enabled,
            slack_enabled=self.slack_enabled
        )

    async def send_alert(
        self,
        alert: Alert,
        user_id: str,
        channels: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Send alert through configured channels

        Args:
            alert: Alert to send
            user_id: User ID for context
            channels: List of channels to use (email, slack, webhook)
                     If None, uses all enabled channels

        Returns:
            Dict mapping channel -> delivery status
        """
        if not self.enabled:
            logger.debug("alert_delivery_disabled", alert=alert.title)
            return {"status": "disabled"}

        delivery_results = {}

        # Determine which channels to use
        if channels is None:
            channels = []
            if self.email_enabled:
                channels.append("email")
            if self.slack_enabled:
                channels.append("slack")
            if settings.ALERT_WEBHOOK_URLS:
                channels.append("webhook")

        # Send to each channel
        for channel in channels:
            try:
                if channel == "email" and self.email_enabled:
                    result = await self._send_email(alert, user_id)
                    delivery_results["email"] = result

                elif channel == "slack" and self.slack_enabled:
                    result = await self._send_slack(alert, user_id)
                    delivery_results["slack"] = result

                elif channel == "webhook" and settings.ALERT_WEBHOOK_URLS:
                    result = await self._send_webhook(alert, user_id)
                    delivery_results["webhook"] = result

                else:
                    delivery_results[channel] = "not_configured"

            except Exception as e:
                logger.error(
                    "alert_delivery_failed",
                    channel=channel,
                    alert=alert.title,
                    error=str(e),
                    exc_info=True
                )
                delivery_results[channel] = f"error: {str(e)}"

        logger.info(
            "alert_sent",
            alert=alert.title,
            user_id=user_id,
            channels=list(delivery_results.keys())
        )

        return delivery_results

    async def _send_email(self, alert: Alert, user_id: str) -> str:
        """Send alert via email"""
        if not all([
            settings.SMTP_USERNAME,
            settings.SMTP_PASSWORD,
            settings.SMTP_FROM,
            settings.ALERT_EMAIL_TO
        ]):
            return "email_not_configured"

        try:
            msg = MIMEMultipart()
            msg['From'] = settings.SMTP_FROM
            msg['To'] = settings.ALERT_EMAIL_TO
            msg['Subject'] = f"Trading Alert: {alert.title}"

            # Map severity to color
            severity_colors = {
                "info": "#36a64f",      # Green
                "warning": "#ff9500",   # Orange
                "error": "#ff0000"      # Red
            }
            color = severity_colors.get(alert.severity, "#808080")

            # Create HTML email
            html_message = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="border-left: 4px solid {color}; padding-left: 20px; margin: 20px 0;">
                    <h2 style="color: {color}; margin: 0 0 10px 0;">
                        {alert.severity.upper()}: {alert.title}
                    </h2>
                    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        <p style="margin: 0; font-size: 14px; line-height: 1.6;">
                            {alert.message}
                        </p>
                    </div>
                    {f'<p style="color: #d32f2f; font-weight: bold;">⚠️ Action Required</p>' if alert.action_required else ''}
                    <p style="color: #666; font-size: 12px; margin: 15px 0 0 0;">
                        User: {user_id}<br>
                        Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
                    </p>
                </div>
                <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 20px 0;">
                <p style="color: #999; font-size: 11px; text-align: center;">
                    Signal Service Alert Notification
                </p>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_message, 'html'))

            # Send email (run in executor to avoid blocking)
            def send_email():
                with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
                    server.starttls()
                    server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
                    server.send_message(msg)

            await asyncio.get_event_loop().run_in_executor(None, send_email)

            logger.info("email_alert_sent", alert=alert.title, to=settings.ALERT_EMAIL_TO)
            return "sent"

        except Exception as e:
            logger.error("email_send_failed", error=str(e))
            return f"error: {str(e)}"

    async def _send_slack(self, alert: Alert, user_id: str) -> str:
        """Send alert to Slack"""
        if not settings.ALERT_SLACK_WEBHOOK:
            return "slack_not_configured"

        try:
            # Map severity to Slack color
            severity_colors = {
                "info": "#36a64f",      # Green
                "warning": "#ff9500",   # Orange
                "error": "#ff0000"      # Red
            }
            color = severity_colors.get(alert.severity, "#808080")

            # Map severity to emoji
            severity_emojis = {
                "info": ":information_source:",
                "warning": ":warning:",
                "error": ":rotating_light:"
            }
            emoji = severity_emojis.get(alert.severity, ":bell:")

            payload = {
                "username": "Signal Service Alerts",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"{emoji} {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "User",
                                "value": user_id,
                                "short": True
                            },
                            {
                                "title": "Action Required",
                                "value": "Yes" if alert.action_required else "No",
                                "short": True
                            }
                        ],
                        "footer": "Signal Service",
                        "ts": int(datetime.now(timezone.utc).timestamp())
                    }
                ]
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    settings.ALERT_SLACK_WEBHOOK,
                    json=payload,
                    timeout=5.0
                )
                response.raise_for_status()

            logger.info("slack_alert_sent", alert=alert.title)
            return "sent"

        except Exception as e:
            logger.error("slack_send_failed", error=str(e))
            return f"error: {str(e)}"

    async def _send_webhook(self, alert: Alert, user_id: str) -> str:
        """Send alert to generic webhooks"""
        if not settings.ALERT_WEBHOOK_URLS:
            return "webhook_not_configured"

        try:
            webhook_urls = settings.ALERT_WEBHOOK_URLS.split(",")
            results = []

            payload = {
                "service": "signal-service",
                "alert_type": "trading_alert",
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "action_required": alert.action_required,
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            async with httpx.AsyncClient() as client:
                for url in webhook_urls:
                    url = url.strip()
                    if not url:
                        continue

                    try:
                        response = await client.post(
                            url,
                            json=payload,
                            timeout=5.0
                        )
                        response.raise_for_status()
                        results.append("sent")
                    except Exception as e:
                        logger.error("webhook_send_failed", url=url, error=str(e))
                        results.append(f"error: {str(e)}")

            logger.info("webhook_alerts_sent", alert=alert.title, count=len(results))
            return f"sent_to_{len(results)}_webhooks"

        except Exception as e:
            logger.error("webhook_send_failed", error=str(e))
            return f"error: {str(e)}"

    async def send_batch_alerts(
        self,
        alerts: List[Alert],
        user_id: str,
        channels: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Send multiple alerts

        Args:
            alerts: List of alerts to send
            user_id: User ID
            channels: Channels to use

        Returns:
            Dict mapping alert title -> delivery results
        """
        results = {}

        for alert in alerts:
            # Only send actionable or high-severity alerts
            if alert.action_required or alert.severity in ["warning", "error"]:
                delivery_result = await self.send_alert(alert, user_id, channels)
                results[alert.title] = delivery_result

        return results


# Singleton instance
_alert_delivery_service: Optional[AlertDeliveryService] = None


def get_alert_delivery_service() -> AlertDeliveryService:
    """Get or create alert delivery service singleton"""
    global _alert_delivery_service
    if _alert_delivery_service is None:
        _alert_delivery_service = AlertDeliveryService()
    return _alert_delivery_service
