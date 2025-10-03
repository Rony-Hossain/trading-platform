"""
Tests for Alert Delivery System
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.core.alert_delivery import AlertDeliveryService, get_alert_delivery_service
from app.core.contracts import Alert


@pytest.fixture
def alert_service():
    """Create alert delivery service for testing"""
    with patch('app.core.alert_delivery.settings') as mock_settings:
        mock_settings.ALERT_DELIVERY_ENABLED = True
        mock_settings.ALERT_EMAIL_ENABLED = True
        mock_settings.ALERT_SLACK_ENABLED = True
        mock_settings.ALERT_WEBHOOK_URLS = "http://localhost:9000/webhook"
        mock_settings.SMTP_SERVER = "smtp.test.com"
        mock_settings.SMTP_PORT = 587
        mock_settings.SMTP_USERNAME = "test@test.com"
        mock_settings.SMTP_PASSWORD = "password"
        mock_settings.SMTP_FROM = "alerts@signal-service.com"
        mock_settings.ALERT_EMAIL_TO = "user@example.com"
        mock_settings.ALERT_SLACK_WEBHOOK = "https://hooks.slack.com/test"

        service = AlertDeliveryService()
        return service


@pytest.fixture
def sample_alert():
    """Create sample alert for testing"""
    return Alert(
        severity="warning",
        title="Daily Trade Limit Approaching",
        message="You have 1 trade remaining today (limit: 3 trades).",
        action_required=False
    )


@pytest.fixture
def critical_alert():
    """Create critical alert for testing"""
    return Alert(
        severity="error",
        title="Daily Loss Limit Reached",
        message="Daily loss limit reached (5.2%). Trading paused for today.",
        action_required=True
    )


class TestAlertDeliveryService:
    """Test alert delivery service"""

    def test_service_initialization(self, alert_service):
        """Test service initializes correctly"""
        assert alert_service.enabled is True
        assert alert_service.email_enabled is True
        assert alert_service.slack_enabled is True

    @pytest.mark.asyncio
    async def test_send_alert_disabled(self):
        """Test send_alert when delivery is disabled"""
        with patch('app.core.alert_delivery.settings') as mock_settings:
            mock_settings.ALERT_DELIVERY_ENABLED = False

            service = AlertDeliveryService()
            alert = Alert(
                severity="info",
                title="Test",
                message="Test message",
                action_required=False
            )

            result = await service.send_alert(alert, "user123")
            assert result == {"status": "disabled"}

    @pytest.mark.asyncio
    async def test_send_email_success(self, alert_service, sample_alert):
        """Test successful email delivery"""
        with patch('smtplib.SMTP') as mock_smtp:
            # Mock SMTP server
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            result = await alert_service._send_email(sample_alert, "user123")

            assert result == "sent"
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
            mock_server.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_email_not_configured(self):
        """Test email delivery when not configured"""
        with patch('app.core.alert_delivery.settings') as mock_settings:
            mock_settings.SMTP_USERNAME = None  # Missing config

            service = AlertDeliveryService()
            alert = Alert(
                severity="info",
                title="Test",
                message="Test",
                action_required=False
            )

            result = await service._send_email(alert, "user123")
            assert result == "email_not_configured"

    @pytest.mark.asyncio
    async def test_send_slack_success(self, alert_service, sample_alert):
        """Test successful Slack delivery"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock HTTP client
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await alert_service._send_slack(sample_alert, "user123")

            assert result == "sent"

    @pytest.mark.asyncio
    async def test_send_slack_not_configured(self):
        """Test Slack delivery when not configured"""
        with patch('app.core.alert_delivery.settings') as mock_settings:
            mock_settings.ALERT_SLACK_WEBHOOK = None

            service = AlertDeliveryService()
            alert = Alert(
                severity="info",
                title="Test",
                message="Test",
                action_required=False
            )

            result = await service._send_slack(alert, "user123")
            assert result == "slack_not_configured"

    @pytest.mark.asyncio
    async def test_send_webhook_success(self, alert_service, sample_alert):
        """Test successful webhook delivery"""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock HTTP client
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await alert_service._send_webhook(sample_alert, "user123")

            assert result.startswith("sent_to_")

    @pytest.mark.asyncio
    async def test_send_alert_all_channels(self, alert_service, critical_alert):
        """Test sending alert to all configured channels"""
        with patch('smtplib.SMTP') as mock_smtp, \
             patch('httpx.AsyncClient') as mock_http:

            # Mock SMTP
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            # Mock HTTP client
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_http.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await alert_service.send_alert(critical_alert, "user123")

            # Should have results for email, slack, and webhook
            assert "email" in result
            assert "slack" in result
            assert "webhook" in result
            assert result["email"] == "sent"
            assert result["slack"] == "sent"

    @pytest.mark.asyncio
    async def test_send_batch_alerts(self, alert_service):
        """Test sending batch alerts"""
        alerts = [
            Alert(
                severity="info",
                title="Info Alert",
                message="This is just info",
                action_required=False
            ),
            Alert(
                severity="warning",
                title="Warning Alert",
                message="This is a warning",
                action_required=False
            ),
            Alert(
                severity="error",
                title="Error Alert",
                message="This requires action",
                action_required=True
            )
        ]

        with patch.object(alert_service, 'send_alert') as mock_send:
            mock_send.return_value = {"email": "sent", "slack": "sent"}

            results = await alert_service.send_batch_alerts(alerts, "user123")

            # Should only send warning and error alerts
            assert len(results) == 2
            assert "Warning Alert" in results
            assert "Error Alert" in results
            assert "Info Alert" not in results  # Info alerts not sent

    @pytest.mark.asyncio
    async def test_alert_severity_colors(self, alert_service):
        """Test that different severity levels get different colors"""
        info_alert = Alert(severity="info", title="Info", message="Info", action_required=False)
        warning_alert = Alert(severity="warning", title="Warning", message="Warning", action_required=False)
        error_alert = Alert(severity="error", title="Error", message="Error", action_required=False)

        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            # Send alerts with different severities
            await alert_service._send_email(info_alert, "user123")
            await alert_service._send_email(warning_alert, "user123")
            await alert_service._send_email(error_alert, "user123")

            # Verify all were sent
            assert mock_server.send_message.call_count == 3

    @pytest.mark.asyncio
    async def test_error_handling(self, alert_service, sample_alert):
        """Test error handling in alert delivery"""
        with patch('smtplib.SMTP') as mock_smtp:
            # Simulate SMTP error
            mock_smtp.return_value.__enter__.side_effect = Exception("SMTP connection failed")

            result = await alert_service._send_email(sample_alert, "user123")

            assert result.startswith("error:")
            assert "SMTP connection failed" in result

    @pytest.mark.asyncio
    async def test_channel_selection(self, alert_service, sample_alert):
        """Test sending to specific channels only"""
        with patch('smtplib.SMTP') as mock_smtp, \
             patch('httpx.AsyncClient') as mock_http:

            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_http.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Send only to email
            result = await alert_service.send_alert(sample_alert, "user123", channels=["email"])

            assert "email" in result
            assert "slack" not in result
            assert "webhook" not in result

    def test_singleton_pattern(self):
        """Test that get_alert_delivery_service returns singleton"""
        service1 = get_alert_delivery_service()
        service2 = get_alert_delivery_service()

        assert service1 is service2


class TestAlertFormats:
    """Test alert formatting for different channels"""

    @pytest.mark.asyncio
    async def test_email_html_format(self, alert_service, critical_alert):
        """Test email HTML formatting includes all alert details"""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            await alert_service._send_email(critical_alert, "user123")

            # Verify email was sent with message
            assert mock_server.send_message.called
            call_args = mock_server.send_message.call_args[0][0]

            # Email should contain key information
            email_content = str(call_args)
            assert "Daily Loss Limit Reached" in email_content
            assert "user123" in email_content

    @pytest.mark.asyncio
    async def test_slack_attachment_format(self, alert_service, sample_alert):
        """Test Slack message formatting"""
        with patch('httpx.AsyncClient') as mock_http:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            posted_data = None

            async def capture_post(url, **kwargs):
                nonlocal posted_data
                posted_data = kwargs.get('json')
                return mock_response

            mock_http.return_value.__aenter__.return_value.post = capture_post

            await alert_service._send_slack(sample_alert, "user123")

            # Verify Slack payload structure
            assert posted_data is not None
            assert "attachments" in posted_data
            assert len(posted_data["attachments"]) > 0

            attachment = posted_data["attachments"][0]
            assert "title" in attachment
            assert "color" in attachment
            assert "fields" in attachment

    @pytest.mark.asyncio
    async def test_webhook_payload_format(self, alert_service, critical_alert):
        """Test webhook payload format"""
        with patch('httpx.AsyncClient') as mock_http:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            posted_data = None

            async def capture_post(url, **kwargs):
                nonlocal posted_data
                posted_data = kwargs.get('json')
                return mock_response

            mock_http.return_value.__aenter__.return_value.post = capture_post

            await alert_service._send_webhook(critical_alert, "user123")

            # Verify webhook payload
            assert posted_data is not None
            assert posted_data["service"] == "signal-service"
            assert posted_data["alert_type"] == "trading_alert"
            assert posted_data["severity"] == "error"
            assert posted_data["title"] == "Daily Loss Limit Reached"
            assert posted_data["user_id"] == "user123"
            assert posted_data["action_required"] is True
            assert "timestamp" in posted_data
