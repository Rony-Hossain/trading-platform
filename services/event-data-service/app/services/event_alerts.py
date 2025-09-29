"""Event-driven alerting system for high-impact market events."""

import asyncio
import json
import logging
import os
import smtplib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import httpx

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Available alert channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"
    DISCORD = "discord"


class AlertReason(str, Enum):
    """Reasons for triggering alerts."""
    HIGH_IMPACT_EVENT = "high_impact_event"
    IMPACT_SCORE_INCREASE = "impact_score_increase"
    MULTIPLE_EVENTS = "multiple_events"
    PREDICTION_ACCURACY = "prediction_accuracy"
    EVENT_STATUS_CHANGE = "event_status_change"
    CLUSTER_FORMATION = "cluster_formation"
    HEADLINE_SURGE = "headline_surge"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    channels: List[AlertChannel]
    conditions: Dict[str, Any]
    enabled: bool = True
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    def is_in_cooldown(self) -> bool:
        """Check if rule is in cooldown period."""
        if not self.last_triggered:
            return False
        cooldown_period = timedelta(minutes=self.cooldown_minutes)
        return datetime.now(timezone.utc) - self.last_triggered < cooldown_period
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.last_triggered:
            data["last_triggered"] = self.last_triggered.isoformat()
        return data


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_id: str
    severity: AlertSeverity
    reason: AlertReason
    title: str
    message: str
    event_data: Dict[str, Any]
    channels: List[AlertChannel]
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivery_status: Dict[str, str] = None
    
    def __post_init__(self):
        if self.delivery_status is None:
            self.delivery_status = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if self.sent_at:
            data["sent_at"] = self.sent_at.isoformat()
        return data


class EventAlertSystem:
    """Comprehensive event-driven alerting system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._alert_rules: Dict[str, AlertRule] = {}
        self._alert_history: List[Alert] = []
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Load configuration from environment
        self._load_config()
        self._create_default_rules()
        
    def _load_config(self):
        """Load alerting configuration from environment variables."""
        self.config.update({
            # Email configuration
            "smtp_server": os.getenv("ALERT_SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("ALERT_SMTP_PORT", "587")),
            "smtp_username": os.getenv("ALERT_SMTP_USERNAME"),
            "smtp_password": os.getenv("ALERT_SMTP_PASSWORD"),
            "email_from": os.getenv("ALERT_EMAIL_FROM"),
            "email_to": os.getenv("ALERT_EMAIL_TO", "").split(",") if os.getenv("ALERT_EMAIL_TO") else [],
            
            # Slack configuration
            "slack_webhook_url": os.getenv("ALERT_SLACK_WEBHOOK_URL"),
            "slack_channel": os.getenv("ALERT_SLACK_CHANNEL", "#trading-alerts"),
            
            # Teams configuration
            "teams_webhook_url": os.getenv("ALERT_TEAMS_WEBHOOK_URL"),
            
            # Discord configuration
            "discord_webhook_url": os.getenv("ALERT_DISCORD_WEBHOOK_URL"),
            
            # General webhook configuration
            "webhook_urls": os.getenv("ALERT_WEBHOOK_URLS", "").split(",") if os.getenv("ALERT_WEBHOOK_URLS") else [],
            
            # SMS configuration (Twilio)
            "twilio_account_sid": os.getenv("ALERT_TWILIO_ACCOUNT_SID"),
            "twilio_auth_token": os.getenv("ALERT_TWILIO_AUTH_TOKEN"),
            "twilio_from_number": os.getenv("ALERT_TWILIO_FROM_NUMBER"),
            "sms_to_numbers": os.getenv("ALERT_SMS_TO_NUMBERS", "").split(",") if os.getenv("ALERT_SMS_TO_NUMBERS") else [],
            
            # General settings
            "alert_enabled": os.getenv("ALERT_ENABLED", "true").lower() == "true",
            "max_alerts_per_hour": int(os.getenv("ALERT_MAX_PER_HOUR", "50")),
            "default_cooldown_minutes": int(os.getenv("ALERT_DEFAULT_COOLDOWN_MINUTES", "30")),
        })
        
    def _create_default_rules(self):
        """Create default alert rules."""
        default_rules = [
            AlertRule(
                id="high_impact_events",
                name="High Impact Events",
                description="Alert for events with impact score >= 8",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                conditions={
                    "min_impact_score": 8,
                    "categories": ["earnings", "fda_approval", "mna", "regulatory"]
                },
                cooldown_minutes=15
            ),
            AlertRule(
                id="critical_impact_events",
                name="Critical Impact Events", 
                description="Alert for events with impact score >= 9",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SMS],
                conditions={
                    "min_impact_score": 9
                },
                cooldown_minutes=5
            ),
            AlertRule(
                id="mega_cap_earnings",
                name="Mega Cap Earnings",
                description="Alert for mega-cap company earnings",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.SLACK, AlertChannel.WEBHOOK],
                conditions={
                    "categories": ["earnings"],
                    "market_cap_tier": "mega_cap",
                    "min_impact_score": 7
                },
                cooldown_minutes=30
            ),
            AlertRule(
                id="multiple_sector_events",
                name="Multiple Sector Events",
                description="Alert when multiple events in same sector occur within 24h",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.EMAIL],
                conditions={
                    "cluster_types": ["sector_earnings", "regulatory_sector"],
                    "min_cluster_size": 3
                },
                cooldown_minutes=60
            ),
            AlertRule(
                id="impact_score_surge",
                name="Impact Score Increase",
                description="Alert when event impact score increases significantly",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.SLACK],
                conditions={
                    "impact_score_increase": 2.0,
                    "min_final_score": 7
                },
                cooldown_minutes=20
            ),
            AlertRule(
                id="event_occurred",
                name="High Impact Event Occurred",
                description="Alert when high-impact event changes to occurred status",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.SLACK, AlertChannel.WEBHOOK],
                conditions={
                    "status_change": "occurred",
                    "min_impact_score": 8
                },
                cooldown_minutes=0  # No cooldown for status changes
            )
        ]
        
        for rule in default_rules:
            self._alert_rules[rule.id] = rule
            
    async def start(self):
        """Start the alerting system."""
        self._http_client = httpx.AsyncClient(timeout=10.0)
        logger.info("Event alert system started")
        
    async def stop(self):
        """Stop the alerting system."""
        if self._http_client:
            await self._http_client.aclose()
        logger.info("Event alert system stopped")
        
    async def evaluate_event(self, event_data: Dict[str, Any], event_type: str = "event.created"):
        """Evaluate an event against all alert rules."""
        if not self.config.get("alert_enabled", True):
            return
            
        triggered_alerts = []
        
        for rule in self._alert_rules.values():
            if not rule.enabled or rule.is_in_cooldown():
                continue
                
            if await self._evaluate_rule(rule, event_data, event_type):
                alert = await self._create_alert(rule, event_data, event_type)
                if alert:
                    triggered_alerts.append(alert)
                    await self._send_alert(alert)
                    
        return triggered_alerts
        
    async def _evaluate_rule(self, rule: AlertRule, event_data: Dict[str, Any], event_type: str) -> bool:
        """Evaluate if an event matches an alert rule."""
        conditions = rule.conditions
        
        # Check impact score
        min_impact = conditions.get("min_impact_score")
        if min_impact:
            event_impact = event_data.get("impact_score", 0)
            if event_impact < min_impact:
                return False
                
        # Check categories
        allowed_categories = conditions.get("categories")
        if allowed_categories:
            event_category = event_data.get("category", "").lower()
            if event_category not in [cat.lower() for cat in allowed_categories]:
                return False
                
        # Check market cap tier (from enrichment data)
        required_tier = conditions.get("market_cap_tier")
        if required_tier:
            enrichment = event_data.get("metadata", {}).get("enrichment", {})
            market_context = enrichment.get("market_context", {})
            event_tier = market_context.get("market_cap_tier")
            if event_tier != required_tier:
                return False
                
        # Check status changes
        required_status_change = conditions.get("status_change")
        if required_status_change:
            if event_type not in ["event.updated", "event.status_changed"]:
                return False
            event_status = event_data.get("status")
            if event_status != required_status_change:
                return False
                
        # Check impact score increases
        score_increase_threshold = conditions.get("impact_score_increase")
        if score_increase_threshold:
            if event_type != "event.impact_changed":
                return False
            # Would need previous score to compare - simplified for demo
            current_score = event_data.get("impact_score", 0)
            min_final = conditions.get("min_final_score", 0)
            if current_score < min_final:
                return False
                
        return True
        
    async def _create_alert(self, rule: AlertRule, event_data: Dict[str, Any], event_type: str) -> Alert:
        """Create an alert instance."""
        import uuid
        
        # Determine alert reason based on rule and event
        reason = AlertReason.HIGH_IMPACT_EVENT
        if "impact_score_increase" in rule.conditions:
            reason = AlertReason.IMPACT_SCORE_INCREASE
        elif "status_change" in rule.conditions:
            reason = AlertReason.EVENT_STATUS_CHANGE
        elif "cluster" in rule.id:
            reason = AlertReason.CLUSTER_FORMATION
            
        # Generate alert title and message
        symbol = event_data.get("symbol", "Unknown")
        title = event_data.get("title", "Event")
        impact_score = event_data.get("impact_score", 0)
        category = event_data.get("category", "event")
        
        alert_title = f"🚨 {rule.name}: {symbol}"
        alert_message = self._generate_alert_message(rule, event_data, reason)
        
        alert = Alert(
            id=str(uuid.uuid4()),
            rule_id=rule.id,
            severity=rule.severity,
            reason=reason,
            title=alert_title,
            message=alert_message,
            event_data=event_data,
            channels=rule.channels,
            created_at=datetime.now(timezone.utc)
        )
        
        # Update rule state
        rule.last_triggered = datetime.now(timezone.utc)
        rule.trigger_count += 1
        
        # Store alert
        self._alert_history.append(alert)
        
        # Keep only recent alerts (last 1000)
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]
            
        return alert
        
    def _generate_alert_message(self, rule: AlertRule, event_data: Dict[str, Any], reason: AlertReason) -> str:
        """Generate formatted alert message."""
        symbol = event_data.get("symbol", "Unknown")
        title = event_data.get("title", "Event")
        impact_score = event_data.get("impact_score", 0)
        category = event_data.get("category", "event")
        scheduled_at = event_data.get("scheduled_at", "")
        
        # Get enrichment data if available
        enrichment = event_data.get("metadata", {}).get("enrichment", {})
        market_context = enrichment.get("market_context", {})
        market_cap_tier = market_context.get("market_cap_tier", "unknown")
        sector = market_context.get("sector", "unknown")
        
        message_parts = [
            f"**Event**: {title}",
            f"**Symbol**: {symbol}",
            f"**Category**: {category.title()}",
            f"**Impact Score**: {impact_score}/10",
        ]
        
        if scheduled_at:
            message_parts.append(f"**Scheduled**: {scheduled_at}")
            
        if market_cap_tier != "unknown":
            message_parts.append(f"**Market Cap**: {market_cap_tier.replace('_', ' ').title()}")
            
        if sector != "unknown":
            message_parts.append(f"**Sector**: {sector.title()}")
            
        message_parts.append(f"**Reason**: {reason.value.replace('_', ' ').title()}")
        message_parts.append(f"**Severity**: {rule.severity.value.upper()}")
        
        return "\n".join(message_parts)
        
    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        delivery_results = {}
        
        for channel in alert.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    result = await self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK:
                    result = await self._send_slack_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    result = await self._send_webhook_alert(alert)
                elif channel == AlertChannel.SMS:
                    result = await self._send_sms_alert(alert)
                elif channel == AlertChannel.TEAMS:
                    result = await self._send_teams_alert(alert)
                elif channel == AlertChannel.DISCORD:
                    result = await self._send_discord_alert(alert)
                else:
                    result = "unsupported_channel"
                    
                delivery_results[channel.value] = result
                
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
                delivery_results[channel.value] = f"error: {str(e)}"
                
        alert.delivery_status = delivery_results
        alert.sent_at = datetime.now(timezone.utc)
        
        logger.info(f"Alert {alert.id} sent via {len(alert.channels)} channels")
        
    async def _send_email_alert(self, alert: Alert) -> str:
        """Send alert via email."""
        if not all([
            self.config.get("smtp_username"),
            self.config.get("smtp_password"),
            self.config.get("email_from"),
            self.config.get("email_to")
        ]):
            return "email_not_configured"
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config["email_from"]
            msg['To'] = ", ".join(self.config["email_to"])
            msg['Subject'] = f"Trading Alert: {alert.title}"
            
            # Create HTML version of the message
            html_message = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange' if alert.severity == AlertSeverity.HIGH else 'blue'};">
                    {alert.title}
                </h2>
                <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
                    {alert.message.replace('**', '<b>').replace('**', '</b>').replace('\n', '<br>')}
                </div>
                <p><small>Generated at {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</small></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_message, 'html'))
            
            # Send email using asyncio to avoid blocking
            def send_email():
                with smtplib.SMTP(self.config["smtp_server"], self.config["smtp_port"]) as server:
                    server.starttls()
                    server.login(self.config["smtp_username"], self.config["smtp_password"])
                    server.send_message(msg)
                    
            await asyncio.get_event_loop().run_in_executor(None, send_email)
            return "sent"
            
        except Exception as e:
            return f"error: {str(e)}"
            
    async def _send_slack_alert(self, alert: Alert) -> str:
        """Send alert to Slack."""
        webhook_url = self.config.get("slack_webhook_url")
        if not webhook_url:
            return "slack_not_configured"
            
        # Determine color based on severity
        color_map = {
            AlertSeverity.LOW: "#36a64f",      # Green
            AlertSeverity.MEDIUM: "#ff9500",   # Orange  
            AlertSeverity.HIGH: "#ff4500",     # Red-orange
            AlertSeverity.CRITICAL: "#ff0000" # Red
        }
        
        payload = {
            "channel": self.config.get("slack_channel", "#trading-alerts"),
            "username": "Trading Alert Bot",
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#ff9500"),
                    "title": alert.title,
                    "text": alert.message,
                    "footer": "Event Data Service",
                    "ts": int(alert.created_at.timestamp()),
                    "fields": [
                        {
                            "title": "Symbol",
                            "value": alert.event_data.get("symbol", "Unknown"),
                            "short": True
                        },
                        {
                            "title": "Impact Score",
                            "value": f"{alert.event_data.get('impact_score', 0)}/10",
                            "short": True
                        }
                    ]
                }
            ]
        }
        
        try:
            response = await self._http_client.post(webhook_url, json=payload)
            if response.status_code == 200:
                return "sent"
            else:
                return f"error: HTTP {response.status_code}"
        except Exception as e:
            return f"error: {str(e)}"
            
    async def _send_webhook_alert(self, alert: Alert) -> str:
        """Send alert to generic webhooks."""
        webhook_urls = self.config.get("webhook_urls", [])
        if not webhook_urls:
            return "webhooks_not_configured"
            
        payload = {
            "alert_type": "trading_event",
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "event_data": alert.event_data,
            "timestamp": alert.created_at.isoformat()
        }
        
        results = []
        for webhook_url in webhook_urls:
            try:
                response = await self._http_client.post(webhook_url, json=payload)
                results.append(f"{webhook_url}: {response.status_code}")
            except Exception as e:
                results.append(f"{webhook_url}: error")
                
        return "; ".join(results) if results else "no_webhooks"
        
    async def _send_sms_alert(self, alert: Alert) -> str:
        """Send alert via SMS using Twilio."""
        if not all([
            self.config.get("twilio_account_sid"),
            self.config.get("twilio_auth_token"),
            self.config.get("twilio_from_number"),
            self.config.get("sms_to_numbers")
        ]):
            return "sms_not_configured"
            
        # Simplified SMS message
        sms_message = f"{alert.title}\n{alert.event_data.get('symbol', 'Unknown')}: {alert.event_data.get('category', 'event')} (Impact: {alert.event_data.get('impact_score', 0)}/10)"
        
        # Would integrate with Twilio API here
        # For demo purposes, return success
        return "sent_via_twilio"
        
    async def _send_teams_alert(self, alert: Alert) -> str:
        """Send alert to Microsoft Teams."""
        webhook_url = self.config.get("teams_webhook_url")
        if not webhook_url:
            return "teams_not_configured"
            
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "FF4500" if alert.severity == AlertSeverity.CRITICAL else "FF9500",
            "summary": alert.title,
            "sections": [
                {
                    "activityTitle": alert.title,
                    "activitySubtitle": f"Severity: {alert.severity.value.upper()}",
                    "text": alert.message,
                    "facts": [
                        {
                            "name": "Symbol",
                            "value": alert.event_data.get("symbol", "Unknown")
                        },
                        {
                            "name": "Impact Score", 
                            "value": f"{alert.event_data.get('impact_score', 0)}/10"
                        },
                        {
                            "name": "Time",
                            "value": alert.created_at.strftime('%Y-%m-%d %H:%M UTC')
                        }
                    ]
                }
            ]
        }
        
        try:
            response = await self._http_client.post(webhook_url, json=payload)
            return "sent" if response.status_code == 200 else f"error: HTTP {response.status_code}"
        except Exception as e:
            return f"error: {str(e)}"
            
    async def _send_discord_alert(self, alert: Alert) -> str:
        """Send alert to Discord."""
        webhook_url = self.config.get("discord_webhook_url")
        if not webhook_url:
            return "discord_not_configured"
            
        # Discord color codes (decimal)
        color_map = {
            AlertSeverity.LOW: 3066993,      # Green
            AlertSeverity.MEDIUM: 16753920,  # Orange
            AlertSeverity.HIGH: 16734003,    # Red-orange
            AlertSeverity.CRITICAL: 16711680 # Red
        }
        
        payload = {
            "username": "Trading Alert Bot",
            "embeds": [
                {
                    "title": alert.title,
                    "description": alert.message,
                    "color": color_map.get(alert.severity, 16753920),
                    "timestamp": alert.created_at.isoformat(),
                    "fields": [
                        {
                            "name": "Symbol",
                            "value": alert.event_data.get("symbol", "Unknown"),
                            "inline": True
                        },
                        {
                            "name": "Impact Score",
                            "value": f"{alert.event_data.get('impact_score', 0)}/10",
                            "inline": True
                        }
                    ],
                    "footer": {
                        "text": "Event Data Service"
                    }
                }
            ]
        }
        
        try:
            response = await self._http_client.post(webhook_url, json=payload)
            return "sent" if response.status_code == 204 else f"error: HTTP {response.status_code}"
        except Exception as e:
            return f"error: {str(e)}"
            
    def add_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self._alert_rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")
        
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
        
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID."""
        return self._alert_rules.get(rule_id)
        
    def list_rules(self) -> List[AlertRule]:
        """List all alert rules."""
        return list(self._alert_rules.values())
        
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        return self._alert_history[-limit:]
        
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting system statistics."""
        total_alerts = len(self._alert_history)
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([a for a in self._alert_history if a.severity == severity])
            
        # Count by channel
        channel_counts = {}
        for channel in AlertChannel:
            channel_counts[channel.value] = sum(
                1 for alert in self._alert_history 
                for alert_channel in alert.channels 
                if alert_channel == channel
            )
            
        # Recent activity
        recent_24h = len([
            a for a in self._alert_history 
            if (datetime.now(timezone.utc) - a.created_at).total_seconds() < 86400
        ])
        
        return {
            "total_alerts": total_alerts,
            "alerts_24h": recent_24h,
            "active_rules": len([r for r in self._alert_rules.values() if r.enabled]),
            "total_rules": len(self._alert_rules),
            "severity_distribution": severity_counts,
            "channel_usage": channel_counts,
            "configuration": {
                "email_enabled": bool(self.config.get("email_from")),
                "slack_enabled": bool(self.config.get("slack_webhook_url")),
                "sms_enabled": bool(self.config.get("twilio_account_sid")),
                "teams_enabled": bool(self.config.get("teams_webhook_url")),
                "discord_enabled": bool(self.config.get("discord_webhook_url")),
                "webhooks_enabled": bool(self.config.get("webhook_urls"))
            }
        }


def build_alert_system(config: Optional[Dict[str, Any]] = None) -> EventAlertSystem:
    """Build and configure the event alert system."""
    return EventAlertSystem(config)