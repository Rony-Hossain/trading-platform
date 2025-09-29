import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertType(Enum):
    PRICE_THRESHOLD = "price_threshold"
    PRICE_CHANGE = "price_change"
    VOLUME_SPIKE = "volume_spike"
    TECHNICAL_SIGNAL = "technical_signal"

class AlertCondition(Enum):
    ABOVE = "above"
    BELOW = "below"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"

@dataclass
class AlertRule:
    id: str
    symbol: str
    alert_type: AlertType
    condition: AlertCondition
    threshold: float
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class AlertTrigger:
    rule_id: str
    symbol: str
    message: str
    current_value: float
    threshold: float
    triggered_at: datetime = None
    
    def __post_init__(self):
        if self.triggered_at is None:
            self.triggered_at = datetime.now()

class AlertEngine:
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.triggers: List[AlertTrigger] = []
        self.price_history: Dict[str, List[Dict]] = {}  # symbol -> recent prices
        self.running = False
        
    def add_rule(self, rule: AlertRule) -> str:
        """Add an alert rule"""
        self.rules[rule.id] = rule
        logger.info(f"Added alert rule {rule.id} for {rule.symbol}")
        return rule.id
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule {rule_id}")
            return True
        return False
    
    def get_rules_for_symbol(self, symbol: str) -> List[AlertRule]:
        """Get all active rules for a symbol"""
        return [rule for rule in self.rules.values() 
                if rule.symbol == symbol and rule.enabled]
    
    def is_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period"""
        if rule.last_triggered is None:
            return False
        cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def update_price_history(self, symbol: str, price_data: Dict):
        """Update price history for technical analysis"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'timestamp': datetime.now(),
            'price': price_data.get('price', 0),
            'volume': price_data.get('volume', 0),
            'high': price_data.get('high', price_data.get('price', 0)),
            'low': price_data.get('low', price_data.get('price', 0)),
        })
        
        # Keep only last 200 data points for efficiency
        if len(self.price_history[symbol]) > 200:
            self.price_history[symbol] = self.price_history[symbol][-200:]
    
    def check_price_threshold(self, rule: AlertRule, current_price: float) -> Optional[AlertTrigger]:
        """Check price threshold alerts"""
        if rule.condition == AlertCondition.ABOVE and current_price > rule.threshold:
            return AlertTrigger(
                rule_id=rule.id,
                symbol=rule.symbol,
                message=f"{rule.symbol} price ${current_price:.2f} is above threshold ${rule.threshold:.2f}",
                current_value=current_price,
                threshold=rule.threshold
            )
        elif rule.condition == AlertCondition.BELOW and current_price < rule.threshold:
            return AlertTrigger(
                rule_id=rule.id,
                symbol=rule.symbol,
                message=f"{rule.symbol} price ${current_price:.2f} is below threshold ${rule.threshold:.2f}",
                current_value=current_price,
                threshold=rule.threshold
            )
        return None
    
    def check_price_change(self, rule: AlertRule, symbol: str) -> Optional[AlertTrigger]:
        """Check price change percentage alerts"""
        history = self.price_history.get(symbol, [])
        if len(history) < 2:
            return None
            
        current_price = history[-1]['price']
        previous_price = history[-2]['price']
        
        if previous_price == 0:
            return None
            
        change_percent = ((current_price - previous_price) / previous_price) * 100
        
        if abs(change_percent) >= rule.threshold:
            direction = "increased" if change_percent > 0 else "decreased"
            return AlertTrigger(
                rule_id=rule.id,
                symbol=rule.symbol,
                message=f"{rule.symbol} price {direction} by {abs(change_percent):.1f}% to ${current_price:.2f}",
                current_value=change_percent,
                threshold=rule.threshold
            )
        return None
    
    def check_volume_spike(self, rule: AlertRule, symbol: str) -> Optional[AlertTrigger]:
        """Check for volume spikes"""
        history = self.price_history.get(symbol, [])
        if len(history) < 10:  # Need some history for average
            return None
            
        current_volume = history[-1]['volume']
        recent_volumes = [h['volume'] for h in history[-10:-1]]  # Last 9 volumes
        avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            if volume_ratio >= rule.threshold:
                return AlertTrigger(
                    rule_id=rule.id,
                    symbol=rule.symbol,
                    message=f"{rule.symbol} volume spike: {current_volume:,} ({volume_ratio:.1f}x average)",
                    current_value=volume_ratio,
                    threshold=rule.threshold
                )
        return None
    
    def process_price_update(self, symbol: str, price_data: Dict):
        """Process a price update and check all rules"""
        self.update_price_history(symbol, price_data)
        
        rules = self.get_rules_for_symbol(symbol)
        current_price = price_data.get('price', 0)
        
        for rule in rules:
            if self.is_in_cooldown(rule):
                continue
                
            trigger = None
            
            if rule.alert_type == AlertType.PRICE_THRESHOLD:
                trigger = self.check_price_threshold(rule, current_price)
            elif rule.alert_type == AlertType.PRICE_CHANGE:
                trigger = self.check_price_change(rule, symbol)
            elif rule.alert_type == AlertType.VOLUME_SPIKE:
                trigger = self.check_volume_spike(rule, symbol)
                
            if trigger:
                self.triggers.append(trigger)
                rule.last_triggered = datetime.now()
                logger.info(f"Alert triggered: {trigger.message}")
    
    def get_recent_triggers(self, limit: int = 50) -> List[AlertTrigger]:
        """Get recent alert triggers"""
        return sorted(self.triggers, key=lambda x: x.triggered_at, reverse=True)[:limit]
    
    def get_stats(self) -> Dict:
        """Get alert engine statistics"""
        return {
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_triggers": len(self.triggers),
            "recent_triggers": len([t for t in self.triggers 
                                  if t.triggered_at > datetime.now() - timedelta(hours=24)]),
            "symbols_monitored": len(set(rule.symbol for rule in self.rules.values()))
        }

# Global alert engine instance
alert_engine = AlertEngine()