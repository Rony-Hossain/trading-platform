"""
Decision Explanations and "Why/Why-Not" Trace System
Provides detailed explanations for trading decisions and rejections
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import threading
from decimal import Decimal

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of trading decisions"""
    ENTRY_ALLOWED = "entry_allowed"
    ENTRY_REJECTED = "entry_rejected"
    SIZE_ADJUSTED = "size_adjusted"
    EXIT_TRIGGERED = "exit_triggered"
    POSITION_HELD = "position_held"

class RejectionReason(Enum):
    """Specific rejection reasons mapped to pipeline stages"""
    # Regime Filter
    REGIME_BEARISH = "regime_bearish"
    REGIME_HIGH_VOLATILITY = "regime_high_volatility"
    REGIME_LOW_LIQUIDITY = "regime_low_liquidity"
    
    # VaR Calculation
    VAR_EXCEEDED = "var_exceeded"
    VAR_PORTFOLIO_RISK = "var_portfolio_risk"
    VAR_CONCENTRATION = "var_concentration"
    
    # Borrow Check
    BORROW_UNAVAILABLE = "borrow_unavailable"
    BORROW_EXPENSIVE = "borrow_expensive"
    BORROW_LIMITED_QUANTITY = "borrow_limited_quantity"
    
    # Position Sizing
    SIZE_TOO_SMALL = "size_too_small"
    SIZE_EXCEEDS_LIMIT = "size_exceeds_limit"
    CAPITAL_INSUFFICIENT = "capital_insufficient"
    
    # Venue Rules
    HALTED = "halted"
    LULD_VIOLATION = "luld_violation"
    POST_REOPEN_DELAY = "post_reopen_delay"
    
    # SPA/DSR Gates
    SPA_GATE_FAILED = "spa_gate_failed"
    DSR_GATE_FAILED = "dsr_gate_failed"
    CORRELATION_HIGH = "correlation_high"
    
    # Caps and Limits
    POSITION_CAP_REACHED = "position_cap_reached"
    EXPOSURE_CAP_REACHED = "exposure_cap_reached"
    SECTOR_CAP_REACHED = "sector_cap_reached"
    
    # Signal Quality
    SIGNAL_WEAK = "signal_weak"
    SIGNAL_STALE = "signal_stale"
    SIGNAL_CONFLICTING = "signal_conflicting"

@dataclass
class RejectionDetail:
    """Detailed information about a rejection reason"""
    reason: RejectionReason
    stage: str
    message: str
    threshold: Optional[Union[float, int]] = None
    actual_value: Optional[Union[float, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason.value,
            "stage": self.stage,
            "message": self.message,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "metadata": self.metadata
        }

@dataclass
class SignalAnalysis:
    """Analysis of the trading signal"""
    signal_strength: float
    confidence: float
    signal_type: str
    features_used: List[str]
    feature_importance: Dict[str, float] = field(default_factory=dict)
    signal_age_ms: Optional[float] = None
    conflicting_signals: List[str] = field(default_factory=list)

@dataclass
class RiskAnalysis:
    """Risk analysis details"""
    portfolio_var: float
    position_var: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    regime_risk: str
    var_threshold: float
    var_utilization_pct: float

@dataclass
class ExecutionConstraints:
    """Execution constraints and limits"""
    max_position_size: Optional[int] = None
    available_capital: Optional[float] = None
    borrow_available: Optional[bool] = None
    borrow_rate: Optional[float] = None
    venue_restrictions: List[str] = field(default_factory=list)
    position_caps: Dict[str, float] = field(default_factory=dict)

@dataclass
class TradeDecision:
    """Complete trading decision with full explanation"""
    decision_id: str
    trade_id: Optional[str]
    timestamp: datetime
    symbol: str
    side: str
    original_quantity: int
    suggested_quantity: Optional[int]
    decision_type: DecisionType
    
    # Core analysis
    signal_analysis: SignalAnalysis
    risk_analysis: RiskAnalysis
    execution_constraints: ExecutionConstraints
    
    # Decision outcome
    is_approved: bool
    final_message: str
    rejection_details: List[RejectionDetail] = field(default_factory=list)
    
    # Timing and performance
    decision_latency_ms: Optional[float] = None
    trace_id: Optional[str] = None
    
    # Additional context
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    portfolio_context: Dict[str, Any] = field(default_factory=dict)
    
    def add_rejection(self, reason: RejectionReason, stage: str, message: str,
                     threshold: Optional[Union[float, int]] = None,
                     actual_value: Optional[Union[float, int]] = None,
                     metadata: Dict[str, Any] = None):
        """Add a rejection reason"""
        detail = RejectionDetail(
            reason=reason,
            stage=stage,
            message=message,
            threshold=threshold,
            actual_value=actual_value,
            metadata=metadata or {}
        )
        self.rejection_details.append(detail)
        self.is_approved = False
    
    def get_primary_rejection_reason(self) -> Optional[str]:
        """Get the primary (first) rejection reason"""
        if self.rejection_details:
            return self.rejection_details[0].reason.value
        return None
    
    def get_rejection_summary(self) -> Dict[str, Any]:
        """Get summary of all rejection reasons"""
        if not self.rejection_details:
            return {}
        
        reasons_by_stage = defaultdict(list)
        for detail in self.rejection_details:
            reasons_by_stage[detail.stage].append(detail.reason.value)
        
        return {
            "primary_reason": self.get_primary_rejection_reason(),
            "total_rejections": len(self.rejection_details),
            "reasons_by_stage": dict(reasons_by_stage),
            "all_reasons": [detail.reason.value for detail in self.rejection_details]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "decision_id": self.decision_id,
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "original_quantity": self.original_quantity,
            "suggested_quantity": self.suggested_quantity,
            "decision_type": self.decision_type.value,
            "is_approved": self.is_approved,
            "final_message": self.final_message,
            "decision_latency_ms": self.decision_latency_ms,
            "trace_id": self.trace_id,
            
            "signal_analysis": {
                "signal_strength": self.signal_analysis.signal_strength,
                "confidence": self.signal_analysis.confidence,
                "signal_type": self.signal_analysis.signal_type,
                "features_used": self.signal_analysis.features_used,
                "feature_importance": self.signal_analysis.feature_importance,
                "signal_age_ms": self.signal_analysis.signal_age_ms,
                "conflicting_signals": self.signal_analysis.conflicting_signals
            },
            
            "risk_analysis": {
                "portfolio_var": self.risk_analysis.portfolio_var,
                "position_var": self.risk_analysis.position_var,
                "correlation_risk": self.risk_analysis.correlation_risk,
                "concentration_risk": self.risk_analysis.concentration_risk,
                "liquidity_risk": self.risk_analysis.liquidity_risk,
                "regime_risk": self.risk_analysis.regime_risk,
                "var_threshold": self.risk_analysis.var_threshold,
                "var_utilization_pct": self.risk_analysis.var_utilization_pct
            },
            
            "execution_constraints": {
                "max_position_size": self.execution_constraints.max_position_size,
                "available_capital": self.execution_constraints.available_capital,
                "borrow_available": self.execution_constraints.borrow_available,
                "borrow_rate": self.execution_constraints.borrow_rate,
                "venue_restrictions": self.execution_constraints.venue_restrictions,
                "position_caps": self.execution_constraints.position_caps
            },
            
            "rejection_details": [detail.to_dict() for detail in self.rejection_details],
            "rejection_summary": self.get_rejection_summary(),
            
            "market_conditions": self.market_conditions,
            "portfolio_context": self.portfolio_context
        }

class DecisionExplainer:
    """Main decision explanation system"""
    
    def __init__(self, max_decisions: int = 10000):
        self.max_decisions = max_decisions
        self.decisions: Dict[str, TradeDecision] = {}
        self.decisions_by_trade: Dict[str, str] = {}
        self.recent_decisions = deque(maxlen=max_decisions)
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "total_decisions": 0,
            "approved_decisions": 0,
            "rejected_decisions": 0,
            "rejection_reasons": defaultdict(int),
            "avg_decision_latency_ms": 0.0,
            "last_reset": datetime.now()
        }
    
    def create_decision(self, symbol: str, side: str, quantity: int, 
                       trade_id: Optional[str] = None, 
                       trace_id: Optional[str] = None) -> TradeDecision:
        """Create a new trading decision"""
        decision_id = str(uuid.uuid4())
        
        decision = TradeDecision(
            decision_id=decision_id,
            trade_id=trade_id,
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            original_quantity=quantity,
            suggested_quantity=quantity,
            decision_type=DecisionType.ENTRY_ALLOWED,
            signal_analysis=SignalAnalysis(
                signal_strength=0.0,
                confidence=0.0,
                signal_type="unknown",
                features_used=[]
            ),
            risk_analysis=RiskAnalysis(
                portfolio_var=0.0,
                position_var=0.0,
                correlation_risk=0.0,
                concentration_risk=0.0,
                liquidity_risk=0.0,
                regime_risk="unknown",
                var_threshold=0.0,
                var_utilization_pct=0.0
            ),
            execution_constraints=ExecutionConstraints(),
            is_approved=True,
            final_message="Decision pending analysis",
            trace_id=trace_id
        )
        
        with self.lock:
            self.decisions[decision_id] = decision
            if trade_id:
                self.decisions_by_trade[trade_id] = decision_id
            self.recent_decisions.append(decision_id)
            self.stats["total_decisions"] += 1
        
        return decision
    
    def finalize_decision(self, decision: TradeDecision, latency_ms: Optional[float] = None):
        """Finalize a decision with timing information"""
        decision.decision_latency_ms = latency_ms
        
        if decision.is_approved:
            decision.decision_type = DecisionType.ENTRY_ALLOWED
            if decision.suggested_quantity != decision.original_quantity:
                decision.decision_type = DecisionType.SIZE_ADJUSTED
                decision.final_message = f"Position size adjusted from {decision.original_quantity} to {decision.suggested_quantity}"
            else:
                decision.final_message = f"Trade approved: {decision.side} {decision.suggested_quantity} {decision.symbol}"
        else:
            decision.decision_type = DecisionType.ENTRY_REJECTED
            primary_reason = decision.get_primary_rejection_reason()
            decision.final_message = f"Trade rejected: {primary_reason or 'multiple_reasons'}"
        
        with self.lock:
            # Update statistics
            if decision.is_approved:
                self.stats["approved_decisions"] += 1
            else:
                self.stats["rejected_decisions"] += 1
                for detail in decision.rejection_details:
                    self.stats["rejection_reasons"][detail.reason.value] += 1
            
            # Update average latency
            if latency_ms is not None:
                current_avg = self.stats["avg_decision_latency_ms"]
                total_decisions = self.stats["total_decisions"]
                self.stats["avg_decision_latency_ms"] = (
                    (current_avg * (total_decisions - 1) + latency_ms) / total_decisions
                )
        
        logger.info(f"Finalized decision {decision.decision_id}: {decision.final_message}")
    
    def get_decision(self, decision_id: str) -> Optional[TradeDecision]:
        """Get decision by ID"""
        return self.decisions.get(decision_id)
    
    def get_decision_by_trade(self, trade_id: str) -> Optional[TradeDecision]:
        """Get decision by trade ID"""
        decision_id = self.decisions_by_trade.get(trade_id)
        if decision_id:
            return self.decisions.get(decision_id)
        return None
    
    def get_recent_decisions(self, limit: int = 100) -> List[TradeDecision]:
        """Get recent decisions"""
        with self.lock:
            recent_ids = list(self.recent_decisions)[-limit:]
        
        return [self.decisions[decision_id] for decision_id in recent_ids 
                if decision_id in self.decisions]
    
    def get_rejected_decisions(self, limit: int = 100) -> List[TradeDecision]:
        """Get recent rejected decisions"""
        recent = self.get_recent_decisions(limit * 2)  # Get more to filter
        rejected = [d for d in recent if not d.is_approved]
        return rejected[:limit]
    
    def explain_rejection(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed explanation for a rejection"""
        decision = self.get_decision(decision_id)
        if not decision or decision.is_approved:
            return None
        
        explanation = {
            "decision_id": decision_id,
            "trade_id": decision.trade_id,
            "symbol": decision.symbol,
            "side": decision.side,
            "original_quantity": decision.original_quantity,
            "timestamp": decision.timestamp.isoformat(),
            "rejection_summary": decision.get_rejection_summary(),
            "detailed_reasons": [],
            "market_context": decision.market_conditions,
            "portfolio_context": decision.portfolio_context
        }
        
        # Add detailed explanations for each rejection
        for detail in decision.rejection_details:
            detailed_reason = {
                "reason": detail.reason.value,
                "stage": detail.stage,
                "message": detail.message,
                "explanation": self._get_reason_explanation(detail.reason),
                "remediation": self._get_remediation_suggestion(detail.reason),
                "threshold": detail.threshold,
                "actual_value": detail.actual_value,
                "metadata": detail.metadata
            }
            explanation["detailed_reasons"].append(detailed_reason)
        
        return explanation
    
    def _get_reason_explanation(self, reason: RejectionReason) -> str:
        """Get human-readable explanation for a rejection reason"""
        explanations = {
            RejectionReason.REGIME_BEARISH: "Market regime indicates bearish conditions with elevated downside risk",
            RejectionReason.REGIME_HIGH_VOLATILITY: "Market volatility exceeds acceptable levels for this strategy",
            RejectionReason.REGIME_LOW_LIQUIDITY: "Market liquidity is insufficient for reliable execution",
            
            RejectionReason.VAR_EXCEEDED: "Position would exceed Value-at-Risk limits for the portfolio",
            RejectionReason.VAR_PORTFOLIO_RISK: "Adding this position would increase portfolio risk beyond acceptable levels",
            RejectionReason.VAR_CONCENTRATION: "Position creates excessive concentration risk in portfolio",
            
            RejectionReason.BORROW_UNAVAILABLE: "Shares are not available for borrowing to execute short sale",
            RejectionReason.BORROW_EXPENSIVE: "Borrowing costs exceed maximum acceptable rate for this position",
            RejectionReason.BORROW_LIMITED_QUANTITY: "Insufficient borrowable shares available for desired position size",
            
            RejectionReason.SIZE_TOO_SMALL: "Position size falls below minimum economic threshold",
            RejectionReason.SIZE_EXCEEDS_LIMIT: "Position size exceeds maximum allowed limit",
            RejectionReason.CAPITAL_INSUFFICIENT: "Insufficient capital available to execute this trade",
            
            RejectionReason.HALTED: "Trading is halted for this symbol due to regulatory restrictions",
            RejectionReason.LULD_VIOLATION: "Order price violates Limit Up/Limit Down bands",
            RejectionReason.POST_REOPEN_DELAY: "Post-halt reopen delay is active due to gap pricing protection",
            
            RejectionReason.SPA_GATE_FAILED: "Systematic Performance Attribution gate failed risk checks",
            RejectionReason.DSR_GATE_FAILED: "Dynamic Style Rotation gate indicates unfavorable conditions",
            RejectionReason.CORRELATION_HIGH: "Position exhibits high correlation with existing holdings",
            
            RejectionReason.POSITION_CAP_REACHED: "Maximum position size cap has been reached for this symbol",
            RejectionReason.EXPOSURE_CAP_REACHED: "Portfolio exposure limit reached for this sector/theme",
            RejectionReason.SECTOR_CAP_REACHED: "Sector allocation cap prevents additional exposure",
            
            RejectionReason.SIGNAL_WEAK: "Trading signal strength is below minimum confidence threshold",
            RejectionReason.SIGNAL_STALE: "Trading signal is too old and may no longer be reliable",
            RejectionReason.SIGNAL_CONFLICTING: "Multiple conflicting signals detected for this symbol"
        }
        
        return explanations.get(reason, f"Rejection reason: {reason.value}")
    
    def _get_remediation_suggestion(self, reason: RejectionReason) -> str:
        """Get remediation suggestion for a rejection reason"""
        suggestions = {
            RejectionReason.REGIME_BEARISH: "Wait for regime change or consider defensive positioning",
            RejectionReason.REGIME_HIGH_VOLATILITY: "Reduce position size or wait for volatility to decline",
            RejectionReason.REGIME_LOW_LIQUIDITY: "Trade smaller size or use limit orders with patience",
            
            RejectionReason.VAR_EXCEEDED: "Reduce position size or close other positions to free up risk budget",
            RejectionReason.VAR_PORTFOLIO_RISK: "Consider portfolio rebalancing or risk-off positioning",
            RejectionReason.VAR_CONCENTRATION: "Diversify holdings or reduce concentration in correlated assets",
            
            RejectionReason.BORROW_UNAVAILABLE: "Try again later or consider alternative instruments",
            RejectionReason.BORROW_EXPENSIVE: "Reduce position size or wait for better borrow rates",
            RejectionReason.BORROW_LIMITED_QUANTITY: "Reduce order size to available quantity",
            
            RejectionReason.SIZE_TOO_SMALL: "Increase position size to meet minimum threshold",
            RejectionReason.SIZE_EXCEEDS_LIMIT: "Reduce position size to comply with limits",
            RejectionReason.CAPITAL_INSUFFICIENT: "Free up capital by closing other positions",
            
            RejectionReason.HALTED: "Wait for trading to resume or monitor halt updates",
            RejectionReason.LULD_VIOLATION: "Adjust order price within acceptable bands",
            RejectionReason.POST_REOPEN_DELAY: "Wait for delay period to expire",
            
            RejectionReason.SPA_GATE_FAILED: "Review attribution factors and strategy alignment",
            RejectionReason.DSR_GATE_FAILED: "Consider alternative style exposures or timing",
            RejectionReason.CORRELATION_HIGH: "Diversify holdings or choose uncorrelated alternatives",
            
            RejectionReason.POSITION_CAP_REACHED: "Close existing position to make room for new trade",
            RejectionReason.EXPOSURE_CAP_REACHED: "Rebalance sector allocations or reduce other exposures",
            RejectionReason.SECTOR_CAP_REACHED: "Consider opportunities in other sectors",
            
            RejectionReason.SIGNAL_WEAK: "Wait for stronger signal or reduce position size",
            RejectionReason.SIGNAL_STALE: "Refresh signal data or wait for new signal generation",
            RejectionReason.SIGNAL_CONFLICTING: "Resolve signal conflicts or wait for clarity"
        }
        
        return suggestions.get(reason, "Review conditions and retry when appropriate")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics"""
        with self.lock:
            stats = self.stats.copy()
        
        # Calculate approval rate
        total = stats["total_decisions"]
        approval_rate = (stats["approved_decisions"] / total * 100) if total > 0 else 0
        
        # Get top rejection reasons
        top_rejections = sorted(
            stats["rejection_reasons"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_decisions": total,
            "approved_decisions": stats["approved_decisions"],
            "rejected_decisions": stats["rejected_decisions"],
            "approval_rate_pct": approval_rate,
            "avg_decision_latency_ms": stats["avg_decision_latency_ms"],
            "top_rejection_reasons": top_rejections,
            "last_reset": stats["last_reset"].isoformat()
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        with self.lock:
            self.stats = {
                "total_decisions": 0,
                "approved_decisions": 0,
                "rejected_decisions": 0,
                "rejection_reasons": defaultdict(int),
                "avg_decision_latency_ms": 0.0,
                "last_reset": datetime.now()
            }
        
        logger.info("Reset decision statistics")

# Global decision explainer instance
decision_explainer = DecisionExplainer()

# Convenience functions
def create_trade_decision(symbol: str, side: str, quantity: int, 
                         trade_id: Optional[str] = None,
                         trace_id: Optional[str] = None) -> TradeDecision:
    """Create a new trade decision"""
    return decision_explainer.create_decision(symbol, side, quantity, trade_id, trace_id)

def finalize_trade_decision(decision: TradeDecision, latency_ms: Optional[float] = None):
    """Finalize a trade decision"""
    decision_explainer.finalize_decision(decision, latency_ms)

def explain_trade_rejection(decision_id: str) -> Optional[Dict[str, Any]]:
    """Get explanation for trade rejection"""
    return decision_explainer.explain_rejection(decision_id)

def get_decision_statistics() -> Dict[str, Any]:
    """Get decision statistics"""
    return decision_explainer.get_statistics()