"""
Data Contracts - Pydantic Models
All request/response schemas for API endpoints
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime
from enum import Enum


# =============================================================================
# SHARED METADATA
# =============================================================================

class SourceModel(BaseModel):
    """Model metadata"""
    name: str
    version: str
    sha: str
    confidence: Optional[float] = None


class ResponseMetadata(BaseModel):
    """Shared response metadata"""
    request_id: str = Field(..., description="ULID for tracing")
    generated_at: datetime
    version: str  # e.g., "plan.v1"
    latency_ms: float
    source_models: List[SourceModel]


class ErrorResponse(BaseModel):
    """Error response structure"""
    code: str  # UPSTREAM_TIMEOUT, INSUFFICIENT_DATA, etc.
    message: str
    retry_after_seconds: Optional[int] = None
    degraded_fields: Optional[List[str]] = None  # ["sentiment", "target"]


# =============================================================================
# PLAN ENDPOINT
# =============================================================================

class ReasonCode(str, Enum):
    """Machine-readable reason codes"""
    SUPPORT_BOUNCE = "support_bounce"
    BUYER_PRESSURE = "buyer_pressure"
    NEWS_SENTIMENT = "news_sentiment"
    MOMENTUM_SHIFT = "momentum_shift"
    VOLATILITY_SPIKE = "volatility_spike"
    EARNINGS_SURPRISE = "earnings_surprise"
    TECHNICAL_BREAKOUT = "technical_breakout"
    OVERSOLD_RSI = "oversold_rsi"
    VOLUME_SURGE = "volume_surge"
    TREND_REVERSAL = "trend_reversal"


class Constraints(BaseModel):
    """Position constraints"""
    stop_loss: float
    max_position_value_usd: float
    min_holding_period_sec: int = 300  # 5min min hold for beginners


class LimitsApplied(BaseModel):
    """Guardrails that were applied"""
    volatility_brake: bool = False
    earnings_window: bool = False
    cap_hit: bool = False
    drift_downgrade: bool = False


class Compliance(BaseModel):
    """Compliance restrictions"""
    leverage_ok: bool = False
    options_allowed: bool = False
    short_allowed: bool = False
    paper_trade_only: bool = True


class Driver(BaseModel):
    """Feature importance driver (for expert mode)"""
    name: str
    contribution: float  # 0.0-1.0
    value: Optional[float] = None  # Actual value if numeric


class Pick(BaseModel):
    """Trading pick/recommendation"""
    symbol: str
    action: Literal["BUY", "SELL", "HOLD", "AVOID"]
    shares: int
    entry_hint: float
    safety_line: float
    target: Optional[float] = None  # None if forecast unavailable
    confidence: Literal["low", "medium", "high"]
    reason: str  # Plain English
    reason_codes: List[ReasonCode]
    max_risk_usd: float
    budget_impact: Dict[str, float]  # {"cash_left": 982.50}
    constraints: Constraints
    limits_applied: LimitsApplied
    compliance: Compliance
    decision_path: str  # "ALPHA>THRESH>RISK_OK>SENTIMENT_OK"
    drivers: Optional[List[Driver]] = None  # Expert mode only
    reason_score: float = 0.0  # Quality score 0-1


class DailyCap(BaseModel):
    """Daily loss cap"""
    max_loss_usd: float
    used_usd: float
    status: Literal["ok", "warning", "hit"]
    reset_at: datetime
    reason: Optional[str] = None  # Reason when status != ok


class IndicatorSignal(BaseModel):
    """Technical indicator signal"""
    value: float
    signal: Literal["bullish", "bearish", "neutral", "oversold", "overbought"]
    color: Literal["green", "red", "yellow"]


class ExpertPanels(BaseModel):
    """Expert mode additional data"""
    indicators: Dict[str, IndicatorSignal]  # {"rsi": {...}, "macd": {...}}
    options: Dict[str, float]  # {"iv_rank": 65, "max_pain": 185}
    diagnostics: Dict  # {"drift_status": "green", "top_drivers": [...]}
    explain_tokens: List[str]  # Preload hints for tooltips


class PlanResponse(BaseModel):
    """Today's Plan response"""
    metadata: ResponseMetadata
    mode: Literal["beginner", "expert"]
    daily_cap: DailyCap
    picks: List[Pick]
    expert_panels: Optional[ExpertPanels] = None
    error: Optional[ErrorResponse] = None
    ui_hints: Optional[Dict] = None  # {"show_banner": "degraded_sentiment"}


# =============================================================================
# ALERTS ENDPOINT
# =============================================================================

class AlertThrottle(BaseModel):
    """Alert throttling info"""
    cooldown_sec: int = 900  # 15min default
    dedupe_key: str
    suppressed: bool = False
    suppressed_reason: Optional[str] = None


class AlertSafety(BaseModel):
    """Alert safety metrics"""
    max_loss_usd: float
    estimated_slippage_bps: float
    execution_confidence: float


class Alert(BaseModel):
    """Alert notification"""
    id: str  # ULID
    type: Literal["opportunity", "protect"]
    symbol: str
    message: str
    actions: List[Literal["buy_now", "sell_now", "snooze"]]
    safety: AlertSafety
    throttle: AlertThrottle
    paper_trade_only: bool
    expires_at: datetime
    created_at: datetime


class AlertsResponse(BaseModel):
    """Alerts list response"""
    metadata: ResponseMetadata
    alerts: List[Alert]
    armed: bool  # Whether alerts are enabled
    quiet_hours: List[str]  # ["22:00-07:00"]


class AlertArmRequest(BaseModel):
    """Configure alert preferences"""
    opportunity: bool = True
    protect: bool = True
    quiet_hours: List[str] = []  # ["22:00-07:00"]


# =============================================================================
# EXPLAIN ENDPOINT
# =============================================================================

class ExplainMath(BaseModel):
    """Mathematical explanation"""
    formula: str
    example: str


class ExplainResponse(BaseModel):
    """Term explanation"""
    term: str
    plain: str  # Plain English (<=60 words)
    how_we_use: str  # How we use it (<=60 words)
    math: ExplainMath
    last_reviewed: str  # YYYY-MM-DD
    related_terms: List[str] = []


# =============================================================================
# ACTIONS ENDPOINT
# =============================================================================

class BuyRequest(BaseModel):
    """Buy action request"""
    symbol: str
    shares: int
    limit_price: Optional[float] = None
    alert_id: Optional[str] = None  # If triggered by alert


class SellRequest(BaseModel):
    """Sell action request"""
    symbol: str
    shares: int
    limit_price: Optional[float] = None
    alert_id: Optional[str] = None


class ActionResponse(BaseModel):
    """Action execution response"""
    action_id: str
    status: Literal["pending", "executed", "failed"]
    symbol: str
    shares: int
    estimated_cost: float
    idempotency_key: str
    result: Optional[Dict] = None


# =============================================================================
# POSITIONS ENDPOINT
# =============================================================================

class Position(BaseModel):
    """Simplified position view"""
    symbol: str
    shares: int
    entry_price: float
    current_price: float
    pnl_usd: float
    pnl_pct: float
    safety_line: float
    max_planned_loss_usd: float
    message: str  # "You're up $12 (3.2%)"


class PositionsResponse(BaseModel):
    """Positions list response"""
    metadata: ResponseMetadata
    positions: List[Position]
    total_value: float
    cash_available: float
    total_pnl_usd: float
    total_pnl_pct: float


# =============================================================================
# INTERNAL ENDPOINTS
# =============================================================================

class SLOStatus(BaseModel):
    """SLO status"""
    availability: Dict
    latency_p95_ms: Dict
    error_budget_pct: float
    total_requests: int
    failed_requests: int


class HealthStatus(BaseModel):
    """Detailed health status"""
    status: Literal["healthy", "degraded", "unhealthy"]
    service: str
    version: str
    uptime_seconds: float
    dependencies: Dict[str, str]  # {"redis": "healthy", "inference": "healthy"}
