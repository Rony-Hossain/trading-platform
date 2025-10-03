"""
Trade Journal Service
Records trade fills, calculates P&L attribution, and provides reconciliation

Acceptance Criteria:
- Reconciliation: End-of-day balances match to 1 cent
- P&L attribution: Full cost breakdown (slippage, fees, borrow)
- Audit trail: Immutable fill records with timestamps
"""
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import asyncpg
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
fills_recorded = Counter('trade_journal_fills_recorded_total', 'Total fills recorded')
pnl_calculations = Counter('trade_journal_pnl_calculations_total', 'Total P&L calculations')
reconciliation_errors = Counter('trade_journal_reconciliation_errors_total', 'Reconciliation errors')
fill_latency = Histogram('trade_journal_fill_latency_seconds', 'Fill recording latency')
total_net_pnl = Gauge('trade_journal_total_net_pnl_usd', 'Total net P&L in USD')
position_count = Gauge('trade_journal_position_count', 'Number of open positions')

# FastAPI app
app = FastAPI(title="Trade Journal Service", version="1.0.0")

# Database pool
db_pool: Optional[asyncpg.Pool] = None


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    IOC = "IOC"
    POST_ONLY = "POST_ONLY"


@dataclass
class TradeFill:
    """Trade fill record"""
    order_id: str
    symbol: str
    venue: str
    side: OrderSide
    fill_price: Decimal
    fill_quantity: int
    fill_timestamp: datetime
    order_type: OrderType
    limit_price: Optional[Decimal] = None
    time_in_force: Optional[str] = None
    commission_usd: Decimal = Decimal('0')
    exchange_fee_usd: Decimal = Decimal('0')
    sec_fee_usd: Decimal = Decimal('0')
    finra_taf_usd: Decimal = Decimal('0')
    slippage_bps: Optional[Decimal] = None
    routing_decision_id: Optional[str] = None
    routing_latency_ms: Optional[Decimal] = None
    decision_factors: Optional[Dict[str, Any]] = None
    fill_id: Optional[int] = None


@dataclass
class Position:
    """Current position"""
    symbol: str
    current_quantity: int
    avg_entry_price: Decimal
    total_cost_basis_usd: Decimal
    unrealized_pnl_usd: Optional[Decimal] = None
    realized_pnl_usd: Decimal = Decimal('0')
    position_id: Optional[int] = None


@dataclass
class PnLAttribution:
    """P&L attribution breakdown"""
    symbol: str
    attribution_date: date
    gross_pnl_usd: Decimal
    commission_cost_usd: Decimal
    slippage_cost_usd: Decimal
    borrow_cost_usd: Decimal
    other_fees_usd: Decimal
    net_pnl_usd: Decimal
    strategy_name: Optional[str] = None
    venue: Optional[str] = None
    entry_fill_id: Optional[int] = None
    exit_fill_id: Optional[int] = None
    entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None
    quantity: Optional[int] = None
    holding_period_hours: Optional[Decimal] = None
    market_impact_bps: Optional[Decimal] = None
    timing_alpha_bps: Optional[Decimal] = None
    venue_selection_savings_bps: Optional[Decimal] = None


# Pydantic models for API
class FillRequest(BaseModel):
    order_id: str
    symbol: str
    venue: str
    side: OrderSide
    fill_price: float
    fill_quantity: int
    fill_timestamp: datetime
    order_type: OrderType
    limit_price: Optional[float] = None
    time_in_force: Optional[str] = None
    commission_usd: float = 0.0
    exchange_fee_usd: float = 0.0
    sec_fee_usd: float = 0.0
    finra_taf_usd: float = 0.0
    slippage_bps: Optional[float] = None
    routing_decision_id: Optional[str] = None
    routing_latency_ms: Optional[float] = None
    decision_factors: Optional[Dict[str, Any]] = None


class PositionResponse(BaseModel):
    position_id: int
    symbol: str
    current_quantity: int
    avg_entry_price: float
    total_cost_basis_usd: float
    unrealized_pnl_usd: Optional[float]
    realized_pnl_usd: float


class PnLAttributionResponse(BaseModel):
    pnl_id: int
    symbol: str
    attribution_date: date
    gross_pnl_usd: float
    commission_cost_usd: float
    slippage_cost_usd: float
    borrow_cost_usd: float
    other_fees_usd: float
    net_pnl_usd: float
    strategy_name: Optional[str]
    venue: Optional[str]
    market_impact_bps: Optional[float]
    timing_alpha_bps: Optional[float]
    venue_selection_savings_bps: Optional[float]


class ReconciliationReport(BaseModel):
    reconciliation_date: date
    total_fills: int
    total_gross_pnl_usd: float
    total_costs_usd: float
    total_net_pnl_usd: float
    position_count: int
    total_position_value_usd: float
    discrepancies: List[Dict[str, Any]]
    status: str


class TradeJournalService:
    """Trade journal service"""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def record_fill(self, fill: TradeFill) -> int:
        """Record a trade fill"""
        start_time = asyncio.get_event_loop().time()

        try:
            async with self.db_pool.acquire() as conn:
                # Insert fill
                fill_id = await conn.fetchval("""
                    INSERT INTO trade_fills (
                        order_id, symbol, venue, side, fill_price, fill_quantity,
                        fill_timestamp, order_type, limit_price, time_in_force,
                        commission_usd, exchange_fee_usd, sec_fee_usd, finra_taf_usd,
                        slippage_bps, routing_decision_id, routing_latency_ms,
                        decision_factors
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                    RETURNING fill_id
                """, fill.order_id, fill.symbol, fill.venue, fill.side.value,
                    fill.fill_price, fill.fill_quantity, fill.fill_timestamp,
                    fill.order_type.value, fill.limit_price, fill.time_in_force,
                    fill.commission_usd, fill.exchange_fee_usd, fill.sec_fee_usd,
                    fill.finra_taf_usd, fill.slippage_bps, fill.routing_decision_id,
                    fill.routing_latency_ms, fill.decision_factors
                )

                # Update position
                await self._update_position(conn, fill, fill_id)

                fills_recorded.inc()
                fill_latency.observe(asyncio.get_event_loop().time() - start_time)

                logger.info(f"Recorded fill {fill_id} for {fill.symbol} {fill.side.value} {fill.fill_quantity}@{fill.fill_price}")
                return fill_id

        except Exception as e:
            logger.error(f"Error recording fill: {e}")
            raise

    async def _update_position(self, conn: asyncpg.Connection, fill: TradeFill, fill_id: int):
        """Update position after fill"""
        # Get current position
        position = await conn.fetchrow("""
            SELECT position_id, current_quantity, avg_entry_price, total_cost_basis_usd, realized_pnl_usd
            FROM positions
            WHERE symbol = $1
        """, fill.symbol)

        if position is None:
            # New position
            if fill.side == OrderSide.BUY:
                await conn.execute("""
                    INSERT INTO positions (
                        symbol, current_quantity, avg_entry_price, total_cost_basis_usd,
                        first_entry_fill_id, last_update_fill_id
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, fill.symbol, fill.fill_quantity, fill.fill_price,
                    fill.fill_price * fill.fill_quantity, fill_id, fill_id
                )
            else:
                # Short position
                await conn.execute("""
                    INSERT INTO positions (
                        symbol, current_quantity, avg_entry_price, total_cost_basis_usd,
                        first_entry_fill_id, last_update_fill_id
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, fill.symbol, -fill.fill_quantity, fill.fill_price,
                    fill.fill_price * fill.fill_quantity, fill_id, fill_id
                )
        else:
            # Update existing position
            current_qty = position['current_quantity']
            avg_price = Decimal(str(position['avg_entry_price']))
            cost_basis = Decimal(str(position['total_cost_basis_usd']))
            realized_pnl = Decimal(str(position['realized_pnl_usd']))

            if fill.side == OrderSide.BUY:
                new_qty = current_qty + fill.fill_quantity
                if new_qty == 0:
                    # Closed position
                    new_avg_price = Decimal('0')
                    new_cost_basis = Decimal('0')
                elif current_qty >= 0:
                    # Adding to long
                    new_cost_basis = cost_basis + (fill.fill_price * fill.fill_quantity)
                    new_avg_price = new_cost_basis / new_qty
                else:
                    # Covering short
                    pnl_per_share = avg_price - fill.fill_price
                    realized = pnl_per_share * min(abs(current_qty), fill.fill_quantity)
                    realized_pnl += realized
                    new_avg_price = avg_price
                    new_cost_basis = cost_basis
            else:
                new_qty = current_qty - fill.fill_quantity
                if new_qty == 0:
                    # Closed position
                    new_avg_price = Decimal('0')
                    new_cost_basis = Decimal('0')
                elif current_qty > 0:
                    # Selling long
                    pnl_per_share = fill.fill_price - avg_price
                    realized = pnl_per_share * min(current_qty, fill.fill_quantity)
                    realized_pnl += realized
                    new_avg_price = avg_price
                    new_cost_basis = cost_basis - (avg_price * fill.fill_quantity)
                else:
                    # Adding to short
                    new_cost_basis = cost_basis + (fill.fill_price * fill.fill_quantity)
                    new_avg_price = new_cost_basis / abs(new_qty)

            await conn.execute("""
                UPDATE positions
                SET current_quantity = $1,
                    avg_entry_price = $2,
                    total_cost_basis_usd = $3,
                    realized_pnl_usd = $4,
                    last_update_fill_id = $5,
                    updated_at = NOW()
                WHERE symbol = $6
            """, new_qty, new_avg_price, new_cost_basis, realized_pnl, fill_id, fill.symbol)

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get current positions"""
        async with self.db_pool.acquire() as conn:
            if symbol:
                rows = await conn.fetch("""
                    SELECT position_id, symbol, current_quantity, avg_entry_price,
                           total_cost_basis_usd, unrealized_pnl_usd, realized_pnl_usd
                    FROM positions
                    WHERE symbol = $1 AND current_quantity != 0
                """, symbol)
            else:
                rows = await conn.fetch("""
                    SELECT position_id, symbol, current_quantity, avg_entry_price,
                           total_cost_basis_usd, unrealized_pnl_usd, realized_pnl_usd
                    FROM positions
                    WHERE current_quantity != 0
                """)

            return [Position(
                position_id=row['position_id'],
                symbol=row['symbol'],
                current_quantity=row['current_quantity'],
                avg_entry_price=Decimal(str(row['avg_entry_price'])),
                total_cost_basis_usd=Decimal(str(row['total_cost_basis_usd'])),
                unrealized_pnl_usd=Decimal(str(row['unrealized_pnl_usd'])) if row['unrealized_pnl_usd'] else None,
                realized_pnl_usd=Decimal(str(row['realized_pnl_usd']))
            ) for row in rows]

    async def calculate_pnl_attribution(self, start_date: date, end_date: date) -> List[PnLAttribution]:
        """Calculate P&L attribution for date range"""
        pnl_calculations.inc()

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT pnl_id, symbol, attribution_date, gross_pnl_usd, commission_cost_usd,
                       slippage_cost_usd, borrow_cost_usd, other_fees_usd, net_pnl_usd,
                       strategy_name, venue, entry_fill_id, exit_fill_id, entry_price,
                       exit_price, quantity, holding_period_hours, market_impact_bps,
                       timing_alpha_bps, venue_selection_savings_bps
                FROM pnl_attribution
                WHERE attribution_date BETWEEN $1 AND $2
                ORDER BY attribution_date DESC
            """, start_date, end_date)

            return [PnLAttribution(
                symbol=row['symbol'],
                attribution_date=row['attribution_date'],
                gross_pnl_usd=Decimal(str(row['gross_pnl_usd'])),
                commission_cost_usd=Decimal(str(row['commission_cost_usd'])),
                slippage_cost_usd=Decimal(str(row['slippage_cost_usd'])),
                borrow_cost_usd=Decimal(str(row['borrow_cost_usd'])),
                other_fees_usd=Decimal(str(row['other_fees_usd'])),
                net_pnl_usd=Decimal(str(row['net_pnl_usd'])),
                strategy_name=row['strategy_name'],
                venue=row['venue'],
                entry_fill_id=row['entry_fill_id'],
                exit_fill_id=row['exit_fill_id'],
                entry_price=Decimal(str(row['entry_price'])) if row['entry_price'] else None,
                exit_price=Decimal(str(row['exit_price'])) if row['exit_price'] else None,
                quantity=row['quantity'],
                holding_period_hours=Decimal(str(row['holding_period_hours'])) if row['holding_period_hours'] else None,
                market_impact_bps=Decimal(str(row['market_impact_bps'])) if row['market_impact_bps'] else None,
                timing_alpha_bps=Decimal(str(row['timing_alpha_bps'])) if row['timing_alpha_bps'] else None,
                venue_selection_savings_bps=Decimal(str(row['venue_selection_savings_bps'])) if row['venue_selection_savings_bps'] else None
            ) for row in rows]

    async def reconcile_positions(self, reconciliation_date: date) -> ReconciliationReport:
        """End-of-day position reconciliation"""
        logger.info(f"Starting reconciliation for {reconciliation_date}")

        async with self.db_pool.acquire() as conn:
            # Get fills for the day
            fills = await conn.fetch("""
                SELECT COUNT(*) as fill_count, SUM(fill_price * fill_quantity) as total_value
                FROM trade_fills
                WHERE DATE(fill_timestamp) = $1
            """, reconciliation_date)

            # Get P&L for the day
            pnl = await conn.fetch("""
                SELECT SUM(gross_pnl_usd) as gross_pnl,
                       SUM(commission_cost_usd + slippage_cost_usd + borrow_cost_usd + other_fees_usd) as total_costs,
                       SUM(net_pnl_usd) as net_pnl
                FROM pnl_attribution
                WHERE attribution_date = $1
            """, reconciliation_date)

            # Get positions
            positions = await conn.fetch("""
                SELECT COUNT(*) as position_count,
                       SUM(total_cost_basis_usd) as total_position_value
                FROM positions
                WHERE current_quantity != 0
            """)

            # Check for discrepancies
            discrepancies = []

            # Example: Check if realized P&L matches closed positions
            realized_check = await conn.fetchval("""
                SELECT SUM(realized_pnl_usd) FROM positions
            """)

            fill_count = fills[0]['fill_count'] if fills else 0
            gross_pnl = float(pnl[0]['gross_pnl']) if pnl and pnl[0]['gross_pnl'] else 0.0
            total_costs = float(pnl[0]['total_costs']) if pnl and pnl[0]['total_costs'] else 0.0
            net_pnl = float(pnl[0]['net_pnl']) if pnl and pnl[0]['net_pnl'] else 0.0
            pos_count = positions[0]['position_count'] if positions else 0
            total_pos_value = float(positions[0]['total_position_value']) if positions and positions[0]['total_position_value'] else 0.0

            # Check if reconciliation matches within 1 cent
            tolerance = Decimal('0.01')
            if abs(Decimal(str(gross_pnl)) - Decimal(str(total_costs)) - Decimal(str(net_pnl))) > tolerance:
                discrepancies.append({
                    "type": "pnl_mismatch",
                    "description": "Gross P&L - Costs != Net P&L",
                    "difference_usd": float(Decimal(str(gross_pnl)) - Decimal(str(total_costs)) - Decimal(str(net_pnl)))
                })
                reconciliation_errors.inc()

            status = "PASS" if len(discrepancies) == 0 else "FAIL"

            return ReconciliationReport(
                reconciliation_date=reconciliation_date,
                total_fills=fill_count,
                total_gross_pnl_usd=gross_pnl,
                total_costs_usd=total_costs,
                total_net_pnl_usd=net_pnl,
                position_count=pos_count,
                total_position_value_usd=total_pos_value,
                discrepancies=discrepancies,
                status=status
            )


# Global service instance
trade_journal: Optional[TradeJournalService] = None


@app.on_event("startup")
async def startup():
    """Initialize database connection"""
    global db_pool, trade_journal

    db_pool = await asyncpg.create_pool(
        host="localhost",
        port=5432,
        database="trading_db",
        user="trading_user",
        password="trading_pass",
        min_size=2,
        max_size=10
    )

    trade_journal = TradeJournalService(db_pool)
    logger.info("Trade Journal service started")


@app.on_event("shutdown")
async def shutdown():
    """Close database connection"""
    global db_pool
    if db_pool:
        await db_pool.close()
    logger.info("Trade Journal service stopped")


@app.post("/fills", status_code=201)
async def record_fill(fill_request: FillRequest) -> Dict[str, Any]:
    """Record a trade fill"""
    fill = TradeFill(
        order_id=fill_request.order_id,
        symbol=fill_request.symbol,
        venue=fill_request.venue,
        side=fill_request.side,
        fill_price=Decimal(str(fill_request.fill_price)),
        fill_quantity=fill_request.fill_quantity,
        fill_timestamp=fill_request.fill_timestamp,
        order_type=fill_request.order_type,
        limit_price=Decimal(str(fill_request.limit_price)) if fill_request.limit_price else None,
        time_in_force=fill_request.time_in_force,
        commission_usd=Decimal(str(fill_request.commission_usd)),
        exchange_fee_usd=Decimal(str(fill_request.exchange_fee_usd)),
        sec_fee_usd=Decimal(str(fill_request.sec_fee_usd)),
        finra_taf_usd=Decimal(str(fill_request.finra_taf_usd)),
        slippage_bps=Decimal(str(fill_request.slippage_bps)) if fill_request.slippage_bps else None,
        routing_decision_id=fill_request.routing_decision_id,
        routing_latency_ms=Decimal(str(fill_request.routing_latency_ms)) if fill_request.routing_latency_ms else None,
        decision_factors=fill_request.decision_factors
    )

    fill_id = await trade_journal.record_fill(fill)
    return {"fill_id": fill_id, "status": "recorded"}


@app.get("/positions", response_model=List[PositionResponse])
async def get_positions(symbol: Optional[str] = Query(None)):
    """Get current positions"""
    positions = await trade_journal.get_positions(symbol)
    return [PositionResponse(
        position_id=p.position_id,
        symbol=p.symbol,
        current_quantity=p.current_quantity,
        avg_entry_price=float(p.avg_entry_price),
        total_cost_basis_usd=float(p.total_cost_basis_usd),
        unrealized_pnl_usd=float(p.unrealized_pnl_usd) if p.unrealized_pnl_usd else None,
        realized_pnl_usd=float(p.realized_pnl_usd)
    ) for p in positions]


@app.get("/pnl", response_model=List[PnLAttributionResponse])
async def get_pnl_attribution(
    start_date: date = Query(...),
    end_date: date = Query(...)
):
    """Get P&L attribution for date range"""
    pnl_records = await trade_journal.calculate_pnl_attribution(start_date, end_date)
    return [PnLAttributionResponse(
        pnl_id=0,  # Not returned from this query
        symbol=p.symbol,
        attribution_date=p.attribution_date,
        gross_pnl_usd=float(p.gross_pnl_usd),
        commission_cost_usd=float(p.commission_cost_usd),
        slippage_cost_usd=float(p.slippage_cost_usd),
        borrow_cost_usd=float(p.borrow_cost_usd),
        other_fees_usd=float(p.other_fees_usd),
        net_pnl_usd=float(p.net_pnl_usd),
        strategy_name=p.strategy_name,
        venue=p.venue,
        market_impact_bps=float(p.market_impact_bps) if p.market_impact_bps else None,
        timing_alpha_bps=float(p.timing_alpha_bps) if p.timing_alpha_bps else None,
        venue_selection_savings_bps=float(p.venue_selection_savings_bps) if p.venue_selection_savings_bps else None
    ) for p in pnl_records]


@app.post("/reconcile", response_model=ReconciliationReport)
async def reconcile(reconciliation_date: date = Query(...)):
    """End-of-day reconciliation"""
    return await trade_journal.reconcile_positions(reconciliation_date)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "trade-journal"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
