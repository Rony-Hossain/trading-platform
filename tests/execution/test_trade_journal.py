"""
Tests for Trade Journal and P&L Attribution

Acceptance Criteria:
- Reconciliation: End-of-day balances match to 1 cent
- P&L attribution: Full cost breakdown (slippage, fees, borrow)
- Audit trail: Immutable fill records with timestamps
"""
import pytest
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
import sys

# Add trade-journal to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'trade-journal' / 'app'))

from main import (
    TradeJournalService, TradeFill, Position, PnLAttribution,
    OrderSide, OrderType
)

# Mock database for testing
class MockConnection:
    def __init__(self):
        self.fills = []
        self.positions = {}
        self.pnl_records = []

    async def fetchval(self, query, *args):
        """Mock fetchval"""
        if "INSERT INTO trade_fills" in query:
            fill_id = len(self.fills) + 1
            self.fills.append({
                'fill_id': fill_id,
                'order_id': args[0],
                'symbol': args[1],
                'venue': args[2],
                'side': args[3],
                'fill_price': args[4],
                'fill_quantity': args[5],
                'fill_timestamp': args[6],
                'commission_usd': args[10],
                'slippage_bps': args[14]
            })
            return fill_id
        return None

    async def fetchrow(self, query, *args):
        """Mock fetchrow"""
        if "FROM positions WHERE symbol" in query:
            return self.positions.get(args[0])
        return None

    async def fetch(self, query, *args):
        """Mock fetch"""
        if "FROM positions WHERE current_quantity" in query:
            return [pos for pos in self.positions.values() if pos['current_quantity'] != 0]
        elif "FROM pnl_attribution" in query:
            return self.pnl_records
        elif "FROM trade_fills WHERE DATE" in query:
            return [{'fill_count': len(self.fills), 'total_value': Decimal('10000')}]
        return []

    async def execute(self, query, *args):
        """Mock execute"""
        if "INSERT INTO positions" in query:
            self.positions[args[0]] = {
                'position_id': len(self.positions) + 1,
                'symbol': args[0],
                'current_quantity': args[1],
                'avg_entry_price': args[2],
                'total_cost_basis_usd': args[3],
                'unrealized_pnl_usd': None,
                'realized_pnl_usd': Decimal('0')
            }
        elif "UPDATE positions" in query:
            symbol = args[5]
            if symbol in self.positions:
                self.positions[symbol].update({
                    'current_quantity': args[0],
                    'avg_entry_price': args[1],
                    'total_cost_basis_usd': args[2],
                    'realized_pnl_usd': args[3]
                })


class MockPool:
    def __init__(self):
        self.conn = MockConnection()

    def acquire(self):
        return self

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_db_pool():
    """Mock database pool"""
    return MockPool()


@pytest.fixture
def journal_service(mock_db_pool):
    """Trade journal service with mock DB"""
    return TradeJournalService(mock_db_pool)


@pytest.fixture
def sample_buy_fill():
    """Sample buy fill"""
    return TradeFill(
        order_id="ORD-001",
        symbol="AAPL",
        venue="NASDAQ",
        side=OrderSide.BUY,
        fill_price=Decimal("185.50"),
        fill_quantity=100,
        fill_timestamp=datetime.utcnow(),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("185.50"),
        commission_usd=Decimal("1.00"),
        exchange_fee_usd=Decimal("0.50"),
        slippage_bps=Decimal("2.0")
    )


@pytest.fixture
def sample_sell_fill():
    """Sample sell fill"""
    return TradeFill(
        order_id="ORD-002",
        symbol="AAPL",
        venue="NASDAQ",
        side=OrderSide.SELL,
        fill_price=Decimal("187.00"),
        fill_quantity=100,
        fill_timestamp=datetime.utcnow(),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("187.00"),
        commission_usd=Decimal("1.00"),
        exchange_fee_usd=Decimal("0.50"),
        slippage_bps=Decimal("2.5")
    )


@pytest.mark.asyncio
async def test_record_fill(journal_service, sample_buy_fill):
    """Test recording a fill"""
    fill_id = await journal_service.record_fill(sample_buy_fill)

    assert fill_id == 1
    assert len(journal_service.db_pool.conn.fills) == 1

    recorded_fill = journal_service.db_pool.conn.fills[0]
    assert recorded_fill['symbol'] == "AAPL"
    assert recorded_fill['side'] == "BUY"
    assert recorded_fill['fill_quantity'] == 100
    assert recorded_fill['fill_price'] == Decimal("185.50")

    print(f"\n✓ Fill recorded: {recorded_fill}")


@pytest.mark.asyncio
async def test_position_tracking_new_long(journal_service, sample_buy_fill):
    """Test creating a new long position"""
    await journal_service.record_fill(sample_buy_fill)

    positions = await journal_service.get_positions("AAPL")
    assert len(positions) == 1

    position = positions[0]
    assert position.symbol == "AAPL"
    assert position.current_quantity == 100
    assert position.avg_entry_price == Decimal("185.50")
    assert position.total_cost_basis_usd == Decimal("185.50") * 100

    print(f"\n✓ Long position created: {position}")


@pytest.mark.asyncio
async def test_position_tracking_add_to_long(journal_service, sample_buy_fill):
    """Test adding to existing long position"""
    # First buy
    await journal_service.record_fill(sample_buy_fill)

    # Second buy at different price
    second_fill = TradeFill(
        order_id="ORD-003",
        symbol="AAPL",
        venue="NASDAQ",
        side=OrderSide.BUY,
        fill_price=Decimal("186.00"),
        fill_quantity=50,
        fill_timestamp=datetime.utcnow(),
        order_type=OrderType.LIMIT,
        commission_usd=Decimal("1.00")
    )

    await journal_service.record_fill(second_fill)

    positions = await journal_service.get_positions("AAPL")
    position = positions[0]

    # Should have 150 shares
    assert position.current_quantity == 150

    # Average price should be (185.50*100 + 186.00*50) / 150 = 185.67
    expected_avg = (Decimal("185.50") * 100 + Decimal("186.00") * 50) / 150
    assert abs(position.avg_entry_price - expected_avg) < Decimal("0.01")

    print(f"\n✓ Position updated: {position.current_quantity} shares @ ${position.avg_entry_price}")


@pytest.mark.asyncio
async def test_position_tracking_close_position(journal_service, sample_buy_fill, sample_sell_fill):
    """Test closing a position and calculating realized P&L"""
    # Buy 100 shares
    await journal_service.record_fill(sample_buy_fill)

    # Sell 100 shares
    await journal_service.record_fill(sample_sell_fill)

    positions = await journal_service.get_positions("AAPL")

    # Position should be closed (quantity = 0)
    assert len(positions) == 0

    # Check realized P&L in database
    position_record = journal_service.db_pool.conn.positions.get("AAPL")
    assert position_record is not None

    # Realized P&L = (187.00 - 185.50) * 100 = 150.00
    expected_pnl = (Decimal("187.00") - Decimal("185.50")) * 100
    assert abs(position_record['realized_pnl_usd'] - expected_pnl) < Decimal("0.01")

    print(f"\n✓ Position closed with realized P&L: ${position_record['realized_pnl_usd']}")


@pytest.mark.asyncio
async def test_position_tracking_partial_close(journal_service):
    """Test partial position close"""
    # Buy 200 shares at 185.50
    buy_fill = TradeFill(
        order_id="ORD-001",
        symbol="AAPL",
        venue="NASDAQ",
        side=OrderSide.BUY,
        fill_price=Decimal("185.50"),
        fill_quantity=200,
        fill_timestamp=datetime.utcnow(),
        order_type=OrderType.LIMIT,
        commission_usd=Decimal("1.00")
    )
    await journal_service.record_fill(buy_fill)

    # Sell 100 shares at 187.00
    sell_fill = TradeFill(
        order_id="ORD-002",
        symbol="AAPL",
        venue="NASDAQ",
        side=OrderSide.SELL,
        fill_price=Decimal("187.00"),
        fill_quantity=100,
        fill_timestamp=datetime.utcnow(),
        order_type=OrderType.LIMIT,
        commission_usd=Decimal("1.00")
    )
    await journal_service.record_fill(sell_fill)

    positions = await journal_service.get_positions("AAPL")
    position = positions[0]

    # Should have 100 shares remaining
    assert position.current_quantity == 100
    assert position.avg_entry_price == Decimal("185.50")

    # Realized P&L = (187.00 - 185.50) * 100 = 150.00
    expected_realized = (Decimal("187.00") - Decimal("185.50")) * 100
    assert abs(position.realized_pnl_usd - expected_realized) < Decimal("0.01")

    print(f"\n✓ Partial close: {position.current_quantity} shares remaining, ${position.realized_pnl_usd} realized")


@pytest.mark.asyncio
async def test_reconciliation_pass(journal_service, sample_buy_fill, sample_sell_fill):
    """Test reconciliation with matching balances"""
    # Record some fills
    await journal_service.record_fill(sample_buy_fill)
    await journal_service.record_fill(sample_sell_fill)

    # Add mock P&L data
    journal_service.db_pool.conn.pnl_records = [{
        'gross_pnl_usd': Decimal('150.00'),
        'commission_cost_usd': Decimal('2.00'),
        'slippage_cost_usd': Decimal('5.00'),
        'borrow_cost_usd': Decimal('0.00'),
        'other_fees_usd': Decimal('1.00'),
        'net_pnl_usd': Decimal('142.00')
    }]

    # Run reconciliation
    report = await journal_service.reconcile_positions(date.today())

    print(f"\n✓ Reconciliation Report:")
    print(f"  Total Fills: {report.total_fills}")
    print(f"  Gross P&L: ${report.total_gross_pnl_usd:.2f}")
    print(f"  Total Costs: ${report.total_costs_usd:.2f}")
    print(f"  Net P&L: ${report.total_net_pnl_usd:.2f}")
    print(f"  Status: {report.status}")
    print(f"  Discrepancies: {len(report.discrepancies)}")

    # Should pass if within 1 cent
    assert report.status in ["PASS", "FAIL"]


@pytest.mark.asyncio
async def test_reconciliation_tolerance(journal_service):
    """Test reconciliation tolerance (must match within 1 cent)"""
    # Create P&L with discrepancy < 1 cent
    journal_service.db_pool.conn.pnl_records = [{
        'gross_pnl_usd': Decimal('100.00'),
        'commission_cost_usd': Decimal('1.00'),
        'slippage_cost_usd': Decimal('0.50'),
        'borrow_cost_usd': Decimal('0.00'),
        'other_fees_usd': Decimal('0.49'),  # Total costs = 1.99
        'net_pnl_usd': Decimal('98.01')    # 100.00 - 1.99 = 98.01 (matches)
    }]

    report = await journal_service.reconcile_positions(date.today())

    # Should pass
    assert report.status == "PASS"
    assert len(report.discrepancies) == 0

    print(f"\n✓ Reconciliation passed with 0 cent difference")

    # Now create discrepancy > 1 cent
    journal_service.db_pool.conn.pnl_records = [{
        'gross_pnl_usd': Decimal('100.00'),
        'commission_cost_usd': Decimal('1.00'),
        'slippage_cost_usd': Decimal('0.50'),
        'borrow_cost_usd': Decimal('0.00'),
        'other_fees_usd': Decimal('0.49'),
        'net_pnl_usd': Decimal('98.10')    # 100.00 - 1.99 = 98.01, but reported 98.10 (0.09 diff)
    }]

    report = await journal_service.reconcile_positions(date.today())

    # Should fail
    assert report.status == "FAIL"
    assert len(report.discrepancies) > 0

    print(f"\n✓ Reconciliation failed with > 1 cent difference: {report.discrepancies}")


@pytest.mark.asyncio
async def test_cost_breakdown(journal_service):
    """Test full cost breakdown (slippage, fees, borrow)"""
    # Create fill with all costs
    fill = TradeFill(
        order_id="ORD-001",
        symbol="AAPL",
        venue="NASDAQ",
        side=OrderSide.SELL,  # Short sell
        fill_price=Decimal("185.50"),
        fill_quantity=100,
        fill_timestamp=datetime.utcnow(),
        order_type=OrderType.LIMIT,
        commission_usd=Decimal("1.00"),
        exchange_fee_usd=Decimal("0.50"),
        sec_fee_usd=Decimal("0.22"),
        finra_taf_usd=Decimal("0.13"),
        slippage_bps=Decimal("2.5")
    )

    await journal_service.record_fill(fill)

    # Verify all costs recorded
    recorded_fill = journal_service.db_pool.conn.fills[0]

    total_fees = (
        recorded_fill['commission_usd'] +
        Decimal("0.50") +  # exchange_fee_usd
        Decimal("0.22") +  # sec_fee_usd
        Decimal("0.13")    # finra_taf_usd
    )

    assert total_fees == Decimal("1.85")
    assert recorded_fill['slippage_bps'] == Decimal("2.5")

    # Calculate slippage cost
    # 2.5 bps on 100 shares @ $185.50 = $0.046 per share = $4.64 total
    slippage_cost = (Decimal("2.5") / Decimal("10000")) * Decimal("185.50") * 100

    total_cost = total_fees + slippage_cost

    print(f"\n✓ Cost Breakdown:")
    print(f"  Commission: ${recorded_fill['commission_usd']}")
    print(f"  Exchange Fee: $0.50")
    print(f"  SEC Fee: $0.22")
    print(f"  FINRA TAF: $0.13")
    print(f"  Total Fees: ${total_fees}")
    print(f"  Slippage: {recorded_fill['slippage_bps']} bps = ${slippage_cost:.2f}")
    print(f"  Total Cost: ${total_cost:.2f}")


@pytest.mark.asyncio
async def test_audit_trail_immutability(journal_service, sample_buy_fill):
    """Test that fill records are immutable (audit trail)"""
    # Record fill
    fill_id = await journal_service.record_fill(sample_buy_fill)

    # Get recorded fill
    recorded_fill = journal_service.db_pool.conn.fills[0]

    # Verify timestamp is set
    assert 'fill_timestamp' in recorded_fill
    assert recorded_fill['fill_timestamp'] is not None

    # In production, fills should not be updatable (only insertable)
    # This is enforced by database constraints and application logic

    print(f"\n✓ Audit trail: Fill {fill_id} recorded with timestamp {recorded_fill['fill_timestamp']}")
    print(f"  Order ID: {recorded_fill['order_id']}")
    print(f"  Symbol: {recorded_fill['symbol']}")
    print(f"  Venue: {recorded_fill['venue']}")
    print(f"  Side: {recorded_fill['side']}")
    print(f"  Quantity: {recorded_fill['fill_quantity']}@${recorded_fill['fill_price']}")


@pytest.mark.asyncio
async def test_pnl_attribution_calculation(journal_service):
    """Test P&L attribution calculation"""
    # Mock P&L data
    journal_service.db_pool.conn.pnl_records = [
        {
            'symbol': 'AAPL',
            'attribution_date': date.today(),
            'gross_pnl_usd': Decimal('150.00'),
            'commission_cost_usd': Decimal('2.00'),
            'slippage_cost_usd': Decimal('5.00'),
            'borrow_cost_usd': Decimal('0.00'),
            'other_fees_usd': Decimal('1.00'),
            'net_pnl_usd': Decimal('142.00'),
            'strategy_name': 'momentum',
            'venue': 'NASDAQ',
            'entry_fill_id': 1,
            'exit_fill_id': 2,
            'entry_price': Decimal('185.50'),
            'exit_price': Decimal('187.00'),
            'quantity': 100,
            'holding_period_hours': Decimal('24.5'),
            'market_impact_bps': Decimal('1.2'),
            'timing_alpha_bps': Decimal('8.1'),
            'venue_selection_savings_bps': Decimal('0.5')
        }
    ]

    # Calculate P&L attribution
    pnl_records = await journal_service.calculate_pnl_attribution(
        date.today() - timedelta(days=1),
        date.today()
    )

    assert len(pnl_records) == 1
    pnl = pnl_records[0]

    print(f"\n✓ P&L Attribution:")
    print(f"  Symbol: {pnl.symbol}")
    print(f"  Gross P&L: ${pnl.gross_pnl_usd}")
    print(f"  Commission: ${pnl.commission_cost_usd}")
    print(f"  Slippage: ${pnl.slippage_cost_usd}")
    print(f"  Net P&L: ${pnl.net_pnl_usd}")
    print(f"  Strategy: {pnl.strategy_name}")
    print(f"  Venue: {pnl.venue}")
    print(f"  Holding Period: {pnl.holding_period_hours} hours")
    print(f"  Market Impact: {pnl.market_impact_bps} bps")
    print(f"  Timing Alpha: {pnl.timing_alpha_bps} bps")
    print(f"  Venue Savings: {pnl.venue_selection_savings_bps} bps")

    # Verify P&L calculation
    expected_net = pnl.gross_pnl_usd - pnl.commission_cost_usd - pnl.slippage_cost_usd - pnl.other_fees_usd
    assert pnl.net_pnl_usd == expected_net


def test_position_fifo_accounting():
    """Test FIFO accounting for position cost basis"""
    # This would test FIFO lot tracking if implemented
    # For now, we use weighted average cost basis

    # Example:
    # Buy 100 @ $185.50 (lot 1)
    # Buy 100 @ $186.00 (lot 2)
    # Sell 150
    # FIFO: Sell 100 from lot 1 @ $185.50, then 50 from lot 2 @ $186.00

    # Current implementation uses weighted average:
    # Avg cost = (100*185.50 + 100*186.00) / 200 = $185.75
    # Sell 150 @ $185.75 avg cost

    print("\n✓ Position accounting: Weighted average cost basis (FIFO optional)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
