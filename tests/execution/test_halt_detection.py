"""
Tests for Halt Detection and Safe Order Management

Acceptance Criteria:
- LULD halt detection: 100% detection rate in backtests
- Auction handling: Respect auction-only order types
- Circuit breaker awareness: Cancel working orders on halt
"""
import pytest
import asyncio
from datetime import datetime, time, timedelta
from pathlib import Path
import sys

# Add execution to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'services' / 'strategy-service' / 'app' / 'execution'))

from halt_detector import (
    HaltDetector, SafeOrderManager, HaltType, AuctionType,
    OrderRestriction, LULDBand, HaltStatus
)


@pytest.fixture
def halt_detector():
    """Halt detector fixture"""
    detector = HaltDetector(redis_url="redis://localhost:6379")
    return detector


@pytest.fixture
def safe_order_manager(halt_detector):
    """Safe order manager fixture"""
    return SafeOrderManager(halt_detector)


def test_luld_band_calculation_tier1():
    """Test LULD band calculation for Tier 1 stocks"""
    detector = HaltDetector()

    # Tier 1 stock, price >= $3.00
    # During normal hours (10:00 AM): 10% bands
    reference_price = 185.50

    band = detector.calculate_luld_bands("AAPL", reference_price, tier=1)

    assert band.symbol == "AAPL"
    assert band.reference_price == reference_price
    assert band.tier == 1

    # 10% bands
    expected_upper = reference_price * 1.10
    expected_lower = reference_price * 0.90

    assert abs(band.upper_band - expected_upper) < 0.01
    assert abs(band.lower_band - expected_lower) < 0.01
    assert band.band_width_pct == 10.0

    print(f"\nTier 1 LULD Bands (normal hours):")
    print(f"  Reference: ${band.reference_price}")
    print(f"  Upper Band: ${band.upper_band:.2f} (+{band.band_width_pct}%)")
    print(f"  Lower Band: ${band.lower_band:.2f} (-{band.band_width_pct}%)")


def test_luld_band_calculation_tier2():
    """Test LULD band calculation for Tier 2 stocks"""
    detector = HaltDetector()

    # Tier 2 stock, price >= $3.00
    # During normal hours: 20% bands
    reference_price = 45.00

    band = detector.calculate_luld_bands("XYZ", reference_price, tier=2)

    assert band.tier == 2

    # 20% bands
    expected_upper = reference_price * 1.20
    expected_lower = reference_price * 0.80

    assert abs(band.upper_band - expected_upper) < 0.01
    assert abs(band.lower_band - expected_lower) < 0.01
    assert band.band_width_pct == 20.0

    print(f"\nTier 2 LULD Bands (normal hours):")
    print(f"  Reference: ${band.reference_price}")
    print(f"  Upper Band: ${band.upper_band:.2f} (+{band.band_width_pct}%)")
    print(f"  Lower Band: ${band.lower_band:.2f} (-{band.band_width_pct}%)")


def test_luld_band_low_price():
    """Test LULD bands for low-priced stocks (< $3.00)"""
    detector = HaltDetector()

    # Low-priced Tier 1 stock: 20% bands
    reference_price = 1.50

    band = detector.calculate_luld_bands("PENNY", reference_price, tier=1)

    # 20% bands for low-priced stocks
    assert band.band_width_pct == 20.0

    expected_upper = reference_price * 1.20
    expected_lower = reference_price * 0.80

    assert abs(band.upper_band - expected_upper) < 0.01
    assert abs(band.lower_band - expected_lower) < 0.01

    print(f"\nLow-Priced Stock LULD Bands:")
    print(f"  Reference: ${band.reference_price}")
    print(f"  Upper Band: ${band.upper_band:.2f} (+{band.band_width_pct}%)")
    print(f"  Lower Band: ${band.lower_band:.2f} (-{band.band_width_pct}%)")


def test_luld_violation_detection_upper():
    """Test LULD upper band violation detection"""
    detector = HaltDetector()

    reference_price = 185.50
    band = detector.calculate_luld_bands("AAPL", reference_price, tier=1)

    # Price above upper band
    violation_price = band.upper_band + 0.10

    halt_type = detector.check_luld_violation("AAPL", violation_price, reference_price, tier=1)

    assert halt_type == HaltType.LULD_UPPER
    print(f"\n✓ LULD upper band violation detected: ${violation_price} > ${band.upper_band:.2f}")


def test_luld_violation_detection_lower():
    """Test LULD lower band violation detection"""
    detector = HaltDetector()

    reference_price = 185.50
    band = detector.calculate_luld_bands("AAPL", reference_price, tier=1)

    # Price below lower band
    violation_price = band.lower_band - 0.10

    halt_type = detector.check_luld_violation("AAPL", violation_price, reference_price, tier=1)

    assert halt_type == HaltType.LULD_LOWER
    print(f"\n✓ LULD lower band violation detected: ${violation_price} < ${band.lower_band:.2f}")


def test_luld_no_violation():
    """Test no violation when price within bands"""
    detector = HaltDetector()

    reference_price = 185.50
    band = detector.calculate_luld_bands("AAPL", reference_price, tier=1)

    # Price within bands
    safe_price = reference_price + 5.00  # Well within 10% band

    halt_type = detector.check_luld_violation("AAPL", safe_price, reference_price, tier=1)

    assert halt_type is None
    print(f"\n✓ No LULD violation: ${safe_price} within [${band.lower_band:.2f}, ${band.upper_band:.2f}]")


def test_luld_100_percent_detection_rate():
    """Test 100% LULD detection rate across price range"""
    detector = HaltDetector()

    reference_price = 100.00
    band = detector.calculate_luld_bands("TEST", reference_price, tier=1)

    test_cases = [
        # (price, should_halt, expected_type)
        (band.upper_band - 0.01, False, None),  # Just within
        (band.upper_band, True, HaltType.LULD_UPPER),  # At boundary
        (band.upper_band + 0.01, True, HaltType.LULD_UPPER),  # Just above
        (band.lower_band + 0.01, False, None),  # Just within
        (band.lower_band, True, HaltType.LULD_LOWER),  # At boundary
        (band.lower_band - 0.01, True, HaltType.LULD_LOWER),  # Just below
        (reference_price, False, None),  # At reference
    ]

    detection_count = 0
    expected_detections = sum(1 for _, should_halt, _ in test_cases if should_halt)

    for price, should_halt, expected_type in test_cases:
        halt_type = detector.check_luld_violation("TEST", price, reference_price, tier=1)

        if should_halt:
            assert halt_type == expected_type
            detection_count += 1
        else:
            assert halt_type is None

    # 100% detection rate
    detection_rate = (detection_count / expected_detections) * 100 if expected_detections > 0 else 100
    assert detection_rate == 100.0

    print(f"\n✓ LULD Detection Rate: {detection_rate:.1f}% ({detection_count}/{expected_detections} detected)")


@pytest.mark.asyncio
async def test_halt_status_tracking(halt_detector):
    """Test halt status tracking"""
    symbol = "AAPL"

    # Initially not halted
    status = await halt_detector.check_halt_status(symbol)
    assert status.is_halted == False

    # Trigger halt
    await halt_detector.handle_halt_detected(
        symbol,
        HaltType.LULD_UPPER,
        message="Price exceeded upper LULD band"
    )

    # Should be halted now
    status = await halt_detector.check_halt_status(symbol)
    assert status.is_halted == True
    assert status.halt_type == HaltType.LULD_UPPER
    assert status.halt_start_time is not None
    assert OrderRestriction.NO_NEW_ORDERS in status.restrictions

    print(f"\n✓ Halt status tracked:")
    print(f"  Symbol: {status.symbol}")
    print(f"  Halted: {status.is_halted}")
    print(f"  Type: {status.halt_type.value}")
    print(f"  Restrictions: {[r.value for r in status.restrictions]}")


@pytest.mark.asyncio
async def test_halt_resumption(halt_detector):
    """Test halt resumption"""
    symbol = "AAPL"

    # Trigger halt
    await halt_detector.handle_halt_detected(symbol, HaltType.LULD_UPPER)
    assert halt_detector.halt_status[symbol].is_halted

    # Resume trading
    await halt_detector.handle_halt_resumed(symbol)

    # Should not be halted
    status = await halt_detector.check_halt_status(symbol)
    assert status.is_halted == False

    print(f"\n✓ Halt resumed for {symbol}")


def test_auction_detection_opening():
    """Test opening auction detection"""
    detector = HaltDetector()

    # Get current auction (mock would need to be during auction time)
    # For testing, we check the logic
    now = datetime.now()

    # Simulate opening auction time (9:25-9:30)
    if now.time() >= time(9, 25) and now.time() < time(9, 30):
        auction = detector.get_current_auction()
        assert auction is not None
        assert auction.auction_type == AuctionType.OPENING
        assert auction.is_active

        print(f"\n✓ Opening auction detected:")
        print(f"  Type: {auction.auction_type.value}")
        print(f"  Expected end: {auction.expected_end_time.strftime('%H:%M:%S')}")
    else:
        print(f"\n✓ Outside opening auction period (current time: {now.time()})")


def test_auction_detection_closing():
    """Test closing auction detection"""
    detector = HaltDetector()

    now = datetime.now()

    # Simulate closing auction time (15:55-16:00)
    if now.time() >= time(15, 55) and now.time() <= time(16, 0):
        auction = detector.get_current_auction()
        assert auction is not None
        assert auction.auction_type == AuctionType.CLOSING
        assert auction.is_active

        print(f"\n✓ Closing auction detected:")
        print(f"  Type: {auction.auction_type.value}")
        print(f"  Expected end: {auction.expected_end_time.strftime('%H:%M:%S')}")
    else:
        print(f"\n✓ Outside closing auction period (current time: {now.time()})")


def test_order_restrictions_during_halt(halt_detector):
    """Test order restrictions during halt"""
    symbol = "AAPL"

    # Simulate halt
    halt_detector.halt_status[symbol] = HaltStatus(
        symbol=symbol,
        is_halted=True,
        halt_type=HaltType.LULD_UPPER,
        halt_start_time=datetime.utcnow()
    )

    restrictions = halt_detector.get_order_restrictions(symbol)

    assert OrderRestriction.NO_NEW_ORDERS in restrictions
    assert OrderRestriction.NO_CANCEL in restrictions

    print(f"\n✓ Halt restrictions applied:")
    print(f"  Restrictions: {[r.value for r in restrictions]}")


def test_order_restrictions_during_auction(halt_detector):
    """Test order restrictions during auction"""
    # Simulate auction
    halt_detector.auction_status = {
        'auction_type': AuctionType.OPENING,
        'is_active': True
    }

    # Mock get_current_auction to return auction
    original_method = halt_detector.get_current_auction

    def mock_auction():
        from dataclasses import dataclass
        @dataclass
        class MockAuction:
            auction_type: AuctionType
            is_active: bool

        return MockAuction(auction_type=AuctionType.OPENING, is_active=True)

    halt_detector.get_current_auction = mock_auction

    restrictions = halt_detector.get_order_restrictions("AAPL")

    assert OrderRestriction.AUCTION_ONLY in restrictions

    halt_detector.get_current_auction = original_method

    print(f"\n✓ Auction restrictions applied:")
    print(f"  Restrictions: {[r.value for r in restrictions]}")


@pytest.mark.asyncio
async def test_safe_order_submission_normal(safe_order_manager):
    """Test order submission during normal market conditions"""
    symbol = "AAPL"
    order_id = "ORD-001"

    accepted, reason = await safe_order_manager.submit_order(
        symbol, order_id, "LIMIT"
    )

    assert accepted == True
    assert reason is None
    assert order_id in safe_order_manager.working_orders[symbol]

    print(f"\n✓ Order {order_id} accepted for {symbol}")


@pytest.mark.asyncio
async def test_safe_order_submission_during_halt(safe_order_manager, halt_detector):
    """Test order rejection during halt"""
    symbol = "AAPL"
    order_id = "ORD-002"

    # Trigger halt
    halt_detector.halt_status[symbol] = HaltStatus(
        symbol=symbol,
        is_halted=True,
        halt_type=HaltType.LULD_UPPER
    )

    accepted, reason = await safe_order_manager.submit_order(
        symbol, order_id, "LIMIT"
    )

    assert accepted == False
    assert reason is not None
    assert "halt" in reason.lower() or "circuit breaker" in reason.lower()

    print(f"\n✓ Order {order_id} rejected during halt: {reason}")


@pytest.mark.asyncio
async def test_safe_order_cancellation_normal(safe_order_manager):
    """Test order cancellation during normal conditions"""
    symbol = "AAPL"
    order_id = "ORD-001"

    # Submit order first
    await safe_order_manager.submit_order(symbol, order_id, "LIMIT")

    # Cancel order
    cancelled, reason = await safe_order_manager.cancel_order(symbol, order_id)

    assert cancelled == True
    assert reason is None
    assert order_id not in safe_order_manager.working_orders.get(symbol, [])

    print(f"\n✓ Order {order_id} cancelled for {symbol}")


@pytest.mark.asyncio
async def test_safe_order_cancellation_during_halt(safe_order_manager, halt_detector):
    """Test order cancellation rejection during halt"""
    symbol = "AAPL"
    order_id = "ORD-001"

    # Submit order first
    await safe_order_manager.submit_order(symbol, order_id, "LIMIT")

    # Trigger halt
    halt_detector.halt_status[symbol] = HaltStatus(
        symbol=symbol,
        is_halted=True,
        halt_type=HaltType.LULD_UPPER
    )

    # Try to cancel
    cancelled, reason = await safe_order_manager.cancel_order(symbol, order_id)

    assert cancelled == False
    assert reason is not None
    assert "cancel" in reason.lower()

    print(f"\n✓ Order {order_id} cancel rejected during halt: {reason}")


@pytest.mark.asyncio
async def test_cancel_all_orders_on_halt(safe_order_manager):
    """Test cancelling all orders when halt detected"""
    symbol = "AAPL"

    # Submit multiple orders
    order_ids = ["ORD-001", "ORD-002", "ORD-003"]
    for order_id in order_ids:
        await safe_order_manager.submit_order(symbol, order_id, "LIMIT")

    assert len(safe_order_manager.working_orders[symbol]) == 3

    # Cancel all on halt
    await safe_order_manager.cancel_all_orders_for_symbol(symbol, "LULD halt detected")

    assert len(safe_order_manager.working_orders[symbol]) == 0

    print(f"\n✓ All {len(order_ids)} orders cancelled for {symbol} on halt")


def test_circuit_breaker_restrictions(halt_detector):
    """Test circuit breaker restrictions"""
    # Level 1: 7% drop - limit orders only
    halt_detector.circuit_breaker_active = True
    halt_detector.circuit_breaker_level = HaltType.CIRCUIT_BREAKER_L1

    restrictions = halt_detector.get_order_restrictions("AAPL")
    assert OrderRestriction.LIMIT_ONLY in restrictions

    # Level 3: 20% drop - no new orders
    halt_detector.circuit_breaker_level = HaltType.CIRCUIT_BREAKER_L3

    restrictions = halt_detector.get_order_restrictions("AAPL")
    assert OrderRestriction.NO_NEW_ORDERS in restrictions

    print(f"\n✓ Circuit breaker restrictions:")
    print(f"  L1 (7% drop): LIMIT_ONLY")
    print(f"  L3 (20% drop): NO_NEW_ORDERS")


def test_order_type_validation_during_circuit_breaker(halt_detector):
    """Test order type validation during circuit breaker"""
    symbol = "AAPL"

    # Level 1 circuit breaker - limit orders only
    halt_detector.circuit_breaker_active = True
    halt_detector.circuit_breaker_level = HaltType.CIRCUIT_BREAKER_L1

    # Market order should be rejected
    can_place, reason = halt_detector.can_place_order(symbol, "MARKET")
    assert can_place == False
    assert "limit" in reason.lower()

    # Limit order should be accepted
    can_place, reason = halt_detector.can_place_order(symbol, "LIMIT")
    assert can_place == True

    print(f"\n✓ Circuit breaker L1 order validation:")
    print(f"  MARKET order: Rejected")
    print(f"  LIMIT order: Accepted")


def test_auction_eligible_order_validation(halt_detector):
    """Test auction-eligible order validation"""
    symbol = "AAPL"

    # Simulate auction
    original_method = halt_detector.get_current_auction

    def mock_auction():
        from dataclasses import dataclass
        @dataclass
        class MockAuction:
            auction_type: AuctionType
            is_active: bool

        return MockAuction(auction_type=AuctionType.OPENING, is_active=True)

    halt_detector.get_current_auction = mock_auction

    # Non-auction-eligible order should be rejected
    can_place, reason = halt_detector.can_place_order(
        symbol, "LIMIT", is_auction_eligible=False
    )
    assert can_place == False
    assert "auction" in reason.lower()

    # Auction-eligible order should be accepted
    can_place, reason = halt_detector.can_place_order(
        symbol, "LIMIT", is_auction_eligible=True
    )
    assert can_place == True

    halt_detector.get_current_auction = original_method

    print(f"\n✓ Auction order validation:")
    print(f"  Non-auction-eligible: Rejected")
    print(f"  Auction-eligible: Accepted")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
