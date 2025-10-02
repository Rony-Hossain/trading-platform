"""
Tests for Halt / LULD Handling

Comprehensive test suite ensuring zero entries executed during halts
and proper reopen handling with gap pricing protection.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from execution.venue_rules import (
    VenueRuleEngine, HaltMonitor, LULDCalculator, ReopenAnalyzer,
    HaltStatus, HaltReason, VenueType, HaltInfo, LULDBand, ReopenAnalysis,
    validate_trade_order, get_halt_status, simulate_trading_halt, simulate_trading_reopen
)


class TestHaltMonitor:
    """Test halt status monitoring."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = HaltMonitor(update_interval=1)
    
    def test_add_symbol(self):
        """Test adding symbol to monitoring."""
        self.monitor.add_symbol("AAPL", VenueType.NASDAQ)
        
        assert "AAPL" in self.monitor.halt_status
        halt_info = self.monitor.halt_status["AAPL"]
        assert halt_info.symbol == "AAPL"
        assert halt_info.status == HaltStatus.NORMAL
        assert halt_info.venue == VenueType.NASDAQ
    
    def test_trading_allowed_normal(self):
        """Test trading allowed during normal conditions."""
        self.monitor.add_symbol("AAPL")
        
        allowed, reason = self.monitor.is_trading_allowed("AAPL")
        assert allowed is True
        assert reason is None
    
    def test_trading_blocked_during_halt(self):
        """Test trading blocked during halt."""
        self.monitor.add_symbol("AAPL")
        self.monitor.simulate_halt("AAPL", HaltReason.VOLATILITY)
        
        allowed, reason = self.monitor.is_trading_allowed("AAPL")
        assert allowed is False
        assert "Trading blocked" in reason
        assert "halted" in reason
    
    def test_trading_blocked_during_luld(self):
        """Test trading blocked during LULD pause."""
        self.monitor.add_symbol("AAPL")
        
        # Simulate LULD limit up
        halt_info = self.monitor.halt_status["AAPL"]
        halt_info.status = HaltStatus.LIMIT_UP
        halt_info.reason = HaltReason.LULD_PAUSE
        
        allowed, reason = self.monitor.is_trading_allowed("AAPL")
        assert allowed is False
        assert "limit_up" in reason
    
    def test_simulate_halt_and_reopen(self):
        """Test halt simulation and reopen."""
        self.monitor.add_symbol("AAPL")
        
        # Simulate halt
        self.monitor.simulate_halt("AAPL", HaltReason.NEWS_PENDING, 10)
        halt_info = self.monitor.get_halt_status("AAPL")
        
        assert halt_info.status == HaltStatus.HALTED
        assert halt_info.reason == HaltReason.NEWS_PENDING
        assert halt_info.halt_start is not None
        assert halt_info.expected_reopen is not None
        assert halt_info.is_trading_blocked is True
        
        # Simulate reopen
        self.monitor.simulate_reopen("AAPL", 105.50)
        halt_info = self.monitor.get_halt_status("AAPL")
        
        assert halt_info.status == HaltStatus.REOPENED
        assert halt_info.reopen_price == 105.50
        assert halt_info.is_trading_blocked is False
    
    def test_no_halt_status_blocks_trading(self):
        """Test that missing halt status blocks trading."""
        allowed, reason = self.monitor.is_trading_allowed("UNKNOWN")
        assert allowed is False
        assert "No halt status available" in reason


class TestLULDCalculator:
    """Test LULD band calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = LULDCalculator()
    
    def test_tier1_above_3_dollars(self):
        """Test Tier 1 LULD bands for stocks >= $3."""
        band = self.calculator.calculate_luld_bands("AAPL", 100.0, tier=1)
        
        assert band.symbol == "AAPL"
        assert band.reference_price == 100.0
        assert band.tier == 1
        assert band.band_percentage == 0.05  # 5%
        assert band.upper_band == 105.0
        assert band.lower_band == 95.0
        assert band.is_within_bands(100.0) is True
        assert band.is_within_bands(106.0) is False
        assert band.is_within_bands(94.0) is False
    
    def test_tier2_above_3_dollars(self):
        """Test Tier 2 LULD bands for stocks >= $3."""
        band = self.calculator.calculate_luld_bands("TSLA", 200.0, tier=2)
        
        assert band.tier == 2
        assert band.band_percentage == 0.10  # 10%
        assert band.upper_band == 220.0
        assert band.lower_band == 180.0
    
    def test_below_3_dollars(self):
        """Test LULD bands for stocks < $3."""
        band = self.calculator.calculate_luld_bands("PENNY", 2.50, tier=1)
        
        assert band.band_percentage == 0.20  # 20%
        assert band.upper_band == 3.0
        assert band.lower_band == 2.0
    
    def test_below_75_cents(self):
        """Test LULD bands for stocks < $0.75."""
        band = self.calculator.calculate_luld_bands("MICRO", 0.50, tier=1)
        
        assert band.band_percentage == 0.75  # 75%
        assert band.upper_band == 0.875
        assert band.lower_band == 0.125
    
    def test_band_distance_calculation(self):
        """Test LULD band distance calculations."""
        band = self.calculator.calculate_luld_bands("TEST", 100.0, tier=1)
        
        # Test price at upper band
        distance = band.distance_to_band(105.0)
        assert distance['to_upper'] == 0.0
        assert distance['to_lower'] == 10.0
        
        # Test price at lower band
        distance = band.distance_to_band(95.0)
        assert distance['to_upper'] == 10.0
        assert distance['to_lower'] == 0.0
        
        # Test basis points calculation
        distance = band.distance_to_band(102.0)
        assert abs(distance['to_upper_bps'] - 294.12) < 1  # Approximately 2.94%


class TestReopenAnalyzer:
    """Test post-halt reopen analysis."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = ReopenAnalyzer()
    
    def test_small_gap_analysis(self):
        """Test analysis for small price gaps."""
        analysis = self.analyzer.analyze_reopen(
            "AAPL", pre_halt_price=100.0, reopen_price=101.0, volume=1000
        )
        
        assert analysis.symbol == "AAPL"
        assert analysis.gap_percentage == 1.0
        assert analysis.gap_bps == 100
        assert analysis.has_significant_gap is False
        assert analysis.risk_level == "low"
        assert analysis.should_delay_entry is False
    
    def test_large_gap_analysis(self):
        """Test analysis for large price gaps."""
        analysis = self.analyzer.analyze_reopen(
            "VOLATILE", pre_halt_price=50.0, reopen_price=60.0, volume=10000
        )
        
        assert analysis.gap_percentage == 20.0
        assert analysis.gap_bps == 2000
        assert analysis.has_significant_gap is True
        assert analysis.risk_level == "extreme"
        assert analysis.should_delay_entry is True
        assert analysis.recommended_entry_delay.total_seconds() > 600  # > 10 minutes
    
    def test_negative_gap_analysis(self):
        """Test analysis for negative price gaps."""
        analysis = self.analyzer.analyze_reopen(
            "DOWN", pre_halt_price=100.0, reopen_price=90.0, volume=5000
        )
        
        assert analysis.gap_percentage == -10.0
        assert analysis.gap_bps == -1000
        assert analysis.has_significant_gap is True
        assert analysis.risk_level == "high"
        assert analysis.should_delay_entry is True
    
    def test_risk_level_assessment(self):
        """Test risk level assessment logic."""
        # Low risk
        low_risk = self.analyzer.analyze_reopen("TEST", 100.0, 101.0, 1000)
        assert low_risk.risk_level == "low"
        
        # Medium risk  
        med_risk = self.analyzer.analyze_reopen("TEST", 100.0, 106.0, 1000)
        assert med_risk.risk_level == "medium"
        
        # High risk
        high_risk = self.analyzer.analyze_reopen("TEST", 100.0, 112.0, 1000)
        assert high_risk.risk_level == "high"
        
        # Extreme risk
        extreme_risk = self.analyzer.analyze_reopen("TEST", 100.0, 125.0, 1000)
        assert extreme_risk.risk_level == "extreme"
    
    def test_entry_delay_calculation(self):
        """Test entry delay calculation."""
        # Low risk - 30 seconds
        low_analysis = self.analyzer.analyze_reopen("TEST", 100.0, 101.0)
        assert low_analysis.recommended_entry_delay.total_seconds() == 30
        
        # Medium risk - 2 minutes
        med_analysis = self.analyzer.analyze_reopen("TEST", 100.0, 106.0)
        assert med_analysis.recommended_entry_delay.total_seconds() == 120
        
        # High risk - 5 minutes
        high_analysis = self.analyzer.analyze_reopen("TEST", 100.0, 112.0)
        assert high_analysis.recommended_entry_delay.total_seconds() == 300
        
        # Extreme risk with large gap - >15 minutes
        extreme_analysis = self.analyzer.analyze_reopen("TEST", 100.0, 150.0)
        assert extreme_analysis.recommended_entry_delay.total_seconds() > 900


class TestVenueRuleEngine:
    """Test main venue rule engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = VenueRuleEngine()
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test engine initialization."""
        await self.engine.initialize(["AAPL", "GOOGL", "MSFT"])
        
        # Check that symbols are monitored
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            halt_info = self.engine.halt_monitor.get_halt_status(symbol)
            assert halt_info is not None
            assert halt_info.status == HaltStatus.NORMAL
    
    @pytest.mark.asyncio
    async def test_order_validation_normal_conditions(self):
        """Test order validation under normal conditions."""
        await self.engine.initialize(["AAPL"])
        
        allowed, reason = await self.engine.validate_order("AAPL", "buy", 100, 150.0, "limit")
        assert allowed is True
        assert reason is None
    
    @pytest.mark.asyncio
    async def test_order_validation_during_halt(self):
        """Test order validation during halt - CRITICAL TEST."""
        await self.engine.initialize(["AAPL"])
        
        # Simulate halt
        self.engine.halt_monitor.simulate_halt("AAPL", HaltReason.VOLATILITY)
        
        # ALL orders should be blocked during halt
        market_allowed, market_reason = await self.engine.validate_order("AAPL", "buy", 100)
        limit_allowed, limit_reason = await self.engine.validate_order("AAPL", "sell", 50, 149.0, "limit")
        
        assert market_allowed is False
        assert limit_allowed is False
        assert "Trading blocked" in market_reason
        assert "halted" in market_reason
        assert "Trading blocked" in limit_reason
    
    @pytest.mark.asyncio
    async def test_luld_compliance_validation(self):
        """Test LULD compliance validation."""
        await self.engine.initialize(["AAPL"])
        
        # Add LULD band
        luld_band = LULDBand(
            timestamp=datetime.now(),
            symbol="AAPL",
            reference_price=150.0,
            upper_band=157.5,  # 5% above
            lower_band=142.5,  # 5% below
            band_percentage=0.05,
            tier=1
        )
        self.engine.halt_monitor.luld_bands["AAPL"] = luld_band
        
        # Valid price within bands
        allowed, reason = await self.engine.validate_order("AAPL", "buy", 100, 150.0, "limit")
        assert allowed is True
        
        # Invalid price above upper band
        above_allowed, above_reason = await self.engine.validate_order("AAPL", "buy", 100, 158.0, "limit")
        assert above_allowed is False
        assert "exceeds upper LULD band" in above_reason
        
        # Invalid price below lower band
        below_allowed, below_reason = await self.engine.validate_order("AAPL", "sell", 100, 142.0, "limit")
        assert below_allowed is False
        assert "below lower LULD band" in below_reason
    
    @pytest.mark.asyncio
    async def test_reopen_delay_validation(self):
        """Test post-reopen delay validation."""
        await self.engine.initialize(["AAPL"])
        
        # Simulate halt and reopen with large gap
        self.engine.halt_monitor.simulate_halt("AAPL", HaltReason.VOLATILITY)
        halt_info = self.engine.halt_monitor.get_halt_status("AAPL")
        halt_info.pre_halt_price = 100.0
        
        # Simulate reopen with 15% gap (extreme risk)
        self.engine.halt_monitor.simulate_reopen("AAPL", 115.0)
        
        # Orders should be blocked due to reopen delay
        allowed, reason = await self.engine.validate_order("AAPL", "buy", 100)
        assert allowed is False
        assert "Post-reopen delay active" in reason
        assert "extreme risk" in reason
    
    @pytest.mark.asyncio
    async def test_compliance_logging(self):
        """Test compliance logging functionality."""
        await self.engine.initialize(["AAPL"])
        
        # Generate some events
        await self.engine.validate_order("AAPL", "buy", 100)  # Should pass
        
        self.engine.halt_monitor.simulate_halt("AAPL", HaltReason.VOLATILITY)
        await self.engine.validate_order("AAPL", "sell", 50)  # Should be blocked
        
        # Check compliance report
        report = self.engine.get_compliance_report("AAPL", hours=1)
        
        assert report['symbol_filter'] == "AAPL"
        assert report['total_events'] >= 2
        assert 'ORDER_VALIDATED' in report['event_breakdown']
        assert 'ORDER_BLOCKED' in report['event_breakdown']
        assert report['blocked_orders_count'] >= 1
        assert report['compliance_status'] == 'REVIEW_REQUIRED'  # Due to blocked order
    
    def test_blocked_order_audit_trail(self):
        """Test that blocked orders are properly logged for audit."""
        # This is critical for proving zero entries during halts
        self.engine._log_blocked_order("AAPL", "buy", 1000, 150.0, "Halt active")
        
        assert len(self.engine.blocked_orders) >= 1
        blocked_order = self.engine.blocked_orders[-1]
        
        assert blocked_order['symbol'] == "AAPL"
        assert blocked_order['side'] == "buy"
        assert blocked_order['quantity'] == 1000
        assert blocked_order['price'] == 150.0
        assert blocked_order['reason'] == "Halt active"
        assert blocked_order['compliance_status'] == 'BLOCKED'


class TestGlobalFunctions:
    """Test global utility functions."""
    
    @pytest.mark.asyncio
    async def test_validate_trade_order_function(self):
        """Test global validate_trade_order function."""
        # Mock the global venue_rules instance
        from execution.venue_rules import venue_rules
        venue_rules.validate_order = AsyncMock(return_value=(True, None))
        
        allowed, reason = await validate_trade_order("AAPL", "buy", 100, 150.0, "limit")
        assert allowed is True
        assert reason is None
        
        venue_rules.validate_order.assert_called_once_with("AAPL", "buy", 100, 150.0, "limit")
    
    def test_get_halt_status_function(self):
        """Test global get_halt_status function."""
        from execution.venue_rules import venue_rules
        mock_halt_info = HaltInfo(
            timestamp=datetime.now(),
            symbol="AAPL",
            status=HaltStatus.NORMAL,
            reason=HaltReason.UNKNOWN,
            venue=VenueType.NASDAQ
        )
        venue_rules.halt_monitor.get_halt_status = Mock(return_value=mock_halt_info)
        
        halt_info = get_halt_status("AAPL")
        assert halt_info == mock_halt_info
        
        venue_rules.halt_monitor.get_halt_status.assert_called_once_with("AAPL")
    
    def test_simulate_functions(self):
        """Test simulation functions."""
        from execution.venue_rules import venue_rules
        venue_rules.halt_monitor.simulate_halt = Mock()
        venue_rules.halt_monitor.simulate_reopen = Mock()
        
        # Test halt simulation
        simulate_trading_halt("AAPL", HaltReason.NEWS_PENDING, 20)
        venue_rules.halt_monitor.simulate_halt.assert_called_once_with("AAPL", HaltReason.NEWS_PENDING, 20)
        
        # Test reopen simulation
        simulate_trading_reopen("AAPL", 155.0)
        venue_rules.halt_monitor.simulate_reopen.assert_called_once_with("AAPL", 155.0)


class TestHaltComplianceIntegration:
    """Integration tests for halt compliance - the critical acceptance criteria."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = VenueRuleEngine()
    
    @pytest.mark.asyncio
    async def test_zero_entries_during_halt_guarantee(self):
        """
        CRITICAL TEST: Guarantee zero entries executed during halts.
        This test validates the primary acceptance criteria.
        """
        await self.engine.initialize(["AAPL", "GOOGL", "MSFT"])
        
        # Simulate halts on all symbols
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            self.engine.halt_monitor.simulate_halt(symbol, HaltReason.VOLATILITY)
        
        # Attempt various order types - ALL should be blocked
        test_orders = [
            ("AAPL", "buy", 100, None, "market"),
            ("AAPL", "sell", 50, 149.0, "limit"),
            ("GOOGL", "buy", 25, 2500.0, "limit"),
            ("GOOGL", "sell", 10, None, "market"),
            ("MSFT", "buy", 200, 300.0, "limit"),
            ("MSFT", "sell", 150, None, "market"),
        ]
        
        blocked_count = 0
        for symbol, side, qty, price, order_type in test_orders:
            allowed, reason = await self.engine.validate_order(symbol, side, qty, price, order_type)
            
            # CRITICAL: Every order must be blocked
            assert allowed is False, f"Order {symbol} {side} {qty} should be blocked during halt"
            assert "Trading blocked" in reason, f"Blocking reason should mention trading blocked: {reason}"
            blocked_count += 1
        
        # Verify all orders were blocked
        assert blocked_count == len(test_orders)
        
        # Verify audit trail
        report = self.engine.get_compliance_report(hours=1)
        assert report['blocked_orders_count'] == len(test_orders)
        assert report['event_breakdown']['ORDER_BLOCKED'] == len(test_orders)
        
        # Critical compliance check
        assert report['compliance_status'] == 'REVIEW_REQUIRED'  # Blocked orders require review
        
        print(f"✅ COMPLIANCE VERIFIED: {blocked_count} orders blocked during halt conditions")
    
    @pytest.mark.asyncio 
    async def test_reopen_with_gap_protection(self):
        """Test that large gaps trigger appropriate delays."""
        await self.engine.initialize(["GAPPY"])
        
        # Simulate halt
        self.engine.halt_monitor.simulate_halt("GAPPY", HaltReason.NEWS_PENDING)
        halt_info = self.engine.halt_monitor.get_halt_status("GAPPY")
        halt_info.pre_halt_price = 100.0
        
        # Simulate reopen with 25% gap (extreme)
        self.engine.halt_monitor.simulate_reopen("GAPPY", 125.0)
        
        # Orders should be blocked due to gap protection
        allowed, reason = await self.engine.validate_order("GAPPY", "buy", 100)
        assert allowed is False
        assert "Post-reopen delay active" in reason
        assert "extreme risk" in reason
        assert "2500bps gap" in reason  # 25% = 2500bps
    
    @pytest.mark.asyncio
    async def test_luld_pause_blocking(self):
        """Test that LULD pauses block trading."""
        await self.engine.initialize(["LULD"])
        
        # Set LULD limit up status
        halt_info = self.engine.halt_monitor.get_halt_status("LULD")
        halt_info.status = HaltStatus.LIMIT_UP
        halt_info.reason = HaltReason.LULD_PAUSE
        
        # All orders should be blocked
        buy_allowed, buy_reason = await self.engine.validate_order("LULD", "buy", 100)
        sell_allowed, sell_reason = await self.engine.validate_order("LULD", "sell", 100)
        
        assert buy_allowed is False
        assert sell_allowed is False
        assert "limit_up" in buy_reason
        assert "limit_up" in sell_reason
    
    @pytest.mark.asyncio
    async def test_comprehensive_halt_scenario(self):
        """Test complete halt lifecycle with compliance tracking."""
        await self.engine.initialize(["LIFECYCLE"])
        
        # Phase 1: Normal trading
        allowed, reason = await self.engine.validate_order("LIFECYCLE", "buy", 100)
        assert allowed is True
        
        # Phase 2: Halt occurs
        self.engine.halt_monitor.simulate_halt("LIFECYCLE", HaltReason.VOLATILITY, 15)
        
        # Multiple blocked attempts
        for i in range(5):
            allowed, reason = await self.engine.validate_order("LIFECYCLE", "buy", 100 + i*10)
            assert allowed is False
            assert "halted" in reason
        
        # Phase 3: Reopen with gap
        halt_info = self.engine.halt_monitor.get_halt_status("LIFECYCLE")
        halt_info.pre_halt_price = 50.0
        self.engine.halt_monitor.simulate_reopen("LIFECYCLE", 58.0)  # 16% gap
        
        # Orders still blocked due to gap
        allowed, reason = await self.engine.validate_order("LIFECYCLE", "buy", 100)
        assert allowed is False
        assert "Post-reopen delay" in reason
        
        # Verify complete audit trail
        report = self.engine.get_compliance_report("LIFECYCLE", hours=1)
        assert report['blocked_orders_count'] >= 6  # 5 during halt + 1 during reopen delay
        assert 'ORDER_VALIDATED' in report['event_breakdown']  # Initial valid order
        assert 'ORDER_BLOCKED' in report['event_breakdown']    # Blocked orders
        
        print(f"✅ LIFECYCLE TEST PASSED: Complete halt protection verified")


if __name__ == "__main__":
    # Run critical compliance tests
    pytest.main([__file__, "-v", "-k", "test_zero_entries_during_halt_guarantee"])