"""
Borrow/Locate & Hard-To-Borrow Fees Checker

This module provides functionality to check borrow availability and fees for short positions.
It integrates with broker APIs or proxy services to ensure short trades can be executed
and tracks borrowing costs in trade records.

Key Features:
- Borrow availability checking
- Hard-to-borrow fee estimation
- Integration with broker APIs
- Fee embedding in P&L calculations
- Trade gating for unavailable borrows
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import aiohttp
import pandas as pd
from decimal import Decimal

logger = logging.getLogger(__name__)


class BorrowStatus(Enum):
    """Borrow availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LIMITED = "limited"
    HARD_TO_BORROW = "hard_to_borrow"
    EASY_TO_BORROW = "easy_to_borrow"
    UNKNOWN = "unknown"


@dataclass
class BorrowInfo:
    """Borrow information for a security."""
    symbol: str
    status: BorrowStatus
    quantity_available: Optional[int] = None
    borrow_rate: Optional[float] = None  # Annual rate
    fee_per_share: Optional[float] = None  # Daily fee per share
    locate_fee: Optional[float] = None  # One-time locate fee
    rebate_rate: Optional[float] = None  # Rebate on cash collateral
    updated_at: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    expires_at: Optional[datetime] = None
    
    @property
    def is_available(self) -> bool:
        """Check if borrow is available for trading."""
        return self.status in [BorrowStatus.AVAILABLE, BorrowStatus.EASY_TO_BORROW, BorrowStatus.HARD_TO_BORROW]
    
    @property
    def daily_cost_per_dollar(self) -> float:
        """Daily borrowing cost per dollar of position."""
        if self.borrow_rate is None:
            return 0.0
        return self.borrow_rate / 365.0
    
    def calculate_daily_fee(self, position_value: float) -> float:
        """Calculate daily borrowing fee for a position."""
        if not self.is_available or self.borrow_rate is None:
            return 0.0
        
        daily_rate = self.daily_cost_per_dollar
        return position_value * daily_rate
    
    def calculate_total_cost(self, position_value: float, holding_days: int) -> float:
        """Calculate total borrowing cost over holding period."""
        daily_fee = self.calculate_daily_fee(position_value)
        locate_cost = self.locate_fee or 0.0
        
        return (daily_fee * holding_days) + locate_cost


@dataclass
class BorrowRequest:
    """Request for borrow information."""
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str = "market"
    urgency: str = "normal"  # normal, urgent
    account_id: Optional[str] = None


class BorrowDataProvider:
    """Base class for borrow data providers."""
    
    def __init__(self, name: str):
        self.name = name
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)  # Cache for 15 minutes
    
    async def get_borrow_info(self, symbol: str) -> BorrowInfo:
        """Get borrow information for a symbol."""
        raise NotImplementedError
    
    async def check_borrow_availability(self, request: BorrowRequest) -> BorrowInfo:
        """Check borrow availability for a specific request."""
        if request.side.lower() == 'buy':
            # No borrow needed for long positions
            return BorrowInfo(
                symbol=request.symbol,
                status=BorrowStatus.AVAILABLE,
                borrow_rate=0.0,
                source=self.name
            )
        
        return await self.get_borrow_info(request.symbol)
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid."""
        if symbol not in self.cache:
            return False
        
        cached_time = self.cache[symbol].updated_at
        return datetime.now() - cached_time < self.cache_duration
    
    def _cache_result(self, symbol: str, borrow_info: BorrowInfo):
        """Cache borrow information."""
        self.cache[symbol] = borrow_info


class InteractiveBrokersProvider(BorrowDataProvider):
    """Interactive Brokers borrow data provider."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.interactivebrokers.com"):
        super().__init__("Interactive Brokers")
        self.api_key = api_key
        self.base_url = base_url
    
    async def get_borrow_info(self, symbol: str) -> BorrowInfo:
        """Get borrow info from Interactive Brokers API."""
        
        # Check cache first
        if self._is_cache_valid(symbol):
            return self.cache[symbol]
        
        try:
            # Mock implementation - replace with actual IB API calls
            borrow_info = await self._fetch_ib_borrow_data(symbol)
            self._cache_result(symbol, borrow_info)
            return borrow_info
            
        except Exception as e:
            logger.error(f"Error fetching borrow data from IB for {symbol}: {e}")
            return BorrowInfo(
                symbol=symbol,
                status=BorrowStatus.UNKNOWN,
                source=self.name
            )
    
    async def _fetch_ib_borrow_data(self, symbol: str) -> BorrowInfo:
        """Fetch borrow data from IB API (mock implementation)."""
        
        # This is a mock implementation
        # In production, you would make actual API calls to IB
        
        # Simulate different borrow scenarios based on symbol
        if symbol in ['SPY', 'QQQ', 'IWM']:  # Easy to borrow ETFs
            return BorrowInfo(
                symbol=symbol,
                status=BorrowStatus.EASY_TO_BORROW,
                quantity_available=1000000,
                borrow_rate=0.0025,  # 0.25% annual
                source=self.name
            )
        elif symbol in ['AAPL', 'MSFT', 'GOOGL']:  # Large cap stocks
            return BorrowInfo(
                symbol=symbol,
                status=BorrowStatus.AVAILABLE,
                quantity_available=100000,
                borrow_rate=0.01,  # 1% annual
                source=self.name
            )
        elif symbol.startswith('GME') or symbol.startswith('AMC'):  # Meme stocks
            return BorrowInfo(
                symbol=symbol,
                status=BorrowStatus.HARD_TO_BORROW,
                quantity_available=1000,
                borrow_rate=0.25,  # 25% annual
                locate_fee=0.05,  # $0.05 per share
                source=self.name
            )
        else:
            return BorrowInfo(
                symbol=symbol,
                status=BorrowStatus.AVAILABLE,
                quantity_available=50000,
                borrow_rate=0.05,  # 5% annual
                source=self.name
            )


class MockBrokerProvider(BorrowDataProvider):
    """Mock broker provider for testing."""
    
    def __init__(self):
        super().__init__("Mock Broker")
        self.custom_rates = {}
    
    def set_custom_rate(self, symbol: str, rate: float, status: BorrowStatus = BorrowStatus.AVAILABLE):
        """Set custom borrow rate for testing."""
        self.custom_rates[symbol] = (rate, status)
    
    async def get_borrow_info(self, symbol: str) -> BorrowInfo:
        """Get mock borrow information."""
        
        if symbol in self.custom_rates:
            rate, status = self.custom_rates[symbol]
            return BorrowInfo(
                symbol=symbol,
                status=status,
                quantity_available=10000,
                borrow_rate=rate,
                source=self.name
            )
        
        # Default mock data
        return BorrowInfo(
            symbol=symbol,
            status=BorrowStatus.AVAILABLE,
            quantity_available=10000,
            borrow_rate=0.03,  # 3% annual
            source=self.name
        )


class BorrowChecker:
    """Main borrow checking and fee calculation engine."""
    
    def __init__(self, providers: List[BorrowDataProvider] = None):
        self.providers = providers or [MockBrokerProvider()]
        self.fee_cache = {}
        self.blocked_symbols = set()
        
    def add_provider(self, provider: BorrowDataProvider):
        """Add a borrow data provider."""
        self.providers.append(provider)
    
    def block_symbol(self, symbol: str, reason: str = "Unavailable for short selling"):
        """Block a symbol from short selling."""
        self.blocked_symbols.add(symbol)
        logger.warning(f"Symbol {symbol} blocked for short selling: {reason}")
    
    def unblock_symbol(self, symbol: str):
        """Unblock a symbol for short selling."""
        self.blocked_symbols.discard(symbol)
        logger.info(f"Symbol {symbol} unblocked for short selling")
    
    async def check_borrow_availability(self, request: BorrowRequest) -> BorrowInfo:
        """Check borrow availability across all providers."""
        
        if request.side.lower() == 'buy':
            # No borrow needed for long positions
            return BorrowInfo(
                symbol=request.symbol,
                status=BorrowStatus.AVAILABLE,
                borrow_rate=0.0,
                source="No borrow needed for long position"
            )
        
        # Check if symbol is blocked
        if request.symbol in self.blocked_symbols:
            return BorrowInfo(
                symbol=request.symbol,
                status=BorrowStatus.UNAVAILABLE,
                source="Blocked by risk management"
            )
        
        # Try providers in order until we get a valid response
        for provider in self.providers:
            try:
                borrow_info = await provider.check_borrow_availability(request)
                
                if borrow_info.status != BorrowStatus.UNKNOWN:
                    logger.info(f"Borrow info for {request.symbol} from {provider.name}: {borrow_info.status.value}")
                    return borrow_info
                    
            except Exception as e:
                logger.error(f"Error checking borrow with {provider.name}: {e}")
                continue
        
        # If all providers fail, return unknown status
        return BorrowInfo(
            symbol=request.symbol,
            status=BorrowStatus.UNKNOWN,
            source="All providers failed"
        )
    
    async def validate_short_order(self, symbol: str, quantity: int, account_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate if a short order can be executed."""
        
        request = BorrowRequest(
            symbol=symbol,
            quantity=quantity,
            side='sell',
            account_id=account_id
        )
        
        borrow_info = await self.check_borrow_availability(request)
        
        validation_result = {
            'symbol': symbol,
            'quantity': quantity,
            'can_execute': borrow_info.is_available,
            'borrow_info': borrow_info,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        if not borrow_info.is_available:
            validation_result['rejection_reason'] = f"Borrow unavailable: {borrow_info.status.value}"
            logger.warning(f"Short order rejected for {symbol}: {validation_result['rejection_reason']}")
        
        # Check quantity limits
        if borrow_info.quantity_available and quantity > borrow_info.quantity_available:
            validation_result['can_execute'] = False
            validation_result['rejection_reason'] = f"Insufficient borrow quantity: {borrow_info.quantity_available} available"
        
        return validation_result
    
    def calculate_borrow_cost(self, symbol: str, position_value: float, holding_days: int, borrow_info: BorrowInfo) -> Dict[str, float]:
        """Calculate detailed borrow costs."""
        
        if not borrow_info.is_available:
            return {
                'daily_fee': 0.0,
                'total_borrow_cost': 0.0,
                'locate_fee': 0.0,
                'net_position_value': position_value
            }
        
        daily_fee = borrow_info.calculate_daily_fee(abs(position_value))
        total_borrow_cost = borrow_info.calculate_total_cost(abs(position_value), holding_days)
        locate_fee = borrow_info.locate_fee or 0.0
        
        # Net position value after fees
        net_position_value = position_value - total_borrow_cost
        
        return {
            'daily_fee': daily_fee,
            'total_borrow_cost': total_borrow_cost,
            'locate_fee': locate_fee,
            'net_position_value': net_position_value,
            'borrow_rate_annual': borrow_info.borrow_rate or 0.0,
            'borrow_rate_daily': borrow_info.daily_cost_per_dollar
        }
    
    async def get_borrow_costs_batch(self, positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Get borrow costs for multiple positions."""
        
        results = {}
        
        for position in positions:
            symbol = position['symbol']
            side = position.get('side', 'buy')
            quantity = position.get('quantity', 0)
            position_value = position.get('position_value', 0)
            holding_days = position.get('holding_days', 1)
            
            if side.lower() == 'sell' and quantity > 0:
                request = BorrowRequest(symbol=symbol, quantity=quantity, side=side)
                borrow_info = await self.check_borrow_availability(request)
                
                costs = self.calculate_borrow_cost(symbol, position_value, holding_days, borrow_info)
                costs['borrow_available'] = borrow_info.is_available
                costs['borrow_status'] = borrow_info.status.value
                
                results[symbol] = costs
            else:
                # Long position or zero quantity
                results[symbol] = {
                    'daily_fee': 0.0,
                    'total_borrow_cost': 0.0,
                    'locate_fee': 0.0,
                    'net_position_value': position_value,
                    'borrow_available': True,
                    'borrow_status': 'not_applicable'
                }
        
        return results
    
    def create_trade_record_with_fees(self, trade_data: Dict[str, Any], borrow_info: BorrowInfo) -> Dict[str, Any]:
        """Create trade record with embedded borrow fees."""
        
        trade_record = trade_data.copy()
        
        # Add borrow information
        trade_record['borrow_info'] = {
            'status': borrow_info.status.value,
            'borrow_rate': borrow_info.borrow_rate,
            'locate_fee': borrow_info.locate_fee,
            'provider': borrow_info.source,
            'updated_at': borrow_info.updated_at.isoformat()
        }
        
        # Calculate fees if it's a short position
        if trade_data.get('side', '').lower() == 'sell' and trade_data.get('quantity', 0) > 0:
            position_value = abs(trade_data.get('notional_value', 0))
            holding_days = trade_data.get('expected_holding_days', 1)
            
            costs = self.calculate_borrow_cost(trade_data['symbol'], position_value, holding_days, borrow_info)
            
            # Embed costs in trade record
            trade_record['borrow_costs'] = costs
            trade_record['expected_net_pnl'] = trade_data.get('expected_pnl', 0) - costs['total_borrow_cost']
            
            # Adjust execution price to account for fees
            if trade_data.get('price') and costs['total_borrow_cost'] > 0:
                shares = trade_data.get('quantity', 1)
                fee_per_share = costs['total_borrow_cost'] / shares
                trade_record['adjusted_price'] = trade_data['price'] - fee_per_share
        
        return trade_record


# Configuration and Utility Functions

def create_borrow_checker(config: Dict[str, Any] = None) -> BorrowChecker:
    """Create a borrow checker with configured providers."""
    
    config = config or {}
    providers = []
    
    # Add Interactive Brokers if configured
    if config.get('enable_ib', False):
        ib_provider = InteractiveBrokersProvider(
            api_key=config.get('ib_api_key'),
            base_url=config.get('ib_base_url', "https://api.interactivebrokers.com")
        )
        providers.append(ib_provider)
    
    # Add mock provider for testing
    if config.get('enable_mock', True):
        mock_provider = MockBrokerProvider()
        providers.append(mock_provider)
    
    checker = BorrowChecker(providers)
    
    # Configure blocked symbols
    blocked_symbols = config.get('blocked_symbols', [])
    for symbol in blocked_symbols:
        checker.block_symbol(symbol)
    
    return checker


async def validate_portfolio_shorts(positions: List[Dict[str, Any]], checker: BorrowChecker) -> Dict[str, Any]:
    """Validate all short positions in a portfolio."""
    
    validation_results = {}
    short_positions = [p for p in positions if p.get('side', '').lower() == 'sell' and p.get('quantity', 0) > 0]
    
    for position in short_positions:
        symbol = position['symbol']
        quantity = position['quantity']
        
        validation = await checker.validate_short_order(symbol, quantity)
        validation_results[symbol] = validation
    
    # Summary statistics
    total_shorts = len(short_positions)
    available_shorts = sum(1 for v in validation_results.values() if v['can_execute'])
    blocked_shorts = total_shorts - available_shorts
    
    return {
        'validation_results': validation_results,
        'summary': {
            'total_short_positions': total_shorts,
            'available_positions': available_shorts,
            'blocked_positions': blocked_shorts,
            'availability_rate': available_shorts / total_shorts if total_shorts > 0 else 1.0
        },
        'timestamp': datetime.now().isoformat()
    }


# Example usage and testing

async def example_usage():
    """Example of using the borrow checker."""
    
    # Create borrow checker
    config = {
        'enable_mock': True,
        'blocked_symbols': ['SPAC1', 'PENNY1']
    }
    
    checker = create_borrow_checker(config)
    
    # Test borrow availability
    request = BorrowRequest(
        symbol='AAPL',
        quantity=1000,
        side='sell'
    )
    
    borrow_info = await checker.check_borrow_availability(request)
    print(f"Borrow info for AAPL: {borrow_info}")
    
    # Validate short order
    validation = await checker.validate_short_order('AAPL', 1000)
    print(f"Validation result: {validation}")
    
    # Calculate borrow costs
    costs = checker.calculate_borrow_cost('AAPL', 50000, 30, borrow_info)
    print(f"Borrow costs: {costs}")
    
    # Test portfolio validation
    portfolio = [
        {'symbol': 'AAPL', 'side': 'sell', 'quantity': 1000},
        {'symbol': 'TSLA', 'side': 'sell', 'quantity': 500},
        {'symbol': 'SPAC1', 'side': 'sell', 'quantity': 2000}  # This should be blocked
    ]
    
    portfolio_validation = await validate_portfolio_shorts(portfolio, checker)
    print(f"Portfolio validation: {portfolio_validation}")


if __name__ == "__main__":
    asyncio.run(example_usage())