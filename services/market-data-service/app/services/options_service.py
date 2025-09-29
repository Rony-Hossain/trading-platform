"""
Options data service for fetching real options chains and computing Greeks.
Provides comprehensive options information for day trading decisions.
"""

import asyncio
import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from scipy.stats import norm
import numpy as np
from ..providers.options_provider import (
    OptionQuote,
    OptionsChainPayload,
    OptionsDataError,
    options_data_provider,
)

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Represents a single options contract with full Greeks and pricing info"""
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    # Derived metrics
    mid_price: float
    bid_ask_spread: float
    dte: int  # Days to expiration
    intrinsic_value: float
    extrinsic_value: float
    break_even: float

@dataclass
class OptionsChain:
    """Complete options chain for a symbol"""
    symbol: str
    underlying_price: float
    calls: List[OptionContract]
    puts: List[OptionContract]
    expiries: List[datetime]
    risk_free_rate: float = 0.05  # Default 5%

@dataclass
class OptionMetrics:
    """Summary metrics derived from an options chain."""
    symbol: str
    as_of: datetime
    expiry: datetime
    underlying_price: float
    atm_strike: Optional[float]
    atm_iv: Optional[float]
    call_volume: int
    put_volume: int
    call_open_interest: int
    put_open_interest: int
    put_call_volume_ratio: Optional[float]
    put_call_oi_ratio: Optional[float]
    straddle_price: Optional[float]
    implied_move_pct: Optional[float]
    implied_move_upper: Optional[float]
    implied_move_lower: Optional[float]
    iv_25d_call: Optional[float]
    iv_25d_put: Optional[float]
    iv_skew_25d: Optional[float]
    iv_skew_25d_pct: Optional[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['as_of'] = self.as_of.isoformat()
        data['expiry'] = self.expiry.isoformat()
        data['metadata'] = dict(self.metadata)
        return data

    def to_db_record(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'as_of': self.as_of,
            'expiry': self.expiry.date() if isinstance(self.expiry, datetime) else self.expiry,
            'underlying_price': self.underlying_price,
            'atm_strike': self.atm_strike,
            'atm_iv': self.atm_iv,
            'call_volume': self.call_volume,
            'put_volume': self.put_volume,
            'call_open_interest': self.call_open_interest,
            'put_open_interest': self.put_open_interest,
            'put_call_volume_ratio': self.put_call_volume_ratio,
            'put_call_oi_ratio': self.put_call_oi_ratio,
            'straddle_price': self.straddle_price,
            'implied_move_pct': self.implied_move_pct,
            'implied_move_upper': self.implied_move_upper,
            'implied_move_lower': self.implied_move_lower,
            'iv_25d_call': self.iv_25d_call,
            'iv_25d_put': self.iv_25d_put,
            'iv_skew_25d': self.iv_skew_25d,
            'iv_skew_25d_pct': self.iv_skew_25d_pct,
            'metadata': dict(self.metadata),
        }

@dataclass
class UnusualActivity:
    """Represents unusual options activity detection"""
    symbol: str
    contract_symbol: str
    strike: float
    expiry: datetime
    option_type: str
    
    # Activity metrics
    volume: int
    avg_volume_20d: int
    volume_ratio: float  # today/average
    open_interest: int
    oi_change: int
    
    # Unusual signals
    volume_spike: bool
    large_single_trades: bool
    sweep_activity: bool
    unusual_volume_vs_oi: bool
    
    # Context
    underlying_price: float
    strike_distance_pct: float  # % OTM/ITM
    days_to_expiration: int
    
    # Scores
    unusual_score: float  # 0-100
    confidence_level: float  # 0-1
    
    # Trade details
    large_trades: List[Dict[str, Any]]
    timestamp: datetime

@dataclass
class OptionsFlow:
    """Options flow analysis for smart money tracking"""
    symbol: str
    timestamp: datetime
    
    # Flow metrics
    total_call_volume: int
    total_put_volume: int
    call_put_ratio: float
    
    # Size analysis
    large_trades_count: int
    block_trades_value: float
    sweep_trades_count: int
    
    # Unusual activity summary
    unusual_activities: List[UnusualActivity]
    flow_sentiment: str  # bullish, bearish, neutral
    smart_money_score: float  # 0-100
    
    # Premium flows
    call_premium_bought: float
    put_premium_bought: float
    net_premium_flow: float

@dataclass
class TradeSuggestion:
    """Detailed trade suggestion with specific contracts and Greeks"""
    strategy: str
    sentiment: str  # 'bullish', 'bearish', 'neutral'
    contracts: List[OptionContract]
    max_profit: float
    max_loss: float
    break_evens: List[float]
    probability_profit: float
    # Greeks summary for the position
    net_delta: float
    net_theta: float
    net_vega: float
    net_gamma: float
    capital_required: float
    roi_potential: float
    liquidity_score: float  # 0-100 based on volume/spreads
    # Advanced strategy details
    strategy_type: str  # 'spread', 'straddle', 'strangle', 'single'
    complexity: str  # 'beginner', 'intermediate', 'advanced'
    time_decay_impact: str  # 'positive', 'negative', 'neutral'
    volatility_impact: str  # 'positive', 'negative', 'neutral'

class OptionsService:
    """Service for fetching and analyzing options data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.provider = options_data_provider
        
    def _calculate_greeks(self, 
                         spot: float, 
                         strike: float, 
                         time_to_expiry: float, 
                         risk_free_rate: float, 
                         volatility: float, 
                         option_type: str) -> Dict[str, float]:
        """
        Calculate Black-Scholes Greeks for an option
        """
        if time_to_expiry <= 0:
            return {
                'delta': 1.0 if (option_type == 'call' and spot > strike) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        # Black-Scholes calculations
        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        
        # Greeks calculations
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (-(spot * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry)) 
                    - risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = (-(spot * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry)) 
                    + risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
        
        gamma = norm.pdf(d1) / (spot * volatility * math.sqrt(time_to_expiry))
        vega = spot * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100
        
        if option_type == 'call':
            rho = strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
        else:
            rho = -strike * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _calculate_liquidity_score(self, contract: OptionContract) -> float:
        """Calculate liquidity score based on volume, open interest, and spreads"""
        volume_score = min(100, contract.volume / 100 * 50)  # Max 50 points for volume
        oi_score = min(100, contract.open_interest / 500 * 30)  # Max 30 points for OI
        spread_score = max(0, 20 - (contract.bid_ask_spread / contract.mid_price * 100))  # Penalty for wide spreads
        
        return min(100, volume_score + oi_score + spread_score)
    


    async def fetch_options_chain(self, symbol: str, expiry_filter: Optional[str] = None) -> OptionsChain:
        """Fetch options chain via provider with caching and fallback."""
        cache_key = f"{symbol}_{expiry_filter}"
        cached = self.cache.get(cache_key)
        if cached:
            cache_time, data = cached
            if datetime.now().timestamp() - cache_time < self.cache_ttl:
                return data

        try:
            payload = await self.provider.get_chain(symbol, expiry_filter)
            chain = self._build_chain_from_payload(payload)
        except OptionsDataError as exc:
            logger.warning("Falling back to synthetic options chain for %s: %s", symbol, exc)
            chain = await self._generate_mock_chain(symbol)

        self.cache[cache_key] = (datetime.now().timestamp(), chain)
        return chain

    async def _generate_mock_chain(self, symbol: str) -> OptionsChain:
        """Synthetic fallback options chain used when real data is unavailable."""
        underlying_price = 150.0
        calls: List[OptionContract] = []
        puts: List[OptionContract] = []

        base_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
        expiries = [
            base_date + timedelta(days=1),
            base_date + timedelta(days=7),
            base_date + timedelta(days=14),
            base_date + timedelta(days=21),
            base_date + timedelta(days=35),
        ]

        risk_free_rate = 0.05

        for expiry in expiries:
            dte = max((expiry - datetime.now()).days, 0)
            time_to_expiry = max(dte / 365.0, 0.001)
            strikes = [round(underlying_price + i * 5, 2) for i in range(-10, 11)]

            for strike in strikes:
                moneyness = strike / underlying_price
                base_iv = 0.25 + abs(moneyness - 1.0) * 0.5

                call_greeks = self._calculate_greeks(
                    underlying_price, strike, time_to_expiry, risk_free_rate, base_iv, 'call'
                )
                put_greeks = self._calculate_greeks(
                    underlying_price, strike, time_to_expiry, risk_free_rate, base_iv, 'put'
                )

                call_theoretical = max(0.05, underlying_price - strike)
                put_theoretical = max(0.05, strike - underlying_price)

                call_bid = max(0.01, call_theoretical * 0.95)
                call_ask = call_theoretical * 1.05
                put_bid = max(0.01, put_theoretical * 0.95)
                put_ask = put_theoretical * 1.05

                distance_from_atm = abs(strike - underlying_price)
                base_volume = max(10, 1000 - distance_from_atm * 20)
                call_volume = int(base_volume * (0.8 + 0.4 * np.random.random()))
                put_volume = int(base_volume * (0.8 + 0.4 * np.random.random()))

                call_contract = OptionContract(
                    symbol=f"{symbol}_{expiry.strftime('%m%d%y')}C{strike}",
                    strike=strike,
                    expiry=expiry,
                    option_type='call',
                    bid=call_bid,
                    ask=call_ask,
                    last=(call_bid + call_ask) / 2,
                    volume=call_volume,
                    open_interest=call_volume * 5,
                    implied_volatility=base_iv,
                    delta=call_greeks['delta'],
                    gamma=call_greeks['gamma'],
                    theta=call_greeks['theta'],
                    vega=call_greeks['vega'],
                    rho=call_greeks['rho'],
                    mid_price=(call_bid + call_ask) / 2,
                    bid_ask_spread=call_ask - call_bid,
                    dte=dte,
                    intrinsic_value=max(0, underlying_price - strike),
                    extrinsic_value=max(0, ((call_bid + call_ask) / 2) - max(0, underlying_price - strike)),
                    break_even=strike + (call_bid + call_ask) / 2,
                )

                put_contract = OptionContract(
                    symbol=f"{symbol}_{expiry.strftime('%m%d%y')}P{strike}",
                    strike=strike,
                    expiry=expiry,
                    option_type='put',
                    bid=put_bid,
                    ask=put_ask,
                    last=(put_bid + put_ask) / 2,
                    volume=put_volume,
                    open_interest=put_volume * 5,
                    implied_volatility=base_iv,
                    delta=put_greeks['delta'],
                    gamma=put_greeks['gamma'],
                    theta=put_greeks['theta'],
                    vega=put_greeks['vega'],
                    rho=put_greeks['rho'],
                    mid_price=(put_bid + put_ask) / 2,
                    bid_ask_spread=put_ask - put_bid,
                    dte=dte,
                    intrinsic_value=max(0, strike - underlying_price),
                    extrinsic_value=max(0, ((put_bid + put_ask) / 2) - max(0, strike - underlying_price)),
                    break_even=strike - (put_bid + put_ask) / 2,
                )

                calls.append(call_contract)
                puts.append(put_contract)

        return OptionsChain(
            symbol=symbol,
            underlying_price=underlying_price,
            calls=calls,
            puts=puts,
            expiries=expiries,
            risk_free_rate=risk_free_rate,
        )

    def _build_chain_from_payload(self, payload: OptionsChainPayload) -> OptionsChain:
        risk_free_rate = 0.05
        calls = [self._calculate_contract_from_quote(q, payload.underlying_price, risk_free_rate) for q in payload.calls]
        puts = [self._calculate_contract_from_quote(q, payload.underlying_price, risk_free_rate) for q in payload.puts]
        expiries = [exp.replace(tzinfo=None) if exp.tzinfo else exp for exp in payload.expiries]
        return OptionsChain(
            symbol=payload.symbol,
            underlying_price=payload.underlying_price,
            calls=calls,
            puts=puts,
            expiries=expiries,
            risk_free_rate=risk_free_rate,
        )

    def _estimate_mid_price(self, bid: float, ask: float, last: float) -> float:
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        if last > 0:
            return last
        return max(bid, ask, last)

    def _calculate_contract_from_quote(self, quote: OptionQuote, underlying_price: float, risk_free_rate: float) -> OptionContract:
        now = datetime.now(timezone.utc)
        expiry_aware = quote.expiry if quote.expiry.tzinfo else quote.expiry.replace(tzinfo=timezone.utc)
        time_to_expiry = max((expiry_aware - now).total_seconds() / (365.0 * 24 * 3600), 0.0)
        dte = max(int((expiry_aware - now).days), 0)
        iv = quote.implied_volatility if quote.implied_volatility and quote.implied_volatility > 0 else 0.2
        greeks = self._calculate_greeks(
            underlying_price,
            quote.strike,
            max(time_to_expiry, 1e-4),
            risk_free_rate,
            max(iv, 1e-4),
            quote.option_type,
        )
        mid_price = self._estimate_mid_price(quote.bid, quote.ask, quote.last)
        intrinsic_value = max(0.0, underlying_price - quote.strike) if quote.option_type == 'call' else max(0.0, quote.strike - underlying_price)
        extrinsic_value = max(mid_price - intrinsic_value, 0.0)
        expiry_naive = expiry_aware.replace(tzinfo=None)
        return OptionContract(
            symbol=quote.symbol,
            strike=quote.strike,
            expiry=expiry_naive,
            option_type=quote.option_type,
            bid=quote.bid,
            ask=quote.ask,
            last=quote.last,
            volume=quote.volume,
            open_interest=quote.open_interest,
            implied_volatility=iv,
            delta=greeks['delta'],
            gamma=greeks['gamma'],
            theta=greeks['theta'],
            vega=greeks['vega'],
            rho=greeks['rho'],
            mid_price=mid_price,
            bid_ask_spread=max(0.0, quote.ask - quote.bid),
            dte=dte,
            intrinsic_value=intrinsic_value,
            extrinsic_value=extrinsic_value,
            break_even=quote.strike + extrinsic_value if quote.option_type == 'call' else quote.strike - extrinsic_value,
        )

    def _find_atm_contracts(
        self,
        calls: List[OptionContract],
        puts: List[OptionContract],
        underlying_price: float,
    ) -> tuple[Optional[OptionContract], Optional[OptionContract]]:
        atm_call = min(calls, key=lambda c: abs(c.strike - underlying_price), default=None)
        atm_put = min(puts, key=lambda p: abs(p.strike - underlying_price), default=None)
        return atm_call, atm_put

    def _nearest_delta_contract(
        self,
        contracts: List[OptionContract],
        target_delta: float,
    ) -> Optional[OptionContract]:
        if not contracts:
            return None
        return min(contracts, key=lambda c: abs(c.delta - target_delta))

    def calculate_chain_metrics(self, chain: OptionsChain) -> OptionMetrics:
        now = datetime.now(timezone.utc)
        expiries = [exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc) for exp in chain.expiries]
        future_expiries = [exp for exp in expiries if exp > now]
        target_expiry = min(future_expiries, default=(expiries[0] if expiries else now))

        calls = [c for c in chain.calls if (c.expiry.replace(tzinfo=timezone.utc) if c.expiry.tzinfo is None else c.expiry) == target_expiry]
        puts = [p for p in chain.puts if (p.expiry.replace(tzinfo=timezone.utc) if p.expiry.tzinfo is None else p.expiry) == target_expiry]

        total_call_vol = sum(c.volume for c in calls)
        total_put_vol = sum(p.volume for p in puts)
        total_call_oi = sum(c.open_interest for c in calls)
        total_put_oi = sum(p.open_interest for p in puts)

        atm_call, atm_put = self._find_atm_contracts(calls, puts, chain.underlying_price)
        atm_strike = atm_call.strike if atm_call else (atm_put.strike if atm_put else None)
        atm_iv_values = [iv for iv in [atm_call.implied_volatility if atm_call else None, atm_put.implied_volatility if atm_put else None] if iv is not None]
        atm_iv = sum(atm_iv_values) / len(atm_iv_values) if atm_iv_values else None

        straddle_price = None
        implied_move_pct = None
        implied_move_upper = None
        implied_move_lower = None
        if atm_call and atm_put:
            straddle_price = max(atm_call.mid_price + atm_put.mid_price, 0.0)
            if chain.underlying_price > 0 and straddle_price is not None:
                implied_move_pct = straddle_price / chain.underlying_price
                implied_move_upper = chain.underlying_price + straddle_price
                implied_move_lower = max(chain.underlying_price - straddle_price, 0.0)

        call_25 = self._nearest_delta_contract(calls, 0.25)
        put_25 = self._nearest_delta_contract(puts, -0.25)
        iv_25d_call = call_25.implied_volatility if call_25 else None
        iv_25d_put = put_25.implied_volatility if put_25 else None
        iv_skew_25d = None
        iv_skew_25d_pct = None
        if iv_25d_call is not None and iv_25d_put is not None and iv_25d_put != 0:
            iv_skew_25d = iv_25d_call - iv_25d_put
            iv_skew_25d_pct = (iv_25d_call / iv_25d_put) - 1

        put_call_vol_ratio = (total_put_vol / total_call_vol) if total_call_vol > 0 else None
        put_call_oi_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

        metadata = {
            "expiry": target_expiry.isoformat(),
            "atm_call": atm_call.symbol if atm_call else None,
            "atm_put": atm_put.symbol if atm_put else None,
        }

        return OptionMetrics(
            symbol=chain.symbol,
            as_of=now,
            expiry=target_expiry.replace(tzinfo=None),
            underlying_price=chain.underlying_price,
            atm_strike=atm_strike,
            atm_iv=atm_iv,
            call_volume=total_call_vol,
            put_volume=total_put_vol,
            call_open_interest=total_call_oi,
            put_open_interest=total_put_oi,
            put_call_volume_ratio=put_call_vol_ratio,
            put_call_oi_ratio=put_call_oi_ratio,
            straddle_price=straddle_price,
            implied_move_pct=implied_move_pct,
            implied_move_upper=implied_move_upper,
            implied_move_lower=implied_move_lower,
            iv_25d_call=iv_25d_call,
            iv_25d_put=iv_25d_put,
            iv_skew_25d=iv_skew_25d,
            iv_skew_25d_pct=iv_skew_25d_pct,
            metadata=metadata,
        )

    def suggest_day_trade(self, 
                         symbol: str, 
                         sentiment: str, 
                         underlying_price: float,
                         target_delta: float = 0.3,
                         max_dte: int = 7,
                         min_liquidity: float = 50) -> List[TradeSuggestion]:
        """
        Generate specific day trading suggestions with real contracts and Greeks
        """
        # This would use the real options chain
        # For now, creating realistic suggestions
        
        suggestions = []
        
        if sentiment == 'bullish':
            # Call debit spread suggestion
            long_strike = underlying_price + 5
            short_strike = underlying_price + 15
            
            # Mock contract data for illustration
            long_call = OptionContract(
                symbol=f"{symbol}_CALL_{long_strike}",
                strike=long_strike,
                expiry=datetime.now() + timedelta(days=3),
                option_type='call',
                bid=2.85, ask=2.95, last=2.90,
                volume=150, open_interest=750,
                implied_volatility=0.28,
                delta=0.35, gamma=0.08, theta=-0.15, vega=0.12, rho=0.05,
                mid_price=2.90, bid_ask_spread=0.10, dte=3,
                intrinsic_value=max(0, underlying_price - long_strike),
                extrinsic_value=2.90 - max(0, underlying_price - long_strike),
                break_even=long_strike + 2.90
            )
            
            short_call = OptionContract(
                symbol=f"{symbol}_CALL_{short_strike}",
                strike=short_strike,
                expiry=datetime.now() + timedelta(days=3),
                option_type='call',
                bid=1.15, ask=1.25, last=1.20,
                volume=80, open_interest=400,
                implied_volatility=0.30,
                delta=0.18, gamma=0.05, theta=-0.08, vega=0.08, rho=0.02,
                mid_price=1.20, bid_ask_spread=0.10, dte=3,
                intrinsic_value=max(0, underlying_price - short_strike),
                extrinsic_value=1.20 - max(0, underlying_price - short_strike),
                break_even=short_strike + 1.20
            )
            
            net_cost = long_call.mid_price - short_call.mid_price
            max_profit = (short_strike - long_strike) - net_cost
            max_loss = net_cost
            
            suggestion = TradeSuggestion(
                strategy="Call Debit Spread",
                sentiment="bullish",
                contracts=[long_call, short_call],
                max_profit=max_profit,
                max_loss=max_loss,
                break_evens=[long_strike + net_cost],
                probability_profit=0.45,
                net_delta=long_call.delta - short_call.delta,
                net_theta=long_call.theta - short_call.theta,
                net_vega=long_call.vega - short_call.vega,
                net_gamma=long_call.gamma - short_call.gamma,
                capital_required=net_cost * 100,
                roi_potential=(max_profit / max_loss) * 100,
                liquidity_score=min(
                    self._calculate_liquidity_score(long_call),
                    self._calculate_liquidity_score(short_call)
                ),
                strategy_type="spread",
                complexity="beginner",
                time_decay_impact="negative",
                volatility_impact="neutral"
            )
            suggestions.append(suggestion)
            
        elif sentiment == 'bearish':
            # Put debit spread suggestion
            long_strike = underlying_price - 5
            short_strike = underlying_price - 15
            
            # Similar mock implementation for put spread
            # ... (implementation similar to call spread above)
            
        return suggestions
    
    def create_straddle_strategy(self, 
                               chain: OptionsChain, 
                               strike: Optional[float] = None,
                               expiry_target: int = 30) -> TradeSuggestion:
        """Create a long straddle strategy (neutral sentiment, expects volatility)"""
        if strike is None:
            strike = chain.underlying_price
        
        # Find closest expiry
        target_expiry = min(chain.expiries, 
                           key=lambda x: abs((x - datetime.now()).days - expiry_target))
        
        # Find call and put at strike
        call = next((c for c in chain.calls 
                    if c.strike == strike and c.expiry == target_expiry), None)
        put = next((p for p in chain.puts 
                   if p.strike == strike and p.expiry == target_expiry), None)
        
        if not call or not put:
            raise ValueError(f"Could not find straddle contracts for strike {strike}")
        
        cost = call.mid_price + put.mid_price
        upper_be = strike + cost
        lower_be = strike - cost
        
        return TradeSuggestion(
            strategy="Long Straddle",
            sentiment="neutral",
            contracts=[call, put],
            max_profit=float('inf'),  # Unlimited
            max_loss=cost,
            break_evens=[upper_be, lower_be],
            probability_profit=0.35,  # Requires significant move
            net_delta=call.delta + put.delta,  # Should be near 0
            net_theta=call.theta + put.theta,
            net_vega=call.vega + put.vega,
            net_gamma=call.gamma + put.gamma,
            capital_required=cost * 100,
            roi_potential=300,  # High potential but requires movement
            liquidity_score=min(
                self._calculate_liquidity_score(call),
                self._calculate_liquidity_score(put)
            ),
            strategy_type="straddle",
            complexity="intermediate",
            time_decay_impact="negative",
            volatility_impact="positive"
        )
    
    def create_strangle_strategy(self, 
                               chain: OptionsChain,
                               call_strike: Optional[float] = None,
                               put_strike: Optional[float] = None,
                               expiry_target: int = 30) -> TradeSuggestion:
        """Create a long strangle strategy (neutral, cheaper than straddle)"""
        if call_strike is None:
            call_strike = chain.underlying_price + 10
        if put_strike is None:
            put_strike = chain.underlying_price - 10
            
        target_expiry = min(chain.expiries, 
                           key=lambda x: abs((x - datetime.now()).days - expiry_target))
        
        call = next((c for c in chain.calls 
                    if c.strike == call_strike and c.expiry == target_expiry), None)
        put = next((p for p in chain.puts 
                   if p.strike == put_strike and p.expiry == target_expiry), None)
        
        if not call or not put:
            raise ValueError(f"Could not find strangle contracts")
        
        cost = call.mid_price + put.mid_price
        upper_be = call_strike + cost
        lower_be = put_strike - cost
        
        return TradeSuggestion(
            strategy="Long Strangle",
            sentiment="neutral",
            contracts=[call, put],
            max_profit=float('inf'),
            max_loss=cost,
            break_evens=[upper_be, lower_be],
            probability_profit=0.25,  # Requires larger move than straddle
            net_delta=call.delta + put.delta,
            net_theta=call.theta + put.theta,
            net_vega=call.vega + put.vega,
            net_gamma=call.gamma + put.gamma,
            capital_required=cost * 100,
            roi_potential=400,
            liquidity_score=min(
                self._calculate_liquidity_score(call),
                self._calculate_liquidity_score(put)
            ),
            strategy_type="strangle",
            complexity="intermediate",
            time_decay_impact="negative",
            volatility_impact="positive"
        )
    
    def create_iron_condor_strategy(self, 
                                  chain: OptionsChain,
                                  width: float = 10,
                                  expiry_target: int = 30) -> TradeSuggestion:
        """Create an iron condor strategy (neutral, limited risk/reward)"""
        underlying = chain.underlying_price
        
        # Iron condor strikes
        put_long_strike = underlying - width * 2
        put_short_strike = underlying - width
        call_short_strike = underlying + width
        call_long_strike = underlying + width * 2
        
        target_expiry = min(chain.expiries, 
                           key=lambda x: abs((x - datetime.now()).days - expiry_target))
        
        # Find all four contracts
        put_long = next((p for p in chain.puts 
                        if p.strike == put_long_strike and p.expiry == target_expiry), None)
        put_short = next((p for p in chain.puts 
                         if p.strike == put_short_strike and p.expiry == target_expiry), None)
        call_short = next((c for c in chain.calls 
                          if c.strike == call_short_strike and c.expiry == target_expiry), None)
        call_long = next((c for c in chain.calls 
                         if c.strike == call_long_strike and c.expiry == target_expiry), None)
        
        if not all([put_long, put_short, call_short, call_long]):
            raise ValueError("Could not find all iron condor contracts")
        
        # Net credit received
        net_credit = (put_short.mid_price + call_short.mid_price - 
                     put_long.mid_price - call_long.mid_price)
        
        max_profit = net_credit
        max_loss = width - net_credit
        
        return TradeSuggestion(
            strategy="Iron Condor",
            sentiment="neutral",
            contracts=[put_long, put_short, call_short, call_long],
            max_profit=max_profit,
            max_loss=max_loss,
            break_evens=[put_short_strike - net_credit, call_short_strike + net_credit],
            probability_profit=0.60,  # High probability, limited profit
            net_delta=(put_long.delta + put_short.delta + 
                      call_short.delta + call_long.delta),
            net_theta=(put_long.theta + put_short.theta + 
                      call_short.theta + call_long.theta),
            net_vega=(put_long.vega + put_short.vega + 
                     call_short.vega + call_long.vega),
            net_gamma=(put_long.gamma + put_short.gamma + 
                      call_short.gamma + call_long.gamma),
            capital_required=max_loss * 100,
            roi_potential=(max_profit / max_loss) * 100,
            liquidity_score=min([
                self._calculate_liquidity_score(put_long),
                self._calculate_liquidity_score(put_short),
                self._calculate_liquidity_score(call_short),
                self._calculate_liquidity_score(call_long)
            ]),
            strategy_type="spread",
            complexity="advanced",
            time_decay_impact="positive",
            volatility_impact="negative"
        )
    
    def create_bull_call_spread(self, 
                              chain: OptionsChain,
                              long_strike: Optional[float] = None,
                              short_strike: Optional[float] = None,
                              expiry_target: int = 30) -> TradeSuggestion:
        """Create a bull call spread strategy (bullish, limited risk/reward)"""
        if long_strike is None:
            long_strike = chain.underlying_price
        if short_strike is None:
            short_strike = chain.underlying_price + 10
            
        target_expiry = min(chain.expiries, 
                           key=lambda x: abs((x - datetime.now()).days - expiry_target))
        
        long_call = next((c for c in chain.calls 
                         if c.strike == long_strike and c.expiry == target_expiry), None)
        short_call = next((c for c in chain.calls 
                          if c.strike == short_strike and c.expiry == target_expiry), None)
        
        if not long_call or not short_call:
            raise ValueError("Could not find bull call spread contracts")
        
        net_debit = long_call.mid_price - short_call.mid_price
        max_profit = (short_strike - long_strike) - net_debit
        max_loss = net_debit
        break_even = long_strike + net_debit
        
        return TradeSuggestion(
            strategy="Bull Call Spread",
            sentiment="bullish",
            contracts=[long_call, short_call],
            max_profit=max_profit,
            max_loss=max_loss,
            break_evens=[break_even],
            probability_profit=0.45,
            net_delta=long_call.delta - short_call.delta,
            net_theta=long_call.theta - short_call.theta,
            net_vega=long_call.vega - short_call.vega,
            net_gamma=long_call.gamma - short_call.gamma,
            capital_required=net_debit * 100,
            roi_potential=(max_profit / max_loss) * 100,
            liquidity_score=min(
                self._calculate_liquidity_score(long_call),
                self._calculate_liquidity_score(short_call)
            ),
            strategy_type="spread",
            complexity="beginner",
            time_decay_impact="negative",
            volatility_impact="neutral"
        )
    
    def create_bear_put_spread(self, 
                             chain: OptionsChain,
                             long_strike: Optional[float] = None,
                             short_strike: Optional[float] = None,
                             expiry_target: int = 30) -> TradeSuggestion:
        """Create a bear put spread strategy (bearish, limited risk/reward)"""
        if long_strike is None:
            long_strike = chain.underlying_price
        if short_strike is None:
            short_strike = chain.underlying_price - 10
            
        target_expiry = min(chain.expiries, 
                           key=lambda x: abs((x - datetime.now()).days - expiry_target))
        
        long_put = next((p for p in chain.puts 
                        if p.strike == long_strike and p.expiry == target_expiry), None)
        short_put = next((p for p in chain.puts 
                         if p.strike == short_strike and p.expiry == target_expiry), None)
        
        if not long_put or not short_put:
            raise ValueError("Could not find bear put spread contracts")
        
        net_debit = long_put.mid_price - short_put.mid_price
        max_profit = (long_strike - short_strike) - net_debit
        max_loss = net_debit
        break_even = long_strike - net_debit
        
        return TradeSuggestion(
            strategy="Bear Put Spread",
            sentiment="bearish",
            contracts=[long_put, short_put],
            max_profit=max_profit,
            max_loss=max_loss,
            break_evens=[break_even],
            probability_profit=0.45,
            net_delta=long_put.delta - short_put.delta,
            net_theta=long_put.theta - short_put.theta,
            net_vega=long_put.vega - short_put.vega,
            net_gamma=long_put.gamma - short_put.gamma,
            capital_required=net_debit * 100,
            roi_potential=(max_profit / max_loss) * 100,
            liquidity_score=min(
                self._calculate_liquidity_score(long_put),
                self._calculate_liquidity_score(short_put)
            ),
            strategy_type="spread",
            complexity="beginner",
            time_decay_impact="negative",
            volatility_impact="neutral"
        )
    
    async def get_advanced_strategies(self, 
                                    symbol: str, 
                                    sentiment: str = "all",
                                    complexity: str = "all") -> List[TradeSuggestion]:
        """Get comprehensive list of advanced options strategies"""
        chain = await self.fetch_options_chain(symbol)
        strategies = []
        
        try:
            if sentiment in ["neutral", "all"]:
                # Neutral strategies
                strategies.append(self.create_straddle_strategy(chain))
                strategies.append(self.create_strangle_strategy(chain))
                if complexity in ["advanced", "all"]:
                    strategies.append(self.create_iron_condor_strategy(chain))
            
            if sentiment in ["bullish", "all"]:
                # Bullish strategies
                strategies.append(self.create_bull_call_spread(chain))
            
            if sentiment in ["bearish", "all"]:
                # Bearish strategies
                strategies.append(self.create_bear_put_spread(chain))
                
        except ValueError as e:
            logger.warning(f"Could not create some strategies for {symbol}: {e}")
        
        # Sort by ROI potential and liquidity
        strategies.sort(key=lambda x: (x.roi_potential, x.liquidity_score), reverse=True)
        return strategies
    
    async def detect_unusual_activity(self, symbol: str, lookback_days: int = 20) -> List[UnusualActivity]:
        """Detect unusual options activity using volume, open interest, and trade patterns"""
        chain = await self.fetch_options_chain(symbol)
        unusual_activities = []
        
        current_price = chain.underlying_price
        
        # Analyze each contract for unusual patterns
        all_contracts = chain.calls + chain.puts
        
        for contract in all_contracts:
            try:
                # Calculate historical average volume (simulated for now)
                avg_volume_20d = max(100, contract.open_interest * 0.1)  # Rough estimate
                volume_ratio = contract.volume / avg_volume_20d if avg_volume_20d > 0 else 0
                
                # Calculate strike distance percentage
                if contract.option_type == 'call':
                    strike_distance_pct = ((contract.strike - current_price) / current_price) * 100
                else:
                    strike_distance_pct = ((current_price - contract.strike) / current_price) * 100
                
                # Detect unusual signals
                volume_spike = volume_ratio > 5.0  # 5x normal volume
                large_single_trades = contract.volume > 1000  # Large size threshold
                unusual_volume_vs_oi = contract.volume > (contract.open_interest * 0.5) if contract.open_interest > 0 else False
                sweep_activity = volume_ratio > 10.0 and contract.volume > 500  # Potential sweep
                
                # Calculate unusual score
                unusual_score = self._calculate_unusual_score(
                    volume_ratio, contract.volume, contract.open_interest, 
                    strike_distance_pct, contract.dte
                )
                
                # Only include if significantly unusual
                if unusual_score > 30:  # Threshold for "unusual"
                    # Simulate some large trades for demonstration
                    large_trades = []
                    if large_single_trades:
                        large_trades.append({
                            "size": min(contract.volume, 2000),
                            "price": contract.last,
                            "time": datetime.now().isoformat(),
                            "side": "buy" if volume_ratio > 7 else "sell"
                        })
                    
                    activity = UnusualActivity(
                        symbol=symbol,
                        contract_symbol=f"{symbol}_{contract.expiry.strftime('%y%m%d')}_{contract.option_type[0].upper()}{contract.strike}",
                        strike=contract.strike,
                        expiry=contract.expiry,
                        option_type=contract.option_type,
                        volume=contract.volume,
                        avg_volume_20d=int(avg_volume_20d),
                        volume_ratio=volume_ratio,
                        open_interest=contract.open_interest,
                        oi_change=0,  # Would need historical data
                        volume_spike=volume_spike,
                        large_single_trades=large_single_trades,
                        sweep_activity=sweep_activity,
                        unusual_volume_vs_oi=unusual_volume_vs_oi,
                        underlying_price=current_price,
                        strike_distance_pct=strike_distance_pct,
                        days_to_expiration=contract.dte,
                        unusual_score=unusual_score,
                        confidence_level=min(1.0, volume_ratio / 10),
                        large_trades=large_trades,
                        timestamp=datetime.now()
                    )
                    
                    unusual_activities.append(activity)
                    
            except Exception as e:
                logger.warning(f"Error analyzing contract {contract.strike} {contract.option_type}: {e}")
                continue
        
        # Sort by unusual score
        unusual_activities.sort(key=lambda x: x.unusual_score, reverse=True)
        return unusual_activities[:20]  # Return top 20 most unusual
    
    def _calculate_unusual_score(self, volume_ratio: float, volume: int, 
                                open_interest: int, strike_distance_pct: float, 
                                dte: int) -> float:
        """Calculate a composite unusual activity score (0-100)"""
        
        # Volume ratio component (0-40 points)
        volume_score = min(40, volume_ratio * 4)
        
        # Absolute volume component (0-25 points)
        volume_abs_score = min(25, volume / 100)
        
        # Volume vs OI component (0-20 points)
        vol_oi_ratio = volume / max(1, open_interest)
        vol_oi_score = min(20, vol_oi_ratio * 20)
        
        # Strike selection component (0-15 points)
        # Favor ATM and slightly OTM options
        strike_score = 0
        if abs(strike_distance_pct) < 5:  # ATM
            strike_score = 15
        elif abs(strike_distance_pct) < 15:  # Near the money
            strike_score = 10
        elif abs(strike_distance_pct) < 30:  # Reasonable OTM
            strike_score = 5
        
        # Time decay factor - penalize very short term or very long term
        if 7 <= dte <= 45:  # Sweet spot for unusual activity
            time_score = 0  # No penalty
        elif dte < 7:  # Very short term
            time_score = -10
        else:  # Too long term
            time_score = -5
        
        total_score = volume_score + volume_abs_score + vol_oi_score + strike_score + time_score
        return max(0, min(100, total_score))
    
    async def analyze_options_flow(self, symbol: str) -> OptionsFlow:
        """Analyze options flow to detect smart money activity"""
        chain = await self.fetch_options_chain(symbol)
        unusual_activities = await self.detect_unusual_activity(symbol)
        
        # Calculate flow metrics
        total_call_volume = sum(c.volume for c in chain.calls)
        total_put_volume = sum(p.volume for p in chain.puts)
        call_put_ratio = total_call_volume / max(1, total_put_volume)
        
        # Analyze large trades and sweeps
        large_trades_count = sum(1 for activity in unusual_activities if activity.large_single_trades)
        sweep_trades_count = sum(1 for activity in unusual_activities if activity.sweep_activity)
        
        # Calculate block trades value (estimate)
        block_trades_value = sum(
            activity.volume * activity.underlying_price * 100 
            for activity in unusual_activities 
            if activity.volume > 500
        )
        
        # Calculate premium flows
        call_premium_bought = sum(
            c.volume * c.mid_price * 100 for c in chain.calls
        )
        put_premium_bought = sum(
            p.volume * p.mid_price * 100 for p in chain.puts
        )
        net_premium_flow = call_premium_bought - put_premium_bought
        
        # Determine flow sentiment
        if call_put_ratio > 1.5 and net_premium_flow > 0:
            flow_sentiment = "bullish"
        elif call_put_ratio < 0.67 and net_premium_flow < 0:
            flow_sentiment = "bearish"
        else:
            flow_sentiment = "neutral"
        
        # Calculate smart money score
        smart_money_score = self._calculate_smart_money_score(
            unusual_activities, call_put_ratio, block_trades_value, sweep_trades_count
        )
        
        return OptionsFlow(
            symbol=symbol,
            timestamp=datetime.now(),
            total_call_volume=total_call_volume,
            total_put_volume=total_put_volume,
            call_put_ratio=call_put_ratio,
            large_trades_count=large_trades_count,
            block_trades_value=block_trades_value,
            sweep_trades_count=sweep_trades_count,
            unusual_activities=unusual_activities,
            flow_sentiment=flow_sentiment,
            smart_money_score=smart_money_score,
            call_premium_bought=call_premium_bought,
            put_premium_bought=put_premium_bought,
            net_premium_flow=net_premium_flow
        )
    
    def _calculate_smart_money_score(self, unusual_activities: List[UnusualActivity],
                                   call_put_ratio: float, block_trades_value: float,
                                   sweep_trades_count: int) -> float:
        """Calculate smart money score based on activity patterns"""
        
        score = 0
        
        # Unusual activity count (0-30 points)
        activity_score = min(30, len(unusual_activities) * 3)
        score += activity_score
        
        # Block trades value (0-25 points)
        block_score = min(25, block_trades_value / 1000000)  # $1M = 25 points
        score += block_score
        
        # Sweep activity (0-20 points)
        sweep_score = min(20, sweep_trades_count * 4)
        score += sweep_score
        
        # Directional conviction (0-25 points)
        if call_put_ratio > 2.0 or call_put_ratio < 0.5:  # Strong directional bias
            conviction_score = 25
        elif call_put_ratio > 1.5 or call_put_ratio < 0.67:  # Moderate bias
            conviction_score = 15
        else:  # Neutral
            conviction_score = 5
        score += conviction_score
        
        return min(100, score)

# Global instance
options_service = OptionsService()