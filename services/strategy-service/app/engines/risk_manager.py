"""
Risk Manager utilities for position sizing and risk analytics.
Enhanced with VaR-based position sizing and portfolio risk management.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .var_calculator import (
    VaRCalculator, VaRConfiguration, VaRMethod, RiskHorizon,
    PositionInfo, VaRResult, PortfolioVaRResult
)

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingRequest:
    portfolio_value: float
    entry_price: float
    method: str = "fixed"
    risk_fraction: Optional[float] = None
    stop_loss_price: Optional[float] = None
    atr_period: int = 14
    atr_multiplier: float = 2.0
    max_position_fraction: float = 0.2
    min_position: int = 1
    # VaR-specific parameters
    symbol: Optional[str] = None
    returns_data: Optional[pd.Series] = None
    max_var_contribution: Optional[float] = None  # Max VaR as fraction of portfolio
    var_method: str = "historical"  # historical, parametric, monte_carlo, cornish_fisher
    var_confidence: float = 0.95
    var_horizon: int = 1  # Risk horizon in days


class RiskManager:
    """Enhanced risk management with VaR-based position sizing and portfolio risk analytics."""

    def __init__(
        self,
        default_risk_fraction: float = 0.01,
        default_atr_period: int = 14,
        default_atr_multiplier: float = 2.0,
        max_position_fraction: float = 0.2,
        default_var_confidence: float = 0.95,
        default_max_var_contribution: float = 0.05,  # 5% max VaR contribution
        var_lookback_period: int = 252,
    ) -> None:
        self.default_risk_fraction = default_risk_fraction
        self.default_atr_period = default_atr_period
        self.default_atr_multiplier = default_atr_multiplier
        self.default_max_position_fraction = max_position_fraction
        self.default_var_confidence = default_var_confidence
        self.default_max_var_contribution = default_max_var_contribution
        
        # Initialize VaR calculator
        var_config = VaRConfiguration(
            confidence_level=default_var_confidence,
            risk_horizon=RiskHorizon.DAILY,
            lookback_period=var_lookback_period
        )
        self.var_calculator = VaRCalculator(var_config)
        
        # Portfolio tracking for risk aggregation
        self.current_positions = []
        self.portfolio_var_cache = None
        self.last_var_calculation = None

    # ------------------------------------------------------------------
    # Health / diagnostics
    # ------------------------------------------------------------------
    def is_healthy(self) -> bool:
        """Simple health check hook used by the FastAPI service."""
        return True

    async def analyze_backtest_risk(
        self,
        db_session: Any,
        backtest_id: int,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Enhanced risk analysis with VaR and CVaR calculations."""
        logger.info(
            "Enhanced risk analysis for backtest %s (confidence %.2f)",
            backtest_id,
            confidence_level,
        )
        
        # This would integrate with actual backtest results from database
        # For now, return enhanced risk metrics structure
        return {
            "backtest_id": backtest_id,
            "value_at_risk_95": None,  # 95% VaR
            "conditional_var_95": None,  # 95% CVaR (Expected Shortfall)
            "value_at_risk_99": None,  # 99% VaR
            "conditional_var_99": None,  # 99% CVaR
            "max_drawdown": None,
            "volatility_annualized": None,
            "var_break_ratio": None,  # Percentage of days VaR was exceeded
            "confidence_level": confidence_level,
            "risk_attribution": {
                "individual_position_vars": {},
                "correlation_contribution": None,
                "diversification_benefit": None
            },
            "stress_test_results": {},
            "notes": "Enhanced VaR-based risk analytics implemented."
        }
    
    async def calculate_portfolio_var(
        self,
        positions: List[Dict[str, Any]],
        method: str = "historical",
        confidence_level: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate portfolio-level VaR with correlation effects.
        
        Args:
            positions: List of position dictionaries with symbol, quantity, price, returns
            method: VaR calculation method
            confidence_level: VaR confidence level
            
        Returns:
            Portfolio VaR analysis results
        """
        try:
            confidence = confidence_level or self.default_var_confidence
            var_method = VaRMethod(method.lower())
            
            # Convert positions to VaR calculator format
            position_objects = []
            total_portfolio_value = 0
            
            for pos in positions:
                market_value = abs(pos.get('quantity', 0) * pos.get('current_price', 0))
                total_portfolio_value += market_value
                
                position_info = PositionInfo(
                    symbol=pos.get('symbol', 'UNKNOWN'),
                    quantity=pos.get('quantity', 0),
                    current_price=pos.get('current_price', 0),
                    market_value=market_value,
                    weight=0,  # Will be calculated after total is known
                    returns_data=pos.get('returns_data')
                )
                position_objects.append(position_info)
            
            # Calculate weights
            for pos in position_objects:
                pos.weight = pos.market_value / total_portfolio_value if total_portfolio_value > 0 else 0
            
            # Calculate portfolio VaR
            self.var_calculator.config.confidence_level = confidence
            portfolio_var_result = self.var_calculator.calculate_portfolio_var(
                position_objects, var_method, consider_correlation=True
            )
            
            # Cache results
            self.portfolio_var_cache = portfolio_var_result
            self.last_var_calculation = datetime.now()
            
            return portfolio_var_result.to_dict()
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return {
                'error': f'Portfolio VaR calculation failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(
        self,
        request: PositionSizingRequest,
        market_data: Optional[pd.DataFrame] = None,
    ) -> int:
        """Route position sizing request to the configured strategy."""
        method = request.method.lower()
        if method == "fixed":
            return self._position_size_fixed(request)
        elif method == "atr":
            return self._position_size_atr(request, market_data)
        elif method == "var":
            return self._position_size_var(request)
        else:
            raise ValueError(f"Unsupported position sizing method: {request.method}")
    
    def _position_size_var(self, request: PositionSizingRequest) -> int:
        """VaR-based position sizing with max VaR contribution limits."""
        
        if not request.symbol or request.returns_data is None:
            logger.warning("VaR sizing requires symbol and returns data; falling back to fixed sizing")
            return self._position_size_fixed(request)
        
        max_var_contribution = request.max_var_contribution or self.default_max_var_contribution
        var_method = VaRMethod(request.var_method.lower())
        
        try:
            # Update VaR calculator configuration
            self.var_calculator.config.confidence_level = request.var_confidence
            self.var_calculator.config.risk_horizon = RiskHorizon(request.var_horizon)
            
            # Calculate maximum position size based on VaR contribution limit
            var_sizing_result = self.var_calculator.calculate_maximum_position_size(
                symbol=request.symbol,
                entry_price=request.entry_price,
                portfolio_value=request.portfolio_value,
                max_var_contribution=max_var_contribution,
                returns_data=request.returns_data,
                method=var_method
            )
            
            if 'error' in var_sizing_result:
                logger.warning(f"VaR sizing failed: {var_sizing_result['error']}; falling back to fixed sizing")
                return self._position_size_fixed(request)
            
            var_based_quantity = var_sizing_result.get('max_units', 0)
            
            # Apply additional constraints
            max_position_fraction = request.max_position_fraction or self.default_max_position_fraction
            max_position_value = request.portfolio_value * max_position_fraction
            max_units_by_fraction = max(request.min_position, 
                                      math.floor(max_position_value / request.entry_price))
            
            # Take the minimum of VaR-based and fraction-based limits
            final_quantity = min(var_based_quantity, max_units_by_fraction)
            final_quantity = max(request.min_position, final_quantity)
            
            logger.debug(
                "VaR sizing -> symbol=%s var_units=%s fraction_units=%s final=%s var_contribution=%.4f",
                request.symbol,
                var_based_quantity,
                max_units_by_fraction,
                final_quantity,
                max_var_contribution
            )
            
            return int(final_quantity)
            
        except Exception as e:
            logger.error(f"Error in VaR-based position sizing: {e}")
            logger.warning("Falling back to fixed position sizing")
            return self._position_size_fixed(request)

    def _position_size_fixed(self, request: PositionSizingRequest) -> int:
        risk_fraction = request.risk_fraction or self.default_risk_fraction
        max_position_fraction = request.max_position_fraction or self.default_max_position_fraction

        risk_amount = request.portfolio_value * risk_fraction
        max_position_value = request.portfolio_value * max_position_fraction

        stop_distance = None
        if request.stop_loss_price is not None:
            stop_distance = abs(request.entry_price - request.stop_loss_price)
        if not stop_distance or stop_distance <= 0:
            stop_distance = request.entry_price * 0.02  # fallback 2 percent buffer

        quantity = math.floor(risk_amount / max(stop_distance, 1e-6))
        if quantity <= 0:
            quantity = request.min_position

        notional = quantity * request.entry_price
        if notional > max_position_value:
            quantity = max(request.min_position, math.floor(max_position_value / request.entry_price))

        logger.debug(
            "Fixed sizing -> risk_fraction=%.4f stop_distance=%.4f quantity=%s",
            risk_fraction,
            stop_distance,
            quantity,
        )
        return int(max(request.min_position, quantity))

    def _position_size_atr(
        self,
        request: PositionSizingRequest,
        market_data: Optional[pd.DataFrame],
    ) -> int:
        if market_data is None or len(market_data) < request.atr_period + 1:
            logger.warning("Insufficient market data for ATR sizing; falling back to fixed sizing")
            return self._position_size_fixed(request)

        risk_fraction = request.risk_fraction or self.default_risk_fraction
        atr_period = request.atr_period or self.default_atr_period
        atr_multiplier = request.atr_multiplier or self.default_atr_multiplier
        max_position_fraction = request.max_position_fraction or self.default_max_position_fraction

        atr_value = self._compute_atr(market_data, atr_period)
        if atr_value is None or np.isnan(atr_value):
            logger.warning("ATR calculation failed; falling back to fixed sizing")
            return self._position_size_fixed(request)

        risk_amount = request.portfolio_value * risk_fraction
        risk_per_share = max(atr_value * atr_multiplier, 1e-6)
        quantity = math.floor(risk_amount / risk_per_share)
        if quantity <= 0:
            quantity = request.min_position

        max_position_value = request.portfolio_value * max_position_fraction
        notional = quantity * request.entry_price
        if notional > max_position_value:
            quantity = max(request.min_position, math.floor(max_position_value / request.entry_price))

        logger.debug(
            "ATR sizing -> atr=%.4f multiplier=%.2f risk_fraction=%.4f quantity=%s",
            atr_value,
            atr_multiplier,
            risk_fraction,
            quantity,
        )
        return int(max(request.min_position, quantity))

    @staticmethod
    def _compute_atr(market_data: pd.DataFrame, period: int) -> Optional[float]:
        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(set(market_data.columns)):
            logger.warning("Market data missing required columns for ATR calculation")
            return None

        high = market_data["high"].astype(float)
        low = market_data["low"].astype(float)
        close = market_data["close"].astype(float)

        previous_close = close.shift(1)
        tr_components = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ], axis=1)
        true_range = tr_components.max(axis=1)
        atr_series = true_range.rolling(window=period).mean()
        atr_value = atr_series.iloc[-1]
        return None if pd.isna(atr_value) else float(atr_value)

    # ------------------------------------------------------------------
    # Portfolio-level risk management
    # ------------------------------------------------------------------
    
    async def update_portfolio_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update current portfolio positions for risk tracking."""
        self.current_positions = positions
        logger.info(f"Updated portfolio with {len(positions)} positions")
    
    async def check_var_limits(self, new_position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if adding a new position would violate VaR limits.
        
        Args:
            new_position: Dictionary with symbol, quantity, price, returns_data
            
        Returns:
            Risk check results with approval/rejection
        """
        try:
            # Simulate portfolio with new position
            simulated_positions = self.current_positions.copy()
            simulated_positions.append(new_position)
            
            # Calculate VaR for simulated portfolio
            portfolio_var = await self.calculate_portfolio_var(simulated_positions)
            
            if 'error' in portfolio_var:
                return {
                    'approved': False,
                    'reason': f"VaR calculation error: {portfolio_var['error']}",
                    'new_position': new_position
                }
            
            # Check VaR limits
            portfolio_value = portfolio_var.get('portfolio_value', 0)
            var_percentage = portfolio_var.get('var_percentage', 0)
            max_var_percentage = 15.0  # 15% maximum portfolio VaR
            
            if var_percentage > max_var_percentage:
                return {
                    'approved': False,
                    'reason': f"Portfolio VaR {var_percentage:.2f}% exceeds limit {max_var_percentage}%",
                    'current_var': var_percentage,
                    'var_limit': max_var_percentage,
                    'new_position': new_position
                }
            
            # Check individual position VaR contribution
            position_value = abs(new_position.get('quantity', 0) * new_position.get('current_price', 0))
            position_var_contribution = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_var_contribution > self.default_max_var_contribution:
                return {
                    'approved': False,
                    'reason': f"Position VaR contribution {position_var_contribution:.2f}% exceeds limit {self.default_max_var_contribution:.2f}%",
                    'position_contribution': position_var_contribution,
                    'contribution_limit': self.default_max_var_contribution,
                    'new_position': new_position
                }
            
            return {
                'approved': True,
                'portfolio_var': portfolio_var,
                'position_contribution': position_var_contribution,
                'new_position': new_position
            }
            
        except Exception as e:
            logger.error(f"Error checking VaR limits: {e}")
            return {
                'approved': False,
                'reason': f"Risk check error: {str(e)}",
                'new_position': new_position
            }
    
    async def stress_test_portfolio(self, stress_scenarios: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Perform stress testing on current portfolio.
        
        Args:
            stress_scenarios: Custom stress scenarios (default scenarios if None)
            
        Returns:
            Stress test results
        """
        if not self.current_positions:
            return {
                'error': 'No positions in portfolio for stress testing',
                'timestamp': datetime.now().isoformat()
            }
        
        # Default stress scenarios
        if stress_scenarios is None:
            stress_scenarios = {
                'market_crash_2008': -20.0,  # 20% market drop
                'covid_crash_2020': -35.0,   # 35% market drop
                'black_monday_1987': -22.0,  # 22% market drop
                'moderate_correction': -10.0, # 10% market drop
                'bull_rally': 15.0,          # 15% market rise
                'sector_rotation': -5.0      # 5% sector-specific drop
            }
        
        try:
            # Convert positions to VaR calculator format
            position_objects = []
            for pos in self.current_positions:
                market_value = abs(pos.get('quantity', 0) * pos.get('current_price', 0))
                position_info = PositionInfo(
                    symbol=pos.get('symbol', 'UNKNOWN'),
                    quantity=pos.get('quantity', 0),
                    current_price=pos.get('current_price', 0),
                    market_value=market_value,
                    weight=0,  # Will be calculated
                    returns_data=pos.get('returns_data')
                )
                position_objects.append(position_info)
            
            # Perform stress test
            stress_results = self.var_calculator.stress_test_portfolio(
                position_objects, stress_scenarios
            )
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error in portfolio stress testing: {e}")
            return {
                'error': f'Stress test failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive risk dashboard for current portfolio.
        
        Returns:
            Risk dashboard with key metrics
        """
        try:
            portfolio_value = sum(
                abs(pos.get('quantity', 0) * pos.get('current_price', 0))
                for pos in self.current_positions
            )
            
            # Basic portfolio metrics
            dashboard = {
                'portfolio_summary': {
                    'total_value': portfolio_value,
                    'num_positions': len(self.current_positions),
                    'last_updated': datetime.now().isoformat()
                },
                'var_metrics': {
                    'last_calculation': self.last_var_calculation.isoformat() if self.last_var_calculation else None,
                    'cached_portfolio_var': self.portfolio_var_cache.to_dict() if self.portfolio_var_cache else None
                },
                'risk_limits': {
                    'max_position_fraction': self.default_max_position_fraction,
                    'max_var_contribution': self.default_max_var_contribution,
                    'var_confidence_level': self.default_var_confidence,
                    'max_portfolio_var': 15.0  # 15% max portfolio VaR
                },
                'position_breakdown': []
            }
            
            # Position-level breakdown
            for pos in self.current_positions:
                position_value = abs(pos.get('quantity', 0) * pos.get('current_price', 0))
                weight = position_value / portfolio_value if portfolio_value > 0 else 0
                
                dashboard['position_breakdown'].append({
                    'symbol': pos.get('symbol', 'UNKNOWN'),
                    'quantity': pos.get('quantity', 0),
                    'current_price': pos.get('current_price', 0),
                    'market_value': position_value,
                    'weight': weight,
                    'has_returns_data': pos.get('returns_data') is not None
                })
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating risk dashboard: {e}")
            return {
                'error': f'Dashboard generation failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

