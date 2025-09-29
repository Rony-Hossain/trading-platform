"""
Backtesting Engine - Core backtesting logic and execution
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
import quantstats as qs
import empyrical

from ..models.schemas import (
    Backtest, BacktestCreate, BacktestResult, Trade, Position,
    PerformanceMetrics, OptimizationRequest, RiskMetrics
)

logger = logging.getLogger(__name__)

@dataclass
class BacktestState:
    """Current state of backtest execution"""
    current_date: date
    cash: float
    positions: Dict[str, int]  # symbol -> quantity
    portfolio_value: float
    trades: List[Trade]
    daily_returns: List[float]
    drawdown_series: List[float]
    benchmark_returns: List[float]
    day_start_value: float
    circuit_breaker_triggered: bool
    running_peak_value: float
    circuit_breaker_log: List[Dict[str, Any]]
    slippage_config: Dict[str, Any]

class BacktestEngine:
    """Core backtesting engine with vectorized operations"""
    
    def __init__(self):
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.001   # 0.1% slippage
        self.min_trade_size = 1
        self.max_position_size = 0.2  # 20% of portfolio
        
    def create_backtest(self, db: Session, backtest_create: BacktestCreate) -> Backtest:
        """Create new backtest record"""
        # Implementation would store in database
        # For now, return mock backtest
        return Backtest(
            id=1,
            strategy_id=backtest_create.strategy_id,
            symbol=backtest_create.symbol,
            start_date=backtest_create.start_date,
            end_date=backtest_create.end_date,
            initial_capital=backtest_create.initial_capital,
            parameters=backtest_create.parameters,
            status="created",
            created_at=datetime.now()
        )
    
    def get_backtest(self, db: Session, backtest_id: int) -> Optional[Backtest]:
        """Get backtest by ID"""
        # Implementation would query database
        return None
    
    async def run_backtest_async(self, backtest_id: int, db: Session):
        """Run backtest asynchronously"""
        try:
            backtest = self.get_backtest(db, backtest_id)
            if not backtest:
                raise ValueError(f"Backtest {backtest_id} not found")
            
            # Update status to running
            self.update_backtest_status(db, backtest_id, "running")
            
            # Run the backtest
            result = await self.run_backtest(backtest, db)
            
            # Store results and update status
            self.store_backtest_results(db, backtest_id, result)
            self.update_backtest_status(db, backtest_id, "completed")
            
            logger.info(f"Backtest {backtest_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Backtest {backtest_id} failed: {e}")
            self.update_backtest_status(db, backtest_id, "failed")
            self.store_backtest_error(db, backtest_id, str(e))
    
    async def run_backtest(self, backtest: Backtest, db: Session) -> BacktestResult:
        """Execute backtest with given parameters"""
        try:
            # Load market data
            market_data = await self.load_market_data(
                backtest.symbol, backtest.start_date, backtest.end_date
            )
            
            if market_data.empty:
                raise ValueError(f"No market data available for {backtest.symbol}")
            
            # Load strategy
            strategy = self.load_strategy(backtest.strategy_id, backtest.parameters)

            # Risk configuration
            parameters = backtest.parameters or {}
            risk_params = parameters.get('risk', {}) if isinstance(parameters, dict) else {}
            daily_loss_limit = float(risk_params.get('daily_loss_limit', 0.03))
            max_drawdown_limit = float(risk_params.get('max_drawdown_limit', 0.2))
            resume_next_day = bool(risk_params.get('resume_after_hit', True))

            execution_params = parameters.get('execution', {}) if isinstance(parameters, dict) else {}
            slippage_config = execution_params.get('slippage', {}) if isinstance(execution_params, dict) else {}
            if not isinstance(slippage_config, dict):
                slippage_config = {}

            # Initialize backtest state
            state = BacktestState(
                current_date=backtest.start_date,
                cash=backtest.initial_capital,
                positions={},
                portfolio_value=backtest.initial_capital,
                trades=[],
                daily_returns=[],
                drawdown_series=[],
                benchmark_returns=[],
                day_start_value=backtest.initial_capital,
                circuit_breaker_triggered=False,
                running_peak_value=backtest.initial_capital,
                circuit_breaker_log=[],
                slippage_config=slippage_config
            )

            # Load benchmark data (SPY)
            benchmark_data = await self.load_market_data("SPY", backtest.start_date, backtest.end_date)

            # Run day-by-day simulation
            current_day = None
            drawdown_break_triggered = False
            for i, (date, row) in enumerate(market_data.iterrows()):
                # Update current date
                state.current_date = date.date() if hasattr(date, 'date') else date

                # Reset or check circuit breakers for new trading day
                loop_day = date.date() if hasattr(date, 'date') else date
                if current_day != loop_day:
                    current_day = loop_day
                    state.day_start_value = state.portfolio_value
                    if resume_next_day:
                        state.circuit_breaker_triggered = False

                # Determine if trading is allowed
                if state.circuit_breaker_triggered or drawdown_break_triggered:
                    signals = {}
                else:
                    signals = strategy.generate_signals(market_data.iloc[:i+1], state)

                # Execute trades based on signals
                trades = self.execute_signals(signals, row, state, backtest.symbol)
                state.trades.extend(trades)
                
                # Update portfolio value
                portfolio_value = self.calculate_portfolio_value(state, row, backtest.symbol)

                # Calculate daily return
                if state.portfolio_value > 0:
                    daily_return = (portfolio_value - state.portfolio_value) / state.portfolio_value
                    state.daily_returns.append(daily_return)

                state.portfolio_value = portfolio_value

                # Track running peak for drawdown calculations
                state.running_peak_value = max(state.running_peak_value, state.portfolio_value)

                # Daily loss breaker
                if state.day_start_value > 0:
                    daily_change = (state.portfolio_value - state.day_start_value) / state.day_start_value
                    if daily_change <= -daily_loss_limit and not state.circuit_breaker_triggered:
                        state.circuit_breaker_triggered = True
                        state.circuit_breaker_log.append({
                            'type': 'daily_loss',
                            'date': str(state.current_date),
                            'loss_pct': float(daily_change)
                        })
                        logger.warning("Daily loss limit reached (%.2f%%) on %s", daily_change * 100, state.current_date)
                        self._flatten_positions(state, row)
                        state.portfolio_value = self.calculate_portfolio_value(state, row, backtest.symbol)
                        state.cash = state.portfolio_value
                        drawdown_after_limit = (state.portfolio_value - state.running_peak_value) / state.running_peak_value if state.running_peak_value > 0 else 0
                        state.drawdown_series.append(drawdown_after_limit)
                        if not resume_next_day:
                            drawdown_break_triggered = True
                        continue

                # Portfolio drawdown breaker
                if state.running_peak_value > 0:
                    drawdown = (state.portfolio_value - state.running_peak_value) / state.running_peak_value
                else:
                    drawdown = 0
                state.drawdown_series.append(drawdown)

                if drawdown <= -max_drawdown_limit and not drawdown_break_triggered:
                    drawdown_break_triggered = True
                    state.circuit_breaker_log.append({
                        'type': 'max_drawdown',
                        'date': str(state.current_date),
                        'drawdown_pct': float(drawdown)
                    })
                    logger.error("Max drawdown limit reached (%.2f%%). Flattening positions and stopping backtest.", drawdown * 100)
                    self._flatten_positions(state, row)
                    state.portfolio_value = self.calculate_portfolio_value(state, row, backtest.symbol)
                    state.cash = state.portfolio_value
                    break

                # Calculate benchmark return
                if i > 0 and not benchmark_data.empty:
                    if i < len(benchmark_data):
                        benchmark_return = (benchmark_data.iloc[i]['close'] - benchmark_data.iloc[i-1]['close']) / benchmark_data.iloc[i-1]['close']
                        state.benchmark_returns.append(benchmark_return)
            
            # Calculate final metrics
            metrics = self.calculate_performance_metrics(state, backtest)
            
            return BacktestResult(
                backtest_id=backtest.id,
                total_return=metrics['total_return'],
                annual_return=metrics['annual_return'],
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                total_trades=len(state.trades),
                final_portfolio_value=state.portfolio_value,
                trades=state.trades,
                daily_returns=state.daily_returns,
                benchmark_returns=state.benchmark_returns,
                drawdown_series=state.drawdown_series,
                metadata={"strategy_params": backtest.parameters, "circuit_breakers": state.circuit_breaker_log, "slippage_config": state.slippage_config}
            )
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    async def load_market_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Load market data for backtesting"""
        try:
            # This would typically fetch from your market data service
            # For now, simulate with yfinance
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            # Convert to standard format
            data.columns = [col.lower() for col in data.columns]
            data.reset_index(inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def load_strategy(self, strategy_id: int, parameters: Dict[str, Any]):
        """Load strategy implementation"""
        # This would load the actual strategy class
        # For now, return a simple moving average strategy
        return SimpleMovingAverageStrategy(parameters)
    
    def _flatten_positions(self, state: BacktestState, market_row: pd.Series) -> None:
        """Flatten all open positions at the current price."""
        flattened_trades = []
        try:
            for pos_symbol, quantity in list(state.positions.items()):
                if quantity <= 0:
                    continue
                trade = self.execute_sell_signal({'quantity': quantity}, market_row, state, pos_symbol)
                if trade:
                    flattened_trades.append(trade)
            if flattened_trades:
                state.trades.extend(flattened_trades)
        except Exception as exc:
            logger.error(f"Error flattening positions: {exc}")

    def _calculate_slippage_cost(self, notional: float, price: float, quantity: int, config: Dict[str, Any]) -> float:
        """Compute slippage cost according to configured model."""
        cfg = config or {}
        mode = str(cfg.get('type', 'percentage')).lower()

        try:
            if mode in {'per_share', 'per_share_fee', 'share'}:
                per_share = float(cfg.get('per_share', cfg.get('amount', 0.0)))
                return max(0.0, per_share * quantity)

            if mode in {'fixed', 'flat'}:
                amount = float(cfg.get('amount', 0.0))
                return max(0.0, amount)

            if mode in {'bps', 'basis_points'}:
                bps = float(cfg.get('bps', cfg.get('basis_points', self.slippage * 10000)))
                rate = bps / 10000.0
                return max(0.0, notional * rate)

            if mode in {'percentage', 'percent', 'rate'}:
                rate = float(cfg.get('rate', self.slippage))
                return max(0.0, notional * rate)

            if mode in {'per_share_percent', 'share_percent'}:
                rate = float(cfg.get('rate', self.slippage))
                return max(0.0, price * rate * quantity)

        except Exception as exc:
            logger.warning('Slippage configuration error: %s', exc)

        return max(0.0, notional * self.slippage)

    def execute_signals(self, signals: Dict[str, Any], market_row: pd.Series, 
                       state: BacktestState, symbol: str) -> List[Trade]:
        """Execute trading signals"""
        trades = []
        
        try:
            for signal_type, signal_data in signals.items():
                if signal_type == "BUY":
                    trade = self.execute_buy_signal(signal_data, market_row, state, symbol)
                    if trade:
                        trades.append(trade)
                        
                elif signal_type == "SELL":
                    trade = self.execute_sell_signal(signal_data, market_row, state, symbol)
                    if trade:
                        trades.append(trade)
                        
        except Exception as e:
            logger.error(f"Error executing signals: {e}")
            
        return trades
    
    def execute_buy_signal(self, signal_data: Dict, market_row: pd.Series, 
                          state: BacktestState, symbol: str) -> Optional[Trade]:
        """Execute buy signal"""
        try:
            price = market_row['close']
            quantity = signal_data.get('quantity', 0)
            
            if quantity <= 0:
                return None
            
            # Apply position sizing limits
            max_position_value = state.portfolio_value * self.max_position_size
            max_quantity = int(max_position_value / price)
            quantity = min(quantity, max_quantity)
            
            # Determine feasible quantity respecting available cash
            feasible_quantity = max(quantity, self.min_trade_size)
            chosen_quantity = 0
            commission_cost = 0.0
            slippage_cost = 0.0
            total_cost = 0.0

            while feasible_quantity >= self.min_trade_size:
                cost = price * feasible_quantity
                commission_cost = cost * self.commission
                slippage_cost = self._calculate_slippage_cost(cost, price, feasible_quantity, state.slippage_config)
                total_cost = cost + commission_cost + slippage_cost

                if total_cost <= state.cash + 1e-9:
                    chosen_quantity = feasible_quantity
                    break

                feasible_quantity -= 1

            if chosen_quantity < self.min_trade_size:
                return None

            quantity = chosen_quantity
            cost = price * quantity
            commission_cost = cost * self.commission
            slippage_cost = self._calculate_slippage_cost(cost, price, quantity, state.slippage_config)
            total_cost = cost + commission_cost + slippage_cost
            # Execute trade
            state.cash -= total_cost
            current_position = state.positions.get(symbol, 0)
            state.positions[symbol] = current_position + quantity
            
            return Trade(
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                price=price,
                timestamp=state.current_date,
                commission=commission_cost,
                slippage=slippage_cost,
                total_cost=total_cost
            )
            
        except Exception as e:
            logger.error(f"Error executing buy signal: {e}")
            return None
    
    def execute_sell_signal(self, signal_data: Dict, market_row: pd.Series,
                           state: BacktestState, symbol: str) -> Optional[Trade]:
        """Execute sell signal"""
        try:
            price = market_row['close']
            quantity = signal_data.get('quantity', 0)
            
            current_position = state.positions.get(symbol, 0)
            if current_position <= 0:
                return None
            
            # Limit sell quantity to current position
            quantity = min(quantity, current_position)
            
            if quantity < self.min_trade_size:
                return None
            
            # Calculate proceeds
            gross_proceeds = price * quantity
            commission_cost = gross_proceeds * self.commission
            slippage_cost = self._calculate_slippage_cost(gross_proceeds, price, quantity, state.slippage_config)
            net_proceeds = gross_proceeds - commission_cost - slippage_cost
            
            # Execute trade
            state.cash += net_proceeds
            state.positions[symbol] = current_position - quantity
            
            return Trade(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=price,
                timestamp=state.current_date,
                commission=commission_cost,
                slippage=slippage_cost,
                total_cost=net_proceeds
            )
            
        except Exception as e:
            logger.error(f"Error executing sell signal: {e}")
            return None
    
    def calculate_portfolio_value(self, state: BacktestState, market_row: pd.Series, symbol: str) -> float:
        """Calculate current portfolio value"""
        try:
            position_value = 0
            
            # Calculate value of all positions
            for pos_symbol, quantity in state.positions.items():
                if pos_symbol == symbol:
                    position_value += quantity * market_row['close']
                else:
                    # For simplicity, assume other positions maintain their value
                    # In reality, you'd need current prices for all symbols
                    position_value += quantity * 100  # Placeholder
            
            return state.cash + position_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return state.cash
    
    def calculate_performance_metrics(self, state: BacktestState, backtest: Backtest) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            if not state.daily_returns:
                return self._empty_metrics()
            
            returns = np.array(state.daily_returns)
            
            # Basic metrics
            total_return = (state.portfolio_value - backtest.initial_capital) / backtest.initial_capital
            
            # Annualized return
            trading_days = len(returns)
            years = trading_days / 252  # Assuming 252 trading days per year
            annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            
            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = empyrical.sharpe_ratio(returns) if len(returns) > 1 else 0
            max_drawdown = abs(min(state.drawdown_series)) if state.drawdown_series else 0
            
            # Trade metrics
            winning_trades = [t for t in state.trades if self._is_winning_trade(t)]
            total_trades = len(state.trades)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(t.total_cost for t in winning_trades)
            gross_loss = sum(abs(t.total_cost) for t in state.trades if not self._is_winning_trade(t))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'avg_trade_return': total_return / total_trades if total_trades > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._empty_metrics()
    
    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if trade was profitable"""
        # This is simplified - in reality you'd need to track entry/exit pairs
        return trade.side == "SELL" and trade.total_cost > 0
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'avg_trade_return': 0.0
        }
    
    # Placeholder methods for database operations
    def update_backtest_status(self, db: Session, backtest_id: int, status: str):
        """Update backtest status in database"""
        pass
    
    def store_backtest_results(self, db: Session, backtest_id: int, result: BacktestResult):
        """Store backtest results in database"""
        pass
    
    def store_backtest_error(self, db: Session, backtest_id: int, error: str):
        """Store backtest error in database"""
        pass
    
    def get_backtest_results(self, db: Session, backtest_id: int, 
                           include_trades: bool, include_metrics: bool) -> BacktestResult:
        """Get stored backtest results"""
        # Placeholder implementation
        return BacktestResult()
    
    def get_backtest_trades(self, db: Session, backtest_id: int, skip: int, limit: int) -> List[Trade]:
        """Get trades from backtest"""
        return []
    
    async def calculate_performance_metrics(self, db: Session, backtest_id: int, 
                                          benchmark_symbol: str) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Placeholder implementation
        return PerformanceMetrics()
    
    def start_optimization(self, db: Session, request: OptimizationRequest) -> int:
        """Start parameter optimization"""
        return 1  # Return optimization ID
    
    async def run_optimization_async(self, optimization_id: int, db: Session):
        """Run parameter optimization"""
        pass
    
    def get_optimization_results(self, db: Session, optimization_id: int):
        """Get optimization results"""
        return {}
    
    async def compare_backtests(self, db: Session, backtest_ids: List[int], metrics: List[str]):
        """Compare multiple backtests"""
        return {}
    
    def get_strategy_leaderboard(self, db: Session, metric: str, symbol: Optional[str],
                               timeframe: Optional[str], limit: int):
        """Get strategy performance leaderboard"""
        return []
    
    def get_backtest_count(self, db: Session) -> int:
        """Get total number of backtests"""
        return 0
    
    def is_healthy(self) -> bool:
        """Check if backtest engine is healthy"""
        try:
            # Quick test of core functionality
            test_returns = [0.01, -0.005, 0.02, -0.01]
            sharpe = empyrical.sharpe_ratio(test_returns)
            return not np.isnan(sharpe)
        except Exception:
            return False


class SimpleMovingAverageStrategy:
    """Simple moving average crossover strategy for testing"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.short_window = parameters.get('short_window', 20)
        self.long_window = parameters.get('long_window', 50)
        self.position_size = parameters.get('position_size', 0.1)  # 10% of portfolio
    
    def generate_signals(self, market_data: pd.DataFrame, state: BacktestState) -> Dict[str, Any]:
        """Generate trading signals based on moving average crossover"""
        signals = {}
        
        if len(market_data) < self.long_window:
            return signals
        
        # Calculate moving averages
        short_ma = market_data['close'].rolling(window=self.short_window).mean().iloc[-1]
        long_ma = market_data['close'].rolling(window=self.long_window).mean().iloc[-1]
        
        # Previous values for crossover detection
        if len(market_data) >= self.long_window + 1:
            prev_short_ma = market_data['close'].rolling(window=self.short_window).mean().iloc[-2]
            prev_long_ma = market_data['close'].rolling(window=self.long_window).mean().iloc[-2]
            
            current_price = market_data['close'].iloc[-1]
            
            # Buy signal: short MA crosses above long MA
            if short_ma > long_ma and prev_short_ma <= prev_long_ma:
                position_value = state.portfolio_value * self.position_size
                quantity = int(position_value / current_price)
                signals['BUY'] = {'quantity': quantity, 'reason': 'MA_CROSSOVER_UP'}
            
            # Sell signal: short MA crosses below long MA
            elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
                # Sell current position
                current_position = state.positions.get(market_data.iloc[-1].name, 0)
                if current_position > 0:
                    signals['SELL'] = {'quantity': current_position, 'reason': 'MA_CROSSOVER_DOWN'}
        
        return signals
