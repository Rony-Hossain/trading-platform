"""
Daily P&L Attribution

Decomposes daily P&L into: alpha, timing, selection, fees, slippage, borrow
Creates API: GET /analytics/attribution?date=YYYY-MM-DD
Acceptance: Report generated for 100% trading days; stored in artifacts/reports/pnl/
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class PnLComponent(Enum):
    """P&L attribution components"""
    ALPHA = "alpha"
    TIMING = "timing"
    SELECTION = "selection"
    FEES = "fees"
    SLIPPAGE = "slippage"
    BORROW = "borrow"
    OTHER = "other"


@dataclass
class PnLAttribution:
    """
    Daily P&L attribution breakdown
    """
    date: date
    portfolio_id: str
    total_pnl: float

    # Attribution components
    alpha_pnl: float  # Pure alpha from signal quality
    timing_pnl: float  # Impact of execution timing
    selection_pnl: float  # Asset selection contribution
    fees_pnl: float  # Trading fees (negative)
    slippage_pnl: float  # Slippage costs (negative)
    borrow_pnl: float  # Borrow/lending costs (negative for shorts)
    other_pnl: float  # Unexplained residual

    # Supporting metrics
    trade_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    # Reconciliation
    attribution_sum: float = field(init=False)
    reconciliation_error: float = field(init=False)

    def __post_init__(self):
        """Calculate derived fields"""
        self.attribution_sum = (
            self.alpha_pnl +
            self.timing_pnl +
            self.selection_pnl +
            self.fees_pnl +
            self.slippage_pnl +
            self.borrow_pnl +
            self.other_pnl
        )
        self.reconciliation_error = abs(self.total_pnl - self.attribution_sum)

        # Calculate gross/net if not provided
        if self.gross_pnl == 0.0:
            self.gross_pnl = self.alpha_pnl + self.timing_pnl + self.selection_pnl
        if self.net_pnl == 0.0:
            self.net_pnl = self.total_pnl

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = asdict(self)
        result['date'] = self.date.isoformat()
        return result

    def get_component_percentages(self) -> Dict[str, float]:
        """Get each component as percentage of total P&L"""
        if abs(self.total_pnl) < 1e-10:
            return {comp.value: 0.0 for comp in PnLComponent}

        return {
            PnLComponent.ALPHA.value: (self.alpha_pnl / self.total_pnl) * 100,
            PnLComponent.TIMING.value: (self.timing_pnl / self.total_pnl) * 100,
            PnLComponent.SELECTION.value: (self.selection_pnl / self.total_pnl) * 100,
            PnLComponent.FEES.value: (self.fees_pnl / self.total_pnl) * 100,
            PnLComponent.SLIPPAGE.value: (self.slippage_pnl / self.total_pnl) * 100,
            PnLComponent.BORROW.value: (self.borrow_pnl / self.total_pnl) * 100,
            PnLComponent.OTHER.value: (self.other_pnl / self.total_pnl) * 100,
        }


class PnLAttributionEngine:
    """
    P&L Attribution Engine

    Decomposes daily P&L into contributing factors using transaction-level data
    and performance analytics.
    """

    def __init__(self,
                 reports_dir: str = "artifacts/reports/pnl",
                 max_reconciliation_error: float = 0.01):
        """
        Initialize P&L attribution engine.

        Args:
            reports_dir: Directory to store attribution reports
            max_reconciliation_error: Maximum acceptable reconciliation error ($)
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.max_reconciliation_error = max_reconciliation_error

        # Cache of attributions
        self.attribution_cache: Dict[Tuple[str, date], PnLAttribution] = {}

    def calculate_alpha_pnl(self,
                           trades_df: pd.DataFrame,
                           positions_df: pd.DataFrame) -> float:
        """
        Calculate alpha P&L from signal quality.

        Alpha P&L = Expected returns from signals - Market beta returns

        Args:
            trades_df: DataFrame with trade data
            positions_df: DataFrame with position data

        Returns:
            Alpha P&L contribution
        """
        if trades_df.empty and positions_df.empty:
            return 0.0

        alpha_pnl = 0.0

        # Method 1: From trades with signal strength
        if 'signal_strength' in trades_df.columns and 'pnl' in trades_df.columns:
            # Weight P&L by signal strength
            trades_df['alpha_contribution'] = (
                trades_df['pnl'] * trades_df['signal_strength']
            )
            alpha_pnl += trades_df['alpha_contribution'].sum()

        # Method 2: From positions with expected alpha
        if 'expected_alpha' in positions_df.columns and 'realized_pnl' in positions_df.columns:
            # Sum expected alpha that was realized
            alpha_pnl += positions_df['expected_alpha'].sum()

        return alpha_pnl

    def calculate_timing_pnl(self,
                            trades_df: pd.DataFrame,
                            benchmark_prices: pd.DataFrame) -> float:
        """
        Calculate timing P&L from execution timing decisions.

        Timing P&L = (Actual execution price - TWAP/VWAP) * Quantity

        Args:
            trades_df: DataFrame with trade data
            benchmark_prices: DataFrame with benchmark prices (TWAP/VWAP)

        Returns:
            Timing P&L contribution
        """
        if trades_df.empty:
            return 0.0

        timing_pnl = 0.0

        # Calculate timing impact for each trade
        for _, trade in trades_df.iterrows():
            symbol = trade.get('symbol')
            execution_price = trade.get('execution_price', trade.get('price'))
            quantity = trade.get('quantity', 0)
            side = trade.get('side', 'buy')

            # Get benchmark price (VWAP for the period)
            if symbol in benchmark_prices.index:
                benchmark_price = benchmark_prices.loc[symbol, 'vwap']

                # Calculate timing impact
                price_diff = execution_price - benchmark_price

                # Timing benefit if we bought below or sold above benchmark
                if side == 'buy':
                    timing_impact = -price_diff * quantity
                else:  # sell
                    timing_impact = price_diff * quantity

                timing_pnl += timing_impact

        return timing_pnl

    def calculate_selection_pnl(self,
                                positions_df: pd.DataFrame,
                                benchmark_returns: pd.Series) -> float:
        """
        Calculate selection P&L from asset selection.

        Selection P&L = Sum of (Position weight * (Asset return - Benchmark return))

        Args:
            positions_df: DataFrame with position data
            benchmark_returns: Series with benchmark returns

        Returns:
            Selection P&L contribution
        """
        if positions_df.empty:
            return 0.0

        selection_pnl = 0.0

        for _, position in positions_df.iterrows():
            symbol = position.get('symbol')
            position_value = position.get('position_value', 0)
            asset_return = position.get('return', 0)

            # Get benchmark return
            benchmark_return = benchmark_returns.get(symbol, 0.0)

            # Selection contribution
            excess_return = asset_return - benchmark_return
            selection_pnl += position_value * excess_return

        return selection_pnl

    def calculate_fees_pnl(self, trades_df: pd.DataFrame) -> float:
        """
        Calculate fees P&L (always negative).

        Args:
            trades_df: DataFrame with trade data including fees

        Returns:
            Fees P&L (negative)
        """
        if trades_df.empty:
            return 0.0

        total_fees = 0.0

        # Sum all fee columns
        fee_columns = ['commission', 'exchange_fee', 'sec_fee', 'taf_fee', 'total_fees']

        for col in fee_columns:
            if col in trades_df.columns:
                total_fees += trades_df[col].sum()

        return -abs(total_fees)  # Fees are always negative

    def calculate_slippage_pnl(self, trades_df: pd.DataFrame) -> float:
        """
        Calculate slippage P&L (typically negative).

        Slippage = (Execution price - Expected price) * Quantity

        Args:
            trades_df: DataFrame with trade data

        Returns:
            Slippage P&L (typically negative)
        """
        if trades_df.empty or 'slippage' not in trades_df.columns:
            return 0.0

        # Slippage should already be calculated per trade
        total_slippage = trades_df['slippage'].sum()

        return -abs(total_slippage)  # Slippage is a cost

    def calculate_borrow_pnl(self,
                            positions_df: pd.DataFrame,
                            borrow_costs: pd.DataFrame) -> float:
        """
        Calculate borrow/lending P&L.

        For shorts: negative (borrow cost)
        For longs: can be positive (rebate) or negative (margin interest)

        Args:
            positions_df: DataFrame with position data
            borrow_costs: DataFrame with borrow cost data

        Returns:
            Borrow P&L
        """
        if positions_df.empty:
            return 0.0

        total_borrow_cost = 0.0

        for _, position in positions_df.iterrows():
            symbol = position.get('symbol')
            quantity = position.get('quantity', 0)
            position_value = position.get('position_value', 0)

            # Check if short position
            if quantity < 0:
                # Get borrow cost for this symbol
                if symbol in borrow_costs.index:
                    daily_borrow_rate = borrow_costs.loc[symbol, 'daily_rate']
                    borrow_cost = abs(position_value) * daily_borrow_rate
                    total_borrow_cost += borrow_cost

        return -abs(total_borrow_cost)  # Borrow costs are negative

    def calculate_daily_attribution(self,
                                   date: date,
                                   portfolio_id: str,
                                   trades_df: pd.DataFrame,
                                   positions_df: pd.DataFrame,
                                   total_pnl: float,
                                   benchmark_prices: Optional[pd.DataFrame] = None,
                                   benchmark_returns: Optional[pd.Series] = None,
                                   borrow_costs: Optional[pd.DataFrame] = None) -> PnLAttribution:
        """
        Calculate complete daily P&L attribution.

        Args:
            date: Date for attribution
            portfolio_id: Portfolio identifier
            trades_df: DataFrame with trade data
            positions_df: DataFrame with position data
            total_pnl: Total P&L for the day
            benchmark_prices: Benchmark prices (TWAP/VWAP)
            benchmark_returns: Benchmark returns
            borrow_costs: Borrow cost data

        Returns:
            PnLAttribution object
        """
        # Calculate each component
        alpha_pnl = self.calculate_alpha_pnl(trades_df, positions_df)

        timing_pnl = 0.0
        if benchmark_prices is not None:
            timing_pnl = self.calculate_timing_pnl(trades_df, benchmark_prices)

        selection_pnl = 0.0
        if benchmark_returns is not None:
            selection_pnl = self.calculate_selection_pnl(positions_df, benchmark_returns)

        fees_pnl = self.calculate_fees_pnl(trades_df)
        slippage_pnl = self.calculate_slippage_pnl(trades_df)

        borrow_pnl = 0.0
        if borrow_costs is not None:
            borrow_pnl = self.calculate_borrow_pnl(positions_df, borrow_costs)

        # Calculate residual (unexplained P&L)
        explained_pnl = (
            alpha_pnl + timing_pnl + selection_pnl +
            fees_pnl + slippage_pnl + borrow_pnl
        )
        other_pnl = total_pnl - explained_pnl

        # Create attribution object
        attribution = PnLAttribution(
            date=date,
            portfolio_id=portfolio_id,
            total_pnl=total_pnl,
            alpha_pnl=alpha_pnl,
            timing_pnl=timing_pnl,
            selection_pnl=selection_pnl,
            fees_pnl=fees_pnl,
            slippage_pnl=slippage_pnl,
            borrow_pnl=borrow_pnl,
            other_pnl=other_pnl,
            trade_count=len(trades_df)
        )

        # Check reconciliation
        if attribution.reconciliation_error > self.max_reconciliation_error:
            logger.warning(
                f"High reconciliation error for {date}: "
                f"${attribution.reconciliation_error:.2f}"
            )

        # Cache the attribution
        self.attribution_cache[(portfolio_id, date)] = attribution

        return attribution

    def save_attribution_report(self, attribution: PnLAttribution) -> str:
        """
        Save attribution report to file.

        Args:
            attribution: PnLAttribution object

        Returns:
            Path to saved report
        """
        # Create date-based directory structure
        year_month = attribution.date.strftime("%Y-%m")
        date_dir = self.reports_dir / year_month
        date_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"pnl_attribution_{attribution.portfolio_id}_{attribution.date.isoformat()}.json"
        filepath = date_dir / filename

        # Save report
        report = {
            "attribution": attribution.to_dict(),
            "component_percentages": attribution.get_component_percentages(),
            "generated_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Attribution report saved to {filepath}")

        return str(filepath)

    def load_attribution_report(self,
                               portfolio_id: str,
                               date: date) -> Optional[PnLAttribution]:
        """
        Load attribution report from file.

        Args:
            portfolio_id: Portfolio identifier
            date: Date for attribution

        Returns:
            PnLAttribution object or None if not found
        """
        # Check cache first
        if (portfolio_id, date) in self.attribution_cache:
            return self.attribution_cache[(portfolio_id, date)]

        # Load from file
        year_month = date.strftime("%Y-%m")
        filename = f"pnl_attribution_{portfolio_id}_{date.isoformat()}.json"
        filepath = self.reports_dir / year_month / filename

        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            report = json.load(f)

        # Reconstruct PnLAttribution object
        attr_data = report['attribution']
        attr_data['date'] = datetime.fromisoformat(attr_data['date']).date()

        attribution = PnLAttribution(**attr_data)

        # Cache it
        self.attribution_cache[(portfolio_id, date)] = attribution

        return attribution

    def get_attribution_range(self,
                             portfolio_id: str,
                             start_date: date,
                             end_date: date) -> List[PnLAttribution]:
        """
        Get attribution reports for a date range.

        Args:
            portfolio_id: Portfolio identifier
            start_date: Start date
            end_date: End date

        Returns:
            List of PnLAttribution objects
        """
        attributions = []
        current_date = start_date

        while current_date <= end_date:
            attribution = self.load_attribution_report(portfolio_id, current_date)
            if attribution:
                attributions.append(attribution)

            current_date += timedelta(days=1)

        return attributions

    def generate_summary_statistics(self,
                                   attributions: List[PnLAttribution]) -> Dict:
        """
        Generate summary statistics for multiple attributions.

        Args:
            attributions: List of PnLAttribution objects

        Returns:
            Summary statistics dictionary
        """
        if not attributions:
            return {}

        df = pd.DataFrame([attr.to_dict() for attr in attributions])

        summary = {
            "total_days": len(attributions),
            "total_pnl": df['total_pnl'].sum(),
            "avg_daily_pnl": df['total_pnl'].mean(),
            "pnl_volatility": df['total_pnl'].std(),
            "component_totals": {
                "alpha": df['alpha_pnl'].sum(),
                "timing": df['timing_pnl'].sum(),
                "selection": df['selection_pnl'].sum(),
                "fees": df['fees_pnl'].sum(),
                "slippage": df['slippage_pnl'].sum(),
                "borrow": df['borrow_pnl'].sum(),
                "other": df['other_pnl'].sum()
            },
            "component_averages": {
                "alpha": df['alpha_pnl'].mean(),
                "timing": df['timing_pnl'].mean(),
                "selection": df['selection_pnl'].mean(),
                "fees": df['fees_pnl'].mean(),
                "slippage": df['slippage_pnl'].mean(),
                "borrow": df['borrow_pnl'].mean(),
                "other": df['other_pnl'].mean()
            },
            "total_trades": df['trade_count'].sum(),
            "avg_reconciliation_error": df['reconciliation_error'].mean(),
            "max_reconciliation_error": df['reconciliation_error'].max()
        }

        # Calculate percentages of total P&L
        if abs(summary['total_pnl']) > 1e-10:
            summary['component_percentages'] = {
                comp: (total / summary['total_pnl']) * 100
                for comp, total in summary['component_totals'].items()
            }

        return summary


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    engine = PnLAttributionEngine()

    # Mock data
    test_date = date.today()

    trades_df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'quantity': [100, 50, 75],
        'price': [150.0, 2800.0, 350.0],
        'execution_price': [150.1, 2798.0, 351.0],
        'side': ['buy', 'sell', 'buy'],
        'commission': [1.0, 2.0, 1.5],
        'slippage': [10.0, -100.0, 75.0],
        'signal_strength': [0.8, 0.9, 0.7],
        'pnl': [200.0, 500.0, 150.0]
    })

    positions_df = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'quantity': [100, -50, 75],
        'position_value': [15000, -140000, 26250],
        'expected_alpha': [100, 200, 75],
        'realized_pnl': [200, 500, 150],
        'return': [0.01, -0.02, 0.015]
    })

    total_pnl = 850.0

    # Calculate attribution
    attribution = engine.calculate_daily_attribution(
        date=test_date,
        portfolio_id="portfolio_001",
        trades_df=trades_df,
        positions_df=positions_df,
        total_pnl=total_pnl
    )

    print(f"\nDaily P&L Attribution for {test_date}:")
    print(f"Total P&L: ${attribution.total_pnl:.2f}")
    print(f"  Alpha: ${attribution.alpha_pnl:.2f}")
    print(f"  Timing: ${attribution.timing_pnl:.2f}")
    print(f"  Selection: ${attribution.selection_pnl:.2f}")
    print(f"  Fees: ${attribution.fees_pnl:.2f}")
    print(f"  Slippage: ${attribution.slippage_pnl:.2f}")
    print(f"  Borrow: ${attribution.borrow_pnl:.2f}")
    print(f"  Other: ${attribution.other_pnl:.2f}")
    print(f"Reconciliation Error: ${attribution.reconciliation_error:.4f}")

    # Save report
    report_path = engine.save_attribution_report(attribution)
    print(f"\nReport saved to: {report_path}")
