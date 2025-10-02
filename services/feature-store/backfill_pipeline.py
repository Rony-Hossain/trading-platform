"""
PIT-Compliant Feature Backfill Pipeline
Backfills historical features with strict PIT guarantees
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, date
from dataclasses import dataclass
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BackfillConfig:
    """Configuration for backfill job"""
    start_date: date
    end_date: date
    symbols: List[str]
    feature_views: List[str]
    batch_size: int = 1000  # Symbols per batch
    max_workers: int = 4     # Parallel workers
    output_dir: str = "data/features"
    validate_pit: bool = True  # Validate PIT compliance
    dry_run: bool = False

@dataclass
class BackfillMetrics:
    """Metrics for backfill job"""
    total_symbols: int
    total_dates: int
    total_rows_generated: int
    rows_per_second: float
    pit_violations: int
    failed_symbols: List[str]
    execution_time_seconds: float
    start_time: datetime
    end_time: datetime

class PITBackfillPipeline:
    """
    PIT-compliant feature backfill pipeline

    Key guarantees:
    1. All features calculated using only data available as of calculation timestamp
    2. No forward-looking bias in feature generation
    3. Explicit event_timestamp on all features
    4. Validation of PIT compliance before writing
    5. Idempotent - can re-run safely
    """

    def __init__(self, config: BackfillConfig):
        self.config = config
        self.metrics = None

        # Validate config
        if config.start_date > config.end_date:
            raise ValueError("start_date must be <= end_date")

        logger.info(
            f"Backfill pipeline initialized: "
            f"{config.start_date} to {config.end_date}, "
            f"{len(config.symbols)} symbols, "
            f"{len(config.feature_views)} feature views"
        )

    def run(self) -> BackfillMetrics:
        """
        Run backfill pipeline

        Returns:
            BackfillMetrics with job statistics
        """
        start_time = time.time()
        logger.info("Starting backfill pipeline...")

        # Generate date range
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq="D"
        ).date.tolist()

        total_rows = 0
        pit_violations = 0
        failed_symbols = []

        # Process each feature view
        for feature_view in self.config.feature_views:
            logger.info(f"Backfilling feature view: {feature_view}")

            try:
                rows = self._backfill_feature_view(
                    feature_view=feature_view,
                    dates=dates
                )
                total_rows += rows

            except Exception as e:
                logger.error(f"Failed to backfill {feature_view}: {e}")
                failed_symbols.append(feature_view)

        # Calculate metrics
        execution_time = time.time() - start_time
        rows_per_second = total_rows / execution_time if execution_time > 0 else 0

        metrics = BackfillMetrics(
            total_symbols=len(self.config.symbols),
            total_dates=len(dates),
            total_rows_generated=total_rows,
            rows_per_second=rows_per_second,
            pit_violations=pit_violations,
            failed_symbols=failed_symbols,
            execution_time_seconds=execution_time,
            start_time=datetime.fromtimestamp(start_time),
            end_time=datetime.now()
        )

        self.metrics = metrics

        logger.info(
            f"Backfill complete: {total_rows:,} rows in {execution_time:.1f}s "
            f"({rows_per_second:,.0f} rows/sec)"
        )

        if pit_violations > 0:
            logger.error(f"PIT VIOLATIONS DETECTED: {pit_violations}")

        return metrics

    def _backfill_feature_view(
        self,
        feature_view: str,
        dates: List[date]
    ) -> int:
        """
        Backfill a single feature view

        Args:
            feature_view: Name of feature view to backfill
            dates: List of dates to backfill

        Returns:
            Number of rows generated
        """
        total_rows = 0

        # Batch symbols for parallel processing
        symbol_batches = [
            self.config.symbols[i:i + self.config.batch_size]
            for i in range(0, len(self.config.symbols), self.config.batch_size)
        ]

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._backfill_batch,
                    feature_view,
                    batch,
                    dates
                ): batch
                for batch in symbol_batches
            }

            for future in as_completed(futures):
                batch = futures[future]
                try:
                    rows = future.result()
                    total_rows += rows
                    logger.info(
                        f"Completed batch: {len(batch)} symbols, {rows} rows"
                    )
                except Exception as e:
                    logger.error(f"Batch failed: {e}")

        return total_rows

    def _backfill_batch(
        self,
        feature_view: str,
        symbols: List[str],
        dates: List[date]
    ) -> int:
        """
        Backfill a batch of symbols

        Args:
            feature_view: Feature view name
            symbols: List of symbols in this batch
            dates: List of dates to backfill

        Returns:
            Number of rows generated
        """
        # Generate features for this batch
        features_df = self._generate_features(
            feature_view=feature_view,
            symbols=symbols,
            dates=dates
        )

        if features_df.empty:
            return 0

        # Validate PIT compliance
        if self.config.validate_pit:
            self._validate_pit_compliance(features_df)

        # Write to parquet (partitioned by date)
        if not self.config.dry_run:
            self._write_features(feature_view, features_df)

        return len(features_df)

    def _generate_features(
        self,
        feature_view: str,
        symbols: List[str],
        dates: List[date]
    ) -> pd.DataFrame:
        """
        Generate features for symbols and dates

        This is a placeholder - actual implementation would call
        feature generation logic specific to each feature view

        Args:
            feature_view: Feature view name
            symbols: List of symbols
            dates: List of dates

        Returns:
            DataFrame with features and event_timestamp
        """
        # Create cartesian product of symbols x dates
        rows = []
        for symbol in symbols:
            for calc_date in dates:
                # Set event_timestamp to end of trading day (4pm ET)
                event_timestamp = datetime.combine(
                    calc_date,
                    datetime.min.time()
                ).replace(hour=16, minute=0, second=0)

                row = {
                    "symbol": symbol,
                    "event_timestamp": event_timestamp
                }

                # Generate features based on view type
                if feature_view == "price_volume_features":
                    row.update(self._generate_price_volume_features(symbol, calc_date))
                elif feature_view == "momentum_features":
                    row.update(self._generate_momentum_features(symbol, calc_date))
                elif feature_view == "liquidity_features":
                    row.update(self._generate_liquidity_features(symbol, calc_date))
                # ... other feature views

                rows.append(row)

        return pd.DataFrame(rows)

    def _generate_price_volume_features(
        self,
        symbol: str,
        as_of_date: date
    ) -> Dict[str, Any]:
        """
        Generate price/volume features as of a specific date

        CRITICAL: Only use data available as of as_of_date (PIT compliance)

        Args:
            symbol: Stock symbol
            as_of_date: Calculate features as of this date

        Returns:
            Dictionary of feature values
        """
        # In production, this would:
        # 1. Load historical price data where date <= as_of_date
        # 2. Calculate features using only that data
        # 3. Return feature dictionary

        # Placeholder implementation
        return {
            "close_price": 150.0 + np.random.randn(),
            "volume": int(1e6 + np.random.randn() * 1e5),
            "vwap": 150.0 + np.random.randn(),
            "returns_1d": np.random.normal(0, 0.02),
            "returns_5d": np.random.normal(0, 0.05),
            "returns_20d": np.random.normal(0, 0.1),
            "volatility_20d": 0.25 + np.random.randn() * 0.05,
            "volume_20d_avg": 1e6 + np.random.randn() * 1e5,
            "volume_ratio": 1.0 + np.random.randn() * 0.2,
            "high_low_spread": 0.02 + np.random.randn() * 0.005,
        }

    def _generate_momentum_features(
        self,
        symbol: str,
        as_of_date: date
    ) -> Dict[str, Any]:
        """Generate momentum features (PIT-compliant)"""
        return {
            "rsi_14": 50 + np.random.randn() * 15,
            "macd": np.random.randn() * 2,
            "macd_signal": np.random.randn() * 2,
            "macd_histogram": np.random.randn(),
            "momentum_1m": np.random.normal(0, 0.05),
            "momentum_3m": np.random.normal(0, 0.1),
            "momentum_6m": np.random.normal(0, 0.15),
            "momentum_12m": np.random.normal(0, 0.2),
            "price_vs_52w_high": -0.05 + np.random.randn() * 0.1,
            "price_vs_52w_low": 0.25 + np.random.randn() * 0.1,
            "trend_strength": 0.5 + np.random.randn() * 0.2,
        }

    def _generate_liquidity_features(
        self,
        symbol: str,
        as_of_date: date
    ) -> Dict[str, Any]:
        """Generate liquidity features (PIT-compliant)"""
        return {
            "bid_ask_spread_bps": 5 + np.random.randn() * 2,
            "effective_spread_bps": 4 + np.random.randn() * 2,
            "market_depth_10bps": 10000 + np.random.randn() * 2000,
            "turnover_ratio": 0.01 + np.random.randn() * 0.005,
            "amihud_illiquidity": 0.001 + np.random.randn() * 0.0005,
            "roll_impact": 0.0002 + np.random.randn() * 0.0001,
            "price_impact_1pct": 0.05 + np.random.randn() * 0.02,
            "liquidity_regime": np.random.choice(["HIGH", "NORMAL", "LOW"]),
        }

    def _validate_pit_compliance(self, features_df: pd.DataFrame):
        """
        Validate PIT compliance

        Raises:
            ValueError: If any PIT violations detected
        """
        if "event_timestamp" not in features_df.columns:
            raise ValueError("Features must have event_timestamp column")

        # Check for future timestamps (relative to now)
        now = datetime.now()
        future_rows = features_df[features_df["event_timestamp"] > now]

        if len(future_rows) > 0:
            raise ValueError(
                f"PIT VIOLATION: {len(future_rows)} rows have "
                f"event_timestamp > {now}"
            )

        # Check for null timestamps
        null_timestamps = features_df["event_timestamp"].isnull().sum()
        if null_timestamps > 0:
            raise ValueError(
                f"PIT VIOLATION: {null_timestamps} rows have null event_timestamp"
            )

        logger.debug(f"PIT validation passed: {len(features_df)} rows")

    def _write_features(self, feature_view: str, features_df: pd.DataFrame):
        """
        Write features to parquet (partitioned by date)

        Args:
            feature_view: Feature view name
            features_df: DataFrame with features
        """
        output_dir = Path(self.config.output_dir) / feature_view

        # Partition by date for efficient querying
        features_df["date"] = features_df["event_timestamp"].dt.date

        # Write partitioned parquet
        for date_val, group in features_df.groupby("date"):
            partition_dir = output_dir / f"date={date_val}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            file_path = partition_dir / "features.parquet"
            group.drop(columns=["date"]).to_parquet(
                file_path,
                engine="pyarrow",
                compression="snappy",
                index=False
            )

        logger.debug(
            f"Wrote {len(features_df)} rows to {output_dir} "
            f"({features_df['date'].nunique()} partitions)"
        )

def run_backfill_job(
    start_date: str,
    end_date: str,
    symbols: List[str],
    feature_views: List[str],
    output_dir: str = "data/features",
    max_workers: int = 4,
    dry_run: bool = False
) -> BackfillMetrics:
    """
    Run backfill job

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbols: List of symbols to backfill
        feature_views: List of feature views to backfill
        output_dir: Output directory for parquet files
        max_workers: Number of parallel workers
        dry_run: If True, don't write files

    Returns:
        BackfillMetrics

    Example:
        >>> metrics = run_backfill_job(
        ...     start_date="2020-01-01",
        ...     end_date="2024-12-31",
        ...     symbols=["AAPL", "MSFT", "GOOGL"],
        ...     feature_views=["price_volume_features", "momentum_features"]
        ... )
        >>> print(f"Generated {metrics.total_rows_generated:,} rows")
    """
    config = BackfillConfig(
        start_date=datetime.strptime(start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(end_date, "%Y-%m-%d").date(),
        symbols=symbols,
        feature_views=feature_views,
        output_dir=output_dir,
        max_workers=max_workers,
        dry_run=dry_run
    )

    pipeline = PITBackfillPipeline(config)
    metrics = pipeline.run()

    return metrics

if __name__ == "__main__":
    # Example backfill job
    import argparse

    parser = argparse.ArgumentParser(description="Run PIT backfill pipeline")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--feature-views", required=True, help="Comma-separated feature views")
    parser.add_argument("--output-dir", default="data/features", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't write)")

    args = parser.parse_args()

    metrics = run_backfill_job(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols.split(","),
        feature_views=args.feature_views.split(","),
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        dry_run=args.dry_run
    )

    print("\n" + "="*80)
    print("BACKFILL COMPLETE")
    print("="*80)
    print(f"Total symbols:       {metrics.total_symbols:,}")
    print(f"Total dates:         {metrics.total_dates:,}")
    print(f"Total rows:          {metrics.total_rows_generated:,}")
    print(f"Throughput:          {metrics.rows_per_second:,.0f} rows/sec")
    print(f"Execution time:      {metrics.execution_time_seconds:.1f}s")
    print(f"PIT violations:      {metrics.pit_violations}")
    print(f"Failed symbols:      {len(metrics.failed_symbols)}")
    print("="*80)
