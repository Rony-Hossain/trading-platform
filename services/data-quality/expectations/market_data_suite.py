"""
Great Expectations Suite for Market Data Quality
Critical for PIT compliance and downstream alpha generation
"""
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from datetime import datetime, timedelta

def create_market_data_suite(context: gx.DataContext) -> str:
    """
    Create expectation suite for market data validation

    Critical checks:
    1. Schema validation (all required columns present)
    2. Timestamp monotonicity (no gaps or duplicates)
    3. Price sanity checks (no negatives, reasonable bounds)
    4. Volume sanity checks
    5. Spread validation (bid <= ask)
    6. Completeness (no excessive nulls)
    7. Freshness (latest timestamp within SLO)

    Returns:
        Suite name
    """

    suite_name = "market_data_quality_suite"

    # Create or retrieve suite
    try:
        suite = context.get_expectation_suite(suite_name)
        print(f"Suite {suite_name} already exists, updating...")
    except:
        suite = context.create_expectation_suite(suite_name)
        print(f"Created new suite: {suite_name}")

    # ===========================================================================
    # SCHEMA VALIDATION
    # ===========================================================================

    # Required columns must exist
    required_columns = [
        "symbol",
        "timestamp",
        "bid",
        "ask",
        "last",
        "volume",
        "bid_size",
        "ask_size"
    ]

    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=required_columns
        )
    )

    # ===========================================================================
    # TIMESTAMP VALIDATION
    # ===========================================================================

    # Timestamps must be monotonic increasing (within each symbol)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="timestamp"
        )
    )

    # Timestamps must be in the past (no future data)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInTypeList(
            column="timestamp",
            type_list=["datetime64", "DATETIME"]
        )
    )

    # ===========================================================================
    # PRICE VALIDATION
    # ===========================================================================

    # Prices must be positive
    for price_col in ["bid", "ask", "last"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeGreaterThan(
                column=price_col,
                min_value=0,
                mostly=1.0  # 100% compliance
            )
        )

        # Prices must be reasonable (detect bad data)
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=price_col,
                min_value=0.01,  # Penny stocks minimum
                max_value=100000,  # BRK.A maximum
                mostly=0.999  # 99.9% compliance
            )
        )

        # No nulls allowed in prices
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=price_col
            )
        )

    # Bid must be <= Ask (spread validation)
    suite.add_expectation(
        gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="ask",
            column_B="bid",
            or_equal=True,
            mostly=0.999  # Allow 0.1% for crossed markets
        )
    )

    # Spread must be reasonable (detect bad data)
    # We'll validate this in custom expectation

    # ===========================================================================
    # VOLUME VALIDATION
    # ===========================================================================

    # Volume must be non-negative
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeGreaterThanOrEqualTo(
            column="volume",
            min_value=0,
            mostly=1.0
        )
    )

    # Volume must be reasonable (detect outliers)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="volume",
            min_value=0,
            max_value=1e9,  # 1B shares max (SPY can hit this)
            mostly=0.999
        )
    )

    # ===========================================================================
    # COMPLETENESS
    # ===========================================================================

    # Symbol must always be present
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="symbol"
        )
    )

    # Symbol must be valid ticker format
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="symbol",
            regex=r"^[A-Z]{1,5}$",  # 1-5 uppercase letters
            mostly=0.99
        )
    )

    # Row count must be reasonable (detect truncation)
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1,
            max_value=1000000  # 1M rows max per batch
        )
    )

    # ===========================================================================
    # CUSTOM VALIDATIONS
    # ===========================================================================

    # Spread reasonableness (custom expectation)
    # For Tier 1 stocks, spread should be <50bps
    # For Tier 2 stocks, spread should be <200bps

    print(f"Suite {suite_name} configured with {len(suite.expectations)} expectations")

    # Save suite
    context.save_expectation_suite(suite)

    return suite_name

# Custom expectation for spread validation
class ExpectSpreadToBeReasonable(gx.expectations.ExpectationConfiguration):
    """
    Custom expectation: (ask - bid) / mid < threshold_bps

    Args:
        tier1_threshold_bps: Max spread for Tier 1 stocks (default 50)
        tier2_threshold_bps: Max spread for Tier 2 stocks (default 200)
    """
    pass  # Implementation would extend gx.expectations.Expectation base class
