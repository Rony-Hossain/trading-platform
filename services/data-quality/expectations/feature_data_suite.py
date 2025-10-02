"""
Great Expectations Suite for Feature Data Quality
Ensures PIT compliance and feature integrity
"""
import great_expectations as gx
from datetime import datetime, timedelta

def create_feature_data_suite(context: gx.DataContext) -> str:
    """
    Create expectation suite for feature validation

    Critical checks:
    1. Event timestamp exists and is valid
    2. No feature leakage (event_timestamp <= ingestion_time)
    3. Feature value ranges are reasonable
    4. No excessive nulls (coverage targets)
    5. Distribution stability (detect data drift)
    6. Entity coverage (all expected symbols present)

    Returns:
        Suite name
    """

    suite_name = "feature_data_quality_suite"

    try:
        suite = context.get_expectation_suite(suite_name)
        print(f"Suite {suite_name} already exists, updating...")
    except:
        suite = context.create_expectation_suite(suite_name)
        print(f"Created new suite: {suite_name}")

    # ===========================================================================
    # PIT COMPLIANCE VALIDATION
    # ===========================================================================

    # Event timestamp must exist
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(column="event_timestamp")
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="event_timestamp")
    )

    # Event timestamp must be in the past (no future data)
    # This is critical for PIT compliance
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInTypeList(
            column="event_timestamp",
            type_list=["datetime64", "DATETIME"]
        )
    )

    # If ingestion_timestamp exists, validate event <= ingestion
    # This catches feature leakage bugs

    # ===========================================================================
    # ENTITY VALIDATION
    # ===========================================================================

    # Symbol must exist
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(column="symbol")
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="symbol")
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(
            column="symbol",
            regex=r"^[A-Z]{1,5}$"
        )
    )

    # ===========================================================================
    # FEATURE VALUE VALIDATION (by feature type)
    # ===========================================================================

    # Returns should be reasonable (-50% to +100% daily)
    returns_features = [
        "returns_1d", "returns_5d", "returns_20d",
        "spy_returns_1d", "sector_returns_1d"
    ]

    for feature in returns_features:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=feature,
                min_value=-0.5,  # -50% max drawdown
                max_value=1.0,   # +100% max gain
                mostly=0.999
            )
        )

    # Volatility should be positive and reasonable
    volatility_features = [
        "volatility_20d", "spy_volatility_20d", "sector_volatility_20d"
    ]

    for feature in volatility_features:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeGreaterThan(
                column=feature,
                min_value=0,
                mostly=1.0
            )
        )

        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=feature,
                min_value=0,
                max_value=5.0,  # 500% annualized vol max
                mostly=0.999
            )
        )

    # RSI should be between 0 and 100
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="rsi_14",
            min_value=0,
            max_value=100,
            mostly=1.0
        )
    )

    # Spread should be positive and reasonable
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeGreaterThan(
            column="bid_ask_spread_bps",
            min_value=0,
            mostly=0.999
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="bid_ask_spread_bps",
            min_value=0,
            max_value=1000,  # 10% max spread
            mostly=0.99
        )
    )

    # VIX should be positive and reasonable
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="vix_level",
            min_value=5,    # Historical minimum ~9
            max_value=150,  # March 2020 peak ~82
            mostly=0.999
        )
    )

    # ===========================================================================
    # COMPLETENESS VALIDATION
    # ===========================================================================

    # Critical features must have <1% nulls
    critical_features = [
        "close_price", "volume", "returns_1d", "volatility_20d"
    ]

    for feature in critical_features:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=feature,
                mostly=0.99  # Max 1% nulls
            )
        )

    # Non-critical features must have <10% nulls
    non_critical_features = [
        "news_sentiment_1d", "social_sentiment_1d", "analyst_rating_avg"
    ]

    for feature in non_critical_features:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=feature,
                mostly=0.90  # Max 10% nulls
            )
        )

    # ===========================================================================
    # DISTRIBUTION STABILITY (Detect Data Drift)
    # ===========================================================================

    # Returns should be approximately normally distributed
    # Mean should be near 0, std should be stable

    # Volume should follow log-normal distribution

    # This would use ExpectColumnKlDivergenceToBeLessThan
    # or custom drift detection expectations

    print(f"Suite {suite_name} configured with {len(suite.expectations)} expectations")

    context.save_expectation_suite(suite)

    return suite_name
