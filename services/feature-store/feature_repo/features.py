"""
Feast Feature View Definitions
All features are PIT-compliant with explicit event timestamps
"""
from datetime import timedelta
from feast import FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64, String
from entities import symbol_entity, portfolio_entity, sector_entity

# ==============================================================================
# PRICE & VOLUME FEATURES
# ==============================================================================

price_volume_source = FileSource(
    name="price_volume_source",
    path="data/features/price_volume.parquet",
    timestamp_field="event_timestamp",
)

price_volume_features = FeatureView(
    name="price_volume_features",
    entities=[symbol_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="close_price", dtype=Float64),
        Field(name="volume", dtype=Int64),
        Field(name="vwap", dtype=Float64),
        Field(name="returns_1d", dtype=Float64),
        Field(name="returns_5d", dtype=Float64),
        Field(name="returns_20d", dtype=Float64),
        Field(name="volatility_20d", dtype=Float64),
        Field(name="volume_20d_avg", dtype=Float64),
        Field(name="volume_ratio", dtype=Float64),  # volume / 20d avg
        Field(name="high_low_spread", dtype=Float64),
    ],
    source=price_volume_source,
    online=True,
)

# ==============================================================================
# MOMENTUM FEATURES
# ==============================================================================

momentum_source = FileSource(
    name="momentum_source",
    path="data/features/momentum.parquet",
    timestamp_field="event_timestamp",
)

momentum_features = FeatureView(
    name="momentum_features",
    entities=[symbol_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="rsi_14", dtype=Float64),
        Field(name="macd", dtype=Float64),
        Field(name="macd_signal", dtype=Float64),
        Field(name="macd_histogram", dtype=Float64),
        Field(name="momentum_1m", dtype=Float64),
        Field(name="momentum_3m", dtype=Float64),
        Field(name="momentum_6m", dtype=Float64),
        Field(name="momentum_12m", dtype=Float64),
        Field(name="price_vs_52w_high", dtype=Float64),
        Field(name="price_vs_52w_low", dtype=Float64),
        Field(name="trend_strength", dtype=Float64),
    ],
    source=momentum_source,
    online=True,
)

# ==============================================================================
# LIQUIDITY FEATURES
# ==============================================================================

liquidity_source = FileSource(
    name="liquidity_source",
    path="data/features/liquidity.parquet",
    timestamp_field="event_timestamp",
)

liquidity_features = FeatureView(
    name="liquidity_features",
    entities=[symbol_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="bid_ask_spread_bps", dtype=Float64),
        Field(name="effective_spread_bps", dtype=Float64),
        Field(name="market_depth_10bps", dtype=Float64),  # shares within 10bps
        Field(name="turnover_ratio", dtype=Float64),  # volume / shares outstanding
        Field(name="amihud_illiquidity", dtype=Float64),
        Field(name="roll_impact", dtype=Float64),
        Field(name="price_impact_1pct", dtype=Float64),  # cost to trade 1% ADV
        Field(name="liquidity_regime", dtype=String),  # HIGH, NORMAL, LOW, CRISIS
    ],
    source=liquidity_source,
    online=True,
)

# ==============================================================================
# FUNDAMENTAL FEATURES
# ==============================================================================

fundamental_source = FileSource(
    name="fundamental_source",
    path="data/features/fundamental.parquet",
    timestamp_field="event_timestamp",
)

fundamental_features = FeatureView(
    name="fundamental_features",
    entities=[symbol_entity],
    ttl=timedelta(days=90),  # Longer TTL for fundamentals
    schema=[
        Field(name="market_cap", dtype=Float64),
        Field(name="pe_ratio", dtype=Float64),
        Field(name="pb_ratio", dtype=Float64),
        Field(name="dividend_yield", dtype=Float64),
        Field(name="roe", dtype=Float64),
        Field(name="roa", dtype=Float64),
        Field(name="debt_to_equity", dtype=Float64),
        Field(name="current_ratio", dtype=Float64),
        Field(name="earnings_growth_yoy", dtype=Float64),
        Field(name="revenue_growth_yoy", dtype=Float64),
    ],
    source=fundamental_source,
    online=True,
)

# ==============================================================================
# SENTIMENT FEATURES
# ==============================================================================

sentiment_source = FileSource(
    name="sentiment_source",
    path="data/features/sentiment.parquet",
    timestamp_field="event_timestamp",
)

sentiment_features = FeatureView(
    name="sentiment_features",
    entities=[symbol_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="news_sentiment_1d", dtype=Float64),
        Field(name="news_sentiment_7d", dtype=Float64),
        Field(name="social_sentiment_1d", dtype=Float64),
        Field(name="social_volume_1d", dtype=Int64),
        Field(name="analyst_rating_avg", dtype=Float64),
        Field(name="analyst_target_premium", dtype=Float64),  # (target - price) / price
        Field(name="earnings_surprise_pct", dtype=Float64),
    ],
    source=sentiment_source,
    online=True,
)

# ==============================================================================
# MACRO FEATURES (Portfolio-level)
# ==============================================================================

macro_source = FileSource(
    name="macro_source",
    path="data/features/macro.parquet",
    timestamp_field="event_timestamp",
)

macro_features = FeatureView(
    name="macro_features",
    entities=[portfolio_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="spy_returns_1d", dtype=Float64),
        Field(name="spy_volatility_20d", dtype=Float64),
        Field(name="vix_level", dtype=Float64),
        Field(name="vix_change_1d", dtype=Float64),
        Field(name="term_spread", dtype=Float64),  # 10Y - 2Y
        Field(name="credit_spread", dtype=Float64),  # HY - IG
        Field(name="treasury_10y", dtype=Float64),
        Field(name="dollar_index", dtype=Float64),
        Field(name="oil_price", dtype=Float64),
        Field(name="market_regime", dtype=String),  # BULL, BEAR, VOLATILE, CRISIS
    ],
    source=macro_source,
    online=True,
)

# ==============================================================================
# SECTOR FEATURES
# ==============================================================================

sector_source = FileSource(
    name="sector_source",
    path="data/features/sector.parquet",
    timestamp_field="event_timestamp",
)

sector_features = FeatureView(
    name="sector_features",
    entities=[sector_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="sector_returns_1d", dtype=Float64),
        Field(name="sector_returns_5d", dtype=Float64),
        Field(name="sector_volatility_20d", dtype=Float64),
        Field(name="sector_momentum_3m", dtype=Float64),
        Field(name="sector_relative_strength", dtype=Float64),  # vs SPY
    ],
    source=sector_source,
    online=True,
)
