# Feature Store Service

PIT-compliant feature serving using Feast with Redis online store.

## Architecture

```
┌─────────────────┐
│  Offline Store  │  ← Historical features for training
│  (Parquet)      │
└─────────────────┘
         ↓
    Materialization
         ↓
┌─────────────────┐
│  Online Store   │  ← Real-time inference (<10ms p95)
│  (Redis)        │
└─────────────────┘
```

## Setup

```bash
# Install Feast
pip install feast[redis]

# Initialize feature repo
cd feature_repo
feast apply

# Materialize features to online store
feast materialize-incremental $(date +%Y-%m-%d)
```

## Features

### Price & Volume (5min freshness)
- close_price, vwap, volume
- returns (1d, 5d, 20d)
- volatility_20d
- volume_ratio

### Momentum (1hr freshness)
- RSI, MACD, trend strength
- Momentum (1m, 3m, 6m, 12m)
- 52-week high/low position

### Liquidity (15min freshness)
- bid_ask_spread_bps
- market_depth, turnover_ratio
- Amihud illiquidity
- price_impact_1pct
- liquidity_regime

### Fundamental (1day freshness)
- PE, PB ratios
- ROE, ROA
- Debt ratios
- Growth metrics

### Sentiment (4hr freshness)
- News sentiment (1d, 7d)
- Social sentiment
- Analyst ratings

### Macro (portfolio-level, 1hr freshness)
- SPY returns/volatility
- VIX, term spread, credit spread
- Treasury yields
- market_regime

## Usage

```python
from feature_client import FeatureStoreClient

client = FeatureStoreClient(repo_path="feature_repo")

# Online features (for inference)
features = client.get_online_features(
    feature_refs=[
        "price_volume_features:close_price",
        "momentum_features:rsi_14",
        "liquidity_features:bid_ask_spread_bps"
    ],
    entity_rows=[{"symbol": "AAPL"}],
    as_of_timestamp=datetime.now()  # PIT compliance
)

# Historical features (for training)
entity_df = pd.DataFrame({
    "symbol": ["AAPL", "MSFT"] * 100,
    "event_timestamp": pd.date_range("2024-01-01", periods=200, freq="1H")
})

training_data = client.get_historical_features(
    entity_df=entity_df,
    feature_refs=["price_volume_features:close_price", ...]
)
```

## PIT Guarantees

1. **Strict Validation**: All features validated that `event_timestamp <= as_of_timestamp`
2. **Automatic Rejection**: PIT violations raise `ValueError` immediately
3. **Zero Leakage**: No future data can leak into historical retrievals

## Monitoring

```python
# Check freshness
freshness = client.get_feature_freshness("price_volume_features")

# Check coverage
coverage = client.validate_feature_coverage(
    feature_refs=["price_volume_features:close_price"],
    entity_rows=[{"symbol": s} for s in ["AAPL", "MSFT", ...]]
)

# Retrieval stats
stats = client.get_retrieval_stats(window_minutes=60)
# Returns: p50/p95/p99 latency, null rates, PIT violations
```

## Acceptance Criteria

- ✅ 100% PIT compliance (zero feature leakage)
- ✅ Online retrieval <10ms p95 for 100 features
- ✅ Offline generation >1M rows/min
- ✅ Feature coverage ≥99% for Tier 1 symbols
- ✅ Freshness SLOs per feature view
- ✅ Automatic staleness alerts

## Data Pipeline

Features are materialized on schedule:

```bash
# Cron job (every 5 minutes for price/volume)
*/5 * * * * cd /app/feature_repo && feast materialize-incremental $(date +%Y-%m-%d)
```

## Backfill

For historical backfills, see `backfill_pipeline.py`
