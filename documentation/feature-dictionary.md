# Feature Dictionary – Advanced Forecasting Stack

This dictionary captures feature categories, a canonical name, a short formula/spec, the default time window, suggested data sources, and notes.

## 1. Options & Volatility Surface

| Feature | Category | Formula / Definition | Window | Source | Notes |
|---------|----------|----------------------|--------|--------|-------|
| `iv30` | options_vol | 30-day implied volatility (ATM) | daily | Options vendor (OPRA/CBOE) or Finnhub derivatives | Use interpolated IV surface. |
| `iv60` | options_vol | 60-day implied volatility (ATM) | daily | Options vendor | Needed for term-structure slope. |
| `iv_rank_1y` | options_vol | (current_iv30 - min_iv30_252) / (max_iv30_252 - min_iv30_252) | 252 trading days | Options vendor | Percentile rank of IV in last year. |
| `skew_25d` | options_vol | 25Δ put IV - 25Δ call IV | daily | Options vendor | Proxy for downside risk pricing. |
| `kurtosis_smile` | options_vol | Excess kurtosis of IV across strikes at fixed tenor | daily | Options vendor | Highlights wings pricing. |
| `oi_delta_bucket_{strike}` | options_flow | Change in open interest for strike bucket (e.g., ±5% moneyness) | daily | Options vendor | Use net change between sessions. |
| `sweep_notional_1d` | options_flow | Sum of aggressive sweep trade notional per symbol | intraday/daily | Options flow feed (SpotGamma, UnusualWhales) | Identify large directional bets. |
| `put_call_ratio_equity` | options_flow | total put volume / total call volume | daily | CBOE / OCC | Track crowd positioning. |
| `put_call_ratio_index` | options_flow | Index-level put/call ratio | daily | CBOE | Divergence from single-name ratio useful. |

## 2. Market Microstructure & Order Flow

| Feature | Category | Formula / Definition | Window | Source | Notes |
|---------|----------|----------------------|--------|--------|-------|
| `bid_ask_imbalance_lvl{1-3}` | order_flow | (bid_vol - ask_vol) / (bid_vol + ask_vol) per level | per snapshot (1s) | Level 2/3 feed (NASDAQ TotalView, ARCA) | Evaluate pressure at the top of book. |
| `spread_pct` | order_flow | (ask_price - bid_price) / mid_price | per snapshot | L2 feed | For slippage modeling. |
| `order_book_slope` | order_flow | Regression slope of depth vs. price (top N levels) | per snapshot | L3 feed | Detect liquidity pockets/holes. |
| `cum_delta_1m` | order_flow | Σ (trade_size * side) over 1 minute | rolling 1 minute | TAQ or proprietary trade feed | Captures net aggression. |
| `auction_imbalance_open` | order_flow | Net imbalance at open auction | daily | Exchange auction feed | Signals potential gaps. |
| `queue_position_metrics` | order_flow | Avg queued shares per order / cancellation rate | intraday | L3 feed | Understand execution risk. |

## 3. Event & Macro Calendar

| Feature | Category | Definition | Window | Source | Notes |
|---------|----------|------------|--------|--------|-------|
| `macro_event_flag_{event}` | macro | Binary indicator if event within H hours | rolling (H=0.5,1,4) | Econ calendars (Econoday, TradingEconomics) | Build per-event type (CPI, FOMC, etc.). |
| `macro_surprise_score` | macro | (actual - consensus) / stdev(last 12 releases) | event time | Econ feed | Drives regime shifts. |
| `earnings_date_flag` | corporate | 1 if earnings within {3d,1d} lookahead | daily | Earnings calendar (FactSet, Finnhub) | Use for gap risk. |
| `guidance_event_flag` | corporate | 1 if company issued guidance this week | weekly | Corporate filings | Combines with fundamentals 2.0. |
| `regulatory_alert_score` | macro | Weighted count of regulatory/geopolitical headlines per day | daily | News feed (Dow Jones, Reuters) | Use weights by severity. |

## 4. Alternative Data

| Feature | Category | Definition | Frequency | Source | Notes |
|---------|----------|------------|-----------|--------|-------|
| `card_spend_yoy` | alt_data | YoY change in aggregated card spending | weekly | Third-party card data (Affirm, Yodlee) | Proxy for revenue. |
| `job_postings_velocity` | alt_data | Δ job postings count / total | weekly | LinkedIn, Indeed API | Hiring + expansion. |
| `app_usage_index` | alt_data | Daily active users (normalized) | daily | App analytics (app store, telemetry) | Tech names early signal. |
| `satellite_store_traffic` | alt_data | Foot traffic index | weekly | Satellite providers | Retail/energy. |
| `supply_chain_delay_index` | alt_data | Avg shipping delay vs. baseline | weekly | Freight/supply data | Warning for inventories. |

## 5. Cross-Asset & Factor Context

| Feature | Category | Definition | Window | Source | Notes |
|---------|----------|------------|--------|--------|-------|
| `rates_2y`, `rates_10y` | cross_asset | Treasury yields | daily/intraday | FRED, Bloomberg | Build curve slope `rates_2y - rates_10y`. |
| `credit_spread_ig` | cross_asset | IG spread vs. Treasuries | daily | ICE/Bloomberg | Credit stress indicator. |
| `dxy_index` | cross_asset | USD index level | daily/intraday | FX feed | Affects exporters/commodities. |
| `commodity_{oil,copper}` | cross_asset | Spot/futures price | daily/intraday | CME/ICE | Macro sensitivity. |
| `crypto_beta` | cross_asset | Rolling beta of equity vs. BTC/ETH | 30d window | Price feeds | For crypto-exposed names. |
| `factor_value`, `factor_momentum`, etc. | factor | Factor returns (long-short) | daily | Barra, AQR, Quandl | Map single names to factor exposures. |

## 6. Technical & Regime Features

| Feature | Category | Definition | Window | Notes |
|---------|----------|------------|--------|-------|
| `atr_band_upper/lower` | technical | Price ± k * ATR | ATR window 14, k configurable | Defines range. |
| `realized_vol_term` | technical | σ returns on {5d, 21d, 63d} horizons | multiple | Compare term structure. |
| `donchian_breakout` | technical | Price > max high(n) or < min low(n) | n=20,55 | Trend entries. |
| `adx_14` | technical | ADX measure of trend strength | 14 | Combine with DI. |
| `regime_tag` | regime | {low-vol trend, high-vol chop, event-risk} from HMM or rules | daily | Use states to filter signals. |

## 7. Fundamentals 2.0

| Feature | Category | Definition | Window | Source | Notes |
|---------|----------|------------|--------|--------|-------|
| `earnings_surprise_pct` | fundamentals | (actual - consensus) / abs(consensus) | quarterly | Earnings APIs | Map to post-earnings drift. |
| `guidance_delta` | fundamentals | New guidance - prior guidance / prior | per event | Company filings | Track signals direction. |
| `estimate_revision_momentum` | fundamentals | EMA of analyst EPS revisions | daily/weekly | FactSet/I/B/E/S | Predicts medium-term drift. |
| `accrual_ratio` | fundamentals | (NI - CFO) / total assets | quarterly | Financial statements | Quality flag. |
| `buyback_intensity` | fundamentals | Shares repurchased / shares outstanding | trailing 4 quarters | Filings | Supports capital return analysis. |

## 8. Insider, Institutional & Ownership Flows

| Feature | Category | Definition | Frequency | Notes |
|---------|----------|------------|-----------|-------|
| `insider_cluster_buy_score` | ownership | Weighted sum of insider purchases grouped by insider type | weekly | EDGAR Form 4 | Strong signal when clustered. |
| `short_interest_pct_float` | ownership | Short interest / float | bi-weekly | FINRA | Squeeze risk. |
| `etf_flow_score` | ownership | Net ETF flow impacting ticker (sector/theme) | daily | ETF providers | Use to infer passive flow pressure. |
| `institutional_hold_change` | ownership | Δ institutional holdings (13F) | quarterly | SEC 13F filings | Slow signal but high conviction. |

## 9. Risk, Execution & Costs

| Feature | Category | Definition | Frequency | Notes |
|---------|----------|------------|-----------|-------|
| `slippage_estimate` | execution | Expected slippage = f(spread_pct, vol, participation_pct) | per trade | Model response surfaces. |
| `max_drawdown_trailing` | risk | Max loss from peak over past N trades | rolling window | Enforce circuit breakers. |
| `kelly_scaled_size` | risk | Kelly fraction * cap factor | per signal | Keeps sizing disciplined. |
| `trade_fill_prob` | execution | Probability of fill given venue/route | per order | Derived from historical fill data. |

## 10. Data Quality & ML Ops

| Feature | Category | Definition | Frequency | Notes |
|---------|----------|------------|-----------|-------|
| `timestamp_lag_check` | datavalidation | Max |timestamp_model_input - timestamp_source| | per batch | Ensure no look-ahead. |
| `feature_drift_score` | datavalidation | KS-statistic vs. training distribution | daily | Trigger retraining. |
| `label_leakage_flag` | datavalidation | Boolean if future info detected in feature | per batch | Run automated tests. |

## 11. Explainability & Monitoring

| Feature | Category | Definition | Frequency | Notes |
|---------|----------|------------|-----------|-------|
| `shap_importance_{feature}` | explainability | Mean absolute SHAP for feature conditioned on regime | batch/daily | Compare across regimes. |
| `error_bucket_{state}` | explainability | Aggregated error metrics grouped by liquidity/vol/regime | daily | Identify failing contexts. |
| `post_trade_pnl_analysis` | explainability | Actual vs. expected PnL with attribution to features | per trade/day | Feeds risk review. |

---

Usage Notes:
* Treat this dictionary as a living document; version it with each model release.
* Add column for computation owner (data engineering vs. quant research) and build checks around missing data or stale feeds.
* Align naming with feature store / warehouse conventions to avoid ambiguity.

