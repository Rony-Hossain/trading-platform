# High-End Upgrade Roadmap

This roadmap combines the immediate high-impact upgrades with the longer-term future feature program so you have a single source of truth. It builds directly on the current platform strengths (see `documentation/current-capabilities.md`) and targets the gaps listed there.

---

## Phase 1 – Data Enrichment & Factor Fusion

Focus: close the most significant factor gaps quickly and feed them into the existing forecast and strategy pipelines while formalizing baseline risk controls.

### 1. Market Data Service (`services/market-data-service`)

| Capability | Action | Impact | Key Files |
|------------|--------|--------|-----------|
| Cross-Asset Context | Fetch daily VIX, 2Y/10Y Treasury yields, EUR/USD, and crude oil prices via Finnhub/Yahoo. Store in TimescaleDB hypertables with retention policies. | Adds macro regime awareness to every forecast. | `providers/finnhub_provider.py`, `services/market_data.py`, DB migrations |
| Options Analytics (Primer) | Pull EOD options chains (yfinance or paid feed) for core symbols; compute put/call ratios and ATM implied volatility; persist summary series. | Introduces forward-looking crowd positioning into analytics. | New `providers/options_provider.py`, schema update |
| Implied Move Modeling | For each scheduled event, calculate implied move from the nearest-dated ATM straddle (price ÷ spot); store expected move bands for use in analysis/strategy services. | Quantifies the market’s priced-in expectation so surprise can be measured. | Options provider module, new factor table |
| IV Skew / Implied Move Ratio | Track IV skew (25Δ put-call) and the ratio of event IV vs. rolling IV benchmarks (e.g., IV_event ÷ IV30). | Highlights abnormal risk pricing specific to catalysts. | Options provider, analytics helpers |

### 2. Fundamentals Service (`services/fundamentals-service`)

| Capability | Action | Impact | Key Files |
|------------|--------|--------|-----------|
| Surprise Delta | Ingest analyst consensus (EPS/Revenue); extend earnings monitor to compute actual minus consensus metrics and store them as factor features. | Upgrades fundamentals signal from raw YoY to expectation-aware deltas. | `app/services/earnings_monitor.py`, migrations |
| Ownership & Flow (Kick-off) | Start Form 4/13F ingestion using the SEC parser scaffolding; label cluster buying and institutional net change. | Lays groundwork for “smart money” factors in later phases. | `app/services/sec_parser.py`, new storage helpers |
| Analyst Revision Tracking | Track and aggregate analyst rating/price-target revisions in the 30 days pre-event; expose momentum metrics via API. | Captures shifting consensus ahead of catalysts. | New ingestion job, fundamentals schema update |

### 3. Analysis Service (`services/analysis`)

| Capability | Action | Impact | Key Files |
|------------|--------|--------|-----------|
| Multi-Factor Feature Set | Extend feature engineering to pull sentiment summaries, surprise deltas, macro/IV series, and factor interaction terms (e.g., Momentum×Rates slope, Value×Credit spread). | Converts the model from single-factor (technical) to multi-factor with interaction awareness. | `app/services/forecast_service.py`, helper utilities |
| Options Volume Dynamics | Model options OI/volume delta ramp using time-series techniques to capture positioning build-up into events. | Measures conviction of derivatives traders pre-event. | Options analytics module |

### 4. Strategy Service (`services/strategy-service`)

| Capability | Action | Impact | Key Files |
|------------|--------|--------|-----------|
| Risk & Execution Baselines | Implement configurable position sizing (fixed %, ATR-based), daily loss limits, and a simple slippage model in `risk_manager` and `backtest_engine`. | Ensures backtests and live trials respect capital constraints and trading friction. | `app/engines/risk_manager.py`, `app/engines/backtest_engine.py`, config |

### 5. Event Data Service (new)

| Capability | Action | Impact | Key Files |
|------------|--------|--------|-----------|
| Event Calendar Feed | Stand up `services/event-data-service` to ingest scheduled catalysts (earnings, product launches, analyst days, regulatory decisions) via an external calendar API. Normalize and store forward-looking events in TimescaleDB. | Provides proactive knowledge of upcoming catalysts. | new service scaffold, DB migrations |
| Real-Time Headline Capture | Integrate a low-latency news/headline API to record actual outcomes tied to scheduled events. | Enables immediate surprise scoring once results drop. | event service collectors, webhook handlers |

**Deliverables**: API endpoints `/events/upcoming`, `/events/recent`, `/events/{symbol}`; health checks; documentation of data providers and rate limits.

---

## Phase 2 – Model Sophistication & MLOps Foundation

Focus: evolve predictive models, add deep learning, and introduce experiment tracking and monitoring.

### 1. Analysis Service Enhancements

| Capability | Action | Impact |
|------------|--------|--------|
| Neural Sequence Pilot | Implement an LSTM/GRU pipeline using the enriched feature set; run in parallel with the RandomForestRegressor. | Captures nonlinear temporal dependencies and prepares for ensemble forecasts. |
| Regime Tagging | Compute ATR bands, realized volatility term structures, and a simple HMM state classifier (trend vs. chop vs. event). | Gives models data-driven market state awareness. |

### 2. Sentiment Service Deepening (`services/sentiment-service`)

| Capability | Action | Impact |
|------------|--------|--------|
| Transformer Fine-Tuning | Fine-tune FinBERT/DistilBERT (or similar) on an annotated retail finance dataset; replace or augment the VADER/TextBlob stack. | Provides higher accuracy, domain-specific sentiment scores. |
| Topic Modeling | Use BERTopic (transformer embeddings + UMAP + HDBSCAN + c-TF-IDF) to surface narratives. Expose `/topics/{symbol}` endpoints and store topic probabilities. | Supplies topical factors for forecasting and narrative monitoring. |
| Sentiment Momentum | Calculate short-term sentiment acceleration (e.g., EMA of net sentiment over 1/5/10-day windows) to detect pre-event build-up or information leakage. | Adds anticipatory signal ahead of catalysts. |
| Novelty & Source Weighting | Compute novelty scores (dedupe syndicated headlines) and apply source credibility weights to sentiment aggregates. | Cleans noise and prevents double-counting of information shocks. |

### 3. Infrastructure & Tooling

| Capability | Action | Impact |
|------------|--------|--------|
| Experiment Tracking | Integrate MLflow (or similar) into training scripts; log hyperparameters, data cuts, metrics, and model artifacts. | Enables reproducible research, comparisons, and rollbacks. |
| Monitoring Expansion | Add Prometheus metrics for collector coverage, model latency, and data freshness; extend Grafana dashboards to cover new pipelines. | Faster diagnosis of production issues and data outages. |

### 4. Event-Driven Strategy Layer

| Capability | Action | Impact |
|------------|--------|--------|
| Catalyst Trigger Logic | Require event occurrence + surprise delta above threshold + sentiment spike confirmation before generating signals. | Filters to high-conviction information shocks. |
| Gap Trading Modules | Implement continuation vs. fade logic for gap scenarios, supporting pre/post-market price handling. | Formalizes post-event trade decision process. |
| Event Profile Modeling | Build CAR (Cumulative Abnormal Return) studies for past events to inform exit horizon and holding period per catalyst type. | Grounds strategy exits in historical reaction profiles. |
| Surprise Threshold Calibration | Calibrate surprise thresholds by sector and event type (e.g., product launch vs. regulatory update). | Ensures triggers fire only on truly material shocks. |
| Regime-Conditioned Signals | Require favorable market regime tags (e.g., low-vol trend) before executing event trades. | Avoids firing signals in hostile market environments. |
| Event-Aware Backtests | Extend `backtest_engine` to replay pre/after-hours prices with latency assumptions and default tight stop-loss rules. | Produces realistic performance metrics for event-driven strategies. |
| Execution Latency & Order Types | Model execution delay (e.g., 200 ms) and support aggressive limit/IOC orders instead of pure market orders in strategy/backtest. | Aligns backtests with real-world fill dynamics. |

---

## Phase 3 – Institutional Features & Explainability

Focus: add institutional-grade data, interpretability, and a polished UI/UX to close the loop with traders.

### 1. Fundamentals Service

| Capability | Action | Impact |
|------------|--------|--------|
| Institutional/Insider Flow | Complete parsing of Form 4 (cluster buys/sells) and Form 13F (holding changes); store aggregated flow signals in TimescaleDB. | Provides medium-term conviction and squeeze indicators. |

### 2. Analysis & Strategy Services

| Capability | Action | Impact |
|------------|--------|--------|
| SHAP Explainability | Compute SHAP values for RF/LSTM outputs; surface via API and dashboards. | Offers transparent model decisions and audit readiness. |
| Post-Trade Analytics | Extend the backtest and (eventual) live paper trading outputs with error buckets (slippage, missed trades, risk stops). | Highlights failure modes and areas for improvement. |

### 3. Frontend (`trading-frontend`)

| Capability | Action | Impact |
|------------|--------|--------|
| Sentiment & Factor Dashboards | Build pages that visualize sentiment trends, topics, ownership flows, and SHAP feature importances with drill-down to raw data. | Puts advanced analytics in front of end users. |
| Paper Trading Integration | Connect the Strategy Service to a paper trading API (Alpaca/IB) with UI controls for strategy selection and risk settings. | Closes the loop for real-time validation without real capital. |

---

## Phase 4 �?" Platform Hardening & Alpha Expansion (18–36 weeks)

Focus: institutionalize data/model operations, unlock low-latency alpha, and harden the production stack.

### 1. Data Infrastructure & Quality

| Capability | Action | Impact |
|------------|--------|--------|
| Feature Store / Point-in-Time Governance | Deploy a feature store (e.g., Feast or temporal Postgres tables) that records availability timestamps and versions every factor. | Eliminates look-ahead leakage and simplifies feature reuse. |
| Automated Data Quality Gates | Integrate Great Expectations/Monte Carlo checks for macro/options/event feeds; alert on anomalies, missing data, and duplicates. | Keeps inputs clean and highlights pipeline failures early. |

### 2. Model Ops & Streaming

| Capability | Action | Impact |
|------------|--------|--------|
| Continuous Retraining & Drift Monitoring | Schedule monthly retrains for RF/LSTM models with drift metrics (PSI/KS) and champion–challenger promotion. | Maintains performance and prevents silent model decay. |
| Real-Time Feature Streaming | Introduce Kafka or Redis Streams for streaming sentiment spikes, options flow, and microstructure metrics; serve sub-second inference via TorchServe/ONNX Runtime. | Enables rapid response in event-driven strategies. |

### 3. Execution Enhancements

| Capability | Action | Impact |
|------------|--------|--------|
| Smart Order Routing | Add venue-aware routing logic, dark-pool access, and trade journaling. | Improves fills and creates execution data for calibration. |
| Trade Journal & Attribution | Store actual fills, slippage, P&L attribution, and link to SHAP/explainability outputs. | Builds an institutional audit trail and feedback loop. |

### 4. Security, Compliance & Deployment

| Capability | Action | Impact |
|------------|--------|--------|
| Secrets & IaC Hardening | Move secrets to Vault/Secrets Manager, enforce rotation, and manage infrastructure via Terraform. | Hardens production deployments. |
| Container Security | Integrate Trivy (or similar) into CI, enforce signed non-root images, and add policy checks. | Reduces supply-chain risk. |

### 5. Testing, Collaboration & Alerts

| Capability | Action | Impact |
|------------|--------|--------|
| CI/CD & Synthetic Testing | Expand unit/integration/synthetic-data tests; build GitHub Actions pipelines for lint/test/build/deploy. | Ensures reliability with faster releases. |
| Research Workspace & Alerting | Provide JupyterHub/Deepnote access tied to the feature store; set up Slack/Teams alert workflows with deep links to dashboards. | Accelerates quant research and operational awareness. |

### 6. Stress Testing & Alternative Data

| Capability | Action | Impact |
|------------|--------|--------|
| Scenario Simulation | Build engines to replay historic crises or synthetic shocks for strategy stress testing. | Validates robustness under extreme regimes. |
| Alternative Data Onboarding | Establish connectors and ROI tracking for new datasets (satellite, card spend, web scraping, ESG). | Continually expands the alpha surface responsibly. |

---

## Implementation Notes
- **TimescaleDB**: Each new factor will require schema changes (hypertables + retention). Plan migrations carefully.
- **Secrets Management**: New feeds (options, macro) may need additional API keys—update `.env` docs and consider Vault/Parameter Store in production.
- **Testing**: Expand beyond `test-all-services.py` with unit/integration tests for new collectors, pipelines, and models.
- **Documentation**: Keep setup/testing guides and the feature dictionary in sync as capabilities grow.
- **Point-in-Time Data**: Adopt temporal tables or a feature store to enforce availability timestamps for fundamentals/ownership features; normalize event/news timestamps to UTC with millisecond precision to eliminate look-ahead.
- **Backtesting Realism**: Enhance slippage models with depth-aware fill probability curves and use VWAP/TWAP for pre/after-hours fills when events gap the open.

## Success Metrics
- Forecast accuracy (RMSE, hit rate) improves after multi-factor fusion and deep learning adoption.
- Backtest realism: risk rules and slippage reduce the gap between simulated and paper/live trading P&L.
- Operational coverage: collectors, model training, and pipelines are monitored with dashboards; ML experiments are reproducible.
- Explainability: ability to answer “why” for each forecast/trade (SHAP reports, post-trade analytics).

---

## Appendix: Extended Future Feature Program

The broader future program (originally captured in `future-feature-upgrade-plan.txt`) pushes the platform toward a comprehensive, multi-modal intelligence stack. It is retained here verbatim so you have both the near-term high-impact plan and the deeper R&D roadmap.

### Phase 1 – Data & Labeling Foundation
**Goals**
- Assemble trustworthy datasets for retail-market sentiment and topic modeling.
- Establish governance for data versions, labeling quality, and reproducibility.

**Key Tasks**
- Collect raw posts from Reddit, Twitter/X, StockTwits, Discord, Telegram (backfill where permitted).
- Stand up an annotation workflow (Label Studio or similar) with multi-class labels: {Bullish, Bearish, Neutral, Sarcastic-Bullish, Sarcastic-Bearish, Hype/Spam, FUD}.
- Build quality controls (consensus labeling, reviewer spot checks, active learning for ambiguous samples).
- Version datasets with DVC/Delta Lake; document schema and feature definitions.

**Deliverables**
- Versioned raw + labeled datasets with metadata (symbols, timestamps, language).
- Labeling guideline documentation and QA metrics (agreement ratios, reviewer throughput).

**Cross-Cutting**
- Confirm API usage compliance; add backoff logic for each collector.
- Track annotation throughput, inter-rater agreement, and dataset freshness in dashboards.

### Phase 2 – Domain-Specific NLP Stack
**Goals**
- Train and deploy sentiment/sarcasm models tuned to retail finance language.
- Provide robust inference APIs for downstream services.

**Key Tasks**
- Fine-tune FinBERT/RoBERTa on financial + retail datasets using small LR (1e-5 backbone, 1e-4 classifier).
- Build sarcasm detection ensemble (transformer embeddings + lexicon incongruity + emoji/punctuation CNN + Bi-LSTM/attention).
- Apply adversarial training (FGSM) to improve robustness to slang/noise.
- Deploy inference service (FastAPI + TorchServe/ONNX) with GPU support, caching, autoscaling.

**Deliverables**
- Sentiment API v2 returning label probabilities, sarcasm flag, attention/metadata.
- Model cards documenting datasets, metrics (accuracy, F1, ROC), limitations.

**Cross-Cutting**
- Automate training with experiment tracking (MLflow/W&B).
- Monitor latency, embedding drift, and health via dashboards.

### Phase 3 – Topic & Narrative Intelligence
**Goals**
- Identify emerging narratives and track their evolution by symbol.

**Key Tasks**
- Generate sentence embeddings from Phase 2 transformer; run BERTopic (UMAP + HDBSCAN + c-TF-IDF).
- Store topic assignments, probabilities, novelty scores, and keywords in TimescaleDB.
- Expose `/topics/{symbol}` and `/topics/{symbol}/history` endpoints.

**Deliverables**
- Topic pipeline orchestrated with Prefect/Airflow; backfill scripts and alerts for distribution shifts.
- Dashboards for top narratives, coverage gaps, topic churn, data freshness.

**Cross-Cutting**
- Evaluate cluster coherence and silhouette scores; perform human review.
- Issue alerts (Slack/webhooks) when new high-confidence topics surface.

### Phase 4 – Multi-Modal Forecasting & Fusion
**Goals**
- Build predictive models that merge price/technical indicators, macro, sentiment vectors, sarcasm rates, and topic dominance.

**Key Tasks**
- Engineer feature store with aligned hourly/daily samples: EMA-weighted sentiment, sentiment volatility, sarcasm ratio, topic probabilities, market indicators (VIX, Put/Call, AAII), technical signals (RSI, MACD, log returns, volume features, wavelet denoising).
- Architect multi-input network (Temporal Fusion Transformer for price/macro, 1D CNN for sentiment/topic sequences) and fuse through an MLP head. Experiment with LSTM/GRU baselines.
- Implement walk-forward validation/backtesting harness; benchmark against RandomForest baseline.
- Integrate explainability (SHAP, attention visualizations); log feature importances.

**Deliverables**
- Forecast service v2 returning point predictions, confidence intervals, and driver explanations.
- Evaluation report with RMSE, hit rate, direction precision/recall vs. baseline.

**Cross-Cutting**
- Apply regularization (dropout, L2), early stopping, stress tests; plan failover to legacy model if needed.

### Phase 5 – UI & Experience Enhancements
**Goals**
- Surface enriched insights through dashboards and trading tools.

**Key Tasks**
- Build sentiment command center (trend lines, heat maps, source breakdown, live narrative feed).
- Overlay sentiment/topic annotations on price charts; add narrative watchlists and alerts.
- Blend fundamentals + sentiment overlays into strategy/portfolio views.
- Enrich options tooling with sentiment-driven recommendations and explanations.

**Deliverables**
- New Next.js components/pages with responsive design, dark mode, SWR/React Query caching.
- User documentation and onboarding guides for analysts/traders.

**Cross-Cutting**
- Conduct UX validation; address accessibility (ARIA, formatting).

### Phase 6 – Operationalization & Governance
**Goals**
- Ensure long-term reliability, compliance, and maintenance.

**Key Tasks**
- Establish monthly retraining cadence with automated tests and rollback; integrate with CI/CD.
- Extend observability (Prometheus/Grafana) for model drift, annotation backlog, API usage.
- Harden security/compliance: key rotation, secrets management, audit logs.
- Review cost/performance; tune autoscaling and caching to meet SLOs.

**Deliverables**
- Runbooks for incidents, retraining, dataset updates.
- Compliance checklist covering data licensing, privacy, ToS adherence.

**Cross-Cutting**
- Conduct post-launch retrospectives; plan future enhancements (multilingual sentiment, reinforcement learning, synthetic data augmentation).

### Getting Started Checklist
1. Rotate and secure API credentials (Twitter/X, Reddit, Finnhub, NewsAPI, etc.).
2. Deploy labeling tooling and document annotation schema.
3. Define success metrics (precision/recall, forecast RMSE, P&L impact) and baselines.
4. Staff data engineering, NLP research, MLOps, UI, QA, and compliance roles.
5. Create project board (Jira/Linear) with milestones and owners.
