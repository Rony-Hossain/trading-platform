# Model Card: Momentum Alpha Predictor

**Version**: v1.0.0
**Date**: 2025-10-01
**Owner**: Quantitative Research Team
**Status**: Production
**MLflow Run ID**: abc123def456
**Governance Tag**: `governance_ready=true`

---

## 1. Model Overview

### 1.1 Objective
**What does this model do?**

This model predicts intraday price momentum for liquid US equities (market cap > $1B, ADV > $10M) to generate short-term alpha signals. The model outputs a probability score (0-1) indicating the likelihood of positive returns over the next 30 minutes.

### 1.2 Model Type
- [x] Supervised Learning (Classification)
- [ ] Supervised Learning (Regression)
- [ ] Unsupervised Learning
- [ ] Reinforcement Learning
- [ ] Rule-Based System
- [ ] Hybrid

**Algorithm**: XGBoost Gradient Boosting Classifier

### 1.3 Use Case
- [x] Alpha Generation
- [x] Risk Assessment
- [ ] Portfolio Construction
- [ ] Execution Optimization
- [ ] Market Regime Detection
- [ ] Anomaly Detection
- [ ] Other: N/A

---

## 2. Data

### 2.1 Training Data

**Data Sources**:
- NYSE TAQ Level 1 quotes and trades
- NASDAQ TotalView order book data
- Reuters news headlines (sentiment scores)
- CBOE VIX and sector ETF volumes

**Time Period**: 2023-01-01 to 2024-06-30 (18 months)

**Sample Size**:
- Training: 2,450,000 samples
- Validation: 350,000 samples
- Test: 420,000 samples

**Data Splits**:
- Training: 75%
- Validation: 10%
- Test: 15%

**Split Method**: Time-based (chronological to avoid look-ahead bias)

### 2.2 Feature Engineering

**Input Features** (Total: 47):

| Feature Name | Type | Description | Source | PIT Compliant? |
|--------------|------|-------------|--------|----------------|
| price_momentum_5min | Numeric | 5-min price change | TAQ | ✅ |
| price_momentum_15min | Numeric | 15-min price change | TAQ | ✅ |
| volume_ratio_5min | Numeric | Vol vs 20-day avg | TAQ | ✅ |
| bid_ask_spread_bps | Numeric | Spread in basis points | TAQ | ✅ |
| order_imbalance | Numeric | Bid-ask imbalance | NASDAQ | ✅ |
| rsi_14 | Numeric | Relative Strength Index | Derived | ✅ |
| macd_signal | Numeric | MACD - Signal line | Derived | ✅ |
| vwap_distance_pct | Numeric | Price vs VWAP | Derived | ✅ |
| sector_etf_momentum | Numeric | Sector ETF 30min return | External | ✅ |
| vix_change_pct | Numeric | VIX % change (5min) | CBOE | ✅ |
| news_sentiment_score | Numeric | Sentiment (-1 to 1) | Reuters | ✅ |
| ... (37 more) | ... | ... | ... | ... |

**Feature Groups**:
- **Price/Volume**: price_momentum (multiple timeframes), volume_ratio, volatility
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Microstructure**: bid-ask spread, order imbalance, trade intensity
- **Market Context**: VIX, sector momentum, market breadth
- **Sentiment**: news sentiment score, social media buzz (when available)

### 2.3 Point-In-Time (PIT) Compliance

**PIT Rules Applied**:
- [x] All features use only data available at prediction time
- [x] No look-ahead bias in feature construction
- [x] Corporate actions adjusted correctly (splits, dividends)
- [x] Restatement handling for fundamental data
- [x] Survivorship bias eliminated
- [x] Time-zone alignment verified

**PIT Validation Process**:
- Automated tests verify all features have timestamp <= prediction_time
- Manual code review by 2 senior quants
- Historical replay on random sample (100 days) with zero violations

**PIT Violations Identified**: None

### 2.4 Data Quality

**Missing Data Handling**:
- Strategy: Forward-fill for market data (max 5 seconds), drop for features
- Missing data threshold: < 0.5% per feature (all features meet this)

**Outlier Treatment**:
- Detection method: Winsorization at 1st/99th percentile
- Handling: Cap extreme values to prevent training instability

**Data Freshness Requirements**:
- Maximum staleness: 10 seconds for price/volume data, 60 seconds for sentiment
- Fallback behavior: Use last valid value, set staleness_flag=True

---

## 3. Model Architecture & Training

### 3.1 Architecture

**Model Structure**:
```
XGBoost Classifier
- Objective: binary:logistic
- Max depth: 6
- Learning rate: 0.05
- N estimators: 200
- Subsample: 0.8
- Colsample_bytree: 0.8
- Min_child_weight: 5
- Gamma: 0.1
- Reg_alpha: 0.01 (L1)
- Reg_lambda: 1.0 (L2)
- Scale_pos_weight: 1.0 (balanced classes)
```

### 3.2 Training Process

**Training Infrastructure**:
- Hardware: AWS c5.4xlarge (16 vCPU, 32GB RAM)
- Training time: 45 minutes
- Framework: XGBoost 1.7.4, scikit-learn 1.3.0, Python 3.10

**Optimization**:
- Loss function: Binary log loss
- Optimization algorithm: Gradient boosting with histogram-based splits
- Early stopping: Yes, patience=20 rounds on validation AUC

**Hyperparameter Tuning**:
- Method: Bayesian optimization (Optuna, 100 trials)
- Cross-validation: 5-fold time-series CV
- Parameters tuned: max_depth, learning_rate, n_estimators, regularization

---

## 4. Performance Metrics

### 4.1 Offline Performance

**Classification Metrics**:
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | 61.2% | 58.7% | 58.3% |
| Precision | 59.8% | 57.2% | 56.9% |
| Recall | 63.5% | 60.1% | 59.7% |
| F1 Score | 61.6% | 58.6% | 58.3% |
| AUC-ROC | 0.672 | 0.651 | 0.647 |

**Business Metrics** (from backtest):
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Sharpe Ratio | 2.15 | > 1.5 | ✅ |
| Information Ratio | 1.82 | > 1.0 | ✅ |
| Hit Rate | 56.9% | > 52% | ✅ |
| Max Drawdown | -8.3% | < 15% | ✅ |

### 4.2 Feature Importance

**Top 10 Features** (SHAP values):
1. price_momentum_5min: 0.142
2. order_imbalance: 0.098
3. volume_ratio_5min: 0.087
4. vwap_distance_pct: 0.074
5. bid_ask_spread_bps: 0.061
6. rsi_14: 0.053
7. sector_etf_momentum: 0.049
8. macd_signal: 0.045
9. vix_change_pct: 0.038
10. news_sentiment_score: 0.031

**Feature Importance Method**: SHAP (SHapley Additive exPlanations)

---

## 5. Risk & Validation Gates

### 5.1 SPA (Symmetric Prediction Accuracy)

**SPA Score**: 0.52 (Target: ≥ 0.45)

**Test**: Does the model predict equally well in both directions?
- Long predictions accuracy: 57.8%
- Short predictions accuracy: 56.0%
- Symmetry score: 0.52 (well-balanced)

**Status**: ✅ Pass

### 5.2 DSR (Downside Sensitivity Ratio)

**DSR Score**: 1.12 (Target: ≤ 1.5)

**Test**: Is the model more sensitive to downside than upside?
- Downside volatility: 1.38%
- Upside volatility: 1.23%
- Ratio: 1.12

**Status**: ✅ Pass

### 5.3 PBO (Probability of Backtest Overfitting)

**PBO Score**: 32% (Target: < 50%)

**Test**: Combinatorially symmetric backtest over 16 partitions
- Number of partitions: 16
- IS/OOS performance ratio: 1.08

**Status**: ✅ Pass

### 5.4 Walk-Forward Validation

**Configuration**:
- Window size: 3 months training, 1 month testing
- Step size: 1 month
- Number of windows: 12

**Results**:
- Average OOS Sharpe: 1.89
- Consistency: 83% of positive windows (10/12)

**Status**: ✅ Pass

---

## 6. Model Monitoring & Drift

### 6.1 Drift Detection Metrics

**Statistical Drift**:
- [x] KS Test on feature distributions (p-value threshold: 0.05)
- [x] PSI (Population Stability Index) on predictions (threshold: 0.1)
- [x] Jensen-Shannon Divergence on label distribution

**Performance Drift**:
- [x] Rolling Sharpe ratio (30-day window)
- [x] Prediction accuracy degradation
- [x] Calibration drift (Brier score)

**Alert Thresholds**:
| Metric | Warning | Critical |
|--------|---------|----------|
| PSI | > 0.1 | > 0.25 |
| KS p-value | < 0.05 | < 0.01 |
| Sharpe decline | > 20% | > 40% |
| Accuracy drop | > 5% | > 10% |

### 6.2 Monitoring Dashboard

**Grafana Dashboard**: http://grafana.internal/d/momentum-alpha-v1

**Monitored Metrics**:
- Real-time prediction distribution (30min rolling)
- Feature drift scores (daily PSI per feature)
- Model latency (p50/p95/p99)
- Error rates and exceptions
- Business metrics (daily Sharpe, hit rate, P&L attribution)

### 6.3 Retraining Policy

**Trigger Conditions** (OR logic):
- [x] Scheduled: Every 3 months
- [x] Drift detected: PSI > 0.25 for 3 consecutive days
- [x] Performance degradation: Sharpe < 1.0 for 5 trading days
- [x] Data regime change: Detected by macro regime filter
- [x] Manual trigger: Model owner discretion

**Retraining Process**: See `docs/runbooks/model_retraining.md`

---

## 7. Assumptions & Limitations

### 7.1 Key Assumptions

1. **Market microstructure stability**: Model assumes bid-ask spreads and order book depth remain within historical norms (2-20 bps for large caps).
   - **Risk if violated**: Execution costs may exceed predictions, degrading net alpha.

2. **Intraday mean reversion**: Assumes 30-minute horizon is sufficient for momentum signals to play out.
   - **Risk if violated**: Holding period may need extension, changing risk profile.

3. **News sentiment accuracy**: Reuters sentiment scores assumed 70%+ accurate.
   - **Risk if violated**: Sentiment feature may introduce noise; can be disabled via feature flag.

### 7.2 Known Limitations

1. **Earnings announcement windows**: Model performance degrades ±2 days around earnings.
   - **Mitigation**: Exclude symbols with earnings in 2-day window (implemented in pre-trade filter).

2. **Low-liquidity securities**: Performance drops for ADV < $5M due to wider spreads.
   - **Mitigation**: Hard minimum liquidity filter (ADV ≥ $10M) enforced pre-trade.

3. **Overnight gaps**: Model trained on intraday data, doesn't predict overnight moves.
   - **Mitigation**: Flatten all positions 30 minutes before close.

### 7.3 Out-of-Scope Scenarios

**Model should NOT be used for**:
- Illiquid micro-cap stocks (market cap < $1B)
- Earnings announcement windows (±2 days)
- After-hours or pre-market trading
- Securities with trading halts
- Options or derivatives (equity only)

---

## 8. Failure Modes & Contingencies

### 8.1 Known Failure Modes

| Failure Mode | Probability | Impact | Detection | Mitigation |
|--------------|-------------|--------|-----------|------------|
| Feature service outage | Low | High | Latency > 5s or errors | Use cached features (5min TTL), flag as stale |
| Model returns NaN/Inf | Low | High | Output validation | Use previous valid prediction, alert |
| VIX spike (>40) | Medium | Medium | Real-time VIX monitor | Reduce allocation to 50%, widen risk limits |
| News sentiment API down | Low | Low | API health check | Disable sentiment feature, continue with others |

### 8.2 Circuit Breakers

**Automatic Deactivation Triggers**:
1. Prediction latency > 500ms for > 10 consecutive predictions
2. Error rate > 5% over 5-minute window
3. Daily P&L loss > $50k attributed to this model
4. PSI > 0.35 (severe drift)

**Manual Override**: Authorized by Head of Trading or CTO via feature flag

### 8.3 Rollback Plan

**Rollback Triggers**:
- Sharpe ratio < 0.5 for 3 consecutive trading days
- Critical bug causing systematic errors
- VaR breach directly attributed to model
- Regulatory or compliance issue

**Rollback Procedure**:
1. Disable model via feature flag: `models.momentum_alpha.enabled = false`
2. Flatten all open positions from this model (< 5 min)
3. Notify stakeholders: Slack #trading-alerts, email trading desk
4. Post-mortem scheduled within 48 hours

**Previous Stable Version**: N/A (initial deployment)

**Estimated Rollback Time**: 5 minutes

---

## 9. Deployment & Operations

### 9.1 Deployment History

| Version | Date | Changes | Deployed By | Status |
|---------|------|---------|-------------|--------|
| v1.0.0 | 2025-10-01 | Initial production deployment | J. Smith | Active |

### 9.2 Infrastructure

**Serving Infrastructure**:
- Platform: Kubernetes (AWS EKS, us-east-1)
- Container image: quant-models/momentum-alpha:v1.0.0
- Resources: 4 vCPU, 8GB RAM per pod
- Scaling: Min 2, max 10 replicas (HPA on CPU > 70%)

**Dependencies**:
- Python: 3.10.12
- XGBoost: 1.7.4
- NumPy: 1.24.3
- Pandas: 2.0.1
- External services: Feature Store API, Market Data Feed

### 9.3 API / Interface

**Endpoint**: `POST /api/v1/predictions/momentum-alpha`

**Input Schema**:
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-10-01T14:30:00Z",
  "features": {
    "price_momentum_5min": 0.0023,
    "volume_ratio_5min": 1.45,
    ...
  }
}
```

**Output Schema**:
```json
{
  "prediction": 0.687,
  "confidence": 0.82,
  "model_version": "v1.0.0",
  "timestamp": "2025-10-01T14:30:00.123Z",
  "latency_ms": 87
}
```

**SLA**:
- Latency: p95 < 100ms, p99 < 200ms
- Availability: 99.9% during market hours
- Throughput: 1000 requests/sec

---

## 10. Compliance & Governance

### 10.1 Regulatory Considerations

**Applicable Regulations**:
- [x] SEC Rule 15c3-5 (Market Access) - Pre-trade risk checks enabled
- [x] FINRA Rule 3110 (Supervision) - Daily surveillance logs
- [ ] MiFID II Algorithmic Trading - N/A (US only)
- [ ] Other: N/A

**Compliance Controls**:
- Pre-trade risk checks: Enabled (position limits, liquidity)
- Kill switch: Available via feature flag + manual override
- Audit trail: Enabled (all predictions and decisions logged to S3)

### 10.2 Model Approval

**Approval Workflow**:
- [x] Model owner sign-off: J. Smith (2025-09-28)
- [x] Quant team review: Senior Quants Review Committee (2025-09-29)
- [x] Risk management review: Chief Risk Officer (2025-09-30)
- [x] Compliance review: Compliance Officer (2025-09-30)
- [x] Executive approval: CTO (2025-10-01)

**Approval Date**: 2025-10-01
**Approved By**: J. Smith (Model Owner), A. Chen (CRO), M. Patel (CTO)

### 10.3 Review Cadence

**Scheduled Reviews**:
- Performance review: Daily (automated report)
- Risk review: Weekly (Tuesday 10am meeting)
- Full model review: Quarterly (Jan, Apr, Jul, Oct)

**Next Review Date**: 2025-12-15

---

## 11. References & Artifacts

### 11.1 Documentation

- **Research Paper**: `research/momentum_alpha_research_memo.pdf`
- **Backtest Report**: `artifacts/backtests/momentum_alpha_v1_backtest.html`
- **Deployment Memo**: `docs/deploy_memos/momentum_alpha_v1.0.0.md`
- **Runbook**: `docs/runbooks/momentum_alpha_operations.md`

### 11.2 MLflow Artifacts

**MLflow Experiment**: momentum-alpha-production
**MLflow Run ID**: abc123def456

**Attached Artifacts**:
- [x] Trained model (model.xgb)
- [x] Feature list (features.json)
- [x] Training data schema (schema.json)
- [x] Performance plots (performance.png, shap_summary.png)
- [x] This model card (model_card.md)
- [x] Deployment memo (deploy_memo.md)

**MLflow Tags**:
```
governance_ready: true
status: production
owner: quant-research
version: v1.0.0
deployment_date: 2025-10-01
```

### 11.3 Code Repository

**Repository**: https://github.com/trading-platform/models/momentum-alpha
**Branch**: main
**Commit Hash**: a7f3c2d9e8b1f4a6c3e5d8f9a2b7c4e1

**Key Files**:
- Training script: `src/train.py`
- Inference script: `src/predict.py`
- Feature engineering: `src/features/momentum_features.py`
- Tests: `tests/test_momentum_alpha.py`

---

## 12. Contact & Support

**Model Owner**: Jane Smith, jane.smith@tradingplatform.com
**Team**: Quantitative Research
**Slack Channel**: #quant-momentum-alpha
**On-Call**: PagerDuty schedule "Quant Models"

**Escalation Path**:
1. Jane Smith (Model Owner)
2. Alex Chen (Lead Quant)
3. Morgan Patel (Head of Research)
4. CTO

---

## 13. Changelog

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-10-01 | 1.0.0 | J. Smith | Initial production deployment |

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-10-01
**Governance Review**: ✅ Approved for production
