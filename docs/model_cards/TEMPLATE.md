# Model Card: [Model Name]

**Version**: [e.g., v1.2.0]
**Date**: [YYYY-MM-DD]
**Owner**: [Team/Individual Name]
**Status**: [Development / Staging / Production / Deprecated]
**MLflow Run ID**: [run_id if applicable]
**Governance Tag**: `governance_ready=true`

---

## 1. Model Overview

### 1.1 Objective
**What does this model do?**

[Clear, concise statement of the model's purpose and business objective]

**Example**: This model predicts intraday price momentum for liquid US equities to generate short-term alpha signals. The model outputs a probability score (0-1) indicating the likelihood of positive returns over the next 30 minutes.

### 1.2 Model Type
- [ ] Supervised Learning (Classification)
- [ ] Supervised Learning (Regression)
- [ ] Unsupervised Learning
- [ ] Reinforcement Learning
- [ ] Rule-Based System
- [ ] Hybrid

**Algorithm**: [e.g., XGBoost, Random Forest, Neural Network, etc.]

### 1.3 Use Case
- [ ] Alpha Generation
- [ ] Risk Assessment
- [ ] Portfolio Construction
- [ ] Execution Optimization
- [ ] Market Regime Detection
- [ ] Anomaly Detection
- [ ] Other: [specify]

---

## 2. Data

### 2.1 Training Data

**Data Sources**:
- [List all data sources, e.g., "NYSE TAQ trades", "Reuters news headlines", "SEC 10-K filings"]

**Time Period**: [Start Date] to [End Date]

**Sample Size**:
- Training: [N samples]
- Validation: [N samples]
- Test: [N samples]

**Data Splits**:
- Training: [%]
- Validation: [%]
- Test: [%]

**Split Method**: [Time-based / Random / Stratified / Walk-forward]

### 2.2 Feature Engineering

**Input Features** (Total: [N]):

| Feature Name | Type | Description | Source | PIT Compliant? |
|--------------|------|-------------|--------|----------------|
| [feature_1] | Numeric | [description] | [source] | ✅ / ❌ |
| [feature_2] | Categorical | [description] | [source] | ✅ / ❌ |
| ... | ... | ... | ... | ... |

**Feature Groups**:
- **Price/Volume**: [list features]
- **Technical Indicators**: [list features]
- **Fundamental**: [list features]
- **Alternative Data**: [list features]
- **Derived/Engineered**: [list features]

### 2.3 Point-In-Time (PIT) Compliance

**PIT Rules Applied**:
- [ ] All features use only data available at prediction time
- [ ] No look-ahead bias in feature construction
- [ ] Corporate actions adjusted correctly (splits, dividends)
- [ ] Restatement handling for fundamental data
- [ ] Survivorship bias eliminated
- [ ] Time-zone alignment verified

**PIT Validation Process**: [Describe validation methodology]

**PIT Violations Identified**: [None / List any violations and remediation]

### 2.4 Data Quality

**Missing Data Handling**:
- Strategy: [Forward-fill / Interpolation / Imputation / Drop]
- Missing data threshold: [%]

**Outlier Treatment**:
- Detection method: [IQR / Z-score / Domain knowledge]
- Handling: [Winsorization / Capping / Removal]

**Data Freshness Requirements**:
- Maximum staleness: [e.g., "5 minutes for price data"]
- Fallback behavior: [e.g., "Use last valid value, flag as stale"]

---

## 3. Model Architecture & Training

### 3.1 Architecture

**Model Structure**:
```
[Describe model architecture, hyperparameters, etc.]

Example for XGBoost:
- Objective: binary:logistic
- Max depth: 6
- Learning rate: 0.1
- N estimators: 100
- Subsample: 0.8
```

### 3.2 Training Process

**Training Infrastructure**:
- Hardware: [e.g., "16 vCPU, 64GB RAM"]
- Training time: [e.g., "2 hours"]
- Framework: [e.g., "scikit-learn 1.3.0, XGBoost 1.7.0"]

**Optimization**:
- Loss function: [e.g., "Log loss", "MSE"]
- Optimization algorithm: [e.g., "Gradient boosting"]
- Early stopping: [Yes/No, criteria]

**Hyperparameter Tuning**:
- Method: [Grid search / Random search / Bayesian optimization]
- Cross-validation: [K-fold, time-series CV]
- Parameters tuned: [list]

---

## 4. Performance Metrics

### 4.1 Offline Performance

**Classification Metrics** (if applicable):
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | [%] | [%] | [%] |
| Precision | [%] | [%] | [%] |
| Recall | [%] | [%] | [%] |
| F1 Score | [%] | [%] | [%] |
| AUC-ROC | [%] | [%] | [%] |

**Regression Metrics** (if applicable):
| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| RMSE | [value] | [value] | [value] |
| MAE | [value] | [value] | [value] |
| R² | [value] | [value] | [value] |

**Business Metrics**:
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Sharpe Ratio | [value] | [target] | ✅ / ⚠️ / ❌ |
| Information Ratio | [value] | [target] | ✅ / ⚠️ / ❌ |
| Hit Rate | [%] | [target] | ✅ / ⚠️ / ❌ |
| Max Drawdown | [%] | [target] | ✅ / ⚠️ / ❌ |

### 4.2 Feature Importance

**Top 10 Features**:
1. [feature_name]: [importance score]
2. [feature_name]: [importance score]
3. ...

**Feature Importance Method**: [SHAP / Permutation / Gain-based]

---

## 5. Risk & Validation Gates

### 5.1 SPA (Symmetric Prediction Accuracy)

**SPA Score**: [value] (Target: ≥ 0.45)

**Test**: Does the model predict equally well in both directions?
- Long predictions accuracy: [%]
- Short predictions accuracy: [%]
- Symmetry score: [value]

**Status**: ✅ Pass / ❌ Fail

### 5.2 DSR (Downside Sensitivity Ratio)

**DSR Score**: [value] (Target: ≤ 1.5)

**Test**: Is the model more sensitive to downside than upside?
- Downside volatility: [value]
- Upside volatility: [value]
- Ratio: [value]

**Status**: ✅ Pass / ❌ Fail

### 5.3 PBO (Probability of Backtest Overfitting)

**PBO Score**: [%] (Target: < 50%)

**Test**: Combinatorially symmetric backtest over N partitions
- Number of partitions: [N]
- IS/OOS performance ratio: [value]

**Status**: ✅ Pass / ❌ Fail

### 5.4 Walk-Forward Validation

**Configuration**:
- Window size: [N months/years]
- Step size: [N months]
- Number of windows: [N]

**Results**:
- Average OOS Sharpe: [value]
- Consistency: [% of positive windows]

**Status**: ✅ Pass / ❌ Fail

---

## 6. Model Monitoring & Drift

### 6.1 Drift Detection Metrics

**Statistical Drift**:
- [ ] KS Test on feature distributions (p-value threshold: 0.05)
- [ ] PSI (Population Stability Index) on predictions (threshold: 0.1)
- [ ] Jensen-Shannon Divergence on label distribution

**Performance Drift**:
- [ ] Rolling Sharpe ratio (30-day window)
- [ ] Prediction accuracy degradation
- [ ] Calibration drift (Brier score)

**Alert Thresholds**:
| Metric | Warning | Critical |
|--------|---------|----------|
| PSI | > 0.1 | > 0.25 |
| KS p-value | < 0.05 | < 0.01 |
| Sharpe decline | > 20% | > 40% |
| Accuracy drop | > 5% | > 10% |

### 6.2 Monitoring Dashboard

**Grafana Dashboard**: [Link or name]

**Monitored Metrics**:
- Real-time prediction distribution
- Feature drift scores
- Model latency (p95, p99)
- Error rates
- Business metrics (daily Sharpe, hit rate)

### 6.3 Retraining Policy

**Trigger Conditions** (OR logic):
- [ ] Scheduled: Every [N weeks/months]
- [ ] Drift detected: PSI > 0.25 for 3 consecutive days
- [ ] Performance degradation: Sharpe < threshold for 5 days
- [ ] Data regime change: Detected by regime filter
- [ ] Manual trigger: Model owner discretion

**Retraining Process**: [Reference to retraining runbook]

---

## 7. Assumptions & Limitations

### 7.1 Key Assumptions

1. **[Assumption 1]**: [Description and justification]
   - **Risk if violated**: [Impact]

2. **[Assumption 2]**: [Description and justification]
   - **Risk if violated**: [Impact]

**Example**:
1. **Market microstructure remains stable**: Model assumes bid-ask spreads and market depth characteristics stay within historical ranges.
   - **Risk if violated**: Execution costs may exceed model predictions, degrading alpha.

### 7.2 Known Limitations

1. **[Limitation 1]**: [Description]
   - **Mitigation**: [Strategy]

2. **[Limitation 2]**: [Description]
   - **Mitigation**: [Strategy]

**Example**:
1. **Low-liquidity securities**: Model performance degrades for stocks with ADV < $10M
   - **Mitigation**: Apply minimum liquidity filter in pre-trade screening

### 7.3 Out-of-Scope Scenarios

**Model should NOT be used for**:
- [Scenario 1]
- [Scenario 2]
- [Scenario 3]

**Example**:
- Illiquid micro-cap stocks (market cap < $100M)
- Earnings announcement windows (±2 days)
- Halted securities

---

## 8. Failure Modes & Contingencies

### 8.1 Known Failure Modes

| Failure Mode | Probability | Impact | Detection | Mitigation |
|--------------|-------------|--------|-----------|------------|
| [Mode 1] | Low/Med/High | Low/Med/High | [Method] | [Strategy] |
| [Mode 2] | Low/Med/High | Low/Med/High | [Method] | [Strategy] |

**Example**:
| Failure Mode | Probability | Impact | Detection | Mitigation |
|--------------|-------------|--------|-----------|------------|
| Feature service outage | Medium | High | Latency > 5s | Use cached features, flag stale |
| Model returns NaN | Low | High | Output validation | Fallback to previous valid prediction |

### 8.2 Circuit Breakers

**Automatic Deactivation Triggers**:
1. Prediction latency > [N ms] for > [M] consecutive predictions
2. Error rate > [%] over [time window]
3. VaR breach attributed to model
4. Drift metrics exceed critical thresholds

**Manual Override**: [Process and authorization required]

### 8.3 Rollback Plan

**Rollback Triggers**:
- Performance degradation beyond acceptable threshold
- Critical bug discovered
- Regulatory or compliance issue
- Business decision

**Rollback Procedure**:
1. Disable model via feature flag: `models.{model_name}.enabled = false`
2. Revert to previous version: [Specify version or fallback strategy]
3. Notify stakeholders: [Slack channel, email list]
4. Post-mortem within 48 hours

**Previous Stable Version**: [Version number, MLflow run ID]

**Estimated Rollback Time**: [N minutes]

---

## 9. Deployment & Operations

### 9.1 Deployment History

| Version | Date | Changes | Deployed By | Status |
|---------|------|---------|-------------|--------|
| v1.0.0 | YYYY-MM-DD | Initial deployment | [Name] | Deprecated |
| v1.1.0 | YYYY-MM-DD | Added features X, Y | [Name] | Deprecated |
| v1.2.0 | YYYY-MM-DD | Current version | [Name] | Active |

### 9.2 Infrastructure

**Serving Infrastructure**:
- Platform: [e.g., "Kubernetes", "AWS SageMaker"]
- Container image: [image:tag]
- Resources: [CPU, memory, GPU]
- Scaling: [Min/max replicas, autoscaling policy]

**Dependencies**:
- Python: [version]
- Libraries: [list with versions]
- External services: [list]

### 9.3 API / Interface

**Endpoint**: `[URL or function signature]`

**Input Schema**:
```json
{
  "symbol": "string",
  "timestamp": "ISO-8601",
  "features": {
    "feature_1": float,
    "feature_2": int,
    ...
  }
}
```

**Output Schema**:
```json
{
  "prediction": float,
  "confidence": float,
  "model_version": "string",
  "timestamp": "ISO-8601"
}
```

**SLA**:
- Latency: p95 < [N ms]
- Availability: [%]
- Throughput: [requests/sec]

---

## 10. Compliance & Governance

### 10.1 Regulatory Considerations

**Applicable Regulations**:
- [ ] SEC Rule 15c3-5 (Market Access)
- [ ] MiFID II Algorithmic Trading
- [ ] FINRA Rule 3110 (Supervision)
- [ ] Other: [specify]

**Compliance Controls**:
- Pre-trade risk checks: [Enabled/Disabled]
- Kill switch: [Available/Not Available]
- Audit trail: [Enabled/Disabled]

### 10.2 Model Approval

**Approval Workflow**:
- [ ] Model owner sign-off
- [ ] Quant team review
- [ ] Risk management review
- [ ] Compliance review
- [ ] Executive approval (if required)

**Approval Date**: [YYYY-MM-DD]
**Approved By**: [Names and roles]

### 10.3 Review Cadence

**Scheduled Reviews**:
- Performance review: [Frequency, e.g., "Weekly"]
- Risk review: [Frequency, e.g., "Monthly"]
- Full model review: [Frequency, e.g., "Quarterly"]

**Next Review Date**: [YYYY-MM-DD]

---

## 11. References & Artifacts

### 11.1 Documentation

- **Research Paper**: [Link or reference]
- **Backtest Report**: [Link to artifacts/backtests/]
- **Deployment Memo**: [Link to docs/deploy_memos/]
- **Runbook**: [Link to docs/runbooks/]

### 11.2 MLflow Artifacts

**MLflow Experiment**: [Experiment name]
**MLflow Run ID**: [run_id]

**Attached Artifacts**:
- [ ] Trained model (model.pkl)
- [ ] Feature list (features.json)
- [ ] Training data schema (schema.json)
- [ ] Performance plots (performance.png)
- [ ] This model card (model_card.md)
- [ ] Deployment memo (deploy_memo.md)

**MLflow Tags**:
```
governance_ready: true
status: production
owner: [name]
version: [version]
```

### 11.3 Code Repository

**Repository**: [GitHub/GitLab URL]
**Branch**: [e.g., main]
**Commit Hash**: [hash]

**Key Files**:
- Training script: [path]
- Inference script: [path]
- Feature engineering: [path]
- Tests: [path]

---

## 12. Contact & Support

**Model Owner**: [Name, email]
**Team**: [Team name]
**Slack Channel**: [#channel-name]
**On-Call**: [PagerDuty schedule or contact]

**Escalation Path**:
1. Model owner
2. Lead quant
3. Head of research
4. CTO

---

## 13. Changelog

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| YYYY-MM-DD | 1.0.0 | [Name] | Initial model card |
| YYYY-MM-DD | 1.1.0 | [Name] | Updated performance metrics |
| YYYY-MM-DD | 1.2.0 | [Name] | Added new features, re-validated |

---

**Document Status**: ✅ Complete / ⚠️ Draft / ❌ Incomplete
**Last Updated**: [YYYY-MM-DD]
**Governance Review**: ✅ Approved for production
