# Alt-Data Onboarding Checklist

This checklist ensures systematic evaluation and integration of alternative data sources.

## 1. Legal & Compliance ⚖️

- [ ] **License agreement reviewed**
  - Terms of use documented
  - Usage restrictions identified
  - Renewal terms understood
  - Termination clauses reviewed

- [ ] **Data usage rights confirmed**
  - Production use permitted
  - Redistribution rights (if needed)
  - Derivative works allowed
  - Geographic restrictions noted

- [ ] **MNPI screening completed**
  - Data source verified as public
  - First-print timestamps available
  - No insider information risk
  - Compliance team sign-off

- [ ] **Privacy compliance (GDPR/CCPA) verified**
  - Personal data identified
  - Consent mechanisms in place
  - Data retention policy compliant
  - Right to deletion supported

---

## 2. Technical Evaluation 🔧

- [ ] **Data format documented**
  - Schema defined (JSON/CSV/Parquet/etc.)
  - Field types specified
  - Nested structures mapped
  - Sample data reviewed

- [ ] **Delivery mechanism tested (API/FTP/S3)**
  - Connection established
  - Authentication working
  - Rate limits documented
  - Failover procedures defined

- [ ] **PIT compliance verified**
  - First-print timestamps available
  - Revisions tracked
  - Look-ahead bias prevented
  - Temporal integrity validated

- [ ] **Backfill available (min 2 years)**
  - Historical depth confirmed
  - Data consistency verified
  - Missing periods identified
  - Load time estimated

---

## 3. Quality Assessment 📊

- [ ] **Data quality expectations defined**
  - Completeness threshold (e.g., ≥95%)
  - Accuracy requirements
  - Timeliness SLA
  - Update frequency

- [ ] **Historical consistency checked**
  - Point-in-time restatements analyzed
  - Survivor ship bias checked
  - Corporate actions handled
  - Time zone consistency

- [ ] **Missing data patterns analyzed**
  - Gaps documented
  - Seasonality identified
  - Coverage by symbol/sector
  - Imputation strategy defined

- [ ] **Revision frequency documented**
  - How often data is revised
  - Revision window (e.g., T+2)
  - Impact on strategies
  - Version control approach

---

## 4. Alpha Assessment 🎯

- [ ] **Backtest with alt-data feature**
  - PIT-compliant backtest run
  - Baseline performance measured
  - Alt-data feature added
  - Performance comparison done

- [ ] **Calculate IR uplift**
  - Information Ratio before: `___`
  - Information Ratio after: `___`
  - IR uplift: `___`
  - Statistical significance: p-value `___`

- [ ] **Estimate cost per IR improvement**
  - Annual data cost: $`___`
  - IR uplift: `___`
  - Cost per 0.1 IR: $`___`
  - Threshold: $10,000 per 0.1 IR

- [ ] **Compare to threshold (gate decision)**
  - [ ] IR uplift per dollar ≥ threshold
  - [ ] Sharpe improvement ≥ 0.1
  - [ ] P&L improvement ≥ data cost
  - **Decision: ONBOARD / REJECT**

---

## 5. Production Integration 🚀

- [ ] **Connector implemented**
  - Python connector class created
  - Error handling implemented
  - Retry logic added
  - Logging configured

- [ ] **Monitoring alerts configured**
  - Data freshness alert
  - Quality threshold alert
  - Volume anomaly detection
  - Delivery failure notification

- [ ] **Fallback strategy defined**
  - Behavior when data unavailable
  - Staleness tolerance (e.g., 24h)
  - Alternative data sources
  - Graceful degradation

- [ ] **Documentation completed**
  - Data dictionary created
  - Integration guide written
  - Troubleshooting FAQ
  - Contact information

---

## 6. ROI Validation 💰

- [ ] **Initial ROI estimate**
  - Expected Sharpe uplift: `___`
  - Expected annual benefit: $`___`
  - Expected ROI: `___%`
  - Break-even capital: $`___`

- [ ] **3-month review**
  - Actual Sharpe uplift: `___`
  - Actual benefit: $`___`
  - Actual ROI: `___%`
  - Continue: YES / NO

- [ ] **6-month review**
  - Cumulative benefit: $`___`
  - Cumulative ROI: `___%`
  - Data quality issues: `___`
  - Continue: YES / NO

- [ ] **Annual review**
  - Full year benefit: $`___`
  - Full year ROI: `___%`
  - Renew: YES / NO
  - Renegotiate pricing: YES / NO

---

## Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Quant Lead | ____________ | ____________ | ________ |
| Risk Manager | ____________ | ____________ | ________ |
| Compliance | ____________ | ____________ | ________ |
| IT/Data Eng | ____________ | ____________ | ________ |
| CFO (if > $50K) | ____________ | ____________ | ________ |

---

## Notes & Action Items

```
[Add any notes, concerns, or action items here]








```

---

**Onboarding Date**: ________________
**Expected Production Date**: ________________
**Data Source**: ________________
**Vendor**: ________________
**Annual Cost**: $________________
