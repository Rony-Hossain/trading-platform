# Feature Contract Template

## Overview
This template defines the mandatory Point-in-Time (PIT) contract for all features in the trading platform. Every feature must have a completed contract before being used in training or production.

## Required Fields

### 1. Feature Identification
- **feature_name**: Unique identifier for the feature
- **feature_type**: Category (technical, fundamental, sentiment, macro, options, derived)
- **data_source**: Primary data provider (yahoo, finnhub, custom, computed)
- **version**: Semantic version (e.g., 1.0.0)

### 2. Point-in-Time Constraints
- **as_of_ts**: Timestamp when feature value is known/available for use
- **effective_ts**: Timestamp when underlying event occurred 
- **arrival_latency**: Expected delay from effective_ts to as_of_ts (in minutes)
- **point_in_time_rule**: Specific PIT access rules and constraints

### 3. Data Quality & SLA
- **vendor_SLA**: Service level agreement from data provider
  - **availability**: Expected uptime (e.g., 99.9%)
  - **latency**: Maximum delivery delay (e.g., 15 minutes)
  - **quality**: Expected accuracy/completeness (e.g., 99.5%)
- **revision_policy**: How and when data gets revised
  - **revision_frequency**: How often revisions occur (never, daily, weekly)
  - **revision_window**: How far back revisions can occur (e.g., 30 days)
  - **notification_method**: How revisions are communicated

### 4. Business Logic
- **computation_logic**: How feature is calculated (for derived features)
- **dependencies**: Other features this depends on
- **lookback_period**: Historical data required (e.g., 30 days)
- **update_frequency**: How often feature updates (real-time, daily, weekly)

### 5. Validation & Monitoring
- **valid_range**: Expected value range (min, max)
- **null_handling**: How null/missing values are treated
- **outlier_detection**: Method for identifying anomalous values
- **monitoring_alerts**: Conditions that trigger data quality alerts

### 6. Compliance & Audit
- **pii_classification**: Whether feature contains personal information (none, masked, anonymized)
- **regulatory_notes**: Any regulatory considerations
- **audit_trail**: How changes to this feature are tracked
- **retention_policy**: How long historical data is kept

## Example Contract

```yaml
# Example: VIX Fear Index Contract
feature_name: "vix_close"
feature_type: "macro"
data_source: "yahoo"
version: "1.0.0"

# Point-in-Time Constraints
as_of_ts: "T+1 09:30 ET"  # Available next trading day at market open
effective_ts: "T 16:00 ET"  # VIX close value from previous day
arrival_latency: 17.5  # hours (overnight + 30 min processing)
point_in_time_rule: "No access to T+0 VIX close until T+1 market open"

# Data Quality & SLA  
vendor_SLA:
  availability: 99.9%
  latency: "15 minutes after market close"
  quality: 99.99%
revision_policy:
  revision_frequency: "never"
  revision_window: "0 days"
  notification_method: "none"

# Business Logic
computation_logic: "Direct feed from CBOE via Yahoo Finance"
dependencies: []
lookback_period: "365 days"
update_frequency: "daily"

# Validation & Monitoring
valid_range: [5.0, 80.0]
null_handling: "forward_fill_max_3_days"
outlier_detection: "3_sigma_rolling_30d"
monitoring_alerts: 
  - "null_for_2_days"
  - "outside_historical_range"
  - "latency_exceed_sla"

# Compliance & Audit
pii_classification: "none"
regulatory_notes: "Public market data, no restrictions"
audit_trail: "git_commits + mlflow_tags"
retention_policy: "7_years"
```

## Contract Lifecycle

1. **Creation**: New features must have contract before first use
2. **Review**: Contracts reviewed by data team + compliance
3. **Approval**: Signed off by lead data scientist + risk manager  
4. **Monitoring**: Automated validation against contract terms
5. **Updates**: Changes require new version + re-approval

## Enforcement

- CI/CD pipeline blocks deployment if features lack contracts
- Automated validation runs nightly to check contract compliance
- Violations generate alerts and block model deployment
- All contract changes tracked in audit log

## Templates by Feature Type

- [Technical Indicators](./templates/technical_contract.yml)
- [Fundamental Data](./templates/fundamental_contract.yml) 
- [Sentiment Features](./templates/sentiment_contract.yml)
- [Options Data](./templates/options_contract.yml)
- [Macro Factors](./templates/macro_contract.yml)
- [Derived Features](./templates/derived_contract.yml)