# Feature Contract Template

This template defines the required contract specification for all features used in the trading platform to ensure Point-in-Time (PIT) compliance and data integrity.

## Contract Fields

### Core Temporal Fields

#### `as_of_ts` (Required)
- **Type**: ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS.sssZ)
- **Description**: The point-in-time timestamp representing when this feature value was "known" or "observable" in the real world
- **PIT Rule**: Must be <= prediction timestamp to avoid future data leakage
- **Example**: `"2024-01-15T09:30:00.000Z"`
- **Validation**: Must be valid timestamp, cannot be future relative to prediction time

#### `effective_ts` (Required)
- **Type**: ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS.sssZ)
- **Description**: The timestamp when this feature becomes available for use in the trading system
- **PIT Rule**: Must be >= as_of_ts + arrival_latency
- **Example**: `"2024-01-15T09:35:00.000Z"`
- **Validation**: effective_ts >= as_of_ts + arrival_latency

#### `arrival_latency` (Required)
- **Type**: Duration in seconds (integer) or ISO 8601 duration
- **Description**: Expected delay from when data is "as of" to when it arrives in our system
- **Purpose**: Models real-world data delivery delays for realistic backtesting
- **Example**: `300` (5 minutes) or `"PT5M"`
- **Validation**: Must be non-negative integer or valid duration

### Data Quality & SLA Fields

#### `point_in_time_rule` (Required)
- **Type**: String (enum: "strict", "relaxed", "custom")
- **Description**: PIT access constraint level for this feature
- **Values**:
  - `"strict"`: Zero tolerance for future data access
  - `"relaxed"`: Allow minor timing discrepancies (<1min)
  - `"custom"`: Custom rule defined in additional_rules field
- **Example**: `"strict"`
- **Validation**: Must be one of allowed enum values

#### `vendor_SLA` (Required)
- **Type**: Object with service level agreement details
- **Description**: Data vendor's guaranteed service levels
- **Structure**:
  ```json
  {
    "availability_pct": 99.5,
    "max_latency_seconds": 300,
    "update_frequency": "1min",
    "backfill_policy": "24h_window",
    "quality_guarantee": "99.9%"
  }
  ```
- **Validation**: All SLA fields must be present and valid

#### `revision_policy` (Required)
- **Type**: Object describing how/when data gets revised
- **Description**: Policy for handling data corrections and revisions
- **Structure**:
  ```json
  {
    "allows_revisions": true,
    "revision_window_hours": 24,
    "revision_notification": "immediate",
    "backfill_on_revision": true,
    "version_tracking": true
  }
  ```
- **Validation**: Policy object must be complete and consistent

### Metadata Fields

#### `feature_id` (Required)
- **Type**: String (snake_case identifier)
- **Description**: Unique identifier for this feature
- **Pattern**: `^[a-z][a-z0-9_]*[a-z0-9]$`
- **Example**: `"stock_price_close_1min"`

#### `feature_name` (Required)
- **Type**: String (human-readable name)
- **Description**: Descriptive name for the feature
- **Example**: `"1-Minute Stock Close Price"`

#### `data_source` (Required)
- **Type**: String (data provider identifier)
- **Description**: Primary data source/vendor
- **Example**: `"bloomberg_api"`, `"yahoo_finance"`, `"internal_calculation"`

#### `feature_type` (Required)
- **Type**: String (enum)
- **Description**: Classification of feature type
- **Values**: `"price"`, `"volume"`, `"sentiment"`, `"fundamental"`, `"technical"`, `"macro"`, `"derived"`

#### `granularity` (Required)
- **Type**: String (time interval)
- **Description**: Time granularity of the feature
- **Example**: `"1min"`, `"5min"`, `"1hour"`, `"1day"`

#### `currency` (Optional)
- **Type**: String (ISO 4217 currency code)
- **Description**: Currency denomination for monetary features
- **Example**: `"USD"`, `"EUR"`

#### `unit` (Optional)
- **Type**: String
- **Description**: Unit of measurement
- **Example**: `"dollars"`, `"shares"`, `"percentage"`, `"ratio"`

### Compliance Fields

#### `pit_audit_status` (System Generated)
- **Type**: String (enum: "compliant", "warning", "violation")
- **Description**: Current PIT compliance status
- **Auto-generated**: Set by PIT auditor system

#### `last_audit_ts` (System Generated)
- **Type**: ISO 8601 timestamp
- **Description**: Timestamp of last PIT audit
- **Auto-generated**: Set by PIT auditor system

#### `contract_version` (Required)
- **Type**: String (semantic version)
- **Description**: Version of this contract specification
- **Example**: `"1.0.0"`

#### `created_by` (Required)
- **Type**: String (user identifier)
- **Description**: User who created this contract
- **Example**: `"john.doe@company.com"`

#### `approved_by` (Optional)
- **Type**: String (user identifier)
- **Description**: User who approved this contract for production
- **Example**: `"jane.smith@company.com"`

#### `approval_date` (Optional)
- **Type**: ISO 8601 timestamp
- **Description**: When this contract was approved
- **Example**: `"2024-01-15T14:30:00.000Z"`

### Extended Configuration

#### `additional_rules` (Optional)
- **Type**: Object
- **Description**: Custom PIT rules when point_in_time_rule is "custom"
- **Example**:
  ```json
  {
    "custom_lag_seconds": 120,
    "weekend_handling": "forward_fill",
    "holiday_handling": "skip"
  }
  ```

#### `dependencies` (Optional)
- **Type**: Array of feature IDs
- **Description**: Other features this feature depends on
- **Example**: `["stock_price_open_1min", "stock_volume_1min"]`

#### `tags` (Optional)
- **Type**: Array of strings
- **Description**: Classification tags for feature discovery
- **Example**: `["equity", "price", "real_time", "core"]`

## Complete Contract Example

```json
{
  "feature_id": "aapl_close_price_1min",
  "feature_name": "Apple Inc. 1-Minute Close Price",
  "data_source": "bloomberg_api",
  "feature_type": "price",
  "granularity": "1min",
  "currency": "USD",
  "unit": "dollars",
  
  "as_of_ts": "2024-01-15T09:30:00.000Z",
  "effective_ts": "2024-01-15T09:35:00.000Z",
  "arrival_latency": 300,
  
  "point_in_time_rule": "strict",
  "vendor_SLA": {
    "availability_pct": 99.9,
    "max_latency_seconds": 300,
    "update_frequency": "1min",
    "backfill_policy": "24h_window",
    "quality_guarantee": "99.95%"
  },
  "revision_policy": {
    "allows_revisions": false,
    "revision_window_hours": 0,
    "revision_notification": "none",
    "backfill_on_revision": false,
    "version_tracking": true
  },
  
  "contract_version": "1.0.0",
  "created_by": "quant.team@company.com",
  "approved_by": "risk.manager@company.com",
  "approval_date": "2024-01-15T14:30:00.000Z",
  
  "tags": ["equity", "price", "real_time", "core", "aapl"],
  "dependencies": []
}
```

## Validation Rules

### Temporal Consistency
1. `as_of_ts <= effective_ts`
2. `effective_ts >= as_of_ts + arrival_latency`
3. `as_of_ts <= prediction_timestamp` (enforced at runtime)
4. All timestamps must be valid ISO 8601 format

### PIT Compliance
1. Strict rule: `effective_ts` must be available before feature use
2. No future data access: `as_of_ts` cannot exceed prediction time
3. Latency modeling: `arrival_latency` must be realistic for data source

### Business Logic
1. `vendor_SLA.availability_pct` must be between 0.0 and 100.0
2. `vendor_SLA.max_latency_seconds` should align with `arrival_latency`
3. `revision_policy` must be consistent with `vendor_SLA`

### Required Fields
All fields marked as "Required" must be present and non-null in every contract.

## Usage in Code

Feature contracts must be registered before feature use:

```python
from core.feature_contracts import FeatureContractValidator

validator = FeatureContractValidator()

# Register contract
contract = {
    "feature_id": "aapl_close_price_1min",
    # ... full contract as above
}
validator.register_contract(contract)

# Validate feature usage
feature_data = {
    "as_of_timestamp": "2024-01-15T09:30:00.000Z",
    "effective_timestamp": "2024-01-15T09:35:00.000Z",
    "value": 150.25
}

violations = validator.check_pit_compliance("aapl_close_price_1min", feature_data)
if violations:
    raise PitComplianceError(f"PIT violations: {violations}")
```

## CI/CD Integration

All feature contracts are automatically validated in CI/CD:

```bash
# Pre-commit hook
python ci/pit_contract_check.py --validate-all

# Environment variable for enforcement
export PIT_CONTRACTS_ENFORCE=true
```

Deployment will fail if:
- New features lack contracts
- Contracts fail validation
- PIT compliance violations detected
- Required approvals missing