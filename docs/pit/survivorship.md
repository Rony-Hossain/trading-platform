# Survivorship Bias Prevention in Trading Platform

## Overview

Survivorship bias is a critical issue in quantitative finance that can lead to overly optimistic backtest results. This bias occurs when historical analysis only includes securities that have "survived" to the present day, excluding those that were delisted, merged, or went bankrupt during the study period.

Our trading platform implements comprehensive survivorship bias prevention through:
- Point-in-time universe construction
- Inclusion of delisted securities
- Proper data lineage tracking
- Validation and testing frameworks

## Why Survivorship Bias Matters

### Impact on Backtests
- **Inflated Returns**: Excluding failed companies artificially boosts historical performance
- **Reduced Risk Measures**: Eliminating bankruptcies and delisting events underestimates downside risk
- **False Strategy Validation**: Strategies may appear profitable only because they exclude negative outcomes

### Real-World Examples
- A strategy tested on only surviving S&P 500 companies since 1990 would miss hundreds of delisted firms
- Technology sector analysis excluding dot-com failures (1999-2001) would show unrealistic resilience
- Small-cap strategies without delisted names could overestimate alpha generation

## Implementation Architecture

### Universe Management System

```
┌─────────────────────────────────────────────────────────────────┐
│                    Universe Data Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  External Sources    │    Database Storage    │   Point-in-Time │
│  ┌─────────────┐     │   ┌─────────────────┐  │   ┌───────────┐ │
│  │ S&P 500     │────┐│   │ universe_       │  │   │ Backtest  │ │
│  │ Constituents│    ││   │ constituents    │  │   │ Universe  │ │
│  └─────────────┘    ├┼──▶│                 │──┼──▶│ Builder   │ │
│  ┌─────────────┐    ││   │ - symbol        │  │   │           │ │
│  │ NASDAQ 100  │────┘│   │ - effective_date│  │   └───────────┘ │
│  │ History     │     │   │ - expire_date   │  │                 │
│  └─────────────┘     │   │ - status        │  │                 │
│  ┌─────────────┐     │   └─────────────────┘  │                 │
│  │ Delisting   │─────┼──▶┌─────────────────┐  │                 │
│  │ Events      │     │   │ delisting_      │  │                 │
│  └─────────────┘     │   │ events          │  │                 │
│                      │   └─────────────────┘  │                 │
└─────────────────────────────────────────────────────────────────┘
```

### Database Schema

#### Universe Constituents Table
```sql
CREATE TABLE universe_constituents (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    universe_type VARCHAR(50) NOT NULL,  -- 'sp500', 'nasdaq100', etc.
    effective_date DATE NOT NULL,        -- When symbol joined universe
    expiration_date DATE,                -- When symbol left (NULL if still active)
    listing_status VARCHAR(20) NOT NULL, -- 'active', 'delisted', 'merged', etc.
    delisting_reason VARCHAR(200),       -- Reason for delisting
    successor_symbol VARCHAR(20),        -- Acquiring company symbol
    data_source VARCHAR(50) NOT NULL,
    
    UNIQUE(symbol, universe_type, effective_date)
);
```

#### Delisting Events Table
```sql
CREATE TABLE delisting_events (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    delisting_date DATE NOT NULL,
    delisting_reason VARCHAR(200),
    last_trading_date DATE,
    final_price DECIMAL(10,4),
    event_type VARCHAR(50),              -- 'merger', 'acquisition', 'bankruptcy'
    successor_symbol VARCHAR(20),
    
    UNIQUE(symbol, delisting_date)
);
```

## Usage Examples

### 1. Loading Complete Universe for Backtesting

```python
from universe.ptfs_loader import get_universe_for_backtest, UniverseType
from datetime import date

# Get S&P 500 universe including delisted companies for 10-year backtest
start_date = date(2014, 1, 1)
end_date = date(2024, 1, 1)

universe_symbols = await get_universe_for_backtest(
    universe_type=UniverseType.SP500,
    start_date=start_date,
    end_date=end_date,
    include_delisted=True  # Critical for bias prevention
)

print(f"Universe contains {len(universe_symbols)} symbols")
# Output: Universe contains 847 symbols (includes ~200 delisted companies)
```

### 2. Point-in-Time Universe Construction

```python
from universe.ptfs_loader import PortfolioUniverseLoader

loader = PortfolioUniverseLoader()

# Get active universe as of specific historical date
historical_date = date(2020, 3, 15)  # COVID crash
active_symbols = await loader.get_active_universe(
    universe_type=UniverseType.SP500,
    as_of_date=historical_date
)

# This correctly excludes companies that weren't in S&P 500 yet as of March 2020
print(f"S&P 500 on {historical_date}: {len(active_symbols)} symbols")
```

### 3. Survivorship Bias Validation

```python
# Validate that backtest includes appropriate delisted securities
validation = await loader.validate_survivorship_bias_prevention(
    universe_type=UniverseType.SP500,
    backtest_start=date(2015, 1, 1),
    backtest_end=date(2024, 1, 1)
)

print(f"Backtest universe analysis:")
print(f"- Total symbols: {validation['universe_composition']['total_constituents']}")
print(f"- Delisted symbols: {validation['universe_composition']['delisted_constituents']}")
print(f"- Delisted percentage: {validation['universe_composition']['delisted_percentage']:.1f}%")
print(f"- Survivorship bias risk: {validation['survivorship_bias_assessment']['bias_risk']}")
```

### 4. Analyzing Delisting Events

```python
# Get detailed delisting information for analysis
delisting_events = await loader.load_delisting_events(
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31)
)

# Analyze delisting reasons
from collections import Counter
reasons = Counter(event['delisting_reason'] for event in delisting_events)

print("Delisting reasons (2020-2023):")
for reason, count in reasons.most_common():
    print(f"- {reason}: {count} companies")
```

## Testing Framework

### Automated Survivorship Tests

Our testing framework includes comprehensive checks for survivorship bias:

```python
async def test_survivorship_bias_prevention():
    """Test that backtests include delisted securities"""
    
    # Test 1: Verify delisted companies are included
    universe = await get_universe_for_backtest(
        UniverseType.SP500, 
        date(2010, 1, 1), 
        date(2020, 1, 1),
        include_delisted=True
    )
    
    delisted_symbols = await get_delisted_symbols_in_period(
        date(2010, 1, 1), 
        date(2020, 1, 1)
    )
    
    included_delisted = universe.intersection(delisted_symbols)
    
    assert len(included_delisted) > 0, "No delisted companies in universe"
    assert len(included_delisted) / len(delisted_symbols) > 0.8, \
        "Missing significant portion of delisted companies"
    
    # Test 2: Verify point-in-time accuracy
    # A company delisted in 2015 should not appear in 2010 universe
    for symbol in delisted_symbols:
        delisting_date = await get_delisting_date(symbol)
        pit_universe_before = await get_active_universe(
            UniverseType.SP500, 
            delisting_date - timedelta(days=30)
        )
        pit_universe_after = await get_active_universe(
            UniverseType.SP500, 
            delisting_date + timedelta(days=30)
        )
        
        if symbol in pit_universe_before:
            assert symbol not in pit_universe_after, \
                f"{symbol} still in universe after delisting"
```

### Manual Validation Procedures

#### Quarterly Universe Audit
1. **Compare Current vs Historical**: Verify that current S&P 500 differs from historical composition
2. **Delisting Rate Check**: Confirm 3-5% annual delisting rate (historical average)
3. **Sector Distribution**: Ensure delisted companies represent diverse sectors
4. **Random Sampling**: Manually verify delisting dates for sample of companies

#### Backtest Validation Checklist
- [ ] Universe includes companies delisted during backtest period
- [ ] Point-in-time universe construction prevents look-ahead bias
- [ ] Delisting events properly reflected in performance calculations
- [ ] Survivorship rate aligns with historical expectations (>85% survival over 10 years)

## Common Pitfalls and Solutions

### Pitfall 1: Using Current Index Composition
**Problem**: Using today's S&P 500 list for historical backtests
**Solution**: Implement point-in-time universe reconstruction

### Pitfall 2: Excluding "Penny Stocks"
**Problem**: Filtering out stocks below $5 may exclude pre-delisting companies
**Solution**: Apply price filters at point-in-time, not retrospectively

### Pitfall 3: Missing Corporate Actions
**Problem**: Not tracking mergers, spin-offs, and symbol changes
**Solution**: Maintain comprehensive corporate action database

### Pitfall 4: Inadequate Delisting Data
**Problem**: Using only bankruptcy delistings, missing acquisitions
**Solution**: Track all delisting types: bankruptcy, merger, acquisition, voluntary

## Data Sources and Maintenance

### Primary Data Sources
1. **Index Providers**: S&P Dow Jones, NASDAQ, Russell Investments
2. **Exchange Data**: NYSE, NASDAQ delisting notifications
3. **Corporate Actions**: Bloomberg, Refinitiv corporate action feeds
4. **Regulatory Filings**: SEC EDGAR system for merger/acquisition announcements

### Data Quality Assurance
- **Daily Updates**: Process delisting notifications within 24 hours
- **Historical Validation**: Cross-reference multiple sources for historical changes
- **Corporate Action Tracking**: Maintain full audit trail of universe changes
- **Automated Testing**: Run survivorship bias tests on each data update

## Best Practices

### For Strategy Development
1. **Always Include Delisted**: Default to including delisted securities unless specifically excluded
2. **Document Exclusions**: If excluding delisted companies, document rationale and impact
3. **Sensitivity Analysis**: Test strategy performance with and without delisted companies
4. **Period Analysis**: Break down results by time periods to identify bias impact

### For Risk Management
1. **Stress Testing**: Include crisis periods with high delisting rates (2008, 2020)
2. **Concentration Risk**: Monitor exposure to companies with elevated delisting risk
3. **Sector Analysis**: Understand delisting patterns by sector and market cap
4. **Forward-Looking**: Use survivorship statistics to inform position sizing

### For Model Validation
1. **Out-of-Sample Testing**: Reserve recent delisting events for model validation
2. **Cross-Validation**: Test models on different time periods and universes
3. **Benchmark Comparison**: Compare results against survivorship-biased benchmarks
4. **Statistical Testing**: Use hypothesis tests to validate bias mitigation effectiveness

## Monitoring and Alerts

### Automated Monitoring
- **Daily Delisting Checks**: Alert on new delisting announcements
- **Universe Drift**: Monitor changes in universe composition over time
- **Bias Metrics**: Track survivorship bias indicators in live strategies
- **Data Quality**: Alert on missing or inconsistent universe data

### Quarterly Review Process
1. Review delisting events and corporate actions
2. Validate universe reconstruction accuracy
3. Update data sources and vendor relationships
4. Assess impact on live trading strategies

## Conclusion

Survivorship bias prevention is not optional—it's essential for realistic strategy development and risk management. Our comprehensive system ensures:

- **Accurate Historical Analysis**: Including all securities that existed during backtest periods
- **Realistic Risk Assessment**: Capturing true downside risks including bankruptcy and delisting
- **Robust Strategy Development**: Building strategies that work in real-world conditions
- **Regulatory Compliance**: Meeting institutional standards for model validation

By implementing these procedures and continuously monitoring for bias, we ensure our trading platform produces reliable, institutional-quality results that translate effectively to live trading environments.

---

*For technical implementation details, see:*
- `universe/ptfs_loader.py` - Universe loading implementation
- `tests/pit/test_survivorship.sql` - Automated testing suite
- `migrations/012_universe_tables.sql` - Database schema