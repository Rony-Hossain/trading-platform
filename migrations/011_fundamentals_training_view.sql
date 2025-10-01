-- Migration: Create training view for fundamentals (first-print only)
-- Purpose: Ensure training queries only use first-print data to prevent look-ahead bias

-- Create view that only exposes first-print fundamentals for training
CREATE OR REPLACE VIEW vw_fundamentals_training AS
SELECT 
    fp.id,
    fp.symbol,
    fp.report_date,
    fp.period_end_date,
    fp.fiscal_year,
    fp.fiscal_quarter,
    fp.first_print_timestamp AS as_of_timestamp,
    
    -- Core financial metrics (first-print values only)
    fp.revenue,
    fp.net_income,
    fp.total_assets,
    fp.total_equity,
    fp.total_debt,
    fp.cash_and_equivalents,
    fp.operating_cash_flow,
    fp.free_cash_flow,
    
    -- Per-share metrics
    fp.earnings_per_share,
    fp.book_value_per_share,
    fp.shares_outstanding,
    
    -- Derived ratios (calculated from first-print values)
    fp.debt_to_equity_ratio,
    fp.current_ratio,
    fp.return_on_equity,
    fp.return_on_assets,
    fp.gross_margin,
    fp.operating_margin,
    fp.net_margin,
    
    -- Data lineage
    fp.data_source,
    fp.revision_number,
    fp.is_preliminary,
    fp.created_at,
    
    -- Training-specific metadata
    'first_print_only' AS data_type,
    CASE 
        WHEN fp.first_print_timestamp <= fp.period_end_date + INTERVAL '90 days' THEN 'timely'
        WHEN fp.first_print_timestamp <= fp.period_end_date + INTERVAL '180 days' THEN 'delayed'
        ELSE 'very_delayed'
    END AS reporting_timeliness,
    
    -- Point-in-time validation flags
    CASE 
        WHEN fp.first_print_timestamp > NOW() THEN 'INVALID_FUTURE_TIMESTAMP'
        WHEN fp.first_print_timestamp::DATE < fp.period_end_date + INTERVAL '30 days' THEN 'INVALID_TOO_EARLY'
        ELSE 'VALID'
    END AS pit_validation_status

FROM fundamentals_first_print fp
WHERE 
    -- Only include valid first-print records
    fp.first_print_timestamp <= NOW()
    AND fp.first_print_timestamp::DATE >= fp.period_end_date + INTERVAL '30 days'
    
    -- Exclude obviously invalid data
    AND fp.revenue IS NOT NULL
    AND fp.period_end_date IS NOT NULL
    AND fp.fiscal_year BETWEEN 1990 AND EXTRACT(YEAR FROM NOW()) + 1;

-- Create materialized view for better performance on large datasets
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_fundamentals_training_features AS
SELECT 
    ft.*,
    
    -- Additional derived features for training
    CASE 
        WHEN LAG(ft.revenue) OVER (PARTITION BY ft.symbol ORDER BY ft.period_end_date) > 0 
        THEN (ft.revenue - LAG(ft.revenue) OVER (PARTITION BY ft.symbol ORDER BY ft.period_end_date)) 
             / LAG(ft.revenue) OVER (PARTITION BY ft.symbol ORDER BY ft.period_end_date)
        ELSE NULL
    END AS revenue_growth_qoq,
    
    CASE 
        WHEN LAG(ft.net_income, 4) OVER (PARTITION BY ft.symbol ORDER BY ft.period_end_date) > 0 
        THEN (ft.net_income - LAG(ft.net_income, 4) OVER (PARTITION BY ft.symbol ORDER BY ft.period_end_date)) 
             / LAG(ft.net_income, 4) OVER (PARTITION BY ft.symbol ORDER BY ft.period_end_date)
        ELSE NULL
    END AS net_income_growth_yoy,
    
    -- Trend indicators
    AVG(ft.return_on_equity) OVER (
        PARTITION BY ft.symbol 
        ORDER BY ft.period_end_date 
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS roe_4q_avg,
    
    STDDEV(ft.earnings_per_share) OVER (
        PARTITION BY ft.symbol 
        ORDER BY ft.period_end_date 
        ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
    ) AS eps_8q_volatility,
    
    -- Quality scores
    CASE 
        WHEN ft.operating_cash_flow > ft.net_income * 0.8 THEN 'high'
        WHEN ft.operating_cash_flow > ft.net_income * 0.5 THEN 'medium'
        ELSE 'low'
    END AS earnings_quality,
    
    -- Sector percentiles (approximated)
    PERCENT_RANK() OVER (
        PARTITION BY ft.fiscal_year, ft.fiscal_quarter 
        ORDER BY ft.return_on_equity
    ) AS roe_percentile_by_quarter,
    
    -- Data freshness indicators
    EXTRACT(DAYS FROM ft.as_of_timestamp - ft.period_end_date) AS days_to_report,
    
    -- Training-specific features
    ROW_NUMBER() OVER (
        PARTITION BY ft.symbol 
        ORDER BY ft.period_end_date
    ) AS quarter_sequence,
    
    COUNT(*) OVER (
        PARTITION BY ft.symbol
    ) AS total_quarters_available

FROM vw_fundamentals_training ft;

-- Create indexes for the materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_fundamentals_training_unique
ON mv_fundamentals_training_features(symbol, report_date, period_end_date);

CREATE INDEX IF NOT EXISTS idx_mv_fundamentals_training_symbol_date
ON mv_fundamentals_training_features(symbol, period_end_date);

CREATE INDEX IF NOT EXISTS idx_mv_fundamentals_training_date
ON mv_fundamentals_training_features(period_end_date);

CREATE INDEX IF NOT EXISTS idx_mv_fundamentals_training_fiscal
ON mv_fundamentals_training_features(fiscal_year, fiscal_quarter);

-- Function to refresh the materialized view
CREATE OR REPLACE FUNCTION refresh_fundamentals_training_features()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_fundamentals_training_features;
    
    -- Log the refresh
    INSERT INTO materialized_view_refreshes (view_name, refresh_timestamp, row_count)
    SELECT 
        'mv_fundamentals_training_features',
        NOW(),
        COUNT(*)
    FROM mv_fundamentals_training_features;
    
END;
$$ LANGUAGE plpgsql;

-- Create table to track materialized view refreshes
CREATE TABLE IF NOT EXISTS materialized_view_refreshes (
    id SERIAL PRIMARY KEY,
    view_name VARCHAR(100) NOT NULL,
    refresh_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    row_count BIGINT,
    duration_seconds DECIMAL(10,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mv_refreshes_view_time
ON materialized_view_refreshes(view_name, refresh_timestamp DESC);

-- Point-in-Time training data function
CREATE OR REPLACE FUNCTION get_training_fundamentals_pit(
    p_symbol VARCHAR(20),
    p_as_of_date TIMESTAMP WITH TIME ZONE,
    p_lookback_quarters INTEGER DEFAULT 12
) RETURNS TABLE (
    symbol VARCHAR(20),
    period_end_date DATE,
    revenue DECIMAL(20,2),
    net_income DECIMAL(20,2),
    earnings_per_share DECIMAL(10,4),
    return_on_equity DECIMAL(8,4),
    revenue_growth_qoq DECIMAL(8,4),
    net_income_growth_yoy DECIMAL(8,4),
    days_to_report INTEGER,
    quarter_sequence BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ft.symbol,
        ft.period_end_date,
        ft.revenue,
        ft.net_income,
        ft.earnings_per_share,
        ft.return_on_equity,
        ft.revenue_growth_qoq,
        ft.net_income_growth_yoy,
        ft.days_to_report::INTEGER,
        ft.quarter_sequence
    FROM mv_fundamentals_training_features ft
    WHERE ft.symbol = p_symbol
    AND ft.as_of_timestamp <= p_as_of_date
    ORDER BY ft.period_end_date DESC
    LIMIT p_lookback_quarters;
END;
$$ LANGUAGE plpgsql;

-- Training data validation function
CREATE OR REPLACE FUNCTION validate_training_fundamentals_pit()
RETURNS TABLE (
    violation_type VARCHAR(50),
    symbol VARCHAR(20),
    period_end_date DATE,
    as_of_timestamp TIMESTAMP WITH TIME ZONE,
    description TEXT,
    severity VARCHAR(20)
) AS $$
BEGIN
    -- Check 1: No future-dated training data
    RETURN QUERY
    SELECT 
        'future_training_data'::VARCHAR(50),
        ft.symbol,
        ft.period_end_date,
        ft.as_of_timestamp,
        'Training data has future timestamp'::TEXT,
        'critical'::VARCHAR(20)
    FROM vw_fundamentals_training ft
    WHERE ft.as_of_timestamp > NOW();
    
    -- Check 2: Check for potential look-ahead bias
    RETURN QUERY
    SELECT 
        'potential_lookahead_bias'::VARCHAR(50),
        ft.symbol,
        ft.period_end_date,
        ft.as_of_timestamp,
        'Data available too soon after period end'::TEXT,
        'high'::VARCHAR(20)
    FROM vw_fundamentals_training ft
    WHERE ft.as_of_timestamp::DATE < ft.period_end_date + INTERVAL '30 days';
    
    -- Check 3: Validate against latest data contamination
    RETURN QUERY
    SELECT 
        'latest_data_contamination'::VARCHAR(50),
        ft.symbol,
        ft.period_end_date,
        ft.as_of_timestamp,
        'Training view may contain revised data'::TEXT,
        'medium'::VARCHAR(20)
    FROM vw_fundamentals_training ft
    INNER JOIN fundamentals_latest fl ON 
        ft.symbol = fl.symbol 
        AND ft.period_end_date = fl.period_end_date
        AND fl.revision_number > 0
    WHERE NOT EXISTS (
        SELECT 1 FROM fundamentals_first_print fp 
        WHERE fp.symbol = ft.symbol 
        AND fp.period_end_date = ft.period_end_date
    );
    
    -- Check 4: Data completeness
    RETURN QUERY
    SELECT 
        'incomplete_training_data'::VARCHAR(50),
        ft.symbol,
        ft.period_end_date,
        ft.as_of_timestamp,
        'Missing critical training features'::TEXT,
        'medium'::VARCHAR(20)
    FROM vw_fundamentals_training ft
    WHERE ft.revenue IS NULL 
    OR ft.net_income IS NULL 
    OR ft.earnings_per_share IS NULL;
    
END;
$$ LANGUAGE plpgsql;

-- Automated refresh schedule function
CREATE OR REPLACE FUNCTION schedule_fundamentals_training_refresh()
RETURNS VOID AS $$
BEGIN
    -- This would typically be called by a cron job or scheduler
    PERFORM refresh_fundamentals_training_features();
    
    -- Validate after refresh
    INSERT INTO pit_compliance_checks (
        check_type, check_timestamp, violations_found, check_status
    )
    SELECT 
        'training_fundamentals_validation',
        NOW(),
        COUNT(*),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM validate_training_fundamentals_pit();
    
END;
$$ LANGUAGE plpgsql;

-- Create table for compliance check results
CREATE TABLE IF NOT EXISTS pit_compliance_checks (
    id SERIAL PRIMARY KEY,
    check_type VARCHAR(50) NOT NULL,
    check_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    violations_found INTEGER NOT NULL DEFAULT 0,
    check_status VARCHAR(20) NOT NULL, -- 'PASS', 'FAIL', 'WARNING'
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pit_compliance_checks_type_time
ON pit_compliance_checks(check_type, check_timestamp DESC);

-- Grant permissions
GRANT SELECT ON vw_fundamentals_training TO PUBLIC;
GRANT SELECT ON mv_fundamentals_training_features TO PUBLIC;
GRANT SELECT ON materialized_view_refreshes TO PUBLIC;
GRANT SELECT ON pit_compliance_checks TO PUBLIC;

-- Comments
COMMENT ON VIEW vw_fundamentals_training IS 'Training view exposing only first-print fundamentals data (PIT compliant)';
COMMENT ON MATERIALIZED VIEW mv_fundamentals_training_features IS 'Pre-computed training features from first-print fundamentals only';
COMMENT ON FUNCTION get_training_fundamentals_pit IS 'Point-in-Time lookup for training fundamentals as known at specific date';
COMMENT ON FUNCTION validate_training_fundamentals_pit IS 'Validates training data for Point-in-Time compliance violations';