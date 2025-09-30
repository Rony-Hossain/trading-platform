-- Migration 012: Fundamentals Training Views and PIT Compliance
-- Creates strict training-only views that enforce first-print data usage
-- Adds additional constraints and monitoring for Point-in-Time compliance

-- Enhanced training view that enforces strict first-print only access
-- This view should be used by ALL training pipelines to prevent look-ahead bias
CREATE OR REPLACE VIEW vw_fundamentals_training AS
WITH training_constraints AS (
    SELECT 
        COALESCE(MIN(CASE WHEN constraint_type = 'fundamentals_lag' THEN min_lag_days END), 45) as min_lag_days,
        COALESCE(MIN(CASE WHEN constraint_type = 'data_quality_threshold' THEN min_quality_score END), 0.85) as min_quality,
        COALESCE(MAX(CASE WHEN constraint_type = 'revision_exclusion' THEN 1 ELSE 0 END), 1) as exclude_amendments
    FROM training_data_constraints 
    WHERE is_active = TRUE
),
valid_data AS (
    SELECT 
        fp.*,
        tc.min_lag_days,
        tc.min_quality,
        tc.exclude_amendments,
        -- Calculate availability timestamp (when data becomes usable for training)
        fp.first_print_timestamp + INTERVAL '1 day' * tc.min_lag_days as training_available_timestamp,
        -- Add data staleness check
        EXTRACT(DAYS FROM NOW() - fp.first_print_timestamp) as days_since_first_print,
        -- Quality flags
        CASE 
            WHEN fp.data_quality_score >= tc.min_quality THEN TRUE 
            ELSE FALSE 
        END as meets_quality_threshold,
        CASE 
            WHEN fp.amendment_flag = FALSE OR tc.exclude_amendments = 0 THEN TRUE 
            ELSE FALSE 
        END as meets_amendment_rules
    FROM fundamentals_first_print fp
    CROSS JOIN training_constraints tc
    WHERE fp.data_quality_score IS NOT NULL
)
SELECT 
    -- Core identifiers
    symbol,
    report_date,
    filing_date,
    first_print_timestamp,
    training_available_timestamp,
    period_type,
    fiscal_year,
    
    -- Financial metrics (first-print only)
    revenue,
    net_income,
    total_assets,
    total_liabilities,
    shareholders_equity,
    operating_income,
    gross_profit,
    total_debt,
    cash_and_equivalents,
    free_cash_flow,
    
    -- Per-share metrics
    earnings_per_share,
    book_value_per_share,
    diluted_shares_outstanding,
    
    -- Key ratios
    roe,
    roa,
    debt_to_equity,
    current_ratio,
    gross_margin,
    operating_margin,
    net_margin,
    
    -- Data quality and compliance fields
    data_source,
    filing_type,
    amendment_flag,
    revision_count,
    data_quality_score,
    meets_quality_threshold,
    meets_amendment_rules,
    days_since_first_print,
    
    -- Metadata
    created_at,
    updated_at,
    
    -- Training compliance flags
    CASE 
        WHEN meets_quality_threshold AND meets_amendment_rules THEN TRUE 
        ELSE FALSE 
    END as training_compliant,
    
    -- Availability check for point-in-time usage
    CASE 
        WHEN NOW() >= training_available_timestamp THEN TRUE 
        ELSE FALSE 
    END as available_for_training

FROM valid_data
WHERE meets_quality_threshold = TRUE 
  AND meets_amendment_rules = TRUE
ORDER BY symbol, report_date DESC, first_print_timestamp DESC;

-- Materialized view for better performance on large datasets
CREATE MATERIALIZED VIEW IF NOT EXISTS mvw_fundamentals_training_fast AS
SELECT 
    *,
    -- Pre-calculate common training features
    ROW_NUMBER() OVER (PARTITION BY symbol, period_type ORDER BY report_date DESC) as recency_rank,
    LAG(revenue) OVER (PARTITION BY symbol, period_type ORDER BY report_date) as prev_revenue,
    LAG(net_income) OVER (PARTITION BY symbol, period_type ORDER BY report_date) as prev_net_income,
    LAG(earnings_per_share) OVER (PARTITION BY symbol, period_type ORDER BY report_date) as prev_eps,
    
    -- Growth calculations (year-over-year for quarterly data)
    CASE 
        WHEN period_type IN ('Q1', 'Q2', 'Q3', 'Q4') THEN
            LAG(revenue, 4) OVER (PARTITION BY symbol, period_type ORDER BY fiscal_year, period_type)
        WHEN period_type = 'FY' THEN
            LAG(revenue, 1) OVER (PARTITION BY symbol, period_type ORDER BY fiscal_year)
        ELSE NULL
    END as revenue_yoy_comparison,
    
    CASE 
        WHEN period_type IN ('Q1', 'Q2', 'Q3', 'Q4') THEN
            LAG(net_income, 4) OVER (PARTITION BY symbol, period_type ORDER BY fiscal_year, period_type)
        WHEN period_type = 'FY' THEN
            LAG(net_income, 1) OVER (PARTITION BY symbol, period_type ORDER BY fiscal_year)
        ELSE NULL
    END as net_income_yoy_comparison
    
FROM vw_fundamentals_training
WHERE available_for_training = TRUE;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_mvw_fundamentals_training_fast_unique 
ON mvw_fundamentals_training_fast (symbol, report_date, period_type, fiscal_year);

-- Index for common query patterns
CREATE INDEX IF NOT EXISTS idx_mvw_fundamentals_training_fast_symbol_date 
ON mvw_fundamentals_training_fast (symbol, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_mvw_fundamentals_training_fast_available 
ON mvw_fundamentals_training_fast (training_available_timestamp DESC) 
WHERE available_for_training = TRUE;

-- Function to refresh the materialized view
CREATE OR REPLACE FUNCTION refresh_fundamentals_training_view()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mvw_fundamentals_training_fast;
    
    -- Log the refresh
    INSERT INTO fundamentals_access_log (
        user_id, query_type, data_version, access_granted, 
        records_returned, timestamp
    ) VALUES (
        'system', 'training_data', 'first_print', TRUE,
        (SELECT COUNT(*) FROM mvw_fundamentals_training_fast),
        NOW()
    );
END;
$$ LANGUAGE plpgsql;

-- Training data compliance validation function
CREATE OR REPLACE FUNCTION validate_training_data_compliance(
    p_symbols TEXT[] DEFAULT NULL,
    p_as_of_date DATE DEFAULT CURRENT_DATE,
    p_max_age_days INTEGER DEFAULT 90
) RETURNS TABLE (
    symbol VARCHAR(20),
    report_date DATE,
    compliance_status VARCHAR(20),
    violation_details TEXT[],
    data_age_days INTEGER,
    quality_score DECIMAL(3,2),
    recommended_action TEXT
) AS $$
DECLARE
    min_lag_days INTEGER;
    min_quality_score DECIMAL(3,2);
    rec RECORD;
    violations TEXT[];
BEGIN
    -- Get current constraints
    SELECT 
        COALESCE(MIN(CASE WHEN constraint_type = 'fundamentals_lag' THEN min_lag_days END), 45),
        COALESCE(MIN(CASE WHEN constraint_type = 'data_quality_threshold' THEN min_quality_score END), 0.85)
    INTO min_lag_days, min_quality_score
    FROM training_data_constraints 
    WHERE is_active = TRUE;
    
    FOR rec IN 
        SELECT 
            fp.symbol,
            fp.report_date,
            fp.first_print_timestamp,
            fp.data_quality_score,
            fp.amendment_flag,
            fp.revision_count,
            EXTRACT(DAYS FROM p_as_of_date::TIMESTAMPTZ - fp.first_print_timestamp) as age_days
        FROM fundamentals_first_print fp
        WHERE (p_symbols IS NULL OR fp.symbol = ANY(p_symbols))
        AND fp.report_date >= (p_as_of_date - INTERVAL '1 year')
        AND EXTRACT(DAYS FROM p_as_of_date::TIMESTAMPTZ - fp.first_print_timestamp) <= p_max_age_days
    LOOP
        violations := '{}';
        
        -- Check lag requirement
        IF rec.age_days < min_lag_days THEN
            violations := array_append(violations, 'insufficient_lag: ' || rec.age_days || ' < ' || min_lag_days);
        END IF;
        
        -- Check quality score
        IF rec.data_quality_score < min_quality_score THEN
            violations := array_append(violations, 'low_quality: ' || rec.data_quality_score || ' < ' || min_quality_score);
        END IF;
        
        -- Check for amendments
        IF rec.amendment_flag = TRUE THEN
            violations := array_append(violations, 'amendment_filing_excluded');
        END IF;
        
        -- Check for excessive revisions
        IF rec.revision_count > 2 THEN
            violations := array_append(violations, 'excessive_revisions: ' || rec.revision_count);
        END IF;
        
        -- Determine status and recommendation
        RETURN QUERY SELECT 
            rec.symbol,
            rec.report_date,
            CASE 
                WHEN array_length(violations, 1) IS NULL THEN 'COMPLIANT'::VARCHAR(20)
                WHEN 'insufficient_lag' = ANY(violations) THEN 'PENDING'::VARCHAR(20)
                ELSE 'VIOLATION'::VARCHAR(20)
            END,
            violations,
            rec.age_days::INTEGER,
            rec.data_quality_score,
            CASE 
                WHEN array_length(violations, 1) IS NULL THEN 'Safe for training use'
                WHEN 'insufficient_lag' = ANY(violations) THEN 'Wait ' || (min_lag_days - rec.age_days) || ' more days'
                ELSE 'Exclude from training data'
            END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to get training-safe fundamentals with automatic compliance checking
CREATE OR REPLACE FUNCTION get_training_safe_fundamentals(
    p_symbols TEXT[],
    p_as_of_date DATE DEFAULT CURRENT_DATE,
    p_lookback_years INTEGER DEFAULT 5,
    p_min_observations INTEGER DEFAULT 8
) RETURNS TABLE (
    symbol VARCHAR(20),
    report_date DATE,
    period_type VARCHAR(10),
    fiscal_year INTEGER,
    revenue BIGINT,
    net_income BIGINT,
    earnings_per_share DECIMAL(10,4),
    roe DECIMAL(8,4),
    roa DECIMAL(8,4),
    debt_to_equity DECIMAL(8,4),
    revenue_growth_yoy DECIMAL(8,4),
    eps_growth_yoy DECIMAL(8,4),
    data_quality_score DECIMAL(3,2),
    training_available_timestamp TIMESTAMPTZ,
    compliance_certified BOOLEAN
) AS $$
BEGIN
    -- Log the access request
    INSERT INTO fundamentals_access_log (
        user_id, query_type, symbols_requested, date_range_start, date_range_end,
        data_version, constraints_applied, access_granted, timestamp
    ) VALUES (
        session_user, 'training_data', p_symbols, 
        p_as_of_date - INTERVAL '1 year' * p_lookback_years, p_as_of_date,
        'first_print', ARRAY['fundamentals_lag', 'data_quality_threshold', 'revision_exclusion'],
        TRUE, NOW()
    );
    
    RETURN QUERY
    WITH training_data AS (
        SELECT 
            mv.*,
            -- Calculate growth rates
            CASE 
                WHEN mv.revenue_yoy_comparison IS NOT NULL AND mv.revenue_yoy_comparison > 0 THEN
                    ((mv.revenue::DECIMAL - mv.revenue_yoy_comparison::DECIMAL) / mv.revenue_yoy_comparison::DECIMAL) * 100
                ELSE NULL
            END as revenue_growth_yoy_calc,
            CASE 
                WHEN mv.net_income_yoy_comparison IS NOT NULL AND mv.net_income_yoy_comparison != 0 THEN
                    ((mv.net_income::DECIMAL - mv.net_income_yoy_comparison::DECIMAL) / ABS(mv.net_income_yoy_comparison::DECIMAL)) * 100
                ELSE NULL
            END as eps_growth_yoy_calc
        FROM mvw_fundamentals_training_fast mv
        WHERE mv.symbol = ANY(p_symbols)
        AND mv.report_date >= (p_as_of_date - INTERVAL '1 year' * p_lookback_years)
        AND mv.training_available_timestamp <= p_as_of_date::TIMESTAMPTZ
        AND mv.available_for_training = TRUE
    ),
    symbol_counts AS (
        SELECT symbol, COUNT(*) as observation_count
        FROM training_data
        GROUP BY symbol
        HAVING COUNT(*) >= p_min_observations
    )
    SELECT 
        td.symbol,
        td.report_date,
        td.period_type,
        td.fiscal_year,
        td.revenue,
        td.net_income,
        td.earnings_per_share,
        td.roe,
        td.roa,
        td.debt_to_equity,
        td.revenue_growth_yoy_calc,
        CASE 
            WHEN td.earnings_per_share IS NOT NULL AND td.prev_eps IS NOT NULL THEN
                ((td.earnings_per_share - td.prev_eps) / ABS(td.prev_eps)) * 100
            ELSE td.eps_growth_yoy_calc
        END,
        td.data_quality_score,
        td.training_available_timestamp,
        TRUE as compliance_certified -- All data from this function is certified compliant
    FROM training_data td
    INNER JOIN symbol_counts sc ON td.symbol = sc.symbol
    ORDER BY td.symbol, td.report_date DESC;
END;
$$ LANGUAGE plpgsql;

-- Row Level Security for training views
ALTER TABLE fundamentals_first_print ENABLE ROW LEVEL SECURITY;
ALTER TABLE fundamentals_latest ENABLE ROW LEVEL SECURITY;

-- Policy to restrict training access to first-print data only
CREATE POLICY training_access_first_print_only ON fundamentals_first_print
    FOR SELECT
    TO trading_user
    USING (
        amendment_flag = FALSE 
        AND data_quality_score >= 0.85
        AND first_print_timestamp <= (NOW() - INTERVAL '45 days')
    );

-- Policy to prevent training pipelines from accessing latest data
CREATE POLICY no_training_access_latest ON fundamentals_latest
    FOR SELECT
    TO trading_user
    USING (
        -- Only allow access for non-training purposes
        -- This would need to be configured based on application roles
        current_setting('app.context', true) != 'training'
    );

-- Create a dedicated training role with restricted access
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trading_training_user') THEN
        CREATE ROLE trading_training_user;
    END IF;
END $$;

-- Grant permissions to training role
GRANT SELECT ON vw_fundamentals_training TO trading_training_user;
GRANT SELECT ON mvw_fundamentals_training_fast TO trading_training_user;
GRANT EXECUTE ON FUNCTION get_training_safe_fundamentals TO trading_training_user;
GRANT EXECUTE ON FUNCTION validate_training_data_compliance TO trading_training_user;

-- Deny access to latest data for training role
REVOKE SELECT ON fundamentals_latest FROM trading_training_user;

-- Create monitoring view for compliance tracking
CREATE OR REPLACE VIEW vw_training_compliance_monitor AS
SELECT 
    DATE_TRUNC('day', fal.timestamp) as access_date,
    fal.user_id,
    fal.query_type,
    fal.data_version,
    COUNT(*) as access_count,
    SUM(CASE WHEN fal.access_granted THEN 1 ELSE 0 END) as granted_count,
    SUM(CASE WHEN NOT fal.access_granted THEN 1 ELSE 0 END) as denied_count,
    SUM(fal.records_returned) as total_records_accessed,
    ARRAY_AGG(DISTINCT fal.violations_detected) FILTER (WHERE fal.violations_detected IS NOT NULL) as violations_summary
FROM fundamentals_access_log fal
WHERE fal.timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', fal.timestamp), fal.user_id, fal.query_type, fal.data_version
ORDER BY access_date DESC, user_id;

-- Function to audit fundamentals access patterns
CREATE OR REPLACE FUNCTION audit_fundamentals_access_patterns(
    p_days_back INTEGER DEFAULT 30
) RETURNS TABLE (
    user_id TEXT,
    data_version_used TEXT,
    total_accesses BIGINT,
    training_accesses BIGINT,
    latest_data_accesses BIGINT,
    compliance_violations BIGINT,
    risk_score INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fal.user_id::TEXT,
        fal.data_version::TEXT,
        COUNT(*) as total_accesses,
        SUM(CASE WHEN fal.query_type = 'training_data' THEN 1 ELSE 0 END) as training_accesses,
        SUM(CASE WHEN fal.data_version = 'latest' AND fal.query_type = 'training_data' THEN 1 ELSE 0 END) as latest_data_accesses,
        SUM(CASE WHEN array_length(fal.violations_detected, 1) > 0 THEN 1 ELSE 0 END) as compliance_violations,
        -- Risk score: 0-100 based on compliance violations and inappropriate access patterns
        LEAST(100, 
            (SUM(CASE WHEN fal.data_version = 'latest' AND fal.query_type = 'training_data' THEN 10 ELSE 0 END)::INTEGER +
             SUM(CASE WHEN array_length(fal.violations_detected, 1) > 0 THEN 5 ELSE 0 END)::INTEGER)
        ) as risk_score
    FROM fundamentals_access_log fal
    WHERE fal.timestamp >= CURRENT_DATE - INTERVAL '1 day' * p_days_back
    GROUP BY fal.user_id, fal.data_version
    ORDER BY risk_score DESC, total_accesses DESC;
END;
$$ LANGUAGE plpgsql;

-- Convert new tables to TimescaleDB hypertables if available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Convert access log to hypertable for better performance
        PERFORM create_hypertable(
            'fundamentals_access_log',
            'timestamp',
            chunk_time_interval => INTERVAL '1 week',
            if_not_exists => TRUE
        );
        
        -- Set up retention policy for access logs
        PERFORM add_retention_policy(
            'fundamentals_access_log',
            INTERVAL '2 years',
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'TimescaleDB hypertable created for fundamentals_access_log';
    END IF;
END $$;

-- Refresh materialized view initially
SELECT refresh_fundamentals_training_view();

-- Schedule automatic refresh (this would typically be done via cron or application scheduler)
-- For now, just document the recommended refresh schedule
COMMENT ON FUNCTION refresh_fundamentals_training_view IS 
'Should be called daily after new fundamentals data is loaded. Recommended schedule: 0 1 * * * (1 AM daily)';

-- Final permissions and comments
GRANT SELECT ON vw_training_compliance_monitor TO trading_user;
GRANT EXECUTE ON FUNCTION audit_fundamentals_access_patterns TO trading_user;

-- Table and view comments
COMMENT ON VIEW vw_fundamentals_training IS 
'Training-only view that enforces first-print data usage and PIT compliance. Use this view for ALL training pipelines.';

COMMENT ON MATERIALIZED VIEW mvw_fundamentals_training_fast IS 
'High-performance materialized view for training data with pre-calculated features. Refreshed daily.';

COMMENT ON FUNCTION get_training_safe_fundamentals IS 
'Primary function for accessing fundamentals in training. Automatically enforces all compliance constraints and logs access.';

COMMENT ON FUNCTION validate_training_data_compliance IS 
'Validates that proposed fundamentals usage complies with PIT constraints. Use before training to check compliance.';

COMMENT ON VIEW vw_training_compliance_monitor IS 
'Monitor fundamentals data access patterns for compliance violations and inappropriate usage.';

COMMENT ON FUNCTION audit_fundamentals_access_patterns IS 
'Audit user access patterns to identify potential compliance violations or misuse of latest data in training.';

-- Insert initial compliance validation for existing data
INSERT INTO fundamentals_access_log (
    user_id, query_type, data_version, constraints_applied, access_granted, 
    records_returned, timestamp
) 
SELECT 
    'system_migration', 'data_validation', 'first_print', 
    ARRAY['fundamentals_lag', 'data_quality_threshold'],
    TRUE,
    COUNT(*),
    NOW()
FROM vw_fundamentals_training;