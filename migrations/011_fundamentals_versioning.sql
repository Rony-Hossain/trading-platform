-- Migration 011: Fundamentals Data Versioning System
-- Separates first-print vs latest fundamentals to prevent look-ahead bias
-- Ensures training only uses data available at the time

-- First-Print Fundamentals Table (original releases)
CREATE TABLE IF NOT EXISTS fundamentals_first_print (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL, -- The date the financial period ended
    filing_date DATE NOT NULL, -- The date the filing was submitted
    first_print_timestamp TIMESTAMPTZ NOT NULL, -- When this data first became available
    period_type VARCHAR(10) NOT NULL CHECK (period_type IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY', 'TTM')),
    fiscal_year INTEGER NOT NULL,
    
    -- Core financial metrics (as originally reported)
    revenue BIGINT, -- in thousands
    net_income BIGINT,
    total_assets BIGINT,
    total_liabilities BIGINT,
    shareholders_equity BIGINT,
    operating_income BIGINT,
    gross_profit BIGINT,
    total_debt BIGINT,
    cash_and_equivalents BIGINT,
    free_cash_flow BIGINT,
    
    -- Per-share metrics
    earnings_per_share DECIMAL(10,4),
    book_value_per_share DECIMAL(10,4),
    diluted_shares_outstanding BIGINT,
    
    -- Key ratios (calculated from first-print data)
    roe DECIMAL(8,4), -- Return on Equity
    roa DECIMAL(8,4), -- Return on Assets
    debt_to_equity DECIMAL(8,4),
    current_ratio DECIMAL(8,4),
    gross_margin DECIMAL(8,4),
    operating_margin DECIMAL(8,4),
    net_margin DECIMAL(8,4),
    
    -- Data quality and source tracking
    data_source VARCHAR(50) NOT NULL, -- 'sec_edgar', 'factset', 'refinitiv', etc.
    filing_type VARCHAR(20), -- '10-K', '10-Q', '8-K', etc.
    amendment_flag BOOLEAN DEFAULT FALSE,
    revision_count INTEGER DEFAULT 0,
    data_quality_score DECIMAL(3,2) DEFAULT 1.0,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(symbol, report_date, period_type, fiscal_year, first_print_timestamp)
);

-- Latest/Revised Fundamentals Table (most current versions)
CREATE TABLE IF NOT EXISTS fundamentals_latest (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    filing_date DATE NOT NULL,
    first_print_timestamp TIMESTAMPTZ NOT NULL, -- When data first became available
    latest_revision_timestamp TIMESTAMPTZ NOT NULL, -- When latest revision was published
    period_type VARCHAR(10) NOT NULL CHECK (period_type IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY', 'TTM')),
    fiscal_year INTEGER NOT NULL,
    
    -- Core financial metrics (latest revised values)
    revenue BIGINT,
    net_income BIGINT,
    total_assets BIGINT,
    total_liabilities BIGINT,
    shareholders_equity BIGINT,
    operating_income BIGINT,
    gross_profit BIGINT,
    total_debt BIGINT,
    cash_and_equivalents BIGINT,
    free_cash_flow BIGINT,
    
    -- Per-share metrics
    earnings_per_share DECIMAL(10,4),
    book_value_per_share DECIMAL(10,4),
    diluted_shares_outstanding BIGINT,
    
    -- Key ratios (calculated from latest data)
    roe DECIMAL(8,4),
    roa DECIMAL(8,4),
    debt_to_equity DECIMAL(8,4),
    current_ratio DECIMAL(8,4),
    gross_margin DECIMAL(8,4),
    operating_margin DECIMAL(8,4),
    net_margin DECIMAL(8,4),
    
    -- Revision tracking
    revision_count INTEGER DEFAULT 0,
    major_revision BOOLEAN DEFAULT FALSE, -- Significant change from first print
    revision_magnitude DECIMAL(8,4), -- Percentage change from first print
    revision_reason TEXT, -- Restatement, error correction, etc.
    
    -- Data source and quality
    data_source VARCHAR(50) NOT NULL,
    filing_type VARCHAR(20),
    amendment_flag BOOLEAN DEFAULT FALSE,
    data_quality_score DECIMAL(3,2) DEFAULT 1.0,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(symbol, report_date, period_type, fiscal_year)
);

-- Fundamentals Revision History Table
CREATE TABLE IF NOT EXISTS fundamentals_revision_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    period_type VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    revision_timestamp TIMESTAMPTZ NOT NULL,
    revision_number INTEGER NOT NULL,
    
    -- The specific field(s) that were revised
    revised_field VARCHAR(50) NOT NULL,
    old_value DECIMAL(20,4),
    new_value DECIMAL(20,4),
    change_percentage DECIMAL(8,4),
    
    -- Revision metadata
    revision_reason VARCHAR(100),
    source_document VARCHAR(100), -- SEC filing, press release, etc.
    materiality VARCHAR(20) CHECK (materiality IN ('minor', 'moderate', 'significant', 'material')),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Earnings Surprises (first-print vs consensus)
CREATE TABLE IF NOT EXISTS earnings_surprises (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    fiscal_year INTEGER NOT NULL,
    period_type VARCHAR(10) NOT NULL,
    
    -- Consensus estimates (before earnings)
    consensus_eps_estimate DECIMAL(10,4),
    consensus_revenue_estimate BIGINT,
    consensus_estimates_count INTEGER,
    estimate_standard_deviation DECIMAL(10,4),
    
    -- First-print actual results
    actual_eps_first_print DECIMAL(10,4),
    actual_revenue_first_print BIGINT,
    
    -- Latest revised actuals
    actual_eps_latest DECIMAL(10,4),
    actual_revenue_latest BIGINT,
    
    -- Surprise calculations
    eps_surprise_first_print DECIMAL(10,4), -- actual - consensus
    eps_surprise_percentage_first_print DECIMAL(8,4),
    revenue_surprise_first_print BIGINT,
    revenue_surprise_percentage_first_print DECIMAL(8,4),
    
    -- Surprise based on latest revisions
    eps_surprise_latest DECIMAL(10,4),
    eps_surprise_percentage_latest DECIMAL(8,4),
    revenue_surprise_latest BIGINT,
    revenue_surprise_percentage_latest DECIMAL(8,4),
    
    -- Market reaction
    announcement_time TIMESTAMPTZ,
    stock_price_before DECIMAL(10,4),
    stock_price_after_1day DECIMAL(10,4),
    stock_price_after_5day DECIMAL(10,4),
    volume_before_avg BIGINT, -- 20-day average before announcement
    volume_after_1day BIGINT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(symbol, report_date, period_type, fiscal_year)
);

-- Training Data Constraints Table
CREATE TABLE IF NOT EXISTS training_data_constraints (
    id SERIAL PRIMARY KEY,
    constraint_name VARCHAR(100) UNIQUE NOT NULL,
    constraint_type VARCHAR(50) NOT NULL CHECK (constraint_type IN (
        'fundamentals_lag', 'earnings_lag', 'revision_exclusion', 'data_quality_threshold'
    )),
    
    -- Lag constraints (prevents look-ahead)
    min_lag_days INTEGER, -- Minimum days between report date and usage
    min_lag_business_days INTEGER,
    filing_lag_required BOOLEAN DEFAULT TRUE, -- Must wait for actual filing
    
    -- Data quality constraints
    min_quality_score DECIMAL(3,2),
    max_revision_count INTEGER,
    exclude_amendments BOOLEAN DEFAULT FALSE,
    
    -- Scope and applicability
    applies_to_symbols TEXT[], -- Specific symbols or NULL for all
    applies_to_sectors TEXT[], -- Specific sectors or NULL for all
    start_date DATE, -- When constraint becomes effective
    end_date DATE, -- When constraint expires
    
    -- Rule description and rationale
    description TEXT NOT NULL,
    rationale TEXT,
    regulatory_requirement BOOLEAN DEFAULT FALSE,
    
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(100),
    approved_by VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fundamentals Data Access Log (audit trail)
CREATE TABLE IF NOT EXISTS fundamentals_access_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    query_type VARCHAR(50) NOT NULL CHECK (query_type IN (
        'training_data', 'backtest', 'live_trading', 'research', 'reporting'
    )),
    
    -- Query details
    symbols_requested TEXT[],
    date_range_start DATE,
    date_range_end DATE,
    data_version VARCHAR(20) CHECK (data_version IN ('first_print', 'latest', 'both')),
    
    -- Compliance checks
    constraints_applied TEXT[],
    violations_detected TEXT[],
    access_granted BOOLEAN,
    
    -- Query metadata
    query_sql TEXT,
    records_returned INTEGER,
    execution_time_ms INTEGER,
    
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundamentals_first_print_symbol_date 
ON fundamentals_first_print (symbol, report_date DESC, first_print_timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundamentals_first_print_filing_date 
ON fundamentals_first_print (filing_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundamentals_latest_symbol_date 
ON fundamentals_latest (symbol, report_date DESC, latest_revision_timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundamentals_revision_history_symbol_date 
ON fundamentals_revision_history (symbol, report_date DESC, revision_timestamp);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_earnings_surprises_symbol_date 
ON earnings_surprises (symbol, report_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_training_constraints_type_active 
ON training_data_constraints (constraint_type, is_active);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundamentals_access_log_timestamp 
ON fundamentals_access_log (timestamp DESC);

-- Functions for training data compliance

-- Function to get compliant fundamentals data for training
CREATE OR REPLACE FUNCTION get_training_fundamentals(
    p_symbols TEXT[],
    p_as_of_date DATE,
    p_lookback_days INTEGER DEFAULT 365
) RETURNS TABLE (
    symbol VARCHAR(20),
    report_date DATE,
    filing_date DATE,
    first_print_timestamp TIMESTAMPTZ,
    period_type VARCHAR(10),
    fiscal_year INTEGER,
    revenue BIGINT,
    net_income BIGINT,
    earnings_per_share DECIMAL(10,4),
    roe DECIMAL(8,4),
    data_quality_score DECIMAL(3,2)
) AS $$
DECLARE
    min_lag_days INTEGER;
    min_quality_score DECIMAL(3,2);
BEGIN
    -- Get active constraints
    SELECT COALESCE(MIN(c.min_lag_days), 45) INTO min_lag_days
    FROM training_data_constraints c
    WHERE c.constraint_type = 'fundamentals_lag' AND c.is_active = TRUE;
    
    SELECT COALESCE(MIN(c.min_quality_score), 0.8) INTO min_quality_score
    FROM training_data_constraints c
    WHERE c.constraint_type = 'data_quality_threshold' AND c.is_active = TRUE;
    
    -- Return compliant data only
    RETURN QUERY
    SELECT 
        fp.symbol,
        fp.report_date,
        fp.filing_date,
        fp.first_print_timestamp,
        fp.period_type,
        fp.fiscal_year,
        fp.revenue,
        fp.net_income,
        fp.earnings_per_share,
        fp.roe,
        fp.data_quality_score
    FROM fundamentals_first_print fp
    WHERE fp.symbol = ANY(p_symbols)
    AND fp.first_print_timestamp <= (p_as_of_date - INTERVAL '1 day' * min_lag_days)::TIMESTAMPTZ
    AND fp.report_date >= (p_as_of_date - INTERVAL '1 day' * p_lookback_days)
    AND fp.data_quality_score >= min_quality_score
    AND fp.amendment_flag = FALSE
    ORDER BY fp.symbol, fp.report_date DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to check if fundamentals usage complies with constraints
CREATE OR REPLACE FUNCTION check_fundamentals_compliance(
    p_symbol VARCHAR(20),
    p_report_date DATE,
    p_usage_date DATE,
    p_data_version VARCHAR(20) DEFAULT 'first_print'
) RETURNS TABLE (
    compliant BOOLEAN,
    violations TEXT[]
) AS $$
DECLARE
    violation_list TEXT[] := '{}';
    min_lag_days INTEGER;
    min_quality_score DECIMAL(3,2);
    data_quality DECIMAL(3,2);
    first_print_ts TIMESTAMPTZ;
    filing_ts TIMESTAMPTZ;
BEGIN
    -- Get constraints
    SELECT COALESCE(MIN(c.min_lag_days), 45) INTO min_lag_days
    FROM training_data_constraints c
    WHERE c.constraint_type = 'fundamentals_lag' AND c.is_active = TRUE;
    
    SELECT COALESCE(MIN(c.min_quality_score), 0.8) INTO min_quality_score
    FROM training_data_constraints c
    WHERE c.constraint_type = 'data_quality_threshold' AND c.is_active = TRUE;
    
    -- Get data details
    IF p_data_version = 'first_print' THEN
        SELECT fp.first_print_timestamp, fp.filing_date::TIMESTAMPTZ, fp.data_quality_score
        INTO first_print_ts, filing_ts, data_quality
        FROM fundamentals_first_print fp
        WHERE fp.symbol = p_symbol AND fp.report_date = p_report_date
        LIMIT 1;
    ELSE
        SELECT fl.first_print_timestamp, fl.filing_date::TIMESTAMPTZ, fl.data_quality_score
        INTO first_print_ts, filing_ts, data_quality
        FROM fundamentals_latest fl
        WHERE fl.symbol = p_symbol AND fl.report_date = p_report_date
        LIMIT 1;
    END IF;
    
    -- Check violations
    IF first_print_ts IS NULL THEN
        violation_list := array_append(violation_list, 'data_not_found');
    ELSE
        -- Check lag requirement
        IF p_usage_date::TIMESTAMPTZ < (first_print_ts + INTERVAL '1 day' * min_lag_days) THEN
            violation_list := array_append(violation_list, 'insufficient_lag');
        END IF;
        
        -- Check data quality
        IF data_quality < min_quality_score THEN
            violation_list := array_append(violation_list, 'low_quality_data');
        END IF;
    END IF;
    
    RETURN QUERY SELECT (array_length(violation_list, 1) IS NULL), violation_list;
END;
$$ LANGUAGE plpgsql;

-- Triggers for audit trail
CREATE OR REPLACE FUNCTION log_fundamentals_access()
RETURNS TRIGGER AS $$
BEGIN
    -- This would be called by application code to log access
    -- Implementation depends on how queries are structured
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Update triggers
CREATE TRIGGER update_fundamentals_first_print_updated_at 
BEFORE UPDATE ON fundamentals_first_print 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fundamentals_latest_updated_at 
BEFORE UPDATE ON fundamentals_latest 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_earnings_surprises_updated_at 
BEFORE UPDATE ON earnings_surprises 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_data_constraints_updated_at 
BEFORE UPDATE ON training_data_constraints 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default training data constraints
INSERT INTO training_data_constraints (
    constraint_name, constraint_type, min_lag_days, min_quality_score, 
    description, rationale, regulatory_requirement, created_by, approved_by
) VALUES 
(
    'standard_fundamentals_lag', 'fundamentals_lag', 45, NULL,
    'Minimum 45-day lag between report date and model training usage',
    'Ensures data was actually available to market participants at training time',
    FALSE, 'system', 'data_team'
),
(
    'high_quality_data_only', 'data_quality_threshold', NULL, 0.85,
    'Only use fundamentals data with quality score >= 0.85',
    'Prevents training on low-quality or incomplete data',
    FALSE, 'system', 'data_team'
),
(
    'no_amendment_filings', 'revision_exclusion', NULL, NULL,
    'Exclude amended filings from training data',
    'Amended filings may contain information not available at original filing time',
    TRUE, 'system', 'compliance_team'
)
ON CONFLICT (constraint_name) DO NOTHING;

-- Sample data for testing
INSERT INTO fundamentals_first_print (
    symbol, report_date, filing_date, first_print_timestamp, period_type, fiscal_year,
    revenue, net_income, earnings_per_share, roe, data_source, filing_type, data_quality_score
) VALUES 
(
    'SPY', '2024-03-31', '2024-05-10', '2024-05-10 16:30:00-04'::TIMESTAMPTZ, 'Q1', 2024,
    50000000, 5000000, 1.25, 15.5, 'sec_edgar', '10-Q', 0.95
),
(
    'SPY', '2024-06-30', '2024-08-09', '2024-08-09 16:30:00-04'::TIMESTAMPTZ, 'Q2', 2024,
    52000000, 5200000, 1.30, 16.0, 'sec_edgar', '10-Q', 0.92
)
ON CONFLICT DO NOTHING;

-- Comments
COMMENT ON TABLE fundamentals_first_print IS 'Original fundamentals data as first reported - used for training to prevent look-ahead bias';
COMMENT ON TABLE fundamentals_latest IS 'Latest revised fundamentals data - used for current analysis and reporting';
COMMENT ON TABLE fundamentals_revision_history IS 'Tracks all revisions to fundamentals data for audit and analysis';
COMMENT ON TABLE earnings_surprises IS 'Earnings surprise analysis comparing consensus vs actual (first-print and latest)';
COMMENT ON TABLE training_data_constraints IS 'Rules and constraints for accessing fundamentals data in training pipelines';
COMMENT ON TABLE fundamentals_access_log IS 'Audit log of all fundamentals data access for compliance tracking';

COMMENT ON FUNCTION get_training_fundamentals IS 'Returns fundamentals data that complies with training constraints to prevent look-ahead bias';
COMMENT ON FUNCTION check_fundamentals_compliance IS 'Validates that fundamentals data usage complies with established constraints';