-- Fundamentals and Earnings Data Storage Tables
-- Stores SEC filings, earnings data, and financial analysis results

-- Create earnings events table
CREATE TABLE IF NOT EXISTS earnings_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    
    -- Event details
    report_date DATE NOT NULL,
    period_ending DATE NOT NULL,
    period_type VARCHAR(10) NOT NULL CHECK (period_type IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY')),
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter INTEGER CHECK (fiscal_quarter BETWEEN 1 AND 4),
    
    -- Estimates and actuals
    estimated_eps DECIMAL(10,4),
    actual_eps DECIMAL(10,4),
    estimated_revenue BIGINT,
    actual_revenue BIGINT,
    surprise_percent DECIMAL(8,4),
    
    -- Event metadata
    announcement_time VARCHAR(10) CHECK (announcement_time IN ('BMO', 'AMC', 'TAS')),
    status VARCHAR(20) DEFAULT 'upcoming' CHECK (status IN ('upcoming', 'reported', 'estimated')),
    guidance_updated BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Unique constraint per symbol/period
    UNIQUE(symbol, fiscal_year, fiscal_quarter, period_type)
);

-- Create quarterly performance table
CREATE TABLE IF NOT EXISTS quarterly_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    quarter VARCHAR(5) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    period_ending DATE NOT NULL,
    
    -- Financial metrics
    revenue BIGINT,
    revenue_growth_yoy DECIMAL(8,4),
    revenue_growth_qoq DECIMAL(8,4),
    net_income BIGINT,
    earnings_per_share DECIMAL(10,4),
    eps_growth_yoy DECIMAL(8,4),
    
    -- Margins
    gross_margin DECIMAL(8,4),
    operating_margin DECIMAL(8,4),
    net_margin DECIMAL(8,4),
    
    -- Profitability ratios
    roe DECIMAL(8,4),
    roa DECIMAL(8,4),
    roic DECIMAL(8,4),
    
    -- Cash flow
    free_cash_flow BIGINT,
    operating_cash_flow BIGINT,
    capex BIGINT,
    
    -- Guidance
    guidance_revenue_low BIGINT,
    guidance_revenue_high BIGINT,
    guidance_eps_low DECIMAL(10,4),
    guidance_eps_high DECIMAL(10,4),
    
    -- Balance sheet
    total_assets BIGINT,
    total_debt BIGINT,
    cash_and_equivalents BIGINT,
    shareholders_equity BIGINT,
    
    -- Timestamps
    report_date DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Unique constraint
    UNIQUE(symbol, fiscal_year, quarter)
);

-- Create SEC filings table
CREATE TABLE IF NOT EXISTS sec_filings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    cik VARCHAR(20),
    
    -- Filing details
    filing_type VARCHAR(10) NOT NULL CHECK (filing_type IN ('10-K', '10-Q', '8-K', 'DEF 14A', '20-F')),
    filing_date DATE NOT NULL,
    period_end_date DATE,
    fiscal_year INTEGER,
    fiscal_period VARCHAR(10),
    
    -- Filing content
    filing_url TEXT NOT NULL,
    raw_content TEXT,
    processed_data JSONB,
    
    -- Extracted data
    income_statement JSONB,
    balance_sheet JSONB,
    cash_flow_statement JSONB,
    
    -- Text analysis
    risk_factors TEXT[],
    management_discussion TEXT,
    business_description TEXT,
    key_metrics JSONB,
    
    -- Processing status
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    processing_error TEXT,
    
    -- Metadata
    document_count INTEGER DEFAULT 0,
    size_kb INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Unique constraint per filing
    UNIQUE(symbol, filing_type, filing_date)
);

-- Create earnings trends analysis table
CREATE TABLE IF NOT EXISTS earnings_trends (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    analysis_date DATE NOT NULL DEFAULT CURRENT_DATE,
    
    -- Revenue trends
    revenue_trend_direction VARCHAR(20),
    revenue_avg_growth DECIMAL(8,4),
    revenue_acceleration DECIMAL(8,4),
    
    -- EPS trends
    eps_trend_direction VARCHAR(20),
    eps_avg_growth DECIMAL(8,4),
    eps_acceleration DECIMAL(8,4),
    
    -- Margin trends
    margin_trend_direction VARCHAR(20),
    current_margin DECIMAL(8,4),
    avg_margin DECIMAL(8,4),
    
    -- Quality metrics
    earnings_surprise_rate DECIMAL(6,4),
    consistency_score DECIMAL(6,2),
    growth_quality VARCHAR(20),
    growth_quality_score INTEGER,
    
    -- Guidance analysis
    guidance_accuracy JSONB,
    
    -- Analysis period
    quarters_analyzed INTEGER DEFAULT 8,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- One analysis per symbol per day
    UNIQUE(symbol, analysis_date)
);

-- Create sector earnings analysis table
CREATE TABLE IF NOT EXISTS sector_earnings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sector VARCHAR(100) NOT NULL,
    period VARCHAR(20) NOT NULL,
    analysis_date DATE NOT NULL DEFAULT CURRENT_DATE,
    
    -- Sector metrics
    companies_count INTEGER DEFAULT 0,
    reporting_complete INTEGER DEFAULT 0,
    beat_estimates INTEGER DEFAULT 0,
    missed_estimates INTEGER DEFAULT 0,
    avg_surprise DECIMAL(8,4),
    revenue_growth_avg DECIMAL(8,4),
    margin_expansion INTEGER DEFAULT 0,
    guidance_raises INTEGER DEFAULT 0,
    guidance_lowers INTEGER DEFAULT 0,
    
    -- Company details
    company_details JSONB,
    
    -- Market impact
    total_market_cap BIGINT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- One analysis per sector per period per day
    UNIQUE(sector, period, analysis_date)
);

-- Create earnings alerts table
CREATE TABLE IF NOT EXISTS earnings_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    user_id UUID, -- For future user integration
    
    -- Alert settings
    days_before_earnings INTEGER DEFAULT 7 CHECK (days_before_earnings BETWEEN 1 AND 30),
    surprise_threshold DECIMAL(6,2) DEFAULT 5.0 CHECK (surprise_threshold >= 0),
    guidance_changes BOOLEAN DEFAULT true,
    revenue_miss BOOLEAN DEFAULT true,
    margin_compression_threshold DECIMAL(6,2) DEFAULT 2.0,
    
    -- Alert status
    is_active BOOLEAN DEFAULT true,
    last_triggered_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create earnings calendar view for easy querying
CREATE TABLE IF NOT EXISTS earnings_calendar (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calendar_date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    market_cap_billions DECIMAL(10,2),
    period_type VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    announcement_time VARCHAR(10),
    estimated_eps DECIMAL(10,4),
    estimated_revenue BIGINT,
    is_high_impact BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Unique per date/symbol
    UNIQUE(calendar_date, symbol)
);

-- Create indexes for performance
-- Earnings events indexes
CREATE INDEX IF NOT EXISTS idx_earnings_events_symbol_date 
    ON earnings_events (symbol, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_earnings_events_date 
    ON earnings_events (report_date DESC);

CREATE INDEX IF NOT EXISTS idx_earnings_events_status 
    ON earnings_events (status, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_earnings_events_fiscal 
    ON earnings_events (fiscal_year, fiscal_quarter);

-- Quarterly performance indexes
CREATE INDEX IF NOT EXISTS idx_quarterly_perf_symbol_date 
    ON quarterly_performance (symbol, report_date DESC);

CREATE INDEX IF NOT EXISTS idx_quarterly_perf_fiscal 
    ON quarterly_performance (symbol, fiscal_year DESC, quarter);

CREATE INDEX IF NOT EXISTS idx_quarterly_perf_metrics 
    ON quarterly_performance (revenue_growth_yoy, eps_growth_yoy) 
    WHERE revenue_growth_yoy IS NOT NULL AND eps_growth_yoy IS NOT NULL;

-- SEC filings indexes
CREATE INDEX IF NOT EXISTS idx_filings_symbol_type_date 
    ON sec_filings (symbol, filing_type, filing_date DESC);

CREATE INDEX IF NOT EXISTS idx_filings_processing_status 
    ON sec_filings (processing_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_filings_fiscal 
    ON sec_filings (fiscal_year DESC, fiscal_period);

-- Earnings trends indexes
CREATE INDEX IF NOT EXISTS idx_earnings_trends_symbol_date 
    ON earnings_trends (symbol, analysis_date DESC);

CREATE INDEX IF NOT EXISTS idx_earnings_trends_quality 
    ON earnings_trends (growth_quality, consistency_score DESC);

-- Sector earnings indexes
CREATE INDEX IF NOT EXISTS idx_sector_earnings_sector_date 
    ON sector_earnings (sector, analysis_date DESC);

CREATE INDEX IF NOT EXISTS idx_sector_earnings_period 
    ON sector_earnings (period, analysis_date DESC);

-- Earnings alerts indexes
CREATE INDEX IF NOT EXISTS idx_earnings_alerts_symbol_active 
    ON earnings_alerts (symbol, is_active) WHERE is_active = true;

-- Earnings calendar indexes
CREATE INDEX IF NOT EXISTS idx_earnings_calendar_date 
    ON earnings_calendar (calendar_date DESC);

CREATE INDEX IF NOT EXISTS idx_earnings_calendar_symbol_date 
    ON earnings_calendar (symbol, calendar_date DESC);

CREATE INDEX IF NOT EXISTS idx_earnings_calendar_high_impact 
    ON earnings_calendar (calendar_date DESC, is_high_impact) WHERE is_high_impact = true;

-- Create trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_fundamentals_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to relevant tables
CREATE TRIGGER update_earnings_events_updated_at 
    BEFORE UPDATE ON earnings_events 
    FOR EACH ROW 
    EXECUTE PROCEDURE update_fundamentals_updated_at_column();

CREATE TRIGGER update_quarterly_performance_updated_at 
    BEFORE UPDATE ON quarterly_performance 
    FOR EACH ROW 
    EXECUTE PROCEDURE update_fundamentals_updated_at_column();

CREATE TRIGGER update_sec_filings_updated_at 
    BEFORE UPDATE ON sec_filings 
    FOR EACH ROW 
    EXECUTE PROCEDURE update_fundamentals_updated_at_column();

CREATE TRIGGER update_earnings_alerts_updated_at 
    BEFORE UPDATE ON earnings_alerts 
    FOR EACH ROW 
    EXECUTE PROCEDURE update_fundamentals_updated_at_column();

-- Convert tables to TimescaleDB hypertables if available
DO $$
BEGIN
    -- Check if TimescaleDB extension exists
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Convert earnings_events to hypertable
        PERFORM create_hypertable(
            'earnings_events', 
            'report_date',
            chunk_time_interval => INTERVAL '1 month',
            if_not_exists => TRUE
        );
        
        -- Convert quarterly_performance to hypertable
        PERFORM create_hypertable(
            'quarterly_performance',
            'report_date',
            chunk_time_interval => INTERVAL '3 months',
            if_not_exists => TRUE
        );
        
        -- Convert sec_filings to hypertable
        PERFORM create_hypertable(
            'sec_filings',
            'filing_date',
            chunk_time_interval => INTERVAL '1 month',
            if_not_exists => TRUE
        );
        
        -- Convert earnings_calendar to hypertable
        PERFORM create_hypertable(
            'earnings_calendar',
            'calendar_date',
            chunk_time_interval => INTERVAL '1 month',
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'TimescaleDB hypertables created for fundamentals service';
        
        -- Set up retention policies
        PERFORM add_retention_policy(
            'earnings_events',
            INTERVAL '10 years',
            if_not_exists => TRUE
        );
        
        PERFORM add_retention_policy(
            'quarterly_performance',
            INTERVAL '15 years',
            if_not_exists => TRUE
        );
        
        PERFORM add_retention_policy(
            'sec_filings',
            INTERVAL '7 years',
            if_not_exists => TRUE
        );
        
        PERFORM add_retention_policy(
            'earnings_calendar',
            INTERVAL '5 years',
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'TimescaleDB retention policies set for fundamentals service';
        
    ELSE
        RAISE NOTICE 'TimescaleDB extension not available, using regular tables';
    END IF;
END $$;

-- Create useful views for analysis
CREATE OR REPLACE VIEW earnings_summary_view AS
SELECT 
    symbol,
    fiscal_year,
    fiscal_quarter,
    actual_eps,
    actual_revenue,
    surprise_percent,
    report_date,
    CASE 
        WHEN surprise_percent > 0 THEN 'beat'
        WHEN surprise_percent < 0 THEN 'miss'
        ELSE 'inline'
    END as surprise_direction
FROM earnings_events 
WHERE status = 'reported' 
    AND actual_eps IS NOT NULL 
    AND report_date >= CURRENT_DATE - INTERVAL '5 years'
ORDER BY symbol, fiscal_year DESC, fiscal_quarter DESC;

-- Create view for quarterly trends
CREATE OR REPLACE VIEW quarterly_trends_view AS
SELECT 
    symbol,
    quarter,
    fiscal_year,
    revenue_growth_yoy,
    eps_growth_yoy,
    net_margin,
    roe,
    LAG(revenue_growth_yoy) OVER (PARTITION BY symbol ORDER BY fiscal_year, quarter) as prev_revenue_growth,
    LAG(eps_growth_yoy) OVER (PARTITION BY symbol ORDER BY fiscal_year, quarter) as prev_eps_growth,
    LAG(net_margin) OVER (PARTITION BY symbol ORDER BY fiscal_year, quarter) as prev_net_margin
FROM quarterly_performance 
WHERE report_date >= CURRENT_DATE - INTERVAL '5 years'
ORDER BY symbol, fiscal_year DESC, quarter DESC;

-- Create view for upcoming earnings
CREATE OR REPLACE VIEW upcoming_earnings_view AS
SELECT 
    symbol,
    company_name,
    calendar_date as report_date,
    period_type,
    fiscal_year,
    announcement_time,
    estimated_eps,
    estimated_revenue,
    market_cap_billions,
    is_high_impact,
    calendar_date - CURRENT_DATE as days_until
FROM earnings_calendar 
WHERE calendar_date >= CURRENT_DATE 
    AND calendar_date <= CURRENT_DATE + INTERVAL '90 days'
ORDER BY calendar_date ASC, market_cap_billions DESC;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON earnings_events TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON quarterly_performance TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON sec_filings TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON earnings_trends TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON sector_earnings TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON earnings_alerts TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON earnings_calendar TO trading_user;
GRANT SELECT ON earnings_summary_view TO trading_user;
GRANT SELECT ON quarterly_trends_view TO trading_user;
GRANT SELECT ON upcoming_earnings_view TO trading_user;

-- Add table comments
COMMENT ON TABLE earnings_events IS 'Earnings events with estimates, actuals, and surprises';
COMMENT ON TABLE quarterly_performance IS 'Quarterly financial performance metrics and ratios';
COMMENT ON TABLE sec_filings IS 'SEC filing documents with parsed financial data';
COMMENT ON TABLE earnings_trends IS 'Earnings trend analysis and quality scoring';
COMMENT ON TABLE sector_earnings IS 'Sector-wide earnings performance analysis';
COMMENT ON TABLE earnings_alerts IS 'User-configured earnings monitoring alerts';
COMMENT ON TABLE earnings_calendar IS 'Upcoming earnings calendar with market impact';
COMMENT ON VIEW earnings_summary_view IS 'Summary of earnings surprises and results';
COMMENT ON VIEW quarterly_trends_view IS 'Quarterly performance trends with growth rates';
COMMENT ON VIEW upcoming_earnings_view IS 'Upcoming earnings events with timing';