-- Migration: Create first-print vs latest fundamentals tables
-- Purpose: Separate first-print fundamentals (for training) from latest (for live serving)
-- This ensures Point-in-Time compliance by preventing look-ahead bias

-- Enable TimescaleDB extension if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- First-print fundamentals table (for training only)
CREATE TABLE IF NOT EXISTS fundamentals_first_print (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    period_end_date DATE NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter INTEGER NOT NULL,
    
    -- First print timestamp (when data was first reported)
    first_print_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Core financial metrics (first reported values)
    revenue DECIMAL(20,2),
    net_income DECIMAL(20,2),
    total_assets DECIMAL(20,2),
    total_equity DECIMAL(20,2),
    total_debt DECIMAL(20,2),
    cash_and_equivalents DECIMAL(20,2),
    operating_cash_flow DECIMAL(20,2),
    free_cash_flow DECIMAL(20,2),
    
    -- Per-share metrics
    earnings_per_share DECIMAL(10,4),
    book_value_per_share DECIMAL(10,4),
    shares_outstanding BIGINT,
    
    -- Derived ratios (calculated from first-print values)
    debt_to_equity_ratio DECIMAL(8,4),
    current_ratio DECIMAL(8,4),
    return_on_equity DECIMAL(8,4),
    return_on_assets DECIMAL(8,4),
    gross_margin DECIMAL(8,4),
    operating_margin DECIMAL(8,4),
    net_margin DECIMAL(8,4),
    
    -- Data lineage and quality
    data_source VARCHAR(50) NOT NULL,
    revision_number INTEGER DEFAULT 0,
    is_preliminary BOOLEAN DEFAULT FALSE,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure only one first-print record per symbol/period
    UNIQUE(symbol, report_date, period_end_date)
);

-- Latest fundamentals table (for live serving)
CREATE TABLE IF NOT EXISTS fundamentals_latest (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    period_end_date DATE NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter INTEGER NOT NULL,
    
    -- Latest update timestamp
    last_updated_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Core financial metrics (latest revised values)
    revenue DECIMAL(20,2),
    net_income DECIMAL(20,2),
    total_assets DECIMAL(20,2),
    total_equity DECIMAL(20,2),
    total_debt DECIMAL(20,2),
    cash_and_equivalents DECIMAL(20,2),
    operating_cash_flow DECIMAL(20,2),
    free_cash_flow DECIMAL(20,2),
    
    -- Per-share metrics
    earnings_per_share DECIMAL(10,4),
    book_value_per_share DECIMAL(10,4),
    shares_outstanding BIGINT,
    
    -- Derived ratios (calculated from latest values)
    debt_to_equity_ratio DECIMAL(8,4),
    current_ratio DECIMAL(8,4),
    return_on_equity DECIMAL(8,4),
    return_on_assets DECIMAL(8,4),
    gross_margin DECIMAL(8,4),
    operating_margin DECIMAL(8,4),
    net_margin DECIMAL(8,4),
    
    -- Data lineage and quality
    data_source VARCHAR(50) NOT NULL,
    revision_number INTEGER DEFAULT 0,
    total_revisions INTEGER DEFAULT 0,
    is_preliminary BOOLEAN DEFAULT FALSE,
    
    -- Reference to first print
    first_print_id INTEGER REFERENCES fundamentals_first_print(id),
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure only one latest record per symbol/period
    UNIQUE(symbol, report_date, period_end_date)
);

-- Convert to TimescaleDB hypertables for better performance
SELECT create_hypertable('fundamentals_first_print', 'first_print_timestamp', 
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE);

SELECT create_hypertable('fundamentals_latest', 'last_updated_timestamp', 
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_fundamentals_first_print_symbol_date 
ON fundamentals_first_print(symbol, report_date, first_print_timestamp);

CREATE INDEX IF NOT EXISTS idx_fundamentals_first_print_timestamp 
ON fundamentals_first_print(first_print_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_fundamentals_first_print_period 
ON fundamentals_first_print(symbol, fiscal_year, fiscal_quarter);

CREATE INDEX IF NOT EXISTS idx_fundamentals_latest_symbol_date 
ON fundamentals_latest(symbol, report_date, last_updated_timestamp);

CREATE INDEX IF NOT EXISTS idx_fundamentals_latest_timestamp 
ON fundamentals_latest(last_updated_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_fundamentals_latest_period 
ON fundamentals_latest(symbol, fiscal_year, fiscal_quarter);

-- Index for fast first_print_id lookups
CREATE INDEX IF NOT EXISTS idx_fundamentals_latest_first_print 
ON fundamentals_latest(first_print_id);

-- Revision tracking table for audit trail
CREATE TABLE IF NOT EXISTS fundamentals_revisions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    period_end_date DATE NOT NULL,
    
    -- Revision metadata
    revision_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    revision_number INTEGER NOT NULL,
    revision_type VARCHAR(20) NOT NULL, -- 'first_print', 'revision', 'restatement'
    
    -- What changed
    changed_fields JSONB,
    previous_values JSONB,
    new_values JSONB,
    
    -- Data source and reason
    data_source VARCHAR(50) NOT NULL,
    revision_reason TEXT,
    
    -- References
    first_print_id INTEGER REFERENCES fundamentals_first_print(id),
    latest_id INTEGER REFERENCES fundamentals_latest(id),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Convert revisions to hypertable
SELECT create_hypertable('fundamentals_revisions', 'revision_timestamp', 
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE);

-- Index for revision tracking
CREATE INDEX IF NOT EXISTS idx_fundamentals_revisions_symbol_date 
ON fundamentals_revisions(symbol, report_date, revision_timestamp);

CREATE INDEX IF NOT EXISTS idx_fundamentals_revisions_type 
ON fundamentals_revisions(revision_type, revision_timestamp);

-- Point-in-Time lookup function
CREATE OR REPLACE FUNCTION get_fundamentals_pit(
    p_symbol VARCHAR(20),
    p_as_of_date TIMESTAMP WITH TIME ZONE,
    p_lookback_quarters INTEGER DEFAULT 8
) RETURNS TABLE (
    symbol VARCHAR(20),
    report_date DATE,
    period_end_date DATE,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    revenue DECIMAL(20,2),
    net_income DECIMAL(20,2),
    earnings_per_share DECIMAL(10,4),
    first_print_timestamp TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fp.symbol,
        fp.report_date,
        fp.period_end_date,
        fp.fiscal_year,
        fp.fiscal_quarter,
        fp.revenue,
        fp.net_income,
        fp.earnings_per_share,
        fp.first_print_timestamp
    FROM fundamentals_first_print fp
    WHERE fp.symbol = p_symbol
    AND fp.first_print_timestamp <= p_as_of_date
    ORDER BY fp.period_end_date DESC
    LIMIT p_lookback_quarters;
END;
$$ LANGUAGE plpgsql;

-- Data validation function
CREATE OR REPLACE FUNCTION validate_fundamentals_pit_compliance() 
RETURNS TABLE (
    violation_type VARCHAR(50),
    symbol VARCHAR(20),
    report_date DATE,
    description TEXT,
    severity VARCHAR(20)
) AS $$
BEGIN
    -- Check 1: Ensure no first-print data has future timestamps
    RETURN QUERY
    SELECT 
        'future_first_print'::VARCHAR(50),
        fp.symbol,
        fp.report_date,
        'First print timestamp is after current time'::TEXT,
        'critical'::VARCHAR(20)
    FROM fundamentals_first_print fp
    WHERE fp.first_print_timestamp > NOW();
    
    -- Check 2: Ensure first-print timestamp is after period end
    RETURN QUERY
    SELECT 
        'invalid_print_timing'::VARCHAR(50),
        fp.symbol,
        fp.report_date,
        'First print timestamp is before period end date'::TEXT,
        'high'::VARCHAR(20)
    FROM fundamentals_first_print fp
    WHERE fp.first_print_timestamp::DATE < fp.period_end_date + INTERVAL '45 days';
    
    -- Check 3: Check for missing first-print records that have latest
    RETURN QUERY
    SELECT 
        'missing_first_print'::VARCHAR(50),
        fl.symbol,
        fl.report_date,
        'Latest record exists without corresponding first-print'::TEXT,
        'high'::VARCHAR(20)
    FROM fundamentals_latest fl
    LEFT JOIN fundamentals_first_print fp ON fp.id = fl.first_print_id
    WHERE fp.id IS NULL;
    
    -- Check 4: Validate revision sequence
    RETURN QUERY
    SELECT 
        'invalid_revision_sequence'::VARCHAR(50),
        fl.symbol,
        fl.report_date,
        'Revision number inconsistency detected'::TEXT,
        'medium'::VARCHAR(20)
    FROM fundamentals_latest fl
    WHERE fl.revision_number > fl.total_revisions;
    
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically create revision records
CREATE OR REPLACE FUNCTION track_fundamentals_revisions()
RETURNS TRIGGER AS $$
DECLARE
    changed_fields JSONB := '{}';
    prev_values JSONB := '{}';
    new_values JSONB := '{}';
    field_name TEXT;
BEGIN
    -- Only track changes on updates
    IF TG_OP = 'UPDATE' THEN
        -- Check each numeric field for changes
        FOR field_name IN SELECT unnest(ARRAY['revenue', 'net_income', 'total_assets', 
                                              'total_equity', 'earnings_per_share']) LOOP
            IF (to_jsonb(OLD) ->> field_name)::DECIMAL != (to_jsonb(NEW) ->> field_name)::DECIMAL THEN
                changed_fields := changed_fields || jsonb_build_object(field_name, true);
                prev_values := prev_values || jsonb_build_object(field_name, to_jsonb(OLD) ->> field_name);
                new_values := new_values || jsonb_build_object(field_name, to_jsonb(NEW) ->> field_name);
            END IF;
        END LOOP;
        
        -- Insert revision record if any fields changed
        IF jsonb_object_keys(changed_fields) IS NOT NULL THEN
            INSERT INTO fundamentals_revisions (
                symbol, report_date, period_end_date, revision_timestamp,
                revision_number, revision_type, changed_fields, previous_values,
                new_values, data_source, latest_id, first_print_id
            ) VALUES (
                NEW.symbol, NEW.report_date, NEW.period_end_date, NOW(),
                NEW.revision_number, 'revision', changed_fields, prev_values,
                new_values, NEW.data_source, NEW.id, NEW.first_print_id
            );
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for revision tracking
DROP TRIGGER IF EXISTS trigger_track_fundamentals_revisions ON fundamentals_latest;
CREATE TRIGGER trigger_track_fundamentals_revisions
    AFTER UPDATE ON fundamentals_latest
    FOR EACH ROW
    EXECUTE FUNCTION track_fundamentals_revisions();

-- Create compression policies for older data
SELECT add_compression_policy('fundamentals_first_print', INTERVAL '12 months');
SELECT add_compression_policy('fundamentals_latest', INTERVAL '12 months');
SELECT add_compression_policy('fundamentals_revisions', INTERVAL '6 months');

-- Create retention policies
SELECT add_retention_policy('fundamentals_revisions', INTERVAL '7 years');

-- Grant permissions
GRANT SELECT ON fundamentals_first_print TO PUBLIC;
GRANT SELECT ON fundamentals_latest TO PUBLIC;
GRANT SELECT ON fundamentals_revisions TO PUBLIC;

-- Comments for documentation
COMMENT ON TABLE fundamentals_first_print IS 'First-print fundamentals data for training (Point-in-Time compliant)';
COMMENT ON TABLE fundamentals_latest IS 'Latest fundamentals data for live serving (may include revisions)';
COMMENT ON TABLE fundamentals_revisions IS 'Audit trail of fundamentals data revisions';

COMMENT ON FUNCTION get_fundamentals_pit IS 'Point-in-Time lookup of fundamentals data as known at specific date';
COMMENT ON FUNCTION validate_fundamentals_pit_compliance IS 'Validates Point-in-Time compliance of fundamentals data';