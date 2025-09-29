-- TimescaleDB Optimization Migration
-- Converts candles tables to hypertables and adds continuous aggregates
-- Run after enabling TimescaleDB extension

-- Enable TimescaleDB extension (requires superuser privileges)
-- This should be run manually: CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert existing candles table to hypertable
-- First, need to check if data exists and handle accordingly
DO $$
BEGIN
    -- Check if timescaledb extension is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Convert candles table to hypertable (partitioned by time)
        -- Chunk interval: 7 days for daily data
        PERFORM create_hypertable(
            'candles', 
            'ts',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE
        );
        
        -- Convert intraday candles to hypertable  
        -- Chunk interval: 1 day for minute data
        PERFORM create_hypertable(
            'candles_intraday',
            'ts', 
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'TimescaleDB hypertables created successfully';
    ELSE
        RAISE NOTICE 'TimescaleDB extension not available, skipping hypertable creation';
    END IF;
END $$;

-- Create continuous aggregates for common time windows
-- Daily aggregates from minute data
DROP MATERIALIZED VIEW IF EXISTS candles_daily_agg CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_daily_agg
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 day', ts) AS day,
    first(open, ts) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, ts) AS close,
    sum(volume) AS volume,
    count(*) AS bar_count
FROM candles_intraday
GROUP BY symbol, day
WITH NO DATA;

-- Hourly aggregates from minute data  
DROP MATERIALIZED VIEW IF EXISTS candles_hourly_agg CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_hourly_agg
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', ts) AS hour,
    first(open, ts) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, ts) AS close,
    sum(volume) AS volume,
    count(*) AS bar_count
FROM candles_intraday
GROUP BY symbol, hour
WITH NO DATA;

-- Set up refresh policies for continuous aggregates
-- Refresh every 30 minutes for hourly data
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM add_continuous_aggregate_policy(
            'candles_hourly_agg',
            start_offset => INTERVAL '2 hours',
            end_offset => INTERVAL '30 minutes',
            schedule_interval => INTERVAL '30 minutes',
            if_not_exists => TRUE
        );
        
        -- Refresh daily at 2 AM
        PERFORM add_continuous_aggregate_policy(
            'candles_daily_agg', 
            start_offset => INTERVAL '2 days',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'Continuous aggregate policies created';
    END IF;
END $$;

-- Set up data retention policies
-- Keep minute data for 3 months, daily data for 5 years
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Retain intraday data for 3 months
        PERFORM add_retention_policy(
            'candles_intraday',
            INTERVAL '3 months',
            if_not_exists => TRUE
        );
        
        -- Retain daily data for 5 years  
        PERFORM add_retention_policy(
            'candles',
            INTERVAL '5 years',
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'Data retention policies created';
    END IF;
END $$;

-- Create optimized indexes for time-series queries
-- Drop existing indexes first to avoid conflicts
DROP INDEX IF EXISTS idx_candles_symbol_ts;
DROP INDEX IF EXISTS idx_candles_ts;
DROP INDEX IF EXISTS idx_candles_intraday_symbol_ts;

-- Recreate with TimescaleDB optimizations
CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts_btree 
    ON candles (symbol, ts DESC);

CREATE INDEX IF NOT EXISTS idx_candles_intraday_symbol_ts_btree
    ON candles_intraday (symbol, ts DESC);

-- Create BRIN indexes for time column (better for time-series)
CREATE INDEX IF NOT EXISTS idx_candles_ts_brin 
    ON candles USING BRIN (ts);

CREATE INDEX IF NOT EXISTS idx_candles_intraday_ts_brin
    ON candles_intraday USING BRIN (ts);

-- Create indexes for volume-based queries
CREATE INDEX IF NOT EXISTS idx_candles_volume 
    ON candles (symbol, ts DESC) WHERE volume > 0;

CREATE INDEX IF NOT EXISTS idx_candles_intraday_volume
    ON candles_intraday (symbol, ts DESC) WHERE volume > 0;

-- Create compression policy for older data (requires TimescaleDB 2.0+)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Compress data older than 7 days for intraday
        PERFORM add_compression_policy(
            'candles_intraday',
            INTERVAL '7 days',
            if_not_exists => TRUE
        );
        
        -- Compress data older than 30 days for daily
        PERFORM add_compression_policy(
            'candles', 
            INTERVAL '30 days',
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'Compression policies created';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Compression policies not available in this TimescaleDB version';
END $$;

-- Create helper functions for time-series queries
CREATE OR REPLACE FUNCTION get_latest_price(p_symbol TEXT)
RETURNS TABLE(
    symbol TEXT,
    price NUMERIC,
    timestamp TIMESTAMPTZ
) 
LANGUAGE SQL STABLE AS $$
    SELECT 
        c.symbol,
        c.close,
        c.ts
    FROM candles_intraday c
    WHERE c.symbol = p_symbol
    ORDER BY c.ts DESC
    LIMIT 1;
$$;

CREATE OR REPLACE FUNCTION get_ohlcv_range(
    p_symbol TEXT,
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ,
    p_interval TEXT DEFAULT '1h'
)
RETURNS TABLE(
    symbol TEXT,
    ts TIMESTAMPTZ,
    open NUMERIC,
    high NUMERIC, 
    low NUMERIC,
    close NUMERIC,
    volume BIGINT
)
LANGUAGE SQL STABLE AS $$
    SELECT 
        ci.symbol,
        time_bucket(p_interval::INTERVAL, ci.ts) as ts,
        first(ci.open, ci.ts) as open,
        max(ci.high) as high,
        min(ci.low) as low,
        last(ci.close, ci.ts) as close,
        sum(ci.volume) as volume
    FROM candles_intraday ci
    WHERE ci.symbol = p_symbol
        AND ci.ts >= p_start_time
        AND ci.ts <= p_end_time
    GROUP BY ci.symbol, time_bucket(p_interval::INTERVAL, ci.ts)
    ORDER BY ts;
$$;

-- Statistics and monitoring views
CREATE OR REPLACE VIEW timeseries_stats AS
SELECT
    'candles' as table_name,
    count(*) as row_count,
    count(DISTINCT symbol) as symbol_count,
    min(ts) as earliest_data,
    max(ts) as latest_data,
    pg_size_pretty(pg_total_relation_size('candles')) as table_size
FROM candles
UNION ALL
SELECT
    'candles_intraday' as table_name,
    count(*) as row_count,
    count(DISTINCT symbol) as symbol_count, 
    min(ts) as earliest_data,
    max(ts) as latest_data,
    pg_size_pretty(pg_total_relation_size('candles_intraday')) as table_size
FROM candles_intraday;

-- Create trigger to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Comment with usage instructions
COMMENT ON SCHEMA public IS 'TimescaleDB optimized schema for trading platform. 

Usage:
- Use candles_intraday for minute-level data
- Use candles for daily/higher timeframes  
- Use continuous aggregates for fast querying of common timeframes
- Check timeseries_stats view for monitoring

Example queries:
SELECT * FROM get_latest_price(''AAPL'');
SELECT * FROM get_ohlcv_range(''AAPL'', now() - interval ''1 day'', now(), ''1h'');
SELECT * FROM timeseries_stats;
';