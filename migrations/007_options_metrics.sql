-- Options Metrics Storage
-- Hypertable for ATM IV, skew, and implied move aggregates

CREATE TABLE IF NOT EXISTS options_metrics (
    symbol TEXT NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    expiry DATE,
    underlying_price NUMERIC,
    atm_strike NUMERIC,
    atm_iv NUMERIC,
    call_volume BIGINT,
    put_volume BIGINT,
    call_open_interest BIGINT,
    put_open_interest BIGINT,
    put_call_volume_ratio NUMERIC,
    put_call_oi_ratio NUMERIC,
    straddle_price NUMERIC,
    implied_move_pct NUMERIC,
    implied_move_upper NUMERIC,
    implied_move_lower NUMERIC,
    iv_25d_call NUMERIC,
    iv_25d_put NUMERIC,
    iv_skew_25d NUMERIC,
    iv_skew_25d_pct NUMERIC,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, as_of, expiry)
);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'options_metrics',
            'as_of',
            chunk_time_interval => INTERVAL '30 days',
            if_not_exists => TRUE
        );

        PERFORM add_retention_policy(
            'options_metrics',
            INTERVAL '2 years',
            if_not_exists => TRUE
        );

        BEGIN
            PERFORM add_compression_policy(
                'options_metrics',
                INTERVAL '14 days',
                if_not_exists => TRUE
            );
        EXCEPTION WHEN others THEN
            RAISE NOTICE 'Compression policy unavailable for options_metrics';
        END;
    ELSE
        RAISE NOTICE 'TimescaleDB extension not available, options_metrics left as standard table';
    END IF;
END $$ LANGUAGE plpgsql;

CREATE INDEX IF NOT EXISTS idx_options_metrics_symbol_ts
    ON options_metrics (symbol, as_of DESC);

CREATE INDEX IF NOT EXISTS idx_options_metrics_expiry
    ON options_metrics (expiry, symbol);

