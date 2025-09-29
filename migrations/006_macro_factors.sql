-- Macro Factors Storage
-- Creates hypertable for macro and cross-asset factor time series

CREATE TABLE IF NOT EXISTS macro_factors (
    factor_key TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    value NUMERIC NOT NULL,
    source TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (factor_key, ts)
);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'macro_factors',
            'ts',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE
        );

        -- Retain macro readings for ten years by default
        PERFORM add_retention_policy(
            'macro_factors',
            INTERVAL '10 years',
            if_not_exists => TRUE
        );

        BEGIN
            PERFORM add_compression_policy(
                'macro_factors',
                INTERVAL '30 days',
                if_not_exists => TRUE
            );
        EXCEPTION WHEN others THEN
            RAISE NOTICE 'Compression policy unavailable for macro_factors';
        END;
    ELSE
        RAISE NOTICE 'TimescaleDB extension not available, macro_factors left as standard table';
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_macro_factors_ts
    ON macro_factors (ts DESC);

CREATE INDEX IF NOT EXISTS idx_macro_factors_key_ts
    ON macro_factors (factor_key, ts DESC);

CREATE INDEX IF NOT EXISTS idx_macro_factors_value
    ON macro_factors (factor_key, value);
