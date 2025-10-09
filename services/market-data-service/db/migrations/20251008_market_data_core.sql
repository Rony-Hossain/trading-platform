-- Market Data Service Core Schema Migration
-- Version: 20251008
-- Description: Adds PIT-safe candles, universe, cursors, and backfill infrastructure

-- === OPTIONAL (if using TimescaleDB) ===
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- === Core types/enums ===
DO $$ BEGIN
  CREATE TYPE bar_interval AS ENUM ('1m','5m','15m','1h','1d');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE bar_status AS ENUM ('open','final');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- === Candles (PIT-safe) ===
CREATE TABLE IF NOT EXISTS candles_intraday (
  symbol            TEXT        NOT NULL,
  ts                TIMESTAMPTZ NOT NULL,
  interval          bar_interval NOT NULL,
  o                 DOUBLE PRECISION NOT NULL,
  h                 DOUBLE PRECISION NOT NULL,
  l                 DOUBLE PRECISION NOT NULL,
  c                 DOUBLE PRECISION NOT NULL,
  v                 DOUBLE PRECISION NOT NULL DEFAULT 0,
  provider          TEXT        NOT NULL,
  as_of             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  status            bar_status  NOT NULL DEFAULT 'final',
  PRIMARY KEY (symbol, ts, interval)
);

CREATE INDEX IF NOT EXISTS ix_candles_symbol_ts ON candles_intraday (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS ix_candles_ts ON candles_intraday (ts DESC);

-- === Quotes (L1) ===
CREATE TABLE IF NOT EXISTS quotes_l1 (
  symbol            TEXT        NOT NULL,
  ts                TIMESTAMPTZ NOT NULL,
  bid               DOUBLE PRECISION,
  ask               DOUBLE PRECISION,
  bid_size          DOUBLE PRECISION,
  ask_size          DOUBLE PRECISION,
  provider          TEXT        NOT NULL,
  as_of             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS ix_quotes_symbol_ts ON quotes_l1 (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS ix_quotes_ts ON quotes_l1 (ts DESC);

-- === Symbol universe + mapping ===
CREATE TABLE IF NOT EXISTS symbol_universe (
  symbol               TEXT PRIMARY KEY,
  exchange             TEXT NOT NULL,
  asset_type           TEXT NOT NULL,
  adv_21d              DOUBLE PRECISION,
  mkt_cap              DOUBLE PRECISION,
  tier                 TEXT NOT NULL,       -- 'T0','T1','T2'
  active               BOOLEAN NOT NULL DEFAULT TRUE,
  provider_symbol_map  JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_universe_active_tier ON symbol_universe (active, tier);
CREATE INDEX IF NOT EXISTS ix_universe_adv ON symbol_universe (adv_21d DESC NULLS LAST);

-- === Universe versioning ===
CREATE TABLE IF NOT EXISTS universe_versions (
  id                BIGSERIAL PRIMARY KEY,
  version_tag       TEXT NOT NULL UNIQUE,
  source_meta       JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- === Ingestion cursor (continuity tracking) ===
CREATE TABLE IF NOT EXISTS ingestion_cursor (
  symbol        TEXT        NOT NULL,
  interval      TEXT        NOT NULL,  -- using TEXT instead of enum for flexibility
  source        TEXT        NOT NULL,  -- provider id or 'auto'
  last_ts       TIMESTAMPTZ NOT NULL,
  last_status   TEXT        NOT NULL DEFAULT 'ok',
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (symbol, interval, source)
);

CREATE INDEX IF NOT EXISTS ix_cursor_updated_at ON ingestion_cursor (updated_at DESC);

-- === Backfill jobs (gap repair queue) ===
CREATE TABLE IF NOT EXISTS backfill_jobs (
  id           BIGSERIAL PRIMARY KEY,
  symbol       TEXT        NOT NULL,
  interval     TEXT        NOT NULL,
  start_ts     TIMESTAMPTZ NOT NULL,
  end_ts       TIMESTAMPTZ NOT NULL,
  priority     TEXT        NOT NULL DEFAULT 'T2',  -- 'T0','T1','T2'
  status       TEXT        NOT NULL DEFAULT 'queued',  -- 'queued','leased','done','failed'
  attempts     INTEGER     NOT NULL DEFAULT 0,
  leased_at    TIMESTAMPTZ,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_error   TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_backfill_window
  ON backfill_jobs (symbol, interval, start_ts, end_ts, priority);

CREATE INDEX IF NOT EXISTS ix_backfill_status_priority_created
  ON backfill_jobs (status, priority, created_at);

CREATE INDEX IF NOT EXISTS ix_backfill_created_status
  ON backfill_jobs (created_at DESC, status);

-- === Macro factors table (already exists, ensuring consistency) ===
CREATE TABLE IF NOT EXISTS macro_factors (
  id           BIGSERIAL PRIMARY KEY,
  factor_key   TEXT        NOT NULL,
  ts           TIMESTAMPTZ NOT NULL,
  value        DOUBLE PRECISION NOT NULL,
  source       TEXT,
  metadata     JSONB,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(factor_key, ts)
);

CREATE INDEX IF NOT EXISTS ix_macro_factor_ts ON macro_factors (factor_key, ts DESC);

-- === Options metrics table (already exists, ensuring consistency) ===
CREATE TABLE IF NOT EXISTS options_metrics (
  id                      BIGSERIAL PRIMARY KEY,
  symbol                  TEXT NOT NULL,
  as_of                   TIMESTAMPTZ NOT NULL,
  expiry                  TIMESTAMPTZ,
  underlying_price        DOUBLE PRECISION,
  atm_strike              DOUBLE PRECISION,
  atm_iv                  DOUBLE PRECISION,
  call_volume             INTEGER DEFAULT 0,
  put_volume              INTEGER DEFAULT 0,
  call_open_interest      INTEGER DEFAULT 0,
  put_open_interest       INTEGER DEFAULT 0,
  put_call_volume_ratio   DOUBLE PRECISION,
  put_call_oi_ratio       DOUBLE PRECISION,
  straddle_price          DOUBLE PRECISION,
  implied_move_pct        DOUBLE PRECISION,
  implied_move_upper      DOUBLE PRECISION,
  implied_move_lower      DOUBLE PRECISION,
  iv_25d_call             DOUBLE PRECISION,
  iv_25d_put              DOUBLE PRECISION,
  iv_skew_25d             DOUBLE PRECISION,
  iv_skew_25d_pct         DOUBLE PRECISION,
  metadata                JSONB,
  created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(symbol, as_of, expiry)
);

CREATE INDEX IF NOT EXISTS ix_options_symbol_as_of ON options_metrics (symbol, as_of DESC);

-- === OPTIONAL Timescale policies (uncomment if using TimescaleDB) ===
-- SELECT create_hypertable('candles_intraday','ts', if_not_exists=>TRUE);
-- -- compress by symbol+interval for better columnar grouping
-- ALTER TABLE candles_intraday SET (
--   timescaledb.compress,
--   timescaledb.compress_segmentby = 'symbol,interval'
-- );
-- -- keep raw 14d hot, compress older; adjust to taste
-- SELECT add_compression_policy('candles_intraday', INTERVAL '14 days');
-- -- example retention: keep 365d of 1m bars; adjust if you have colder storage
-- SELECT add_retention_policy('candles_intraday', INTERVAL '365 days');

-- Optional continuous aggregates examples:
-- CREATE MATERIALIZED VIEW IF NOT EXISTS candles_5m
-- WITH (timescaledb.continuous) AS
--   SELECT symbol, time_bucket(INTERVAL '5 minutes', ts) AS ts, '5m'::bar_interval AS interval,
--          first(o, ts) AS o, max(h) AS h, min(l) AS l, last(c, ts) AS c, sum(v) AS v
--   FROM candles_intraday
--   WHERE interval='1m'
--   GROUP BY symbol, time_bucket(INTERVAL '5 minutes', ts);
-- SELECT add_continuous_aggregate_policy('candles_5m',
--   start_offset => INTERVAL '30 days',
--   end_offset   => INTERVAL '0',
--   schedule_interval => INTERVAL '5 minutes');

-- === Housekeeping triggers (timestamps) ===
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END; $$ LANGUAGE plpgsql;

DO $$ BEGIN
  CREATE TRIGGER trg_touch_universe BEFORE UPDATE ON symbol_universe
  FOR EACH ROW EXECUTE PROCEDURE touch_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TRIGGER trg_touch_backfill BEFORE UPDATE ON backfill_jobs
  FOR EACH ROW EXECUTE PROCEDURE touch_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- === Migration complete ===
-- Run this migration with:
-- psql -U <user> -d <db> -f db/migrations/20251008_market_data_core.sql
