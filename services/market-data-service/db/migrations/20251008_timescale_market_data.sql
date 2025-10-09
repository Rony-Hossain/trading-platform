-- Market Data Service - TimescaleDB Migration
-- Version: 20251008
-- Description: TimescaleDB-optimized schema with hypertables, compression, and continuous aggregates

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ==== Types (enums) ====
DO $$ BEGIN
  CREATE TYPE bar_interval AS ENUM ('1m','5m','15m','1h','1d');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE bar_status AS ENUM ('open','final');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ==== Core tables (PIT-safe) ====
CREATE TABLE IF NOT EXISTS candles_intraday (
  symbol   TEXT        NOT NULL,
  ts       TIMESTAMPTZ NOT NULL,
  interval bar_interval NOT NULL,
  o        DOUBLE PRECISION NOT NULL,
  h        DOUBLE PRECISION NOT NULL,
  l        DOUBLE PRECISION NOT NULL,
  c        DOUBLE PRECISION NOT NULL,
  v        DOUBLE PRECISION NOT NULL DEFAULT 0,
  provider TEXT        NOT NULL,
  as_of    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  status   bar_status  NOT NULL DEFAULT 'final',
  PRIMARY KEY (symbol, ts, interval)
);

CREATE TABLE IF NOT EXISTS quotes_l1 (
  symbol   TEXT        NOT NULL,
  ts       TIMESTAMPTZ NOT NULL,
  bid      DOUBLE PRECISION,
  ask      DOUBLE PRECISION,
  bid_size DOUBLE PRECISION,
  ask_size DOUBLE PRECISION,
  provider TEXT        NOT NULL,
  as_of    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS symbol_universe (
  symbol  TEXT PRIMARY KEY,
  exchange TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  adv_21d DOUBLE PRECISION,
  mkt_cap  DOUBLE PRECISION,
  tier     TEXT NOT NULL,
  active   BOOLEAN NOT NULL DEFAULT TRUE,
  provider_symbol_map JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS universe_versions (
  id SERIAL PRIMARY KEY,
  version_tag TEXT NOT NULL UNIQUE,
  source_meta JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ingestion_cursor (
  symbol     TEXT NOT NULL,
  interval   bar_interval NOT NULL,
  source     TEXT NOT NULL,
  last_ts    TIMESTAMPTZ NOT NULL,
  last_status TEXT NOT NULL DEFAULT 'ok',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (symbol, interval, source)
);

CREATE TABLE IF NOT EXISTS backfill_jobs (
  id BIGSERIAL PRIMARY KEY,
  symbol   TEXT NOT NULL,
  interval bar_interval NOT NULL,
  start_ts TIMESTAMPTZ NOT NULL,
  end_ts   TIMESTAMPTZ NOT NULL,
  priority TEXT NOT NULL DEFAULT 'T2',
  status   TEXT NOT NULL DEFAULT 'queued',
  attempts INT  NOT NULL DEFAULT 0,
  leased_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_error TEXT
);

-- ==== Indexes (before converting to hypertables) ====
CREATE INDEX IF NOT EXISTS ix_universe_active_tier ON symbol_universe (active, tier);
CREATE INDEX IF NOT EXISTS ix_universe_adv ON symbol_universe (adv_21d DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS ix_cursor_updated_at ON ingestion_cursor (updated_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_backfill_window
  ON backfill_jobs (symbol, interval, start_ts, end_ts, priority);

CREATE INDEX IF NOT EXISTS ix_backfill_status_priority_created
  ON backfill_jobs (status, priority, created_at);

-- ==== Convert to Hypertables ====
SELECT create_hypertable(
  'candles_intraday', 'ts',
  chunk_time_interval => INTERVAL '1 day',
  if_not_exists => TRUE
);

SELECT create_hypertable(
  'quotes_l1', 'ts',
  chunk_time_interval => INTERVAL '1 day',
  if_not_exists => TRUE
);

-- ==== Compression Settings ====
ALTER TABLE candles_intraday SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'symbol,interval',
  timescaledb.compress_orderby = 'ts DESC'
);

ALTER TABLE quotes_l1 SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'symbol',
  timescaledb.compress_orderby = 'ts DESC'
);

-- ==== Compression Policies ====
-- Compress candles older than 14 days
SELECT add_compression_policy('candles_intraday', INTERVAL '14 days');

-- Compress quotes older than 7 days
SELECT add_compression_policy('quotes_l1', INTERVAL '7 days');

-- ==== Retention Policies ====
-- Keep 365 days of candles
SELECT add_retention_policy('candles_intraday', INTERVAL '365 days');

-- Keep 14 days of quotes
SELECT add_retention_policy('quotes_l1', INTERVAL '14 days');

-- ==== Continuous Aggregates ====
-- 5-minute aggregate from 1m bars
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_5m
WITH (timescaledb.continuous) AS
SELECT
  symbol,
  time_bucket(INTERVAL '5 minutes', ts) AS ts,
  '5m'::bar_interval AS interval,
  first(o, ts) AS o,
  max(h) AS h,
  min(l) AS l,
  last(c, ts) AS c,
  sum(v) AS v,
  mode() WITHIN GROUP (ORDER BY provider) AS provider
FROM candles_intraday
WHERE interval = '1m'
GROUP BY symbol, time_bucket(INTERVAL '5 minutes', ts);

-- 1-hour aggregate from 1m bars
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_1h
WITH (timescaledb.continuous) AS
SELECT
  symbol,
  time_bucket(INTERVAL '1 hour', ts) AS ts,
  '1h'::bar_interval AS interval,
  first(o, ts) AS o,
  max(h) AS h,
  min(l) AS l,
  last(c, ts) AS c,
  sum(v) AS v,
  mode() WITHIN GROUP (ORDER BY provider) AS provider
FROM candles_intraday
WHERE interval = '1m'
GROUP BY symbol, time_bucket(INTERVAL '1 hour', ts);

-- ==== Continuous Aggregate Refresh Policies ====
SELECT add_continuous_aggregate_policy(
  'candles_5m',
  start_offset => INTERVAL '30 days',
  end_offset   => INTERVAL '1 minute',
  schedule_interval => INTERVAL '5 minutes'
);

SELECT add_continuous_aggregate_policy(
  'candles_1h',
  start_offset => INTERVAL '180 days',
  end_offset   => INTERVAL '1 hour',
  schedule_interval => INTERVAL '15 minutes'
);

-- ==== Additional indexes on hypertables ====
CREATE INDEX IF NOT EXISTS ix_candles_symbol_ts
  ON candles_intraday (symbol, ts DESC);

CREATE INDEX IF NOT EXISTS ix_candles_ts
  ON candles_intraday (ts DESC)
  WHERE status = 'final';

CREATE INDEX IF NOT EXISTS ix_quotes_symbol_ts
  ON quotes_l1 (symbol, ts DESC);

-- ==== Triggers for timestamp updates ====
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
  CREATE TRIGGER trg_touch_universe
    BEFORE UPDATE ON symbol_universe
    FOR EACH ROW
    EXECUTE PROCEDURE touch_updated_at();
EXCEPTION
  WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TRIGGER trg_touch_backfill
    BEFORE UPDATE ON backfill_jobs
    FOR EACH ROW
    EXECUTE PROCEDURE touch_updated_at();
EXCEPTION
  WHEN duplicate_object THEN NULL;
END $$;

-- ==== Informational Views ====
CREATE OR REPLACE VIEW candles_stats AS
SELECT
  interval,
  COUNT(DISTINCT symbol) as symbol_count,
  COUNT(*) as total_bars,
  MIN(ts) as earliest_bar,
  MAX(ts) as latest_bar,
  pg_size_pretty(hypertable_size('candles_intraday')) as total_size
FROM candles_intraday
GROUP BY interval;

-- ==== Migration complete ====
COMMENT ON TABLE candles_intraday IS 'TimescaleDB hypertable for OHLCV bars with compression and retention';
COMMENT ON TABLE quotes_l1 IS 'TimescaleDB hypertable for L1 quotes with 14-day retention';
COMMENT ON VIEW candles_5m IS 'Continuous aggregate: 5-minute bars from 1m data';
COMMENT ON VIEW candles_1h IS 'Continuous aggregate: 1-hour bars from 1m data';
