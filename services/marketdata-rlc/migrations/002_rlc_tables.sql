-- Core RLC tables
CREATE TABLE IF NOT EXISTS rlc_predictions(
  ts timestamptz NOT NULL,
  provider text NOT NULL,
  predicted_p95_ms integer,
  predicted_error_prob numeric,
  recommended_refill numeric,
  recommended_burst integer,
  budget_envelope_usd_per_min numeric(12,6),
  policy_mode text,
  PRIMARY KEY (provider, ts)
);

CREATE TABLE IF NOT EXISTS token_bucket_state(
  ts timestamptz NOT NULL,
  provider text NOT NULL,
  refill_rate numeric,
  burst integer,
  tokens numeric,
  source text,
  PRIMARY KEY (provider, ts)
);

CREATE TABLE IF NOT EXISTS cost_ledger(
  ts timestamptz NOT NULL,
  provider text NOT NULL,
  tier text NOT NULL,
  symbols_covered integer,
  cost_usd numeric(12,6),
  regime text,
  PRIMARY KEY (provider, tier, ts)
);

CREATE TABLE IF NOT EXISTS latency_sip_series(
  ts timestamptz NOT NULL,
  tape text NOT NULL,
  p50_ms integer,
  p95_ms integer,
  source text,
  PRIMARY KEY (tape, ts)
);

CREATE TABLE IF NOT EXISTS tick_density(
  ts timestamptz NOT NULL,
  security_id uuid NOT NULL,
  expected numeric,
  observed numeric,
  zscore numeric,
  PRIMARY KEY (security_id, ts)
);

CREATE TABLE IF NOT EXISTS dq_violations(
  ts timestamptz NOT NULL,
  security_id uuid,
  rule text NOT NULL,
  severity text NOT NULL,
  details jsonb,
  window_start timestamptz,
  window_end timestamptz
);
CREATE INDEX IF NOT EXISTS dq_violations_rule_ts_idx
  ON dq_violations (rule, ts DESC);

-- Hypertables & retention
SELECT create_hypertable('rlc_predictions', 'ts', if_not_exists => TRUE);
SELECT create_hypertable('token_bucket_state', 'ts', if_not_exists => TRUE);
SELECT create_hypertable('cost_ledger', 'ts', if_not_exists => TRUE);
SELECT create_hypertable('latency_sip_series', 'ts', if_not_exists => TRUE);
SELECT create_hypertable('tick_density', 'ts', if_not_exists => TRUE);

SELECT add_retention_policy('rlc_predictions', INTERVAL '60 days', if_not_exists => TRUE);
SELECT add_retention_policy('token_bucket_state', INTERVAL '60 days', if_not_exists => TRUE);
SELECT add_retention_policy('cost_ledger', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('latency_sip_series', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('tick_density', INTERVAL '90 days', if_not_exists => TRUE);

ALTER TABLE rlc_predictions SET (timescaledb.compress, timescaledb.compress_segmentby='provider', timescaledb.compress_orderby='ts DESC');
ALTER TABLE token_bucket_state SET (timescaledb.compress, timescaledb.compress_segmentby='provider', timescaledb.compress_orderby='ts DESC');
ALTER TABLE cost_ledger SET (timescaledb.compress, timescaledb.compress_segmentby='provider,tier', timescaledb.compress_orderby='ts DESC');
ALTER TABLE latency_sip_series SET (timescaledb.compress, timescaledb.compress_segmentby='tape', timescaledb.compress_orderby='ts DESC');
ALTER TABLE tick_density SET (timescaledb.compress, timescaledb.compress_segmentby='security_id', timescaledb.compress_orderby='ts DESC');

SELECT add_compression_policy('rlc_predictions', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('token_bucket_state', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('cost_ledger', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('latency_sip_series', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('tick_density', INTERVAL '7 days', if_not_exists => TRUE);
