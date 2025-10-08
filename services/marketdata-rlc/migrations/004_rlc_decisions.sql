CREATE TABLE IF NOT EXISTS rlc_decisions(
  ts timestamptz NOT NULL,
  provider text NOT NULL,
  regime text NOT NULL,
  mode text NOT NULL,
  arm_id int,
  batch_size int,
  delay_ms int,
  target_rps numeric,
  token_refill_rate numeric,
  token_burst int,
  token_jitter_ms int,
  token_ttl_s int,
  PRIMARY KEY (provider, ts)
);

CREATE INDEX IF NOT EXISTS rlc_decisions_provider_ts_idx
  ON rlc_decisions(provider, ts DESC);
