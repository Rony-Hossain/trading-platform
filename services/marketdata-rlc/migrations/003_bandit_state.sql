CREATE TABLE IF NOT EXISTS rlc_bandit_state(
  provider text NOT NULL,
  regime text NOT NULL,
  arm_id int NOT NULL,
  batch_size int NOT NULL,
  delay_ms int NOT NULL,
  n int NOT NULL DEFAULT 0,
  mean_reward double precision NOT NULL DEFAULT 0,
  m2 double precision NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (provider, regime, arm_id)
);
