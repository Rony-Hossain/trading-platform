-- Flyway SQL: core tables and constraints
CREATE TABLE IF NOT EXISTS content (
  content_id uuid primary key default gen_random_uuid(),
  source text not null,
  external_id text not null,
  url text not null,
  title text not null,
  body text null,
  authors jsonb null,
  categories jsonb null,
  language text null,
  published_at timestamptz not null,
  ingested_at timestamptz not null default now(),
  as_of_ts timestamptz not null,
  revision_seq int not null default 0,
  status text not null default 'active',
  metadata jsonb null,
  exact_dedupe_key text not null
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_source_external ON content(source, external_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_exact_dedupe ON content(exact_dedupe_key);
CREATE INDEX IF NOT EXISTS idx_content_published_at ON content(published_at desc);

CREATE TABLE IF NOT EXISTS outbox (
  id bigserial primary key,
  payload jsonb not null,
  status text not null default 'PENDING',
  retries int not null default 0,
  created_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS outbox_dlq (
  id bigserial primary key,
  payload jsonb not null,
  last_error text null,
  retries int not null,
  created_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS ingest_audit (
  id bigserial primary key,
  content_id uuid null,
  source text not null,
  external_id text not null,
  reason text not null,
  dedupe_key text null,
  details jsonb null,
  created_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS backfill_state (
  id bigserial primary key,
  source text not null,
  start_ts timestamptz not null,
  end_ts timestamptz not null,
  status text not null default 'PENDING',
  progress jsonb null,
  created_at timestamptz not null default now(),
  completed_at timestamptz null
);

-- Timescale hypertable optional (enable in V2 migration)
-- SELECT create_hypertable('content','published_at', if_not_exists => TRUE);
