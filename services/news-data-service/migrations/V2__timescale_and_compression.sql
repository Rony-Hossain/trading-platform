-- Optional: enable Timescale compression policy after 7d
-- Uncomment these lines if using TimescaleDB:

-- ALTER TABLE content SET (
--   timescaledb.compress,
--   timescaledb.compress_orderby='published_at',
--   timescaledb.compress_segmentby='source'
-- );

-- SELECT add_compression_policy('content', INTERVAL '7 days');
