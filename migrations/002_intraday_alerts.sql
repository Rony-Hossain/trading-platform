-- Intraday candles (minute bars) and alerting tables

-- Minute candles (optional TimescaleDB hypertable in the future)
CREATE TABLE IF NOT EXISTS candles_intraday (
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    open            NUMERIC(18,6) NOT NULL,
    high            NUMERIC(18,6) NOT NULL,
    low             NUMERIC(18,6) NOT NULL,
    close           NUMERIC(18,6) NOT NULL,
    volume          BIGINT      NOT NULL DEFAULT 0,
    vwap            NUMERIC(18,6),
    PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_intraday_symbol_ts
    ON candles_intraday (symbol, ts DESC);

-- User-defined alert rules
CREATE TABLE IF NOT EXISTS alerts (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID         NOT NULL,
    symbol       TEXT         NOT NULL,
    rule_type    TEXT         NOT NULL, -- vwap_cross | orb_break | pct_change | rsi_extreme | volume_spike
    params       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    is_active    BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_user_symbol
    ON alerts (user_id, symbol);

-- Alert trigger audit log
CREATE TABLE IF NOT EXISTS alert_triggers (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id      UUID         NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    triggered_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    payload       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    delivered     BOOLEAN      NOT NULL DEFAULT FALSE,
    dedupe_key    TEXT         UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_alert_triggers_alert_time
    ON alert_triggers (alert_id, triggered_at DESC);

