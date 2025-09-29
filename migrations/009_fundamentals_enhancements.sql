-- Fundamentals Service Enhancements (Phase 1 Task 2)
-- Creates normalized tables for consensus estimates, analyst revisions, insider transactions,
-- institutional holdings, ownership flow analytics, and revision momentum tracking.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS consensus_estimates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    report_date DATE NOT NULL,
    fiscal_period TEXT NOT NULL,
    fiscal_year INT NOT NULL,
    analyst_count INT,
    estimate_eps NUMERIC(10, 4),
    estimate_eps_high NUMERIC(10, 4),
    estimate_eps_low NUMERIC(10, 4),
    actual_eps NUMERIC(10, 4),
    surprise_percent NUMERIC(8, 4),
    estimate_revenue BIGINT,
    estimate_revenue_high BIGINT,
    estimate_revenue_low BIGINT,
    actual_revenue BIGINT,
    guidance_eps NUMERIC(10, 4),
    guidance_revenue BIGINT,
    source TEXT,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_consensus_estimates_symbol_date
    ON consensus_estimates (symbol, report_date DESC);

CREATE TABLE IF NOT EXISTS analyst_revisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    revision_date DATE NOT NULL,
    analyst TEXT,
    firm TEXT,
    action TEXT,
    from_rating TEXT,
    to_rating TEXT,
    old_price_target NUMERIC(12, 4),
    new_price_target NUMERIC(12, 4),
    rating_score NUMERIC(6, 3),
    notes TEXT,
    source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analyst_revisions_symbol_date
    ON analyst_revisions (symbol, revision_date DESC);

CREATE TABLE IF NOT EXISTS insider_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    insider TEXT NOT NULL,
    relationship TEXT,
    transaction_date DATE NOT NULL,
    transaction_type TEXT,
    shares BIGINT,
    share_change BIGINT,
    price NUMERIC(14, 4),
    total_value BIGINT,
    filing_date DATE,
    link TEXT,
    source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_insider_transactions_symbol_date
    ON insider_transactions (symbol, transaction_date DESC);

CREATE TABLE IF NOT EXISTS institutional_holdings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    institution_name TEXT NOT NULL,
    institution_cik TEXT,
    filing_date DATE NOT NULL,
    quarter_end DATE NOT NULL,
    shares_held BIGINT,
    market_value NUMERIC(15, 2),
    percentage_ownership NUMERIC(8, 4),
    shares_change BIGINT,
    shares_change_pct NUMERIC(8, 4),
    form13f_url TEXT,
    is_new_position BOOLEAN DEFAULT FALSE,
    is_sold_out BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_institutional_holdings_symbol_quarter
    ON institutional_holdings (symbol, quarter_end DESC);

CREATE TABLE IF NOT EXISTS ownership_flow_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    analysis_date DATE NOT NULL,
    period_days INT NOT NULL,
    insider_buy_transactions INT DEFAULT 0,
    insider_sell_transactions INT DEFAULT 0,
    insider_net_shares BIGINT DEFAULT 0,
    insider_net_value NUMERIC(15, 2) DEFAULT 0,
    insider_buy_value NUMERIC(15, 2) DEFAULT 0,
    insider_sell_value NUMERIC(15, 2) DEFAULT 0,
    institutions_increasing INT DEFAULT 0,
    institutions_decreasing INT DEFAULT 0,
    institutions_new_positions INT DEFAULT 0,
    institutions_sold_out INT DEFAULT 0,
    institutional_net_shares BIGINT DEFAULT 0,
    institutional_net_value NUMERIC(15, 2) DEFAULT 0,
    cluster_buying_detected BOOLEAN DEFAULT FALSE,
    cluster_selling_detected BOOLEAN DEFAULT FALSE,
    smart_money_score NUMERIC(5, 4) NOT NULL,
    confidence_level NUMERIC(5, 4) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ownership_flow_analysis_symbol_date
    ON ownership_flow_analysis (symbol, analysis_date DESC);

CREATE TABLE IF NOT EXISTS revision_momentum (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT NOT NULL,
    analysis_period_start DATE NOT NULL,
    analysis_period_end DATE NOT NULL,
    days_analyzed INT NOT NULL,
    total_revisions INT DEFAULT 0,
    upgrades INT DEFAULT 0,
    downgrades INT DEFAULT 0,
    initiations INT DEFAULT 0,
    net_rating_changes INT DEFAULT 0,
    rating_momentum_score NUMERIC(5, 4) DEFAULT 0,
    price_target_revisions INT DEFAULT 0,
    price_target_increases INT DEFAULT 0,
    price_target_decreases INT DEFAULT 0,
    average_price_target_change_pct NUMERIC(8, 4) DEFAULT 0,
    price_target_momentum_score NUMERIC(5, 4) DEFAULT 0,
    consensus_rating_change NUMERIC(5, 4) DEFAULT 0,
    consensus_price_target_change_pct NUMERIC(8, 4),
    consensus_eps_revision_pct NUMERIC(8, 4),
    revision_intensity NUMERIC(8, 6) DEFAULT 0,
    momentum_acceleration NUMERIC(8, 6) DEFAULT 0,
    conviction_score NUMERIC(5, 4) DEFAULT 0,
    pre_earnings_momentum BOOLEAN DEFAULT FALSE,
    unusual_activity_detected BOOLEAN DEFAULT FALSE,
    smart_money_following BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_revision_momentum_symbol_end
    ON revision_momentum (symbol, analysis_period_end DESC);
