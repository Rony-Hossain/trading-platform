-- Migration: Trade Journal and P&L Attribution
-- Creates tables for tracking trade fills, fees, borrow costs, and P&L attribution

-- Trade fills table
CREATE TABLE IF NOT EXISTS trade_fills (
    fill_id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    venue VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    fill_price NUMERIC(20, 6) NOT NULL,
    fill_quantity INTEGER NOT NULL,
    fill_timestamp TIMESTAMPTZ NOT NULL,

    -- Execution details
    order_type VARCHAR(20) NOT NULL,
    limit_price NUMERIC(20, 6),
    time_in_force VARCHAR(20),

    -- Cost breakdown
    commission_usd NUMERIC(10, 4) NOT NULL DEFAULT 0,
    exchange_fee_usd NUMERIC(10, 4) NOT NULL DEFAULT 0,
    sec_fee_usd NUMERIC(10, 4) NOT NULL DEFAULT 0,
    finra_taf_usd NUMERIC(10, 4) NOT NULL DEFAULT 0,
    slippage_bps NUMERIC(10, 4),

    -- Routing metadata
    routing_decision_id VARCHAR(100),
    routing_latency_ms NUMERIC(10, 2),
    decision_factors JSONB,

    -- P&L tracking
    realized_pnl_usd NUMERIC(15, 4),
    unrealized_pnl_usd NUMERIC(15, 4),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for trade fills
CREATE INDEX idx_trade_fills_symbol ON trade_fills(symbol);
CREATE INDEX idx_trade_fills_timestamp ON trade_fills(fill_timestamp);
CREATE INDEX idx_trade_fills_order_id ON trade_fills(order_id);
CREATE INDEX idx_trade_fills_venue ON trade_fills(venue);
CREATE INDEX idx_trade_fills_symbol_timestamp ON trade_fills(symbol, fill_timestamp);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('trade_fills', 'fill_timestamp', if_not_exists => TRUE);

-- Fee breakdown table (for detailed fee analysis)
CREATE TABLE IF NOT EXISTS fee_breakdown (
    fee_id SERIAL PRIMARY KEY,
    fill_id INTEGER NOT NULL REFERENCES trade_fills(fill_id),
    fee_type VARCHAR(50) NOT NULL,
    fee_amount_usd NUMERIC(10, 4) NOT NULL,
    fee_bps NUMERIC(10, 4),
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_fee_breakdown_fill_id ON fee_breakdown(fill_id);
CREATE INDEX idx_fee_breakdown_fee_type ON fee_breakdown(fee_type);

-- Borrow costs table (for short positions)
CREATE TABLE IF NOT EXISTS borrow_costs (
    borrow_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    borrow_date DATE NOT NULL,
    shares_borrowed INTEGER NOT NULL,
    borrow_rate_pct NUMERIC(10, 6) NOT NULL,
    daily_cost_usd NUMERIC(10, 4) NOT NULL,
    trade_fill_id INTEGER REFERENCES trade_fills(fill_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_borrow_costs_symbol ON borrow_costs(symbol);
CREATE INDEX idx_borrow_costs_date ON borrow_costs(borrow_date);
CREATE INDEX idx_borrow_costs_fill_id ON borrow_costs(trade_fill_id);

-- P&L attribution table
CREATE TABLE IF NOT EXISTS pnl_attribution (
    pnl_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    attribution_date DATE NOT NULL,

    -- P&L components
    gross_pnl_usd NUMERIC(15, 4) NOT NULL,
    commission_cost_usd NUMERIC(10, 4) NOT NULL,
    slippage_cost_usd NUMERIC(10, 4) NOT NULL,
    borrow_cost_usd NUMERIC(10, 4) NOT NULL DEFAULT 0,
    other_fees_usd NUMERIC(10, 4) NOT NULL DEFAULT 0,
    net_pnl_usd NUMERIC(15, 4) NOT NULL,

    -- Attribution factors
    strategy_name VARCHAR(100),
    venue VARCHAR(50),
    entry_fill_id INTEGER REFERENCES trade_fills(fill_id),
    exit_fill_id INTEGER REFERENCES trade_fills(fill_id),

    -- Trade metrics
    entry_price NUMERIC(20, 6),
    exit_price NUMERIC(20, 6),
    quantity INTEGER,
    holding_period_hours NUMERIC(10, 2),

    -- Attribution breakdown
    market_impact_bps NUMERIC(10, 4),
    timing_alpha_bps NUMERIC(10, 4),
    venue_selection_savings_bps NUMERIC(10, 4),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for P&L attribution
CREATE INDEX idx_pnl_attribution_symbol ON pnl_attribution(symbol);
CREATE INDEX idx_pnl_attribution_date ON pnl_attribution(attribution_date);
CREATE INDEX idx_pnl_attribution_strategy ON pnl_attribution(strategy_name);
CREATE INDEX idx_pnl_attribution_venue ON pnl_attribution(venue);
CREATE INDEX idx_pnl_attribution_symbol_date ON pnl_attribution(symbol, attribution_date);

-- Position tracking table
CREATE TABLE IF NOT EXISTS positions (
    position_id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    current_quantity INTEGER NOT NULL,
    avg_entry_price NUMERIC(20, 6) NOT NULL,
    total_cost_basis_usd NUMERIC(15, 4) NOT NULL,
    unrealized_pnl_usd NUMERIC(15, 4),
    realized_pnl_usd NUMERIC(15, 4) DEFAULT 0,

    -- Tracking
    first_entry_fill_id INTEGER REFERENCES trade_fills(fill_id),
    last_update_fill_id INTEGER REFERENCES trade_fills(fill_id),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(symbol)
);

CREATE INDEX idx_positions_symbol ON positions(symbol);

-- Daily P&L summary view
CREATE OR REPLACE VIEW vw_daily_pnl_summary AS
SELECT
    attribution_date,
    COUNT(*) as num_trades,
    SUM(gross_pnl_usd) as total_gross_pnl,
    SUM(commission_cost_usd) as total_commissions,
    SUM(slippage_cost_usd) as total_slippage,
    SUM(borrow_cost_usd) as total_borrow_costs,
    SUM(net_pnl_usd) as total_net_pnl,
    AVG(market_impact_bps) as avg_market_impact_bps,
    AVG(timing_alpha_bps) as avg_timing_alpha_bps,
    AVG(venue_selection_savings_bps) as avg_venue_savings_bps
FROM pnl_attribution
GROUP BY attribution_date
ORDER BY attribution_date DESC;

-- Venue performance summary view
CREATE OR REPLACE VIEW vw_venue_performance AS
SELECT
    venue,
    COUNT(*) as fill_count,
    SUM(fill_quantity) as total_quantity,
    AVG(slippage_bps) as avg_slippage_bps,
    AVG(commission_usd + exchange_fee_usd + sec_fee_usd + finra_taf_usd) as avg_total_fees,
    AVG(routing_latency_ms) as avg_routing_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY routing_latency_ms) as p99_routing_latency_ms
FROM trade_fills
WHERE fill_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY venue
ORDER BY fill_count DESC;

-- Per-symbol P&L view
CREATE OR REPLACE VIEW vw_symbol_pnl AS
SELECT
    symbol,
    COUNT(*) as num_trades,
    SUM(net_pnl_usd) as total_net_pnl,
    AVG(net_pnl_usd) as avg_trade_pnl,
    SUM(CASE WHEN net_pnl_usd > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as win_rate,
    AVG(holding_period_hours) as avg_holding_hours
FROM pnl_attribution
WHERE attribution_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY symbol
ORDER BY total_net_pnl DESC;

-- Comments
COMMENT ON TABLE trade_fills IS 'Records every trade fill with execution details and costs';
COMMENT ON TABLE fee_breakdown IS 'Detailed breakdown of fees per fill';
COMMENT ON TABLE borrow_costs IS 'Daily borrow costs for short positions';
COMMENT ON TABLE pnl_attribution IS 'P&L attribution with breakdown by factors';
COMMENT ON TABLE positions IS 'Current position tracking with cost basis';
COMMENT ON VIEW vw_daily_pnl_summary IS 'Daily aggregated P&L metrics';
COMMENT ON VIEW vw_venue_performance IS 'Venue performance metrics over last 30 days';
COMMENT ON VIEW vw_symbol_pnl IS 'Per-symbol P&L summary over last 90 days';
