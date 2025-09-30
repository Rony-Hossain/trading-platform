-- Migration 010: Skew Monitoring Tables
-- Creates tables for offline/online performance skew detection and monitoring

-- Table for storing skew alerts
CREATE TABLE IF NOT EXISTS skew_alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy_name VARCHAR(100) NOT NULL,
    metric VARCHAR(50) NOT NULL,
    offline_value DECIMAL(15,6),
    online_value DECIMAL(15,6),
    skew_magnitude DECIMAL(10,6) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    confidence DECIMAL(5,4) DEFAULT 0.0,
    lookback_days INTEGER NOT NULL DEFAULT 30,
    message TEXT,
    metadata JSONB,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for backtest results (enhanced for skew comparison)
CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    backtest_start_date DATE NOT NULL,
    backtest_end_date DATE NOT NULL,
    total_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(8,4),
    calmar_ratio DECIMAL(8,4),
    information_ratio DECIMAL(8,4),
    daily_returns_mean DECIMAL(12,8),
    daily_returns_std DECIMAL(12,8),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_trade_duration_hours DECIMAL(8,2),
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    parameters JSONB, -- Strategy parameters used
    market_conditions JSONB, -- Market regime during backtest
    data_quality_score DECIMAL(3,2) DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for trade executions (enhanced for online performance tracking)
CREATE TABLE IF NOT EXISTS trade_executions (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('BUY', 'SELL', 'SHORT', 'COVER')),
    quantity DECIMAL(15,6) NOT NULL,
    price DECIMAL(15,6) NOT NULL,
    execution_time TIMESTAMPTZ NOT NULL,
    order_id VARCHAR(50),
    fill_id VARCHAR(50),
    commission DECIMAL(10,4) DEFAULT 0.0,
    realized_pnl DECIMAL(15,6),
    unrealized_pnl DECIMAL(15,6),
    position_size DECIMAL(15,6),
    market_value DECIMAL(15,6),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected')),
    signal_strength DECIMAL(5,4), -- Confidence in trade signal
    market_impact DECIMAL(8,6), -- Estimated market impact
    slippage DECIMAL(8,6), -- Actual vs expected price
    latency_ms INTEGER, -- Order execution latency
    venue VARCHAR(50), -- Execution venue
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for strategy configurations
CREATE TABLE IF NOT EXISTS strategy_configs (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    risk_limit DECIMAL(10,6),
    max_position_size DECIMAL(15,6),
    max_daily_trades INTEGER,
    parameters JSONB,
    performance_thresholds JSONB, -- Skew alert thresholds
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_skew_alerts_strategy_timestamp 
ON skew_alerts (strategy_name, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_skew_alerts_severity_timestamp 
ON skew_alerts (severity, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_skew_alerts_metric_timestamp 
ON skew_alerts (metric, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtest_results_strategy_dates 
ON backtest_results (strategy_name, backtest_end_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trade_executions_strategy_time 
ON trade_executions (strategy_name, execution_time DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trade_executions_symbol_time 
ON trade_executions (symbol, execution_time DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trade_executions_status_time 
ON trade_executions (status, execution_time DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
CREATE TRIGGER update_skew_alerts_updated_at 
BEFORE UPDATE ON skew_alerts 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_backtest_results_updated_at 
BEFORE UPDATE ON backtest_results 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trade_executions_updated_at 
BEFORE UPDATE ON trade_executions 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_strategy_configs_updated_at 
BEFORE UPDATE ON strategy_configs 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Materialized view for performance monitoring dashboard
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_strategy_performance_summary AS
SELECT 
    s.strategy_name,
    s.is_active,
    COUNT(DISTINCT sa.id) as total_alerts_30d,
    COUNT(DISTINCT CASE WHEN sa.severity IN ('high', 'critical') THEN sa.id END) as critical_alerts_30d,
    MAX(sa.timestamp) as last_alert_time,
    COUNT(DISTINCT te.id) as total_trades_30d,
    AVG(te.realized_pnl) as avg_daily_pnl_30d,
    STDDEV(te.realized_pnl) as pnl_volatility_30d,
    MAX(br.sharpe_ratio) as latest_backtest_sharpe,
    MAX(br.created_at) as latest_backtest_date
FROM strategy_configs s
LEFT JOIN skew_alerts sa ON s.strategy_name = sa.strategy_name 
    AND sa.timestamp >= NOW() - INTERVAL '30 days'
LEFT JOIN trade_executions te ON s.strategy_name = te.strategy_name 
    AND te.execution_time >= NOW() - INTERVAL '30 days'
    AND te.status = 'filled'
LEFT JOIN backtest_results br ON s.strategy_name = br.strategy_name
GROUP BY s.strategy_name, s.is_active;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_strategy_performance_summary_name 
ON mv_strategy_performance_summary (strategy_name);

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_strategy_performance_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_strategy_performance_summary;
END;
$$ LANGUAGE plpgsql;

-- Insert sample strategy configurations
INSERT INTO strategy_configs (strategy_name, is_active, parameters, performance_thresholds) 
VALUES 
    ('SPY_MOMENTUM', true, 
     '{"lookback_period": 20, "momentum_threshold": 0.02}',
     '{"sharpe_ratio_threshold": 0.3, "max_drawdown_threshold": 0.5}'
    ),
    ('VIX_MEAN_REVERT', true,
     '{"vix_threshold": 25, "position_size": 0.1}', 
     '{"sharpe_ratio_threshold": 0.25, "max_drawdown_threshold": 0.4}'
    ),
    ('EARNINGS_MOMENTUM', true,
     '{"earnings_window": 5, "volume_threshold": 1.5}',
     '{"win_rate_threshold": 0.6, "profit_factor_threshold": 1.2}'
    )
ON CONFLICT (strategy_name) DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE skew_alerts IS 'Stores performance skew alerts between offline backtests and online production';
COMMENT ON TABLE backtest_results IS 'Stores comprehensive backtest performance metrics for skew comparison';
COMMENT ON TABLE trade_executions IS 'Enhanced trade execution records for online performance calculation';
COMMENT ON TABLE strategy_configs IS 'Strategy configurations and performance thresholds for monitoring';
COMMENT ON MATERIALIZED VIEW mv_strategy_performance_summary IS 'Aggregated performance metrics for monitoring dashboard';

COMMENT ON COLUMN skew_alerts.skew_magnitude IS 'Relative difference between offline and online performance (0.3 = 30% difference)';
COMMENT ON COLUMN skew_alerts.confidence IS 'Statistical confidence in the detected skew (0.0-1.0)';
COMMENT ON COLUMN backtest_results.data_quality_score IS 'Quality score for backtest data (0.0-1.0)';
COMMENT ON COLUMN trade_executions.market_impact IS 'Estimated market impact of the trade';
COMMENT ON COLUMN trade_executions.slippage IS 'Difference between expected and actual execution price';