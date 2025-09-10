-- Seed data for trading platform
-- This file contains initial test data for development

-- Insert test user
INSERT INTO users (id, email, password_hash) VALUES 
  ('123e4567-e89b-12d3-a456-426614174000', 'test@example.com', '$2b$12$example_hash_here'),
  ('123e4567-e89b-12d3-a456-426614174001', 'demo@example.com', '$2b$12$another_hash_here')
ON CONFLICT (email) DO NOTHING;

-- Insert default portfolio for test user
INSERT INTO portfolios (id, user_id, name, description) VALUES 
  ('123e4567-e89b-12d3-a456-426614174010', '123e4567-e89b-12d3-a456-426614174000', 'Main Portfolio', 'Primary investment portfolio'),
  ('123e4567-e89b-12d3-a456-426614174011', '123e4567-e89b-12d3-a456-426614174001', 'Demo Portfolio', 'Demo portfolio for testing')
ON CONFLICT (id) DO NOTHING;

-- Insert some portfolio positions
INSERT INTO portfolio_positions (portfolio_id, symbol, quantity, average_cost) VALUES 
  ('123e4567-e89b-12d3-a456-426614174010', 'AAPL', 100, 150.00),
  ('123e4567-e89b-12d3-a456-426614174010', 'TSLA', 50, 200.00),
  ('123e4567-e89b-12d3-a456-426614174010', 'MSFT', 75, 300.00),
  ('123e4567-e89b-12d3-a456-426614174011', 'SPY', 200, 400.00),
  ('123e4567-e89b-12d3-a456-426614174011', 'QQQ', 100, 350.00)
ON CONFLICT (portfolio_id, symbol) DO NOTHING;

-- Insert default watchlist for test user
INSERT INTO watchlists (id, user_id, name) VALUES 
  ('123e4567-e89b-12d3-a456-426614174020', '123e4567-e89b-12d3-a456-426614174000', 'Tech Stocks'),
  ('123e4567-e89b-12d3-a456-426614174021', '123e4567-e89b-12d3-a456-426614174000', 'Growth Stocks'),
  ('123e4567-e89b-12d3-a456-426614174022', '123e4567-e89b-12d3-a456-426614174001', 'Demo Watchlist')
ON CONFLICT (id) DO NOTHING;

-- Insert watchlist items
INSERT INTO watchlist_items (watchlist_id, symbol) VALUES 
  ('123e4567-e89b-12d3-a456-426614174020', 'AAPL'),
  ('123e4567-e89b-12d3-a456-426614174020', 'MSFT'),
  ('123e4567-e89b-12d3-a456-426614174020', 'GOOGL'),
  ('123e4567-e89b-12d3-a456-426614174020', 'META'),
  ('123e4567-e89b-12d3-a456-426614174021', 'TSLA'),
  ('123e4567-e89b-12d3-a456-426614174021', 'NVDA'),
  ('123e4567-e89b-12d3-a456-426614174021', 'AMZN'),
  ('123e4567-e89b-12d3-a456-426614174022', 'SPY'),
  ('123e4567-e89b-12d3-a456-426614174022', 'QQQ'),
  ('123e4567-e89b-12d3-a456-426614174022', 'VTI')
ON CONFLICT (watchlist_id, symbol) DO NOTHING;

-- Insert test alerts
INSERT INTO alerts (id, user_id, symbol, type, target_value, metadata) VALUES 
  ('123e4567-e89b-12d3-a456-426614174030', '123e4567-e89b-12d3-a456-426614174000', 'AAPL', 'price_above', 160.00, '{"note": "Buy signal when above 160"}'),
  ('123e4567-e89b-12d3-a456-426614174031', '123e4567-e89b-12d3-a456-426614174000', 'TSLA', 'price_below', 180.00, '{"note": "Stop loss alert"}'),
  ('123e4567-e89b-12d3-a456-426614174032', '123e4567-e89b-12d3-a456-426614174000', 'MSFT', 'price_change', 5.0, '{"note": "5% price change alert", "period": "1d"}'),
  ('123e4567-e89b-12d3-a456-426614174033', '123e4567-e89b-12d3-a456-426614174001', 'SPY', 'volume_spike', 1.5, '{"note": "1.5x average volume spike"}')
ON CONFLICT (id) DO NOTHING;

-- Insert sample historical candles for popular symbols
-- This is just sample data - in production this would come from market data feeds
INSERT INTO candles (symbol, ts, open, high, low, close, volume) VALUES 
  ('AAPL', '2024-01-02 14:30:00+00', 185.64, 186.95, 185.55, 186.85, 45234567),
  ('AAPL', '2024-01-02 14:31:00+00', 186.85, 187.12, 186.45, 186.78, 12345678),
  ('AAPL', '2024-01-02 14:32:00+00', 186.78, 186.89, 186.23, 186.45, 8765432),
  ('TSLA', '2024-01-02 14:30:00+00', 248.48, 249.85, 247.12, 248.99, 23456789),
  ('TSLA', '2024-01-02 14:31:00+00', 248.99, 249.45, 248.67, 249.12, 9876543),
  ('TSLA', '2024-01-02 14:32:00+00', 249.12, 249.78, 248.89, 249.34, 7654321),
  ('MSFT', '2024-01-02 14:30:00+00', 376.04, 377.23, 375.89, 376.87, 18765432),
  ('MSFT', '2024-01-02 14:31:00+00', 376.87, 377.45, 376.34, 377.12, 6543219),
  ('MSFT', '2024-01-02 14:32:00+00', 377.12, 377.89, 376.78, 377.56, 5432198)
ON CONFLICT (symbol, ts) DO NOTHING;

-- Insert sample technical analysis cache entries
INSERT INTO technical_analysis_cache (symbol, period, analysis_data, expires_at) VALUES 
  ('AAPL', '1d', '{"sma_20": 185.45, "sma_50": 182.34, "rsi_14": 67.8, "macd": {"macd": 2.34, "signal": 1.89, "histogram": 0.45}}', NOW() + INTERVAL '1 hour'),
  ('TSLA', '1d', '{"sma_20": 245.67, "sma_50": 238.91, "rsi_14": 72.1, "macd": {"macd": 3.45, "signal": 2.78, "histogram": 0.67}}', NOW() + INTERVAL '1 hour'),
  ('MSFT', '1d', '{"sma_20": 374.23, "sma_50": 371.45, "rsi_14": 58.3, "macd": {"macd": 1.89, "signal": 1.56, "histogram": 0.33}}', NOW() + INTERVAL '1 hour')
ON CONFLICT (symbol, period) DO NOTHING;

-- Insert sample forecast cache entries
INSERT INTO forecast_cache (symbol, model_type, horizon, forecast_data, expires_at) VALUES 
  ('AAPL', 'ensemble', 5, '{"predictions": [187.45, 188.23, 189.67, 188.91, 190.12], "confidence": 0.78, "direction": "UP"}', NOW() + INTERVAL '4 hours'),
  ('TSLA', 'lstm', 5, '{"predictions": [250.34, 252.67, 248.91, 251.23, 253.45], "confidence": 0.65, "direction": "SIDEWAYS"}', NOW() + INTERVAL '4 hours'),
  ('MSFT', 'xgboost', 5, '{"predictions": [378.45, 379.23, 380.67, 381.91, 383.12], "confidence": 0.82, "direction": "UP"}', NOW() + INTERVAL '4 hours')
ON CONFLICT (symbol, model_type, horizon) DO NOTHING;