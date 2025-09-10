-- Additional development-only seed data
-- This extends the base seed_data.sql with more test data

-- Insert additional test users for development
INSERT INTO users (id, email, password_hash) VALUES 
  ('123e4567-e89b-12d3-a456-426614174002', 'alice@dev.com', '$2b$12$dev_hash_alice'),
  ('123e4567-e89b-12d3-a456-426614174003', 'bob@dev.com', '$2b$12$dev_hash_bob'),
  ('123e4567-e89b-12d3-a456-426614174004', 'charlie@dev.com', '$2b$12$dev_hash_charlie')
ON CONFLICT (email) DO NOTHING;

-- Insert more portfolios for testing
INSERT INTO portfolios (id, user_id, name, description) VALUES 
  ('123e4567-e89b-12d3-a456-426614174012', '123e4567-e89b-12d3-a456-426614174002', 'Growth Portfolio', 'High growth stocks'),
  ('123e4567-e89b-12d3-a456-426614174013', '123e4567-e89b-12d3-a456-426614174002', 'Value Portfolio', 'Value investing approach'),
  ('123e4567-e89b-12d3-a456-426614174014', '123e4567-e89b-12d3-a456-426614174003', 'Crypto Portfolio', 'Cryptocurrency investments'),
  ('123e4567-e89b-12d3-a456-426614174015', '123e4567-e89b-12d3-a456-426614174004', 'Conservative Portfolio', 'Safe investments')
ON CONFLICT (id) DO NOTHING;

-- Insert more diverse portfolio positions
INSERT INTO portfolio_positions (portfolio_id, symbol, quantity, average_cost) VALUES 
  -- Growth Portfolio
  ('123e4567-e89b-12d3-a456-426614174012', 'NVDA', 25, 450.00),
  ('123e4567-e89b-12d3-a456-426614174012', 'AMD', 50, 120.00),
  ('123e4567-e89b-12d3-a456-426614174012', 'NFLX', 30, 380.00),
  ('123e4567-e89b-12d3-a456-426614174012', 'CRM', 20, 220.00),
  
  -- Value Portfolio  
  ('123e4567-e89b-12d3-a456-426614174013', 'BRK.B', 40, 280.00),
  ('123e4567-e89b-12d3-a456-426614174013', 'JPM', 35, 140.00),
  ('123e4567-e89b-12d3-a456-426614174013', 'JNJ', 60, 160.00),
  ('123e4567-e89b-12d3-a456-426614174013', 'PG', 45, 150.00),
  
  -- Crypto Portfolio
  ('123e4567-e89b-12d3-a456-426614174014', 'COIN', 15, 85.00),
  ('123e4567-e89b-12d3-a456-426614174014', 'MSTR', 10, 320.00),
  ('123e4567-e89b-12d3-a456-426614174014', 'SQ', 25, 75.00),
  
  -- Conservative Portfolio
  ('123e4567-e89b-12d3-a456-426614174015', 'VTI', 100, 220.00),
  ('123e4567-e89b-12d3-a456-426614174015', 'BND', 150, 80.00),
  ('123e4567-e89b-12d3-a456-426614174015', 'SCHD', 80, 75.00)
ON CONFLICT (portfolio_id, symbol) DO NOTHING;

-- Insert more watchlists
INSERT INTO watchlists (id, user_id, name) VALUES 
  ('123e4567-e89b-12d3-a456-426614174023', '123e4567-e89b-12d3-a456-426614174002', 'AI Stocks'),
  ('123e4567-e89b-12d3-a456-426614174024', '123e4567-e89b-12d3-a456-426614174002', 'Dividend Stocks'),
  ('123e4567-e89b-12d3-a456-426614174025', '123e4567-e89b-12d3-a456-426614174003', 'Meme Stocks'),
  ('123e4567-e89b-12d3-a456-426614174026', '123e4567-e89b-12d3-a456-426614174004', 'Blue Chip')
ON CONFLICT (id) DO NOTHING;

-- Insert diverse watchlist items
INSERT INTO watchlist_items (watchlist_id, symbol) VALUES 
  -- AI Stocks
  ('123e4567-e89b-12d3-a456-426614174023', 'NVDA'),
  ('123e4567-e89b-12d3-a456-426614174023', 'AMD'),
  ('123e4567-e89b-12d3-a456-426614174023', 'PLTR'),
  ('123e4567-e89b-12d3-a456-426614174023', 'C3.AI'),
  
  -- Dividend Stocks
  ('123e4567-e89b-12d3-a456-426614174024', 'SCHD'),
  ('123e4567-e89b-12d3-a456-426614174024', 'JEPI'),
  ('123e4567-e89b-12d3-a456-426614174024', 'VYM'),
  ('123e4567-e89b-12d3-a456-426614174024', 'DIVO'),
  
  -- Meme Stocks
  ('123e4567-e89b-12d3-a456-426614174025', 'GME'),
  ('123e4567-e89b-12d3-a456-426614174025', 'AMC'),
  ('123e4567-e89b-12d3-a456-426614174025', 'BBBY'),
  ('123e4567-e89b-12d3-a456-426614174025', 'NOK'),
  
  -- Blue Chip
  ('123e4567-e89b-12d3-a456-426614174026', 'AAPL'),
  ('123e4567-e89b-12d3-a456-426614174026', 'MSFT'),
  ('123e4567-e89b-12d3-a456-426614174026', 'JNJ'),
  ('123e4567-e89b-12d3-a456-426614174026', 'KO')
ON CONFLICT (watchlist_id, symbol) DO NOTHING;

-- Insert more test alerts with various conditions
INSERT INTO alerts (id, user_id, symbol, type, target_value, metadata) VALUES 
  ('123e4567-e89b-12d3-a456-426614174034', '123e4567-e89b-12d3-a456-426614174002', 'NVDA', 'price_above', 500.00, '{"note": "Breakout alert", "strategy": "momentum"}'),
  ('123e4567-e89b-12d3-a456-426614174035', '123e4567-e89b-12d3-a456-426614174002', 'AMD', 'price_below', 100.00, '{"note": "Buy the dip", "strategy": "value"}'),
  ('123e4567-e89b-12d3-a456-426614174036', '123e4567-e89b-12d3-a456-426614174003', 'BTC-USD', 'price_change', 10.0, '{"note": "Bitcoin volatility alert", "period": "1h"}'),
  ('123e4567-e89b-12d3-a456-426614174037', '123e4567-e89b-12d3-a456-426614174003', 'GME', 'volume_spike', 3.0, '{"note": "Meme stock volume spike", "threshold": "3x"}'),
  ('123e4567-e89b-12d3-a456-426614174038', '123e4567-e89b-12d3-a456-426614174004', 'VTI', 'technical_signal', 0.0, '{"note": "RSI oversold signal", "indicator": "rsi", "threshold": 30}')
ON CONFLICT (id) DO NOTHING;

-- Insert more sample candle data for development
INSERT INTO candles (symbol, ts, open, high, low, close, volume) VALUES 
  -- NVDA data
  ('NVDA', '2024-01-02 14:30:00+00', 495.22, 498.67, 494.18, 497.45, 28456789),
  ('NVDA', '2024-01-02 14:31:00+00', 497.45, 499.23, 496.78, 498.12, 15234567),
  ('NVDA', '2024-01-02 14:32:00+00', 498.12, 500.45, 497.89, 499.78, 12876543),
  
  -- AMD data
  ('AMD', '2024-01-02 14:30:00+00', 147.23, 148.95, 146.89, 148.12, 18765432),
  ('AMD', '2024-01-02 14:31:00+00', 148.12, 148.78, 147.45, 147.89, 9876543),
  ('AMD', '2024-01-02 14:32:00+00', 147.89, 149.12, 147.23, 148.67, 8765432),
  
  -- BTC-USD data
  ('BTC-USD', '2024-01-02 14:30:00+00', 42567.89, 43123.45, 42234.12, 42890.34, 567890123),
  ('BTC-USD', '2024-01-02 14:31:00+00', 42890.34, 43234.56, 42678.90, 43012.45, 456789012),
  ('BTC-USD', '2024-01-02 14:32:00+00', 43012.45, 43345.67, 42789.12, 43156.78, 398765432)
ON CONFLICT (symbol, ts) DO NOTHING;

-- Insert more cache entries for testing
INSERT INTO technical_analysis_cache (symbol, period, analysis_data, expires_at) VALUES 
  ('NVDA', '1d', '{"sma_20": 492.34, "sma_50": 478.91, "rsi_14": 76.2, "macd": {"macd": 8.45, "signal": 6.78, "histogram": 1.67}, "signal": "STRONG_BUY"}', NOW() + INTERVAL '1 hour'),
  ('AMD', '1d', '{"sma_20": 145.67, "sma_50": 142.34, "rsi_14": 68.9, "macd": {"macd": 3.45, "signal": 2.89, "histogram": 0.56}, "signal": "BUY"}', NOW() + INTERVAL '1 hour'),
  ('BTC-USD', '1h', '{"sma_20": 42890.45, "sma_50": 41234.67, "rsi_14": 82.1, "macd": {"macd": 456.78, "signal": 389.45, "histogram": 67.33}, "signal": "OVERBOUGHT"}', NOW() + INTERVAL '30 minutes')
ON CONFLICT (symbol, period) DO NOTHING;

INSERT INTO forecast_cache (symbol, model_type, horizon, forecast_data, expires_at) VALUES 
  ('NVDA', 'ensemble', 5, '{"predictions": [502.34, 506.78, 498.45, 512.23, 518.90], "confidence": 0.73, "direction": "UP", "volatility": 0.04}', NOW() + INTERVAL '4 hours'),
  ('AMD', 'xgboost', 5, '{"predictions": [149.45, 151.23, 148.90, 152.67, 155.12], "confidence": 0.68, "direction": "UP", "volatility": 0.03}', NOW() + INTERVAL '4 hours'),
  ('BTC-USD', 'lstm', 3, '{"predictions": [43567.89, 44123.45, 42890.12], "confidence": 0.52, "direction": "SIDEWAYS", "volatility": 0.08}', NOW() + INTERVAL '2 hours')
ON CONFLICT (symbol, model_type, horizon) DO NOTHING;