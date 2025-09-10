-- Test database connection and verify schema
-- Run this with: psql "postgresql://trading_user:trading_pass@localhost:5432/trading_db" -f test-db-connection.sql

-- Test basic connection
SELECT 'Database connection successful!' as status;

-- Show all tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Show all custom types
SELECT typname 
FROM pg_type 
WHERE typtype = 'e' 
ORDER BY typname;

-- Test data counts
SELECT 'users' as table_name, COUNT(*) as row_count FROM users
UNION ALL
SELECT 'portfolios', COUNT(*) FROM portfolios
UNION ALL
SELECT 'alerts', COUNT(*) FROM alerts
UNION ALL
SELECT 'candles', COUNT(*) FROM candles
ORDER BY table_name;

-- Test a sample query
SELECT 
  u.email,
  p.name as portfolio_name,
  COUNT(pp.id) as positions_count
FROM users u
LEFT JOIN portfolios p ON u.id = p.user_id
LEFT JOIN portfolio_positions pp ON p.id = pp.portfolio_id
GROUP BY u.email, p.name
ORDER BY u.email;

SELECT 'Schema test completed successfully!' as final_status;