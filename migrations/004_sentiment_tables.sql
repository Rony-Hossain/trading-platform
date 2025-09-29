-- Sentiment Service Database Tables
-- Stores social media posts, news articles, and sentiment analysis results

-- Create sentiment posts table
CREATE TABLE IF NOT EXISTS sentiment_posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(50) NOT NULL,
    platform_post_id VARCHAR(255) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    author VARCHAR(255),
    content TEXT NOT NULL,
    url TEXT,
    
    -- Sentiment analysis results
    sentiment_score FLOAT CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0),
    sentiment_label VARCHAR(20) CHECK (sentiment_label IN ('BULLISH', 'BEARISH', 'NEUTRAL')),
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Engagement metrics (JSON)
    engagement JSONB,
    
    -- Analysis metadata (JSON)
    metadata JSONB,
    
    -- Timestamps
    post_timestamp TIMESTAMPTZ NOT NULL,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    analyzed_at TIMESTAMPTZ,
    
    -- Ensure unique posts per platform
    UNIQUE(platform, platform_post_id)
);

-- Create sentiment news table
CREATE TABLE IF NOT EXISTS sentiment_news (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(100) NOT NULL,
    article_id VARCHAR(255) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    
    title TEXT NOT NULL,
    content TEXT,
    author VARCHAR(255),
    url TEXT NOT NULL,
    
    -- Sentiment analysis results
    sentiment_score FLOAT CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0),
    sentiment_label VARCHAR(20) CHECK (sentiment_label IN ('BULLISH', 'BEARISH', 'NEUTRAL')),
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Article relevance
    relevance_score FLOAT CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0),
    
    -- Analysis metadata (JSON)
    metadata JSONB,
    
    -- Timestamps
    published_at TIMESTAMPTZ NOT NULL,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    analyzed_at TIMESTAMPTZ,
    
    -- Ensure unique articles per source
    UNIQUE(source, article_id)
);

-- Create sentiment aggregates table
CREATE TABLE IF NOT EXISTS sentiment_aggregates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL CHECK (timeframe IN ('1h', '1d', '1w', '1m')),
    bucket_start TIMESTAMPTZ NOT NULL,
    
    -- Aggregate metrics
    avg_sentiment FLOAT,
    total_mentions INTEGER DEFAULT 0,
    bullish_count INTEGER DEFAULT 0,
    bearish_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,
    
    -- Platform breakdown (JSON)
    platform_breakdown JSONB,
    
    -- Engagement totals
    total_engagement BIGINT DEFAULT 0,
    
    -- Computation timestamp
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Ensure unique buckets per symbol/timeframe
    UNIQUE(symbol, timeframe, bucket_start)
);

-- Create collection status table
CREATE TABLE IF NOT EXISTS collection_status (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    
    -- Collection metrics
    last_collection_at TIMESTAMPTZ,
    posts_collected INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    last_error TEXT,
    
    -- API status
    rate_limit_remaining INTEGER,
    rate_limit_total INTEGER,
    rate_limit_reset_at TIMESTAMPTZ,
    
    -- Health status
    is_healthy BOOLEAN DEFAULT true,
    health_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Ensure unique status per platform/symbol
    UNIQUE(platform, symbol)
);

-- Create indexes for performance
-- Sentiment posts indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_posts_symbol_timestamp 
    ON sentiment_posts (symbol, post_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_posts_platform_timestamp 
    ON sentiment_posts (platform, post_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_posts_score_symbol 
    ON sentiment_posts (sentiment_score, symbol) WHERE sentiment_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sentiment_posts_collected_at 
    ON sentiment_posts (collected_at DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_posts_platform_symbol 
    ON sentiment_posts (platform, symbol);

-- Sentiment news indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_news_symbol_published 
    ON sentiment_news (symbol, published_at DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_news_source_published 
    ON sentiment_news (source, published_at DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_news_score_symbol 
    ON sentiment_news (sentiment_score, symbol) WHERE sentiment_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sentiment_news_relevance 
    ON sentiment_news (relevance_score DESC) WHERE relevance_score IS NOT NULL;

-- Sentiment aggregates indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_agg_symbol_timeframe_bucket 
    ON sentiment_aggregates (symbol, timeframe, bucket_start DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_agg_bucket_start 
    ON sentiment_aggregates (bucket_start DESC);

CREATE INDEX IF NOT EXISTS idx_sentiment_agg_computed_at 
    ON sentiment_aggregates (computed_at DESC);

-- Collection status indexes
CREATE INDEX IF NOT EXISTS idx_collection_status_platform_symbol 
    ON collection_status (platform, symbol);

CREATE INDEX IF NOT EXISTS idx_collection_status_updated_at 
    ON collection_status (updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_collection_status_healthy 
    ON collection_status (is_healthy) WHERE is_healthy = false;

-- Create trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_collection_status_updated_at 
    BEFORE UPDATE ON collection_status 
    FOR EACH ROW 
    EXECUTE PROCEDURE update_updated_at_column();

-- Convert tables to TimescaleDB hypertables if TimescaleDB is available
DO $$
BEGIN
    -- Check if TimescaleDB extension exists
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Convert sentiment_posts to hypertable
        PERFORM create_hypertable(
            'sentiment_posts', 
            'post_timestamp',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
        
        -- Convert sentiment_news to hypertable
        PERFORM create_hypertable(
            'sentiment_news',
            'published_at', 
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
        
        -- Convert sentiment_aggregates to hypertable
        PERFORM create_hypertable(
            'sentiment_aggregates',
            'bucket_start',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'TimescaleDB hypertables created for sentiment service';
        
        -- Set up retention policies
        PERFORM add_retention_policy(
            'sentiment_posts',
            INTERVAL '90 days',
            if_not_exists => TRUE
        );
        
        PERFORM add_retention_policy(
            'sentiment_news',
            INTERVAL '180 days',
            if_not_exists => TRUE
        );
        
        PERFORM add_retention_policy(
            'sentiment_aggregates',
            INTERVAL '5 years',
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'TimescaleDB retention policies set for sentiment service';
        
    ELSE
        RAISE NOTICE 'TimescaleDB extension not available, using regular tables';
    END IF;
END $$;

-- Create views for common queries
CREATE OR REPLACE VIEW sentiment_summary_view AS
SELECT 
    symbol,
    platform,
    DATE_TRUNC('hour', post_timestamp) as hour,
    COUNT(*) as total_posts,
    AVG(sentiment_score) as avg_sentiment,
    COUNT(CASE WHEN sentiment_label = 'BULLISH' THEN 1 END) as bullish_count,
    COUNT(CASE WHEN sentiment_label = 'BEARISH' THEN 1 END) as bearish_count,
    COUNT(CASE WHEN sentiment_label = 'NEUTRAL' THEN 1 END) as neutral_count,
    AVG(confidence) as avg_confidence
FROM sentiment_posts 
WHERE sentiment_score IS NOT NULL 
    AND post_timestamp >= NOW() - INTERVAL '7 days'
GROUP BY symbol, platform, DATE_TRUNC('hour', post_timestamp);

-- Create view for recent activity
CREATE OR REPLACE VIEW recent_sentiment_activity AS
SELECT 
    'post' as type,
    symbol,
    platform as source,
    content as text,
    sentiment_score,
    sentiment_label,
    confidence,
    post_timestamp as timestamp,
    author,
    url
FROM sentiment_posts 
WHERE post_timestamp >= NOW() - INTERVAL '24 hours'
    AND sentiment_score IS NOT NULL

UNION ALL

SELECT 
    'news' as type,
    symbol,
    source,
    title as text,
    sentiment_score,
    sentiment_label,
    confidence,
    published_at as timestamp,
    author,
    url
FROM sentiment_news 
WHERE published_at >= NOW() - INTERVAL '24 hours'
    AND sentiment_score IS NOT NULL

ORDER BY timestamp DESC;

-- Insert some sample data for testing (optional)
INSERT INTO collection_status (platform, symbol, is_healthy, health_message) VALUES
    ('twitter', 'AAPL', true, 'OK'),
    ('reddit', 'AAPL', true, 'OK'),
    ('twitter', 'TSLA', true, 'OK'),
    ('reddit', 'TSLA', true, 'OK')
ON CONFLICT (platform, symbol) DO NOTHING;

-- Grant permissions (adjust as needed for your setup)
GRANT SELECT, INSERT, UPDATE, DELETE ON sentiment_posts TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON sentiment_news TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON sentiment_aggregates TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON collection_status TO trading_user;
GRANT SELECT ON sentiment_summary_view TO trading_user;
GRANT SELECT ON recent_sentiment_activity TO trading_user;

-- Comment with usage information
COMMENT ON TABLE sentiment_posts IS 'Social media posts with sentiment analysis results';
COMMENT ON TABLE sentiment_news IS 'News articles with sentiment analysis results';
COMMENT ON TABLE sentiment_aggregates IS 'Pre-computed sentiment aggregates for fast querying';
COMMENT ON TABLE collection_status IS 'Track collection status and health per platform/symbol';
COMMENT ON VIEW sentiment_summary_view IS 'Hourly sentiment summaries by symbol and platform';
COMMENT ON VIEW recent_sentiment_activity IS 'Recent sentiment activity across all sources';