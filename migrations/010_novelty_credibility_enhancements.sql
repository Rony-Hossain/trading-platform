-- Enhanced Sentiment Data Quality Migration
-- Adds novelty scores and source credibility weights to prevent double-counting

-- Add novelty and credibility columns to sentiment_posts
ALTER TABLE sentiment_posts ADD COLUMN IF NOT EXISTS novelty_score FLOAT DEFAULT 1.0 CHECK (novelty_score >= 0.0 AND novelty_score <= 1.0);
ALTER TABLE sentiment_posts ADD COLUMN IF NOT EXISTS source_credibility_weight FLOAT DEFAULT 1.0 CHECK (source_credibility_weight >= 0.0 AND source_credibility_weight <= 2.0);
ALTER TABLE sentiment_posts ADD COLUMN IF NOT EXISTS author_credibility_weight FLOAT DEFAULT 1.0 CHECK (author_credibility_weight >= 0.0 AND author_credibility_weight <= 2.0);
ALTER TABLE sentiment_posts ADD COLUMN IF NOT EXISTS engagement_weight FLOAT DEFAULT 1.0 CHECK (engagement_weight >= 0.0 AND engagement_weight <= 2.0);
ALTER TABLE sentiment_posts ADD COLUMN IF NOT EXISTS duplicate_risk VARCHAR(20) CHECK (duplicate_risk IN ('none', 'low_similarity', 'moderate_similarity', 'high_similarity', 'exact_duplicate'));
ALTER TABLE sentiment_posts ADD COLUMN IF NOT EXISTS content_hash VARCHAR(32);

-- Add novelty and credibility columns to sentiment_news
ALTER TABLE sentiment_news ADD COLUMN IF NOT EXISTS novelty_score FLOAT DEFAULT 1.0 CHECK (novelty_score >= 0.0 AND novelty_score <= 1.0);
ALTER TABLE sentiment_news ADD COLUMN IF NOT EXISTS source_credibility_weight FLOAT DEFAULT 1.0 CHECK (source_credibility_weight >= 0.0 AND source_credibility_weight <= 2.0);
ALTER TABLE sentiment_news ADD COLUMN IF NOT EXISTS author_credibility_weight FLOAT DEFAULT 1.0 CHECK (author_credibility_weight >= 0.0 AND author_credibility_weight <= 2.0);
ALTER TABLE sentiment_news ADD COLUMN IF NOT EXISTS engagement_weight FLOAT DEFAULT 1.0 CHECK (engagement_weight >= 0.0 AND engagement_weight <= 2.0);
ALTER TABLE sentiment_news ADD COLUMN IF NOT EXISTS duplicate_risk VARCHAR(20) CHECK (duplicate_risk IN ('none', 'low_similarity', 'moderate_similarity', 'high_similarity', 'exact_duplicate'));
ALTER TABLE sentiment_news ADD COLUMN IF NOT EXISTS content_hash VARCHAR(32);

-- Enhance sentiment_aggregates with quality-adjusted metrics
ALTER TABLE sentiment_aggregates ADD COLUMN IF NOT EXISTS weighted_avg_sentiment FLOAT;
ALTER TABLE sentiment_aggregates ADD COLUMN IF NOT EXISTS total_effective_weight FLOAT DEFAULT 0.0;
ALTER TABLE sentiment_aggregates ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 1.0 CHECK (quality_score >= 0.0 AND quality_score <= 1.0);
ALTER TABLE sentiment_aggregates ADD COLUMN IF NOT EXISTS novelty_distribution JSONB;
ALTER TABLE sentiment_aggregates ADD COLUMN IF NOT EXISTS credibility_distribution JSONB;
ALTER TABLE sentiment_aggregates ADD COLUMN IF NOT EXISTS duplicate_count INTEGER DEFAULT 0;

-- Create content deduplication table
CREATE TABLE IF NOT EXISTS content_deduplication (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(32) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    occurrence_count INTEGER DEFAULT 1,
    platforms JSONB, -- Array of platforms where this content appeared
    sources JSONB,   -- Array of sources
    
    -- Representative sample of the content
    representative_content TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Indexes
    UNIQUE(content_hash, symbol)
);

-- Create source credibility tracking table
CREATE TABLE IF NOT EXISTS source_credibility (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name VARCHAR(100) NOT NULL,
    platform VARCHAR(50) NOT NULL,
    
    -- Credibility metrics
    base_credibility_weight FLOAT DEFAULT 1.0 CHECK (base_credibility_weight >= 0.0 AND base_credibility_weight <= 2.0),
    historical_accuracy_score FLOAT, -- Track accuracy over time
    volume_consistency_score FLOAT,  -- Avoid sources that spam
    
    -- Performance tracking
    total_posts INTEGER DEFAULT 0,
    accurate_predictions INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    spam_flags INTEGER DEFAULT 0,
    
    -- Auto-adjustment settings
    auto_adjust_enabled BOOLEAN DEFAULT true,
    last_adjustment_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    UNIQUE(source_name, platform)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_sentiment_posts_novelty_score 
    ON sentiment_posts (novelty_score DESC) WHERE novelty_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sentiment_posts_content_hash 
    ON sentiment_posts (content_hash) WHERE content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sentiment_posts_duplicate_risk 
    ON sentiment_posts (duplicate_risk) WHERE duplicate_risk IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sentiment_news_novelty_score 
    ON sentiment_news (novelty_score DESC) WHERE novelty_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sentiment_news_content_hash 
    ON sentiment_news (content_hash) WHERE content_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sentiment_news_duplicate_risk 
    ON sentiment_news (duplicate_risk) WHERE duplicate_risk IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_content_dedup_hash_symbol 
    ON content_deduplication (content_hash, symbol);

CREATE INDEX IF NOT EXISTS idx_content_dedup_symbol_first_seen 
    ON content_deduplication (symbol, first_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_source_credibility_source_platform 
    ON source_credibility (source_name, platform);

CREATE INDEX IF NOT EXISTS idx_source_credibility_weight 
    ON source_credibility (base_credibility_weight DESC);

-- Create trigger to automatically update updated_at timestamp for new tables
CREATE TRIGGER update_content_dedup_updated_at 
    BEFORE UPDATE ON content_deduplication 
    FOR EACH ROW 
    EXECUTE PROCEDURE update_updated_at_column();

CREATE TRIGGER update_source_credibility_updated_at 
    BEFORE UPDATE ON source_credibility 
    FOR EACH ROW 
    EXECUTE PROCEDURE update_updated_at_column();

-- Create enhanced views for quality-weighted sentiment analysis
CREATE OR REPLACE VIEW weighted_sentiment_summary_view AS
SELECT 
    symbol,
    platform,
    DATE_TRUNC('hour', post_timestamp) as hour,
    
    -- Traditional metrics
    COUNT(*) as total_posts,
    AVG(sentiment_score) as avg_sentiment,
    COUNT(CASE WHEN sentiment_label = 'BULLISH' THEN 1 END) as bullish_count,
    COUNT(CASE WHEN sentiment_label = 'BEARISH' THEN 1 END) as bearish_count,
    COUNT(CASE WHEN sentiment_label = 'NEUTRAL' THEN 1 END) as neutral_count,
    
    -- Quality-weighted metrics
    SUM(COALESCE(novelty_score, 1.0) * COALESCE(source_credibility_weight, 1.0) * COALESCE(author_credibility_weight, 1.0) * COALESCE(engagement_weight, 1.0)) as total_effective_weight,
    SUM(sentiment_score * COALESCE(novelty_score, 1.0) * COALESCE(source_credibility_weight, 1.0) * COALESCE(author_credibility_weight, 1.0) * COALESCE(engagement_weight, 1.0)) / NULLIF(SUM(COALESCE(novelty_score, 1.0) * COALESCE(source_credibility_weight, 1.0) * COALESCE(author_credibility_weight, 1.0) * COALESCE(engagement_weight, 1.0)), 0) as weighted_avg_sentiment,
    
    -- Quality distribution
    AVG(COALESCE(novelty_score, 1.0)) as avg_novelty_score,
    AVG(COALESCE(source_credibility_weight, 1.0)) as avg_source_weight,
    COUNT(CASE WHEN duplicate_risk IN ('high_similarity', 'exact_duplicate') THEN 1 END) as potential_duplicates
    
FROM sentiment_posts 
WHERE sentiment_score IS NOT NULL 
    AND post_timestamp >= NOW() - INTERVAL '7 days'
GROUP BY symbol, platform, DATE_TRUNC('hour', post_timestamp)
HAVING COUNT(*) >= 3; -- Only include hours with at least 3 posts

-- Create view for duplicate content analysis
CREATE OR REPLACE VIEW duplicate_content_analysis_view AS
SELECT 
    cd.symbol,
    cd.content_hash,
    cd.occurrence_count,
    cd.first_seen_at,
    cd.last_seen_at,
    cd.platforms,
    cd.sources,
    cd.representative_content,
    
    -- Calculate impact of duplicates
    CASE 
        WHEN cd.occurrence_count > 10 THEN 'high_duplicate_risk'
        WHEN cd.occurrence_count > 5 THEN 'moderate_duplicate_risk'
        WHEN cd.occurrence_count > 2 THEN 'low_duplicate_risk'
        ELSE 'unique_content'
    END as duplicate_impact_level
    
FROM content_deduplication cd
WHERE cd.occurrence_count > 1
ORDER BY cd.occurrence_count DESC, cd.last_seen_at DESC;

-- Insert default source credibility weights
INSERT INTO source_credibility (source_name, platform, base_credibility_weight) VALUES
    ('reuters', 'news', 1.0),
    ('bloomberg', 'news', 1.0),
    ('wall_street_journal', 'news', 1.0),
    ('financial_times', 'news', 0.95),
    ('cnbc', 'news', 0.9),
    ('marketwatch', 'news', 0.85),
    ('seeking_alpha', 'news', 0.8),
    ('yahoo_finance', 'news', 0.8),
    ('google_finance', 'news', 0.8),
    ('twitter', 'social', 0.6),
    ('reddit', 'social', 0.55),
    ('stocktwits', 'social', 0.65),
    ('discord', 'social', 0.5)
ON CONFLICT (source_name, platform) DO NOTHING;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON content_deduplication TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON source_credibility TO trading_user;
GRANT SELECT ON weighted_sentiment_summary_view TO trading_user;
GRANT SELECT ON duplicate_content_analysis_view TO trading_user;

-- Comments
COMMENT ON TABLE content_deduplication IS 'Tracks duplicate content across platforms to enable novelty scoring';
COMMENT ON TABLE source_credibility IS 'Stores and tracks credibility weights for different sources';
COMMENT ON VIEW weighted_sentiment_summary_view IS 'Quality-weighted sentiment summaries that account for novelty and credibility';
COMMENT ON VIEW duplicate_content_analysis_view IS 'Analysis of duplicate content patterns across the platform';

COMMENT ON COLUMN sentiment_posts.novelty_score IS 'Content novelty score (0-1) based on similarity to recent content';
COMMENT ON COLUMN sentiment_posts.source_credibility_weight IS 'Source credibility weight (0-2) based on historical reliability';
COMMENT ON COLUMN sentiment_posts.duplicate_risk IS 'Risk level of content being duplicate/replicated';
COMMENT ON COLUMN sentiment_posts.content_hash IS 'Hash of normalized content for duplicate detection';