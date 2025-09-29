-- Event Data Service Tables
-- Creates tables for scheduled events, headlines, and surprise scoring

-- Create event type enum
CREATE TYPE event_type AS ENUM (
    'earnings',
    'product_launch',
    'analyst_day',
    'regulatory_decision',
    'merger_acquisition',
    'ipo',
    'conference_call',
    'guidance_update',
    'dividend_announcement',
    'stock_split',
    'fda_approval',
    'clinical_trial'
);

-- Create impact level enum
CREATE TYPE impact_level AS ENUM ('low', 'medium', 'high');

-- Create event status enum
CREATE TYPE event_status AS ENUM ('scheduled', 'in_progress', 'completed', 'cancelled');

-- Create headline type enum
CREATE TYPE headline_type AS ENUM ('breaking', 'earnings', 'merger', 'regulatory', 'analyst', 'general');

-- Scheduled events table
CREATE TABLE scheduled_events (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    event_type event_type NOT NULL,
    event_date DATE NOT NULL,
    event_time TIMESTAMPTZ,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    impact_level impact_level DEFAULT 'medium',
    status event_status DEFAULT 'scheduled',
    source VARCHAR(100),
    source_url VARCHAR(1000),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Event-specific data
    metadata JSONB DEFAULT '{}',
    
    -- Outcome tracking
    actual_outcome TEXT,
    outcome_timestamp TIMESTAMPTZ,
    surprise_score NUMERIC(5,4)
);

-- Headline captures table
CREATE TABLE headline_captures (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    headline VARCHAR(1000) NOT NULL,
    source VARCHAR(100) NOT NULL,
    source_url VARCHAR(1000),
    published_at TIMESTAMPTZ NOT NULL,
    captured_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Content analysis
    headline_type headline_type DEFAULT 'general',
    relevance_score NUMERIC(3,2),
    sentiment_score NUMERIC(3,2),
    urgency_score NUMERIC(3,2),
    
    -- Full content (optional)
    article_content TEXT,
    content_summary TEXT,
    
    -- Processing metadata
    keywords JSONB,
    entities JSONB,
    metadata JSONB DEFAULT '{}'
);

-- Event outcomes table
CREATE TABLE event_outcomes (
    id SERIAL PRIMARY KEY,
    scheduled_event_id INTEGER NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    
    -- Outcome details
    actual_outcome TEXT NOT NULL,
    outcome_timestamp TIMESTAMPTZ NOT NULL,
    outcome_source VARCHAR(100),
    outcome_url VARCHAR(1000),
    
    -- Quantitative results
    actual_value NUMERIC(15,4),
    expected_value NUMERIC(15,4),
    surprise_percent NUMERIC(8,4),
    
    -- Market reaction
    price_impact_1h NUMERIC(8,4),
    price_impact_1d NUMERIC(8,4),
    volume_impact NUMERIC(8,4),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Surprise scores table
CREATE TABLE surprise_scores (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    event_type event_type NOT NULL,
    event_date DATE NOT NULL,
    
    -- Surprise calculation
    expected_value NUMERIC(15,4),
    actual_value NUMERIC(15,4),
    surprise_magnitude NUMERIC(8,4) NOT NULL,
    surprise_score NUMERIC(5,4) NOT NULL,
    
    -- Context
    historical_volatility NUMERIC(8,4),
    event_importance NUMERIC(3,2),
    market_context VARCHAR(50),
    
    -- Calculation metadata
    calculation_method VARCHAR(100),
    confidence_level NUMERIC(3,2),
    data_sources JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for performance
CREATE INDEX idx_scheduled_events_symbol ON scheduled_events(symbol);
CREATE INDEX idx_scheduled_events_date ON scheduled_events(event_date);
CREATE INDEX idx_scheduled_events_type ON scheduled_events(event_type);
CREATE INDEX idx_scheduled_events_status ON scheduled_events(status);
CREATE INDEX idx_scheduled_events_impact ON scheduled_events(impact_level);

CREATE INDEX idx_headline_captures_symbol ON headline_captures(symbol);
CREATE INDEX idx_headline_captures_published ON headline_captures(published_at DESC);
CREATE INDEX idx_headline_captures_captured ON headline_captures(captured_at DESC);
CREATE INDEX idx_headline_captures_relevance ON headline_captures(relevance_score DESC);
CREATE INDEX idx_headline_captures_type ON headline_captures(headline_type);

CREATE INDEX idx_event_outcomes_scheduled_event ON event_outcomes(scheduled_event_id);
CREATE INDEX idx_event_outcomes_symbol ON event_outcomes(symbol);
CREATE INDEX idx_event_outcomes_timestamp ON event_outcomes(outcome_timestamp DESC);

CREATE INDEX idx_surprise_scores_symbol ON surprise_scores(symbol);
CREATE INDEX idx_surprise_scores_date ON surprise_scores(event_date DESC);
CREATE INDEX idx_surprise_scores_type ON surprise_scores(event_type);
CREATE INDEX idx_surprise_scores_score ON surprise_scores(surprise_score DESC);

-- Create updated_at trigger for scheduled_events
CREATE TRIGGER update_scheduled_events_updated_at 
    BEFORE UPDATE ON scheduled_events 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Foreign key constraint
ALTER TABLE event_outcomes 
    ADD CONSTRAINT fk_event_outcomes_scheduled_event 
    FOREIGN KEY (scheduled_event_id) REFERENCES scheduled_events(id) ON DELETE CASCADE;

-- Composite indexes for common queries
CREATE INDEX idx_scheduled_events_symbol_date ON scheduled_events(symbol, event_date);
CREATE INDEX idx_headline_captures_symbol_published ON headline_captures(symbol, published_at DESC);
CREATE INDEX idx_surprise_scores_symbol_date ON surprise_scores(symbol, event_date DESC);