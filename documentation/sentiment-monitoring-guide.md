# Social Sentiment Monitoring Guide

## Overview

The trading platform includes comprehensive social sentiment monitoring that tracks financial discussions across multiple platforms to gauge market sentiment for individual stocks.

## Monitored Platforms

### 1. **Twitter/X** 
- **API**: Twitter API v2 with v1.1 fallback
- **Data Collected**: Tweets mentioning stock symbols, cashtags ($AAPL), financial keywords
- **Rate Limits**: 300 requests per 15-minute window (varies by endpoint)
- **Key Features**:
  - Real-time tweet streaming
  - Engagement metrics (likes, retweets, replies)
  - User verification status
  - Context annotations

**Setup Requirements**:
```env
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
TWITTER_BEARER_TOKEN=your_bearer_token
```

### 2. **Reddit**
- **API**: Reddit API (PRAW)
- **Subreddits Monitored**:
  - r/wallstreetbets
  - r/investing  
  - r/stocks
  - r/SecurityAnalysis
  - r/ValueInvesting
  - r/pennystocks
  - r/StockMarket
  - r/trading
- **Rate Limits**: 60 requests per minute
- **Key Features**:
  - Post titles and content analysis
  - Comment sentiment analysis
  - Upvote/downvote ratios
  - Post engagement metrics

**Setup Requirements**:
```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=TradingPlatform/1.0
```

### 3. **Threads (Meta)**
- **API**: Web scraping (no official API yet)
- **Alternative**: Instagram Basic Display API integration
- **Rate Limits**: Respectful scraping with delays
- **Key Features**:
  - Hashtag and symbol tracking
  - Thread engagement metrics
  - Cross-platform Meta content

**Setup Requirements**:
```env
INSTAGRAM_ACCESS_TOKEN=your_instagram_token  # Optional
```

### 4. **Truth Social**
- **API**: Mastodon-compatible API
- **Rate Limits**: Similar to Mastodon (300 requests per 5 minutes)
- **Key Features**:
  - "Truth" posts mentioning stocks
  - Engagement metrics (favourites, reblogs)
  - User influence tracking

**Setup Requirements**:
```env
TRUTH_SOCIAL_ACCESS_TOKEN=your_token
TRUTH_SOCIAL_CLIENT_ID=your_client_id
TRUTH_SOCIAL_CLIENT_SECRET=your_secret
```

### 5. **StockTwits**
- **API**: StockTwits Public API
- **Rate Limits**: 200 requests per hour (public)
- **Key Features**:
  - Symbol-specific message streams
  - Built-in sentiment labels (Bullish/Bearish)
  - User influence scores
  - Message engagement

**Setup**: No API key required for basic access

### 6. **Discord**
- **API**: Discord Bot API
- **Communities**: Financial Discord servers
- **Rate Limits**: 50 requests per second (bot)
- **Key Features**:
  - Real-time chat monitoring
  - Server-specific financial discussions
  - Message engagement tracking

**Setup Requirements**:
```env
DISCORD_BOT_TOKEN=your_bot_token
DISCORD_WEBHOOK_URL=your_webhook_url
```

### 7. **Telegram**
- **API**: Telegram Bot API
- **Channels**: Financial Telegram channels
- **Rate Limits**: 30 messages per second
- **Key Features**:
  - Channel message monitoring
  - Financial group discussions
  - Message forwarding tracking

**Setup Requirements**:
```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
```

### 8. **Bluesky Social**
- **API**: AT Protocol
- **Rate Limits**: TBD (new platform)
- **Key Features**:
  - Decentralized social posts
  - Financial discussions
  - Cross-platform content

**Setup Requirements**:
```env
BLUESKY_HANDLE=your_handle
BLUESKY_PASSWORD=your_password
```

### 9. **Financial News**
- **Sources**:
  - NewsAPI (Reuters, Bloomberg, CNBC, MarketWatch)
  - Finnhub News API  
  - Yahoo Finance News (free)
  - Alpha Vantage News (optional)
- **Key Features**:
  - Article sentiment analysis
  - Source credibility scoring
  - Publication timing analysis
  - Headline vs content sentiment comparison

**Setup Requirements**:
```env
NEWS_API_KEY=your_newsapi_key
FINNHUB_API_KEY=your_finnhub_key  
ALPHA_VANTAGE_API_KEY=your_alphavantage_key
```

## Sentiment Analysis Engine

### Multi-Model Approach
1. **VADER Sentiment**: Financial lexicon-enhanced
2. **TextBlob**: Polarity and subjectivity analysis  
3. **OpenAI GPT**: Context-aware financial sentiment (optional)

### Financial-Specific Features
- **Custom Lexicon**: 
  - Bullish terms: moon, rocket, lambo, hodl, calls, breakout
  - Bearish terms: crash, dump, puts, rekt, bear market
- **Emoji Analysis**: 🚀📈📉💎🦍🌙
- **Financial Pattern Recognition**: Price targets, timeframes, technical levels

### Sentiment Scoring
- **Range**: -1.0 (extremely bearish) to +1.0 (extremely bullish)
- **Labels**: BULLISH, BEARISH, NEUTRAL
- **Confidence**: 0.0 to 1.0 based on analysis consensus

## Data Collection Process

### Collection Schedule
- **Twitter**: Every 15 minutes
- **Reddit**: Every 30 minutes  
- **StockTwits**: Every 15 minutes
- **News**: Every 60 minutes

### Data Flow
1. **Collection**: Platform-specific collectors gather posts/articles
2. **Preprocessing**: Clean text, extract symbols, normalize content
3. **Analysis**: Multi-model sentiment analysis
4. **Storage**: Store in TimescaleDB with metadata
5. **Aggregation**: Calculate rolling averages and trends
6. **Alerting**: Trigger alerts on sentiment spikes/drops

## Monitoring & Observability

### Prometheus Metrics
```
# Collection metrics
sentiment_posts_collected_total{platform, symbol}
sentiment_collection_errors_total{platform, error_type}

# API health
sentiment_api_rate_limit_remaining{platform}
sentiment_platform_healthy{platform}

# Data quality  
sentiment_confidence_score{platform, symbol}
sentiment_average_score{symbol, timeframe}

# Business metrics
sentiment_symbol_mentions_total{symbol, platform}
sentiment_last_update_timestamp{platform}
```

### Grafana Dashboard
- **Collection Status**: Real-time platform health
- **Data Freshness**: Time since last update per platform
- **Rate Limits**: API quota usage tracking
- **Sentiment Trends**: Symbol sentiment over time
- **Volume Metrics**: Posts/mentions per symbol
- **Error Tracking**: Collection failures and API issues

### Alerting Rules
- Platform down > 1 minute
- Data stale > 30 minutes  
- High error rate > 10%
- Rate limit approaching 90%
- Sentiment spike/drop > 2 standard deviations

## API Endpoints

### Sentiment Data
```http
GET /sentiment/{symbol}?timeframe=1d&sources=twitter,reddit
GET /analysis/{symbol}?timeframe=1w  
GET /summary/{symbol}
```

### Collection Management
```http
POST /collect
GET /posts/{symbol}?source=twitter&hours=24
GET /news/{symbol}?hours=48
```

### Analytics
```http
GET /trends?symbols=AAPL,MSFT,TSLA&timeframe=1d
GET /compare?symbols=AAPL,MSFT&metrics=sentiment,volume
```

## Rate Limiting & Compliance

### Platform Limits
| Platform | Limit | Window | Notes |
|----------|-------|--------|-------|
| Twitter | 300 req | 15 min | Per endpoint |
| Reddit | 60 req | 1 min | Per app |
| StockTwits | 200 req | 1 hour | Public API |
| NewsAPI | 1000 req | 1 day | Free tier |

### Best Practices
- **Respectful Scraping**: Honor robots.txt and rate limits
- **Caching**: Cache responses to minimize API calls
- **Graceful Degradation**: Continue with available sources if one fails
- **Attribution**: Properly attribute data sources
- **ToS Compliance**: Follow each platform's terms of service

## Data Storage Schema

### Core Tables (TimescaleDB)
```sql
-- Social posts
CREATE TABLE sentiment_posts (
    id UUID PRIMARY KEY,
    platform TEXT NOT NULL,
    symbol TEXT NOT NULL,
    content TEXT NOT NULL,
    author TEXT,
    sentiment_score FLOAT,
    sentiment_label TEXT,
    confidence FLOAT,
    engagement JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL
);

-- Sentiment aggregates  
CREATE TABLE sentiment_hourly_agg (
    symbol TEXT,
    hour TIMESTAMPTZ,
    avg_sentiment FLOAT,
    mention_count INTEGER,
    platform_breakdown JSONB
);
```

### Retention Policies
- **Raw posts**: 90 days
- **Hourly aggregates**: 2 years  
- **Daily aggregates**: 5 years

## Configuration Examples

### Development Setup
```yaml
# docker-compose.yml addition
sentiment-service:
  build: ./services/sentiment-service
  ports:
    - "8005:8005"
  environment:
    - TWITTER_API_KEY=${TWITTER_API_KEY}
    - REDDIT_CLIENT_ID=${REDDIT_CLIENT_ID}
    - NEWS_API_KEY=${NEWS_API_KEY}
```

### Production Considerations
- **API Key Rotation**: Regularly rotate API keys
- **Load Balancing**: Multiple collector instances with Redis coordination
- **Backup Collection**: Secondary data sources for redundancy
- **Data Privacy**: Anonymize user data, respect privacy settings

## Troubleshooting

### Common Issues
1. **Rate Limit Exceeded**: 
   - Check API quotas in dashboard
   - Implement exponential backoff
   - Consider upgrading API plans

2. **Platform API Changes**:
   - Monitor platform developer blogs
   - Implement version-specific handlers
   - Graceful fallback to alternative sources

3. **Data Quality Issues**:
   - Review sentiment analysis confidence scores
   - Validate against known market events
   - Adjust financial lexicon based on new terms

### Monitoring Commands
```bash
# Check service health
curl http://localhost:8005/health

# View collection stats  
curl http://localhost:8005/stats

# Test sentiment analysis
curl -X POST http://localhost:8005/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "AAPL to the moon! 🚀", "symbol": "AAPL"}'
```

## Future Enhancements

### Planned Features
- **Discord Integration**: Gaming/crypto communities
- **Telegram Channels**: Financial discussion groups
- **YouTube Transcripts**: Financial influencer content
- **Earnings Call Transcripts**: Management sentiment analysis
- **Real-time Sentiment Alerts**: WebSocket/SSE streaming
- **ML-Enhanced Analysis**: Custom financial sentiment models

### Research Areas
- **Sentiment-Price Correlation**: Lead/lag analysis
- **Influencer Impact**: Weight by follower count/influence
- **Geographic Sentiment**: Regional market sentiment differences
- **Multi-language Support**: Non-English financial discussions