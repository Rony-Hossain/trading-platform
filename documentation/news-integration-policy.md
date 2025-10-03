# News Integration Policy
Last updated: 2025-10-03

## Overview

News integration provides contextual market information to enhance trading decisions. This document defines contracts, sourcing policies, and implementation guidelines.

## News Contracts

### Data Model

Defined in `lib/types/contracts.ts`:

```typescript
export interface NewsItem {
  id: string
  symbols: string[]
  headline: string
  summary: string
  source: string
  url: string
  published_at: string // ISO 8601
  sentiment: 'positive' | 'negative' | 'neutral'
  importance: 'low' | 'medium' | 'high' | 'critical'
  event_type: 'earnings' | 'analyst' | 'regulatory' | 'market' | 'general'
}

export interface NewsFeedParams {
  symbols?: string[]
  lookback_hours?: number
  max_items?: number
  sources?: string[]
  language?: string
}

export interface NewsFeedResponse {
  metadata: ResponseMetadata
  items: NewsItem[]
  total_count: number
}
```

## Provider Allowlist

### Approved News Providers

| Provider | Type | Use Case | Rate Limit | Cost Tier |
|----------|------|----------|------------|-----------|
| **Benzinga** | Real-time | Breaking news, earnings | 1000 req/hour | Premium |
| **Alpha Vantage** | Market data | Company news, sentiment | 500 req/day (free tier) | Free/Pro |
| **Finnhub** | Company-specific | Filings, analyst ratings | 60 req/min | Free/Pro |
| **NewsAPI** | General | Broad market news | 1000 req/day (developer tier) | Free/Pro |
| **Polygon.io** | Market data | News + market data bundle | Included in subscription | Premium |

### Provider Selection Criteria

1. **Reliability**: 99.9% uptime SLA
2. **Latency**: < 2s average response time
3. **Data quality**: Accurate sentiment, proper categorization
4. **Licensing**: Commercial use permitted
5. **Coverage**: Broad symbol coverage (NYSE, NASDAQ minimum)

## Rate Limiting & Caching

### Provider-Specific Rate Limits

```typescript
// lib/news/rate-limits.ts
export const PROVIDER_RATE_LIMITS = {
  benzinga: {
    requests_per_hour: 1000,
    burst_limit: 50,
    retry_after_ms: 3600000, // 1 hour
  },
  alpha_vantage: {
    requests_per_day: 500,
    burst_limit: 5,
    retry_after_ms: 86400000, // 24 hours
  },
  finnhub: {
    requests_per_minute: 60,
    burst_limit: 10,
    retry_after_ms: 60000, // 1 minute
  },
  newsapi: {
    requests_per_day: 1000,
    burst_limit: 10,
    retry_after_ms: 86400000,
  },
  polygon: {
    requests_per_minute: 100, // Depends on tier
    burst_limit: 20,
    retry_after_ms: 60000,
  },
}
```

### Caching Strategy

**TTL by Content Type**:

| Content Type | TTL | Rationale |
|--------------|-----|-----------|
| Breaking news | 5 min | Fast-moving, needs freshness |
| Earnings reports | 1 hour | Important but static after release |
| Analyst ratings | 24 hours | Infrequent updates |
| General news | 15 min | Balance between freshness and load |
| Historical news | 7 days | Archival, rarely changes |

**Cache Implementation**:

```typescript
// lib/news/cache.ts
import { Redis } from 'ioredis'

const redis = new Redis(process.env.REDIS_URL)

export async function getCachedNews(
  cacheKey: string,
  ttlSeconds: number
): Promise<NewsItem[] | null> {
  const cached = await redis.get(cacheKey)
  if (cached) {
    return JSON.parse(cached)
  }
  return null
}

export async function setCachedNews(
  cacheKey: string,
  items: NewsItem[],
  ttlSeconds: number
): Promise<void> {
  await redis.setex(cacheKey, ttlSeconds, JSON.stringify(items))
}

// Generate cache key
export function getNewsCacheKey(params: NewsFeedParams): string {
  const { symbols = [], lookback_hours = 24, sources = [] } = params
  return `news:${symbols.sort().join(',')}:${lookback_hours}:${sources.sort().join(',')}`
}
```

## Attribution Requirements

### Display Requirements

All news items must display:

1. **Source attribution**: Provider name clearly visible
2. **Timestamp**: "Published X minutes ago" or absolute time
3. **Link**: Direct link to original article (if available)
4. **Logo/branding**: Provider logo when required by terms

### Legal Compliance

```tsx
// components/news/NewsItem.tsx
<article className="news-item">
  <h3>{newsItem.headline}</h3>
  <p>{newsItem.summary}</p>

  <footer className="news-meta">
    {/* Required attribution */}
    <span className="source">
      {providerLogo(newsItem.source)}
      {newsItem.source}
    </span>

    <time dateTime={newsItem.published_at}>
      {formatRelativeTime(newsItem.published_at)}
    </time>

    {/* Link to original if available */}
    {newsItem.url && (
      <a
        href={newsItem.url}
        target="_blank"
        rel="noopener noreferrer"
        className="read-more"
      >
        Read full article
      </a>
    )}
  </footer>

  {/* Provider-specific terms notice */}
  {providerRequiresNotice(newsItem.source) && (
    <small className="provider-notice">
      News provided by {newsItem.source}. See their{' '}
      <a href={getProviderTermsUrl(newsItem.source)}>terms of service</a>.
    </small>
  )}
</article>
```

### Provider-Specific Terms

```typescript
// lib/news/provider-terms.ts
export const PROVIDER_TERMS = {
  benzinga: {
    requires_notice: true,
    notice_text: 'News provided by Benzinga',
    terms_url: 'https://www.benzinga.com/terms-of-service',
    logo_required: true,
  },
  alpha_vantage: {
    requires_notice: false,
    logo_required: false,
  },
  finnhub: {
    requires_notice: true,
    notice_text: 'Powered by Finnhub',
    terms_url: 'https://finnhub.io/terms-of-service',
    logo_required: false,
  },
  newsapi: {
    requires_notice: false,
    logo_required: false,
  },
  polygon: {
    requires_notice: true,
    notice_text: 'Data provided by Polygon.io',
    terms_url: 'https://polygon.io/terms',
    logo_required: true,
  },
}
```

## Fallback Behavior

### Multi-Provider Strategy

Implement fallback chain to ensure reliability:

```typescript
// lib/news/fetcher.ts
const PROVIDER_PRIORITY = ['benzinga', 'polygon', 'finnhub', 'newsapi', 'alpha_vantage']

export async function fetchNews(params: NewsFeedParams): Promise<NewsFeedResponse> {
  // Check cache first
  const cacheKey = getNewsCacheKey(params)
  const cached = await getCachedNews(cacheKey, 300) // 5 min TTL

  if (cached) {
    return {
      metadata: { /* ... */ },
      items: cached,
      total_count: cached.length,
    }
  }

  // Try providers in priority order
  for (const provider of PROVIDER_PRIORITY) {
    try {
      // Check rate limit
      if (await isRateLimited(provider)) {
        logger.warn(`Provider ${provider} is rate-limited, trying next`)
        continue
      }

      const items = await fetchFromProvider(provider, params)

      // Cache successful response
      await setCachedNews(cacheKey, items, 300)

      return {
        metadata: {
          request_id: ulid(),
          generated_at: new Date().toISOString(),
          version: 'news.v1',
          latency_ms: /* ... */,
          source_models: [{ name: provider, version: '1.0', sha: '' }],
        },
        items,
        total_count: items.length,
      }
    } catch (error) {
      logger.error(`Provider ${provider} failed`, { error, params })
      // Continue to next provider
    }
  }

  // All providers failed
  throw new Error('All news providers unavailable')
}
```

### Degraded Mode

When all providers fail, show cached data with warning:

```tsx
// components/news/NewsFeed.tsx
const { data, error, isStale } = useNewsQuery(params)

if (error && !data) {
  return <NewsUnavailableState />
}

return (
  <div>
    {isStale && (
      <Banner type="warning">
        News may be outdated. Last updated {formatTime(data.metadata.generated_at)}
      </Banner>
    )}

    {data.items.map((item) => (
      <NewsItem key={item.id} item={item} />
    ))}
  </div>
)
```

## Sentiment Analysis

### Sentiment Validation

Validate provider sentiment against internal model (optional):

```typescript
// lib/news/sentiment.ts
import { analyzeSentiment } from '@/lib/ml/sentiment-model'

export async function enrichWithSentiment(item: NewsItem): Promise<NewsItem> {
  // Use provider sentiment as baseline
  let sentiment = item.sentiment

  // Optionally validate with internal model
  if (process.env.ENABLE_SENTIMENT_VALIDATION === 'true') {
    const analyzed = await analyzeSentiment(item.headline + ' ' + item.summary)

    // If provider and internal model disagree significantly, flag for review
    if (analyzed.confidence > 0.8 && analyzed.sentiment !== sentiment) {
      logger.warn('Sentiment mismatch', {
        item_id: item.id,
        provider_sentiment: sentiment,
        model_sentiment: analyzed.sentiment,
      })

      // Use internal model if high confidence
      sentiment = analyzed.sentiment
    }
  }

  return { ...item, sentiment }
}
```

### Display Guidelines

```tsx
// Display sentiment with icon
const SentimentBadge = ({ sentiment }: { sentiment: NewsSentiment }) => {
  const config = {
    positive: { icon: '↑', color: 'green', label: 'Positive' },
    negative: { icon: '↓', color: 'red', label: 'Negative' },
    neutral: { icon: '→', color: 'gray', label: 'Neutral' },
  }

  const { icon, color, label } = config[sentiment]

  return (
    <span className={`sentiment sentiment-${color}`} aria-label={`Sentiment: ${label}`}>
      {icon} {label}
    </span>
  )
}
```

## Content Filtering

### Relevance Scoring

Filter news by relevance to user portfolio:

```typescript
// lib/news/relevance.ts
export function scoreRelevance(item: NewsItem, userContext: {
  portfolio: string[]
  watchlist: string[]
  recentSearches: string[]
}): number {
  let score = 0

  // High relevance: user owns the stock
  if (item.symbols.some((s) => userContext.portfolio.includes(s))) {
    score += 10
  }

  // Medium relevance: on watchlist
  if (item.symbols.some((s) => userContext.watchlist.includes(s))) {
    score += 5
  }

  // Low relevance: recently searched
  if (item.symbols.some((s) => userContext.recentSearches.includes(s))) {
    score += 2
  }

  // Boost for high importance
  const importanceBoost = {
    critical: 5,
    high: 3,
    medium: 1,
    low: 0,
  }
  score += importanceBoost[item.importance]

  // Boost for recent news
  const ageHours = (Date.now() - new Date(item.published_at).getTime()) / (1000 * 60 * 60)
  if (ageHours < 1) score += 3
  else if (ageHours < 6) score += 2
  else if (ageHours < 24) score += 1

  return score
}

export function filterByRelevance(
  items: NewsItem[],
  userContext: any,
  threshold: number = 5
): NewsItem[] {
  return items
    .map((item) => ({ item, score: scoreRelevance(item, userContext) }))
    .filter(({ score }) => score >= threshold)
    .sort((a, b) => b.score - a.score)
    .map(({ item }) => item)
}
```

### Content Safety

Filter out inappropriate or low-quality content:

```typescript
// lib/news/safety.ts
const BLOCKLIST = [
  // Spam indicators
  /buy now/i,
  /limited time/i,
  /click here/i,

  // Low quality sources (update as needed)
  /promotional/i,
]

export function isSafeContent(item: NewsItem): boolean {
  const text = `${item.headline} ${item.summary}`.toLowerCase()

  // Check blocklist
  if (BLOCKLIST.some((pattern) => pattern.test(text))) {
    return false
  }

  // Minimum content length
  if (item.summary.length < 50) {
    return false
  }

  // Must have valid source
  if (!item.source || item.source === 'unknown') {
    return false
  }

  return true
}
```

## User Personalization

### News Preferences

Allow users to customize news feed:

```typescript
// lib/news/preferences.ts
export interface NewsPreferences {
  sources: string[] // Preferred providers
  event_types: NewsEventType[]
  min_importance: NewsImportance
  language: string
  hide_read: boolean
}

export function applyPreferences(
  items: NewsItem[],
  prefs: NewsPreferences
): NewsItem[] {
  return items.filter((item) => {
    // Filter by source
    if (prefs.sources.length > 0 && !prefs.sources.includes(item.source)) {
      return false
    }

    // Filter by event type
    if (prefs.event_types.length > 0 && !prefs.event_types.includes(item.event_type)) {
      return false
    }

    // Filter by importance
    const importanceLevel = ['low', 'medium', 'high', 'critical']
    if (importanceLevel.indexOf(item.importance) < importanceLevel.indexOf(prefs.min_importance)) {
      return false
    }

    return true
  })
}
```

### Quick Actions

```tsx
// User actions on news items
<NewsItem item={item}>
  <QuickActions>
    <button onClick={() => markAsRead(item.id)}>Mark as read</button>
    <button onClick={() => hideSource(item.source)}>Hide from {item.source}</button>
    <button onClick={() => addSymbolToWatchlist(item.symbols[0])}>Add to watchlist</button>
    <button onClick={() => shareNews(item)}>Share</button>
  </QuickActions>
</NewsItem>
```

## Implementation Checklist

- [ ] Set up provider API keys and rate limit tracking
- [ ] Implement Redis caching layer
- [ ] Create news fetcher with fallback chain
- [ ] Add attribution components for each provider
- [ ] Implement relevance scoring algorithm
- [ ] Add content safety filters
- [ ] Create user preference controls
- [ ] Add telemetry for news engagement (clicks, reads, helpfulness)
- [ ] Test fallback behavior when providers fail
- [ ] Document provider terms compliance in legal review
