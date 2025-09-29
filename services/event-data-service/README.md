# Event Data Service Providers

This service ingests scheduled events and real-time headlines from external providers. To avoid API throttling and authentication issues, set the following environment variables (see .env.example).

## Scheduled Calendar Providers


### Provider Failover
- EVENT_CALENDAR_PROVIDERS_JSON: optional JSON array of provider definitions with 
ame, url, pi_key, headers, etc. Providers are attempted in order.
- EVENT_CALENDAR_PROVIDER_MAX_FAILURES: consecutive failures before backing off a provider (default 3).
- EVENT_CALENDAR_PROVIDER_FAILBACK_SECONDS: backoff duration in seconds after max failures (default 600).


### Data Quality Settings
- EVENT_CALENDAR_DEDUPE_WINDOW_MINUTES: time window for in-memory duplicate suppression (default 60).
- EVENT_CALENDAR_MAX_HORIZON_DAYS: discard events scheduled beyond this horizon (default 365).
- EVENT_CALENDAR_MIN_SYMBOL_LENGTH: minimum ticker length accepted (default 1).
- EVENT_CALENDAR_ALLOWED_CATEGORIES: optional comma-separated whitelist of categories (default accepts all).

| Variable | Description | Notes |
|----------|-------------|-------|
| EVENT_CALENDAR_URL | Base URL returning a JSON payload with upcoming events. | Endpoint must accept GET with optional query params defined by your provider. |
| EVENT_CALENDAR_API_KEY | API key/token for the calendar provider. | Injected as X-API-KEY header. Update code if provider requires a different field. |
| EVENT_CALENDAR_PROVIDER | Friendly name identifying the provider. | Stored in events.source to scope deduplication. |
| EVENT_CALENDAR_POLL_INTERVAL | Seconds between poll cycles (default 900). | Respect provider rate limits. | 

### Expected Payload Shape

The collector expects either:

`json
{
  "events": [
    {
      "symbol": "AAPL",
      "title": "Apple Q4 Earnings Call",
      "category": "earnings",
      "scheduled_at": "2025-10-28T21:00:00Z",
      "timezone": "America/New_York",
      "description": "Conference call details...",
      "id": "external-id-123"
    }
  ]
}
`

or a top-level array. Any additional fields are stored under metadata.

## Headline Providers

| Variable | Description | Notes |
|----------|-------------|-------|
| EVENT_HEADLINE_URL | Endpoint returning breaking headlines in JSON. | Must provide symbol, headline, and published_at. |
| EVENT_HEADLINE_API_KEY | API key/token for the headline feed. | Injected as X-API-KEY header by default. |
| EVENT_HEADLINE_PROVIDER | Friendly name for logging/metadata. | Persists in event_headlines.source. |
| EVENT_HEADLINE_POLL_INTERVAL | Poll interval in seconds (default 120). | Tune per provider policy. |
| EVENT_HEADLINE_MATCH_WINDOW_MINUTES | Minutes around event time to link headlines to scheduled events. | Default 120. |

### Headline Payload Example

`json
{
  "headlines": [
    {
      "symbol": "AAPL",
      "headline": "Apple beats on revenue, stock jumps",
      "published_at": "2025-10-28T21:05:00Z",
      "summary": "Breaking news summary",
      "url": "https://news.example.com/aapl",
      "id": "headline-789"
    }
  ]
}
`

## Rate Limit Guidance

- **Schedule feeds**: polling every 15 minutes (900 seconds) is typically safe for quarterly earnings calendars. Adjust according to provider SLAs.
- **Headline feeds**: high-frequency sources should be polled between 30�120 seconds. Check contractual rate limits and adjust EVENT_HEADLINE_POLL_INTERVAL accordingly.
- **Backoff**: When rates are tight, consider modifying the ingestors to honour provider-specific retry headers (e.g., Retry-After).

## Required Environment Variables

Ensure these keys are set (defaults are provided for development):

- EVENT_DB_URL
- EVENT_CALENDAR_URL, EVENT_CALENDAR_API_KEY, EVENT_CALENDAR_PROVIDER, EVENT_CALENDAR_POLL_INTERVAL
- EVENT_HEADLINE_URL, EVENT_HEADLINE_API_KEY, EVENT_HEADLINE_PROVIDER, EVENT_HEADLINE_POLL_INTERVAL, EVENT_HEADLINE_MATCH_WINDOW_MINUTES

Refer to .env.example for sample values. Update secrets before deploying to production.
## Impact Scoring

The ingestor now derives a market-moving potential score for every event when the provider does not supply one. The heuristic combines:
- Category priors (earnings, regulatory, product launch, etc.) with optional overrides via `EVENT_IMPACT_CATEGORY_BASE` (JSON map of category -> score).
- Company context such as market cap tiers and average daily volume when available in the payload.
- Expected move signals (`implied_move`, `expected_move_pct`, `historical_avg_move`) sourced from metadata or top-level keys.
- Qualitative flags (`is_major`, `importance`, `confidence`, `is_preliminary`).

The final score is clamped to a 1-10 scale. A breakdown is stored in `event.metadata.impact_analysis` so downstream services can audit the contributing factors.
## Feed Health Monitoring

Each provider poll is tracked for success or failure. The `/health` endpoint now includes a `feeds` dictionary, and `/health/feeds` exposes the detailed snapshot. Configuration:

- `EVENT_FEED_ALERT_THRESHOLD` (default `3`): consecutive failures before a feed is marked `down` and an alert is emitted.
- `EVENT_FEED_ALERT_WEBHOOK`: optional HTTP endpoint that receives JSON payloads when a feed degrades or recovers.
- `EVENT_FEED_ALERT_HEADERS`: optional JSON map of headers for the webhook requests (use for auth tokens).

When a provider remains in backoff it is marked `paused`; successful polls clear the failure counter and emit a recovery notification if a feed previously alerted.
## Webhook Notifications

Configure outbound webhooks to forward newly ingested events, manual event mutations, and incoming headlines to downstream services.

- `EVENT_WEBHOOK_TARGETS`: JSON array of targets `[{"url": "https://example.com/webhook", "headers": {"Authorization": "Bearer ..."}, "timeout": 3.0}]`.
- `EVENT_WEBHOOK_URL`: shorthand for a single endpoint (ignored when `EVENT_WEBHOOK_TARGETS` is supplied).
- `EVENT_WEBHOOK_HEADERS`: optional JSON object merged into the POST headers for the single-endpoint configuration.
- `EVENT_WEBHOOK_TIMEOUT`: request timeout in seconds (default 5).

Payload shape:

```json
{
  "type": "event.created",
  "timestamp": "2025-09-27T21:00:00Z",
  "data": { "id": "...", "symbol": "AAPL", ... }
}
```

Supported event types include `event.created`, `event.replaced`, `event.updated`, `event.deleted`, `event.impact_updated`, and `headline.created` (when a headline is linked to an event).
## Event Categorization

Incoming events are normalized into canonical buckets (earnings, fda_approval, mna, regulatory, product_launch, guidance, etc.).
- `EVENT_CATEGORY_OVERRIDES`: Optional JSON map `{ "custom_category": ["keyword1", "keyword2"] }` to extend keyword triggers.
- Each event stores the original provider category, canonical category, confidence score, and matched keywords under `metadata.classification`.
- Query `/events/categories` to inspect the current taxonomy.

## Event Clustering

The clustering engine groups related events across companies, sectors, and supply chains using configurable rules:

### Clustering Types
- **company_same_symbol**: Events from the same company within 24 hours
- **sector_earnings**: Earnings events from the same sector within 1 week  
- **regulatory_sector**: Regulatory/FDA events affecting the same sector within 72 hours
- **mna_wave**: M&A events in the same sector within 30 days
- **supply_chain**: Events affecting supply chain partners within 48 hours

### Configuration
- `EVENT_CLUSTERING_RULES`: Optional JSON array of custom clustering rules
- `EVENT_SECTOR_MAPPING`: JSON map of symbol to sector (e.g., `{"AAPL":"technology","JPM":"financials"}`)
- `EVENT_SUPPLY_CHAIN_RELATIONSHIPS`: JSON map of symbol to related symbols (e.g., `{"AAPL":["TSM","QCOM"]}`)

### API Endpoints
- `GET /events/clusters` - List clusters with optional time range and type filtering
- `GET /events/clusters/{cluster_id}` - Get specific cluster details
- `GET /events/clusters/symbol/{symbol}` - Get clusters involving a specific symbol
- `POST /events/clusters/analyze` - Generate clustering analysis with summary statistics

Each cluster includes cluster type, primary symbol, related symbols, event IDs, confidence score, and rich metadata about time spans, sectors, and categories.

## GraphQL API

The service provides a comprehensive GraphQL endpoint at `/graphql` for complex queries and relationships:

### Key Features
- **Complex Filtering**: Advanced event filtering with multiple criteria (symbols, categories, impact scores, time ranges, text search)
- **Relationship Queries**: Event relationship graphs with configurable relationship types and traversal depth
- **Nested Queries**: Fetch events with related headlines, clusters, and metadata in a single request
- **Real-time Clustering**: Query event clusters with filtering by type, symbols, scores, and time ranges
- **Mutations**: Create, update, and delete events with automatic categorization
- **Feed Health**: Monitor data feed status and health metrics

### Sample Queries

#### Complex Event Search
```graphql
query ComplexEventSearch {
  events(
    filter: {
      symbols: ["AAPL", "MSFT"]
      categories: ["earnings", "product_launch"]
      impact_score_min: 7
      start_time: "2025-01-01T00:00:00Z"
      search_text: "AI"
    }
    limit: 50
  ) {
    id
    symbol
    title
    category
    impact_score
    scheduled_at
    headlines {
      headline
      published_at
      url
    }
    clusters {
      cluster_type
      related_symbols
      cluster_score
    }
  }
}
```

#### Event Relationship Graph
```graphql
query EventRelationships {
  event_relationships(
    input: {
      event_id: "event-123"
      include_supply_chain: true
      include_sector: true
      time_window_hours: 168
      max_distance: 2
    }
  ) {
    central_event {
      symbol
      title
      category
    }
    related_events {
      symbol
      title
      scheduled_at
    }
    relationships {
      relationship_type
      strength
      distance
    }
    clusters {
      cluster_type
      cluster_score
    }
  }
}
```

#### Cluster Analysis
```graphql
query ClusterAnalysis {
  clusters(
    filter: {
      cluster_types: ["sector_earnings", "supply_chain"]
      min_score: 0.7
      symbols: ["AAPL"]
    }
  ) {
    cluster_id
    cluster_type
    primary_symbol
    related_symbols
    cluster_score
    events {
      symbol
      title
      impact_score
    }
  }
}
```

#### Create Event with Mutation
```graphql
mutation CreateEvent {
  create_event(
    input: {
      symbol: "AAPL"
      title: "Apple Q4 Earnings Call"
      category: "earnings"
      scheduled_at: "2025-01-28T21:00:00Z"
      impact_score: 8
      metadata: "{\"analyst_consensus\": 1.25}"
    }
  ) {
    success
    message
    event {
      id
      symbol
      category
      impact_score
    }
  }
}
```

The GraphQL endpoint supports introspection and includes a built-in playground for interactive query development.

## Event Search & Filtering API

The service provides a comprehensive search and filtering API at `/events/search` for advanced event queries:

### Key Features
- **Multi-Criteria Filtering**: Advanced filtering with multiple symbols, categories, statuses, sources, and impact scores
- **Date Range Filtering**: Filter by scheduled_at, created_at, or updated_at timestamps with from/to bounds
- **Text Search**: Full-text search across event title, description, and metadata fields (case-insensitive)
- **Relationship Filtering**: Filter events by presence of headlines, metadata, or external IDs
- **Clustering Integration**: Filter events by cluster membership and cluster types
- **Flexible Pagination**: Configurable limit/offset with multi-field sorting options
- **Response Customization**: Optional includes for headlines, clusters, and metadata to optimize response size

### Query Parameters

#### Filtering
- `symbols`: Comma-separated list of stock symbols (e.g., `AAPL,MSFT,GOOGL`)
- `categories`: Comma-separated list of event categories (e.g., `earnings,fda_approval,mna`)
- `statuses`: Comma-separated list of event statuses (e.g., `scheduled,occurred,cancelled`)
- `sources`: Comma-separated list of data sources (e.g., `provider1,provider2`)
- `impact_score_min`, `impact_score_max`, `impact_score`: Impact score filtering (1-10 scale)
- `scheduled_from`, `scheduled_to`: Date range for scheduled_at timestamp
- `created_from`, `created_to`: Date range for created_at timestamp
- `updated_from`, `updated_to`: Date range for updated_at timestamp
- `search_text`: Text search across title, description, and metadata
- `has_headlines`, `has_metadata`, `has_external_id`: Boolean relationship filters
- `in_clusters`: Filter events that are part of any cluster
- `cluster_types`: Comma-separated list of cluster types

#### Pagination & Sorting
- `limit`: Maximum number of results (default 50, max 1000)
- `offset`: Number of results to skip for pagination
- `sort_by`: Sorting field (`scheduled_at`, `created_at`, `impact_score`, `symbol`)
- `sort_order`: Sort direction (`asc` or `desc`, default `asc`)

#### Response Customization
- `include_headlines`: Include related headlines in response
- `include_clusters`: Include cluster information in response
- `include_metadata`: Include full metadata in response

### Sample Requests

#### Basic Symbol and Category Search
```bash
GET /events/search?symbols=AAPL,MSFT&categories=earnings&limit=20
```

#### High-Impact Events with Date Range
```bash
GET /events/search?impact_score_min=8&scheduled_from=2025-01-01&scheduled_to=2025-12-31&sort_by=impact_score&sort_order=desc
```

#### Text Search with Full Details
```bash
GET /events/search?search_text=AI&include_headlines=true&include_clusters=true&include_metadata=true
```

#### Cluster-Based Analysis
```bash
GET /events/search?in_clusters=true&cluster_types=sector_earnings,supply_chain&sort_by=scheduled_at
```

### Response Format

```json
{
  "events": [
    {
      "id": "event-123",
      "symbol": "AAPL",
      "title": "Apple Q4 Earnings Call",
      "category": "earnings",
      "impact_score": 8,
      "scheduled_at": "2025-01-28T21:00:00Z",
      "status": "scheduled",
      "source": "provider1",
      "headlines": [...],    // If include_headlines=true
      "clusters": [...],     // If include_clusters=true
      "metadata": {...}      // If include_metadata=true
    }
  ],
  "total_count": 150,
  "returned_count": 20,
  "has_more": true,
  "next_offset": 20
}
```

The search API is optimized for performance with selective joins and efficient database queries based on the requested include parameters.

## Event Subscription System

The service provides a real-time subscription system that allows strategy services and other consumers to receive immediate notifications when events occur, are updated, or change in impact. This enables event-driven trading strategies and real-time market analysis.

### Key Features
- **Real-time Notifications**: Instant webhook delivery when events match subscription criteria
- **Advanced Filtering**: Subscribe to specific symbols, categories, impact scores, and event types
- **Reliable Delivery**: Automatic retry logic with exponential backoff and failure tracking
- **Health Monitoring**: Monitor subscription health and delivery statistics
- **Concurrent Delivery**: Efficient async delivery to multiple subscribers
- **Event Lifecycle Tracking**: Notifications for event creation, updates, impact changes, and headline linking

### Subscription Management

#### Create Subscription
```bash
POST /subscriptions
```

**Request Body:**
```json
{
  "service_name": "strategy-service",
  "webhook_url": "https://your-service.com/webhooks/events",
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "categories": ["earnings", "fda_approval", "mna"],
  "min_impact_score": 7,
  "max_impact_score": 10,
  "event_types": ["event.created", "event.impact_changed"],
  "statuses": ["scheduled", "occurred"],
  "headers": {
    "Authorization": "Bearer your-token",
    "X-API-Key": "your-api-key"
  },
  "timeout": 5.0,
  "retry_count": 3,
  "retry_delay": 1.0
}
```

**Response:**
```json
{
  "id": "sub-123e4567-e89b-12d3-a456-426614174000",
  "service_name": "strategy-service",
  "webhook_url": "https://your-service.com/webhooks/events",
  "filters": {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "categories": ["earnings", "fda_approval", "mna"],
    "min_impact_score": 7,
    "event_types": ["event.created", "event.impact_changed"]
  },
  "status": "active",
  "created_at": "2025-01-28T10:00:00Z",
  "updated_at": "2025-01-28T10:00:00Z",
  "last_notification": null,
  "failure_count": 0
}
```

#### List Subscriptions
```bash
GET /subscriptions?service_name=strategy-service
```

#### Get Subscription Details
```bash
GET /subscriptions/{subscription_id}
```

#### Update Subscription Status
```bash
PATCH /subscriptions/{subscription_id}?status=paused
```

#### Delete Subscription
```bash
DELETE /subscriptions/{subscription_id}
```

#### Check Subscription Health
```bash
GET /subscriptions/{subscription_id}/health
```

### Event Types

The subscription system supports the following event types:

- **`event.created`**: New events ingested from providers
- **`event.updated`**: Events modified via API
- **`event.impact_changed`**: Event impact score changes
- **`event.status_changed`**: Event status updates (scheduled → occurred)
- **`headline.linked`**: New headlines linked to events
- **`cluster.formed`**: Events grouped into clusters
- **`*`** (wildcard): All event types

### Notification Payload

When subscriptions match events, the service sends HTTP POST requests to the configured webhook URL:

```json
{
  "subscription_id": "sub-123e4567-e89b-12d3-a456-426614174000",
  "event_type": "event.created",
  "timestamp": "2025-01-28T21:05:00Z",
  "service_name": "event-data-service",
  "data": {
    "id": "event-456",
    "symbol": "AAPL",
    "title": "Apple Q4 Earnings Call",
    "category": "earnings",
    "impact_score": 8,
    "scheduled_at": "2025-01-28T21:00:00Z",
    "status": "scheduled",
    "source": "provider1",
    "metadata": {
      "analyst_consensus": 1.25,
      "classification": {
        "confidence": 0.95,
        "matched_keywords": ["earnings", "quarterly"]
      }
    }
  }
}
```

### Filtering Options

Subscriptions support comprehensive filtering to ensure you only receive relevant notifications:

#### Symbol Filtering
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
}
```

#### Category Filtering
```json
{
  "categories": ["earnings", "fda_approval", "mna", "guidance", "regulatory"]
}
```

#### Impact Score Filtering
```json
{
  "min_impact_score": 7,
  "max_impact_score": 10
}
```

#### Event Type Filtering
```json
{
  "event_types": ["event.created", "event.impact_changed", "headline.linked"]
}
```

#### Status Filtering
```json
{
  "statuses": ["scheduled", "occurred", "cancelled"]
}
```

### Error Handling & Reliability

The subscription system includes robust error handling:

- **Automatic Retries**: Configurable retry attempts with exponential backoff
- **Failure Tracking**: Monitors consecutive failures per subscription
- **Status Management**: Automatically pauses subscriptions after max failures
- **Timeout Handling**: Configurable request timeouts to prevent hanging
- **Health Monitoring**: Real-time subscription health and delivery statistics

### Strategy Service Integration

The system is designed for seamless integration with trading strategy services:

```python
# Example strategy service integration
import httpx

async def subscribe_to_high_impact_events():
    subscription = {
        "service_name": "momentum-strategy",
        "webhook_url": "https://strategy-service.com/webhooks/events",
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "categories": ["earnings", "guidance"],
        "min_impact_score": 8,
        "event_types": ["event.created", "event.impact_changed"]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://event-service:8006/subscriptions",
            json=subscription
        )
        return response.json()

# Webhook handler in strategy service
@app.post("/webhooks/events")
async def handle_event_notification(request: Request):
    event_data = await request.json()
    
    if event_data["event_type"] == "event.created":
        # React to new high-impact event
        await prepare_trading_strategy(event_data["data"])
    
    return {"status": "processed"}
```

See `examples/strategy_service_client.py` for a complete working example of a strategy service that subscribes to events and reacts with trading logic.

### Configuration

Environment variables for subscription system:

- **Webhook Delivery**: Built into the subscription manager
- **Concurrency**: Async delivery to all active subscriptions
- **Memory Management**: In-memory subscription storage (consider Redis for production clustering)

The subscription system provides the foundation for building sophisticated event-driven trading strategies that can react in real-time to market-moving events.

## Real-time Event Enrichment

The service includes a comprehensive event enrichment system that automatically adds market context to events, enabling more sophisticated impact scoring and strategy targeting. Events are enriched with market cap, sector, volatility, and other financial metrics from multiple data sources.

### Key Features
- **Multi-Source Data**: Integrates Finnhub API, Yahoo Finance, and configurable sector mappings
- **Market Context**: Market cap, sector, industry, volatility, beta, average volume, and current price
- **Impact Modifiers**: Automatic impact score adjustments based on company characteristics
- **Performance Optimized**: 30-minute caching, batch processing, and async operations
- **Automatic Integration**: Events are enriched during ingestion and API creation
- **Graceful Fallbacks**: Continues processing even if enrichment fails

### Market Context Data

Each enriched event includes comprehensive market context:

```json
{
  "metadata": {
    "enrichment": {
      "market_context": {
        "symbol": "AAPL",
        "market_cap": 3000000000000,
        "market_cap_tier": "mega_cap",
        "sector": "technology",
        "industry": "consumer_electronics",
        "avg_volume": 89500000,
        "beta": 1.28,
        "volatility_30d": 24.5,
        "volatility_level": "moderate",
        "price": 185.42,
        "shares_outstanding": 16100000000,
        "last_updated": "2025-01-28T10:30:00Z"
      },
      "impact_modifiers": {
        "market_cap_modifier": 2.0,
        "volatility_modifier": 0.5,
        "beta_modifier": 0.5,
        "liquidity_modifier": 0.5
      },
      "enrichment_timestamp": "2025-01-28T10:30:00Z",
      "enrichment_version": "1.0"
    }
  }
}
```

### Market Cap Classification

Events are classified by market capitalization tier with automatic impact modifiers:

- **Mega-cap** (>$200B): +2.0 impact modifier
- **Large-cap** ($10B-$200B): +1.5 impact modifier  
- **Mid-cap** ($2B-$10B): +1.0 impact modifier
- **Small-cap** ($300M-$2B): +0.5 impact modifier
- **Micro-cap** (<$300M): -0.5 impact modifier

### Volatility Analysis

30-day volatility is calculated and classified into risk levels:

- **Very Low** (<10%): -0.5 impact modifier
- **Low** (10-20%): 0.0 impact modifier
- **Moderate** (20-35%): +0.5 impact modifier
- **High** (35-50%): +1.0 impact modifier
- **Very High** (>50%): +1.5 impact modifier

### API Endpoints

#### Get Market Context
```bash
GET /enrichment/market-context/{symbol}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "market_cap": 3000000000000,
  "market_cap_tier": "mega_cap",
  "sector": "technology",
  "volatility_30d": 24.5,
  "volatility_level": "moderate",
  "beta": 1.28,
  "price": 185.42
}
```

#### Enrich Single Event
```bash
POST /enrichment/enrich-event
```

**Request Body:**
```json
{
  "symbol": "AAPL",
  "title": "Apple Q4 Earnings Call",
  "category": "earnings",
  "impact_score": 8,
  "metadata": {
    "analyst_consensus": 1.25
  }
}
```

#### Batch Enrich Events
```bash
POST /enrichment/batch-enrich
```

**Request Body:**
```json
[
  {
    "symbol": "AAPL",
    "title": "Apple Q4 Earnings Call",
    "category": "earnings"
  },
  {
    "symbol": "TSLA", 
    "title": "Tesla FSD Beta Release",
    "category": "product_launch"
  }
]
```

#### Enrichment Statistics
```bash
GET /enrichment/stats
```

**Response:**
```json
{
  "cached_symbols": 150,
  "config": {
    "cache_duration_minutes": 30,
    "max_retries": 3,
    "batch_size": 10,
    "finnhub_enabled": true,
    "yahoo_finance_enabled": true
  }
}
```

### Configuration

Environment variables for enrichment service:

```bash
# API Keys
FINNHUB_API_KEY=your_finnhub_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

# Service Settings
YAHOO_FINANCE_ENABLED=true
ENRICHMENT_CACHE_DURATION_MINUTES=30
ENRICHMENT_MAX_RETRIES=3
ENRICHMENT_TIMEOUT_SECONDS=5.0
ENRICHMENT_BATCH_SIZE=10

# Sector Mapping (JSON)
EVENT_SECTOR_MAPPING={"AAPL":"technology","JPM":"financials"}
```

### Impact on Strategy Services

Enriched events enable more sophisticated trading strategies:

```python
# Strategy service can now react based on market context
@app.post("/webhooks/events")
async def handle_enriched_event(request: Request):
    event_data = await request.json()
    
    # Access enrichment data
    enrichment = event_data["data"]["metadata"]["enrichment"]
    market_context = enrichment["market_context"]
    impact_modifiers = enrichment["impact_modifiers"]
    
    # Strategy decisions based on enrichment
    if market_context["market_cap_tier"] == "mega_cap":
        # High liquidity, can use larger position sizes
        position_size = calculate_position_size(base_size * 1.5)
    
    if market_context["volatility_level"] == "very_high":
        # High volatility, reduce position size and use tighter stops
        position_size *= 0.7
        stop_loss_pct = 0.02  # 2% stop instead of 3%
    
    # Adjust expected impact based on modifiers
    total_modifier = sum(impact_modifiers.values())
    adjusted_impact = min(10, max(1, base_impact + total_modifier))
    
    if adjusted_impact >= 9:
        # Very high impact event, prepare for significant movement
        await prepare_high_impact_strategy(event_data)
```

See `examples/enrichment_demo.py` for a complete demonstration of the enrichment capabilities.

### Data Sources

The enrichment service uses multiple data sources for comprehensive coverage:

1. **Finnhub API**: Market cap, sector, beta, financials, real-time quotes
2. **Yahoo Finance**: Historical prices, volume data, volatility calculations
3. **Sector Mapping**: Configurable symbol-to-sector mapping for classification
4. **Caching Layer**: In-memory caching to optimize API usage and performance

The enrichment system significantly enhances the trading platform's ability to assess event impact and make intelligent strategy decisions based on comprehensive market context.

## Event Lifecycle Tracking

The service includes a sophisticated lifecycle tracking system that monitors events from scheduled state through occurrence to comprehensive impact analysis. This enables automated performance measurement, prediction accuracy tracking, and strategy optimization.

### Key Features
- **Automated Progression**: Events automatically transition through 4 lifecycle stages
- **Status Monitoring**: Tracks 5 event statuses with complete history logging
- **Impact Analysis**: Comprehensive post-event analysis with market data integration
- **Accuracy Measurement**: Compares predicted vs actual impact with scoring
- **Performance Tracking**: Monitors prediction model performance over time
- **Manual Control**: API endpoints for manual status updates and analysis triggers

### Lifecycle Stages

Events progress through four distinct stages:

1. **Pre-Event** (4+ hours before): Preparation and positioning phase
2. **Event Window** (4 hours before to 1 hour after): Active monitoring phase
3. **Post-Event** (1-72 hours after): Impact measurement phase
4. **Analysis Complete** (72+ hours after): Final analysis and archival

### Event Statuses

Events can have the following statuses with automatic and manual transitions:

- **Scheduled**: Event is planned and upcoming
- **Occurred**: Event has taken place
- **Cancelled**: Event was cancelled before occurring
- **Postponed**: Event was delayed to a new date
- **Impact Analyzed**: Complete analysis has been performed

### Impact Analysis

When events transition to "occurred" status, the system automatically schedules comprehensive impact analysis:

```json
{
  "impact_metrics": {
    "event_id": "event-123",
    "symbol": "AAPL",
    "predicted_impact": 8.0,
    "actual_impact": 7.2,
    "accuracy_score": 0.92,
    "max_move_pct": 4.5,
    "min_move_pct": -1.2,
    "volume_change_pct": 180.0,
    "volatility_spike": 2.1,
    "headline_sentiment": 0.3,
    "headline_count": 15,
    "analysis_timestamp": "2025-01-29T15:30:00Z"
  }
}
```

### API Endpoints

#### Get Event Lifecycle
```bash
GET /lifecycle/event/{event_id}
```

**Response:**
```json
{
  "event_id": "event-123",
  "symbol": "AAPL",
  "title": "Apple Q4 Earnings Call",
  "category": "earnings",
  "current_status": "impact_analyzed",
  "current_stage": "analysis_complete",
  "status_history": [
    {
      "status": "scheduled",
      "timestamp": "2025-01-25T10:00:00Z",
      "reason": "event_created"
    },
    {
      "status": "occurred",
      "timestamp": "2025-01-28T21:00:00Z",
      "reason": "automatic_detection"
    },
    {
      "status": "impact_analyzed",
      "timestamp": "2025-01-29T15:30:00Z",
      "reason": "analysis_completed"
    }
  ],
  "impact_metrics": { ... }
}
```

#### Update Event Status
```bash
PATCH /lifecycle/event/{event_id}/status?new_status=occurred&reason=manual_update
```

#### Get Events by Stage
```bash
GET /lifecycle/events/by-stage/pre_event
```

**Response:**
```json
{
  "stage": "pre_event",
  "count": 12,
  "events": [
    {
      "event_id": "event-456",
      "symbol": "TSLA",
      "title": "Tesla Q4 Earnings",
      "scheduled_at": "2025-01-30T21:00:00Z",
      "current_status": "scheduled",
      "current_stage": "pre_event"
    }
  ]
}
```

#### Get Events by Status
```bash
GET /lifecycle/events/by-status/occurred
```

#### Lifecycle Statistics
```bash
GET /lifecycle/stats
```

**Response:**
```json
{
  "total_tracked_events": 85,
  "status_distribution": {
    "scheduled": 23,
    "occurred": 31,
    "impact_analyzed": 28,
    "cancelled": 3
  },
  "stage_distribution": {
    "pre_event": 15,
    "event_window": 8,
    "post_event": 22,
    "analysis_complete": 40
  },
  "analyzed_events": 28,
  "average_accuracy": 0.847
}
```

#### Impact Analysis Results
```bash
GET /lifecycle/impact-analysis?min_accuracy=0.8&category=earnings&limit=20
```

**Response:**
```json
{
  "summary": {
    "total_analyzed": 28,
    "returned_count": 15,
    "average_accuracy": 0.892,
    "best_accuracy": 0.97,
    "worst_accuracy": 0.82
  },
  "events": [
    {
      "event_id": "event-123",
      "symbol": "AAPL",
      "category": "earnings",
      "impact_metrics": {
        "predicted_impact": 8.0,
        "actual_impact": 7.8,
        "accuracy_score": 0.97
      }
    }
  ]
}
```

### Configuration

Environment variables for lifecycle tracking:

```bash
# Monitoring Settings
LIFECYCLE_MONITOR_INTERVAL_MINUTES=15
LIFECYCLE_ANALYSIS_DELAY_HOURS=24
LIFECYCLE_PRE_EVENT_WINDOW_HOURS=4
LIFECYCLE_POST_EVENT_WINDOW_HOURS=72
```

### Strategy Integration

Lifecycle tracking enables sophisticated strategy optimization:

```python
# Strategy service using lifecycle data
@app.post("/webhooks/events")
async def handle_lifecycle_event(request: Request):
    event_data = await request.json()
    
    # Get lifecycle information
    lifecycle_response = await httpx.get(
        f"http://event-service:8006/lifecycle/event/{event_data['id']}"
    )
    lifecycle_data = lifecycle_response.json()
    
    current_stage = lifecycle_data["current_stage"]
    
    if current_stage == "pre_event":
        # Prepare for upcoming event
        await prepare_pre_event_strategy(event_data)
        
    elif current_stage == "event_window":
        # Execute event-driven trades
        await execute_event_strategy(event_data)
        
    elif current_stage == "analysis_complete":
        # Review performance and update models
        impact_metrics = lifecycle_data["impact_metrics"]
        accuracy = impact_metrics["accuracy_score"]
        
        if accuracy < 0.7:
            # Poor prediction - investigate and adjust
            await investigate_prediction_failure(event_data, impact_metrics)
        else:
            # Good prediction - reinforce successful patterns
            await reinforce_successful_prediction(event_data, impact_metrics)

# Performance monitoring dashboard
@app.get("/dashboard/lifecycle-performance")
async def get_lifecycle_performance():
    # Get accuracy by category
    analysis_response = await httpx.get(
        "http://event-service:8006/lifecycle/impact-analysis"
    )
    
    # Analyze performance trends
    return analyze_prediction_performance(analysis_response.json())
```

### Performance Monitoring

The lifecycle system provides comprehensive performance tracking:

- **Prediction Accuracy**: Track how well impact scores predict actual market movements
- **Category Performance**: Identify which event types are easiest/hardest to predict
- **Temporal Patterns**: Analyze how prediction accuracy varies by time of day, day of week, etc.
- **Model Drift**: Monitor prediction performance over time to detect model degradation
- **Strategy Optimization**: Use accuracy data to optimize position sizing and risk management

### Real-world Benefits

1. **Model Validation**: Continuous validation of impact prediction models
2. **Strategy Refinement**: Data-driven optimization of trading strategies
3. **Risk Management**: Better understanding of prediction confidence for position sizing
4. **Research Insights**: Historical data for academic and proprietary research
5. **Operational Monitoring**: Track system performance and identify issues

See `examples/lifecycle_demo.py` for a complete demonstration of the lifecycle tracking capabilities.

The lifecycle tracking system provides the foundation for building self-improving trading strategies that learn from their performance and continuously optimize based on real market outcomes.

## Event Sentiment Analysis Integration

The Event Data Service includes sophisticated sentiment analysis integration that connects with the Sentiment Service to analyze market sentiment around events and their outcomes. This enables comprehensive sentiment-driven event analysis and outcome prediction.

### Key Features

- **Multi-Timeframe Analysis**: Analyzes sentiment across pre-event, event window, and post-event periods
- **Multi-Source Integration**: Aggregates sentiment from Twitter, Reddit, news, and other social platforms
- **Outcome Prediction**: Uses sentiment patterns to predict event outcomes with confidence scoring
- **Sentiment Momentum**: Tracks sentiment changes over time to identify trends
- **Automatic Integration**: Seamlessly integrated into event lifecycle with automatic analysis
- **Caching Layer**: Optimized performance with intelligent caching and refresh mechanisms

### Sentiment Analysis Components

#### EventSentimentService

The core service that orchestrates sentiment analysis:

- **Timeframe Analysis**: Pre-event (24h), event window (2h), post-event (24h)
- **Source Aggregation**: Twitter, Reddit, news, and custom sources
- **Prediction Engine**: ML-based outcome prediction with confidence scoring
- **Performance Optimization**: Caching, batching, and timeout management

#### Sentiment Scoring

Each sentiment analysis includes comprehensive scoring:

```json
{
  "compound": 0.65,          // Overall sentiment (-1.0 to 1.0)
  "positive": 0.45,          // Positive sentiment ratio
  "negative": 0.15,          // Negative sentiment ratio
  "neutral": 0.40,           // Neutral sentiment ratio
  "label": "BULLISH",        // Classification (BULLISH/BEARISH/NEUTRAL)
  "confidence": 0.85,        // Confidence in classification
  "volume": 1250,            // Number of posts/articles analyzed
  "source": "twitter",       // Data source
  "timeframe": "pre_event"   // Analysis timeframe
}
```

### API Endpoints

#### Analyze Event Sentiment
```bash
GET /sentiment/events/{event_id}?force_refresh=false
```

**Response:**
```json
{
  "event_id": "event-123",
  "symbol": "AAPL",
  "category": "earnings",
  "analyzed_at": "2025-01-29T15:30:00Z",
  "overall_sentiment": {
    "compound": 0.45,
    "label": "BULLISH", 
    "confidence": 0.78,
    "volume": 3200
  },
  "sentiment_momentum": 0.12,      // Positive momentum
  "sentiment_divergence": 0.08,    // Low divergence between sources
  "outcome_prediction": "POSITIVE",
  "prediction_confidence": 0.82,
  "timeframes": {
    "pre_event": {
      "compound": 0.38,
      "label": "BULLISH",
      "confidence": 0.72,
      "volume": 1200
    },
    "event_window": {
      "compound": 0.52,
      "label": "BULLISH", 
      "confidence": 0.85,
      "volume": 800
    },
    "post_event": {
      "compound": 0.41,
      "label": "BULLISH",
      "confidence": 0.79,
      "volume": 1200
    }
  },
  "sources": {
    "twitter": {
      "compound": 0.48,
      "label": "BULLISH",
      "confidence": 0.81,
      "volume": 2100
    },
    "reddit": {
      "compound": 0.42,
      "label": "BULLISH", 
      "confidence": 0.75,
      "volume": 650
    },
    "news": {
      "compound": 0.45,
      "label": "BULLISH",
      "confidence": 0.78,
      "volume": 450
    }
  }
}
```

#### Analyze Event Outcome Sentiment
```bash
GET /sentiment/events/{event_id}/outcome
```

Analyzes sentiment specifically around event outcomes, focusing on headlines and immediate post-event social sentiment.

#### Get Sentiment Trends
```bash
GET /sentiment/trends/{symbol}?days=7
```

Retrieves historical sentiment trends for a symbol over a specified time period.

#### Get Sentiment Statistics
```bash
GET /sentiment/stats
```

**Response:**
```json
{
  "service": "event-sentiment-integration",
  "enabled": true,
  "sentiment_service_url": "http://localhost:8007",
  "configuration": {
    "pre_event_hours": 24,
    "post_event_hours": 24,
    "event_window_hours": 2,
    "timeout": 30.0
  },
  "cache_stats": {
    "cached_analyses": 45,
    "cache_ttl_seconds": 1800
  }
}
```

### Configuration

Sentiment analysis integration is configured via environment variables:

```bash
# Event Sentiment Analysis Configuration
EVENT_SENTIMENT_ENABLED=true
SENTIMENT_SERVICE_URL=http://localhost:8007
EVENT_SENTIMENT_TIMEOUT=30.0
EVENT_SENTIMENT_PRE_HOURS=24
EVENT_SENTIMENT_POST_HOURS=24
EVENT_SENTIMENT_WINDOW_HOURS=2
```

### Automatic Integration

Sentiment analysis is automatically triggered in the following scenarios:

1. **Event Creation**: New events are automatically analyzed for baseline sentiment
2. **Event Updates**: Modified events trigger sentiment re-analysis with cache refresh
3. **Lifecycle Transitions**: Events transitioning through lifecycle stages get updated analysis
4. **Manual Requests**: API endpoints allow on-demand analysis with force refresh options

### Use Cases

1. **Pre-Event Analysis**: Assess market sentiment before events to gauge expected reaction
2. **Outcome Prediction**: Use sentiment patterns to predict event outcomes and market moves
3. **Strategy Optimization**: Incorporate sentiment signals into trading strategy decisions
4. **Risk Management**: Monitor sentiment divergence as a risk indicator
5. **Performance Analysis**: Analyze sentiment accuracy vs actual outcomes for model improvement

### Demo and Examples

See `examples/sentiment_demo.py` for a complete demonstration of the sentiment analysis integration capabilities.

The sentiment analysis integration provides powerful insights into market psychology around events, enabling more sophisticated and sentiment-aware trading strategies.

## Historical Data Backfill Capabilities

The Event Data Service includes comprehensive historical data backfill capabilities that automatically populate historical event data for new symbols added to the trading platform. This ensures complete data coverage and enables comprehensive analysis from the moment a new symbol is added.

### Key Features

- **Automatic Symbol Detection**: Automatically detects new symbols and triggers backfill
- **Multi-Source Integration**: Supports multiple premium data sources for comprehensive coverage
- **Concurrent Processing**: Processes multiple symbols simultaneously with configurable limits
- **Progress Tracking**: Real-time progress monitoring and status reporting
- **Intelligent Queuing**: Priority-based queue management with rate limiting
- **Error Resilience**: Robust error handling with retry mechanisms and graceful degradation

### Supported Data Sources

#### Financial Modeling Prep (Primary Source)
- **Coverage**: Earnings calendar, stock splits, dividends, IPO calendar
- **Rate Limit**: 250 requests/minute
- **Data Quality**: High accuracy with comprehensive metadata
- **Configuration**: `FMP_API_KEY`

#### Alpha Vantage
- **Coverage**: Earnings calendar, news sentiment
- **Rate Limit**: 5 requests/minute
- **Data Quality**: Good coverage for major stocks
- **Configuration**: `ALPHA_VANTAGE_API_KEY`

#### Polygon.io
- **Coverage**: Stock splits, dividends, news events
- **Rate Limit**: 5 requests/minute
- **Data Quality**: High precision for corporate actions
- **Configuration**: `POLYGON_API_KEY`

#### Finnhub
- **Coverage**: Earnings calendar, IPO calendar, economic events
- **Rate Limit**: 60 requests/minute
- **Data Quality**: Good coverage with real-time updates
- **Configuration**: `FINNHUB_API_KEY`

### Backfill Process

#### Automatic Triggering

When a new symbol is detected (first event created for a symbol), the system automatically:

1. **Detects New Symbol**: Checks if any historical events exist for the symbol
2. **Queues Backfill Request**: Adds medium-priority backfill request to the queue
3. **Processes Multiple Sources**: Fetches data from all configured sources in parallel
4. **Applies Categorization**: Uses event categorizer for consistent classification
5. **Enriches Data**: Applies market context enrichment and impact scoring
6. **Stores Events**: Creates historical events with proper metadata and relationships

#### Manual Backfill

Users can also manually request backfill for specific symbols with custom parameters:

```bash
POST /backfill/symbols/{symbol}?start_date=2023-01-01&end_date=2024-12-31&categories=earnings,split&priority=1
```

### API Endpoints

#### Request Symbol Backfill
```bash
POST /backfill/symbols/{symbol}
```

**Parameters:**
- `start_date` (optional): Start date in YYYY-MM-DD format (default: 365 days ago)
- `end_date` (optional): End date in YYYY-MM-DD format (default: today)
- `categories` (optional): Comma-separated list of event categories
- `sources` (optional): Comma-separated list of data sources
- `priority` (1-3): Priority level (1=high, 2=medium, 3=low)

**Response:**
```json
{
  "message": "Backfill requested for NVDA",
  "request_id": "backfill-NVDA-2025-01-29T15:30:00",
  "symbol": "NVDA",
  "parameters": {
    "start_date": "2024-01-29T00:00:00",
    "end_date": "2025-01-29T00:00:00",
    "categories": ["earnings", "split"],
    "sources": ["financial_modeling_prep", "polygon"],
    "priority": 1
  }
}
```

#### Get Backfill Status
```bash
GET /backfill/status/{symbol}
```

**Response:**
```json
{
  "symbol": "NVDA",
  "status": "in_progress",
  "progress": {
    "total_requests": 4,
    "completed_requests": 2,
    "completion_percentage": 50.0,
    "current_source": "polygon",
    "current_date_range": "2024-01-01 to 2024-12-31",
    "events_processed": 127,
    "started_at": "2025-01-29T15:30:00Z",
    "estimated_completion": "2025-01-29T15:45:00Z"
  }
}
```

#### List Active Backfills
```bash
GET /backfill/active
```

Returns all currently running backfill operations with their progress status.

#### Get Backfill Statistics
```bash
GET /backfill/stats
```

**Response:**
```json
{
  "service": "historical-backfill",
  "enabled": true,
  "configured_sources": ["financial_modeling_prep", "polygon", "finnhub"],
  "statistics": {
    "total_events": 15420,
    "backfilled_events": 8750,
    "recent_backfills_7d": 125
  },
  "active_backfills": 2,
  "queue_size": 5,
  "configuration": {
    "max_concurrent_symbols": 3,
    "default_lookback_days": 365,
    "max_days_per_request": 90,
    "rate_limit_delay": 1.0,
    "timeout": 30.0,
    "retry_attempts": 3,
    "batch_size": 100
  }
}
```

### Configuration

Historical backfill is configured via environment variables:

```bash
# Historical Data Backfill Configuration
BACKFILL_ENABLED=true
BACKFILL_MAX_CONCURRENT_SYMBOLS=3
BACKFILL_DEFAULT_LOOKBACK_DAYS=365
BACKFILL_MAX_DAYS_PER_REQUEST=90
BACKFILL_RATE_LIMIT_DELAY=1.0
BACKFILL_TIMEOUT=30.0
BACKFILL_RETRY_ATTEMPTS=3
BACKFILL_BATCH_SIZE=100

# Data Source API Keys for Backfill
FMP_API_KEY=your_financial_modeling_prep_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
POLYGON_API_KEY=your_polygon_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
```

### Data Processing Pipeline

#### Event Processing Flow

1. **Source Data Fetch**: Retrieves raw event data from external APIs
2. **Data Normalization**: Converts to standardized event format
3. **Deduplication**: Prevents duplicate events using source + external_id
4. **Categorization**: Applies intelligent event categorization
5. **Impact Scoring**: Calculates impact scores based on event characteristics
6. **Market Enrichment**: Adds market context (market cap, sector, volatility)
7. **Storage**: Persists events with full metadata and relationships

#### Quality Assurance

- **Duplicate Prevention**: Unique constraints on (source, external_id)
- **Data Validation**: Schema validation and business rule enforcement
- **Error Handling**: Graceful handling of API failures and malformed data
- **Retry Logic**: Automatic retry with exponential backoff for transient failures
- **Rate Limiting**: Respects API rate limits to prevent service disruption

### Performance Characteristics

#### Throughput

- **Concurrent Symbols**: Up to 3 symbols processed simultaneously
- **Daily Processing**: 500-1000 events per hour per source
- **Batch Size**: 100 events processed per database transaction
- **Queue Capacity**: Unlimited with memory-efficient design

#### Resource Usage

- **Memory**: Minimal memory footprint with streaming processing
- **CPU**: Low CPU usage with async I/O operations
- **Network**: Optimized API usage with connection pooling
- **Database**: Efficient bulk operations with minimal locking

### Use Cases

1. **New Symbol Onboarding**: Automatically populate historical context for new symbols
2. **Data Gap Filling**: Backfill missing data for existing symbols
3. **Historical Analysis**: Enable comprehensive historical analysis from day one
4. **Strategy Development**: Provide sufficient historical data for strategy development
5. **Compliance**: Ensure complete audit trail and data coverage

### Integration Examples

#### Automatic Backfill on New Symbol
```python
# When creating an event for a new symbol
POST /events
{
  "symbol": "RBLX",
  "title": "Roblox Q4 Earnings",
  "category": "earnings",
  "scheduled_at": "2025-02-15T16:30:00Z"
}

# System automatically detects new symbol and triggers backfill
# Background process fetches 1-year historical data
# Events are categorized, scored, and enriched automatically
```

#### Manual Backfill Request
```python
# Request specific historical data
POST /backfill/symbols/TSLA?start_date=2023-01-01&categories=earnings,split&priority=1

# Monitor progress
GET /backfill/status/TSLA

# Check results
GET /events?symbol=TSLA&limit=100
```

### Demo and Examples

See `examples/backfill_demo.py` for a complete demonstration of the historical backfill capabilities.

The historical backfill system ensures that new symbols have complete historical context from the moment they're added to the platform, enabling sophisticated analysis and strategy development without manual data management.

## Data Retention and Archival Policies

The Event Data Service includes a comprehensive data retention and archival system to manage storage costs, ensure compliance, and optimize performance over time.

### Retention Policy Overview

The system implements a 5-tier data lifecycle management strategy:

1. **Active Data (0-30 days)** - Recent events and headlines stored in primary database
2. **Warm Data (30-180 days)** - Less frequently accessed data, optimized for query performance  
3. **Cold Data (180 days-2 years)** - Archived to compressed formats, available on demand
4. **Compliance Data (2-7 years)** - Long-term archive for regulatory requirements
5. **Deletion (7+ years)** - Permanent removal of expired data

### Default Retention Rules

#### Events
- **Active Events** (30 days): Recent scheduled/occurred events in primary storage
- **Warm Events** (180 days): Completed events with fast query access
- **Cold Events** (2 years): Compressed JSON/Parquet archives
- **Compliance Events** (7 years): Parquet format for regulatory compliance
- **Deletion** (7+ years): Permanent removal of very old events

#### Headlines
- **Active Headlines** (60 days): Recent headlines linked to events
- **Archive Headlines** (1 year): Compressed JSON archives  
- **Deletion** (1+ years): Permanent removal of old headlines

### Configuration

Configure retention policies via environment variables:

```bash
# Enable/disable retention service
RETENTION_ENABLED=true

# Cleanup frequency (hours)
RETENTION_CLEANUP_INTERVAL_HOURS=24

# Archive storage location
RETENTION_ARCHIVE_PATH=/data/archives

# Processing configuration
RETENTION_BATCH_SIZE=1000
RETENTION_MAX_PARALLEL=3

# Custom rules (JSON array)
RETENTION_CUSTOM_RULES='[{"name":"custom_rule","category":"events","policy":"cold","age_days":90,"conditions":{"status":["completed"]},"archive_format":"json","archive_location":"custom_archive","enabled":true,"priority":1}]'
```

### Archive Formats

- **JSON** - Human-readable, compressed with gzip
- **Parquet** - Optimized columnar format for analytics
- **CSV** - Simple format for export/import

### API Endpoints

#### Get Retention Statistics
```bash
GET /retention/stats
```
Returns current storage statistics and data distribution.

#### View Current Rules
```bash
GET /retention/rules
```
Lists all configured retention rules.

#### Manual Cleanup
```bash
POST /retention/cleanup
```
Triggers immediate retention cleanup process.

#### Validate Rules
```bash
POST /retention/validate-rule?rule_name=test&category=events&policy=cold&age_days=90
```
Validates retention rule configuration and estimates impact.

### Custom Retention Rules

Create custom rules for specific business requirements:

```json
{
  "name": "high_impact_events",
  "category": "events", 
  "policy": "compliance",
  "age_days": 1095,
  "conditions": {
    "impact_score": [8, 9, 10],
    "category": ["earnings", "fda_approval"]
  },
  "archive_format": "parquet",
  "archive_location": "compliance/high_impact",
  "enabled": true,
  "priority": 1
}
```

### Compliance and Recovery

- **Audit Trail**: All archival operations are logged with timestamps and file locations
- **Data Recovery**: Archived data can be restored from JSON/Parquet files
- **GDPR Compliance**: Supports deletion policies for data privacy requirements
- **Backup Integration**: Archive files can be backed up to cloud storage or tape

### Performance Impact

- **Minimal Runtime Impact**: Background cleanup runs during off-peak hours
- **Gradual Processing**: Large datasets processed in configurable batches
- **Query Optimization**: Removes old data to improve query performance
- **Storage Savings**: Achieves 70-90% storage reduction through compression and archival

### Monitoring and Alerts

The retention service integrates with the existing health monitoring system:

- **Health Endpoint**: `/health` includes retention service status
- **Archive Metrics**: Track archive sizes, operation duration, and success rates
- **Alert Integration**: Notifications for failed archival operations
- **Statistics Dashboard**: View retention statistics and storage trends

This automated retention system ensures the Event Data Service maintains optimal performance while meeting compliance requirements and managing storage costs effectively.

## Redis Caching Layer

The Event Data Service includes a high-performance Redis caching layer that significantly reduces database load and improves response times for frequently accessed data.

### Cache Architecture

The caching system implements a multi-tier strategy with intelligent cache invalidation:

- **Event Cache**: Individual events cached by ID (1 hour TTL)
- **List Cache**: Event list queries cached with parameters (5 minutes TTL)  
- **Search Cache**: Complex search results cached (10 minutes TTL)
- **Aggregation Cache**: Statistical aggregations cached (15 minutes TTL)
- **Enrichment Cache**: Market context data cached (2 hours TTL)

### Performance Benefits

**Typical Performance Improvements:**
- Single event retrieval: 60-80% faster on cache hit
- Event list queries: 70-90% faster on cache hit
- Complex searches: 80-95% faster on cache hit
- Reduced database load: 40-60% fewer DB queries

### Configuration

Configure the Redis cache via environment variables:

```bash
# Enable/disable Redis caching
EVENT_CACHE_ENABLED=true

# Redis connection (inherits from main Redis config)
REDIS_URL=redis://localhost:6379
EVENT_CACHE_DB=1
EVENT_CACHE_TTL_SECONDS=3600
EVENT_CACHE_MAX_CONNECTIONS=10
EVENT_CACHE_KEY_PREFIX=event_cache

# Performance tuning
EVENT_CACHE_COMPRESSION_THRESHOLD=1024
EVENT_CACHE_BATCH_SIZE=100
EVENT_CACHE_TIMEOUT=5.0
```

### Cache Management API

#### Get Cache Statistics
```bash
GET /cache/stats
```
Returns comprehensive cache performance metrics and Redis statistics.

#### Cache Health Check
```bash
GET /cache/health
```
Returns cache service health status and connection information.

#### Invalidate Specific Event
```bash
POST /cache/invalidate/event/{event_id}?symbol=AAPL
```
Invalidates all cache entries related to a specific event.

#### Invalidate Symbol Data
```bash
POST /cache/invalidate/symbol/{symbol}
```
Invalidates all cache entries for a specific symbol.

#### Pattern-Based Invalidation
```bash
POST /cache/invalidate/pattern?pattern=event_list:*
```
Invalidates cache entries matching a specific pattern.

#### Clear All Cache
```bash
DELETE /cache/clear
```
Clears all cache data (use with caution in production).

### Automatic Cache Invalidation

The cache system automatically invalidates relevant entries when data changes:

**Event Operations:**
- Creating events: Invalidates list and search caches
- Updating events: Invalidates event, list, and symbol-related caches  
- Deleting events: Invalidates all related cache entries

**Smart Invalidation:**
- Symbol-based invalidation for related data
- Pattern-based cleanup for list queries
- Hierarchical invalidation for nested relationships

### Cache Key Structure

Cache keys follow a structured namespace pattern:

```
event_cache:event:{event_id}
event_cache:event_list:list:symbol=AAPL&category=earnings
event_cache:search:{hash}:symbol=AAPL&limit=10
event_cache:agg:stats:{hash}:symbol=AAPL
```

### Data Compression

**Automatic Compression:**
- Data larger than 1KB is automatically compressed with gzip
- Base64 encoding for Redis storage compatibility
- Transparent decompression on retrieval
- Achieves 60-80% size reduction for typical event data

### Monitoring and Analytics

**Built-in Metrics:**
- Hit/miss ratios per cache type
- Response time improvements
- Memory usage and key counts
- Redis server statistics

**Health Integration:**
Cache status is included in the main `/health` endpoint:

```json
{
  "cache": {
    "enabled": true,
    "status": "connected", 
    "total_keys": 1547,
    "hit_rate": 0.73,
    "memory_usage_mb": 12.4
  }
}
```

### Cache Strategies

**Time-Based (TTL):** Default strategy with automatic expiration
**LRU Support:** Least-recently-used eviction for memory management
**Write-Through:** Updates both cache and database simultaneously
**Smart Prefetching:** Proactive caching of related data

### Production Considerations

**Memory Management:**
- Monitor Redis memory usage with `INFO memory`
- Configure max memory and eviction policies
- Use Redis clustering for high-traffic deployments

**Performance Tuning:**
- Adjust TTL values based on data change frequency
- Configure compression threshold for optimal performance
- Monitor cache hit rates and adjust strategies accordingly

**Security:**
- Use Redis AUTH for production deployments
- Configure network security and firewall rules
- Regular backup and monitoring of Redis instances

### Demo and Testing

Run the comprehensive cache demonstration:

```bash
cd examples
python cache_demo.py
```

This demo showcases:
- Performance improvements from caching
- Automatic cache invalidation workflows  
- Cache management API usage
- Statistics and monitoring capabilities

The Redis caching layer provides enterprise-grade performance optimization while maintaining data consistency and offering comprehensive management capabilities.

## Bulk Ingestion for Historical Event Backlogs

The Event Data Service includes a high-performance bulk ingestion system designed to efficiently process large historical event datasets with optimized database operations and intelligent data quality management.

### Bulk Ingestion Architecture

The system provides enterprise-grade bulk data processing with:

- **Multi-Format Support**: CSV, JSON, JSON Lines (JSONL), Parquet, Excel, SQL
- **Intelligent Processing**: Automatic categorization, impact scoring, and data enrichment
- **Performance Optimization**: Batch processing with PostgreSQL UPSERT operations
- **Data Quality**: Configurable validation levels and error handling
- **Progress Monitoring**: Real-time operation tracking and statistics

### Performance Characteristics

**Typical Throughput:**
- CSV files: 1,000-5,000 records/second
- JSON files: 800-3,000 records/second  
- JSONL files: 1,200-4,000 records/second
- Database impact: Optimized batch operations reduce DB load by 80-90%

### Ingestion Modes

**Insert Only** - Only insert new records, skip duplicates
```bash
POST /bulk/ingest?mode=insert_only
```

**Upsert** - Insert new records, update existing ones (default)
```bash
POST /bulk/ingest?mode=upsert
```

**Replace** - Delete existing records for symbols, insert new data
```bash
POST /bulk/ingest?mode=replace
```

**Append** - Append all records regardless of duplicates
```bash
POST /bulk/ingest?mode=append
```

### Validation Levels

**Strict** - Fail on any validation error
**Permissive** - Skip invalid records, continue processing (default)
**None** - No validation, fastest processing

### Configuration

Configure bulk ingestion via environment variables:

```bash
# Enable/disable bulk ingestion service
BULK_INGESTION_ENABLED=true

# File processing limits
BULK_INGESTION_MAX_FILE_SIZE_MB=500
BULK_INGESTION_TEMP_DIR=/tmp
BULK_INGESTION_BATCH_SIZE=1000
BULK_INGESTION_MAX_WORKERS=4

# Performance settings
BULK_INGESTION_MEMORY_LIMIT_MB=1024
BULK_INGESTION_CHUNK_SIZE=10000
BULK_INGESTION_VACUUM_THRESHOLD=100000
BULK_INGESTION_POOL_SIZE=20
```

### API Endpoints

#### Ingest Bulk Data
```bash
POST /bulk/ingest?file_path=/data/events.csv&format_type=csv&batch_size=1000
```

**Parameters:**
- `file_path`: Path to the file to ingest
- `format_type`: File format (csv, json, jsonl)
- `batch_size`: Records per batch (default: 1000)
- `mode`: Ingestion mode (default: upsert)
- `validation_level`: Validation strictness (default: permissive)
- `auto_categorize`: Enable automatic categorization (default: true)
- `auto_enrich`: Enable automatic enrichment (default: false, expensive)
- `skip_cache_invalidation`: Skip cache cleanup for performance (default: false)

#### Validate File Before Ingestion
```bash
POST /bulk/validate?file_path=/data/events.csv&format_type=csv&sample_size=100
```

Returns validation summary, error analysis, and recommendations.

#### Monitor Active Operations
```bash
GET /bulk/operations
```

List all active bulk ingestion operations with progress details.

#### Get Operation Status
```bash
GET /bulk/operations/{operation_id}
```

Get detailed status of a specific ingestion operation.

#### Service Statistics
```bash
GET /bulk/stats
```

Returns service configuration and performance statistics.

### File Format Requirements

#### CSV Format
```csv
symbol,title,category,scheduled_at,description,status,source,external_id,metadata
AAPL,Apple Q1 Earnings,earnings,2024-01-25T16:00:00Z,Quarterly earnings call,scheduled,bulk_import,ext_123,"{""demo"":true}"
```

**Required fields:** symbol, title, scheduled_at
**Optional fields:** category, description, status, timezone, source, external_id, impact_score, metadata

#### JSON Format
```json
{
  "events": [
    {
      "symbol": "AAPL",
      "title": "Apple Q1 Earnings",
      "category": "earnings",
      "scheduled_at": "2024-01-25T16:00:00Z",
      "description": "Quarterly earnings call",
      "status": "scheduled",
      "metadata": {
        "demo": true,
        "quarter": "Q1"
      }
    }
  ]
}
```

#### JSON Lines (JSONL) Format
```jsonl
{"symbol": "AAPL", "title": "Apple Q1 Earnings", "category": "earnings", "scheduled_at": "2024-01-25T16:00:00Z"}
{"symbol": "MSFT", "title": "Microsoft Q1 Earnings", "category": "earnings", "scheduled_at": "2024-01-26T16:00:00Z"}
```

### Data Quality Features

**Automatic Categorization:**
- Uses ML-based categorization engine
- Confidence scoring and keyword matching
- Classification metadata preservation

**Data Validation:**
- Required field validation
- Data type checking
- Format validation (dates, symbols, etc.)
- Configurable error thresholds

**Duplicate Handling:**
- Source + external_id based deduplication
- Configurable duplicate strategies
- Statistics tracking for duplicates

**Error Recovery:**
- Partial success support
- Detailed error reporting
- Resume capability for failed operations

### Performance Optimization

**Batch Processing:**
- Configurable batch sizes (100-10,000 records)
- PostgreSQL UPSERT operations
- Connection pooling and transaction optimization

**Memory Management:**
- Streaming file processing
- Configurable memory limits
- Automatic garbage collection

**Database Optimization:**
- Bulk INSERT/UPSERT operations
- Index-aware processing
- Automatic VACUUM scheduling

### Monitoring and Observability

**Real-time Statistics:**
- Records processed, inserted, updated, failed
- Processing throughput (records/second)
- Memory usage and batch performance
- Error rates and validation statistics

**Operation Tracking:**
- Unique operation IDs
- Start/completion timestamps
- File information and metadata
- Progress percentage and ETA

**Error Analysis:**
- Detailed error messages
- Record-level error tracking
- Validation failure categorization
- Recovery recommendations

### Production Considerations

**Capacity Planning:**
- Monitor file sizes and processing times
- Scale batch sizes based on system performance
- Configure appropriate memory limits

**Error Handling:**
- Set validation levels based on data quality requirements
- Monitor error rates and adjust thresholds
- Implement retry strategies for transient failures

**Security:**
- Validate file paths and access permissions
- Sanitize file contents and metadata
- Audit logging for bulk operations

### Integration with Other Services

**Cache Integration:**
- Automatic cache invalidation after ingestion
- Configurable cache refresh strategies
- Performance optimization during bulk loads

**Event Processing:**
- Automatic categorization via ML pipeline
- Optional enrichment with market data
- Impact scoring for historical events

**Retention Integration:**
- Bulk-imported data follows retention policies
- Archive-friendly metadata tagging
- Compliance-ready audit trails

### Demo and Testing

Run the comprehensive bulk ingestion demonstration:

```bash
cd examples
python bulk_ingestion_demo.py
```

This demo showcases:
- Multi-format file processing (CSV, JSON, JSONL)
- Different ingestion modes and validation levels
- Performance comparison with individual API calls
- File validation and error handling
- Real-time monitoring and statistics

The bulk ingestion system provides enterprise-grade performance for processing large historical datasets while maintaining data quality and system performance.

## Real-Time Event Streaming

The Event Data Service provides real-time event streaming capabilities through multiple protocols to enable immediate notification of event changes and updates. This system supports high-frequency trading applications, real-time dashboards, and event-driven architectures.

### Streaming Backends

The service supports multiple streaming backends for different use cases:

1. **Redis Streams** - Persistent, ordered event streams with consumer groups
2. **Redis Pub/Sub** - Low-latency broadcasting for real-time notifications  
3. **WebSocket** - Bidirectional real-time communication with web clients
4. **Server-Sent Events (SSE)** - Unidirectional streaming to web browsers

### Configuration

Configure streaming in your `.env` file:

```env
# Enable/disable real-time event streaming
EVENT_STREAMING_ENABLED=true

# Streaming backends (comma-separated: redis_streams, redis_pubsub, websocket, sse)
EVENT_STREAMING_BACKENDS=redis_streams,websocket,sse

# Redis configuration for streaming
EVENT_STREAMING_REDIS_DB=2
EVENT_STREAMING_PREFIX=events

# Stream performance settings
EVENT_STREAMING_MAX_LENGTH=10000
EVENT_STREAMING_BATCH_SIZE=100
EVENT_STREAMING_TIMEOUT_MS=5000
EVENT_STREAMING_RETENTION_HOURS=24

# Connection limits
WS_MAX_CONNECTIONS=1000
SSE_MAX_CONNECTIONS=500
```

### WebSocket API

Connect to real-time events via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8010/ws/events');

ws.onopen = function() {
    console.log('Connected to Event Data Service');
    
    // Subscribe to specific event types
    ws.send(JSON.stringify({
        type: 'subscribe',
        topics: ['event.created', 'event.updated'],
        filters: {
            symbols: ['AAPL', 'MSFT'],
            min_priority: 5
        }
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received event:', data);
};
```

**WebSocket Message Types:**

**Client → Server:**
```json
{
    "type": "subscribe",
    "topics": ["event.created", "event.updated"],
    "filters": {
        "symbols": ["AAPL", "MSFT"],
        "min_priority": 5,
        "sources": ["provider1"]
    }
}
```

```json
{
    "type": "unsubscribe", 
    "topics": ["event.updated"]
}
```

```json
{
    "type": "ping"
}
```

**Server → Client:**
```json
{
    "type": "event.created",
    "id": "evt_123",
    "timestamp": "2025-01-28T21:05:00Z",
    "data": {
        "id": "event-456",
        "symbol": "AAPL",
        "title": "Apple Q4 Earnings Call",
        "category": "earnings",
        "impact_score": 8,
        "scheduled_at": "2025-01-28T21:00:00Z"
    },
    "priority": 8,
    "source": "provider1"
}
```

### Server-Sent Events (SSE)

Stream events to web browsers using SSE:

```javascript
const eventSource = new EventSource('/events/stream?topics=event.created,event.updated&symbols=AAPL,MSFT');

eventSource.addEventListener('event.created', function(e) {
    const event = JSON.parse(e.data);
    console.log('New event:', event);
});

eventSource.addEventListener('heartbeat', function(e) {
    console.log('Heartbeat:', e.data);
});
```

**SSE Endpoint:**
```
GET /events/stream?topics=event.created&symbols=AAPL&min_priority=5
```

### Redis Streams Integration

For applications requiring persistent event streams:

```python
import redis

r = redis.Redis(host='localhost', port=6379, db=2)

# Read from event stream
events = r.xread({'events:all': '$'}, block=1000)
for stream, messages in events:
    for message_id, fields in messages:
        event_data = {k.decode(): v.decode() for k, v in fields.items()}
        print(f"Event {message_id}: {event_data}")
```

### Event Types and Filtering

**Supported Event Types:**
- `event.created` - New events from providers
- `event.updated` - Event modifications
- `event.deleted` - Event removals
- `event.status_changed` - Status transitions
- `event.impact_changed` - Impact score updates
- `headline.linked` - New headlines associated
- `cluster.formed` - Event clustering results

**Filtering Options:**
- `symbols` - Array of stock symbols
- `categories` - Array of event categories
- `min_priority` / `max_priority` - Priority range
- `sources` - Array of data providers
- `min_impact_score` - Minimum impact threshold

### Streaming Performance

**Throughput Metrics:**
- Redis Streams: 50,000+ events/second
- WebSocket: 10,000+ concurrent connections
- SSE: 5,000+ concurrent connections
- Average latency: < 5ms for local Redis

**Connection Management:**
- Automatic reconnection handling
- Heartbeat monitoring (30-second intervals)
- Graceful connection cleanup
- Rate limiting and backpressure control

### API Endpoints

#### Stream Management
```bash
GET /streaming/status
GET /streaming/stats
POST /streaming/restart
```

#### Connection Monitoring
```bash
GET /streaming/connections
GET /streaming/connections/websocket
GET /streaming/connections/sse
```

#### Stream Configuration
```bash
GET /streaming/config
PATCH /streaming/config
```

### Integration with CRUD Operations

Event streaming is automatically integrated with all CRUD operations:

```python
# Creating an event automatically publishes to streams
new_event = await create_event(event_data)
# → Publishes 'event.created' to all active streams

# Updating an event publishes change notifications
updated_event = await update_event(event_id, updates)
# → Publishes 'event.updated' to all active streams

# Deleting an event publishes deletion notifications
await delete_event(event_id)
# → Publishes 'event.deleted' to all active streams
```

### Error Handling and Resilience

**Connection Resilience:**
- Automatic reconnection with exponential backoff
- Dead letter queues for failed deliveries
- Circuit breaker pattern for downstream services
- Comprehensive error logging and metrics

**Message Delivery:**
- At-least-once delivery guarantees
- Message deduplication for Redis Streams
- Retry logic for WebSocket/SSE connections
- Configurable timeout and retry policies

### Security Considerations

**Authentication:**
- WebSocket connections support token-based auth
- SSE endpoints require valid API keys
- Redis Streams use connection-level authentication

**Rate Limiting:**
- Per-connection message rate limits
- Global streaming bandwidth controls
- DDoS protection via connection limits

### Monitoring and Observability

**Key Metrics:**
- Active connection counts by protocol
- Message throughput rates
- Connection establishment/termination rates
- Error rates and failure modes
- Streaming latency percentiles

**Health Checks:**
```bash
GET /streaming/health
```

Returns comprehensive streaming service health including Redis connectivity, active connections, and performance metrics.

### Demo and Testing

Run the real-time streaming demonstration:

```bash
cd examples
python streaming_demo.py
```

This demo showcases:
- WebSocket connection management and subscriptions
- SSE streaming with filtering
- Redis Streams integration
- Performance benchmarking across protocols
- Error handling and reconnection logic
- Real-time event publication and consumption

The streaming system enables real-time financial applications with enterprise-grade performance, reliability, and scalability.

## Event Analytics and Reporting Dashboard

The Event Data Service provides a comprehensive analytics platform with interactive dashboards, detailed metrics, trend analysis, and performance reporting. This system enables data-driven insights for trading strategies, market analysis, and operational optimization.

### Analytics Dashboard

Access the interactive web dashboard at: `http://localhost:8010/dashboard`

**Dashboard Features:**
- **Real-time Metrics**: Live event counts, impact scores, and performance indicators
- **Interactive Charts**: Time series visualizations with Chart.js integration
- **Distribution Analysis**: Event breakdowns by category, status, source, and impact
- **Trend Monitoring**: Growth rates, volatility analysis, and directional indicators
- **Performance Tables**: Top symbols, trending categories, and source reliability
- **Auto-refresh**: Automatic data updates every 5 minutes

### Analytics API Endpoints

#### Dashboard Data
```bash
GET /analytics/dashboard
```

Returns comprehensive dashboard data including metrics, charts, and performance indicators:

```json
{
  "summary": {
    "total_events": 15420,
    "weekly_events": 2847,
    "daily_events": 412,
    "avg_impact_score": 6.8,
    "high_impact_events": 284,
    "headline_coverage": 78.5
  },
  "trends": {
    "event_count": {
      "growth_rate": 12.4,
      "trend_direction": "up",
      "volatility": 15.2
    }
  },
  "distributions": {
    "by_category": {"earnings": 3241, "fda_approval": 892},
    "by_status": {"scheduled": 8921, "occurred": 6499}
  },
  "charts": {
    "event_timeline": [...],
    "impact_timeline": [...]
  }
}
```

#### Event Metrics
```bash
GET /analytics/metrics?start_date=2025-01-01&symbols=AAPL,MSFT&categories=earnings
```

**Parameters:**
- `start_date` - Start date filter (ISO format)
- `end_date` - End date filter (ISO format)
- `symbols` - Comma-separated symbol list
- `categories` - Comma-separated category list
- `use_cache` - Enable/disable caching (default: true)

```json
{
  "metrics": {
    "total_events": 2847,
    "events_by_category": {"earnings": 1204, "fda_approval": 892},
    "events_by_status": {"scheduled": 1521, "occurred": 1326},
    "events_by_source": {"provider1": 1847, "provider2": 1000},
    "events_by_symbol": {"AAPL": 284, "MSFT": 192},
    "average_impact_score": 7.2,
    "high_impact_events": 127,
    "events_with_headlines": 2214,
    "total_headlines": 4892
  }
}
```

#### Time Series Analysis
```bash
GET /analytics/timeseries?metric=event_count&start_date=2025-01-01&end_date=2025-01-31&interval=1d
```

**Supported Metrics:**
- `event_count` - Number of events over time
- `impact_score` - Average impact score over time

**Intervals:**
- `5m` - 5-minute buckets
- `15m` - 15-minute buckets
- `1h` - Hourly buckets
- `1d` - Daily buckets
- `1w` - Weekly buckets

```json
{
  "metric": "event_count",
  "interval": "1d",
  "data_points": [
    {
      "timestamp": "2025-01-01T00:00:00Z",
      "value": 124,
      "category": "earnings"
    }
  ]
}
```

#### Trend Analysis
```bash
GET /analytics/trends?metric=event_count&period_days=30&symbols=AAPL
```

```json
{
  "analysis": {
    "period": "30d",
    "growth_rate": 15.7,
    "trend_direction": "up",
    "volatility": 12.4,
    "peak_timestamp": "2025-01-15T00:00:00Z",
    "peak_value": 187
  },
  "data_points": [...]
}
```

#### Performance Reports
```bash
GET /analytics/performance?period_days=7&limit=10
```

```json
{
  "report": {
    "period": "7d",
    "most_active_symbols": [["AAPL", 142], ["MSFT", 98]],
    "trending_categories": [
      {
        "category": "earnings",
        "event_count": 284,
        "growth_rate": 25.7
      }
    ],
    "impact_distribution": {
      "High (8-10)": 127,
      "Medium (5-7)": 892,
      "Low (1-4)": 428
    },
    "source_reliability": {
      "provider1": {
        "event_count": 847,
        "headline_count": 1294,
        "headlines_per_event": 1.53,
        "avg_impact_score": 7.2
      }
    },
    "headline_coverage": {
      "AAPL": 89.4,
      "MSFT": 76.8
    }
  }
}
```

### Analytics Features

**Metrics and KPIs:**
- Total event counts with time filtering
- Event distribution by category, status, source, and symbol
- Impact score statistics and high-impact event identification
- Headline coverage analysis and content metrics
- Source reliability and data quality indicators

**Trend Analysis:**
- Growth rate calculation with percentage changes
- Trend direction detection (up/down/stable)
- Volatility measurement using coefficient of variation
- Peak detection and performance benchmarking
- Time-based pattern recognition

**Performance Monitoring:**
- Most active symbols by event volume
- Trending categories with growth rates
- Source performance and reliability metrics
- Data quality assessments and coverage analysis
- Real-time throughput and processing statistics

**Filtering and Segmentation:**
- Symbol-based filtering for specific stocks
- Category filtering for event types
- Date range analysis for time periods
- Combined multi-dimensional filtering
- Custom metric calculations

### Caching and Performance

**Analytics Caching:**
- Intelligent cache management with 5-minute TTL
- Cache invalidation on data updates
- Per-query parameter caching
- Memory-efficient result storage
- Cache hit/miss monitoring

**Performance Optimization:**
- Database query optimization with indexes
- Batch processing for large datasets
- Asynchronous computation for complex analytics
- Connection pooling for concurrent requests
- Result pagination for large result sets

**Cache Management:**
```bash
POST /analytics/cache/clear
```

Clears all analytics cache for fresh data retrieval.

### Interactive Dashboard Components

**Summary Cards:**
- Total events with trend indicators
- Weekly and daily event volumes
- Average impact scores with changes
- High-impact event counts
- Headline coverage percentages

**Charts and Visualizations:**
- Event volume timeline (7-day trend)
- Impact score progression over time
- Category distribution pie charts
- Status breakdown visualizations
- Source performance comparisons

**Data Tables:**
- Most active symbols with event counts
- Trending categories with growth rates
- Source reliability metrics
- Performance rankings and statistics
- Real-time connection monitoring

### Usage Examples

**Basic Metrics Retrieval:**
```python
import aiohttp

async def get_metrics():
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8010/analytics/metrics') as resp:
            return await resp.json()
```

**Filtered Analysis:**
```python
# Get earnings events for specific symbols
params = {
    'symbols': 'AAPL,MSFT,GOOGL',
    'categories': 'earnings',
    'start_date': '2025-01-01T00:00:00Z'
}

async with session.get('/analytics/metrics', params=params) as resp:
    metrics = await resp.json()
```

**Time Series Analysis:**
```python
# Get daily event counts for the last 30 days
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=30)

params = {
    'metric': 'event_count',
    'start_date': start_date.isoformat(),
    'end_date': end_date.isoformat(),
    'interval': '1d'
}

async with session.get('/analytics/timeseries', params=params) as resp:
    series_data = await resp.json()
```

### Integration with Other Services

**Market Data Integration:**
- Symbol-based event filtering
- Cross-reference with price data
- Impact score correlation analysis
- Volume and volatility relationships

**Portfolio Service Integration:**
- Position-based event filtering
- Risk assessment analytics
- Portfolio impact analysis
- Holdings-specific metrics

**Strategy Service Integration:**
- Strategy performance correlation
- Event-driven signal analysis
- Backtesting support data
- Performance attribution metrics

### Demo and Testing

Run the comprehensive analytics demonstration:

```bash
cd examples
python analytics_demo.py
```

This demo showcases:
- Complete analytics API coverage
- Dashboard data retrieval and visualization
- Metrics calculation with various filters
- Time series analysis and trend detection
- Performance reporting and rankings
- Interactive web dashboard access
- Cache management and optimization

### Monitoring and Observability

**Analytics Metrics:**
- Query performance and response times
- Cache hit ratios and efficiency
- Database query optimization
- Memory usage and resource consumption
- Error rates and failure modes

**Dashboard Performance:**
- Page load times and rendering speed
- Chart rendering performance
- Real-time update frequency
- User interaction responsiveness
- Data freshness indicators

The analytics system provides comprehensive insights into event data patterns, enabling data-driven decision making for trading strategies and market analysis.
