"""
Alternative Data Integration Service
Wraps existing services (sentiment, events, fundamentals) as alternative data sources
and provides unified interface with quality monitoring and ROI tracking
"""
import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


# Prometheus metrics
ALTDATA_FETCH = Counter(
    'altdata_fetch_total',
    'Total alternative data fetches',
    ['source', 'status']
)

ALTDATA_LATENCY = Histogram(
    'altdata_fetch_latency_seconds',
    'Alternative data fetch latency',
    ['source']
)

ALTDATA_QUALITY = Gauge(
    'altdata_quality_score',
    'Alternative data quality score (0-1)',
    ['source']
)


@dataclass
class AltDataPoint:
    """Alternative data point"""
    source: str
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    quality_score: float  # 0.0 to 1.0


class AltDataIntegration:
    """
    Alternative Data Integration Service

    Treats existing platform services as alternative data sources:
    - Sentiment Service: Twitter, Reddit sentiment
    - Event Data Service: Earnings, corporate actions
    - Fundamentals Service: Earnings surprises, analyst ratings
    - Market Data Service: Options flow, unusual volume
    """

    def __init__(
        self,
        sentiment_service_url: str = "http://localhost:8003",
        event_service_url: str = "http://localhost:8010",
        fundamentals_service_url: str = "http://localhost:8002",
        market_data_service_url: str = "http://localhost:8001",
        timeout: int = 30
    ):
        self.sentiment_url = sentiment_service_url
        self.event_url = event_service_url
        self.fundamentals_url = fundamentals_service_url
        self.market_data_url = market_data_service_url
        self.timeout = timeout

        self.client = httpx.AsyncClient(timeout=timeout)

        logger.info("Alternative Data Integration initialized")

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    # ============================================================================
    # SENTIMENT DATA (Twitter, Reddit)
    # ============================================================================

    async def fetch_twitter_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Optional[AltDataPoint]:
        """
        Fetch Twitter sentiment for a symbol

        Args:
            symbol: Stock symbol
            lookback_hours: Hours to look back

        Returns:
            AltDataPoint with sentiment data
        """
        try:
            url = f"{self.sentiment_url}/api/v1/sentiment/twitter/{symbol}"
            params = {"lookback_hours": lookback_hours}

            response = await self.client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Calculate quality score based on sample size and recency
            tweet_count = data.get("tweet_count", 0)
            quality_score = min(1.0, tweet_count / 100)  # 100+ tweets = perfect quality

            ALTDATA_FETCH.labels(source="twitter", status="success").inc()

            return AltDataPoint(
                source="twitter_sentiment",
                symbol=symbol,
                timestamp=datetime.utcnow(),
                data={
                    "sentiment_score": data.get("sentiment_score"),
                    "tweet_count": tweet_count,
                    "positive_ratio": data.get("positive_ratio"),
                    "negative_ratio": data.get("negative_ratio"),
                    "trending": data.get("trending", False)
                },
                quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment for {symbol}: {e}")
            ALTDATA_FETCH.labels(source="twitter", status="error").inc()
            return None

    async def fetch_reddit_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Optional[AltDataPoint]:
        """
        Fetch Reddit sentiment for a symbol

        Args:
            symbol: Stock symbol
            lookback_hours: Hours to look back

        Returns:
            AltDataPoint with sentiment data
        """
        try:
            url = f"{self.sentiment_url}/api/v1/sentiment/reddit/{symbol}"
            params = {"lookback_hours": lookback_hours}

            response = await self.client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Quality based on post/comment count
            post_count = data.get("post_count", 0) + data.get("comment_count", 0)
            quality_score = min(1.0, post_count / 50)

            ALTDATA_FETCH.labels(source="reddit", status="success").inc()

            return AltDataPoint(
                source="reddit_sentiment",
                symbol=symbol,
                timestamp=datetime.utcnow(),
                data={
                    "sentiment_score": data.get("sentiment_score"),
                    "post_count": data.get("post_count"),
                    "comment_count": data.get("comment_count"),
                    "upvote_ratio": data.get("upvote_ratio")
                },
                quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment for {symbol}: {e}")
            ALTDATA_FETCH.labels(source="reddit", status="error").inc()
            return None

    # ============================================================================
    # EVENT DATA (Earnings, Corporate Actions)
    # ============================================================================

    async def fetch_upcoming_events(
        self,
        symbol: str,
        days_ahead: int = 30
    ) -> Optional[AltDataPoint]:
        """
        Fetch upcoming events (earnings, dividends, splits)

        Args:
            symbol: Stock symbol
            days_ahead: Days to look ahead

        Returns:
            AltDataPoint with event data
        """
        try:
            url = f"{self.event_url}/api/v1/events/upcoming/{symbol}"
            params = {"days": days_ahead}

            response = await self.client.get(url, params=params)
            response.raise_for_status()

            events = response.json()

            # Quality based on event count and dates
            quality_score = 1.0 if events else 0.5  # 0.5 if no events

            ALTDATA_FETCH.labels(source="events", status="success").inc()

            return AltDataPoint(
                source="corporate_events",
                symbol=symbol,
                timestamp=datetime.utcnow(),
                data={
                    "earnings_date": events.get("next_earnings"),
                    "dividend_date": events.get("next_dividend"),
                    "has_upcoming_earnings": bool(events.get("next_earnings")),
                    "days_to_earnings": events.get("days_to_earnings"),
                    "events_count": len(events)
                },
                quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Error fetching events for {symbol}: {e}")
            ALTDATA_FETCH.labels(source="events", status="error").inc()
            return None

    # ============================================================================
    # FUNDAMENTALS DATA (Earnings Surprises, Analyst Ratings)
    # ============================================================================

    async def fetch_earnings_surprise(
        self,
        symbol: str
    ) -> Optional[AltDataPoint]:
        """
        Fetch latest earnings surprise data

        Args:
            symbol: Stock symbol

        Returns:
            AltDataPoint with earnings surprise
        """
        try:
            url = f"{self.fundamentals_url}/api/v1/fundamentals/earnings/{symbol}"

            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()

            # Quality based on data recency
            earnings_date_str = data.get("earnings_date", "2020-01-01")
            try:
                earnings_date = datetime.fromisoformat(earnings_date_str.replace('Z', '+00:00'))
            except:
                earnings_date = datetime(2020, 1, 1)

            days_old = (datetime.utcnow() - earnings_date.replace(tzinfo=None)).days
            quality_score = max(0.0, 1.0 - (days_old / 90))  # Degrade over 90 days

            ALTDATA_FETCH.labels(source="earnings", status="success").inc()

            return AltDataPoint(
                source="earnings_surprise",
                symbol=symbol,
                timestamp=datetime.utcnow(),
                data={
                    "eps_surprise": data.get("eps_surprise"),
                    "revenue_surprise": data.get("revenue_surprise"),
                    "earnings_date": data.get("earnings_date"),
                    "beat_expectations": data.get("eps_surprise", 0) > 0
                },
                quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Error fetching earnings surprise for {symbol}: {e}")
            ALTDATA_FETCH.labels(source="earnings", status="error").inc()
            return None

    async def fetch_analyst_ratings(
        self,
        symbol: str
    ) -> Optional[AltDataPoint]:
        """
        Fetch analyst ratings data

        Args:
            symbol: Stock symbol

        Returns:
            AltDataPoint with analyst ratings
        """
        try:
            url = f"{self.fundamentals_url}/api/v1/fundamentals/ratings/{symbol}"

            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()

            # Quality based on number of analysts
            analyst_count = data.get("analyst_count", 0)
            quality_score = min(1.0, analyst_count / 10)  # 10+ analysts = perfect

            ALTDATA_FETCH.labels(source="analyst_ratings", status="success").inc()

            return AltDataPoint(
                source="analyst_ratings",
                symbol=symbol,
                timestamp=datetime.utcnow(),
                data={
                    "rating_avg": data.get("rating_avg"),
                    "price_target_avg": data.get("price_target_avg"),
                    "analyst_count": analyst_count,
                    "upgrades_30d": data.get("upgrades_30d", 0),
                    "downgrades_30d": data.get("downgrades_30d", 0)
                },
                quality_score=quality_score
            )

        except Exception as e:
            logger.error(f"Error fetching analyst ratings for {symbol}: {e}")
            ALTDATA_FETCH.labels(source="analyst_ratings", status="error").inc()
            return None

    # ============================================================================
    # UNIFIED INTERFACE
    # ============================================================================

    async def fetch_all_altdata(
        self,
        symbol: str
    ) -> Dict[str, Optional[AltDataPoint]]:
        """
        Fetch all alternative data for a symbol in parallel

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary mapping source name to data point
        """
        tasks = {
            "twitter_sentiment": self.fetch_twitter_sentiment(symbol),
            "reddit_sentiment": self.fetch_reddit_sentiment(symbol),
            "upcoming_events": self.fetch_upcoming_events(symbol),
            "earnings_surprise": self.fetch_earnings_surprise(symbol),
            "analyst_ratings": self.fetch_analyst_ratings(symbol)
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        altdata = {}
        for source, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {source} for {symbol}: {result}")
                altdata[source] = None
            else:
                altdata[source] = result

                # Update quality metric
                if result and result.quality_score is not None:
                    ALTDATA_QUALITY.labels(source=source).set(result.quality_score)

        return altdata

    def calculate_aggregate_quality(
        self,
        altdata: Dict[str, Optional[AltDataPoint]]
    ) -> float:
        """
        Calculate aggregate quality score across all alt-data sources

        Args:
            altdata: Dictionary of alt-data points

        Returns:
            Aggregate quality score (0.0 to 1.0)
        """
        scores = [
            dp.quality_score for dp in altdata.values()
            if dp is not None and dp.quality_score is not None
        ]

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for all alternative data sources

        Returns:
            Health status for each source
        """
        health = {}

        sources = {
            "sentiment_service": f"{self.sentiment_url}/health",
            "event_service": f"{self.event_url}/health",
            "fundamentals_service": f"{self.fundamentals_url}/health",
            "market_data_service": f"{self.market_data_url}/health"
        }

        for service, url in sources.items():
            try:
                response = await self.client.get(url, timeout=5.0)
                health[service] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "status_code": response.status_code
                }
            except Exception as e:
                health[service] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

        return health


# Example usage
async def main():
    """Example usage of AltDataIntegration"""
    logging.basicConfig(level=logging.INFO)

    integration = AltDataIntegration()

    try:
        # Fetch all alt-data for a symbol
        print("\n=== Fetching all alternative data for AAPL ===\n")
        altdata = await integration.fetch_all_altdata("AAPL")

        for source, data in altdata.items():
            if data:
                print(f"{source}:")
                print(f"  Quality: {data.quality_score:.2f}")
                print(f"  Data: {data.data}")
                print()

        # Calculate aggregate quality
        quality = integration.calculate_aggregate_quality(altdata)
        print(f"Aggregate Quality: {quality:.2f}\n")

        # Health check
        print("=== Health Check ===\n")
        health = await integration.health_check()
        for service, status in health.items():
            print(f"{service}: {status['status']}")

    finally:
        await integration.close()


if __name__ == "__main__":
    asyncio.run(main())
