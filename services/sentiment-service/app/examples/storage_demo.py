"""
Storage Integration Demonstration
Shows how sentiment data flows from collection to database storage
"""

import asyncio
from datetime import datetime
from typing import List
from sqlalchemy.orm import Session

from ..core.database import get_db, SentimentPost, SentimentNews, CollectionStatus
from ..services.sentiment_storage import sentiment_storage
from ..models.schemas import SocialPost, NewsArticle

async def demonstrate_storage_flow():
    """Demonstrate complete storage flow for sentiment data"""
    
    print("🗄️  Sentiment Data Storage Flow Demonstration")
    print("=" * 60)
    
    # Get database session
    db = next(get_db())
    
    try:
        # 1. Store sample social media posts
        print("\n📱 Storing Social Media Posts...")
        
        sample_posts = [
            SocialPost(
                id="tweet_123456",
                platform="twitter",
                author="trader_mike",
                content="$AAPL looking strong today! Bullish on tech 🚀",
                timestamp=datetime.now(),
                url="https://twitter.com/trader_mike/status/123456",
                engagement={"likes": 42, "retweets": 15, "replies": 8}
            ),
            SocialPost(
                id="reddit_789012",
                platform="reddit",
                author="InvestorJoe",
                content="TSLA might face headwinds this quarter. Bearish sentiment growing.",
                timestamp=datetime.now(),
                url="https://reddit.com/r/stocks/789012",
                engagement={"upvotes": 23, "comments": 12}
            ),
            SocialPost(
                id="threads_345678",
                platform="threads",
                author="market_analyst",
                content="NVDA earnings coming up. Market seems neutral, waiting for guidance.",
                timestamp=datetime.now(),
                url="https://threads.net/@market_analyst/345678",
                engagement={"likes": 67, "shares": 5}
            )
        ]
        
        stored_posts = []
        for post in sample_posts:
            # Extract symbol from content (simplified)
            symbol = "AAPL" if "AAPL" in post.content else "TSLA" if "TSLA" in post.content else "NVDA"
            
            stored_post = await sentiment_storage.store_social_post(db, post, symbol)
            stored_posts.append(stored_post)
            
            print(f"  ✅ Stored {post.platform} post for ${symbol}")
            print(f"     Sentiment: {stored_post.sentiment_label} ({stored_post.sentiment_score:.2f})")
            print(f"     Confidence: {stored_post.confidence:.2f}")
        
        # 2. Store sample news articles
        print("\n📰 Storing News Articles...")
        
        sample_news = [
            NewsArticle(
                id="reuters_001",
                source="Reuters",
                title="Apple Reports Strong Q4 Earnings, Beats Expectations",
                content="Apple Inc. reported quarterly earnings that exceeded analyst expectations...",
                author="Tech Reporter",
                url="https://reuters.com/technology/apple-earnings-001",
                published_at=datetime.now()
            ),
            NewsArticle(
                id="bloomberg_002",
                source="Bloomberg",
                title="Tesla Faces Production Challenges in Q1",
                content="Tesla's production numbers fell short of targets due to supply chain issues...",
                author="Auto Industry Analyst",
                url="https://bloomberg.com/news/tesla-production-002",
                published_at=datetime.now()
            )
        ]
        
        stored_news = []
        for article in sample_news:
            symbol = "AAPL" if "Apple" in article.title else "TSLA"
            
            stored_article = await sentiment_storage.store_news_article(db, article, symbol)
            stored_news.append(stored_article)
            
            print(f"  ✅ Stored {article.source} article for ${symbol}")
            print(f"     Sentiment: {stored_article.sentiment_label} ({stored_article.sentiment_score:.2f})")
            print(f"     Relevance: {stored_article.relevance_score:.2f}")
        
        # 3. Show collection status updates
        print("\n📊 Collection Status Updates...")
        
        platforms = ["twitter", "reddit", "threads", "news"]
        symbols = ["AAPL", "TSLA", "NVDA"]
        
        for platform in platforms:
            for symbol in symbols:
                posts_count = len([p for p in stored_posts if p.symbol == symbol and p.platform == platform])
                if posts_count > 0 or (platform == "news" and symbol in ["AAPL", "TSLA"]):
                    count = posts_count if platform != "news" else len([n for n in stored_news if n.symbol == symbol])
                    sentiment_storage.update_collection_status(
                        db, platform, symbol, posts_collected=count, errors=0
                    )
                    print(f"  ✅ Updated {platform} status for ${symbol}: {count} items")
        
        # 4. Demonstrate data retrieval
        print("\n🔍 Retrieving Stored Data...")
        
        for symbol in ["AAPL", "TSLA", "NVDA"]:
            print(f"\n  📈 Data for ${symbol}:")
            
            # Get recent posts
            recent_posts = sentiment_storage.get_recent_posts(db, symbol, hours=24)
            print(f"    Recent posts: {len(recent_posts)}")
            
            # Get sentiment summary
            summary = sentiment_storage.get_sentiment_summary(db, symbol, "1d")
            if summary:
                print(f"    Average sentiment: {summary.get('average_sentiment', 0):.2f}")
                print(f"    Total mentions: {summary.get('total_mentions', 0)}")
                print(f"    Distribution: {summary.get('sentiment_distribution', {})}")
        
        # 5. Show aggregated analytics
        print("\n📈 Computing Aggregated Analytics...")
        
        for symbol in ["AAPL", "TSLA", "NVDA"]:
            await sentiment_storage.compute_aggregates(db, symbol, "1h")
            print(f"  ✅ Computed hourly aggregates for ${symbol}")
        
        # 6. Display collection statistics
        print("\n📊 Overall Collection Statistics...")
        
        stats = sentiment_storage.get_collection_stats(db)
        print(f"  Total platforms monitored: {stats.get('total_platforms', 0)}")
        print(f"  Healthy platforms: {stats.get('healthy_platforms', 0)}")
        print(f"  Total posts collected: {stats.get('total_posts_collected', 0)}")
        
        print(f"\n  Platform breakdown:")
        for platform, data in stats.get('platform_stats', {}).items():
            print(f"    {platform}: {data.get('posts_collected', 0)} posts, {data.get('errors', 0)} errors")
        
        print("\n✅ Storage demonstration completed successfully!")
        print("\n🔑 Key Features Demonstrated:")
        print("   • Complete data persistence for all social media posts")
        print("   • Full sentiment analysis and storage")
        print("   • News article processing and storage")
        print("   • Collection status monitoring")
        print("   • Data retrieval and analytics")
        print("   • Pre-computed aggregations")
        print("   • Platform health tracking")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise
    finally:
        db.close()

# Example of real-time monitoring query
async def show_monitoring_capabilities(db: Session):
    """Show real-time monitoring capabilities"""
    
    print("\n🔴 Real-time Monitoring Capabilities")
    print("=" * 50)
    
    # Recent activity view
    print("\n📊 Recent Activity (SQL View):")
    print("""
    SELECT type, symbol, source, sentiment_label, timestamp 
    FROM recent_sentiment_activity 
    ORDER BY timestamp DESC 
    LIMIT 10;
    """)
    
    # Sentiment trends
    print("\n📈 Sentiment Trends (TimescaleDB Query):")
    print("""
    SELECT 
        DATE_TRUNC('hour', post_timestamp) as hour,
        symbol,
        AVG(sentiment_score) as avg_sentiment,
        COUNT(*) as mentions
    FROM sentiment_posts 
    WHERE post_timestamp >= NOW() - INTERVAL '24 hours'
    GROUP BY hour, symbol 
    ORDER BY hour DESC;
    """)
    
    # Platform comparison
    print("\n🔄 Platform Comparison:")
    print("""
    SELECT 
        platform,
        symbol,
        AVG(sentiment_score) as avg_sentiment,
        COUNT(*) as posts
    FROM sentiment_posts 
    WHERE post_timestamp >= NOW() - INTERVAL '7 days'
    GROUP BY platform, symbol
    ORDER BY symbol, avg_sentiment DESC;
    """)

if __name__ == "__main__":
    asyncio.run(demonstrate_storage_flow())