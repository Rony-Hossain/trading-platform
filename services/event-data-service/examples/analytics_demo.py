#!/usr/bin/env python3
"""
Event Analytics Dashboard Demo

Demonstrates the analytics and reporting capabilities of the Event Data Service,
including metrics calculation, trend analysis, and dashboard functionality.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import webbrowser
import time


class AnalyticsDemoClient:
    """Demo client for Event Data Service Analytics."""
    
    def __init__(self, base_url: str = "http://localhost:8010"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_service_health(self) -> bool:
        """Check if the Event Data Service is running."""
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    print(f"✅ Event Data Service is healthy: {health_data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ Event Data Service health check failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"❌ Failed to connect to Event Data Service: {e}")
            return False
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        async with self.session.get(f"{self.base_url}/analytics/dashboard") as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                raise Exception(f"Failed to get dashboard data: {resp.status}")
    
    async def get_metrics(self, **params) -> Dict[str, Any]:
        """Get event metrics with optional filters."""
        async with self.session.get(f"{self.base_url}/analytics/metrics", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                raise Exception(f"Failed to get metrics: {resp.status}")
    
    async def get_time_series(self, metric: str, start_date: datetime, end_date: datetime, 
                             interval: str = "1d", **params) -> Dict[str, Any]:
        """Get time series data for a metric."""
        params.update({
            'metric': metric,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'interval': interval
        })
        
        async with self.session.get(f"{self.base_url}/analytics/timeseries", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                raise Exception(f"Failed to get time series data: {resp.status}")
    
    async def get_trend_analysis(self, metric: str, period_days: int = 30, **params) -> Dict[str, Any]:
        """Get trend analysis for a metric."""
        params.update({
            'metric': metric,
            'period_days': period_days
        })
        
        async with self.session.get(f"{self.base_url}/analytics/trends", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                raise Exception(f"Failed to get trend analysis: {resp.status}")
    
    async def get_performance_report(self, period_days: int = 7, limit: int = 10) -> Dict[str, Any]:
        """Get performance report."""
        params = {
            'period_days': period_days,
            'limit': limit
        }
        
        async with self.session.get(f"{self.base_url}/analytics/performance", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                raise Exception(f"Failed to get performance report: {resp.status}")
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear analytics cache."""
        async with self.session.post(f"{self.base_url}/analytics/cache/clear") as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                raise Exception(f"Failed to clear cache: {resp.status}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_metrics_summary(metrics: Dict[str, Any]):
    """Print a formatted metrics summary."""
    data = metrics.get('metrics', {})
    
    print(f"📊 Total Events: {data.get('total_events', 0):,}")
    print(f"⭐ Average Impact Score: {data.get('average_impact_score', 0):.2f}")
    print(f"🔥 High Impact Events: {data.get('high_impact_events', 0):,}")
    print(f"📰 Events with Headlines: {data.get('events_with_headlines', 0):,}")
    print(f"📄 Total Headlines: {data.get('total_headlines', 0):,}")
    
    # Top categories
    categories = data.get('events_by_category', {})
    if categories:
        print(f"\n📂 Top Categories:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {category}: {count:,}")
    
    # Top symbols
    symbols = data.get('events_by_symbol', {})
    if symbols:
        print(f"\n🏢 Top Symbols:")
        for symbol, count in list(symbols.items())[:5]:
            print(f"   • {symbol}: {count:,}")


def print_trend_analysis(trend: Dict[str, Any]):
    """Print formatted trend analysis."""
    analysis = trend.get('analysis', {})
    
    growth_rate = analysis.get('growth_rate', 0)
    direction = analysis.get('trend_direction', 'stable')
    volatility = analysis.get('volatility', 0)
    
    # Direction emoji
    direction_emoji = {"up": "📈", "down": "📉", "stable": "➡️"}.get(direction, "➡️")
    
    print(f"{direction_emoji} Trend: {direction.upper()}")
    print(f"📊 Growth Rate: {growth_rate:+.2f}%")
    print(f"🌊 Volatility: {volatility:.2f}%")
    print(f"🎯 Peak: {analysis.get('peak_value', 0)} events")


def print_performance_report(report: Dict[str, Any]):
    """Print formatted performance report."""
    data = report.get('report', {})
    
    # Most active symbols
    active_symbols = data.get('most_active_symbols', [])
    if active_symbols:
        print("🏆 Most Active Symbols:")
        for symbol, count in active_symbols[:5]:
            print(f"   • {symbol}: {count:,} events")
    
    # Trending categories
    trending_categories = data.get('trending_categories', [])
    if trending_categories:
        print(f"\n🔥 Trending Categories:")
        for category_data in trending_categories[:5]:
            category = category_data.get('category', 'Unknown')
            count = category_data.get('event_count', 0)
            growth = category_data.get('growth_rate', 0)
            growth_emoji = "📈" if growth > 0 else "📉" if growth < 0 else "➡️"
            print(f"   • {category}: {count:,} events {growth_emoji} {growth:+.1f}%")
    
    # Source reliability
    source_reliability = data.get('source_reliability', {})
    if source_reliability:
        print(f"\n📡 Source Reliability:")
        for source, stats in list(source_reliability.items())[:3]:
            print(f"   • {source}: {stats.get('event_count', 0)} events, "
                  f"{stats.get('headlines_per_event', 0):.2f} headlines/event")


async def run_analytics_demo():
    """Run comprehensive analytics demonstration."""
    print("🔍 Event Data Service - Analytics Dashboard Demo")
    print("=" * 60)
    
    async with AnalyticsDemoClient() as client:
        # Check service health
        if not await client.check_service_health():
            print("❌ Please ensure the Event Data Service is running on localhost:8010")
            return
        
        try:
            # 1. Dashboard Overview
            print_section("📊 Dashboard Overview")
            dashboard_data = await client.get_dashboard_data()
            
            summary = dashboard_data.get('summary', {})
            print(f"📈 Total Events: {summary.get('total_events', 0):,}")
            print(f"📅 Weekly Events: {summary.get('weekly_events', 0):,}")
            print(f"📆 Daily Events: {summary.get('daily_events', 0):,}")
            print(f"⭐ Avg Impact Score: {summary.get('avg_impact_score', 0):.2f}")
            print(f"📰 Headline Coverage: {summary.get('headline_coverage', 0):.1f}%")
            
            # 2. Detailed Metrics
            print_section("📊 Detailed Metrics")
            metrics = await client.get_metrics()
            print_metrics_summary(metrics)
            
            # 3. Filtered Metrics (example with specific symbols)
            print_section("🔍 Filtered Metrics (AAPL, MSFT, GOOGL)")
            filtered_metrics = await client.get_metrics(symbols="AAPL,MSFT,GOOGL")
            print_metrics_summary(filtered_metrics)
            
            # 4. Time Series Analysis
            print_section("📈 Time Series Analysis")
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            # Event count trend
            event_series = await client.get_time_series(
                metric="event_count",
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            data_points = event_series.get('data_points', [])
            if data_points:
                total_events = sum(point.get('value', 0) for point in data_points)
                avg_daily = total_events / len(data_points) if data_points else 0
                print(f"📊 30-Day Event Timeline:")
                print(f"   • Total Events: {total_events:,}")
                print(f"   • Average Daily: {avg_daily:.1f}")
                print(f"   • Data Points: {len(data_points)}")
            
            # 5. Trend Analysis
            print_section("📈 Trend Analysis")
            
            # Event count trends
            event_trends = await client.get_trend_analysis(
                metric="event_count",
                period_days=30
            )
            print("📊 Event Count Trends (30 days):")
            print_trend_analysis(event_trends)
            
            # Impact score trends
            try:
                impact_trends = await client.get_trend_analysis(
                    metric="impact_score",
                    period_days=30
                )
                print(f"\n⭐ Impact Score Trends (30 days):")
                print_trend_analysis(impact_trends)
            except Exception as e:
                print(f"ℹ️  Impact score trends not available: {e}")
            
            # 6. Performance Report
            print_section("🏆 Performance Report")
            performance = await client.get_performance_report(period_days=7, limit=10)
            print_performance_report(performance)
            
            # 7. Cache Management
            print_section("🗄️ Cache Management")
            cache_result = await client.clear_cache()
            print(f"✅ {cache_result.get('message', 'Cache cleared')}")
            
            # 8. Dashboard Access
            print_section("🌐 Web Dashboard")
            dashboard_url = f"{client.base_url}/dashboard"
            print(f"🔗 Dashboard URL: {dashboard_url}")
            print("📝 Opening dashboard in browser...")
            
            try:
                webbrowser.open(dashboard_url)
                print("✅ Dashboard opened in browser")
            except Exception as e:
                print(f"❌ Failed to open browser: {e}")
                print(f"📝 Please manually navigate to: {dashboard_url}")
            
            # 9. API Endpoints Summary
            print_section("🔌 Available API Endpoints")
            endpoints = [
                ("GET /dashboard", "Interactive web dashboard"),
                ("GET /analytics/dashboard", "Dashboard data (JSON)"),
                ("GET /analytics/metrics", "Event metrics and statistics"),
                ("GET /analytics/timeseries", "Time series data"),
                ("GET /analytics/trends", "Trend analysis"),
                ("GET /analytics/performance", "Performance reports"),
                ("POST /analytics/cache/clear", "Clear analytics cache")
            ]
            
            for endpoint, description in endpoints:
                print(f"   • {endpoint:<30} - {description}")
            
            print_section("✅ Demo Complete")
            print("🎉 Analytics demo completed successfully!")
            print("📊 Check the web dashboard for interactive visualizations")
            print("🔗 API documentation available at /docs")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print("💡 Ensure the Event Data Service is running and has sample data")


def print_usage():
    """Print usage instructions."""
    print("""
🔍 Event Analytics Dashboard Demo

This demo showcases the analytics and reporting capabilities of the Event Data Service.

Features demonstrated:
• 📊 Comprehensive event metrics and statistics
• 📈 Time series analysis and trends
• 🏆 Performance reports and rankings
• 🔍 Filtering by symbols, categories, and date ranges
• 🌐 Interactive web dashboard
• 🗄️ Caching and performance optimization

Prerequisites:
• Event Data Service running on localhost:8010
• Sample event data in the database
• Python 3.8+ with aiohttp installed

Usage:
    python analytics_demo.py

The demo will:
1. Check service health
2. Fetch and display various analytics
3. Open the web dashboard in your browser
4. Show available API endpoints
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_usage()
    else:
        asyncio.run(run_analytics_demo())