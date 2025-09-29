"""
Earnings Monitoring Dashboard Demo
Demonstrates comprehensive quarterly and yearly financial reports tracking
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import json

from ..services.earnings_monitor import earnings_monitor

async def demo_earnings_monitoring():
    """Comprehensive demo of earnings monitoring capabilities"""
    
    print("📊 EARNINGS & FINANCIAL REPORTS MONITORING DEMO")
    print("=" * 70)
    
    # Test symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
    
    # 1. Earnings Calendar
    print("\n📅 EARNINGS CALENDAR (Next 30 Days)")
    print("-" * 50)
    
    start_date = date.today()
    end_date = start_date + timedelta(days=30)
    
    calendar = await earnings_monitor.get_earnings_calendar(start_date, end_date, symbols)
    
    for date_str, earnings_day in calendar.items():
        print(f"\n📅 {date_str}:")
        print(f"   Events: {len(earnings_day.events)}")
        print(f"   High Impact Companies: {earnings_day.high_impact_count}")
        print(f"   Total Market Cap: ${earnings_day.market_cap_total:.1f}B")
        
        for event in earnings_day.events[:3]:  # Show first 3
            print(f"   • {event.symbol} ({event.company_name})")
            print(f"     Period: {event.period_type} {event.fiscal_year}")
            print(f"     Time: {event.announcement_time}")
            if event.estimated_eps:
                print(f"     Est. EPS: ${event.estimated_eps:.2f}")
    
    # 2. Upcoming Earnings (High Impact)
    print("\n🔔 UPCOMING HIGH-IMPACT EARNINGS")
    print("-" * 50)
    
    upcoming = await earnings_monitor.get_upcoming_earnings(days_ahead=14, min_market_cap=50.0)
    
    for event in upcoming[:10]:  # Top 10
        days_until = (event.report_date - date.today()).days
        print(f"• {event.symbol} - {event.company_name}")
        print(f"  Reports in {days_until} days ({event.report_date})")
        print(f"  Period: {event.period_type} {event.fiscal_year}")
        if event.estimated_eps and event.estimated_revenue:
            print(f"  Estimates: EPS ${event.estimated_eps:.2f}, Revenue ${event.estimated_revenue/1e9:.1f}B")
        print()
    
    # 3. Quarterly Performance Tracking
    print("\n📈 QUARTERLY PERFORMANCE TRACKING")
    print("-" * 50)
    
    for symbol in symbols[:3]:  # Track top 3 companies
        print(f"\n🏢 {symbol} - Last 8 Quarters Performance:")
        
        quarterly_data = await earnings_monitor.track_quarterly_performance(symbol, 8)
        
        if quarterly_data:
            print("   Quarter | Revenue Growth | EPS Growth | Net Margin | ROE")
            print("   --------|----------------|------------|------------|-----")
            
            for q in quarterly_data[:6]:  # Show last 6 quarters
                print(f"   {q.quarter} {q.fiscal_year}  |     {q.revenue_growth_yoy:6.1f}%     |   {q.eps_growth_yoy:6.1f}%   |   {q.net_margin:6.1f}%   | {q.roe:5.1f}%")
    
    # 4. Earnings Trends Analysis
    print("\n🔍 EARNINGS TRENDS ANALYSIS")
    print("-" * 50)
    
    for symbol in symbols[:3]:
        print(f"\n📊 {symbol} Trends Analysis:")
        
        trends = await earnings_monitor.analyze_earnings_trends(symbol)
        
        if trends:
            print(f"   Revenue Trend: {trends['revenue_trend']['direction'].upper()}")
            print(f"   - Average Growth: {trends['revenue_trend']['avg_growth']:.1f}%")
            print(f"   - Acceleration: {trends['revenue_trend']['acceleration']:.1f}%")
            
            print(f"   EPS Trend: {trends['eps_trend']['direction'].upper()}")
            print(f"   - Average Growth: {trends['eps_trend']['avg_growth']:.1f}%")
            
            print(f"   Margin Trend: {trends['margin_trend']['direction'].upper()}")
            print(f"   - Current Margin: {trends['margin_trend']['current']:.1f}%")
            
            print(f"   Earnings Surprise Rate: {trends['earnings_surprise_rate']*100:.1f}%")
            print(f"   Consistency Score: {trends['consistency_score']:.1f}/100")
            
            growth_quality = trends['growth_quality']
            print(f"   Growth Quality: {growth_quality['quality'].upper()} ({growth_quality['score']}/4)")
    
    # 5. Sector Analysis
    print("\n🏭 SECTOR EARNINGS ANALYSIS")
    print("-" * 50)
    
    sectors = ["technology", "healthcare", "financials"]
    
    for sector in sectors[:2]:  # Analyze 2 sectors
        print(f"\n🔬 {sector.upper()} Sector Analysis:")
        
        sector_data = await earnings_monitor.monitor_sector_earnings(sector, "current_quarter")
        
        if sector_data:
            print(f"   Companies Analyzed: {sector_data['companies_count']}")
            print(f"   Reporting Complete: {sector_data['reporting_complete']}")
            print(f"   Beat Estimates: {sector_data['beat_estimates']} | Missed: {sector_data['missed_estimates']}")
            print(f"   Average Surprise: {sector_data['avg_surprise']:.1f}%")
            print(f"   Avg Revenue Growth: {sector_data['revenue_growth_avg']:.1f}%")
            
            print(f"   Top Performers:")
            for company in sector_data['company_details'][:3]:
                if company['eps_surprise']:
                    print(f"   • {company['symbol']}: {company['eps_surprise']:+.1f}% surprise, {company['revenue_growth']:+.1f}% growth")
    
    # 6. Financial Reports Monitoring Summary
    print("\n📋 FINANCIAL REPORTS MONITORING CAPABILITIES")
    print("-" * 50)
    
    monitoring_features = {
        "Earnings Calendar": "Track upcoming earnings dates across all symbols",
        "Quarterly Performance": "Monitor revenue, EPS, margins, ROE trends over time",
        "Earnings Trends": "Analyze growth patterns, consistency, and quality",
        "Sector Analysis": "Compare earnings performance across industry sectors", 
        "Earnings Surprises": "Track historical beat/miss rates and surprise patterns",
        "Guidance Analysis": "Monitor management guidance accuracy and updates",
        "Margin Analysis": "Track gross, operating, and net margin trends",
        "Growth Quality": "Assess sustainability and quality of earnings growth",
        "Consistency Scoring": "Rate earnings predictability and reliability",
        "Alert System": "Automated notifications for earnings events and surprises"
    }
    
    for feature, description in monitoring_features.items():
        print(f"✅ {feature:<25} | {description}")
    
    # 7. Key Metrics Dashboard
    print("\n📊 KEY FINANCIAL METRICS DASHBOARD")
    print("-" * 50)
    
    dashboard_metrics = [
        "Revenue Growth (YoY/QoQ)",
        "Earnings Per Share Growth", 
        "Gross/Operating/Net Margins",
        "Return on Equity (ROE)",
        "Return on Assets (ROA)",
        "Free Cash Flow",
        "Debt-to-Equity Ratio",
        "Price-to-Earnings Ratio",
        "Price-to-Book Ratio",
        "Earnings Surprise Rate",
        "Guidance Beat Rate",
        "Consistency Score"
    ]
    
    for i, metric in enumerate(dashboard_metrics, 1):
        print(f"{i:2d}. {metric}")
    
    # 8. Monitoring Frequency & Alerts
    print("\n🔔 MONITORING SCHEDULE & ALERTS")
    print("-" * 50)
    
    monitoring_schedule = {
        "Daily": [
            "Check earnings calendar for next 7 days",
            "Monitor pre-market earnings releases",
            "Track after-hours earnings announcements"
        ],
        "Weekly": [
            "Analyze sector earnings performance",
            "Update quarterly trend analysis",
            "Review guidance changes and updates"
        ],
        "Monthly": [
            "Generate comprehensive earnings reports",
            "Update consistency and quality scores",
            "Benchmark against sector peers"
        ],
        "Quarterly": [
            "Deep dive into financial statement analysis",
            "Update annual trend projections",
            "Refresh earnings models and estimates"
        ]
    }
    
    for frequency, tasks in monitoring_schedule.items():
        print(f"\n{frequency} Tasks:")
        for task in tasks:
            print(f"  • {task}")
    
    print("\n" + "=" * 70)
    print("✅ COMPREHENSIVE EARNINGS MONITORING SYSTEM READY")
    print("🎯 Tracking quarterly & yearly financial reports automatically")
    print("📈 Provides deep insights into earnings trends and performance")
    print("🔔 Automated alerts for key earnings events and surprises")
    print("=" * 70)

async def demo_api_usage():
    """Demo API endpoints for earnings monitoring"""
    
    print("\n🔌 EARNINGS MONITORING API ENDPOINTS")
    print("-" * 50)
    
    api_examples = {
        "GET /earnings/calendar": {
            "description": "Get earnings calendar for date range",
            "example": "?start_date=2024-01-01&end_date=2024-01-31&symbols=AAPL,MSFT",
            "response": "Earnings events with dates, estimates, and market impact"
        },
        "GET /earnings/upcoming": {
            "description": "Get upcoming earnings (next 30 days)",
            "example": "?days_ahead=30&min_market_cap=10",
            "response": "Filtered earnings events by market cap"
        },
        "GET /earnings/{symbol}/monitor": {
            "description": "Monitor quarterly performance for symbol",
            "example": "/earnings/AAPL/monitor?quarters_back=12",
            "response": "Quarterly data + trends analysis"
        },
        "GET /earnings/sector/{sector}": {
            "description": "Monitor sector-wide earnings performance",
            "example": "/earnings/sector/technology?period=current_quarter",
            "response": "Sector-wide earnings analysis and company details"
        },
        "POST /earnings/{symbol}/alerts": {
            "description": "Setup earnings monitoring alerts",
            "example": "Body: {\"days_before_earnings\": 7, \"surprise_threshold\": 5.0}",
            "response": "Alert configuration confirmation"
        },
        "GET /earnings/trends/revenue": {
            "description": "Get revenue growth trends",
            "example": "?symbols=AAPL,MSFT,GOOGL&quarters=8",
            "response": "Revenue growth trends across companies"
        },
        "GET /earnings/trends/margins": {
            "description": "Get margin trends analysis",
            "example": "?symbols=AAPL,MSFT&quarters=8",
            "response": "Gross, operating, net margin trends"
        }
    }
    
    for endpoint, details in api_examples.items():
        print(f"\n📡 {endpoint}")
        print(f"   Purpose: {details['description']}")
        print(f"   Example: {details['example']}")
        print(f"   Returns: {details['response']}")

if __name__ == "__main__":
    print("Starting Earnings Monitoring Demo...")
    asyncio.run(demo_earnings_monitoring())
    asyncio.run(demo_api_usage())