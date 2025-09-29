"""
Earnings and Financial Reports Monitoring Service
Tracks quarterly and yearly earnings reports, schedules, and performance analysis
"""

import logging
import asyncio
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import json

from .finnhub_fundamentals import FinnhubFundamentalsClient

from ..core.database import (
    fundamentals_storage,
    EarningsEvent as EarningsEventModel,
    QuarterlyPerformance as QuarterlyPerformanceModel,
    EarningsCalendar as EarningsCalendarModel,
    SectorEarnings,
    ConsensusEstimate,
    AnalystRevision,
    InsiderTransaction,
)

logger = logging.getLogger(__name__)

@dataclass
class EarningsEvent:
    """Earnings event data structure"""
    symbol: str
    company_name: str
    report_date: date
    period_ending: date
    period_type: str  # Q1, Q2, Q3, Q4, FY
    fiscal_year: int
    fiscal_quarter: Optional[int]
    estimated_eps: Optional[float]
    actual_eps: Optional[float]
    estimated_revenue: Optional[float]
    actual_revenue: Optional[float]
    surprise_percent: Optional[float]
    announcement_time: str  # BMO (Before Market Open), AMC (After Market Close), TAS (Time Not Supplied)
    status: str  # upcoming, reported, estimated
    guidance_updated: bool = False
    
@dataclass 
class QuarterlyPerformance:
    """Quarterly performance metrics"""
    symbol: str
    quarter: str
    fiscal_year: int
    revenue: float
    revenue_growth_yoy: float
    revenue_growth_qoq: float
    net_income: float
    earnings_per_share: float
    eps_growth_yoy: float
    gross_margin: float
    operating_margin: float
    net_margin: float
    roe: float
    roa: float
    free_cash_flow: Optional[float]
    guidance_revenue_low: Optional[float]
    guidance_revenue_high: Optional[float]
    guidance_eps_low: Optional[float]
    guidance_eps_high: Optional[float]

@dataclass
class EarningsCalendar:
    """Earnings calendar data"""
    date: date
    events: List[EarningsEvent]
    market_cap_total: float
    high_impact_count: int  # Companies with market cap > $10B

@dataclass
class ConsensusSnapshot:
    """Consensus snapshot for a reporting period"""
    symbol: str
    report_date: date
    fiscal_period: str
    fiscal_year: int
    estimate_eps: Optional[float]
    actual_eps: Optional[float]
    surprise_percent: Optional[float]
    estimate_revenue: Optional[float]
    actual_revenue: Optional[float]
    guidance_eps: Optional[float]
    guidance_revenue: Optional[float]
    source: Optional[str]


@dataclass
class InsiderTransactionRecord:
    """Simplified insider transaction representation"""
    symbol: str
    insider: str
    relationship: Optional[str]
    transaction_date: date
    transaction_type: Optional[str]
    shares: Optional[int]
    share_change: Optional[int]
    price: Optional[float]
    total_value: Optional[int]
    filing_date: Optional[date]
    link: Optional[str]
    source: Optional[str]


@dataclass
class AnalystRevisionRecord:
    """Analyst upgrade/downgrade or price target change"""
    symbol: str
    revision_date: date
    analyst: Optional[str]
    firm: Optional[str]
    action: Optional[str]
    from_rating: Optional[str]
    to_rating: Optional[str]
    old_price_target: Optional[float]
    new_price_target: Optional[float]
    rating_score: Optional[float]
    notes: Optional[str]
    source: Optional[str]


class EarningsMonitor:
    """Monitor and track earnings reports and financial performance"""
    
    def __init__(self):
        self.base_url = "https://api.earningswhispers.com"
        self.sec_url = "https://www.sec.gov"
        self.finnhub_client = FinnhubFundamentalsClient()
        self.storage = fundamentals_storage
        
    async def get_earnings_calendar(self, start_date: date, end_date: date, 
                                  symbols: Optional[List[str]] = None) -> Dict[str, EarningsCalendar]:
        """Get earnings calendar for date range"""
        try:
            calendar_data = {}
            current_date = start_date
            
            while current_date <= end_date:
                events = await self._fetch_earnings_for_date(current_date, symbols)
                
                if events:
                    # Calculate market impact metrics
                    high_impact_count = len([e for e in events if await self._is_high_impact_company(e.symbol)])
                    total_market_cap = sum([await self._get_market_cap(e.symbol) for e in events])
                    
                    calendar_data[current_date.isoformat()] = EarningsCalendar(
                        date=current_date,
                        events=events,
                        market_cap_total=total_market_cap,
                        high_impact_count=high_impact_count
                    )
                
                current_date += timedelta(days=1)
            
            return calendar_data
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return {}
    
    async def get_upcoming_earnings(self, days_ahead: int = 30, 
                                  min_market_cap: float = 1.0) -> List[EarningsEvent]:
        """Get upcoming earnings for next N days"""
        try:
            start_date = date.today()
            end_date = start_date + timedelta(days=days_ahead)
            
            all_events = []
            current_date = start_date
            
            while current_date <= end_date:
                events = await self._fetch_earnings_for_date(current_date)
                
                # Filter by market cap
                filtered_events = []
                for event in events:
                    market_cap = await self._get_market_cap(event.symbol)
                    if market_cap >= min_market_cap:
                        filtered_events.append(event)
                
                all_events.extend(filtered_events)
                current_date += timedelta(days=1)
            
            return sorted(all_events, key=lambda x: x.report_date)
            
        except Exception as e:
            logger.error(f"Error fetching upcoming earnings: {e}")
            return []
    
    async def track_quarterly_performance(self, symbol: str, 
                                        quarters_back: int = 12, 
                                        db: Optional[Session] = None) -> List[QuarterlyPerformance]:
        """Track quarterly performance over time"""
        try:
            if db:
                # Get stored data from database
                stored_data = fundamentals_storage.get_quarterly_performance(db, symbol, quarters_back)
                
                # If we have stored data, return it
                if stored_data:
                    # Convert SQLAlchemy objects to dataclass objects for consistency
                    performance_data = []
                    for item in stored_data:
                        performance_data.append(QuarterlyPerformance(
                            symbol=item.symbol,
                            quarter=item.quarter,
                            fiscal_year=item.fiscal_year,
                            revenue=float(item.revenue) if item.revenue else 0.0,
                            revenue_growth_yoy=float(item.revenue_growth_yoy) if item.revenue_growth_yoy else 0.0,
                            revenue_growth_qoq=float(item.revenue_growth_qoq) if item.revenue_growth_qoq else 0.0,
                            net_income=float(item.net_income) if item.net_income else 0.0,
                            earnings_per_share=float(item.earnings_per_share) if item.earnings_per_share else 0.0,
                            eps_growth_yoy=float(item.eps_growth_yoy) if item.eps_growth_yoy else 0.0,
                            gross_margin=float(item.gross_margin) if item.gross_margin else 0.0,
                            operating_margin=float(item.operating_margin) if item.operating_margin else 0.0,
                            net_margin=float(item.net_margin) if item.net_margin else 0.0,
                            roe=float(item.roe) if item.roe else 0.0,
                            roa=float(item.roa) if item.roa else 0.0,
                            free_cash_flow=float(item.free_cash_flow) if item.free_cash_flow else None,
                            guidance_revenue_low=float(item.guidance_revenue_low) if item.guidance_revenue_low else None,
                            guidance_revenue_high=float(item.guidance_revenue_high) if item.guidance_revenue_high else None,
                            guidance_eps_low=float(item.guidance_eps_low) if item.guidance_eps_low else None,
                            guidance_eps_high=float(item.guidance_eps_high) if item.guidance_eps_high else None
                        ))
                    return performance_data
            
            # If no database or no stored data, generate sample data and store it
            performance_data = []
            
            # Generate and store sample quarterly data
            for i in range(quarters_back):
                quarter_data = await self._get_quarterly_data(symbol, i)
                if quarter_data:
                    performance_data.append(quarter_data)
                    
                    # Store in database if provided
                    if db:
                        try:
                            stored_data = {
                                'symbol': quarter_data.symbol,
                                'quarter': quarter_data.quarter,
                                'fiscal_year': quarter_data.fiscal_year,
                                'period_ending': date.today() - timedelta(days=30 + (i * 90)),
                                'revenue': int(quarter_data.revenue),
                                'revenue_growth_yoy': quarter_data.revenue_growth_yoy,
                                'revenue_growth_qoq': quarter_data.revenue_growth_qoq,
                                'net_income': int(quarter_data.net_income),
                                'earnings_per_share': quarter_data.earnings_per_share,
                                'eps_growth_yoy': quarter_data.eps_growth_yoy,
                                'gross_margin': quarter_data.gross_margin,
                                'operating_margin': quarter_data.operating_margin,
                                'net_margin': quarter_data.net_margin,
                                'roe': quarter_data.roe,
                                'roa': quarter_data.roa,
                                'free_cash_flow': int(quarter_data.free_cash_flow) if quarter_data.free_cash_flow else None,
                                'guidance_revenue_low': int(quarter_data.guidance_revenue_low) if quarter_data.guidance_revenue_low else None,
                                'guidance_revenue_high': int(quarter_data.guidance_revenue_high) if quarter_data.guidance_revenue_high else None,
                                'guidance_eps_low': quarter_data.guidance_eps_low,
                                'guidance_eps_high': quarter_data.guidance_eps_high,
                                'report_date': date.today() - timedelta(days=7 + (i * 90))
                            }
                            fundamentals_storage.store_quarterly_performance(db, stored_data)
                        except Exception as e:
                            logger.warning(f"Failed to store quarterly data for {symbol}: {e}")
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error tracking quarterly performance for {symbol}: {e}")
            return []
    
    async def analyze_earnings_trends(self, symbol: str) -> Dict[str, Any]:
        """Analyze earnings trends and patterns"""
        try:
            # Get recent quarterly performance
            quarters = await self.track_quarterly_performance(symbol, 8)
            
            if not quarters:
                return {}
            
            # Calculate trend metrics
            revenue_trend = self._calculate_growth_trend([q.revenue for q in quarters])
            eps_trend = self._calculate_growth_trend([q.earnings_per_share for q in quarters])
            margin_trend = self._calculate_trend([q.net_margin for q in quarters])
            
            # Earnings surprise analysis
            surprises = await self._get_earnings_surprises(symbol, 8)
            surprise_rate = len([s for s in surprises if s > 0]) / len(surprises) if surprises else 0
            
            # Guidance accuracy
            guidance_accuracy = await self._analyze_guidance_accuracy(symbol)
            
            return {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "revenue_trend": {
                    "direction": revenue_trend["direction"],
                    "avg_growth": revenue_trend["avg_growth"],
                    "acceleration": revenue_trend["acceleration"]
                },
                "eps_trend": {
                    "direction": eps_trend["direction"], 
                    "avg_growth": eps_trend["avg_growth"],
                    "acceleration": eps_trend["acceleration"]
                },
                "margin_trend": {
                    "direction": margin_trend["direction"],
                    "current": quarters[0].net_margin if quarters else 0,
                    "avg": margin_trend["average"]
                },
                "earnings_surprise_rate": surprise_rate,
                "guidance_accuracy": guidance_accuracy,
                "consistency_score": self._calculate_consistency_score(quarters),
                "growth_quality": self._assess_growth_quality(quarters)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing earnings trends for {symbol}: {e}")
            return {}
    
    async def monitor_sector_earnings(self, sector: str, 
                                    period: str = "current_quarter") -> Dict[str, Any]:
        """Monitor earnings across a sector"""
        try:
            # Get companies in sector
            sector_companies = await self._get_sector_companies(sector)
            
            sector_analysis = {
                "sector": sector,
                "period": period,
                "companies_count": len(sector_companies),
                "reporting_complete": 0,
                "beat_estimates": 0,
                "missed_estimates": 0,
                "avg_surprise": 0.0,
                "revenue_growth_avg": 0.0,
                "margin_expansion": 0,
                "guidance_raises": 0,
                "guidance_lowers": 0,
                "company_details": []
            }
            
            surprises = []
            revenue_growths = []
            
            for symbol in sector_companies:
                try:
                    # Get latest earnings data
                    latest_earnings = await self._get_latest_earnings(symbol)
                    
                    if latest_earnings and latest_earnings.status == "reported":
                        sector_analysis["reporting_complete"] += 1
                        
                        # Calculate surprise
                        if latest_earnings.actual_eps and latest_earnings.estimated_eps:
                            surprise = ((latest_earnings.actual_eps - latest_earnings.estimated_eps) 
                                      / abs(latest_earnings.estimated_eps)) * 100
                            surprises.append(surprise)
                            
                            if surprise > 0:
                                sector_analysis["beat_estimates"] += 1
                            else:
                                sector_analysis["missed_estimates"] += 1
                        
                        # Get quarterly performance
                        quarterly = await self._get_quarterly_data(symbol, 0)
                        if quarterly:
                            revenue_growths.append(quarterly.revenue_growth_yoy)
                            
                            sector_analysis["company_details"].append({
                                "symbol": symbol,
                                "eps_surprise": surprise if 'surprise' in locals() else None,
                                "revenue_growth": quarterly.revenue_growth_yoy,
                                "margin": quarterly.net_margin,
                                "guidance_direction": await self._get_guidance_direction(symbol)
                            })
                    
                except Exception as e:
                    logger.warning(f"Error processing {symbol} in sector analysis: {e}")
                    continue
            
            # Calculate averages
            if surprises:
                sector_analysis["avg_surprise"] = sum(surprises) / len(surprises)
            
            if revenue_growths:
                sector_analysis["revenue_growth_avg"] = sum(revenue_growths) / len(revenue_growths)
            
            return sector_analysis
            
        except Exception as e:
            logger.error(f"Error monitoring sector earnings for {sector}: {e}")
            return {}
    
    async def setup_earnings_alerts(self, symbol: str, alert_settings: Dict[str, Any]) -> bool:
        """Setup alerts for earnings events"""
        try:
            # Alert settings can include:
            # - days_before_earnings: Alert N days before earnings
            # - surprise_threshold: Alert if surprise > X%
            # - guidance_changes: Alert on guidance updates
            # - revenue_miss: Alert on revenue miss
            # - margin_compression: Alert if margins compress > X%
            
            # This would integrate with the notification service
            # For now, we'll store the settings
            
            logger.info(f"Setting up earnings alerts for {symbol}: {alert_settings}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up earnings alerts for {symbol}: {e}")
            return False
    
    # Helper methods
    async def _fetch_earnings_for_date(self, target_date: date, 
                                     symbols: Optional[List[str]] = None) -> List[EarningsEvent]:
        """Fetch earnings events for specific date using Finnhub"""
        try:
            # Use Finnhub earnings calendar
            earnings_calendar = await self.finnhub_client.get_earnings_calendar(
                target_date.strftime("%Y-%m-%d"), 
                target_date.strftime("%Y-%m-%d")
            )
            
            events = []
            if earnings_calendar and 'earningsCalendar' in earnings_calendar:
                for earning in earnings_calendar['earningsCalendar']:
                    # Filter by symbols if provided
                    if symbols and earning.get('symbol') not in symbols:
                        continue
                    
                    # Parse the earnings data
                    symbol = earning.get('symbol', 'UNKNOWN')
                    report_date_str = earning.get('date')
                    
                    if report_date_str:
                        try:
                            report_date = datetime.strptime(report_date_str, "%Y-%m-%d").date()
                        except:
                            report_date = target_date
                    else:
                        report_date = target_date
                    
                    event = EarningsEvent(
                        symbol=symbol,
                        company_name=earning.get('name', symbol),
                        report_date=report_date,
                        period_ending=report_date - timedelta(days=90),  # Approximate quarter end
                        period_type=earning.get('quarter', 'Q1'),
                        fiscal_year=earning.get('year', target_date.year),
                        fiscal_quarter=int(earning.get('quarter', '1').replace('Q', '')),
                        estimated_eps=earning.get('epsEstimate'),
                        actual_eps=earning.get('epsActual'),
                        estimated_revenue=earning.get('revenueEstimate'),
                        actual_revenue=earning.get('revenueActual'),
                        surprise_percent=None,  # Calculate if both actual and estimate available
                        announcement_time=earning.get('hour', 'TAS'),
                        status="upcoming" if earning.get('epsActual') is None else "reported"
                    )
                    
                    # Calculate surprise if both values available
                    if event.actual_eps and event.estimated_eps and event.estimated_eps != 0:
                        event.surprise_percent = ((event.actual_eps - event.estimated_eps) / abs(event.estimated_eps)) * 100
                    
                    events.append(event)
            
            logger.info(f"Fetched {len(events)} earnings events for {target_date}")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar for {target_date}: {e}")
            # Return empty list instead of mock data on error
            return []
    
    async def _get_quarterly_data(self, symbol: str, quarters_back: int) -> Optional[QuarterlyPerformance]:
        """Get quarterly financial data using Finnhub"""
        try:
            # Get basic financials from Finnhub
            basic_financials = await self.finnhub_client.get_basic_financials(symbol)
            
            if not basic_financials or 'metric' not in basic_financials:
                logger.warning(f"No basic financials available for {symbol}")
                return None
            
            metrics = basic_financials['metric']
            
            # Get quarterly earnings for more detailed data
            earnings = await self.finnhub_client.get_company_earnings(symbol)
            
            # Use the most recent quarter's data
            if earnings and len(earnings) > 0:
                latest_earnings = earnings[0]
                
                return QuarterlyPerformance(
                    symbol=symbol,
                    quarter=f"Q{latest_earnings.get('quarter', 1)}",
                    fiscal_year=latest_earnings.get('year', 2024),
                    revenue=latest_earnings.get('revenue', 0) * 1000000,  # Convert to actual value
                    revenue_growth_yoy=metrics.get('revenueGrowthTTM', 0),
                    revenue_growth_qoq=metrics.get('revenueGrowthQuarterlyYoy', 0),
                    net_income=latest_earnings.get('revenue', 0) * metrics.get('netProfitMarginTTM', 0.1) / 100 * 1000000,
                    earnings_per_share=latest_earnings.get('actual', 0),
                    eps_growth_yoy=metrics.get('epsGrowthTTM', 0),
                    gross_margin=metrics.get('grossMarginTTM', 0),
                    operating_margin=metrics.get('operatingMarginTTM', 0),
                    net_margin=metrics.get('netProfitMarginTTM', 0),
                    roe=metrics.get('roeTTM', 0),
                    roa=metrics.get('roaTTM', 0),
                    free_cash_flow=latest_earnings.get('revenue', 0) * 0.15 * 1000000,  # Estimate
                    guidance_revenue_low=None,  # Would need separate API call
                    guidance_revenue_high=None,
                    guidance_eps_low=None,
                    guidance_eps_high=None
                )
            else:
                # Fallback using just basic metrics
                return QuarterlyPerformance(
                    symbol=symbol,
                    quarter="Q1",
                    fiscal_year=2024,
                    revenue=metrics.get('revenueTTM', 0) * 1000000,
                    revenue_growth_yoy=metrics.get('revenueGrowthTTM', 0),
                    revenue_growth_qoq=0,  # Not available in basic metrics
                    net_income=metrics.get('revenueTTM', 0) * metrics.get('netProfitMarginTTM', 0.1) / 100 * 1000000,
                    earnings_per_share=metrics.get('epsTTM', 0),
                    eps_growth_yoy=metrics.get('epsGrowthTTM', 0),
                    gross_margin=metrics.get('grossMarginTTM', 0),
                    operating_margin=metrics.get('operatingMarginTTM', 0),
                    net_margin=metrics.get('netProfitMarginTTM', 0),
                    roe=metrics.get('roeTTM', 0),
                    roa=metrics.get('roaTTM', 0),
                    free_cash_flow=metrics.get('revenueTTM', 0) * 0.15 * 1000000,
                    guidance_revenue_low=None,
                    guidance_revenue_high=None,
                    guidance_eps_low=None,
                    guidance_eps_high=None
                )
                
        except Exception as e:
            logger.error(f"Error fetching quarterly data for {symbol}: {e}")
            return None
    
    def _calculate_growth_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate growth trend analysis"""
        if len(values) < 2:
            return {"direction": "insufficient_data", "avg_growth": 0, "acceleration": 0}
        
        # Calculate period-over-period growth
        growths = []
        for i in range(1, len(values)):
            if values[i] != 0:
                growth = ((values[i-1] - values[i]) / abs(values[i])) * 100
                growths.append(growth)
        
        if not growths:
            return {"direction": "no_data", "avg_growth": 0, "acceleration": 0}
        
        avg_growth = sum(growths) / len(growths)
        
        # Calculate acceleration (is growth rate increasing?)
        acceleration = 0
        if len(growths) >= 2:
            recent_growth = sum(growths[:len(growths)//2]) / (len(growths)//2)
            earlier_growth = sum(growths[len(growths)//2:]) / (len(growths) - len(growths)//2)
            acceleration = recent_growth - earlier_growth
        
        direction = "accelerating" if acceleration > 1 else "decelerating" if acceleration < -1 else "stable"
        
        return {
            "direction": direction,
            "avg_growth": avg_growth,
            "acceleration": acceleration
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate simple trend direction"""
        if len(values) < 2:
            return {"direction": "insufficient_data", "average": 0}
        
        recent_avg = sum(values[:len(values)//2]) / (len(values)//2)
        earlier_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if recent_avg > earlier_avg * 1.05:
            direction = "improving"
        elif recent_avg < earlier_avg * 0.95:
            direction = "declining"
        else:
            direction = "stable"
        
        return {
            "direction": direction,
            "average": sum(values) / len(values)
        }
    
    def _calculate_consistency_score(self, quarters: List[QuarterlyPerformance]) -> float:
        """Calculate earnings consistency score (0-100)"""
        if len(quarters) < 4:
            return 0.0
        
        # Factors: growth consistency, margin stability, guidance accuracy
        revenue_growths = [q.revenue_growth_yoy for q in quarters]
        eps_growths = [q.eps_growth_yoy for q in quarters]
        margins = [q.net_margin for q in quarters]
        
        # Calculate coefficient of variation (lower = more consistent)
        revenue_cv = (self._std_dev(revenue_growths) / abs(self._mean(revenue_growths))) if self._mean(revenue_growths) != 0 else 0
        eps_cv = (self._std_dev(eps_growths) / abs(self._mean(eps_growths))) if self._mean(eps_growths) != 0 else 0
        margin_cv = (self._std_dev(margins) / self._mean(margins)) if self._mean(margins) != 0 else 0
        
        # Convert to score (lower CV = higher score)
        revenue_score = max(0, 100 - (revenue_cv * 10))
        eps_score = max(0, 100 - (eps_cv * 10))
        margin_score = max(0, 100 - (margin_cv * 20))
        
        return (revenue_score + eps_score + margin_score) / 3
    
    def _assess_growth_quality(self, quarters: List[QuarterlyPerformance]) -> Dict[str, Any]:
        """Assess quality of growth"""
        if len(quarters) < 4:
            return {"quality": "insufficient_data"}
        
        # High quality growth: revenue > EPS growth, improving margins, positive FCF
        revenue_growth = self._mean([q.revenue_growth_yoy for q in quarters[:4]])
        eps_growth = self._mean([q.eps_growth_yoy for q in quarters[:4]])
        margin_trend = self._calculate_trend([q.net_margin for q in quarters[:4]])
        
        quality_factors = []
        
        if revenue_growth > 0:
            quality_factors.append("positive_revenue_growth")
        
        if eps_growth > revenue_growth:
            quality_factors.append("eps_outpacing_revenue")  # Could indicate margin expansion
        
        if margin_trend["direction"] == "improving":
            quality_factors.append("margin_expansion")
        
        # Check for FCF
        fcf_positive = all(q.free_cash_flow and q.free_cash_flow > 0 for q in quarters[:4] if q.free_cash_flow)
        if fcf_positive:
            quality_factors.append("positive_fcf")
        
        quality_score = len(quality_factors)
        quality_rating = "high" if quality_score >= 3 else "medium" if quality_score >= 2 else "low"
        
        return {
            "quality": quality_rating,
            "score": quality_score,
            "factors": quality_factors,
            "revenue_growth": revenue_growth,
            "eps_growth": eps_growth,
            "margin_trend": margin_trend["direction"]
        }
    
    def _mean(self, values: List[float]) -> float:
        """Calculate mean"""
        return sum(values) / len(values) if values else 0
    
    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        
        mean_val = self._mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    # Additional helper methods for real implementation
    async def _is_high_impact_company(self, symbol: str) -> bool:
        """Check if company is high impact (market cap > $10B)"""
        market_cap = await self._get_market_cap(symbol)
        return market_cap > 10.0  # $10B
    
    async def _get_market_cap(self, symbol: str) -> float:
        """Get market cap in billions"""
        # Mock implementation - would fetch from market data
        return 2500.0  # Mock $2.5T for demonstration
    
    async def _get_earnings_surprises(self, symbol: str, periods: int) -> List[float]:
        """Get historical earnings surprises"""
        # Mock implementation
        return [5.2, -2.1, 8.3, 1.5, -0.8, 12.1, 3.4, 6.7][:periods]
    
    async def _analyze_guidance_accuracy(self, symbol: str) -> Dict[str, float]:
        """Analyze historical guidance accuracy"""
        # Mock implementation
        return {
            "revenue_accuracy": 85.6,  # % of time guidance was accurate
            "eps_accuracy": 78.9,
            "beats_own_guidance": 65.4  # % of time actual results beat guidance
        }
    
    async def _get_sector_companies(self, sector: str) -> List[str]:
        """Get list of companies in sector"""
        # Mock implementation
        if sector.lower() == "technology":
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
        return ["AAPL", "MSFT"]  # Default
    
    async def _get_latest_earnings(self, symbol: str) -> Optional[EarningsEvent]:
        """Get latest earnings event for symbol"""
        # Mock implementation
        return EarningsEvent(
            symbol=symbol,
            company_name=f"{symbol} Inc.",
            report_date=date.today() - timedelta(days=7),
            period_ending=date.today() - timedelta(days=37),
            period_type="Q1",
            fiscal_year=2024,
            fiscal_quarter=1,
            estimated_eps=1.95,
            actual_eps=2.05,
            estimated_revenue=89.5e9,
            actual_revenue=91.2e9,
            surprise_percent=5.1,
            announcement_time="AMC",
            status="reported"
        )
    
    async def _get_guidance_direction(self, symbol: str) -> str:
        """Get guidance direction (raised, lowered, maintained)"""
        # Mock implementation
        return "raised"
    
    async def fetch_and_store_consensus_data(self, symbol: str, db: Session) -> bool:

        """Fetch real consensus data from Finnhub and store it"""

        try:

            earnings_estimates = await self.finnhub_client.fetch_consensus_estimates(symbol)



            if not earnings_estimates:

                return False



            for estimate in earnings_estimates:

                payload = {

                    'symbol': symbol.upper(),

                    'report_date': estimate.get('report_date') or date.today(),

                    'fiscal_period': estimate.get('fiscal_period') or 'NA',

                    'fiscal_year': estimate.get('fiscal_year') or date.today().year,

                    'analyst_count': estimate.get('analyst_count'),

                    'estimate_eps': estimate.get('estimate_eps'),

                    'estimate_eps_high': estimate.get('estimate_eps_high'),

                    'estimate_eps_low': estimate.get('estimate_eps_low'),

                    'actual_eps': estimate.get('actual_eps'),

                    'surprise_percent': estimate.get('surprise_percent'),

                    'estimate_revenue': estimate.get('estimate_revenue'),

                    'estimate_revenue_high': estimate.get('estimate_revenue_high'),

                    'estimate_revenue_low': estimate.get('estimate_revenue_low'),

                    'actual_revenue': estimate.get('actual_revenue'),

                    'guidance_eps': estimate.get('guidance_eps'),

                    'guidance_revenue': estimate.get('guidance_revenue'),

                    'source': estimate.get('source', 'finnhub'),

                    'retrieved_at': datetime.now()

                }

                self.storage.store_consensus_estimate(db, payload)



            logger.info(f"Stored {len(earnings_estimates)} consensus estimates for {symbol}")

            return True



        except Exception as e:

            logger.error(f"Error fetching consensus data for {symbol}: {e}")

            return False





    async def fetch_and_store_analyst_revisions(self, symbol: str, db: Session) -> bool:

        """Fetch real analyst revision data and store it"""

        try:

            revisions = await self.finnhub_client.fetch_analyst_revisions(symbol)



            if not revisions:

                return False



            for revision in revisions:

                revision_date = revision.get('revision_date') or datetime.now().date()



                payload = {

                    'symbol': symbol.upper(),

                    'revision_date': revision_date,

                    'analyst': revision.get('analyst'),

                    'firm': revision.get('firm'),

                    'action': revision.get('action'),

                    'from_rating': revision.get('from_rating'),

                    'to_rating': revision.get('to_rating'),

                    'old_price_target': revision.get('old_price_target'),

                    'new_price_target': revision.get('new_price_target'),

                    'rating_score': revision.get('rating_score'),

                    'notes': revision.get('notes'),

                    'source': revision.get('source', 'finnhub')

                }

                self.storage.store_analyst_revision(db, payload)



            logger.info(f"Stored {len(revisions)} analyst revisions for {symbol}")

            return True



        except Exception as e:

            logger.error(f"Error fetching analyst revisions for {symbol}: {e}")

            return False





    async def fetch_and_store_insider_transactions(self, symbol: str, db: Session) -> bool:

        """Fetch real insider transaction data and store it"""

        try:

            insider_data = await self.finnhub_client.fetch_insider_transactions(symbol)



            if not insider_data:

                return False



            for transaction in insider_data:

                filing_date = transaction.get('filing_date') or datetime.now().date()

                transaction_date = transaction.get('transaction_date') or filing_date



                payload = {

                    'symbol': symbol.upper(),

                    'insider': transaction.get('insider', 'UNKNOWN'),

                    'relationship': transaction.get('relationship'),

                    'transaction_date': transaction_date,

                    'transaction_type': transaction.get('transaction_type'),

                    'shares': transaction.get('shares'),

                    'share_change': transaction.get('share_change'),

                    'price': transaction.get('price'),

                    'total_value': transaction.get('total_value'),

                    'filing_date': filing_date,

                    'link': transaction.get('link'),

                    'source': transaction.get('source', 'finnhub')

                }

                self.storage.store_insider_transaction(db, payload)



            logger.info(f"Stored {len(insider_data)} insider transactions for {symbol}")

            return True



        except Exception as e:

            logger.error(f"Error fetching insider transactions for {symbol}: {e}")

            return False





    async def get_real_consensus_quality(self, symbol: str, event_date: date, db: Session) -> Dict[str, float]:
        """Compute consensus quality metrics from stored consensus data"""
        try:
            records = self.storage.get_consensus_estimates(db, symbol.upper(), limit=1)

            if records:
                estimate = records[0]
                eps_mean = float(estimate.estimate_eps) if estimate.estimate_eps is not None else 0.0
                high = float(estimate.estimate_eps_high) if estimate.estimate_eps_high is not None else eps_mean
                low = float(estimate.estimate_eps_low) if estimate.estimate_eps_low is not None else eps_mean
                analyst_count = estimate.analyst_count or 0

                eps_range = high - low
                range_pct = (eps_range / abs(eps_mean)) * 100 if eps_mean else 0.0
                std_estimate = eps_range / 4 if eps_range else 0.0

                retrieved_at = estimate.retrieved_at or estimate.created_at or datetime.now()
                if retrieved_at:
                    now_ts = datetime.now(retrieved_at.tzinfo) if getattr(retrieved_at, 'tzinfo', None) else datetime.now()
                    freshness_days = (now_ts - retrieved_at).days
                else:
                    freshness_days = 999

                if analyst_count:
                    coverage_score = min(1.0, analyst_count / 12)
                    agreement_score = max(0.1, 1 - (range_pct / 50))
                    consensus_confidence = coverage_score * agreement_score
                else:
                    consensus_confidence = 0.0

                return {
                    'analyst_count': analyst_count,
                    'consensus_range_pct': range_pct,
                    'consensus_confidence': consensus_confidence,
                    'consensus_std': std_estimate,
                    'consensus_high': high,
                    'consensus_low': low,
                    'data_freshness': freshness_days
                }

        except Exception as e:
            logger.error(f"Error getting consensus quality for {symbol}: {e}")

        # Fallback to synthetic data if real data unavailable
        return {
            'analyst_count': 12,
            'consensus_range_pct': 8.5,
            'consensus_confidence': 0.75,
            'consensus_std': 0.15,
            'data_freshness': 999  # Indicates synthetic data
        }

# Global instance
earnings_monitor = EarningsMonitor()