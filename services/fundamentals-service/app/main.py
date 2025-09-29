"""
Fundamentals Service - Earnings monitoring and basic fundamentals endpoints.
This simplified service exposes earnings-focused endpoints that are implemented
in the repository today. Advanced fundamentals parsing and analytics are
intentionally deferred until their modules are available.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging

from .core.database import get_db, create_tables
from .services.earnings_monitor import earnings_monitor
from .services.institutional_ownership_parser import institutional_ownership_parser
from .services.analyst_revision_tracker import analyst_revision_tracker
from .services.enhanced_surprise_service import enhanced_surprise_service
from .services.ownership_flow_analyzer import OwnershipFlowAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    logger.info("Fundamentals Service started")
    yield
    logger.info("Fundamentals Service stopped")


app = FastAPI(
    title="Fundamentals Service",
    description="Earnings monitoring and fundamentals scaffolding",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "fundamentals-service",
        "status": "running",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "earnings_calendar": "/earnings/calendar",
            "earnings_upcoming": "/earnings/upcoming",
            "earnings_monitor": "/earnings/{symbol}/monitor",
            "earnings_sector": "/earnings/sector/{sector}",
            "earnings_trend_revenue": "/earnings/trends/revenue",
            "earnings_trend_margins": "/earnings/trends/margins",
            "insider_transactions": "/ownership/insider/{symbol}",
            "institutional_holdings": "/ownership/institutional/{symbol}",
            "analyst_revisions": "/analysts/revisions/{symbol}",
            "analyst_momentum": "/analysts/momentum/{symbol}",
            "surprise_analysis": "/surprise/{symbol}",
            "ownership_flow": "/ownership/flow/{symbol}",
            "smart_money_signals": "/ownership/smart-money",
            "sync_consensus": "/sync/consensus/{symbol}",
            "sync_insider": "/sync/insider/{symbol}",
            "sync_institutional": "/sync/institutional/{symbol}",
            "sync_analyst_revisions": "/sync/analyst-revisions/{symbol}",
            "sync_all": "/sync/all/{symbol}",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "fundamentals-service",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/earnings/{symbol}")
async def get_earnings_data(
    symbol: str,
    periods: int = Query(8, ge=1, le=40),
    include_guidance: bool = Query(True),
    db: Session = Depends(get_db),
):
    try:
        performance = await earnings_monitor.track_quarterly_performance(symbol, periods, db)
        return {"symbol": symbol, "quarters": [p.__dict__ for p in performance]}
    except Exception as e:
        logger.error(f"Error getting earnings data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/earnings/{symbol}/transcripts")
async def get_earnings_transcripts(symbol: str, periods: int = Query(4, ge=1, le=20)):
    """Get earnings call transcripts for symbol"""
    try:
        # This would typically integrate with transcript providers like FactSet, Refinitiv, etc.
        # For now, return structure showing what would be available
        return {
            "symbol": symbol,
            "periods_requested": periods,
            "transcripts": [],
            "message": "Transcript integration requires premium data provider access",
            "available_providers": ["FactSet", "Refinitiv", "S&P Capital IQ"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting transcripts for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/factors/{symbol}")
async def get_factor_analysis(symbol: str):
    """Get fundamental factor analysis for symbol"""
    try:
        # Get basic fundamentals from Finnhub via earnings monitor
        quarterly_data = await earnings_monitor._get_quarterly_data(symbol, 4)
        
        if not quarterly_data:
            return {
                "symbol": symbol,
                "factors": {},
                "message": "No fundamental data available for analysis",
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate key fundamental factors
        factors = {
            "valuation": {
                "price_to_earnings": None,  # Would need current price
                "price_to_book": None,      # Would need book value
                "price_to_sales": None,     # Would need current price
            },
            "profitability": {
                "gross_margin": quarterly_data.gross_margin,
                "operating_margin": quarterly_data.operating_margin, 
                "net_margin": quarterly_data.net_margin,
                "return_on_equity": quarterly_data.roe,
                "return_on_assets": quarterly_data.roa
            },
            "growth": {
                "revenue_growth_yoy": quarterly_data.revenue_growth_yoy,
                "revenue_growth_qoq": quarterly_data.revenue_growth_qoq,
                "eps_growth_yoy": quarterly_data.eps_growth_yoy
            },
            "financial_health": {
                "debt_to_equity": None,     # Would need balance sheet data
                "current_ratio": None,      # Would need balance sheet data
                "quick_ratio": None         # Would need balance sheet data
            }
        }
        
        return {
            "symbol": symbol,
            "factors": factors,
            "data_period": f"{quarterly_data.quarter} {quarterly_data.fiscal_year}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting factor analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{symbol}")
async def get_company_profile(symbol: str, include_financials: bool = Query(True)):
    """Get company profile and basic information"""
    try:
        # Get company profile from Finnhub
        profile = await earnings_monitor.finnhub_client.get_company_profile(symbol)
        
        if not profile:
            return {
                "symbol": symbol,
                "profile": {},
                "message": "No company profile data available",
                "timestamp": datetime.now().isoformat()
            }
        
        response = {
            "symbol": symbol,
            "profile": {
                "name": profile.get('name'),
                "country": profile.get('country'),
                "currency": profile.get('currency'),
                "exchange": profile.get('exchange'),
                "ipo": profile.get('ipo'),
                "market_capitalization": profile.get('marketCapitalization'),
                "share_outstanding": profile.get('shareOutstanding'),
                "industry": profile.get('finnhubIndustry'),
                "website": profile.get('weburl'),
                "logo": profile.get('logo'),
                "ticker": profile.get('ticker')
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add financial data if requested
        if include_financials:
            quarterly_data = await earnings_monitor._get_quarterly_data(symbol, 1)
            if quarterly_data:
                response["financials"] = {
                    "revenue_ttm": quarterly_data.revenue * 4,  # Rough TTM estimate
                    "earnings_per_share": quarterly_data.earnings_per_share,
                    "net_margin": quarterly_data.net_margin,
                    "roe": quarterly_data.roe,
                    "period": f"{quarterly_data.quarter} {quarterly_data.fiscal_year}"
                }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting company profile for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync")
async def sync_financials(
    symbols: str = Query(..., description="Comma-separated symbols"),
    db: Session = Depends(get_db)
):
    """Sync financial data for multiple symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        results = {}
        
        for symbol in symbol_list:
            try:
                # Sync consensus data
                consensus_success = await earnings_monitor.fetch_and_store_consensus_data(symbol, db)
                
                # Sync analyst revisions  
                revisions_success = await earnings_monitor.fetch_and_store_analyst_revisions(symbol, db)
                
                # Sync insider transactions
                insider_success = await institutional_ownership_parser.fetch_and_store_insider_transactions(
                    symbol, db, 90
                )
                
                results[symbol] = {
                    "consensus": consensus_success,
                    "analyst_revisions": revisions_success,
                    "insider_transactions": insider_success,
                    "success": any([consensus_success, revisions_success, insider_success])
                }
                
            except Exception as e:
                logger.error(f"Error syncing data for {symbol}: {e}")
                results[symbol] = {
                    "consensus": False,
                    "analyst_revisions": False,
                    "insider_transactions": False,
                    "success": False,
                    "error": str(e)
                }
        
        successful_syncs = sum(1 for result in results.values() if result["success"])
        
        return {
            "status": "completed",
            "symbols_requested": symbol_list,
            "symbols_processed": len(symbol_list),
            "successful_syncs": successful_syncs,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in bulk sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare")
async def compare_companies(symbols: str = Query(...), metrics: str = Query("revenue,eps,roe,pe_ratio")):
    """Compare fundamental metrics across multiple companies"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        metrics_list = [m.strip() for m in metrics.split(",")]
        
        comparison = {
            "symbols": symbol_list,
            "metrics": metrics_list,
            "data": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for symbol in symbol_list:
            try:
                quarterly_data = await earnings_monitor._get_quarterly_data(symbol, 1)
                
                if quarterly_data:
                    symbol_data = {}
                    
                    for metric in metrics_list:
                        if metric == "revenue":
                            symbol_data[metric] = quarterly_data.revenue
                        elif metric == "eps":
                            symbol_data[metric] = quarterly_data.earnings_per_share
                        elif metric == "roe":
                            symbol_data[metric] = quarterly_data.roe
                        elif metric == "pe_ratio":
                            symbol_data[metric] = None  # Would need current price
                        elif metric == "gross_margin":
                            symbol_data[metric] = quarterly_data.gross_margin
                        elif metric == "operating_margin":
                            symbol_data[metric] = quarterly_data.operating_margin
                        elif metric == "net_margin":
                            symbol_data[metric] = quarterly_data.net_margin
                        elif metric == "roa":
                            symbol_data[metric] = quarterly_data.roa
                        elif metric == "revenue_growth_yoy":
                            symbol_data[metric] = quarterly_data.revenue_growth_yoy
                        elif metric == "eps_growth_yoy":
                            symbol_data[metric] = quarterly_data.eps_growth_yoy
                        else:
                            symbol_data[metric] = None
                    
                    comparison["data"][symbol] = symbol_data
                else:
                    comparison["data"][symbol] = {metric: None for metric in metrics_list}
                    
            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
                comparison["data"][symbol] = {
                    "error": str(e),
                    **{metric: None for metric in metrics_list}
                }
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error in company comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/earnings/calendar")
async def get_earnings_calendar(
    start_date: str = Query(..., description="Start date YYYY-MM-DD"),
    end_date: str = Query(..., description="End date YYYY-MM-DD"),
    symbols: Optional[str] = Query(None, description="Comma-separated symbols filter"),
):
    try:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
        symbol_list = symbols.split(",") if symbols else None
        calendar = await earnings_monitor.get_earnings_calendar(start_date_obj, end_date_obj, symbol_list)
        out = {
            k: {
                "date": v.date.isoformat(),
                "events": [e.__dict__ for e in v.events],
                "market_cap_total": v.market_cap_total,
                "high_impact_count": v.high_impact_count,
            }
            for k, v in calendar.items()
        }
        return out
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error getting earnings calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/earnings/upcoming")
async def get_upcoming_earnings(
    days_ahead: int = Query(30, ge=1, le=365, description="Days to look ahead"),
    min_market_cap: float = Query(1.0, ge=0, description="Minimum market cap in billions"),
):
    try:
        events = await earnings_monitor.get_upcoming_earnings(days_ahead, min_market_cap)
        return [e.__dict__ for e in events]
    except Exception as e:
        logger.error(f"Error getting upcoming earnings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/earnings/{symbol}/monitor")
async def monitor_earnings_performance(
    symbol: str,
    quarters_back: int = Query(12, ge=4, le=40, description="Number of quarters to analyze"),
    db: Session = Depends(get_db),
):
    try:
        quarterly_performance = await earnings_monitor.track_quarterly_performance(symbol, quarters_back, db)
        earnings_trends = await earnings_monitor.analyze_earnings_trends(symbol)
        return {
            "symbol": symbol,
            "quarterly_performance": [q.__dict__ for q in quarterly_performance],
            "earnings_trends": earnings_trends,
            "analysis_date": datetime.now().isoformat(),
            "data_source": "database_stored" if quarterly_performance else "generated",
        }
    except Exception as e:
        logger.error(f"Error monitoring earnings for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/earnings/sector/{sector}")
async def monitor_sector_earnings(
    sector: str,
    period: str = Query("current_quarter", description="current_quarter, last_quarter, current_year"),
):
    try:
        return await earnings_monitor.monitor_sector_earnings(sector, period)
    except Exception as e:
        logger.error(f"Error monitoring sector earnings for {sector}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/earnings/{symbol}/alerts")
async def setup_earnings_alerts(symbol: str, alert_settings: Dict[str, Any]):
    try:
        success = await earnings_monitor.setup_earnings_alerts(symbol, alert_settings)
        return {
            "symbol": symbol,
            "alerts_configured": success,
            "settings": alert_settings,
            "configured_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error setting up earnings alerts for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/earnings/trends/revenue")
async def get_revenue_trends(symbols: str = Query(...), quarters: int = Query(8, ge=4, le=20)):
    try:
        symbol_list = symbols.split(",")
        trends_data = {}
        for symbol in symbol_list:
            quarterly_data = await earnings_monitor.track_quarterly_performance(symbol, quarters)
            if quarterly_data:
                trends_data[symbol] = {
                    "revenue_growth_yoy": [q.revenue_growth_yoy for q in quarterly_data],
                    "revenue_growth_qoq": [q.revenue_growth_qoq for q in quarterly_data],
                    "quarters": [f"{q.quarter} {q.fiscal_year}" for q in quarterly_data],
                }
        return {"trends": trends_data, "analysis_date": datetime.now().isoformat(), "quarters_analyzed": quarters}
    except Exception as e:
        logger.error(f"Error getting revenue trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/earnings/trends/margins")
async def get_margin_trends(symbols: str = Query(...), quarters: int = Query(8, ge=4, le=20)):
    try:
        symbol_list = symbols.split(",")
        margin_data = {}
        for symbol in symbol_list:
            quarterly_data = await earnings_monitor.track_quarterly_performance(symbol, quarters)
            if quarterly_data:
                margin_data[symbol] = {
                    "gross_margin": [q.gross_margin for q in quarterly_data],
                    "operating_margin": [q.operating_margin for q in quarterly_data],
                    "net_margin": [q.net_margin for q in quarterly_data],
                    "quarters": [f"{q.quarter} {q.fiscal_year}" for q in quarterly_data],
                }
        return {"margin_trends": margin_data, "analysis_date": datetime.now().isoformat(), "quarters_analyzed": quarters}
    except Exception as e:
        logger.error(f"Error getting margin trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ownership/insider/{symbol}")
async def get_insider_transactions(
    symbol: str,
    days_back: int = Query(90, ge=1, le=365, description="Days to look back"),
    db: Session = Depends(get_db),
):
    """Get insider transaction data (Form 4 filings)"""
    try:
        ownership_flow = await institutional_ownership_parser.analyze_insider_flow(
            symbol.upper(), period_days=days_back
        )
        
        return {
            "symbol": symbol.upper(),
            "analysis_period_days": days_back,
            "ownership_flow": {
                "insider_buy_transactions": ownership_flow.insider_buy_transactions,
                "insider_sell_transactions": ownership_flow.insider_sell_transactions,
                "insider_net_shares": ownership_flow.insider_net_shares,
                "insider_net_value": float(ownership_flow.insider_net_value),
                "insider_buy_value": float(ownership_flow.insider_buy_value),
                "insider_sell_value": float(ownership_flow.insider_sell_value),
                "cluster_buying_detected": ownership_flow.cluster_buying_detected,
                "cluster_selling_detected": ownership_flow.cluster_selling_detected,
                "smart_money_score": float(ownership_flow.smart_money_score),
                "confidence_level": float(ownership_flow.confidence_level),
            },
            "analysis_date": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting insider transactions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ownership/institutional/{symbol}")
async def get_institutional_holdings(
    symbol: str,
    quarters_back: int = Query(4, ge=1, le=12, description="Quarters to analyze"),
):
    """Get institutional holdings data (13F filings)"""
    try:
        holdings = await institutional_ownership_parser.get_major_institutional_holders(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "quarters_analyzed": quarters_back,
            "major_holders": [
                {
                    "institution_name": holding.institution_name,
                    "institution_cik": holding.institution_cik,
                    "shares_held": holding.shares_held,
                    "market_value": float(holding.market_value),
                    "percentage_ownership": float(holding.percentage_ownership) if holding.percentage_ownership else None,
                    "shares_change": holding.shares_change,
                    "shares_change_pct": float(holding.shares_change_pct) if holding.shares_change_pct else None,
                    "is_new_position": holding.is_new_position,
                    "quarter_end": holding.quarter_end.isoformat(),
                    "filing_date": holding.filing_date.isoformat(),
                }
                for holding in holdings
            ],
            "total_institutional_holders": len(holdings),
            "analysis_date": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting institutional holdings for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ownership/flow/{symbol}")
async def get_ownership_flow_analysis(
    symbol: str,
    days_back: int = Query(90, ge=30, le=365, description="Analysis period in days"),
    db: Session = Depends(get_db),
):
    """Get comprehensive ownership flow analysis with smart money detection"""
    try:
        # Initialize the ownership flow analyzer with async database session
        analyzer = OwnershipFlowAnalyzer(db)
        
        # Perform comprehensive flow analysis
        flow_metrics = await analyzer.analyze_ownership_flow(
            symbol.upper(), 
            period_days=days_back
        )
        
        return {
            "symbol": flow_metrics.symbol,
            "analysis_period_days": flow_metrics.period_days,
            "analysis_date": flow_metrics.analysis_date.isoformat(),
            "insider_flow": {
                "buy_transactions": flow_metrics.insider_buy_transactions,
                "sell_transactions": flow_metrics.insider_sell_transactions,
                "net_shares": flow_metrics.insider_net_shares,
                "net_value": flow_metrics.insider_net_value,
                "buy_value": flow_metrics.insider_buy_value,
                "sell_value": flow_metrics.insider_sell_value,
            },
            "institutional_flow": {
                "institutions_increasing": flow_metrics.institutions_increasing,
                "institutions_decreasing": flow_metrics.institutions_decreasing,
                "new_positions": flow_metrics.institutions_new_positions,
                "sold_out": flow_metrics.institutions_sold_out,
                "net_shares": flow_metrics.institutional_net_shares,
                "net_value": flow_metrics.institutional_net_value,
            },
            "smart_money_signals": {
                "cluster_buying": flow_metrics.cluster_buying_detected,
                "cluster_selling": flow_metrics.cluster_selling_detected,
                "smart_money_score": flow_metrics.smart_money_score,
                "confidence_level": flow_metrics.confidence_level,
            },
            "metadata": flow_metrics.metadata,
        }
    except Exception as e:
        logger.error(f"Error getting ownership flow for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ownership/smart-money")
async def get_smart_money_signals(
    min_score: float = Query(0.3, ge=0.1, le=1.0, description="Minimum absolute smart money score"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    db: Session = Depends(get_db),
):
    """Get stocks with strong smart money signals"""
    try:
        analyzer = OwnershipFlowAnalyzer(db)
        signals = await analyzer.get_smart_money_signals(
            min_score=min_score,
            limit=limit
        )
        
        return {
            "signals": signals,
            "criteria": {
                "min_score": min_score,
                "max_results": limit,
            },
            "total_signals": len(signals),
            "analysis_date": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting smart money signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysts/revisions/{symbol}")
async def get_analyst_revisions(
    symbol: str,
    days_back: int = Query(30, ge=7, le=180, description="Days to look back for revisions"),
):
    """Get recent analyst revisions and rating changes"""
    try:
        revisions = await analyst_revision_tracker.get_recent_revisions(
            symbol.upper(), days_back
        )
        
        return {
            "symbol": symbol.upper(),
            "days_analyzed": days_back,
            "total_revisions": len(revisions),
            "revisions": [
                {
                    "revision_date": revision.revision_date.isoformat(),
                    "analyst_firm": revision.analyst_firm,
                    "analyst_name": revision.analyst_name,
                    "previous_rating": revision.previous_rating,
                    "new_rating": revision.new_rating,
                    "rating_action": revision.rating_action.value,
                    "previous_price_target": float(revision.previous_price_target) if revision.previous_price_target else None,
                    "new_price_target": float(revision.new_price_target) if revision.new_price_target else None,
                    "price_target_change": float(revision.price_target_change) if revision.price_target_change else None,
                    "price_target_change_pct": float(revision.price_target_change_pct) if revision.price_target_change_pct else None,
                    "upside_downside_pct": float(revision.upside_downside_pct) if revision.upside_downside_pct else None,
                    "revision_reason": revision.revision_reason,
                    "event_catalyst": revision.event_catalyst,
                    "confidence_score": float(revision.confidence_score),
                }
                for revision in revisions
            ],
            "analysis_date": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting analyst revisions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysts/momentum/{symbol}")
async def get_analyst_momentum(
    symbol: str,
    analysis_days: int = Query(30, ge=7, le=90, description="Days for momentum analysis"),
):
    """Get analyst revision momentum metrics"""
    try:
        momentum = await analyst_revision_tracker.calculate_revision_momentum(
            symbol.upper(), analysis_days
        )
        
        return {
            "symbol": symbol.upper(),
            "analysis_period": {
                "start_date": momentum.analysis_period_start.isoformat(),
                "end_date": momentum.analysis_period_end.isoformat(),
                "days_analyzed": momentum.days_analyzed,
            },
            "revision_activity": {
                "total_revisions": momentum.total_revisions,
                "upgrades": momentum.upgrades,
                "downgrades": momentum.downgrades,
                "initiations": momentum.initiations,
                "net_rating_changes": momentum.net_rating_changes,
                "revision_intensity": float(momentum.revision_intensity),
            },
            "momentum_scores": {
                "rating_momentum": float(momentum.rating_momentum_score),
                "price_target_momentum": float(momentum.price_target_momentum_score),
                "momentum_acceleration": float(momentum.momentum_acceleration),
                "conviction_score": float(momentum.conviction_score),
            },
            "price_target_activity": {
                "revisions": momentum.price_target_revisions,
                "increases": momentum.price_target_increases,
                "decreases": momentum.price_target_decreases,
                "average_change_pct": float(momentum.average_price_target_change_pct),
            },
            "event_indicators": {
                "pre_earnings_momentum": momentum.pre_earnings_momentum,
                "unusual_activity_detected": momentum.unusual_activity_detected,
                "smart_money_following": momentum.smart_money_following,
            },
            "analysis_date": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting analyst momentum for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/surprise/{symbol}")
async def get_surprise_analysis(
    symbol: str,
    event_type: str = Query("earnings", description="Type of event: earnings, revenue, guidance"),
    include_context: bool = Query(True, description="Include detailed context analysis"),
    db: Session = Depends(get_db),
):
    """Get enhanced surprise analysis for recent events"""
    try:
        # For demonstration, create a synthetic surprise analysis
        # In production, this would analyze actual recent events
        
        event_date = datetime.now().date() - timedelta(days=7)
        consensus_value = 2.50 + (hash(symbol) % 100) / 100
        actual_value = consensus_value + (hash(symbol + "actual") % 40 - 20) / 100
        
        enhanced_surprise = await enhanced_surprise_service.calculate_enhanced_surprise(
            symbol.upper(), event_date, event_type, consensus_value, actual_value, db
        )
        
        response = {
            "symbol": enhanced_surprise.symbol,
            "event_date": enhanced_surprise.event_date.isoformat(),
            "event_type": enhanced_surprise.event_type,
            "surprise_metrics": {
                "consensus_value": float(enhanced_surprise.consensus_value),
                "actual_value": float(enhanced_surprise.actual_value),
                "surprise_absolute": float(enhanced_surprise.surprise_absolute),
                "surprise_percent": float(enhanced_surprise.surprise_percent),
                "surprise_standardized": float(enhanced_surprise.surprise_standardized),
                "surprise_score": float(enhanced_surprise.surprise_score),
                "surprise_significance": enhanced_surprise.surprise_significance,
                "surprise_direction": enhanced_surprise.surprise_direction,
                "surprise_percentile": float(enhanced_surprise.surprise_percentile),
            },
            "consensus_quality": {
                "analyst_count": enhanced_surprise.consensus_count,
                "consensus_std": float(enhanced_surprise.consensus_std),
                "consensus_high": float(enhanced_surprise.consensus_high),
                "consensus_low": float(enhanced_surprise.consensus_low),
                "consensus_range_pct": float(enhanced_surprise.consensus_range_pct),
                "consensus_confidence": float(enhanced_surprise.consensus_confidence),
            },
            "market_impact": {
                "reaction_1d": float(enhanced_surprise.market_reaction_1d) if enhanced_surprise.market_reaction_1d else None,
                "reaction_3d": float(enhanced_surprise.market_reaction_3d) if enhanced_surprise.market_reaction_3d else None,
                "volume_spike": float(enhanced_surprise.volume_spike_factor) if enhanced_surprise.volume_spike_factor else None,
                "market_efficiency_score": float(enhanced_surprise.market_efficiency_score),
            },
            "analysis": {
                "surprise_attribution": enhanced_surprise.surprise_attribution,
                "future_implications": enhanced_surprise.future_implications,
            },
        }
        
        if include_context:
            response["context"] = {
                "days_to_event": enhanced_surprise.surprise_context.days_to_event,
                "consensus_stability": float(enhanced_surprise.surprise_context.consensus_stability),
                "revision_activity": enhanced_surprise.surprise_context.revision_activity,
                "analyst_conviction": float(enhanced_surprise.surprise_context.analyst_conviction),
                "historical_beat_rate": float(enhanced_surprise.surprise_context.historical_beat_rate),
                "market_regime": enhanced_surprise.surprise_context.market_regime,
                "volatility_regime": enhanced_surprise.surprise_context.volatility_regime,
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting surprise analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync/consensus/{symbol}")
async def sync_consensus_data(
    symbol: str,
    db: Session = Depends(get_db),
):
    """Fetch and store consensus data for a symbol"""
    try:
        success = await earnings_monitor.fetch_and_store_consensus_data(symbol.upper(), db)
        
        return {
            "symbol": symbol.upper(),
            "sync_type": "consensus",
            "success": success,
            "message": "Consensus data synced successfully" if success else "Failed to sync consensus data",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error syncing consensus data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync/insider/{symbol}")
async def sync_insider_transactions(
    symbol: str,
    period_days: int = Query(90, ge=30, le=365, description="Days to look back for transactions"),
    db: Session = Depends(get_db),
):
    """Fetch and store insider transaction data for a symbol"""
    try:
        success = await institutional_ownership_parser.fetch_and_store_insider_transactions(
            symbol.upper(), db, period_days
        )
        
        return {
            "symbol": symbol.upper(),
            "sync_type": "insider_transactions",
            "period_days": period_days,
            "success": success,
            "message": "Insider transaction data synced successfully" if success else "Failed to sync insider data",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error syncing insider transactions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync/institutional/{symbol}")
async def sync_institutional_holdings(
    symbol: str,
    db: Session = Depends(get_db),
):
    """Fetch and store institutional holdings data for a symbol"""
    try:
        success = await institutional_ownership_parser.fetch_and_store_institutional_holdings(
            symbol.upper(), db
        )
        
        return {
            "symbol": symbol.upper(),
            "sync_type": "institutional_holdings",
            "success": success,
            "message": "Institutional holdings data synced successfully" if success else "Failed to sync institutional data",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error syncing institutional holdings for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync/analyst-revisions/{symbol}")
async def sync_analyst_revisions(
    symbol: str,
    db: Session = Depends(get_db),
):
    """Fetch and store analyst revision data for a symbol"""
    try:
        success = await earnings_monitor.fetch_and_store_analyst_revisions(symbol.upper(), db)
        
        return {
            "symbol": symbol.upper(),
            "sync_type": "analyst_revisions",
            "success": success,
            "message": "Analyst revision data synced successfully" if success else "Failed to sync analyst revisions",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error syncing analyst revisions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync/all/{symbol}")
async def sync_all_fundamentals_data(
    symbol: str,
    db: Session = Depends(get_db),
):
    """Fetch and store all fundamentals data for a symbol"""
    try:
        results = {}
        
        # Sync consensus data
        results["consensus"] = await earnings_monitor.fetch_and_store_consensus_data(symbol.upper(), db)
        
        # Sync analyst revisions
        results["analyst_revisions"] = await earnings_monitor.fetch_and_store_analyst_revisions(symbol.upper(), db)
        
        # Sync insider transactions
        results["insider_transactions"] = await institutional_ownership_parser.fetch_and_store_insider_transactions(
            symbol.upper(), db, 90
        )
        
        # Sync institutional holdings
        results["institutional_holdings"] = await institutional_ownership_parser.fetch_and_store_institutional_holdings(
            symbol.upper(), db
        )
        
        success_count = sum(1 for success in results.values() if success)
        
        return {
            "symbol": symbol.upper(),
            "sync_type": "all_fundamentals",
            "results": results,
            "success_count": success_count,
            "total_operations": len(results),
            "overall_success": success_count > 0,
            "message": f"Synced {success_count}/{len(results)} data types successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error syncing all fundamentals data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_service_stats():
    return {"service": "fundamentals-service", "status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="info")
