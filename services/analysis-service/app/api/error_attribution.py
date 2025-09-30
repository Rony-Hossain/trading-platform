"""
Error Attribution API

FastAPI endpoints for comprehensive error bucket analysis of backtest and paper trading results.
Provides detailed attribution of performance deviations from theoretical signals.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio

from ..services.error_attribution_engine import (
    ErrorAttributionEngine, ErrorAttributionReport, ErrorBucket, SignalExecution,
    ErrorCategory, TradeOutcome
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/error-attribution", tags=["Error Attribution"])

# Global error attribution engine instance
error_engine = ErrorAttributionEngine()

class ErrorAttributionRequest(BaseModel):
    """Request model for error attribution analysis"""
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: datetime = Field(..., description="End date for analysis")
    strategy_ids: Optional[List[str]] = Field(default=None, description="Filter by strategy IDs")
    symbols: Optional[List[str]] = Field(default=None, description="Filter by symbols")
    min_signal_strength: Optional[float] = Field(default=0.0, description="Minimum signal strength")
    include_regime_analysis: bool = Field(default=True, description="Include regime-specific breakdown")

class SignalExecutionInput(BaseModel):
    """Input model for adding signal execution records"""
    signal_id: str
    symbol: str
    signal_timestamp: datetime
    signal_direction: str
    signal_strength: float
    theoretical_entry_price: float
    theoretical_exit_price: Optional[float] = None
    actual_entry_timestamp: Optional[datetime] = None
    actual_entry_price: Optional[float] = None
    actual_exit_timestamp: Optional[datetime] = None
    actual_exit_price: Optional[float] = None
    outcome: Optional[str] = None  # TradeOutcome enum value
    position_size: Optional[float] = None
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_delay_seconds: float = 0.0
    stop_loss_triggered: bool = False
    stop_loss_price: Optional[float] = None
    risk_stop_reason: Optional[str] = None
    theoretical_pnl: float = 0.0
    actual_pnl: float = 0.0

class ErrorBucketResponse(BaseModel):
    """Response model for error buckets"""
    category: str
    description: str
    impact_bps: float
    frequency: int
    total_impact_dollar: float
    avg_impact_per_occurrence: float
    attribution_percentage: float
    examples: List[str]

class ErrorAttributionResponse(BaseModel):
    """Response model for error attribution analysis"""
    period_start: datetime
    period_end: datetime
    total_signals: int
    executed_signals: int
    missed_signals: int
    theoretical_total_pnl: float
    actual_total_pnl: float
    total_error_impact: float
    error_rate_bps: float
    error_buckets: List[ErrorBucketResponse]
    slippage_analysis: Dict[str, Any]
    risk_stop_analysis: Dict[str, Any]
    missed_trade_analysis: Dict[str, Any]
    timing_analysis: Dict[str, Any]
    recommendations: List[str]
    regime_error_breakdown: Dict[str, Dict[str, float]]

@router.post("/add-execution-record")
async def add_execution_record(execution_input: SignalExecutionInput):
    """
    Add a signal execution record for error attribution analysis.
    
    This endpoint allows you to add individual execution records that will be used
    for comprehensive error attribution analysis. Each record represents a signal
    and its corresponding execution details.
    """
    try:
        # Convert input to SignalExecution object
        execution = SignalExecution(
            signal_id=execution_input.signal_id,
            symbol=execution_input.symbol,
            signal_timestamp=execution_input.signal_timestamp,
            signal_direction=execution_input.signal_direction,
            signal_strength=execution_input.signal_strength,
            theoretical_entry_price=execution_input.theoretical_entry_price,
            theoretical_exit_price=execution_input.theoretical_exit_price,
            actual_entry_timestamp=execution_input.actual_entry_timestamp,
            actual_entry_price=execution_input.actual_entry_price,
            actual_exit_timestamp=execution_input.actual_exit_timestamp,
            actual_exit_price=execution_input.actual_exit_price,
            outcome=TradeOutcome(execution_input.outcome) if execution_input.outcome else None,
            position_size=execution_input.position_size,
            slippage_bps=execution_input.slippage_bps,
            market_impact_bps=execution_input.market_impact_bps,
            timing_delay_seconds=execution_input.timing_delay_seconds,
            stop_loss_triggered=execution_input.stop_loss_triggered,
            stop_loss_price=execution_input.stop_loss_price,
            risk_stop_reason=execution_input.risk_stop_reason,
            theoretical_pnl=execution_input.theoretical_pnl,
            actual_pnl=execution_input.actual_pnl
        )
        
        error_engine.add_execution_record(execution)
        
        return {
            "status": "success",
            "message": f"Execution record added for signal {execution_input.signal_id}",
            "total_records": len(error_engine.execution_records)
        }
        
    except Exception as e:
        logger.error(f"Error adding execution record: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-execution-records")
async def add_multiple_execution_records(execution_inputs: List[SignalExecutionInput]):
    """
    Add multiple signal execution records in batch for error attribution analysis.
    
    This endpoint allows you to add multiple execution records at once for efficient
    bulk processing of historical trading data.
    """
    try:
        added_count = 0
        errors = []
        
        for execution_input in execution_inputs:
            try:
                execution = SignalExecution(
                    signal_id=execution_input.signal_id,
                    symbol=execution_input.symbol,
                    signal_timestamp=execution_input.signal_timestamp,
                    signal_direction=execution_input.signal_direction,
                    signal_strength=execution_input.signal_strength,
                    theoretical_entry_price=execution_input.theoretical_entry_price,
                    theoretical_exit_price=execution_input.theoretical_exit_price,
                    actual_entry_timestamp=execution_input.actual_entry_timestamp,
                    actual_entry_price=execution_input.actual_entry_price,
                    actual_exit_timestamp=execution_input.actual_exit_timestamp,
                    actual_exit_price=execution_input.actual_exit_price,
                    outcome=TradeOutcome(execution_input.outcome) if execution_input.outcome else None,
                    position_size=execution_input.position_size,
                    slippage_bps=execution_input.slippage_bps,
                    market_impact_bps=execution_input.market_impact_bps,
                    timing_delay_seconds=execution_input.timing_delay_seconds,
                    stop_loss_triggered=execution_input.stop_loss_triggered,
                    stop_loss_price=execution_input.stop_loss_price,
                    risk_stop_reason=execution_input.risk_stop_reason,
                    theoretical_pnl=execution_input.theoretical_pnl,
                    actual_pnl=execution_input.actual_pnl
                )
                
                error_engine.add_execution_record(execution)
                added_count += 1
                
            except Exception as e:
                errors.append(f"Signal {execution_input.signal_id}: {str(e)}")
        
        return {
            "status": "completed",
            "added_count": added_count,
            "total_records": len(error_engine.execution_records),
            "errors": errors if errors else None
        }
        
    except Exception as e:
        logger.error(f"Error adding execution records: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=ErrorAttributionResponse)
async def analyze_error_attribution(request: ErrorAttributionRequest):
    """
    Generate comprehensive error attribution analysis for the specified period.
    
    This endpoint performs detailed error bucket analysis to understand the sources
    of performance deviation from theoretical signals, including:
    - Slippage impact analysis
    - Risk stop attribution
    - Missed trade analysis
    - Execution timing delays
    - Market impact costs
    - Actionable recommendations
    """
    try:
        logger.info(f"Generating error attribution analysis for period {request.start_date} to {request.end_date}")
        
        # Generate comprehensive report
        report = await error_engine.generate_comprehensive_report(
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Convert error buckets to response format
        error_buckets_response = [
            ErrorBucketResponse(
                category=bucket.category.value,
                description=bucket.description,
                impact_bps=bucket.impact_bps,
                frequency=bucket.frequency,
                total_impact_dollar=bucket.total_impact_dollar,
                avg_impact_per_occurrence=bucket.avg_impact_per_occurrence,
                attribution_percentage=bucket.attribution_percentage,
                examples=bucket.examples
            )
            for bucket in report.error_buckets
        ]
        
        return ErrorAttributionResponse(
            period_start=report.period_start,
            period_end=report.period_end,
            total_signals=report.total_signals,
            executed_signals=report.executed_signals,
            missed_signals=report.missed_signals,
            theoretical_total_pnl=report.theoretical_total_pnl,
            actual_total_pnl=report.actual_total_pnl,
            total_error_impact=report.total_error_impact,
            error_rate_bps=report.error_rate_bps,
            error_buckets=error_buckets_response,
            slippage_analysis=report.slippage_analysis,
            risk_stop_analysis=report.risk_stop_analysis,
            missed_trade_analysis=report.missed_trade_analysis,
            timing_analysis=report.timing_analysis,
            recommendations=report.recommendations,
            regime_error_breakdown=report.regime_error_breakdown
        )
        
    except Exception as e:
        logger.error(f"Error generating error attribution analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/slippage-analysis")
async def get_slippage_analysis(
    start_date: datetime = Query(..., description="Start date for analysis"),
    end_date: datetime = Query(..., description="End date for analysis"),
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """
    Get detailed slippage analysis for the specified period.
    
    This endpoint provides in-depth analysis of slippage costs including:
    - Total and average slippage impact
    - Slippage by signal strength and position size
    - Correlation with timing delays
    - Worst slippage examples
    """
    try:
        # Filter executions by date range and symbol
        filtered_executions = [
            exec_rec for exec_rec in error_engine.execution_records
            if start_date <= exec_rec.signal_timestamp <= end_date
            and (symbol is None or exec_rec.symbol == symbol.upper())
        ]
        
        if not filtered_executions:
            return {"message": "No execution records found for the specified criteria"}
        
        slippage_analysis = error_engine.analyze_slippage_impact(filtered_executions)
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "symbol_filter": symbol
            },
            "analysis": slippage_analysis,
            "sample_size": len(filtered_executions)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing slippage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-stop-analysis")
async def get_risk_stop_analysis(
    start_date: datetime = Query(..., description="Start date for analysis"),
    end_date: datetime = Query(..., description="End date for analysis"),
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """
    Get detailed risk stop analysis for the specified period.
    
    This endpoint analyzes risk stop effectiveness and attribution including:
    - Stop rate and total stops triggered
    - Stop effectiveness (good vs bad stops)
    - PnL impact by stop reason
    - Stop performance by signal strength
    """
    try:
        # Filter executions by date range and symbol
        filtered_executions = [
            exec_rec for exec_rec in error_engine.execution_records
            if start_date <= exec_rec.signal_timestamp <= end_date
            and (symbol is None or exec_rec.symbol == symbol.upper())
        ]
        
        if not filtered_executions:
            return {"message": "No execution records found for the specified criteria"}
        
        risk_stop_analysis = error_engine.analyze_risk_stops(filtered_executions)
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "symbol_filter": symbol
            },
            "analysis": risk_stop_analysis,
            "sample_size": len(filtered_executions)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing risk stops: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/missed-trades-analysis")
async def get_missed_trades_analysis(
    start_date: datetime = Query(..., description="Start date for analysis"),
    end_date: datetime = Query(..., description="End date for analysis"),
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """
    Get detailed missed trades analysis for the specified period.
    
    This endpoint analyzes missed trading opportunities including:
    - Miss rate and total missed signals
    - Opportunity cost analysis
    - Miss patterns by time and signal strength
    - Largest missed opportunities
    """
    try:
        # Filter executions by date range and symbol
        filtered_executions = [
            exec_rec for exec_rec in error_engine.execution_records
            if start_date <= exec_rec.signal_timestamp <= end_date
            and (symbol is None or exec_rec.symbol == symbol.upper())
        ]
        
        if not filtered_executions:
            return {"message": "No execution records found for the specified criteria"}
        
        missed_trades_analysis = error_engine.analyze_missed_trades(filtered_executions)
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "symbol_filter": symbol
            },
            "analysis": missed_trades_analysis,
            "sample_size": len(filtered_executions)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing missed trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/timing-analysis")
async def get_timing_analysis(
    start_date: datetime = Query(..., description="Start date for analysis"),
    end_date: datetime = Query(..., description="End date for analysis"),
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """
    Get detailed timing delay analysis for the specified period.
    
    This endpoint analyzes execution timing delays including:
    - Average, median, and maximum delays
    - Delay impact on PnL
    - Correlation between delays and slippage
    - Worst timing delays examples
    """
    try:
        # Filter executions by date range and symbol
        filtered_executions = [
            exec_rec for exec_rec in error_engine.execution_records
            if start_date <= exec_rec.signal_timestamp <= end_date
            and (symbol is None or exec_rec.symbol == symbol.upper())
        ]
        
        if not filtered_executions:
            return {"message": "No execution records found for the specified criteria"}
        
        timing_analysis = error_engine.analyze_timing_delays(filtered_executions)
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "symbol_filter": symbol
            },
            "analysis": timing_analysis,
            "sample_size": len(filtered_executions)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing timing delays: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary-stats")
async def get_error_attribution_summary():
    """
    Get summary statistics for all loaded execution records.
    
    This endpoint provides a high-level overview of the error attribution data
    including total records, date ranges, and basic performance metrics.
    """
    try:
        if not error_engine.execution_records:
            return {
                "message": "No execution records loaded",
                "total_records": 0
            }
        
        # Calculate summary statistics
        records = error_engine.execution_records
        total_records = len(records)
        
        # Date range
        start_date = min(rec.signal_timestamp for rec in records)
        end_date = max(rec.signal_timestamp for rec in records)
        
        # Outcome distribution
        outcome_counts = {}
        for rec in records:
            outcome = rec.outcome.value if rec.outcome else "unknown"
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        # Performance summary
        total_theoretical_pnl = sum(rec.theoretical_pnl for rec in records)
        total_actual_pnl = sum(rec.actual_pnl for rec in records)
        total_error_impact = total_theoretical_pnl - total_actual_pnl
        
        # Symbol distribution
        symbol_counts = {}
        for rec in records:
            symbol_counts[rec.symbol] = symbol_counts.get(rec.symbol, 0) + 1
        
        return {
            "summary": {
                "total_records": total_records,
                "date_range": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "days": (end_date - start_date).days
                },
                "outcome_distribution": outcome_counts,
                "performance": {
                    "theoretical_total_pnl": total_theoretical_pnl,
                    "actual_total_pnl": total_actual_pnl,
                    "total_error_impact": total_error_impact,
                    "error_rate_bps": (total_error_impact / abs(total_theoretical_pnl) * 10000) if total_theoretical_pnl != 0 else 0
                },
                "top_symbols": dict(sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting summary stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear-records")
async def clear_execution_records():
    """
    Clear all execution records from the error attribution engine.
    
    This endpoint removes all loaded execution records. Use with caution as this
    action cannot be undone.
    """
    try:
        records_count = len(error_engine.execution_records)
        error_engine.execution_records.clear()
        
        return {
            "status": "success",
            "message": f"Cleared {records_count} execution records",
            "remaining_records": len(error_engine.execution_records)
        }
        
    except Exception as e:
        logger.error(f"Error clearing records: {e}")
        raise HTTPException(status_code=500, detail=str(e))