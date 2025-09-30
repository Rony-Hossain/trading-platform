"""
Error Attribution Engine for Backtest and Paper Trading Analysis

This module provides comprehensive error bucket analysis to understand the sources
of performance deviation from theoretical signals, including:
1. Slippage impact analysis
2. Risk stop attribution
3. Missed trade analysis
4. Execution timing delays
5. Market impact costs
6. Technical failure attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import asyncio
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categories of execution errors"""
    SLIPPAGE = "slippage"
    RISK_STOP = "risk_stop"
    MISSED_TRADE = "missed_trade"
    TIMING_DELAY = "timing_delay"
    MARKET_IMPACT = "market_impact"
    TECHNICAL_FAILURE = "technical_failure"
    LIQUIDITY_CONSTRAINT = "liquidity_constraint"
    POSITION_SIZING = "position_sizing"
    REGIME_MISMATCH = "regime_mismatch"

class TradeOutcome(Enum):
    """Trade outcomes for analysis"""
    SUCCESS = "success"
    PARTIAL_FILL = "partial_fill"
    FAILED = "failed"
    STOPPED_OUT = "stopped_out"
    MISSED = "missed"

@dataclass
class ErrorBucket:
    """Individual error bucket for attribution"""
    category: ErrorCategory
    description: str
    impact_bps: float
    frequency: int
    total_impact_dollar: float
    avg_impact_per_occurrence: float
    examples: List[str] = field(default_factory=list)
    attribution_percentage: float = 0.0

@dataclass
class SignalExecution:
    """Signal execution record for error analysis"""
    signal_id: str
    symbol: str
    signal_timestamp: datetime
    signal_direction: str  # "long" or "short"
    signal_strength: float
    theoretical_entry_price: float
    theoretical_exit_price: Optional[float] = None
    
    # Actual execution data
    actual_entry_timestamp: Optional[datetime] = None
    actual_entry_price: Optional[float] = None
    actual_exit_timestamp: Optional[datetime] = None
    actual_exit_price: Optional[float] = None
    
    # Execution details
    outcome: Optional[TradeOutcome] = None
    position_size: Optional[float] = None
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_delay_seconds: float = 0.0
    
    # Risk management
    stop_loss_triggered: bool = False
    stop_loss_price: Optional[float] = None
    risk_stop_reason: Optional[str] = None
    
    # Performance attribution
    theoretical_pnl: float = 0.0
    actual_pnl: float = 0.0
    error_attribution: Dict[ErrorCategory, float] = field(default_factory=dict)

@dataclass
class ErrorAttributionReport:
    """Comprehensive error attribution analysis"""
    period_start: datetime
    period_end: datetime
    total_signals: int
    executed_signals: int
    missed_signals: int
    
    # Performance summary
    theoretical_total_pnl: float
    actual_total_pnl: float
    total_error_impact: float
    error_rate_bps: float
    
    # Error buckets
    error_buckets: List[ErrorBucket]
    
    # Detailed breakdowns
    slippage_analysis: Dict[str, Any]
    risk_stop_analysis: Dict[str, Any]
    missed_trade_analysis: Dict[str, Any]
    timing_analysis: Dict[str, Any]
    
    # Improvement recommendations
    recommendations: List[str]
    
    # Regime-specific analysis
    regime_error_breakdown: Dict[str, Dict[ErrorCategory, float]]

class ErrorAttributionEngine:
    """Engine for comprehensive error attribution analysis"""
    
    def __init__(self):
        self.execution_records: List[SignalExecution] = []
        self.error_thresholds = {
            ErrorCategory.SLIPPAGE: 5.0,  # bps
            ErrorCategory.MARKET_IMPACT: 10.0,  # bps
            ErrorCategory.TIMING_DELAY: 5.0,  # seconds
            ErrorCategory.RISK_STOP: 2.0,  # % of trades
            ErrorCategory.MISSED_TRADE: 1.0,  # % of signals
        }
    
    def add_execution_record(self, execution: SignalExecution):
        """Add execution record for analysis"""
        self.execution_records.append(execution)
        
    def analyze_slippage_impact(self, executions: List[SignalExecution]) -> Dict[str, Any]:
        """Analyze slippage impact across different dimensions"""
        
        slippage_data = []
        for exec_rec in executions:
            if exec_rec.actual_entry_price and exec_rec.theoretical_entry_price:
                slippage_data.append({
                    'symbol': exec_rec.symbol,
                    'signal_strength': exec_rec.signal_strength,
                    'slippage_bps': exec_rec.slippage_bps,
                    'position_size': exec_rec.position_size or 0,
                    'timing_delay': exec_rec.timing_delay_seconds,
                    'timestamp': exec_rec.signal_timestamp,
                    'pnl_impact': exec_rec.theoretical_pnl - exec_rec.actual_pnl
                })
        
        if not slippage_data:
            return {'total_impact_bps': 0, 'frequency': 0, 'analysis': 'No slippage data available'}
            
        df = pd.DataFrame(slippage_data)
        
        analysis = {
            'total_impact_bps': df['slippage_bps'].sum(),
            'avg_slippage_bps': df['slippage_bps'].mean(),
            'median_slippage_bps': df['slippage_bps'].median(),
            'max_slippage_bps': df['slippage_bps'].max(),
            'frequency': len(df),
            'by_signal_strength': df.groupby(pd.cut(df['signal_strength'], bins=5))['slippage_bps'].mean().to_dict(),
            'by_position_size': df.groupby(pd.cut(df['position_size'], bins=5))['slippage_bps'].mean().to_dict(),
            'correlation_with_timing': df[['slippage_bps', 'timing_delay']].corr().iloc[0, 1] if len(df) > 1 else 0,
            'total_dollar_impact': df['pnl_impact'].sum(),
            'worst_slippage_examples': df.nlargest(5, 'slippage_bps')[['symbol', 'slippage_bps', 'pnl_impact']].to_dict('records')
        }
        
        return analysis
    
    def analyze_risk_stops(self, executions: List[SignalExecution]) -> Dict[str, Any]:
        """Analyze risk stop attribution and effectiveness"""
        
        risk_stop_data = []
        total_trades = 0
        
        for exec_rec in executions:
            if exec_rec.outcome in [TradeOutcome.SUCCESS, TradeOutcome.STOPPED_OUT]:
                total_trades += 1
                if exec_rec.stop_loss_triggered:
                    risk_stop_data.append({
                        'symbol': exec_rec.symbol,
                        'signal_strength': exec_rec.signal_strength,
                        'stop_reason': exec_rec.risk_stop_reason,
                        'theoretical_pnl': exec_rec.theoretical_pnl,
                        'actual_pnl': exec_rec.actual_pnl,
                        'stop_loss_price': exec_rec.stop_loss_price,
                        'entry_price': exec_rec.actual_entry_price,
                        'timestamp': exec_rec.signal_timestamp,
                        'pnl_impact': exec_rec.theoretical_pnl - exec_rec.actual_pnl
                    })
        
        if not risk_stop_data:
            return {'stop_rate': 0, 'analysis': 'No risk stops triggered'}
            
        df = pd.DataFrame(risk_stop_data)
        
        analysis = {
            'stop_rate': len(df) / total_trades * 100 if total_trades > 0 else 0,
            'total_stops': len(df),
            'avg_pnl_impact': df['pnl_impact'].mean(),
            'total_dollar_impact': df['pnl_impact'].sum(),
            'by_stop_reason': df.groupby('stop_reason').agg({
                'pnl_impact': ['count', 'mean', 'sum']
            }).round(2).to_dict(),
            'by_signal_strength': df.groupby(pd.cut(df['signal_strength'], bins=5))['pnl_impact'].mean().to_dict(),
            'effectiveness_analysis': self._analyze_stop_effectiveness(df),
            'worst_stops': df.nsmallest(5, 'pnl_impact')[['symbol', 'stop_reason', 'pnl_impact']].to_dict('records')
        }
        
        return analysis
    
    def analyze_missed_trades(self, executions: List[SignalExecution]) -> Dict[str, Any]:
        """Analyze missed trades and their impact"""
        
        missed_trades = [exec_rec for exec_rec in executions if exec_rec.outcome == TradeOutcome.MISSED]
        total_signals = len(executions)
        
        if not missed_trades:
            return {'miss_rate': 0, 'analysis': 'No missed trades'}
        
        missed_data = []
        for exec_rec in missed_trades:
            missed_data.append({
                'symbol': exec_rec.symbol,
                'signal_strength': exec_rec.signal_strength,
                'theoretical_pnl': exec_rec.theoretical_pnl,
                'opportunity_cost': exec_rec.theoretical_pnl,  # Full theoretical PnL is lost
                'timestamp': exec_rec.signal_timestamp,
                'timing_delay': exec_rec.timing_delay_seconds
            })
        
        df = pd.DataFrame(missed_data)
        
        analysis = {
            'miss_rate': len(missed_trades) / total_signals * 100,
            'total_missed': len(missed_trades),
            'total_opportunity_cost': df['opportunity_cost'].sum(),
            'avg_opportunity_cost': df['opportunity_cost'].mean(),
            'by_signal_strength': df.groupby(pd.cut(df['signal_strength'], bins=5))['opportunity_cost'].sum().to_dict(),
            'timing_correlation': df[['opportunity_cost', 'timing_delay']].corr().iloc[0, 1] if len(df) > 1 else 0,
            'largest_missed_opportunities': df.nlargest(5, 'opportunity_cost')[['symbol', 'signal_strength', 'opportunity_cost']].to_dict('records'),
            'miss_patterns': self._analyze_miss_patterns(df)
        }
        
        return analysis
    
    def analyze_timing_delays(self, executions: List[SignalExecution]) -> Dict[str, Any]:
        """Analyze timing delays and their impact"""
        
        timing_data = []
        for exec_rec in executions:
            if exec_rec.actual_entry_timestamp and exec_rec.signal_timestamp:
                delay_seconds = (exec_rec.actual_entry_timestamp - exec_rec.signal_timestamp).total_seconds()
                timing_data.append({
                    'symbol': exec_rec.symbol,
                    'delay_seconds': delay_seconds,
                    'signal_strength': exec_rec.signal_strength,
                    'pnl_impact': exec_rec.theoretical_pnl - exec_rec.actual_pnl,
                    'slippage_bps': exec_rec.slippage_bps,
                    'timestamp': exec_rec.signal_timestamp
                })
        
        if not timing_data:
            return {'avg_delay': 0, 'analysis': 'No timing data available'}
            
        df = pd.DataFrame(timing_data)
        
        analysis = {
            'avg_delay_seconds': df['delay_seconds'].mean(),
            'median_delay_seconds': df['delay_seconds'].median(),
            'max_delay_seconds': df['delay_seconds'].max(),
            'total_delay_impact': df['pnl_impact'].sum(),
            'delay_slippage_correlation': df[['delay_seconds', 'slippage_bps']].corr().iloc[0, 1] if len(df) > 1 else 0,
            'by_delay_buckets': df.groupby(pd.cut(df['delay_seconds'], bins=[0, 1, 5, 15, 60, float('inf')]))['pnl_impact'].sum().to_dict(),
            'worst_delays': df.nlargest(5, 'delay_seconds')[['symbol', 'delay_seconds', 'pnl_impact']].to_dict('records')
        }
        
        return analysis
    
    def _analyze_stop_effectiveness(self, stop_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze whether stop losses were effective or harmful"""
        
        if stop_df.empty:
            return {}
        
        # Calculate if stops were "good" (prevented larger losses) or "bad" (cut winning trades)
        good_stops = stop_df[stop_df['pnl_impact'] > 0]  # Prevented losses
        bad_stops = stop_df[stop_df['pnl_impact'] <= 0]  # Cut winners
        
        return {
            'good_stops_count': len(good_stops),
            'bad_stops_count': len(bad_stops),
            'effectiveness_ratio': len(good_stops) / len(stop_df) if len(stop_df) > 0 else 0,
            'good_stops_value_saved': good_stops['pnl_impact'].sum() if len(good_stops) > 0 else 0,
            'bad_stops_value_lost': abs(bad_stops['pnl_impact'].sum()) if len(bad_stops) > 0 else 0,
            'net_stop_value': stop_df['pnl_impact'].sum()
        }
    
    def _analyze_miss_patterns(self, miss_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missed trades"""
        
        if miss_df.empty:
            return {}
        
        # Group by time of day
        miss_df['hour'] = miss_df['timestamp'].dt.hour
        hourly_misses = miss_df.groupby('hour').size().to_dict()
        
        # Group by day of week
        miss_df['day_of_week'] = miss_df['timestamp'].dt.day_name()
        daily_misses = miss_df.groupby('day_of_week').size().to_dict()
        
        return {
            'hourly_pattern': hourly_misses,
            'daily_pattern': daily_misses,
            'high_strength_misses': len(miss_df[miss_df['signal_strength'] > 0.8]),
            'peak_miss_hour': max(hourly_misses, key=hourly_misses.get) if hourly_misses else None
        }
    
    def generate_error_buckets(self, executions: List[SignalExecution]) -> List[ErrorBucket]:
        """Generate comprehensive error buckets"""
        
        # Analyze each error category
        slippage_analysis = self.analyze_slippage_impact(executions)
        risk_stop_analysis = self.analyze_risk_stops(executions)
        missed_trade_analysis = self.analyze_missed_trades(executions)
        timing_analysis = self.analyze_timing_delays(executions)
        
        error_buckets = []
        
        # Slippage bucket
        if slippage_analysis['frequency'] > 0:
            error_buckets.append(ErrorBucket(
                category=ErrorCategory.SLIPPAGE,
                description="Price slippage during order execution",
                impact_bps=slippage_analysis['total_impact_bps'],
                frequency=slippage_analysis['frequency'],
                total_impact_dollar=slippage_analysis.get('total_dollar_impact', 0),
                avg_impact_per_occurrence=slippage_analysis['avg_slippage_bps'],
                examples=[f"{ex['symbol']}: {ex['slippage_bps']:.1f} bps" for ex in slippage_analysis.get('worst_slippage_examples', [])]
            ))
        
        # Risk stop bucket
        if risk_stop_analysis.get('total_stops', 0) > 0:
            error_buckets.append(ErrorBucket(
                category=ErrorCategory.RISK_STOP,
                description="Impact from risk management stop losses",
                impact_bps=0,  # Calculated differently for stops
                frequency=risk_stop_analysis['total_stops'],
                total_impact_dollar=risk_stop_analysis['total_dollar_impact'],
                avg_impact_per_occurrence=risk_stop_analysis['avg_pnl_impact'],
                examples=[f"{ex['symbol']}: {ex['stop_reason']}" for ex in risk_stop_analysis.get('worst_stops', [])]
            ))
        
        # Missed trade bucket
        if missed_trade_analysis.get('total_missed', 0) > 0:
            error_buckets.append(ErrorBucket(
                category=ErrorCategory.MISSED_TRADE,
                description="Signals that failed to execute",
                impact_bps=0,  # Opportunity cost
                frequency=missed_trade_analysis['total_missed'],
                total_impact_dollar=missed_trade_analysis['total_opportunity_cost'],
                avg_impact_per_occurrence=missed_trade_analysis['avg_opportunity_cost'],
                examples=[f"{ex['symbol']}: {ex['opportunity_cost']:.0f}" for ex in missed_trade_analysis.get('largest_missed_opportunities', [])]
            ))
        
        # Timing delay bucket
        if timing_analysis.get('avg_delay_seconds', 0) > self.error_thresholds[ErrorCategory.TIMING_DELAY]:
            error_buckets.append(ErrorBucket(
                category=ErrorCategory.TIMING_DELAY,
                description="Execution delays from signal generation",
                impact_bps=0,
                frequency=len(executions),
                total_impact_dollar=timing_analysis.get('total_delay_impact', 0),
                avg_impact_per_occurrence=timing_analysis['avg_delay_seconds'],
                examples=[f"{ex['symbol']}: {ex['delay_seconds']:.1f}s delay" for ex in timing_analysis.get('worst_delays', [])]
            ))
        
        # Calculate attribution percentages
        total_impact = sum(abs(bucket.total_impact_dollar) for bucket in error_buckets)
        for bucket in error_buckets:
            if total_impact > 0:
                bucket.attribution_percentage = abs(bucket.total_impact_dollar) / total_impact * 100
        
        return error_buckets
    
    def generate_recommendations(self, error_buckets: List[ErrorBucket], 
                               analyses: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on error analysis"""
        
        recommendations = []
        
        # Slippage recommendations
        slippage_bucket = next((b for b in error_buckets if b.category == ErrorCategory.SLIPPAGE), None)
        if slippage_bucket and slippage_bucket.avg_impact_per_occurrence > 10:
            recommendations.append("Consider using limit orders or implementing TWAP strategies to reduce slippage")
            recommendations.append("Evaluate order size relative to average daily volume")
        
        # Risk stop recommendations
        risk_stop_bucket = next((b for b in error_buckets if b.category == ErrorCategory.RISK_STOP), None)
        if risk_stop_bucket:
            stop_analysis = analyses.get('risk_stop_analysis', {})
            effectiveness = stop_analysis.get('effectiveness_analysis', {})
            if effectiveness.get('effectiveness_ratio', 0) < 0.5:
                recommendations.append("Review stop loss levels - current stops may be too tight")
                recommendations.append("Consider regime-aware stop loss adjustments")
        
        # Missed trade recommendations
        missed_bucket = next((b for b in error_buckets if b.category == ErrorCategory.MISSED_TRADE), None)
        if missed_bucket and missed_bucket.frequency > 0:
            miss_analysis = analyses.get('missed_trade_analysis', {})
            if miss_analysis.get('miss_rate', 0) > 5:
                recommendations.append("Improve signal delivery and execution infrastructure")
                recommendations.append("Implement signal prioritization based on strength")
        
        # Timing recommendations
        timing_bucket = next((b for b in error_buckets if b.category == ErrorCategory.TIMING_DELAY), None)
        if timing_bucket and timing_bucket.avg_impact_per_occurrence > 10:
            recommendations.append("Optimize signal processing and order routing latency")
            recommendations.append("Consider pre-positioning for high-probability signals")
        
        return recommendations
    
    async def generate_comprehensive_report(self, 
                                          start_date: datetime, 
                                          end_date: datetime) -> ErrorAttributionReport:
        """Generate comprehensive error attribution report"""
        
        # Filter executions by date range
        filtered_executions = [
            exec_rec for exec_rec in self.execution_records
            if start_date <= exec_rec.signal_timestamp <= end_date
        ]
        
        if not filtered_executions:
            logger.warning("No execution records found for the specified date range")
            return ErrorAttributionReport(
                period_start=start_date,
                period_end=end_date,
                total_signals=0,
                executed_signals=0,
                missed_signals=0,
                theoretical_total_pnl=0,
                actual_total_pnl=0,
                total_error_impact=0,
                error_rate_bps=0,
                error_buckets=[],
                slippage_analysis={},
                risk_stop_analysis={},
                missed_trade_analysis={},
                timing_analysis={},
                recommendations=[],
                regime_error_breakdown={}
            )
        
        # Calculate summary metrics
        total_signals = len(filtered_executions)
        executed_signals = len([e for e in filtered_executions if e.outcome != TradeOutcome.MISSED])
        missed_signals = len([e for e in filtered_executions if e.outcome == TradeOutcome.MISSED])
        
        theoretical_total_pnl = sum(e.theoretical_pnl for e in filtered_executions)
        actual_total_pnl = sum(e.actual_pnl for e in filtered_executions)
        total_error_impact = theoretical_total_pnl - actual_total_pnl
        error_rate_bps = (total_error_impact / abs(theoretical_total_pnl) * 10000) if theoretical_total_pnl != 0 else 0
        
        # Generate detailed analyses
        slippage_analysis = self.analyze_slippage_impact(filtered_executions)
        risk_stop_analysis = self.analyze_risk_stops(filtered_executions)
        missed_trade_analysis = self.analyze_missed_trades(filtered_executions)
        timing_analysis = self.analyze_timing_delays(filtered_executions)
        
        analyses = {
            'slippage_analysis': slippage_analysis,
            'risk_stop_analysis': risk_stop_analysis,
            'missed_trade_analysis': missed_trade_analysis,
            'timing_analysis': timing_analysis
        }
        
        # Generate error buckets
        error_buckets = self.generate_error_buckets(filtered_executions)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(error_buckets, analyses)
        
        return ErrorAttributionReport(
            period_start=start_date,
            period_end=end_date,
            total_signals=total_signals,
            executed_signals=executed_signals,
            missed_signals=missed_signals,
            theoretical_total_pnl=theoretical_total_pnl,
            actual_total_pnl=actual_total_pnl,
            total_error_impact=total_error_impact,
            error_rate_bps=error_rate_bps,
            error_buckets=error_buckets,
            slippage_analysis=slippage_analysis,
            risk_stop_analysis=risk_stop_analysis,
            missed_trade_analysis=missed_trade_analysis,
            timing_analysis=timing_analysis,
            recommendations=recommendations,
            regime_error_breakdown={}  # TODO: Implement regime-specific analysis
        )