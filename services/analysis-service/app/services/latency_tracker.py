"""
Signal-to-Execution Latency Tracker and Alpha Decay Monitor

Tracks the entire signal generation to trade execution pipeline to measure:
- Signal generation latency
- Service communication delays
- Order processing time
- Alpha decay from execution delays
- Execution bottleneck identification

Key Features:
- Microsecond precision timing
- Cross-service latency measurement
- Alpha decay quantification
- Performance bottleneck identification
- Real-time latency monitoring
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import statistics
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import warnings

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBALANCE = "rebalance"

class LatencyStage(Enum):
    """Stages in the signal-to-execution pipeline"""
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_VALIDATED = "signal_validated"
    SIGNAL_TRANSMITTED = "signal_transmitted"
    ORDER_RECEIVED = "order_received"
    ORDER_VALIDATED = "order_validated"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_EXECUTED = "order_executed"
    EXECUTION_CONFIRMED = "execution_confirmed"

@dataclass
class SignalEvent:
    """Individual signal generation event"""
    signal_id: str
    symbol: str
    signal_type: SignalType
    signal_strength: float  # 0-1, confidence in signal
    expected_return: float  # Expected return from signal
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LatencyMeasurement:
    """Single latency measurement between stages"""
    signal_id: str
    from_stage: LatencyStage
    to_stage: LatencyStage
    latency_microseconds: int
    timestamp: datetime
    component: str  # Which service/component caused the latency
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionEvent:
    """Trade execution event"""
    signal_id: str
    execution_id: str
    symbol: str
    side: str
    quantity: float
    execution_price: float
    execution_time: datetime
    fees: float = 0.0
    slippage_bps: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlphaDecayMeasurement:
    """Alpha decay analysis for a signal-execution pair"""
    signal_id: str
    symbol: str
    signal_type: SignalType
    
    # Timing data
    signal_time: datetime
    execution_time: datetime
    total_latency_ms: float
    
    # Price and return data
    signal_price: float
    execution_price: float
    current_price: float  # Price at analysis time
    
    # Alpha decay metrics
    potential_return_bps: float  # Return if executed immediately
    actual_return_bps: float     # Return after execution delay
    alpha_decay_bps: float       # Lost alpha due to latency
    decay_rate_per_ms: float     # Alpha decay rate per millisecond
    
    # Performance metrics
    signal_strength: float
    execution_quality: float  # 0-1, how well execution matched signal
    
    # Analysis metadata
    analysis_timestamp: datetime
    holding_period_hours: float = 24.0  # Default 24h holding period for analysis
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'signal_time': self.signal_time.isoformat(),
            'execution_time': self.execution_time.isoformat(),
            'total_latency_ms': self.total_latency_ms,
            'signal_price': self.signal_price,
            'execution_price': self.execution_price,
            'current_price': self.current_price,
            'potential_return_bps': self.potential_return_bps,
            'actual_return_bps': self.actual_return_bps,
            'alpha_decay_bps': self.alpha_decay_bps,
            'decay_rate_per_ms': self.decay_rate_per_ms,
            'signal_strength': self.signal_strength,
            'execution_quality': self.execution_quality,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'holding_period_hours': self.holding_period_hours
        }

class LatencyTracker:
    """Comprehensive latency tracking and alpha decay analysis"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # Signal tracking
        self.active_signals: Dict[str, SignalEvent] = {}
        self.signal_timestamps: Dict[str, Dict[LatencyStage, datetime]] = {}
        
        # Latency measurements
        self.latency_measurements: deque = deque(maxlen=max_history)
        self.execution_events: deque = deque(maxlen=max_history)
        
        # Alpha decay analysis
        self.alpha_decay_history: deque = deque(maxlen=max_history)
        
        # Performance tracking by component
        self.component_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Real-time statistics
        self.current_stats = {
            'total_signals_tracked': 0,
            'total_executions_tracked': 0,
            'avg_latency_ms': 0.0,
            'avg_alpha_decay_bps': 0.0,
            'last_updated': datetime.now(timezone.utc)
        }
    
    def generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        return f"sig_{uuid.uuid4().hex[:12]}_{int(time.time_ns() // 1000)}"
    
    async def log_signal_generation(self, signal: SignalEvent) -> str:
        """
        Log signal generation event.
        
        Args:
            signal: Signal event details
            
        Returns:
            Signal ID for tracking
        """
        
        try:
            signal_id = signal.signal_id or self.generate_signal_id()
            signal.signal_id = signal_id
            
            # Store signal
            self.active_signals[signal_id] = signal
            
            # Initialize timestamp tracking
            self.signal_timestamps[signal_id] = {
                LatencyStage.SIGNAL_GENERATED: datetime.now(timezone.utc)
            }
            
            # Update stats
            self.current_stats['total_signals_tracked'] += 1
            
            logger.debug(f"Signal generated: {signal_id} for {signal.symbol}")
            return signal_id
            
        except Exception as e:
            logger.error(f"Error logging signal generation: {e}")
            return signal.signal_id or "error"
    
    async def log_latency_checkpoint(self, signal_id: str, stage: LatencyStage, 
                                   component: str = "unknown", 
                                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log latency checkpoint at specific pipeline stage.
        
        Args:
            signal_id: Signal identifier
            stage: Pipeline stage being logged
            component: Component/service name causing latency
            metadata: Additional metadata
        """
        
        try:
            current_time = datetime.now(timezone.utc)
            
            if signal_id not in self.signal_timestamps:
                logger.warning(f"Signal {signal_id} not found for latency checkpoint")
                return
            
            # Record timestamp for this stage
            self.signal_timestamps[signal_id][stage] = current_time
            
            # Calculate latency from previous stage
            timestamps = self.signal_timestamps[signal_id]
            stage_order = list(LatencyStage)
            
            current_stage_idx = stage_order.index(stage)
            if current_stage_idx > 0:
                previous_stage = stage_order[current_stage_idx - 1]
                
                if previous_stage in timestamps:
                    previous_time = timestamps[previous_stage]
                    latency_microseconds = int((current_time - previous_time).total_seconds() * 1_000_000)
                    
                    # Create latency measurement
                    measurement = LatencyMeasurement(
                        signal_id=signal_id,
                        from_stage=previous_stage,
                        to_stage=stage,
                        latency_microseconds=latency_microseconds,
                        timestamp=current_time,
                        component=component,
                        metadata=metadata or {}
                    )
                    
                    # Store measurement
                    self.latency_measurements.append(measurement)
                    self.component_latencies[component].append(latency_microseconds)
                    
                    logger.debug(f"Latency: {signal_id} {previous_stage.value} -> {stage.value}: {latency_microseconds}μs")
            
        except Exception as e:
            logger.error(f"Error logging latency checkpoint: {e}")
    
    async def log_execution_event(self, execution: ExecutionEvent) -> None:
        """
        Log trade execution event.
        
        Args:
            execution: Execution event details
        """
        
        try:
            # Store execution
            self.execution_events.append(execution)
            
            # Mark execution checkpoint
            await self.log_latency_checkpoint(
                execution.signal_id, 
                LatencyStage.ORDER_EXECUTED,
                "execution_engine",
                {"execution_id": execution.execution_id, "execution_price": execution.execution_price}
            )
            
            # Update stats
            self.current_stats['total_executions_tracked'] += 1
            
            logger.debug(f"Execution logged: {execution.execution_id} for signal {execution.signal_id}")
            
        except Exception as e:
            logger.error(f"Error logging execution event: {e}")
    
    async def calculate_alpha_decay(self, signal_id: str, current_price: float,
                                   holding_period_hours: float = 24.0) -> Optional[AlphaDecayMeasurement]:
        """
        Calculate alpha decay for a completed signal-execution pair.
        
        Args:
            signal_id: Signal identifier
            current_price: Current market price for analysis
            holding_period_hours: Holding period for return calculation
            
        Returns:
            Alpha decay measurement or None if insufficient data
        """
        
        try:
            # Find signal and execution events
            signal = self.active_signals.get(signal_id)
            execution = None
            
            for exec_event in reversed(self.execution_events):
                if exec_event.signal_id == signal_id:
                    execution = exec_event
                    break
            
            if not signal or not execution:
                logger.warning(f"Incomplete data for alpha decay calculation: {signal_id}")
                return None
            
            # Get timing data
            timestamps = self.signal_timestamps.get(signal_id, {})
            signal_time = timestamps.get(LatencyStage.SIGNAL_GENERATED)
            execution_time = execution.execution_time
            
            if not signal_time or not execution_time:
                logger.warning(f"Missing timestamps for alpha decay calculation: {signal_id}")
                return None
            
            # Calculate total latency
            total_latency_ms = (execution_time - signal_time).total_seconds() * 1000
            
            # Get price data
            signal_price = signal.metadata.get('signal_price', execution.execution_price)
            execution_price = execution.execution_price
            
            # Calculate returns
            if signal.signal_type == SignalType.BUY:
                # For buy signals, positive return = price appreciation
                potential_return_bps = ((current_price - signal_price) / signal_price) * 10000
                actual_return_bps = ((current_price - execution_price) / execution_price) * 10000
            elif signal.signal_type == SignalType.SELL:
                # For sell signals, positive return = price depreciation
                potential_return_bps = ((signal_price - current_price) / signal_price) * 10000
                actual_return_bps = ((execution_price - current_price) / execution_price) * 10000
            else:
                # For other signal types, use absolute price difference
                potential_return_bps = abs((current_price - signal_price) / signal_price) * 10000
                actual_return_bps = abs((current_price - execution_price) / execution_price) * 10000
            
            # Calculate alpha decay
            alpha_decay_bps = potential_return_bps - actual_return_bps
            decay_rate_per_ms = alpha_decay_bps / total_latency_ms if total_latency_ms > 0 else 0
            
            # Calculate execution quality
            execution_quality = min(1.0, max(0.0, 1 - (abs(execution_price - signal_price) / signal_price)))
            
            # Create alpha decay measurement
            alpha_decay = AlphaDecayMeasurement(
                signal_id=signal_id,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                signal_time=signal_time,
                execution_time=execution_time,
                total_latency_ms=total_latency_ms,
                signal_price=signal_price,
                execution_price=execution_price,
                current_price=current_price,
                potential_return_bps=potential_return_bps,
                actual_return_bps=actual_return_bps,
                alpha_decay_bps=alpha_decay_bps,
                decay_rate_per_ms=decay_rate_per_ms,
                signal_strength=signal.signal_strength,
                execution_quality=execution_quality,
                analysis_timestamp=datetime.now(timezone.utc),
                holding_period_hours=holding_period_hours
            )
            
            # Store measurement
            self.alpha_decay_history.append(alpha_decay)
            
            # Update running statistics
            await self._update_statistics()
            
            logger.info(f"Alpha decay calculated: {signal_id} - {alpha_decay_bps:.2f}bps decay over {total_latency_ms:.1f}ms")
            return alpha_decay
            
        except Exception as e:
            logger.error(f"Error calculating alpha decay: {e}")
            return None
    
    async def _update_statistics(self) -> None:
        """Update running statistics"""
        
        try:
            # Calculate average latency
            if self.latency_measurements:
                recent_latencies = list(self.latency_measurements)[-1000:]  # Last 1000 measurements
                avg_latency_microseconds = statistics.mean(m.latency_microseconds for m in recent_latencies)
                self.current_stats['avg_latency_ms'] = avg_latency_microseconds / 1000
            
            # Calculate average alpha decay
            if self.alpha_decay_history:
                recent_decays = list(self.alpha_decay_history)[-1000:]  # Last 1000 measurements
                avg_alpha_decay = statistics.mean(m.alpha_decay_bps for m in recent_decays)
                self.current_stats['avg_alpha_decay_bps'] = avg_alpha_decay
            
            self.current_stats['last_updated'] = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def get_latency_analysis(self, lookback_minutes: int = 60) -> Dict[str, Any]:
        """
        Get comprehensive latency analysis.
        
        Args:
            lookback_minutes: Analysis window in minutes
            
        Returns:
            Latency analysis results
        """
        
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
            
            # Filter recent measurements
            recent_measurements = [
                m for m in self.latency_measurements 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_measurements:
                return {
                    'error': 'No recent latency measurements found',
                    'lookback_minutes': lookback_minutes
                }
            
            # Calculate statistics by stage
            stage_stats = {}
            for stage in LatencyStage:
                stage_latencies = [
                    m.latency_microseconds for m in recent_measurements 
                    if m.to_stage == stage
                ]
                
                if stage_latencies:
                    stage_stats[stage.value] = {
                        'count': len(stage_latencies),
                        'avg_microseconds': statistics.mean(stage_latencies),
                        'median_microseconds': statistics.median(stage_latencies),
                        'p95_microseconds': np.percentile(stage_latencies, 95),
                        'p99_microseconds': np.percentile(stage_latencies, 99),
                        'max_microseconds': max(stage_latencies),
                        'min_microseconds': min(stage_latencies)
                    }
            
            # Calculate statistics by component
            component_stats = {}
            for component, latencies in self.component_latencies.items():
                if latencies:
                    recent_component_latencies = [
                        lat for lat, measurement in zip(latencies, 
                        [m for m in self.latency_measurements if m.component == component])
                        if measurement.timestamp >= cutoff_time
                    ]
                    
                    if recent_component_latencies:
                        component_stats[component] = {
                            'count': len(recent_component_latencies),
                            'avg_microseconds': statistics.mean(recent_component_latencies),
                            'median_microseconds': statistics.median(recent_component_latencies),
                            'p95_microseconds': np.percentile(recent_component_latencies, 95),
                            'p99_microseconds': np.percentile(recent_component_latencies, 99)
                        }
            
            # Overall statistics
            all_latencies = [m.latency_microseconds for m in recent_measurements]
            overall_stats = {
                'total_measurements': len(recent_measurements),
                'avg_latency_microseconds': statistics.mean(all_latencies),
                'median_latency_microseconds': statistics.median(all_latencies),
                'p95_latency_microseconds': np.percentile(all_latencies, 95),
                'p99_latency_microseconds': np.percentile(all_latencies, 99),
                'max_latency_microseconds': max(all_latencies),
                'min_latency_microseconds': min(all_latencies)
            }
            
            return {
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'lookback_minutes': lookback_minutes,
                'overall_statistics': overall_stats,
                'stage_breakdown': stage_stats,
                'component_breakdown': component_stats,
                'current_stats': self.current_stats
            }
            
        except Exception as e:
            logger.error(f"Error generating latency analysis: {e}")
            return {
                'error': f'Latency analysis failed: {str(e)}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def get_alpha_decay_analysis(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive alpha decay analysis.
        
        Args:
            lookback_hours: Analysis window in hours
            
        Returns:
            Alpha decay analysis results
        """
        
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
            
            # Filter recent alpha decay measurements
            recent_decays = [
                d for d in self.alpha_decay_history 
                if d.analysis_timestamp >= cutoff_time
            ]
            
            if not recent_decays:
                return {
                    'error': 'No recent alpha decay measurements found',
                    'lookback_hours': lookback_hours
                }
            
            # Calculate overall statistics
            alpha_decays = [d.alpha_decay_bps for d in recent_decays]
            decay_rates = [d.decay_rate_per_ms for d in recent_decays]
            latencies = [d.total_latency_ms for d in recent_decays]
            
            overall_stats = {
                'total_measurements': len(recent_decays),
                'avg_alpha_decay_bps': statistics.mean(alpha_decays),
                'median_alpha_decay_bps': statistics.median(alpha_decays),
                'avg_decay_rate_per_ms': statistics.mean(decay_rates),
                'avg_latency_ms': statistics.mean(latencies),
                'total_alpha_lost_bps': sum(alpha_decays),
                'worst_decay_bps': max(alpha_decays) if alpha_decays else 0,
                'best_decay_bps': min(alpha_decays) if alpha_decays else 0
            }
            
            # Breakdown by symbol
            symbol_stats = {}
            symbols = set(d.symbol for d in recent_decays)
            
            for symbol in symbols:
                symbol_decays = [d for d in recent_decays if d.symbol == symbol]
                symbol_alpha_decays = [d.alpha_decay_bps for d in symbol_decays]
                
                symbol_stats[symbol] = {
                    'count': len(symbol_decays),
                    'avg_alpha_decay_bps': statistics.mean(symbol_alpha_decays),
                    'total_alpha_lost_bps': sum(symbol_alpha_decays),
                    'avg_latency_ms': statistics.mean([d.total_latency_ms for d in symbol_decays])
                }
            
            # Breakdown by signal type
            signal_type_stats = {}
            signal_types = set(d.signal_type for d in recent_decays)
            
            for signal_type in signal_types:
                type_decays = [d for d in recent_decays if d.signal_type == signal_type]
                type_alpha_decays = [d.alpha_decay_bps for d in type_decays]
                
                signal_type_stats[signal_type.value] = {
                    'count': len(type_decays),
                    'avg_alpha_decay_bps': statistics.mean(type_alpha_decays),
                    'total_alpha_lost_bps': sum(type_alpha_decays)
                }
            
            # Correlation analysis
            correlations = {}
            if len(recent_decays) > 10:  # Need sufficient data for correlation
                df = pd.DataFrame([d.to_dict() for d in recent_decays])
                
                correlations = {
                    'latency_vs_decay': float(np.corrcoef(df['total_latency_ms'], df['alpha_decay_bps'])[0, 1]),
                    'signal_strength_vs_decay': float(np.corrcoef(df['signal_strength'], df['alpha_decay_bps'])[0, 1]),
                    'execution_quality_vs_decay': float(np.corrcoef(df['execution_quality'], df['alpha_decay_bps'])[0, 1])
                }
            
            return {
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'lookback_hours': lookback_hours,
                'overall_statistics': overall_stats,
                'symbol_breakdown': symbol_stats,
                'signal_type_breakdown': signal_type_stats,
                'correlations': correlations,
                'recent_measurements': [d.to_dict() for d in recent_decays[-50:]]  # Last 50 measurements
            }
            
        except Exception as e:
            logger.error(f"Error generating alpha decay analysis: {e}")
            return {
                'error': f'Alpha decay analysis failed: {str(e)}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def identify_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify execution bottlenecks in the signal-to-execution pipeline.
        
        Returns:
            Bottleneck analysis with recommendations
        """
        
        try:
            if not self.latency_measurements:
                return {
                    'error': 'No latency measurements available for bottleneck analysis'
                }
            
            # Analyze recent measurements
            recent_measurements = list(self.latency_measurements)[-1000:]
            
            # Find slowest stages
            stage_latencies = {}
            for measurement in recent_measurements:
                stage = measurement.to_stage.value
                if stage not in stage_latencies:
                    stage_latencies[stage] = []
                stage_latencies[stage].append(measurement.latency_microseconds)
            
            # Calculate averages and identify bottlenecks
            stage_averages = {
                stage: statistics.mean(latencies) 
                for stage, latencies in stage_latencies.items()
            }
            
            # Find slowest components
            component_latencies = {}
            for measurement in recent_measurements:
                component = measurement.component
                if component not in component_latencies:
                    component_latencies[component] = []
                component_latencies[component].append(measurement.latency_microseconds)
            
            component_averages = {
                component: statistics.mean(latencies)
                for component, latencies in component_latencies.items()
            }
            
            # Identify top bottlenecks
            top_stage_bottlenecks = sorted(
                stage_averages.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            top_component_bottlenecks = sorted(
                component_averages.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Generate recommendations
            recommendations = []
            
            # Stage-based recommendations
            if top_stage_bottlenecks:
                slowest_stage, slowest_latency = top_stage_bottlenecks[0]
                if slowest_latency > 50000:  # > 50ms
                    recommendations.append(f"Critical bottleneck in {slowest_stage}: {slowest_latency/1000:.1f}ms average")
            
            # Component-based recommendations
            if top_component_bottlenecks:
                slowest_component, slowest_latency = top_component_bottlenecks[0]
                if slowest_latency > 20000:  # > 20ms
                    recommendations.append(f"Optimize {slowest_component} component: {slowest_latency/1000:.1f}ms average")
            
            # General recommendations
            total_avg_latency = statistics.mean([m.latency_microseconds for m in recent_measurements])
            if total_avg_latency > 100000:  # > 100ms total
                recommendations.append("Overall latency exceeds 100ms - consider architectural improvements")
            
            return {
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'measurements_analyzed': len(recent_measurements),
                'top_stage_bottlenecks': [
                    {'stage': stage, 'avg_latency_microseconds': int(latency)}
                    for stage, latency in top_stage_bottlenecks
                ],
                'top_component_bottlenecks': [
                    {'component': component, 'avg_latency_microseconds': int(latency)}
                    for component, latency in top_component_bottlenecks
                ],
                'recommendations': recommendations,
                'overall_avg_latency_microseconds': int(total_avg_latency)
            }
            
        except Exception as e:
            logger.error(f"Error identifying bottlenecks: {e}")
            return {
                'error': f'Bottleneck analysis failed: {str(e)}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }