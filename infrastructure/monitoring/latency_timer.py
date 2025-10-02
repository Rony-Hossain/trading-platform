"""
End-to-End Latency Monitoring for Trading Pipeline
High-precision timing across 8 critical pipeline stages with p95 tracking
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import json
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """8 critical trading pipeline stages for latency monitoring"""
    SIGNAL_GENERATION = "signal_generation"
    REGIME_FILTER = "regime_filter"  
    VAR_CALCULATION = "var_calculation"
    BORROW_CHECK = "borrow_check"
    VENUE_RULES = "venue_rules"
    POSITION_SIZING = "position_sizing"
    EXECUTION_PREP = "execution_prep"
    TRADE_EXECUTION = "trade_execution"

@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage"""
    stage: PipelineStage
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, error: Optional[str] = None) -> float:
        """Mark stage as complete and return duration in milliseconds"""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.error = error
        return self.duration_ms

@dataclass
class PipelineTrace:
    """Complete end-to-end pipeline execution trace"""
    trace_id: str
    symbol: str
    trade_type: str
    quantity: int
    started_at: datetime
    pipeline_start: float
    pipeline_end: Optional[float] = None
    total_duration_ms: Optional[float] = None
    stages: Dict[PipelineStage, StageMetrics] = field(default_factory=dict)
    success: Optional[bool] = None
    final_decision: Optional[str] = None
    rejection_reasons: List[str] = field(default_factory=list)
    
    def complete(self, success: bool, decision: str, reasons: List[str] = None):
        """Mark pipeline as complete"""
        self.pipeline_end = time.perf_counter()
        self.total_duration_ms = (self.pipeline_end - self.pipeline_start) * 1000
        self.success = success
        self.final_decision = decision
        self.rejection_reasons = reasons or []
    
    def get_stage_durations(self) -> Dict[str, float]:
        """Get duration for each completed stage"""
        return {
            stage.value: metrics.duration_ms 
            for stage, metrics in self.stages.items() 
            if metrics.duration_ms is not None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "trace_id": self.trace_id,
            "symbol": self.symbol,
            "trade_type": self.trade_type,
            "quantity": self.quantity,
            "started_at": self.started_at.isoformat(),
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "final_decision": self.final_decision,
            "rejection_reasons": self.rejection_reasons,
            "stage_durations": self.get_stage_durations(),
            "stages": {
                stage.value: {
                    "duration_ms": metrics.duration_ms,
                    "error": metrics.error,
                    "metadata": metrics.metadata
                }
                for stage, metrics in self.stages.items()
            }
        }

class LatencyHistogram:
    """High-performance latency histogram for percentile calculations"""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.samples = deque(maxlen=max_samples)
        self.lock = threading.Lock()
    
    def add_sample(self, duration_ms: float):
        """Add latency sample"""
        with self.lock:
            self.samples.append(duration_ms)
    
    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        with self.lock:
            if not self.samples:
                return {}
            
            sorted_samples = sorted(self.samples)
            n = len(sorted_samples)
            
            return {
                "p50": sorted_samples[int(n * 0.5)],
                "p95": sorted_samples[int(n * 0.95)],
                "p99": sorted_samples[int(n * 0.99)],
                "p999": sorted_samples[int(n * 0.999)],
                "mean": statistics.mean(sorted_samples),
                "min": min(sorted_samples),
                "max": max(sorted_samples),
                "count": n
            }

class LatencyTimer:
    """High-precision latency monitoring for trading pipeline"""
    
    def __init__(self, enable_detailed_logging: bool = True):
        self.enable_detailed_logging = enable_detailed_logging
        self.active_traces: Dict[str, PipelineTrace] = {}
        self.completed_traces: deque = deque(maxlen=1000)
        self.stage_histograms: Dict[PipelineStage, LatencyHistogram] = {
            stage: LatencyHistogram() for stage in PipelineStage
        }
        self.pipeline_histogram = LatencyHistogram()
        self.lock = threading.Lock()
        
        # Performance thresholds (milliseconds)
        self.stage_thresholds = {
            PipelineStage.SIGNAL_GENERATION: 100,
            PipelineStage.REGIME_FILTER: 50,
            PipelineStage.VAR_CALCULATION: 200,
            PipelineStage.BORROW_CHECK: 150,
            PipelineStage.VENUE_RULES: 25,
            PipelineStage.POSITION_SIZING: 50,
            PipelineStage.EXECUTION_PREP: 75,
            PipelineStage.TRADE_EXECUTION: 300
        }
        self.pipeline_threshold = 2000  # 2 seconds for p95
        
        # Statistics
        self.stats = {
            "total_traces": 0,
            "successful_traces": 0,
            "failed_traces": 0,
            "threshold_violations": defaultdict(int),
            "last_reset": datetime.now()
        }
    
    def start_pipeline(self, trace_id: str, symbol: str, trade_type: str, 
                      quantity: int) -> PipelineTrace:
        """Start timing a new pipeline execution"""
        trace = PipelineTrace(
            trace_id=trace_id,
            symbol=symbol,
            trade_type=trade_type,
            quantity=quantity,
            started_at=datetime.now(),
            pipeline_start=time.perf_counter()
        )
        
        with self.lock:
            self.active_traces[trace_id] = trace
            self.stats["total_traces"] += 1
        
        if self.enable_detailed_logging:
            logger.info(f"Started pipeline trace {trace_id} for {symbol} {trade_type} {quantity}")
        
        return trace
    
    def start_stage(self, trace_id: str, stage: PipelineStage, 
                   metadata: Dict[str, Any] = None) -> Optional[StageMetrics]:
        """Start timing a pipeline stage"""
        with self.lock:
            trace = self.active_traces.get(trace_id)
        
        if not trace:
            logger.warning(f"No active trace found for {trace_id}")
            return None
        
        stage_metrics = StageMetrics(
            stage=stage,
            start_time=time.perf_counter(),
            metadata=metadata or {}
        )
        
        trace.stages[stage] = stage_metrics
        
        if self.enable_detailed_logging:
            logger.debug(f"Started stage {stage.value} for trace {trace_id}")
        
        return stage_metrics
    
    def complete_stage(self, trace_id: str, stage: PipelineStage, 
                      error: Optional[str] = None) -> Optional[float]:
        """Complete timing a pipeline stage"""
        with self.lock:
            trace = self.active_traces.get(trace_id)
        
        if not trace or stage not in trace.stages:
            logger.warning(f"No active stage {stage.value} found for trace {trace_id}")
            return None
        
        stage_metrics = trace.stages[stage]
        duration_ms = stage_metrics.complete(error)
        
        # Add to histogram
        self.stage_histograms[stage].add_sample(duration_ms)
        
        # Check threshold violations
        threshold = self.stage_thresholds.get(stage, 1000)
        if duration_ms > threshold:
            self.stats["threshold_violations"][stage] += 1
            logger.warning(
                f"Stage {stage.value} exceeded threshold: {duration_ms:.2f}ms > {threshold}ms "
                f"(trace: {trace_id})"
            )
        
        if self.enable_detailed_logging:
            logger.debug(f"Completed stage {stage.value} in {duration_ms:.2f}ms for trace {trace_id}")
        
        return duration_ms
    
    def complete_pipeline(self, trace_id: str, success: bool, decision: str, 
                         reasons: List[str] = None) -> Optional[PipelineTrace]:
        """Complete pipeline execution timing"""
        with self.lock:
            trace = self.active_traces.pop(trace_id, None)
        
        if not trace:
            logger.warning(f"No active trace found for {trace_id}")
            return None
        
        trace.complete(success, decision, reasons)
        
        # Add to histograms
        self.pipeline_histogram.add_sample(trace.total_duration_ms)
        
        # Update statistics
        with self.lock:
            if success:
                self.stats["successful_traces"] += 1
            else:
                self.stats["failed_traces"] += 1
            
            self.completed_traces.append(trace)
        
        # Check pipeline threshold
        if trace.total_duration_ms > self.pipeline_threshold:
            logger.warning(
                f"Pipeline exceeded threshold: {trace.total_duration_ms:.2f}ms > {self.pipeline_threshold}ms "
                f"(trace: {trace_id})"
            )
        
        if self.enable_detailed_logging:
            logger.info(
                f"Completed pipeline trace {trace_id}: {trace.total_duration_ms:.2f}ms, "
                f"success={success}, decision='{decision}'"
            )
        
        return trace
    
    def get_trace(self, trace_id: str) -> Optional[PipelineTrace]:
        """Get trace by ID (active or completed)"""
        # Check active traces
        with self.lock:
            if trace_id in self.active_traces:
                return self.active_traces[trace_id]
        
        # Check completed traces
        for trace in self.completed_traces:
            if trace.trace_id == trace_id:
                return trace
        
        return None
    
    def get_recent_traces(self, limit: int = 100) -> List[PipelineTrace]:
        """Get recent completed traces"""
        with self.lock:
            return list(self.completed_traces)[-limit:]
    
    def get_pipeline_percentiles(self) -> Dict[str, float]:
        """Get pipeline latency percentiles"""
        return self.pipeline_histogram.get_percentiles()
    
    def get_stage_percentiles(self, stage: PipelineStage) -> Dict[str, float]:
        """Get stage latency percentiles"""
        return self.stage_histograms[stage].get_percentiles()
    
    def get_all_percentiles(self) -> Dict[str, Dict[str, float]]:
        """Get percentiles for all stages and pipeline"""
        result = {
            "pipeline": self.get_pipeline_percentiles()
        }
        
        for stage in PipelineStage:
            result[stage.value] = self.get_stage_percentiles(stage)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self.lock:
            stats = self.stats.copy()
        
        percentiles = self.get_all_percentiles()
        
        # Check if p95 meets threshold
        pipeline_p95 = percentiles["pipeline"].get("p95", 0)
        p95_meets_threshold = pipeline_p95 <= self.pipeline_threshold
        
        return {
            "statistics": stats,
            "percentiles": percentiles,
            "thresholds": {
                "pipeline_threshold_ms": self.pipeline_threshold,
                "stage_thresholds_ms": {stage.value: threshold for stage, threshold in self.stage_thresholds.items()},
                "p95_meets_threshold": p95_meets_threshold,
                "current_p95_ms": pipeline_p95
            },
            "active_traces": len(self.active_traces),
            "completed_traces": len(self.completed_traces)
        }
    
    def reset_statistics(self):
        """Reset all statistics and histograms"""
        with self.lock:
            self.stats = {
                "total_traces": 0,
                "successful_traces": 0,
                "failed_traces": 0,
                "threshold_violations": defaultdict(int),
                "last_reset": datetime.now()
            }
            
            # Clear histograms
            for histogram in self.stage_histograms.values():
                histogram.samples.clear()
            self.pipeline_histogram.samples.clear()
            
            # Clear traces
            self.completed_traces.clear()
        
        logger.info("Reset latency statistics and histograms")

# Global latency timer instance
latency_timer = LatencyTimer()

class PipelineTimer:
    """Context manager for pipeline timing"""
    
    def __init__(self, trace_id: str, symbol: str, trade_type: str, quantity: int):
        self.trace_id = trace_id
        self.symbol = symbol
        self.trade_type = trade_type
        self.quantity = quantity
        self.trace: Optional[PipelineTrace] = None
    
    def __enter__(self) -> 'PipelineTimer':
        self.trace = latency_timer.start_pipeline(
            self.trace_id, self.symbol, self.trade_type, self.quantity
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace:
            success = exc_type is None
            decision = "success" if success else "error"
            reasons = [str(exc_val)] if exc_val else []
            latency_timer.complete_pipeline(self.trace_id, success, decision, reasons)

class StageTimer:
    """Context manager for stage timing"""
    
    def __init__(self, trace_id: str, stage: PipelineStage, metadata: Dict[str, Any] = None):
        self.trace_id = trace_id
        self.stage = stage
        self.metadata = metadata
    
    def __enter__(self) -> 'StageTimer':
        latency_timer.start_stage(self.trace_id, self.stage, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        error = str(exc_val) if exc_val else None
        latency_timer.complete_stage(self.trace_id, self.stage, error)

# Convenience functions
def start_pipeline_timing(trace_id: str, symbol: str, trade_type: str, quantity: int) -> PipelineTimer:
    """Start pipeline timing with context manager"""
    return PipelineTimer(trace_id, symbol, trade_type, quantity)

def start_stage_timing(trace_id: str, stage: PipelineStage, metadata: Dict[str, Any] = None) -> StageTimer:
    """Start stage timing with context manager"""
    return StageTimer(trace_id, stage, metadata)

def get_latency_report() -> Dict[str, Any]:
    """Get comprehensive latency report"""
    return latency_timer.get_statistics()

def is_performance_acceptable() -> Tuple[bool, str]:
    """Check if current performance meets acceptance criteria"""
    stats = latency_timer.get_statistics()
    pipeline_p95 = stats["percentiles"]["pipeline"].get("p95", 0)
    threshold = stats["thresholds"]["pipeline_threshold_ms"]
    
    acceptable = pipeline_p95 <= threshold
    message = (
        f"Pipeline p95: {pipeline_p95:.2f}ms (threshold: {threshold}ms)"
        if pipeline_p95 > 0 else "No performance data available"
    )
    
    return acceptable, message