from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel, Field

from ..services.execution_modeling_engine import (
    ExecutionModelingEngine, OrderType, VenueType, OrderUrgency,
    ExecutionRequest, ExecutionResult, ExecutionOrder
)

router = APIRouter(prefix="/execution-modeling", tags=["execution-modeling"])

class ExecutionModelingRequest(BaseModel):
    symbol: str
    quantity: int
    order_type: OrderType
    venue_preference: Optional[VenueType] = None
    urgency: OrderUrgency = OrderUrgency.NORMAL
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    current_price: float
    bid_price: float
    ask_price: float
    volume: Optional[int] = None

class ExecutionSimulationRequest(BaseModel):
    orders: List[ExecutionModelingRequest]
    simulation_duration_seconds: int = 300
    market_conditions: Optional[Dict] = None

class VenueAnalysisRequest(BaseModel):
    symbol: str
    quantity: int
    order_type: OrderType
    current_price: float
    bid_price: float
    ask_price: float

# Global execution engine instance
execution_engine = ExecutionModelingEngine()

@router.post("/execute-order")
async def execute_order(request: ExecutionModelingRequest) -> Dict:
    """
    Execute a single order with realistic latency modeling
    """
    try:
        execution_request = ExecutionRequest(
            order_id=f"order_{datetime.utcnow().timestamp()}",
            symbol=request.symbol,
            quantity=request.quantity,
            order_type=request.order_type,
            venue_preference=request.venue_preference,
            urgency=request.urgency,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            time_in_force=request.time_in_force
        )
        
        market_data = {
            'current_price': request.current_price,
            'bid_price': request.bid_price,
            'ask_price': request.ask_price,
            'volume': request.volume or 1000000
        }
        
        result = await execution_engine.execute_order(execution_request, market_data)
        
        return {
            "success": True,
            "execution_result": {
                "order_id": result.order_id,
                "execution_status": result.execution_status.value,
                "fill_quantity": result.fill_quantity,
                "fill_price": result.fill_price,
                "total_cost": result.total_cost,
                "commission": result.commission,
                "venue": result.venue.value if result.venue else None,
                "execution_time_ms": result.execution_time_ms,
                "slippage": result.slippage,
                "market_impact": result.market_impact,
                "timestamp": result.timestamp.isoformat(),
                "latency_breakdown": result.latency_breakdown
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/simulate-execution")
async def simulate_execution(request: ExecutionSimulationRequest) -> Dict:
    """
    Simulate execution of multiple orders over time
    """
    try:
        results = []
        start_time = datetime.utcnow()
        
        for order_req in request.orders:
            execution_request = ExecutionRequest(
                order_id=f"sim_order_{datetime.utcnow().timestamp()}",
                symbol=order_req.symbol,
                quantity=order_req.quantity,
                order_type=order_req.order_type,
                venue_preference=order_req.venue_preference,
                urgency=order_req.urgency,
                limit_price=order_req.limit_price,
                stop_price=order_req.stop_price,
                time_in_force=order_req.time_in_force
            )
            
            market_data = {
                'current_price': order_req.current_price,
                'bid_price': order_req.bid_price,
                'ask_price': order_req.ask_price,
                'volume': order_req.volume or 1000000
            }
            
            result = await execution_engine.execute_order(execution_request, market_data)
            results.append(result)
            
            # Add realistic delay between orders
            await asyncio.sleep(0.1)
        
        # Calculate aggregate statistics
        total_orders = len(results)
        filled_orders = len([r for r in results if r.fill_quantity > 0])
        total_cost = sum(r.total_cost for r in results)
        total_commission = sum(r.commission for r in results)
        avg_execution_time = sum(r.execution_time_ms for r in results) / total_orders if total_orders > 0 else 0
        total_slippage = sum(abs(r.slippage) for r in results)
        
        return {
            "success": True,
            "simulation_summary": {
                "total_orders": total_orders,
                "filled_orders": filled_orders,
                "fill_rate": filled_orders / total_orders if total_orders > 0 else 0,
                "total_cost": total_cost,
                "total_commission": total_commission,
                "average_execution_time_ms": avg_execution_time,
                "total_slippage": total_slippage,
                "simulation_duration_seconds": (datetime.utcnow() - start_time).total_seconds()
            },
            "execution_results": [
                {
                    "order_id": r.order_id,
                    "execution_status": r.execution_status.value,
                    "fill_quantity": r.fill_quantity,
                    "fill_price": r.fill_price,
                    "total_cost": r.total_cost,
                    "commission": r.commission,
                    "venue": r.venue.value if r.venue else None,
                    "execution_time_ms": r.execution_time_ms,
                    "slippage": r.slippage,
                    "market_impact": r.market_impact,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze-venues")
async def analyze_venues(request: VenueAnalysisRequest) -> Dict:
    """
    Analyze execution costs across different venues
    """
    try:
        analysis_results = {}
        
        market_data = {
            'current_price': request.current_price,
            'bid_price': request.bid_price,
            'ask_price': request.ask_price,
            'volume': 1000000
        }
        
        for venue_type in VenueType:
            execution_request = ExecutionRequest(
                order_id=f"analysis_{venue_type.value}_{datetime.utcnow().timestamp()}",
                symbol=request.symbol,
                quantity=request.quantity,
                order_type=request.order_type,
                venue_preference=venue_type,
                urgency=OrderUrgency.NORMAL
            )
            
            result = await execution_engine.execute_order(execution_request, market_data)
            
            analysis_results[venue_type.value] = {
                "execution_status": result.execution_status.value,
                "fill_price": result.fill_price,
                "total_cost": result.total_cost,
                "commission": result.commission,
                "execution_time_ms": result.execution_time_ms,
                "slippage": result.slippage,
                "market_impact": result.market_impact,
                "expected_total_cost": result.total_cost + abs(result.slippage * request.quantity)
            }
        
        # Find best venue by total cost
        best_venue = min(analysis_results.items(), 
                        key=lambda x: x[1]["expected_total_cost"])[0]
        
        return {
            "success": True,
            "venue_analysis": analysis_results,
            "recommendation": {
                "best_venue": best_venue,
                "reason": "Lowest expected total cost including slippage"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/latency-profiles")
async def get_latency_profiles() -> Dict:
    """
    Get latency profiles for all venue types
    """
    try:
        profiles = {}
        
        for venue_type, profile in execution_engine.venue_profiles.items():
            profiles[venue_type.value] = {
                "base_latency_ms": profile.base_latency_ms,
                "latency_std_ms": profile.latency_std_ms,
                "processing_time_ms": profile.processing_time_ms,
                "market_data_latency_ms": profile.market_data_latency_ms,
                "network_jitter_ms": profile.network_jitter_ms,
                "commission_rate": profile.commission_rate,
                "min_commission": profile.min_commission
            }
        
        return {
            "success": True,
            "latency_profiles": profiles
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/execution-statistics")
async def get_execution_statistics() -> Dict:
    """
    Get execution statistics from the engine
    """
    try:
        stats = execution_engine.get_execution_statistics()
        
        return {
            "success": True,
            "statistics": {
                "total_orders": stats["total_orders"],
                "filled_orders": stats["filled_orders"],
                "canceled_orders": stats["canceled_orders"],
                "rejected_orders": stats["rejected_orders"],
                "average_execution_time_ms": stats["average_execution_time_ms"],
                "average_slippage": stats["average_slippage"],
                "total_volume": stats["total_volume"],
                "total_commission": stats["total_commission"],
                "venue_distribution": stats["venue_distribution"],
                "order_type_distribution": stats["order_type_distribution"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/reset-statistics")
async def reset_statistics() -> Dict:
    """
    Reset execution statistics
    """
    try:
        execution_engine.reset_statistics()
        
        return {
            "success": True,
            "message": "Execution statistics have been reset"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))