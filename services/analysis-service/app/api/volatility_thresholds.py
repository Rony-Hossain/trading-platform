"""
API endpoints for Volatility-Normalized Surprise Threshold Calibration

This module provides REST API endpoints for:
- Volatility-based threshold calibration
- Adaptive surprise threshold calculation
- Sector-specific threshold adjustment
- Market regime-aware threshold optimization
- Multi-asset threshold comparison and analysis
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

from ..services.volatility_threshold_calibration import (
    VolatilityThresholdCalibrator,
    EventType,
    SectorType,
    MarketRegime,
    create_volatility_threshold_calibrator
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/volatility-thresholds", tags=["Volatility Thresholds"])

# Pydantic models for API
class VolatilityThresholdRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    event_type: str = Field(..., description="Event type (earnings, guidance, etc.)")
    surprise_value: float = Field(..., description="Surprise value to evaluate")
    sector: Optional[str] = Field(None, description="Sector classification")
    target_sigma_level: float = Field(2.0, description="Target sigma level for normalization")

class PriceDataPoint(BaseModel):
    timestamp: datetime
    close: float
    volume: Optional[int] = None

class BulkThresholdRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols")
    event_type: str = Field(..., description="Event type")
    price_data: Optional[Dict[str, List[PriceDataPoint]]] = Field(None, description="Historical price data")
    sector_mappings: Optional[Dict[str, str]] = Field(None, description="Symbol to sector mappings")

class ThresholdCalibrationResponse(BaseModel):
    symbol: str
    event_type: str
    surprise_value: float
    surprise_magnitude: float
    base_threshold: float
    final_threshold: float
    exceeds_threshold: bool
    normalized_surprise: float
    signal_confidence: float
    volatility_metrics: Dict[str, Any]
    adjustments: Dict[str, float]
    confidence_interval: List[float]
    calibration_quality: float
    timestamp: datetime

class SectorAnalysisResponse(BaseModel):
    sector: str
    symbols_analyzed: int
    median_volatility: float
    volatility_range: List[float]
    avg_threshold_adjustment: float
    event_sensitivity_scores: Dict[str, float]
    regime_distribution: Dict[str, int]

# Dependency for calibrator instance
async def get_calibrator() -> VolatilityThresholdCalibrator:
    return await create_volatility_threshold_calibrator()

@router.post("/calculate-adaptive-threshold", response_model=ThresholdCalibrationResponse)
async def calculate_adaptive_threshold(
    request: VolatilityThresholdRequest,
    calibrator: VolatilityThresholdCalibrator = Depends(get_calibrator)
):
    """Calculate adaptive threshold using volatility normalization"""
    try:
        # Convert string enums
        try:
            event_type = EventType(request.event_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid event type: {request.event_type}. Valid types: {[e.value for e in EventType]}"
            )
        
        sector = None
        if request.sector:
            try:
                sector = SectorType(request.sector.lower())
            except ValueError:
                logger.warning(f"Invalid sector: {request.sector}, using None")
        
        # Calculate adaptive threshold
        result = await calibrator.get_adaptive_threshold(
            symbol=request.symbol,
            event_type=event_type,
            surprise_value=request.surprise_value,
            sector=sector
        )
        
        return ThresholdCalibrationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error calculating adaptive threshold: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate-with-price-data", response_model=ThresholdCalibrationResponse)
async def calculate_threshold_with_price_data(
    symbol: str,
    event_type: str,
    surprise_value: float,
    price_data: List[PriceDataPoint],
    sector: Optional[str] = None,
    target_sigma_level: float = 2.0,
    calibrator: VolatilityThresholdCalibrator = Depends(get_calibrator)
):
    """Calculate threshold with provided historical price data"""
    try:
        # Convert string enums
        try:
            event_type_enum = EventType(event_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event type: {event_type}. Valid types: {[e.value for e in EventType]}"
            )
        
        sector_enum = None
        if sector:
            try:
                sector_enum = SectorType(sector.lower())
            except ValueError:
                logger.warning(f"Invalid sector: {sector}, using None")
        
        # Convert price data to DataFrame
        if len(price_data) < 5:
            raise HTTPException(
                status_code=400,
                detail="Minimum 5 data points required for volatility calculation"
            )
        
        df = pd.DataFrame([{
            'timestamp': point.timestamp,
            'close': point.close,
            'volume': point.volume or 0
        } for point in price_data])
        
        # Calculate adaptive threshold
        result = await calibrator.get_adaptive_threshold(
            symbol=symbol,
            event_type=event_type_enum,
            surprise_value=surprise_value,
            price_data=df,
            sector=sector_enum
        )
        
        return ThresholdCalibrationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error calculating threshold with price data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk-threshold-analysis")
async def bulk_threshold_analysis(
    request: BulkThresholdRequest,
    calibrator: VolatilityThresholdCalibrator = Depends(get_calibrator)
):
    """Perform bulk threshold analysis for multiple symbols"""
    try:
        # Convert event type
        try:
            event_type = EventType(request.event_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event type: {request.event_type}"
            )
        
        results = {}
        
        for symbol in request.symbols:
            try:
                # Get sector if provided
                sector = None
                if request.sector_mappings and symbol in request.sector_mappings:
                    try:
                        sector = SectorType(request.sector_mappings[symbol].lower())
                    except ValueError:
                        pass
                
                # Get price data if provided
                price_data = None
                if request.price_data and symbol in request.price_data:
                    df_data = []
                    for point in request.price_data[symbol]:
                        df_data.append({
                            'timestamp': point.timestamp,
                            'close': point.close,
                            'volume': point.volume or 0
                        })
                    price_data = pd.DataFrame(df_data)
                
                # Calculate adaptive threshold (using 0.0 as placeholder surprise)
                result = await calibrator.get_adaptive_threshold(
                    symbol=symbol,
                    event_type=event_type,
                    surprise_value=0.0,  # Placeholder for threshold analysis
                    price_data=price_data,
                    sector=sector
                )
                
                results[symbol] = result
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        return {
            "event_type": request.event_type,
            "symbols_processed": len(request.symbols),
            "results": results,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in bulk threshold analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sector-analysis/{sector}")
async def sector_threshold_analysis(
    sector: str,
    event_type: str = Query(..., description="Event type to analyze"),
    symbols: Optional[List[str]] = Query(None, description="Specific symbols to analyze"),
    calibrator: VolatilityThresholdCalibrator = Depends(get_calibrator)
):
    """Analyze threshold characteristics for a specific sector"""
    try:
        # Convert enums
        try:
            sector_enum = SectorType(sector.lower())
            event_type_enum = EventType(event_type.lower())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Get sector profile
        if sector_enum not in calibrator.sector_profiles:
            raise HTTPException(
                status_code=404,
                detail=f"Sector profile not found for {sector}"
            )
        
        sector_profile = calibrator.sector_profiles[sector_enum]
        
        # If specific symbols provided, analyze them
        symbol_results = {}
        if symbols:
            for symbol in symbols:
                try:
                    result = await calibrator.get_adaptive_threshold(
                        symbol=symbol,
                        event_type=event_type_enum,
                        surprise_value=0.0,  # Placeholder
                        sector=sector_enum
                    )
                    symbol_results[symbol] = result
                except Exception as e:
                    symbol_results[symbol] = {"error": str(e)}
        
        # Calculate sector statistics
        volatilities = []
        threshold_adjustments = []
        regimes = {}
        
        for result in symbol_results.values():
            if "error" not in result:
                vol_metrics = result.get("volatility_metrics", {})
                volatilities.append(vol_metrics.get("realized_vol_30d", 0))
                
                adjustments = result.get("adjustments", {})
                total_adj = (
                    adjustments.get("volatility_adjustment", 0) +
                    adjustments.get("sector_adjustment", 0) +
                    adjustments.get("regime_adjustment", 0)
                )
                threshold_adjustments.append(total_adj)
                
                regime = vol_metrics.get("vol_regime", "normal_vol")
                regimes[regime] = regimes.get(regime, 0) + 1
        
        # Build response
        response = SectorAnalysisResponse(
            sector=sector,
            symbols_analyzed=len(symbol_results),
            median_volatility=float(np.median(volatilities)) if volatilities else sector_profile.median_vol,
            volatility_range=[
                float(np.percentile(volatilities, 25)) if volatilities else sector_profile.vol_range[0],
                float(np.percentile(volatilities, 75)) if volatilities else sector_profile.vol_range[1]
            ],
            avg_threshold_adjustment=float(np.mean(threshold_adjustments)) if threshold_adjustments else 0.0,
            event_sensitivity_scores=sector_profile.event_sensitivity,
            regime_distribution=regimes
        )
        
        return {
            "sector_analysis": response,
            "sector_profile": {
                "median_vol": sector_profile.median_vol,
                "vol_range": sector_profile.vol_range,
                "beta_to_market": sector_profile.beta_to_market,
                "volatility_clustering": sector_profile.volatility_clustering,
                "mean_reversion_speed": sector_profile.mean_reversion_speed
            },
            "symbol_details": symbol_results if symbols else {},
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in sector analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/event-sensitivity-matrix")
async def get_event_sensitivity_matrix(
    sectors: Optional[List[str]] = Query(None, description="Sectors to include"),
    calibrator: VolatilityThresholdCalibrator = Depends(get_calibrator)
):
    """Get event sensitivity matrix across sectors and event types"""
    try:
        # Determine sectors to analyze
        if sectors:
            sector_enums = []
            for sector in sectors:
                try:
                    sector_enums.append(SectorType(sector.lower()))
                except ValueError:
                    logger.warning(f"Invalid sector: {sector}")
        else:
            sector_enums = list(calibrator.sector_profiles.keys())
        
        # Build sensitivity matrix
        matrix = {}
        
        for sector_enum in sector_enums:
            if sector_enum in calibrator.sector_profiles:
                sector_profile = calibrator.sector_profiles[sector_enum]
                
                sector_data = {
                    "sector_name": sector_enum.value,
                    "base_volatility": sector_profile.median_vol,
                    "volatility_range": sector_profile.vol_range,
                    "beta_to_market": sector_profile.beta_to_market,
                    "event_sensitivities": {}
                }
                
                # Get sensitivity for each event type
                for event_type in EventType:
                    # Use sector-specific sensitivity if available
                    if event_type in sector_profile.event_sensitivity:
                        sensitivity = sector_profile.event_sensitivity[event_type]
                    else:
                        # Use global sensitivity
                        sensitivity = calibrator.event_sensitivities.get(event_type, 1.0)
                    
                    # Adjust by sector volatility characteristics
                    vol_adjustment = sector_profile.median_vol / 0.25  # Normalize by 25% reference vol
                    final_sensitivity = sensitivity * vol_adjustment
                    
                    sector_data["event_sensitivities"][event_type.value] = {
                        "base_sensitivity": sensitivity,
                        "volatility_adjusted": final_sensitivity,
                        "confidence": 0.8 if event_type in sector_profile.event_sensitivity else 0.5
                    }
                
                matrix[sector_enum.value] = sector_data
        
        # Calculate cross-sector statistics
        all_sensitivities = {}
        for event_type in EventType:
            event_sensitivities = []
            for sector_data in matrix.values():
                event_sens = sector_data["event_sensitivities"][event_type.value]
                event_sensitivities.append(event_sens["volatility_adjusted"])
            
            all_sensitivities[event_type.value] = {
                "mean": float(np.mean(event_sensitivities)),
                "std": float(np.std(event_sensitivities)),
                "min": float(np.min(event_sensitivities)),
                "max": float(np.max(event_sensitivities))
            }
        
        return {
            "sensitivity_matrix": matrix,
            "cross_sector_statistics": all_sensitivities,
            "methodology": {
                "volatility_normalization": "Thresholds normalized by asset 30-day realized volatility",
                "sector_adjustment": "Sector-specific event sensitivity multipliers applied",
                "confidence_scoring": "Based on sector profile completeness and data quality"
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error generating sensitivity matrix: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/threshold-simulation")
async def threshold_simulation(
    symbol: str,
    event_type: str,
    volatility_scenarios: List[float] = Query(..., description="Volatility scenarios to test"),
    surprise_values: List[float] = Query(..., description="Surprise values to test"),
    sector: Optional[str] = Query(None, description="Sector classification"),
    calibrator: VolatilityThresholdCalibrator = Depends(get_calibrator)
):
    """Simulate threshold behavior across different volatility and surprise scenarios"""
    try:
        # Convert enums
        try:
            event_type_enum = EventType(event_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        
        sector_enum = None
        if sector:
            try:
                sector_enum = SectorType(sector.lower())
            except ValueError:
                logger.warning(f"Invalid sector: {sector}")
        
        simulation_results = []
        
        for vol_scenario in volatility_scenarios:
            for surprise_value in surprise_values:
                # Create synthetic volatility metrics
                from ..services.volatility_threshold_calibration import VolatilityMetrics, MarketRegime
                
                vol_metrics = VolatilityMetrics(
                    symbol=symbol,
                    realized_vol_1d=vol_scenario,
                    realized_vol_5d=vol_scenario,
                    realized_vol_30d=vol_scenario,
                    implied_vol=vol_scenario * 1.15,
                    vol_of_vol=vol_scenario * 0.2,
                    vol_skew=1.0,
                    vol_regime=MarketRegime.NORMAL_VOLATILITY,
                    sector_vol_percentile=0.5
                )
                
                # Calculate threshold
                result = await calibrator.get_adaptive_threshold(
                    symbol=symbol,
                    event_type=event_type_enum,
                    surprise_value=surprise_value,
                    sector=sector_enum
                )
                
                # Override with synthetic volatility metrics
                result["volatility_scenario"] = vol_scenario
                result["original_vol_metrics"] = result["volatility_metrics"]
                result["volatility_metrics"] = {
                    "realized_vol_30d": vol_scenario,
                    "vol_regime": "normal_vol",
                    "sector_vol_percentile": 0.5
                }
                
                simulation_results.append(result)
        
        # Calculate summary statistics
        threshold_range = [r["final_threshold"] for r in simulation_results]
        confidence_range = [r["signal_confidence"] for r in simulation_results]
        
        return {
            "simulation_results": simulation_results,
            "summary_statistics": {
                "threshold_range": {
                    "min": float(np.min(threshold_range)),
                    "max": float(np.max(threshold_range)),
                    "mean": float(np.mean(threshold_range)),
                    "std": float(np.std(threshold_range))
                },
                "confidence_range": {
                    "min": float(np.min(confidence_range)),
                    "max": float(np.max(confidence_range)),
                    "mean": float(np.mean(confidence_range)),
                    "std": float(np.std(confidence_range))
                },
                "scenarios_tested": len(simulation_results)
            },
            "parameters": {
                "symbol": symbol,
                "event_type": event_type,
                "volatility_scenarios": volatility_scenarios,
                "surprise_values": surprise_values,
                "sector": sector
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in threshold simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-sectors")
async def get_supported_sectors():
    """Get list of supported sectors with their characteristics"""
    return {
        "sectors": [
            {
                "code": sector.value,
                "name": sector.value.replace("_", " ").title(),
                "description": f"Sector classification for {sector.value}"
            }
            for sector in SectorType
        ]
    }

@router.get("/supported-event-types")
async def get_supported_event_types():
    """Get list of supported event types with their base sensitivities"""
    calibrator = await create_volatility_threshold_calibrator()
    
    return {
        "event_types": [
            {
                "code": event.value,
                "name": event.value.replace("_", " ").title(),
                "base_sensitivity": calibrator.event_sensitivities.get(event, 1.0),
                "description": f"Event type for {event.value}"
            }
            for event in EventType
        ]
    }