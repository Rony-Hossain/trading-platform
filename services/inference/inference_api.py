"""
Inference API Service
FastAPI service for low-latency predictions
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import logging
from onnx_inference_service import ONNXInferenceService, InferenceServiceManager

logger = logging.getLogger(__name__)

app = FastAPI(title="Inference API", version="1.0.0")

# Global inference service manager
inference_manager = InferenceServiceManager()

# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

class PredictionRequest(BaseModel):
    """Single prediction request"""
    symbol: str
    features: Dict[str, float]
    model_name: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    symbols: List[str]
    features_list: List[Dict[str, float]]
    model_name: Optional[str] = None

class PredictionResponse(BaseModel):
    """Prediction response"""
    symbol: str
    alpha: float
    confidence: float
    features_used: int
    latency_ms: float
    timestamp: datetime
    model_version: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_latency_ms: float
    throughput_per_sec: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models: List[str]
    total_inferences: int
    uptime_seconds: float

class MetricsResponse(BaseModel):
    """Metrics response"""
    model_name: str
    total_inferences: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    throughput_per_sec: float
    errors: int

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    all_metrics = inference_manager.get_all_metrics()

    total_inferences = sum(m.total_inferences for m in all_metrics.values())

    return HealthResponse(
        status="healthy",
        models=list(inference_manager.services.keys()),
        total_inferences=total_inferences,
        uptime_seconds=0.0  # TODO: Track uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single prediction endpoint

    Target latency: <5ms p95
    """
    try:
        result = inference_manager.predict(
            symbol=request.symbol,
            features=request.features,
            model_name=request.model_name
        )

        return PredictionResponse(
            symbol=result.symbol,
            alpha=result.alpha,
            confidence=result.confidence,
            features_used=result.features_used,
            latency_ms=result.latency_ms,
            timestamp=result.timestamp,
            model_version=result.model_version
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint

    Higher throughput than single predictions
    """
    import time
    start_time = time.time()

    try:
        if request.model_name:
            service = inference_manager.services[request.model_name]
        else:
            service = inference_manager.services[inference_manager.default_model]

        results = service.predict_batch(
            symbols=request.symbols,
            features_list=request.features_list,
            model_version=request.model_name or inference_manager.default_model
        )

        total_latency_ms = (time.time() - start_time) * 1000
        throughput = len(results) / (total_latency_ms / 1000)

        predictions = [
            PredictionResponse(
                symbol=r.symbol,
                alpha=r.alpha,
                confidence=r.confidence,
                features_used=r.features_used,
                latency_ms=r.latency_ms,
                timestamp=r.timestamp,
                model_version=r.model_version
            )
            for r in results
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            total_latency_ms=total_latency_ms,
            throughput_per_sec=throughput
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=List[MetricsResponse])
async def get_metrics():
    """Get metrics for all models"""
    all_metrics = inference_manager.get_all_metrics()

    return [
        MetricsResponse(
            model_name=name,
            total_inferences=m.total_inferences,
            latency_p50_ms=m.latency_p50_ms,
            latency_p95_ms=m.latency_p95_ms,
            latency_p99_ms=m.latency_p99_ms,
            latency_max_ms=m.latency_max_ms,
            throughput_per_sec=m.throughput_per_sec,
            errors=m.errors
        )
        for name, m in all_metrics.items()
    ]

@app.get("/metrics/{model_name}", response_model=MetricsResponse)
async def get_model_metrics(model_name: str):
    """Get metrics for specific model"""
    if model_name not in inference_manager.services:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    service = inference_manager.services[model_name]
    metrics = service.get_metrics()

    return MetricsResponse(
        model_name=model_name,
        total_inferences=metrics.total_inferences,
        latency_p50_ms=metrics.latency_p50_ms,
        latency_p95_ms=metrics.latency_p95_ms,
        latency_p99_ms=metrics.latency_p99_ms,
        latency_max_ms=metrics.latency_max_ms,
        throughput_per_sec=metrics.throughput_per_sec,
        errors=metrics.errors
    )

@app.post("/models/reload/{model_name}")
async def reload_model(model_name: str, new_model_path: str):
    """Reload a model (for live updates)"""
    if model_name not in inference_manager.services:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    try:
        service = inference_manager.services[model_name]
        service.reload_model(new_model_path)

        return {"status": "success", "model": model_name, "path": new_model_path}

    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# STARTUP/SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting Inference API...")

    # TODO: Load models from config
    # For now, this would be configured via environment variables or config file

    # Example:
    # inference_manager.register_model(
    #     name="champion",
    #     model_path="models/momentum_alpha_v1.onnx",
    #     feature_names=[...],
    #     is_default=True,
    #     routing_weight=1.0
    # )

    logger.info("Inference API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Inference API...")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False,  # Reduce overhead
        workers=4  # Multiple workers for throughput
    )
