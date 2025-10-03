"""
Low-Latency Inference Service
FastAPI service for real-time model predictions using ONNX Runtime
Targets: p99 latency ≤ 50ms under 2x prod QPS
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import onnxruntime as ort
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

logger = logging.getLogger(__name__)


# Prometheus metrics
INFERENCE_REQUESTS = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['model_name', 'status']
)

INFERENCE_LATENCY = Histogram(
    'inference_latency_ms',
    'Inference latency in milliseconds',
    ['model_name'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

BATCH_SIZE = Histogram(
    'inference_batch_size',
    'Batch size for inference',
    ['model_name'],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Model load time in seconds',
    ['model_name']
)

ACTIVE_MODELS = Gauge(
    'inference_active_models',
    'Number of active models'
)


# Request/Response models
class PredictionRequest(BaseModel):
    """Single prediction request"""
    features: List[float] = Field(..., description="Input features")
    model_name: str = Field(default="default", description="Model to use")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    batch: List[List[float]] = Field(..., description="Batch of input features")
    model_name: str = Field(default="default", description="Model to use")
    max_batch_delay_ms: int = Field(default=10, description="Max delay for batching")


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: float
    model_name: str
    latency_ms: float
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[float]
    model_name: str
    batch_size: int
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: int
    uptime_seconds: float
    version: str


class ONNXModelWrapper:
    """
    Wrapper for ONNX model with optimizations
    """

    def __init__(
        self,
        model_path: Path,
        model_name: str,
        enable_profiling: bool = False
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.warmup_done = False

        # Load model
        self._load_model(enable_profiling)

    def _load_model(self, enable_profiling: bool = False):
        """Load ONNX model"""
        start_time = time.time()

        try:
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            if enable_profiling:
                sess_options.enable_profiling = True

            # Create inference session
            providers = ['CPUExecutionProvider']  # Can add CUDAExecutionProvider if GPU available

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options,
                providers=providers
            )

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            load_time = time.time() - start_time
            MODEL_LOAD_TIME.labels(model_name=self.model_name).set(load_time)

            logger.info(
                f"Loaded model '{self.model_name}' from {self.model_path} "
                f"in {load_time:.3f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def warmup(self, input_shape: tuple, num_iterations: int = 100):
        """
        Warmup model with dummy predictions

        Args:
            input_shape: Shape of input (e.g., (1, 50))
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up model '{self.model_name}'...")
        start_time = time.time()

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(num_iterations):
            self.predict(dummy_input)

        warmup_time = time.time() - start_time
        self.warmup_done = True

        logger.info(
            f"Model '{self.model_name}' warmed up in {warmup_time:.3f}s "
            f"({num_iterations} iterations)"
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference

        Args:
            features: Input features (2D array: batch_size x num_features)

        Returns:
            Predictions
        """
        if self.session is None:
            raise RuntimeError(f"Model {self.model_name} not loaded")

        # Ensure correct dtype
        features = features.astype(np.float32)

        # Run inference
        result = self.session.run(
            [self.output_name],
            {self.input_name: features}
        )

        return result[0]


class InferenceService:
    """
    Inference service managing multiple models
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, ONNXModelWrapper] = {}
        self.start_time = time.time()

    def load_model(
        self,
        model_name: str,
        model_filename: str,
        warmup: bool = True,
        warmup_shape: tuple = (1, 50)
    ):
        """Load a model"""
        model_path = self.models_dir / model_filename

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        model = ONNXModelWrapper(model_path, model_name)

        # Warmup if requested
        if warmup:
            model.warmup(warmup_shape)

        self.models[model_name] = model
        ACTIVE_MODELS.set(len(self.models))

        logger.info(f"Model '{model_name}' loaded and ready")

    def get_model(self, model_name: str) -> ONNXModelWrapper:
        """Get model by name"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        return self.models[model_name]

    async def predict(
        self,
        model_name: str,
        features: np.ndarray
    ) -> np.ndarray:
        """Run prediction"""
        model = self.get_model(model_name)

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model.predict, features)

        return result


# Create FastAPI app
app = FastAPI(
    title="Trading Platform Inference Service",
    description="Low-latency inference service using ONNX Runtime",
    version="1.0.0"
)

# Initialize inference service
inference_service = InferenceService(models_dir=Path("inference/onnx"))


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting inference service...")

    # Load default model (example)
    # In production, load from configuration
    models_dir = Path("inference/onnx")
    if models_dir.exists():
        for model_file in models_dir.glob("*.onnx"):
            model_name = model_file.stem
            try:
                inference_service.load_model(
                    model_name=model_name,
                    model_filename=model_file.name,
                    warmup=True
                )
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")

    logger.info(f"Inference service started with {len(inference_service.models)} models")


@app.post("/inference/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single prediction endpoint
    Target: p99 latency ≤ 50ms
    """
    start_time = time.time()

    try:
        # Convert to numpy array
        features = np.array([request.features], dtype=np.float32)

        # Run prediction
        result = await inference_service.predict(request.model_name, features)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Record metrics
        INFERENCE_REQUESTS.labels(
            model_name=request.model_name,
            status="success"
        ).inc()
        INFERENCE_LATENCY.labels(model_name=request.model_name).observe(latency_ms)

        return PredictionResponse(
            prediction=float(result[0][0]),
            model_name=request.model_name,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
            metadata=request.metadata
        )

    except Exception as e:
        INFERENCE_REQUESTS.labels(
            model_name=request.model_name,
            status="error"
        ).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint
    Higher throughput for batch processing
    """
    start_time = time.time()

    try:
        # Convert to numpy array
        features = np.array(request.batch, dtype=np.float32)
        batch_size = len(request.batch)

        # Run batch prediction
        result = await inference_service.predict(request.model_name, features)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Record metrics
        INFERENCE_REQUESTS.labels(
            model_name=request.model_name,
            status="success"
        ).inc()
        INFERENCE_LATENCY.labels(model_name=request.model_name).observe(latency_ms)
        BATCH_SIZE.labels(model_name=request.model_name).observe(batch_size)

        return BatchPredictionResponse(
            predictions=[float(r[0]) for r in result],
            model_name=request.model_name,
            batch_size=batch_size,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        INFERENCE_REQUESTS.labels(
            model_name=request.model_name,
            status="error"
        ).inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inference/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    uptime = time.time() - inference_service.start_time

    return HealthResponse(
        status="healthy",
        models_loaded=len(inference_service.models),
        uptime_seconds=uptime,
        version="1.0.0"
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
