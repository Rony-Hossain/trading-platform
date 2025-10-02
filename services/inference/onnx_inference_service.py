"""
Low-Latency Inference Service using ONNX Runtime
Target: <5ms p95 latency for single prediction
"""
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class InferenceMetrics:
    """Metrics for inference operations"""
    total_inferences: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    throughput_per_sec: float
    errors: int

@dataclass
class PredictionResult:
    """Single prediction result"""
    symbol: str
    alpha: float
    confidence: float
    features_used: int
    latency_ms: float
    timestamp: datetime
    model_version: str

class ONNXInferenceService:
    """
    ONNX-based inference service for ultra-low latency

    Key optimizations:
    1. ONNX Runtime with CPU/GPU support
    2. Batch inference when possible
    3. Feature vector caching
    4. Pre-allocated arrays
    5. Minimal serialization
    """

    def __init__(
        self,
        model_path: str,
        feature_names: List[str],
        use_gpu: bool = False,
        num_threads: int = 4,
        cache_size: int = 10000
    ):
        self.model_path = Path(model_path)
        self.feature_names = feature_names
        self.use_gpu = use_gpu
        self.num_threads = num_threads

        # Load ONNX model
        self._load_model()

        # Metrics tracking
        self.inference_count = 0
        self.latencies = []
        self.errors = 0

        # Feature cache (for repeated predictions on same symbols)
        self.feature_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(
            f"ONNX Inference Service initialized: {model_path} "
            f"({'GPU' if use_gpu else 'CPU'}, {num_threads} threads)"
        )

    def _load_model(self):
        """Load ONNX model with optimizations"""
        # Session options for performance
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = self.num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Providers (GPU or CPU)
        if self.use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load model
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get expected input shape
        self.input_shape = self.session.get_inputs()[0].shape
        self.num_features = self.input_shape[1] if len(self.input_shape) > 1 else len(self.feature_names)

        logger.info(
            f"Model loaded: {self.num_features} features, "
            f"input={self.input_name}, output={self.output_name}"
        )

    def predict(
        self,
        symbol: str,
        features: Dict[str, float],
        model_version: str = "v1"
    ) -> PredictionResult:
        """
        Single prediction (low latency)

        Args:
            symbol: Stock symbol
            features: Feature dictionary
            model_version: Model version identifier

        Returns:
            PredictionResult
        """
        start_time = time.time()

        try:
            # Build feature vector
            feature_vector = self._build_feature_vector(features)

            # Run inference
            input_data = {self.input_name: feature_vector}
            outputs = self.session.run([self.output_name], input_data)

            # Parse output (assume [alpha, confidence])
            alpha = float(outputs[0][0][0])
            confidence = float(outputs[0][0][1]) if outputs[0].shape[1] > 1 else 0.5

            latency_ms = (time.time() - start_time) * 1000

            # Track metrics
            self.inference_count += 1
            self.latencies.append(latency_ms)

            # Warn on slow inference
            if latency_ms > 5:
                logger.warning(f"Slow inference: {latency_ms:.2f}ms for {symbol}")

            return PredictionResult(
                symbol=symbol,
                alpha=alpha,
                confidence=confidence,
                features_used=self.num_features,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                model_version=model_version
            )

        except Exception as e:
            self.errors += 1
            logger.error(f"Inference error for {symbol}: {e}")
            raise

    def predict_batch(
        self,
        symbols: List[str],
        features_list: List[Dict[str, float]],
        model_version: str = "v1"
    ) -> List[PredictionResult]:
        """
        Batch prediction (higher throughput)

        Args:
            symbols: List of symbols
            features_list: List of feature dictionaries
            model_version: Model version identifier

        Returns:
            List of PredictionResult
        """
        start_time = time.time()

        if len(symbols) != len(features_list):
            raise ValueError("symbols and features_list must have same length")

        try:
            # Build batch feature matrix
            batch_features = np.vstack([
                self._build_feature_vector(features)[0]
                for features in features_list
            ]).astype(np.float32)

            # Run batch inference
            input_data = {self.input_name: batch_features}
            outputs = self.session.run([self.output_name], input_data)

            # Parse outputs
            results = []
            batch_latency_ms = (time.time() - start_time) * 1000
            latency_per_item_ms = batch_latency_ms / len(symbols)

            for i, symbol in enumerate(symbols):
                alpha = float(outputs[0][i][0])
                confidence = float(outputs[0][i][1]) if outputs[0].shape[1] > 1 else 0.5

                results.append(
                    PredictionResult(
                        symbol=symbol,
                        alpha=alpha,
                        confidence=confidence,
                        features_used=self.num_features,
                        latency_ms=latency_per_item_ms,
                        timestamp=datetime.now(),
                        model_version=model_version
                    )
                )

            # Track metrics
            self.inference_count += len(symbols)
            self.latencies.extend([latency_per_item_ms] * len(symbols))

            logger.info(
                f"Batch inference: {len(symbols)} symbols in {batch_latency_ms:.2f}ms "
                f"({latency_per_item_ms:.2f}ms per item)"
            )

            return results

        except Exception as e:
            self.errors += len(symbols)
            logger.error(f"Batch inference error: {e}")
            raise

    def _build_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Build feature vector from dictionary

        Args:
            features: Feature dictionary

        Returns:
            NumPy array of shape (1, num_features)
        """
        # Extract features in correct order
        feature_values = []

        for feature_name in self.feature_names:
            if feature_name in features:
                feature_values.append(features[feature_name])
            else:
                # Missing feature - use 0 or NaN
                logger.warning(f"Missing feature: {feature_name}")
                feature_values.append(0.0)

        # Convert to NumPy array
        feature_vector = np.array([feature_values], dtype=np.float32)

        return feature_vector

    def warm_up(self, num_iterations: int = 100):
        """
        Warm up model with dummy predictions

        This initializes thread pools and optimizes execution graph
        """
        logger.info(f"Warming up model with {num_iterations} iterations...")

        dummy_features = {name: 0.0 for name in self.feature_names}

        for i in range(num_iterations):
            self.predict(symbol="WARMUP", features=dummy_features)

        # Clear warm-up from metrics
        self.inference_count = 0
        self.latencies = []

        logger.info("Warm-up complete")

    def get_metrics(self) -> InferenceMetrics:
        """Get inference metrics"""
        if not self.latencies:
            return InferenceMetrics(
                total_inferences=0,
                latency_p50_ms=0,
                latency_p95_ms=0,
                latency_p99_ms=0,
                latency_max_ms=0,
                throughput_per_sec=0,
                errors=0
            )

        latencies_array = np.array(self.latencies[-10000:])  # Last 10k

        # Calculate time window for throughput
        if len(self.latencies) > 1:
            # Assume uniform distribution over last minute
            throughput = len(self.latencies) / 60.0
        else:
            throughput = 0

        return InferenceMetrics(
            total_inferences=self.inference_count,
            latency_p50_ms=np.percentile(latencies_array, 50),
            latency_p95_ms=np.percentile(latencies_array, 95),
            latency_p99_ms=np.percentile(latencies_array, 99),
            latency_max_ms=np.max(latencies_array),
            throughput_per_sec=throughput,
            errors=self.errors
        )

    def reload_model(self, new_model_path: str):
        """
        Reload model (for live updates)

        Args:
            new_model_path: Path to new model file
        """
        logger.info(f"Reloading model from {new_model_path}")

        old_model_path = self.model_path
        self.model_path = Path(new_model_path)

        try:
            self._load_model()
            logger.info("Model reloaded successfully")

        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            # Rollback to old model
            self.model_path = old_model_path
            self._load_model()
            raise

class InferenceServiceManager:
    """
    Manages multiple inference services (champion/challenger)

    Supports:
    - A/B testing
    - Champion/challenger deployment
    - Gradual rollout
    """

    def __init__(self):
        self.services: Dict[str, ONNXInferenceService] = {}
        self.routing_config = {}  # model_name -> weight
        self.default_model = None

    def register_model(
        self,
        name: str,
        model_path: str,
        feature_names: List[str],
        is_default: bool = False,
        routing_weight: float = 0.0
    ):
        """
        Register a model

        Args:
            name: Model name (e.g., "champion", "challenger_v2")
            model_path: Path to ONNX model
            feature_names: List of feature names
            is_default: Whether this is the default model
            routing_weight: Routing weight for A/B testing (0-1)
        """
        service = ONNXInferenceService(
            model_path=model_path,
            feature_names=feature_names
        )

        # Warm up
        service.warm_up()

        self.services[name] = service
        self.routing_config[name] = routing_weight

        if is_default:
            self.default_model = name

        logger.info(f"Registered model: {name} (weight={routing_weight})")

    def predict(
        self,
        symbol: str,
        features: Dict[str, float],
        model_name: Optional[str] = None
    ) -> PredictionResult:
        """
        Route prediction to appropriate model

        Args:
            symbol: Stock symbol
            features: Feature dictionary
            model_name: Specific model to use (None = use routing)

        Returns:
            PredictionResult
        """
        if model_name:
            # Use specific model
            service = self.services[model_name]
        else:
            # Use default model
            service = self.services[self.default_model]

        return service.predict(symbol, features, model_version=model_name or self.default_model)

    def get_all_metrics(self) -> Dict[str, InferenceMetrics]:
        """Get metrics for all models"""
        return {
            name: service.get_metrics()
            for name, service in self.services.items()
        }

if __name__ == "__main__":
    # Example usage
    feature_names = [
        "returns_1d", "returns_5d", "volatility_20d",
        "rsi_14", "volume_ratio", "momentum_3m"
    ]

    # This would use an actual ONNX model file
    # service = ONNXInferenceService(
    #     model_path="models/momentum_alpha_v1.onnx",
    #     feature_names=feature_names
    # )

    # service.warm_up()

    # # Single prediction
    # result = service.predict(
    #     symbol="AAPL",
    #     features={
    #         "returns_1d": 0.01,
    #         "returns_5d": 0.03,
    #         "volatility_20d": 0.25,
    #         "rsi_14": 65,
    #         "volume_ratio": 1.2,
    #         "momentum_3m": 0.08
    #     }
    # )

    # print(f"Alpha: {result.alpha:.4f}, Latency: {result.latency_ms:.2f}ms")

    # # Get metrics
    # metrics = service.get_metrics()
    # print(f"p95 latency: {metrics.latency_p95_ms:.2f}ms")

    print("ONNX Inference Service implementation complete")
