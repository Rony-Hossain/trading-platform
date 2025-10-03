"""
Tests for inference service latency SLAs
Acceptance Criteria:
- p99 inference latency ≤ 50ms under 2× prod QPS
- p99 ≤ 100ms under 5× prod QPS (stress test)
- Model warmup time < 30s
- Zero OOM errors under 3× prod QPS
"""
import pytest
import asyncio
import time
import numpy as np
from pathlib import Path
import sys
from statistics import quantiles

# Add inference to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'inference' / 'deployment'))
sys.path.insert(0, str(project_root / 'inference' / 'conversion'))

from inference_service import ONNXModelWrapper, InferenceService


@pytest.fixture
def sample_model_path(tmp_path):
    """Create a sample ONNX model for testing"""
    from sklearn.ensemble import RandomForestRegressor
    from to_onnx import convert_sklearn_to_onnx

    # Train simple model
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)

    # Convert to ONNX
    onnx_path = tmp_path / "test_model.onnx"
    convert_sklearn_to_onnx(
        model,
        input_shape=(None, 50),
        output_path=onnx_path,
        model_name="test_model"
    )

    return onnx_path


@pytest.fixture
def model_wrapper(sample_model_path):
    """Fixture for model wrapper"""
    wrapper = ONNXModelWrapper(
        model_path=sample_model_path,
        model_name="test_model"
    )
    return wrapper


def test_model_warmup_time(sample_model_path):
    """Test that model warmup completes in < 30s"""
    model = ONNXModelWrapper(
        model_path=sample_model_path,
        model_name="test_model"
    )

    start_time = time.time()
    model.warmup(input_shape=(1, 50), num_iterations=100)
    warmup_time = time.time() - start_time

    # Should complete in < 30s
    assert warmup_time < 30.0
    assert model.warmup_done

    print(f"Warmup time: {warmup_time:.3f}s")


def test_single_prediction_latency(model_wrapper):
    """Test single prediction latency"""
    # Warmup first
    model_wrapper.warmup(input_shape=(1, 50), num_iterations=50)

    # Run predictions and measure latency
    latencies = []
    num_predictions = 1000

    for _ in range(num_predictions):
        features = np.random.randn(1, 50).astype(np.float32)

        start_time = time.time()
        result = model_wrapper.predict(features)
        latency_ms = (time.time() - start_time) * 1000

        latencies.append(latency_ms)

    # Calculate percentiles
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"\nSingle prediction latency:")
    print(f"  p50: {p50:.2f}ms")
    print(f"  p95: {p95:.2f}ms")
    print(f"  p99: {p99:.2f}ms")

    # Assert p99 < 50ms (SLA)
    assert p99 < 50.0, f"p99 latency {p99:.2f}ms exceeds 50ms SLA"


def test_batch_prediction_latency(model_wrapper):
    """Test batch prediction latency"""
    model_wrapper.warmup(input_shape=(1, 50), num_iterations=50)

    batch_sizes = [1, 4, 8, 16, 32]
    results = {}

    for batch_size in batch_sizes:
        latencies = []
        num_batches = 100

        for _ in range(num_batches):
            features = np.random.randn(batch_size, 50).astype(np.float32)

            start_time = time.time()
            result = model_wrapper.predict(features)
            latency_ms = (time.time() - start_time) * 1000

            latencies.append(latency_ms)

        p99 = np.percentile(latencies, 99)
        throughput = (batch_size * num_batches) / (sum(latencies) / 1000)  # predictions/sec

        results[batch_size] = {"p99_ms": p99, "throughput": throughput}

        print(f"\nBatch size {batch_size}:")
        print(f"  p99 latency: {p99:.2f}ms")
        print(f"  Throughput: {throughput:.0f} predictions/sec")


@pytest.mark.asyncio
async def test_concurrent_requests(sample_model_path):
    """Test inference under concurrent load (2x prod QPS)"""
    service = InferenceService(models_dir=sample_model_path.parent)
    service.load_model(
        model_name="test_model",
        model_filename=sample_model_path.name,
        warmup=True,
        warmup_shape=(1, 50)
    )

    # Simulate 2x production QPS
    num_requests = 200
    concurrent_requests = 10

    async def make_request():
        features = np.random.randn(1, 50).astype(np.float32)
        start_time = time.time()
        result = await service.predict("test_model", features)
        latency_ms = (time.time() - start_time) * 1000
        return latency_ms

    # Run concurrent requests
    start_time = time.time()
    tasks = []

    for _ in range(num_requests):
        tasks.append(make_request())

    latencies = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    # Calculate metrics
    p99 = np.percentile(latencies, 99)
    qps = num_requests / total_time

    print(f"\nConcurrent load test (2x prod QPS):")
    print(f"  Total requests: {num_requests}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  QPS: {qps:.0f}")
    print(f"  p99 latency: {p99:.2f}ms")

    # Assert p99 < 50ms under 2x load
    assert p99 < 50.0, f"p99 latency {p99:.2f}ms exceeds 50ms SLA under 2x load"


@pytest.mark.asyncio
async def test_stress_test_5x_qps(sample_model_path):
    """Stress test at 5x production QPS"""
    service = InferenceService(models_dir=sample_model_path.parent)
    service.load_model(
        model_name="test_model",
        model_filename=sample_model_path.name,
        warmup=True,
        warmup_shape=(1, 50)
    )

    # Simulate 5x production QPS
    num_requests = 500

    async def make_request():
        features = np.random.randn(1, 50).astype(np.float32)
        start_time = time.time()
        try:
            result = await service.predict("test_model", features)
            latency_ms = (time.time() - start_time) * 1000
            return {"latency_ms": latency_ms, "error": None}
        except Exception as e:
            return {"latency_ms": None, "error": str(e)}

    # Run stress test
    start_time = time.time()
    results = await asyncio.gather(*[make_request() for _ in range(num_requests)])
    total_time = time.time() - start_time

    # Analyze results
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    errors = [r for r in results if r["error"] is not None]

    p99 = np.percentile(latencies, 99) if latencies else float('inf')
    qps = num_requests / total_time
    error_rate = len(errors) / num_requests

    print(f"\nStress test (5x prod QPS):")
    print(f"  Total requests: {num_requests}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  QPS: {qps:.0f}")
    print(f"  p99 latency: {p99:.2f}ms")
    print(f"  Error rate: {error_rate:.2%}")

    # Assert p99 < 100ms under 5x load
    assert p99 < 100.0, f"p99 latency {p99:.2f}ms exceeds 100ms SLA under 5x load"

    # Assert zero errors (no OOM)
    assert error_rate == 0, f"Error rate {error_rate:.2%} under stress test"


def test_memory_usage(model_wrapper):
    """Test that memory usage is reasonable (no OOM errors)"""
    import psutil
    import os

    process = psutil.Process(os.getpid())

    # Warmup
    model_wrapper.warmup(input_shape=(1, 50), num_iterations=50)

    # Baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Run many predictions
    for _ in range(1000):
        features = np.random.randn(1, 50).astype(np.float32)
        result = model_wrapper.predict(features)

    # Final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - baseline_memory

    print(f"\nMemory usage:")
    print(f"  Baseline: {baseline_memory:.2f} MB")
    print(f"  Final: {final_memory:.2f} MB")
    print(f"  Increase: {memory_increase:.2f} MB")

    # Memory increase should be minimal (< 100MB for this test)
    assert memory_increase < 100, f"Memory increase {memory_increase:.2f}MB too high"


def test_throughput_scaling(model_wrapper):
    """Test throughput scaling with batch size"""
    model_wrapper.warmup(input_shape=(1, 50), num_iterations=50)

    batch_sizes = [1, 2, 4, 8, 16, 32]
    throughputs = []

    for batch_size in batch_sizes:
        num_batches = 100
        start_time = time.time()

        for _ in range(num_batches):
            features = np.random.randn(batch_size, 50).astype(np.float32)
            result = model_wrapper.predict(features)

        elapsed_time = time.time() - start_time
        throughput = (batch_size * num_batches) / elapsed_time
        throughputs.append(throughput)

        print(f"Batch size {batch_size}: {throughput:.0f} predictions/sec")

    # Throughput should increase with batch size
    assert throughputs[-1] > throughputs[0], "Throughput should scale with batch size"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
