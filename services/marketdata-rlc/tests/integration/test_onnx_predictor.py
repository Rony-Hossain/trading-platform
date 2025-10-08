import importlib.util
import os
import sys

import numpy as np
import pytest

from app.models.onnx_runtime import RlcPredictor


def _run_trainer() -> None:
    trainer_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "tools", "train_to_onnx.py")
    )
    spec = importlib.util.spec_from_file_location("trainer", trainer_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["trainer"] = module
    assert spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predictor_end_to_end(tmp_path) -> None:
    _run_trainer()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    latency_path = os.path.join(base_dir, "models", "rlc_latency.onnx")
    error_path = os.path.join(base_dir, "models", "rlc_error.onnx")
    assert os.path.exists(latency_path), "Latency model missing"
    assert os.path.exists(error_path), "Error model missing"

    predictor = RlcPredictor(latency_path, error_path)
    X = np.random.rand(3, 12).astype(np.float32)
    latency, prob = predictor.predict(X)
    assert latency.shape == (3,)
    assert prob.shape == (3,)
    assert np.isfinite(latency).all()
    assert np.isfinite(prob).all()
