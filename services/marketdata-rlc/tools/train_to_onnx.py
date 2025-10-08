"""
Train tiny LightGBM models (latency + error probability) on synthetic data and export to ONNX.
This script is a placeholder – replace synthetic data with production rollups when available.
"""
from __future__ import annotations

import os

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from onnxmltools import convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(42)
N_SAMPLES = 6000
N_FEATURES = 12


def _make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = RNG.normal(size=(N_SAMPLES, N_FEATURES)).astype(np.float32)

    y_latency = 120 + 40 * X[:, 7] + 25 * np.abs(X[:, 8]) + 10 * RNG.normal(size=N_SAMPLES)
    y_latency = y_latency.astype(np.float32)

    logits = 0.2 * X[:, 8] + 0.15 * X[:, 6] + 0.1 * RNG.normal(size=N_SAMPLES)
    prob = 1 / (1 + np.exp(-logits))
    y_error = (RNG.uniform(size=N_SAMPLES) < prob).astype(np.int32)

    return X, y_latency, y_error


def main() -> None:
    os.makedirs("models", exist_ok=True)
    X, y_latency, y_error = _make_dataset()

    Xtr, Xte, ytr_lat, yte_lat = train_test_split(X, y_latency, test_size=0.2, random_state=42)
    Xtr2, Xte2, ytr_err, yte_err = train_test_split(X, y_error, test_size=0.2, random_state=42)

    reg = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    reg.fit(Xtr, ytr_lat)

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    clf.fit(Xtr2, ytr_err)

    initial_types = [("input", FloatTensorType([None, N_FEATURES]))]
    latency_model = convert_lightgbm(reg, initial_types=initial_types, target_opset=15)
    error_model = convert_lightgbm(clf, initial_types=initial_types, target_opset=15)

    latency_path = os.path.join("models", "rlc_latency.onnx")
    error_path = os.path.join("models", "rlc_error.onnx")
    with open(latency_path, "wb") as f:
        f.write(latency_model.SerializeToString())
    with open(error_path, "wb") as f:
        f.write(error_model.SerializeToString())

    print(f"Saved ONNX models to {latency_path} and {error_path}")


if __name__ == "__main__":
    main()
