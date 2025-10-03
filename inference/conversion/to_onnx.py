"""
Model conversion utilities to ONNX format
Converts trained models (PyTorch, TensorFlow, scikit-learn) to ONNX for fast inference
"""
import logging
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def convert_sklearn_to_onnx(
    model,
    input_shape: Tuple[int, ...],
    output_path: Union[str, Path],
    model_name: str = "sklearn_model"
) -> Path:
    """
    Convert scikit-learn model to ONNX

    Args:
        model: Trained scikit-learn model
        input_shape: Input feature shape (e.g., (1, 50) for 50 features)
        output_path: Path to save ONNX model
        model_name: Name for the model

    Returns:
        Path to saved ONNX model
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        # Define input type
        initial_type = [('float_input', FloatTensorType(input_shape))]

        # Convert to ONNX
        onx = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12,
            options={id(model): {'zipmap': False}}  # Don't use ZipMap for cleaner output
        )

        # Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())

        logger.info(f"Converted sklearn model to ONNX: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error converting sklearn model to ONNX: {e}")
        raise


def convert_pytorch_to_onnx(
    model,
    dummy_input: np.ndarray,
    output_path: Union[str, Path],
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None
) -> Path:
    """
    Convert PyTorch model to ONNX

    Args:
        model: Trained PyTorch model
        dummy_input: Example input tensor
        output_path: Path to save ONNX model
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes for variable-length inputs

    Returns:
        Path to saved ONNX model
    """
    try:
        import torch

        # Set model to eval mode
        model.eval()

        # Convert numpy to torch tensor if needed
        if isinstance(dummy_input, np.ndarray):
            dummy_input = torch.from_numpy(dummy_input).float()

        # Default names
        input_names = input_names or ['input']
        output_names = output_names or ['output']

        # Save model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        logger.info(f"Converted PyTorch model to ONNX: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error converting PyTorch model to ONNX: {e}")
        raise


def convert_tensorflow_to_onnx(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    opset: int = 12
) -> Path:
    """
    Convert TensorFlow/Keras model to ONNX

    Args:
        model_path: Path to saved TensorFlow model
        output_path: Path to save ONNX model
        opset: ONNX opset version

    Returns:
        Path to saved ONNX model
    """
    try:
        import tf2onnx
        import tensorflow as tf

        # Load TF model
        model = tf.saved_model.load(str(model_path))

        # Convert to ONNX
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            opset=opset,
            output_path=str(output_path)
        )

        logger.info(f"Converted TensorFlow model to ONNX: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error converting TensorFlow model to ONNX: {e}")
        raise


def verify_onnx_model(
    onnx_path: Union[str, Path],
    sample_input: np.ndarray,
    expected_output_shape: Optional[Tuple[int, ...]] = None
) -> bool:
    """
    Verify ONNX model can be loaded and run

    Args:
        onnx_path: Path to ONNX model
        sample_input: Sample input for testing
        expected_output_shape: Expected output shape

    Returns:
        True if verification passed
    """
    try:
        import onnxruntime as ort

        # Load model
        session = ort.InferenceSession(str(onnx_path))

        # Get input name
        input_name = session.get_inputs()[0].name

        # Run inference
        result = session.run(None, {input_name: sample_input.astype(np.float32)})

        # Check output shape if provided
        if expected_output_shape:
            actual_shape = result[0].shape
            if actual_shape != expected_output_shape:
                logger.warning(
                    f"Output shape mismatch: expected {expected_output_shape}, "
                    f"got {actual_shape}"
                )
                return False

        logger.info(f"ONNX model verification passed: {onnx_path}")
        return True

    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")
        return False


def optimize_onnx_model(
    input_path: Union[str, Path],
    output_path: Union[str, Path]
) -> Path:
    """
    Optimize ONNX model for inference

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model

    Returns:
        Path to optimized model
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer

        # Load model
        model = onnx.load(str(input_path))

        # Optimize
        optimized_model = optimizer.optimize_model(
            str(input_path),
            model_type='bert',  # Can be adjusted based on model type
            num_heads=0,
            hidden_size=0
        )

        # Save optimized model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        optimized_model.save_model_to_file(str(output_path))

        logger.info(f"Optimized ONNX model saved: {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"ONNX optimization failed (using original): {e}")
        return Path(input_path)


if __name__ == "__main__":
    """Example usage"""
    logging.basicConfig(level=logging.INFO)

    # Example: Convert sklearn RandomForest to ONNX
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    # Train a simple model
    X_train = np.random.randn(1000, 50).astype(np.float32)
    y_train = np.random.randn(1000).astype(np.float32)

    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Convert to ONNX
    onnx_path = convert_sklearn_to_onnx(
        model,
        input_shape=(None, 50),  # None for variable batch size
        output_path="models/random_forest.onnx",
        model_name="random_forest_regressor"
    )

    # Verify the model
    sample_input = np.random.randn(1, 50).astype(np.float32)
    verify_onnx_model(onnx_path, sample_input, expected_output_shape=(1, 1))

    print(f"Model successfully converted and verified: {onnx_path}")
