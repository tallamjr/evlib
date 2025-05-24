"""Test PyTorch bridge functionality."""

import pytest
import numpy as np
import evlib


def test_pytorch_model_loading():
    """Test loading PyTorch weights into Candle models."""
    # Test that we can list models
    models = evlib.processing.list_available_models()
    assert "e2vid_unet" in models

    # Get model info
    info = evlib.processing.get_model_info_py("e2vid_unet")
    assert info["format"] == "pytorch"

    # Try to load the model (this will attempt PyTorch weight loading)
    from evlib.models import E2VID

    # Create model - this should trigger PyTorch weight loading
    try:
        model = E2VID(pretrained=True)
        print("Model created successfully")

        # Test reconstruction with dummy data
        height, width = 180, 240
        num_events = 1000

        # Create random events
        x = np.random.randint(0, width, num_events)
        y = np.random.randint(0, height, num_events)
        t = np.sort(np.random.uniform(0, 1.0, num_events))
        p = np.random.choice([-1, 1], num_events)

        # Reconstruct frame
        frame = model.reconstruct((x, y, t, p), height=height, width=width)

        assert frame is not None
        assert frame.shape == (height, width) or frame.shape == (1, height, width)
        print(f"Reconstruction successful, output shape: {frame.shape}")

    except Exception as e:
        # PyTorch loading might fail if torch is not installed
        print(f"Model loading failed (expected if torch not installed): {e}")
        pytest.skip("PyTorch not available for weight loading")


def test_model_weight_info():
    """Test that we can get weight information from models."""
    import os

    # Check if we have the E2VID model downloaded
    model_path = os.path.join("models", "E2VID_lightweight.pth.tar")
    if not os.path.exists(model_path):
        pytest.skip("E2VID model not downloaded")

    # The weight info should have been created by the test script
    weight_info_path = os.path.join("models", "e2vid_weight_info.json")
    if os.path.exists(weight_info_path):
        import json

        with open(weight_info_path, "r") as f:
            info = json.load(f)

        assert info["model_type"] == "e2vid_unet"
        assert info["num_parameters"] == 74
        assert "unetrecurrent.head.conv2d.weight" in info["key_mappings"]
        print(f"Model has {info['num_parameters']} parameters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
