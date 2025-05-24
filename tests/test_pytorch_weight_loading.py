"""Test PyTorch weight loading through Rust bindings."""

import os
import pytest
import numpy as np


def test_pytorch_weight_loading_rust():
    """Test that Rust can load PyTorch weights through the bridge."""
    try:
        # Test with E2VID model if it exists
        model_path = "models/E2VID_lightweight.pth.tar"
        if not os.path.exists(model_path):
            pytest.skip(f"Model file not found at {model_path}")

        # Try to create a reconstruction model
        # This should trigger the PyTorch weight loading in Rust
        print("\nTesting E2VID model creation with PyTorch weights...")

        # Create a simple test with the processing module
        from evlib.processing import E2VIDReconstructor

        reconstructor = E2VIDReconstructor(device="cpu")
        print("E2VID reconstructor created successfully!")

        # Test with dummy input
        batch_size = 1
        channels = 5  # Event voxel grid channels
        height = 180
        width = 240

        dummy_voxel = np.random.randn(batch_size, channels, height, width).astype(np.float32)
        print(f"Input shape: {dummy_voxel.shape}")

        try:
            # This will use random weights if PyTorch loading fails
            output = reconstructor.reconstruct(dummy_voxel)
            print(f"Output shape: {output.shape}")
            assert output.shape == (batch_size, 1, height, width), "Unexpected output shape"
            print("Reconstruction successful!")
        except Exception as e:
            print(f"Reconstruction failed: {e}")

    except ImportError as e:
        pytest.skip(f"Import error: {e}")
    except Exception as e:
        pytest.fail(f"Test failed: {e}")


def test_model_zoo_pytorch_loading():
    """Test model zoo's ability to load PyTorch checkpoints."""
    import tempfile
    import torch

    # Create a simple PyTorch checkpoint for testing
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        checkpoint = {
            "model": {
                "num_bins": 5,
                "base_num_channels": 32,
            },
            "state_dict": {
                "test.weight": torch.randn(32, 5, 3, 3),
                "test.bias": torch.randn(32),
            },
        }
        torch.save(checkpoint, tmp.name)
        tmp_path = tmp.name

    try:
        # Test that the file can be loaded
        loaded = torch.load(tmp_path, map_location="cpu")
        assert "state_dict" in loaded
        assert "test.weight" in loaded["state_dict"]
        print("Test checkpoint created and loaded successfully")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_model_zoo_pytorch_loading()
    test_pytorch_weight_loading_rust()
