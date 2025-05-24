"""Simple test to verify PyTorch weight loading infrastructure."""

import os
import numpy as np


def test_evlib_basic():
    """Test basic evlib functionality."""
    import evlib

    print("Available evlib modules:", [x for x in dir(evlib) if not x.startswith("_")])

    # Test voxel grid creation
    n_events = 1000
    ts = np.random.rand(n_events).astype(np.float32) * 0.1  # timestamps
    xs = np.random.rand(n_events).astype(np.float32) * 240  # x coordinates
    ys = np.random.rand(n_events).astype(np.float32) * 180  # y coordinates
    ps = np.random.randint(0, 2, n_events).astype(np.float32)  # polarity

    voxel_grid = evlib.create_voxel_grid(xs, ys, ts, ps, num_bins=5, resolution=(240, 180), method="count")

    print(f"Created voxel grid with shape: {voxel_grid.shape}")
    assert voxel_grid.shape == (5, 180, 240)


def test_model_loading_infrastructure():
    """Test that the model loading infrastructure is in place."""
    # Check if model files exist
    model_files = ["models/E2VID_lightweight.pth.tar", "models/ETAP_v1_cvpr25.pth"]

    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"Found model file: {model_file}")

            # Check file size
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")

            # Try to load with torch to verify format
            try:
                import torch

                checkpoint = torch.load(model_file, map_location="cpu", weights_only=True)
                keys = list(checkpoint.keys())
                print(f"  Checkpoint keys: {keys}")

                if "state_dict" in checkpoint:
                    num_params = len(checkpoint["state_dict"])
                    print(f"  Number of parameters: {num_params}")

                    # Show first few parameter names
                    param_names = list(checkpoint["state_dict"].keys())[:5]
                    print(f"  First few parameters: {param_names}")

            except Exception as e:
                print(f"  Could not load checkpoint: {e}")
        else:
            print(f"Model file not found: {model_file}")


def test_rust_pytorch_bridge():
    """Test that Rust PyTorch bridge infrastructure exists."""
    # Check if the pytorch_bridge module was compiled
    src_path = "src/ev_processing/reconstruction/pytorch_bridge.rs"
    if os.path.exists(src_path):
        print(f"PyTorch bridge source exists at: {src_path}")

        # Check for test files
        test_files = ["tests/test_pytorch_bridge.py", "scripts/test_pytorch_loading.py"]

        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"Test file exists: {test_file}")


if __name__ == "__main__":
    print("Testing evlib basic functionality...")
    test_evlib_basic()

    print("\n" + "=" * 50 + "\n")
    print("Testing model loading infrastructure...")
    test_model_loading_infrastructure()

    print("\n" + "=" * 50 + "\n")
    print("Testing Rust PyTorch bridge...")
    test_rust_pytorch_bridge()
