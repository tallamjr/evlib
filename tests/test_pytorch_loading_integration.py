"""Test PyTorch model loading integration with real models."""

import os
import torch
from pathlib import Path


def test_e2vid_model_loading():
    """Test loading the actual E2VID model."""
    model_path = Path("models/E2VID_lightweight.pth.tar")

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    print("\nE2VID Model Structure:")
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")

    # The actual weights are in 'state_dict'
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print(f"\nTotal parameters: {len(state_dict)}")

        # Group parameters by module
        modules = {}
        for key in state_dict.keys():
            module = key.split(".")[0]
            if module not in modules:
                modules[module] = []
            modules[module].append(key)

        print("\nModule breakdown:")
        for module, params in modules.items():
            print(f"  {module}: {len(params)} parameters")

        # Check ConvLSTM parameters
        convlstm_params = [k for k in state_dict.keys() if "convlstm" in k]
        print(f"\nConvLSTM parameters: {len(convlstm_params)}")

        # Sample some parameter shapes
        print("\nSample parameter shapes:")
        for i, (key, value) in enumerate(state_dict.items()):
            if i < 10:
                print(f"  {key}: {value.shape}")

        # Test conversion to numpy
        print("\nTesting tensor conversion:")
        sample_keys = list(state_dict.keys())[:3]
        for key in sample_keys:
            tensor = state_dict[key]
            np_array = tensor.detach().cpu().numpy()
            print(f"  {key}: torch.Size{list(tensor.shape)} -> numpy shape {np_array.shape}")

    # Print model configuration
    if "model" in checkpoint:
        print("\nModel configuration:")
        for key, value in checkpoint["model"].items():
            print(f"  {key}: {value}")


def test_model_zoo_integration():
    """Test model zoo can load PyTorch weights."""
    try:
        from evlib.models import ReconstructionModel

        # Try to create a model with PyTorch weights
        model_path = "models/E2VID_lightweight.pth.tar"
        if os.path.exists(model_path):
            print("\nTesting ReconstructionModel with PyTorch weights:")
            model = ReconstructionModel(model_type="e2vid", device="cpu", model_path=model_path)
            print("Model created successfully!")

            # Test with dummy input
            import numpy as np

            batch_size = 1
            channels = 5  # Event voxel grid channels
            height = 180
            width = 240

            dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
            print(f"Input shape: {dummy_input.shape}")

            try:
                output = model.reconstruct(dummy_input)
                print(f"Output shape: {output.shape}")
            except Exception as e:
                print(f"Reconstruction failed: {e}")

    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_e2vid_model_loading()
    print("\n" + "=" * 50 + "\n")
    test_model_zoo_integration()
