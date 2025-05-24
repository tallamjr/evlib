#!/usr/bin/env python3
"""Recommended workflow for using PyTorch models with evlib

Since direct PyTorch .pth loading is not yet implemented in the Rust backend,
this script demonstrates the recommended workflow:
1. Convert PyTorch models to ONNX format
2. Use the ONNX models with evlib's unified API

Requirements:
    pip install torch onnx onnxruntime
"""

import evlib
import numpy as np
from pathlib import Path


def main():
    print("PyTorch Model Workflow for evlib")
    print("=" * 50)
    print()

    # Step 1: Convert PyTorch model to ONNX (if needed)
    print("Step 1: Converting PyTorch models to ONNX")
    print("-" * 40)
    print("Use the provided converter script:")
    print("  python examples/pytorch_to_onnx_converter.py \\")
    print("    --model e2vid \\")
    print("    --input path/to/model.pth \\")
    print("    --output models/e2vid.onnx")
    print()

    # Step 2: Use ONNX models with evlib
    print("Step 2: Using ONNX models with evlib")
    print("-" * 40)

    # Create synthetic events for demonstration
    n_events = 10000
    width, height = 346, 260

    xs = np.random.randint(0, width, n_events)
    ys = np.random.randint(0, height, n_events)
    ts = np.sort(np.random.uniform(0, 1.0, n_events))
    ps = np.random.choice([-1, 1], n_events)

    # Method 1: Direct ONNX inference (if ONNX model exists)
    onnx_path = Path("models/e2vid_lightweight.onnx")
    if onnx_path.exists():
        print(f"Using ONNX model: {onnx_path}")
        try:
            frame = evlib.processing.events_to_video_advanced(
                xs, ys, ts, ps, height, width, model_type="onnx", model_path=str(onnx_path)
            )
            print(f"✓ ONNX reconstruction successful: shape {frame.shape}")
        except Exception as e:
            print(f"✗ ONNX reconstruction failed: {e}")
    else:
        print(f"ONNX model not found at {onnx_path}")

    print()

    # Method 2: Use unified API with automatic fallback
    print("Using unified API (recommended):")
    try:
        # The unified API will use ONNX if available, otherwise fallback to Candle
        model = evlib.models.E2VID()
        reconstruction = model.reconstruct((xs, ys, ts, ps), height, width)
        print(f"✓ Unified API reconstruction successful: shape {reconstruction.shape}")

        # Show model info
        print(f"  Model type: {model.config.model_type}")
        print(f"  Number of bins: {model.config.num_bins}")
        print(f"  Device: {model.config.device}")
    except Exception as e:
        print(f"✗ Unified API failed: {e}")

    print()

    # Step 3: Working with different models
    print("Step 3: Available models in evlib")
    print("-" * 40)

    models = [
        ("E2VID", evlib.models.E2VID),
        ("FireNet", evlib.models.FireNet),
        ("E2VID+", evlib.models.E2VIDPlus),
        ("FireNet+", evlib.models.FireNetPlus),
    ]

    for name, model_class in models:
        try:
            model = model_class()
            print(f"✓ {name:10} - Available")
        except Exception as e:
            print(f"✗ {name:10} - {str(e)[:50]}...")

    print()

    # Future support notice
    print("Future Support")
    print("-" * 40)
    print("Direct PyTorch .pth loading is planned for future releases.")
    print("Current workarounds:")
    print("1. Convert models to ONNX format (recommended)")
    print("2. Use pre-trained models from the model zoo")
    print("3. Use the Candle backend with random initialization")
    print()
    print("For the latest updates, see:")
    print("  https://github.com/tallamjr/evlib")


if __name__ == "__main__":
    main()
