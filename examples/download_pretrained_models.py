#!/usr/bin/env python3
"""Download pre-trained E2VID models and convert them to ONNX format

This script downloads official pre-trained models and converts them for use with evlib.
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve


def download_file(url, destination, description=""):
    """Download file with progress indicator"""
    print(f"Downloading {description or url}...")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f"\r  Progress: {percent:.1f}%")
        sys.stdout.flush()

    try:
        urlretrieve(url, destination, reporthook=report_progress)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def setup_model_directory():
    """Create models directory if it doesn't exist"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir


def download_e2vid_models():
    """Download pre-trained E2VID models"""
    models_dir = setup_model_directory()

    # Model URLs (from official repositories)
    models = {
        "e2vid_lightweight": {
            "url": "https://github.com/uzh-rpg/rpg_e2vid/raw/master/pretrained/E2VID_lightweight.pth.tar",
            "filename": "e2vid_lightweight.pth.tar",
            "description": "E2VID Lightweight model",
        },
        # Note: FireNet model URL would go here when available
        # "firenet": {
        #     "url": "https://...",
        #     "filename": "firenet.pth",
        #     "description": "FireNet model"
        # }
    }

    downloaded = []
    for model_name, info in models.items():
        destination = models_dir / info["filename"]

        if destination.exists():
            print(f"✓ {info['description']} already exists at {destination}")
            downloaded.append((model_name, destination))
            continue

        if download_file(info["url"], destination, info["description"]):
            print(f"✓ Downloaded {info['description']} to {destination}")
            downloaded.append((model_name, destination))

    return downloaded


def convert_models_to_onnx(models):
    """Convert downloaded models to ONNX format"""
    try:
        from pytorch_to_onnx_converter import (
            convert_to_onnx,
            load_pytorch_model,
            verify_onnx_model,
        )
    except ImportError:
        print(
            "Error: pytorch_to_onnx_converter.py not found in current directory\n"
            "Please run this script from the examples directory"
        )
        return

    print("\nConverting models to ONNX format...")

    for model_name, model_path in models:
        onnx_path = model_path.with_suffix(".onnx")

        if onnx_path.exists():
            print(f"✓ ONNX model already exists: {onnx_path}")
            continue

        print(f"\nConverting {model_name}...")
        try:
            # Determine model type
            if "firenet" in model_name.lower():
                model_type = "firenet"
            else:
                model_type = "e2vid"

            # Load and convert
            model = load_pytorch_model(str(model_path), model_type)
            convert_to_onnx(model, str(onnx_path), verbose=False)

            # Verify
            if verify_onnx_model(onnx_path):
                print(f"✓ Successfully converted to: {onnx_path}")
            else:
                print(f"✗ Conversion failed for: {model_name}")

        except Exception as e:
            print(f"✗ Error converting {model_name}: {e}")


def create_example_script():
    """Create example usage script"""
    example_code = '''#!/usr/bin/env python3
"""Example: Using pre-trained ONNX models with evlib"""

import numpy as np
import evlib


def load_test_events(max_events=10000):
    """Load events from slider_depth dataset"""
    events_path = "../data/slider_depth/events.txt"
    try:
        events_data = np.loadtxt(events_path, max_rows=max_events)
        ts = events_data[:, 0]
        xs = events_data[:, 1].astype(np.int64)
        ys = events_data[:, 2].astype(np.int64)
        ps = events_data[:, 3].astype(np.int64)
        return xs, ys, ts, ps, True
    except FileNotFoundError:
        print("Test data not found, using synthetic events")
        # Generate synthetic events
        n = max_events
        xs = np.random.randint(0, 240, n, dtype=np.int64)
        ys = np.random.randint(0, 180, n, dtype=np.int64)
        ts = np.sort(np.random.uniform(0, 0.1, n))
        ps = np.random.choice([-1, 1], n).astype(np.int64)
        return xs, ys, ts, ps, False


def main():
    # Load events
    xs, ys, ts, ps, real_data = load_test_events()
    height, width = 180, 240

    print(f"Loaded {len(xs)} events (real data: {real_data})")

    # Test with ONNX model
    model_path = "models/e2vid_lightweight.onnx"
    print(f"\\nUsing ONNX model: {model_path}")

    try:
        frame = evlib.processing.events_to_video_advanced(
            xs, ys, ts, ps, height, width,
            num_bins=5,
            model_type="onnx",
            model_path=model_path
        )
        print(f"✓ ONNX reconstruction successful!")
        print(f"  Output shape: {frame.shape}")
        print(f"  Value range: [{frame.min():.3f}, {frame.max():.3f}]")
    except Exception as e:
        print(f"✗ ONNX model failed: {e}")
        print("  Falling back to built-in UNet model...")

        frame = evlib.processing.events_to_video_advanced(
            xs, ys, ts, ps, height, width,
            num_bins=5,
            model_type="unet"
        )
        print(f"✓ UNet reconstruction successful!")
        print(f"  Output shape: {frame.shape}")
        print(f"  Value range: [{frame.min():.3f}, {frame.max():.3f}]")

    # Save output
    if frame is not None:
        import matplotlib.pyplot as plt
        plt.imsave("reconstruction_output.png", frame[:, :, 0], cmap='gray')
        print("\\n✓ Saved output to: reconstruction_output.png")


if __name__ == "__main__":
    main()
'''

    with open("test_pretrained_models.py", "w") as f:
        f.write(example_code)
    print("\n✓ Created example script: test_pretrained_models.py")


def main():
    print("E2VID Pre-trained Model Setup")
    print("=" * 40)

    # Download models
    models = download_e2vid_models()

    if not models:
        print("\nNo models downloaded.")
        return

    # Convert to ONNX
    convert_models_to_onnx(models)

    # Create example
    create_example_script()

    print("\n" + "=" * 40)
    print("Setup complete! To test the models, run:")
    print("  python test_pretrained_models.py")


if __name__ == "__main__":
    main()
