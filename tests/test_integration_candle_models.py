"""Integration test for E2VID Candle architectures with real data"""

import numpy as np
import pytest
import evlib


def test_unet_with_real_data():
    """Test UNet architecture with slider_depth data"""
    # Load test events
    events_path = "data/slider_depth/events.txt"
    try:
        # Read a subset of events
        events_data = np.loadtxt(events_path, max_rows=10000)
        ts = events_data[:, 0]
        xs = events_data[:, 1].astype(np.int64)
        ys = events_data[:, 2].astype(np.int64)
        ps = events_data[:, 3].astype(np.int64)

        # Image dimensions from calib.txt
        height, width = 180, 240

        # Test UNet architecture
        frame_unet = evlib.processing.events_to_video_advanced(
            xs, ys, ts, ps, height, width, num_bins=5, model_type="unet"
        )

        assert frame_unet.shape == (height, width, 1)
        assert frame_unet.dtype == np.float32
        assert np.all(frame_unet >= 0.0) and np.all(frame_unet <= 1.0)

        # Test FireNet architecture (should be faster)
        frame_firenet = evlib.processing.events_to_video_advanced(
            xs, ys, ts, ps, height, width, num_bins=5, model_type="firenet"
        )

        assert frame_firenet.shape == (height, width, 1)
        assert frame_firenet.dtype == np.float32
        assert np.all(frame_firenet >= 0.0) and np.all(frame_firenet <= 1.0)

        print(
            f"✓ UNet output shape: {frame_unet.shape}, range: [{frame_unet.min():.3f}, {frame_unet.max():.3f}]"
        )
        print(
            f"✓ FireNet output shape: {frame_firenet.shape}, range: [{frame_firenet.min():.3f}, {frame_firenet.max():.3f}]"
        )

    except FileNotFoundError:
        pytest.skip("Test data not found at data/slider_depth/events.txt")


def test_model_output_validity():
    """Test that models produce valid outputs"""
    # Generate synthetic events
    n_events = 5000
    width, height = 256, 256
    np.random.seed(42)

    xs = np.random.randint(0, width, n_events, dtype=np.int64)
    ys = np.random.randint(0, height, n_events, dtype=np.int64)
    ts = np.sort(np.random.uniform(0, 1.0, n_events))
    ps = np.random.choice([-1, 1], n_events).astype(np.int64)

    # Test UNet output validity
    frame = evlib.processing.events_to_video_advanced(
        xs, ys, ts, ps, height, width, num_bins=5, model_type="unet"
    )

    # Check output properties
    assert frame.shape == (height, width, 1), f"Expected shape {(height, width, 1)}, got {frame.shape}"
    assert frame.dtype == np.float32, f"Expected dtype float32, got {frame.dtype}"
    assert np.all(frame >= 0.0) and np.all(frame <= 1.0), "Output values not in [0, 1] range"
    assert not np.all(frame == frame.flat[0]), "Output is constant (no variation)"

    print("✓ Model output validity test passed")


def test_model_comparison():
    """Compare outputs of different models"""
    # Generate events
    n_events = 2000
    width, height = 128, 128
    np.random.seed(123)

    xs = np.random.randint(0, width, n_events, dtype=np.int64)
    ys = np.random.randint(0, height, n_events, dtype=np.int64)
    ts = np.sort(np.random.uniform(0, 0.1, n_events))
    ps = np.random.choice([-1, 1], n_events).astype(np.int64)

    # Test all model types
    models = ["unet", "firenet", "simple"]
    results = {}

    for model_type in models:
        frame = evlib.processing.events_to_video_advanced(
            xs, ys, ts, ps, height, width, num_bins=5, model_type=model_type
        )
        results[model_type] = frame

        print(
            f"✓ {model_type:8} - min: {frame.min():.4f}, max: {frame.max():.4f}, "
            f"mean: {frame.mean():.4f}, std: {frame.std():.4f}"
        )

    # Neural models should produce different outputs than simple accumulation
    assert not np.allclose(results["unet"], results["simple"], rtol=0.1)
    assert not np.allclose(results["firenet"], results["simple"], rtol=0.1)

    print("✓ All models produce distinct outputs as expected")


if __name__ == "__main__":
    print("Running E2VID Candle architecture integration tests...")
    test_model_output_validity()
    test_model_comparison()
    test_unet_with_real_data()
    print("\nAll tests passed! ✓")
