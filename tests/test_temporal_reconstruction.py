"""Test temporal reconstruction with E2VID+ and FireNet+"""

import numpy as np
import evlib


def generate_test_events(num_events=1000, width=128, height=128, duration=0.1):
    """Generate synthetic event data for testing"""
    np.random.seed(42)

    # Generate random events
    xs = np.random.randint(0, width, num_events, dtype=np.int64)
    ys = np.random.randint(0, height, num_events, dtype=np.int64)
    ts = np.sort(np.random.uniform(0, duration, num_events))
    ps = np.random.choice([-1, 1], num_events).astype(np.int64)

    return xs, ys, ts, ps


def test_e2vid_plus_reconstruction():
    """Test E2VID+ temporal reconstruction"""
    # Generate test events
    xs, ys, ts, ps = generate_test_events(num_events=5000)
    height, width = 128, 128
    num_frames = 5

    # Run E2VID+ reconstruction
    frames = evlib.processing.events_to_video_temporal(
        xs, ys, ts, ps, height=height, width=width, num_frames=num_frames, num_bins=5, model_type="e2vid_plus"
    )

    # Check output shape
    assert frames.shape == (num_frames, height, width, 1)
    assert frames.dtype == np.float32

    # Check values are in reasonable range
    assert np.all(frames >= 0) and np.all(frames <= 1)

    # Check that frames are different (temporal processing should produce variation)
    frame_diffs = []
    for i in range(1, num_frames):
        diff = np.mean(np.abs(frames[i] - frames[i - 1]))
        frame_diffs.append(diff)

    # At least some frames should be different
    assert any(diff > 0.001 for diff in frame_diffs)


def test_firenet_plus_reconstruction():
    """Test FireNet+ temporal reconstruction"""
    # Generate test events
    xs, ys, ts, ps = generate_test_events(num_events=3000)
    height, width = 64, 64  # Smaller for FireNet+
    num_frames = 3

    # Run FireNet+ reconstruction
    frames = evlib.processing.events_to_video_temporal(
        xs,
        ys,
        ts,
        ps,
        height=height,
        width=width,
        num_frames=num_frames,
        num_bins=5,
        model_type="firenet_plus",
    )

    # Check output shape
    assert frames.shape == (num_frames, height, width, 1)
    assert frames.dtype == np.float32

    # Check values are in reasonable range
    assert np.all(frames >= 0) and np.all(frames <= 1)


def test_e2vid_plus_small_variant():
    """Test E2VID+ small variant"""
    xs, ys, ts, ps = generate_test_events(num_events=2000)
    height, width = 64, 64
    num_frames = 4

    # Run E2VID+ small variant
    frames = evlib.processing.events_to_video_temporal(
        xs,
        ys,
        ts,
        ps,
        height=height,
        width=width,
        num_frames=num_frames,
        num_bins=3,
        model_type="e2vid_plus_small",
    )

    # Check output
    assert frames.shape == (num_frames, height, width, 1)
    assert frames.dtype == np.float32


def test_empty_events():
    """Test with empty events"""
    xs = np.array([], dtype=np.int64)
    ys = np.array([], dtype=np.int64)
    ts = np.array([], dtype=np.float64)
    ps = np.array([], dtype=np.int64)

    frames = evlib.processing.events_to_video_temporal(
        xs, ys, ts, ps, height=32, width=32, num_frames=2, model_type="firenet_plus"
    )

    # Should return frames with shape
    assert frames.shape == (2, 32, 32, 1)
    # With empty events, the network may produce non-zero outputs due to biases
    assert frames.dtype == np.float32


def test_different_num_bins():
    """Test with different numbers of bins"""
    xs, ys, ts, ps = generate_test_events(num_events=1000)

    for num_bins in [3, 5, 10]:
        frames = evlib.processing.events_to_video_temporal(
            xs, ys, ts, ps, height=32, width=32, num_frames=2, num_bins=num_bins, model_type="e2vid_plus"
        )

        assert frames.shape == (2, 32, 32, 1)


if __name__ == "__main__":
    # Run basic tests
    test_e2vid_plus_reconstruction()
    print("✓ E2VID+ reconstruction test passed")

    test_firenet_plus_reconstruction()
    print("✓ FireNet+ reconstruction test passed")

    test_e2vid_plus_small_variant()
    print("✓ E2VID+ small variant test passed")

    test_empty_events()
    print("✓ Empty events test passed")

    test_different_num_bins()
    print("✓ Different num_bins test passed")

    print("\nAll temporal reconstruction tests passed!")
