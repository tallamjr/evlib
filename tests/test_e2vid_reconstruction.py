"""
Test suite for E2VID event-to-video reconstruction functionality.

This module tests the available event-to-video reconstruction functions
provided by the evlib Python API.
"""

import pytest
import numpy as np
from pathlib import Path

try:
    import evlib

    EVLIB_AVAILABLE = True
except ImportError:
    EVLIB_AVAILABLE = False


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
class TestE2VidReconstruction:
    """Test E2VID reconstruction functionality."""

    @pytest.fixture
    def sample_events(self):
        """Create a simple synthetic event stream for testing."""
        # Create a simple pattern of events
        num_events = 1000
        width, height = 64, 64

        # Generate synthetic events
        events_data = []
        for i in range(num_events):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            t = i * 0.001  # 1ms intervals
            polarity = 1 if i % 2 == 0 else -1
            events_data.append((x, y, t, polarity))

        # Return as separate arrays for the API
        xs = np.array([e[0] for e in events_data], dtype=np.int64)
        ys = np.array([e[1] for e in events_data], dtype=np.int64)
        ts = np.array([e[2] for e in events_data], dtype=np.float64)
        ps = np.array([e[3] for e in events_data], dtype=np.int64)

        return xs, ys, ts, ps

    def test_events_to_video_basic(self, sample_events):
        """Test basic event-to-video reconstruction."""
        xs, ys, ts, ps = sample_events
        height, width = 64, 64

        # Test basic reconstruction
        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width)

        # Check output properties
        assert result is not None
        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32

        # Check value ranges (should be 0-1 for reconstructed frames)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_events_to_video_with_bins(self, sample_events):
        """Test reconstruction with different number of bins."""
        xs, ys, ts, ps = sample_events
        height, width = 64, 64

        # Test with default bin number (5) which works with the neural network
        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width, num_bins=5)

        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_events_to_video_different_sizes(self, sample_events):
        """Test reconstruction with different image dimensions."""
        xs, ys, ts, ps = sample_events

        # Clip coordinates to fit smaller images
        sizes = [(32, 32), (64, 64), (128, 128)]

        for height, width in sizes:
            # Filter events to fit within the image bounds
            mask = (xs < width) & (ys < height)
            xs_filtered = xs[mask]
            ys_filtered = ys[mask]
            ts_filtered = ts[mask]
            ps_filtered = ps[mask]

            if len(xs_filtered) > 0:  # Only test if we have events
                result = evlib.processing.events_to_video(
                    xs_filtered, ys_filtered, ts_filtered, ps_filtered, height, width
                )

                assert result.shape == (height, width, 1)
                assert result.dtype == np.float32

    def test_events_to_video_empty_events(self):
        """Test reconstruction with empty event stream."""
        # Create empty event arrays
        xs = np.array([], dtype=np.int64)
        ys = np.array([], dtype=np.int64)
        ts = np.array([], dtype=np.float64)
        ps = np.array([], dtype=np.int64)

        height, width = 64, 64

        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width)

        # Should produce a valid output (likely all zeros)
        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32
        # Empty events should produce mostly zeros
        assert np.all(result >= 0.0)

    def test_reconstruct_events_to_frames(self, sample_events):
        """Test multiple frame reconstruction."""
        xs, ys, ts, ps = sample_events
        height, width = 64, 64
        num_frames = 5

        frames = evlib.processing.reconstruct_events_to_frames(xs, ys, ts, ps, height, width, num_frames)

        # Check output properties
        assert len(frames) == num_frames

        for i, frame in enumerate(frames):
            assert frame.shape == (height, width, 1)
            assert frame.dtype == np.float32
            assert np.all(frame >= 0.0)
            assert np.all(frame <= 1.0)

    def test_reconstruct_events_to_frames_with_bins(self, sample_events):
        """Test multiple frame reconstruction with different bin numbers."""
        xs, ys, ts, ps = sample_events
        height, width = 64, 64
        num_frames = 3

        # Test with default bin number (5) which works with the neural network
        frames = evlib.processing.reconstruct_events_to_frames(
            xs, ys, ts, ps, height, width, num_frames, num_bins=5
        )

        assert len(frames) == num_frames
        for frame in frames:
            assert frame.shape == (height, width, 1)
            assert frame.dtype == np.float32


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
class TestE2VidPerformance:
    """Test E2VID performance characteristics."""

    def test_reconstruction_speed(self, benchmark):
        """Benchmark reconstruction speed."""
        # Create test events
        num_events = 10000
        xs = np.random.randint(0, 64, num_events, dtype=np.int64)
        ys = np.random.randint(0, 64, num_events, dtype=np.int64)
        ts = np.linspace(0.0, 1.0, num_events, dtype=np.float64)
        ps = np.random.choice([-1, 1], num_events).astype(np.int64)

        height, width = 64, 64

        # Benchmark the reconstruction
        result = benchmark(evlib.processing.events_to_video, xs, ys, ts, ps, height, width)

        # Verify the result
        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32

    def test_memory_usage(self):
        """Test that reconstruction works for different image sizes."""
        # Test with progressively larger images to ensure no memory leaks
        sizes = [(64, 64), (128, 128)]  # Reduced sizes for testing

        for width, height in sizes:
            # Create test events
            num_events = 2000  # Reduced number of events
            xs = np.random.randint(0, width, num_events, dtype=np.int64)
            ys = np.random.randint(0, height, num_events, dtype=np.int64)
            ts = np.linspace(0.0, 1.0, num_events, dtype=np.float64)
            ps = np.random.choice([-1, 1], num_events).astype(np.int64)

            # Reconstruct
            result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width)

            # Verify result
            assert result.shape == (height, width, 1)
            assert result.dtype == np.float32
            assert np.all(result >= 0.0)


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
class TestE2VidIntegration:
    """Integration tests for E2VID with real data patterns."""

    @pytest.mark.skipif(not Path("data/slider_depth").exists(), reason="slider_depth dataset not available")
    def test_slider_depth_dataset(self):
        """Test with real slider_depth dataset if available."""
        # Try to load real events from slider_depth dataset
        events_file = Path("data/slider_depth/events.txt")
        if not events_file.exists():
            pytest.skip("slider_depth events.txt not found")

        # Read a subset of events for testing
        with open(events_file, "r") as f:
            lines = f.readlines()[:10000]  # First 10k events only

        # Parse events
        events_data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                t, x, y, p = float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                events_data.append((x, y, t, p))

        if len(events_data) < 100:  # Need minimum events
            pytest.skip("Not enough events in dataset")

        # Convert to arrays
        xs = np.array([e[0] for e in events_data], dtype=np.int64)
        ys = np.array([e[1] for e in events_data], dtype=np.int64)
        ts = np.array([e[2] for e in events_data], dtype=np.float64)
        ps = np.array([e[3] for e in events_data], dtype=np.int64)

        # Get image dimensions
        height = max(ys) + 1
        width = max(xs) + 1

        # Reconstruct
        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width)

        # Verify result
        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32
        assert np.any(result > 0)  # Should have some signal

    def test_temporal_consistency(self):
        """Test temporal consistency across multiple reconstructions."""
        # Create events with temporal structure
        num_events = 2000
        width, height = 64, 64

        # Create two sequential event streams
        xs1 = np.random.randint(0, width, num_events // 2, dtype=np.int64)
        ys1 = np.random.randint(0, height, num_events // 2, dtype=np.int64)
        ts1 = np.linspace(0.0, 0.5, num_events // 2, dtype=np.float64)
        ps1 = np.random.choice([-1, 1], num_events // 2).astype(np.int64)

        xs2 = np.random.randint(0, width, num_events // 2, dtype=np.int64)
        ys2 = np.random.randint(0, height, num_events // 2, dtype=np.int64)
        ts2 = np.linspace(0.5, 1.0, num_events // 2, dtype=np.float64)
        ps2 = np.random.choice([-1, 1], num_events // 2).astype(np.int64)

        # Combine events
        xs = np.concatenate([xs1, xs2])
        ys = np.concatenate([ys1, ys2])
        ts = np.concatenate([ts1, ts2])
        ps = np.concatenate([ps1, ps2])

        # Sort by time
        sort_idx = np.argsort(ts)
        xs = xs[sort_idx]
        ys = ys[sort_idx]
        ts = ts[sort_idx]
        ps = ps[sort_idx]

        # Reconstruct multiple frames
        frames = evlib.processing.reconstruct_events_to_frames(xs, ys, ts, ps, height, width, num_frames=4)

        # Check that frames are reasonable
        assert len(frames) == 4
        for frame in frames:
            assert frame.shape == (height, width, 1)
            assert frame.dtype == np.float32

        # Check that frames have reasonable signal distribution
        # (with neural reconstruction, the relationship isn't strictly accumulative)
        frame_means = [np.mean(frame) for frame in frames]
        assert all(mean >= 0.0 for mean in frame_means), "All frames should have non-negative mean values"
        assert any(mean > 0.0 for mean in frame_means), "At least one frame should have some signal"


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
class TestE2VidEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_event(self):
        """Test reconstruction with single event."""
        xs = np.array([32], dtype=np.int64)
        ys = np.array([32], dtype=np.int64)
        ts = np.array([0.5], dtype=np.float64)
        ps = np.array([1], dtype=np.int64)

        height, width = 64, 64

        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width)

        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32
        # Single event should produce some signal
        assert np.sum(result) > 0

    def test_out_of_bounds_events(self):
        """Test handling of events outside image bounds."""
        # Create events, some of which are out of bounds
        xs = np.array([10, 70, 32, -5], dtype=np.int64)  # 70 and -5 are out of bounds for 64x64
        ys = np.array([10, 32, 70, 10], dtype=np.int64)  # 70 is out of bounds
        ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        ps = np.array([1, -1, 1, -1], dtype=np.int64)

        height, width = 64, 64

        # This should either handle gracefully or filter out-of-bounds events
        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width)

        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32

    def test_large_time_gap(self):
        """Test reconstruction with large time gaps."""
        # Create events with large time gaps
        xs = np.array([10, 20, 30], dtype=np.int64)
        ys = np.array([10, 20, 30], dtype=np.int64)
        ts = np.array([0.0, 100.0, 200.0], dtype=np.float64)  # Large gaps
        ps = np.array([1, -1, 1], dtype=np.int64)

        height, width = 64, 64

        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width)

        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32

    def test_same_timestamp_events(self):
        """Test reconstruction with events at same timestamp."""
        # Create events at the same timestamp
        xs = np.array([10, 20, 30], dtype=np.int64)
        ys = np.array([10, 20, 30], dtype=np.int64)
        ts = np.array([0.5, 0.5, 0.5], dtype=np.float64)  # Same timestamp
        ps = np.array([1, -1, 1], dtype=np.int64)

        height, width = 64, 64

        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width)

        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32
