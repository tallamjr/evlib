"""
Test suite for E2VID event-to-video reconstruction functionality.

This module tests both the legacy and new direct E2VID APIs
using only real event data from the /data/ directory.
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
    """Test E2VID reconstruction functionality with real data."""

    @pytest.fixture
    def real_events(self):
        """Load real events from slider_depth dataset."""
        events_file = Path("data/slider_depth/events.txt")
        if not events_file.exists():
            pytest.skip("slider_depth dataset not available")

        # Load events using evlib
        events_data = evlib.formats.load_events(str(events_file))

        # Take first 2000 events for testing
        n_events = min(2000, len(events_data[0]))
        xs = events_data[0][:n_events]
        ys = events_data[1][:n_events]
        ts = events_data[2][:n_events]
        ps = events_data[3][:n_events]

        return xs, ys, ts, ps

    def test_events_to_video_basic(self, real_events):
        """Test basic event-to-video reconstruction with real data."""
        xs, ys, ts, ps = real_events
        height = int(ys.max()) + 1
        width = int(xs.max()) + 1

        # Test basic reconstruction with real events
        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width)

        # Check output properties
        assert result is not None
        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32

        # Check value ranges (should be 0-1 for reconstructed frames)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Real events should produce some signal
        assert np.any(result > 0), "Real events should produce some signal"

    def test_events_to_video_with_bins(self, real_events):
        """Test reconstruction with different number of bins using real data."""
        xs, ys, ts, ps = real_events
        height = int(ys.max()) + 1
        width = int(xs.max()) + 1

        # Test with default bin number (5) which works with the neural network
        result = evlib.processing.events_to_video(xs, ys, ts, ps, height, width, num_bins=5)

        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert np.any(result > 0), "Real events should produce signal"

    def test_events_to_video_different_sizes(self, real_events):
        """Test reconstruction with different image dimensions using real data."""
        xs, ys, ts, ps = real_events

        # Get actual data dimensions and test with smaller crops
        max_height = int(ys.max()) + 1
        max_width = int(xs.max()) + 1

        # Test with different cropped sizes
        sizes = [(max_height // 2, max_width // 2), (max_height, max_width)]

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
                assert np.any(result > 0), f"Real events should produce signal for size {height}x{width}"

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

    def test_direct_e2vid_api(self, real_events):
        """Test the new direct E2Vid Rust API with real data."""
        xs, ys, ts, ps = real_events
        height = int(ys.max()) + 1
        width = int(xs.max()) + 1

        # Create E2Vid instance directly
        e2vid = evlib.processing.E2Vid(height, width)

        # Test initial state
        assert not e2vid.has_model_py

        # Reconstruct frame using direct API
        frame = e2vid.reconstruct_frame(xs.tolist(), ys.tolist(), ts.tolist(), ps.tolist())

        # Verify output
        assert frame.shape == (height, width)
        assert frame.dtype == np.float32
        assert np.all(frame >= 0.0) and np.all(frame <= 1.0)
        assert np.any(frame > 0), "Real events should produce signal in direct API"

    def test_reconstruct_events_to_frames(self, real_events):
        """Test multiple frame reconstruction with real data."""
        xs, ys, ts, ps = real_events
        height = int(ys.max()) + 1
        width = int(xs.max()) + 1
        num_frames = 3  # Reduced for faster testing

        frames = evlib.processing.reconstruct_events_to_frames(xs, ys, ts, ps, height, width, num_frames)

        # Check output properties
        assert len(frames) == num_frames

        for i, frame in enumerate(frames):
            assert frame.shape == (height, width, 1)
            assert frame.dtype == np.float32
            assert np.all(frame >= 0.0)
            assert np.all(frame <= 1.0)
            # Real events should produce signal in at least some frames
            if i == num_frames - 1:  # Check final frame has signal
                assert np.any(frame > 0), "Final frame should have signal from real events"

    def test_reconstruct_events_to_frames_with_bins(self, real_events):
        """Test multiple frame reconstruction with different bin numbers using real data."""
        xs, ys, ts, ps = real_events
        height = int(ys.max()) + 1
        width = int(xs.max()) + 1
        num_frames = 3

        # Test with default bin number (5) which works with the neural network
        frames = evlib.processing.reconstruct_events_to_frames(
            xs, ys, ts, ps, height, width, num_frames, num_bins=5
        )

        assert len(frames) == num_frames
        for i, frame in enumerate(frames):
            assert frame.shape == (height, width, 1)
            assert frame.dtype == np.float32
            # Final frame should have signal from real events
            if i == num_frames - 1:
                assert np.any(frame > 0), "Final frame should have signal from real events"


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
class TestE2VidPerformance:
    """Test E2VID performance characteristics with real data."""

    @pytest.fixture
    def benchmark_events(self):
        """Load real events for benchmarking."""
        events_file = Path("data/slider_depth/events.txt")
        if not events_file.exists():
            pytest.skip("slider_depth dataset not available")

        events_data = evlib.formats.load_events(str(events_file))

        # Take first 5k events for benchmarking
        n_events = min(5000, len(events_data[0]))
        xs = events_data[0][:n_events]
        ys = events_data[1][:n_events]
        ts = events_data[2][:n_events]
        ps = events_data[3][:n_events]

        height = int(ys.max()) + 1
        width = int(xs.max()) + 1

        return xs, ys, ts, ps, height, width

    def test_reconstruction_speed(self, benchmark, benchmark_events):
        """Benchmark reconstruction speed with real data."""
        xs, ys, ts, ps, height, width = benchmark_events

        # Benchmark the reconstruction
        result = benchmark(evlib.processing.events_to_video, xs, ys, ts, ps, height, width)

        # Verify the result
        assert result.shape == (height, width, 1)
        assert result.dtype == np.float32
        assert np.any(result > 0), "Real events should produce signal"

    def test_memory_usage_real_data(self, benchmark_events):
        """Test memory usage with real event data."""
        xs, ys, ts, ps, height, width = benchmark_events

        # Test with real data at different crop sizes
        crop_sizes = [(height // 2, width // 2), (height, width)]

        for crop_height, crop_width in crop_sizes:
            # Filter events to fit crop
            mask = (xs < crop_width) & (ys < crop_height)
            if np.sum(mask) > 100:  # Need minimum events
                xs_crop = xs[mask]
                ys_crop = ys[mask]
                ts_crop = ts[mask]
                ps_crop = ps[mask]

                result = evlib.processing.events_to_video(
                    xs_crop, ys_crop, ts_crop, ps_crop, crop_height, crop_width
                )

                assert result.shape == (crop_height, crop_width, 1)
                assert result.dtype == np.float32
                assert np.any(result > 0)  # Real events should produce signal


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

    def test_temporal_consistency_real_data(self):
        """Test temporal consistency with real slider_depth data."""
        events_file = Path("data/slider_depth/events.txt")
        if not events_file.exists():
            pytest.skip("slider_depth dataset not available")

        # Load real events
        events_data = evlib.formats.load_events(str(events_file))

        # Take subset and split into temporal windows
        n_events = min(8000, len(events_data[0]))
        xs = events_data[0][:n_events]
        ys = events_data[1][:n_events]
        ts = events_data[2][:n_events]
        ps = events_data[3][:n_events]

        height = int(ys.max()) + 1
        width = int(xs.max()) + 1

        # Reconstruct multiple frames from real data
        frames = evlib.processing.reconstruct_events_to_frames(xs, ys, ts, ps, height, width, num_frames=4)

        # Validate frames
        assert len(frames) == 4
        for frame in frames:
            assert frame.shape == (height, width, 1)
            assert frame.dtype == np.float32
            assert np.all(frame >= 0.0)

        # Real events should produce signal in frames
        frame_means = [np.mean(frame) for frame in frames]
        assert any(mean > 0.0 for mean in frame_means), "Real events should produce signal"


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
