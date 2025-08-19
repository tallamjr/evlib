"""
Tests for evlib visualization functionality.

Tests both the Python visualization module and integration with eTram data.
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Test imports
try:
    import evlib.visualization as viz
    import cv2
    import h5py
except ImportError as e:
    pytest.skip(f"Visualization dependencies not available: {e}", allow_module_level=True)


class TestVisualizationConfig:
    """Test visualization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = viz.VisualizationConfig()

        assert config.width == 640
        assert config.height == 360
        assert config.fps == 30.0
        assert config.positive_color == (0, 0, 255)  # Red in BGR
        assert config.negative_color == (255, 0, 0)  # Blue in BGR
        assert config.background_color == (0, 0, 0)  # Black
        assert config.decay_ms == 100.0
        assert config.show_stats is True
        assert config.codec == "mp4v"

    def test_frame_duration_calculation(self):
        """Test that frame duration is calculated correctly."""
        config = viz.VisualizationConfig(fps=60.0)
        assert abs(config.frame_duration_ms - 16.667) < 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = viz.VisualizationConfig(
            width=1920,
            height=1080,
            fps=60.0,
            positive_color=(255, 255, 0),
            negative_color=(255, 0, 255),
            decay_ms=50.0,
            show_stats=False,
        )

        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 60.0
        assert config.positive_color == (255, 255, 0)
        assert config.negative_color == (255, 0, 255)
        assert config.decay_ms == 50.0
        assert config.show_stats is False


class TesteTramDataLoader:
    """Test eTram data loader functionality."""

    @pytest.fixture
    def mock_etram_data(self):
        """Create mock eTram data structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "test_data"
            data_dir.mkdir()

            # Create the expected directory structure
            repr_dir = data_dir / "event_representations_v2" / "stacked_histogram_dt=50_nbins=10"
            repr_dir.mkdir(parents=True)

            # Create mock HDF5 file
            h5_file = repr_dir / "event_representations_ds2_nearest.h5"
            with h5py.File(h5_file, "w") as f:
                # Shape: (num_frames, num_bins, height, width)
                data = np.random.randint(0, 10, size=(100, 20, 360, 640), dtype=np.uint8)
                f.create_dataset("data", data=data)

            # Create timestamps
            timestamps = np.arange(100) * 50000  # 50ms intervals in microseconds
            np.save(repr_dir / "timestamps_us.npy", timestamps)

            yield data_dir

    def test_find_h5_file_success(self, mock_etram_data):
        """Test successful H5 file finding."""
        loader = viz.eTramDataLoader(mock_etram_data)
        assert loader.h5_file_path is not None
        assert loader.h5_file_path.exists()
        assert loader.h5_file_path.name == "event_representations_ds2_nearest.h5"

    def test_find_h5_file_direct_path(self):
        """Test loading with direct H5 file path."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            # Create a simple HDF5 file
            with h5py.File(tmp.name, "w") as f:
                data = np.random.randint(0, 10, size=(10, 20, 360, 640), dtype=np.uint8)
                f.create_dataset("data", data=data)

            try:
                loader = viz.eTramDataLoader(tmp.name)
                assert loader.h5_file_path == Path(tmp.name)
            finally:
                os.unlink(tmp.name)

    def test_load_metadata(self, mock_etram_data):
        """Test metadata loading."""
        loader = viz.eTramDataLoader(mock_etram_data)

        assert loader.num_frames == 100
        assert loader.num_bins == 20
        assert loader.height == 360
        assert loader.width == 640
        assert loader.dtype == np.uint8

        # Test timestamps
        assert len(loader.timestamps_us) == 100
        assert loader.start_time_s == 0.0
        assert loader.end_time_s == 99 * 0.05  # 99 * 50ms
        assert abs(loader.duration_s - 4.95) < 0.01

    def test_get_frame_data(self, mock_etram_data):
        """Test frame data retrieval."""
        loader = viz.eTramDataLoader(mock_etram_data)

        # Test valid frame
        frame_data = loader.get_frame_data(0)
        assert frame_data.shape == (20, 360, 640)
        assert frame_data.dtype == np.uint8

        # Test frame bounds
        with pytest.raises(ValueError, match="Frame index.*out of range"):
            loader.get_frame_data(-1)

        with pytest.raises(ValueError, match="Frame index.*out of range"):
            loader.get_frame_data(100)

    def test_get_frame_range(self, mock_etram_data):
        """Test frame range retrieval."""
        loader = viz.eTramDataLoader(mock_etram_data)

        # Test valid range
        frame_data = loader.get_frame_range(0, 10)
        assert frame_data.shape == (10, 20, 360, 640)

        # Test invalid ranges
        with pytest.raises(ValueError, match="Invalid frame range"):
            loader.get_frame_range(-1, 10)

        with pytest.raises(ValueError, match="Invalid frame range"):
            loader.get_frame_range(10, 5)  # start > end

    def test_missing_h5_file(self):
        """Test handling of missing H5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No HDF5 file found"):
                viz.eTramDataLoader(tmpdir)


class TestEventFrameRenderer:
    """Test event frame renderer functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return viz.VisualizationConfig(width=100, height=100, fps=30.0, decay_ms=100.0)

    @pytest.fixture
    def renderer(self, config):
        """Create renderer instance."""
        return viz.EventFrameRenderer(config)

    def test_initialization(self, renderer, config):
        """Test renderer initialization."""
        assert renderer.config == config
        assert renderer.decay_buffer.shape == (100, 100, 3)
        assert renderer.frame_count == 0

    def test_render_frame_basic(self, renderer):
        """Test basic frame rendering."""
        # Create mock event data with positive and negative events
        event_data = np.zeros((20, 100, 100), dtype=np.uint8)
        event_data[0, 10, 20] = 100  # Positive event (even bin)
        event_data[1, 30, 40] = 80  # Negative event (odd bin)

        frame = renderer.render_frame(event_data, timestamp_s=1.0)

        assert frame.shape == (100, 100, 3)
        assert frame.dtype == np.uint8
        assert renderer.frame_count == 1

        # Check that positive event created red pixel (BGR format)
        assert frame[10, 20, 2] > 0  # Red channel
        assert frame[10, 20, 0] == 0  # Blue channel
        assert frame[10, 20, 1] == 0  # Green channel

        # Check that negative event created blue pixel
        assert frame[30, 40, 0] > 0  # Blue channel
        assert frame[30, 40, 2] == 0  # Red channel

    def test_render_frame_with_stats(self, renderer):
        """Test frame rendering with statistics overlay."""
        event_data = np.zeros((20, 100, 100), dtype=np.uint8)
        stats = {"fps": 30.0, "events_per_sec": 1000, "total_events": 50}

        frame = renderer.render_frame(event_data, timestamp_s=1.0, show_stats=stats)

        # Should have stats overlay (we can't easily test text, but frame should be modified)
        assert frame.shape == (100, 100, 3)
        assert frame.dtype == np.uint8

    def test_temporal_decay(self, renderer):
        """Test temporal decay functionality."""
        event_data1 = np.zeros((20, 100, 100), dtype=np.uint8)
        event_data1[0, 50, 50] = 255  # Strong positive event

        # First frame
        frame1 = renderer.render_frame(event_data1, timestamp_s=0.0)
        initial_intensity = frame1[50, 50, 2]  # Red channel
        assert initial_intensity == 255

        # Second frame with no events (should show decay)
        event_data2 = np.zeros((20, 100, 100), dtype=np.uint8)
        frame2 = renderer.render_frame(event_data2, timestamp_s=0.033)  # ~33ms later
        decayed_intensity = frame2[50, 50, 2]

        # Should be less than initial due to decay
        assert decayed_intensity < initial_intensity
        assert decayed_intensity > 0  # But not zero yet

    def test_reset(self, renderer):
        """Test renderer reset functionality."""
        # Render a frame to modify state
        event_data = np.zeros((20, 100, 100), dtype=np.uint8)
        event_data[0, 10, 10] = 100
        renderer.render_frame(event_data)

        assert renderer.frame_count == 1
        assert np.any(renderer.decay_buffer > 0)

        # Reset
        renderer.reset()

        assert renderer.frame_count == 0
        assert np.all(renderer.decay_buffer == 0)


class TesteTramVisualizer:
    """Test main eTram visualizer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return viz.VisualizationConfig(width=100, height=100, fps=10.0)  # Low FPS for faster tests

    @pytest.fixture
    def visualizer(self, config):
        """Create visualizer instance."""
        return viz.eTramVisualizer(config)

    @pytest.fixture
    def mock_etram_data(self):
        """Create mock eTram data for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "test_data"
            data_dir.mkdir()

            # Create the expected directory structure
            repr_dir = data_dir / "event_representations_v2" / "stacked_histogram_dt=50_nbins=10"
            repr_dir.mkdir(parents=True)

            # Create small mock HDF5 file for fast testing
            h5_file = repr_dir / "event_representations_ds2_nearest.h5"
            with h5py.File(h5_file, "w") as f:
                # Small test data
                data = np.random.randint(0, 5, size=(10, 20, 100, 100), dtype=np.uint8)
                f.create_dataset("data", data=data)

            # Create timestamps (10 frames at 100ms intervals)
            timestamps = np.arange(10) * 100000  # 100ms intervals in microseconds
            np.save(repr_dir / "timestamps_us.npy", timestamps)

            yield data_dir

    def test_initialization(self, visualizer, config):
        """Test visualizer initialization."""
        assert visualizer.config == config
        assert isinstance(visualizer.renderer, viz.EventFrameRenderer)

    @patch("cv2.VideoWriter")
    def test_process_file_success(self, mock_video_writer, visualizer, mock_etram_data):
        """Test successful file processing."""
        # Mock video writer
        mock_writer = Mock()
        mock_writer.isOpened.return_value = True
        mock_video_writer.return_value = mock_writer

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
            try:
                result = visualizer.process_file(
                    mock_etram_data, output_file.name, duration_s=0.5  # Process only 0.5 seconds
                )

                assert result is True
                mock_video_writer.assert_called_once()
                mock_writer.write.assert_called()
                mock_writer.release.assert_called_once()

            finally:
                if os.path.exists(output_file.name):
                    os.unlink(output_file.name)

    @patch("cv2.VideoWriter")
    def test_process_file_video_writer_failure(self, mock_video_writer, visualizer, mock_etram_data):
        """Test handling of video writer failure."""
        # Mock failed video writer
        mock_writer = Mock()
        mock_writer.isOpened.return_value = False
        mock_video_writer.return_value = mock_writer

        with tempfile.NamedTemporaryFile(suffix=".mp4") as output_file:
            result = visualizer.process_file(mock_etram_data, output_file.name)
            assert result is False

    def test_process_file_invalid_data_path(self, visualizer):
        """Test processing with invalid data path."""
        result = visualizer.process_file("/nonexistent/path", "output.mp4")
        assert result is False

    def test_process_directory(self, visualizer):
        """Test directory processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test data directories
            base_dir = Path(tmpdir)
            output_dir = base_dir / "outputs"

            # Create mock data directories (without actual H5 files for this test)
            for i in range(3):
                data_dir = base_dir / f"test_day_{i:03d}"
                repr_dir = data_dir / "event_representations_v2"
                repr_dir.mkdir(parents=True)

            # This will fail due to missing H5 files, but should handle gracefully
            successful_outputs = visualizer.process_directory(
                base_dir, output_dir, pattern="*/event_representations_v2"
            )

            # Should return empty list due to missing H5 files
            assert isinstance(successful_outputs, list)
            assert len(successful_outputs) == 0


class TestVideoFrame:
    """Test video frame utilities if available."""

    def test_video_frame_creation(self):
        """Test video frame creation and validation."""
        # This would test the Rust video writer functionality
        # For now, we'll just ensure the Python layer doesn't crash
        pass


class TestIntegration:
    """Integration tests with real-like data."""

    def test_end_to_end_visualization(self):
        """Test complete visualization pipeline."""
        # Create a minimal but realistic data structure
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "integration_test"
            data_dir.mkdir()

            # Create expected structure
            repr_dir = data_dir / "event_representations_v2" / "stacked_histogram_dt=50_nbins=10"
            repr_dir.mkdir(parents=True)

            # Create realistic test data with actual event patterns
            h5_file = repr_dir / "event_representations_ds2_nearest.h5"
            with h5py.File(h5_file, "w") as f:
                # Create data with some spatial structure
                num_frames, num_bins, height, width = 20, 20, 50, 50
                data = np.zeros((num_frames, num_bins, height, width), dtype=np.uint8)

                # Add some moving patterns
                for frame_idx in range(num_frames):
                    # Moving positive events (even bins)
                    x = int(10 + frame_idx * 1.5) % width
                    y = int(10 + frame_idx * 0.5) % height
                    data[frame_idx, 0::2, y : y + 5, x : x + 5] = 50 + frame_idx * 2

                    # Moving negative events (odd bins)
                    x = int(width - 10 - frame_idx * 1.5) % width
                    y = int(height - 10 - frame_idx * 0.5) % height
                    data[frame_idx, 1::2, y : y + 3, x : x + 3] = 30 + frame_idx

                f.create_dataset("data", data=data)

            # Create timestamps
            timestamps = np.arange(num_frames) * 50000  # 50ms intervals
            np.save(repr_dir / "timestamps_us.npy", timestamps)

            # Test visualization
            config = viz.VisualizationConfig(width=width, height=height, fps=20.0, decay_ms=80.0)

            _visualizer = viz.eTramVisualizer(config)

            # Test data loading
            loader = viz.eTramDataLoader(data_dir)
            assert loader.num_frames == num_frames
            assert loader.width == width
            assert loader.height == height

            # Test frame rendering
            renderer = viz.EventFrameRenderer(config)

            for frame_idx in range(min(5, num_frames)):  # Test first 5 frames
                event_data = loader.get_frame_data(frame_idx)
                frame = renderer.render_frame(event_data, timestamp_s=frame_idx * 0.05)

                assert frame.shape == (height, width, 3)
                assert frame.dtype == np.uint8

                # Should have some non-zero pixels due to our test pattern
                assert np.sum(frame) > 0

                # Should have both positive (red) and negative (blue) events
                assert np.any(frame[:, :, 2] > 0)  # Red channel
                assert np.any(frame[:, :, 0] > 0)  # Blue channel


# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce log noise during testing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
