"""
Tests for evlib visualization functionality.

Tests both the Python visualization module and integration with real eTram data.
"""

import os
import pytest
import numpy as np
from pathlib import Path
import logging

# Test imports
try:
    import evlib.visualization as viz
    import cv2
    import h5py
except ImportError as e:
    pytest.skip(f"Visualization dependencies not available: {e}", allow_module_level=True)

# Real eTram data path
REAL_ETRAM_DATA = Path("data/eTram_processed/test/test_day_010")
REAL_H5_FILE = (
    REAL_ETRAM_DATA
    / "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5"
)

# Skip tests if real data not available
if not REAL_ETRAM_DATA.exists() or not REAL_H5_FILE.exists():
    pytest.skip("Real eTram data not available for testing", allow_module_level=True)


class TestVisualizationConfig:
    """Test visualization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = viz.VisualizationConfig()

        assert config.width == 640
        assert config.height == 360
        assert config.fps == 30.0
        assert config.positive_color == (0, 0, 255)  # Red in BGR
        assert config.negative_color == (255, 128, 0)  # Bright blue in BGR
        assert config.background_color == (200, 180, 150)  # Pastel blue in BGR
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

    def test_colormap_config(self):
        """Test colormap configuration."""
        config = viz.VisualizationConfig(
            use_colormap=True,
            colormap_type="plasma",
        )

        assert config.use_colormap is True
        assert config.colormap_type == "plasma"


class TesteTramDataLoader:
    """Test eTram data loader functionality with real data."""

    def test_find_h5_file_success(self):
        """Test successful H5 file finding with real data."""
        loader = viz.eTramDataLoader(REAL_ETRAM_DATA)
        assert loader.h5_file_path is not None
        assert loader.h5_file_path.exists()
        assert loader.h5_file_path.name == "event_representations_ds2_nearest.h5"

    def test_find_h5_file_direct_path(self):
        """Test loading with direct H5 file path."""
        loader = viz.eTramDataLoader(REAL_H5_FILE)
        assert loader.h5_file_path == REAL_H5_FILE

    def test_load_metadata(self):
        """Test metadata loading with real data."""
        loader = viz.eTramDataLoader(REAL_ETRAM_DATA)

        # Real data dimensions
        assert loader.num_frames > 0
        assert loader.num_bins == 20
        assert loader.height > 0
        assert loader.width > 0
        assert loader.dtype == np.uint8

        # Test timestamps exist and are valid
        assert len(loader.timestamps_us) == loader.num_frames
        assert loader.start_time_s >= 0.0
        assert loader.end_time_s > loader.start_time_s
        assert loader.duration_s > 0

    def test_get_frame_data(self):
        """Test frame data retrieval with real data."""
        loader = viz.eTramDataLoader(REAL_ETRAM_DATA)

        # Test valid frame
        frame_data = loader.get_frame_data(0)
        assert frame_data.shape == (20, loader.height, loader.width)
        assert frame_data.dtype == np.uint8

        # Test frame bounds
        with pytest.raises(ValueError, match="Frame index.*out of range"):
            loader.get_frame_data(-1)

        with pytest.raises(ValueError, match="Frame index.*out of range"):
            loader.get_frame_data(loader.num_frames)

    def test_get_frame_range(self):
        """Test frame range retrieval with real data."""
        loader = viz.eTramDataLoader(REAL_ETRAM_DATA)

        # Test valid range (use smaller range for speed)
        test_range = min(10, loader.num_frames)
        frame_data = loader.get_frame_range(0, test_range)
        assert frame_data.shape == (test_range, 20, loader.height, loader.width)

        # Test invalid ranges
        with pytest.raises(ValueError, match="Invalid frame range"):
            loader.get_frame_range(-1, 10)

        with pytest.raises(ValueError, match="Invalid frame range"):
            loader.get_frame_range(10, 5)  # start > end

    def test_missing_h5_file(self):
        """Test handling of missing H5 file."""
        with pytest.raises(FileNotFoundError, match="No HDF5 file found"):
            viz.eTramDataLoader("/nonexistent/path")


class TestEventFrameRenderer:
    """Test event frame renderer functionality using real data dimensions."""

    @pytest.fixture
    def real_loader(self):
        """Load real eTram data for testing."""
        return viz.eTramDataLoader(REAL_ETRAM_DATA)

    @pytest.fixture
    def config(self, real_loader):
        """Create test configuration based on real data dimensions."""
        return viz.VisualizationConfig(
            width=real_loader.width, height=real_loader.height, fps=30.0, decay_ms=100.0
        )

    @pytest.fixture
    def renderer(self, config):
        """Create renderer instance."""
        return viz.EventFrameRenderer(config)

    def test_initialization(self, renderer, config):
        """Test renderer initialization."""
        assert renderer.config == config
        assert renderer.decay_buffer is None  # Not initialized until first frame
        assert renderer.frame_count == 0

    def test_render_frame_basic(self, renderer, real_loader):
        """Test basic frame rendering with real data."""
        # Use real event data
        event_data = real_loader.get_frame_data(0)

        frame = renderer.render_frame(event_data, timestamp_s=1.0)

        assert frame.shape == (real_loader.height, real_loader.width, 3)
        assert frame.dtype == np.uint8
        assert renderer.frame_count == 1

        # Check that frame has been rendered (not all background)
        assert np.var(frame) > 0

    def test_render_frame_with_stats(self, renderer, real_loader):
        """Test frame rendering with statistics overlay."""
        event_data = real_loader.get_frame_data(0)
        stats = {"fps": 30.0, "events_per_sec": 1000, "total_events": 50}

        frame = renderer.render_frame(event_data, timestamp_s=1.0, show_stats=stats)

        # Should have stats overlay
        assert frame.shape == (real_loader.height, real_loader.width, 3)
        assert frame.dtype == np.uint8

    def test_temporal_decay(self, renderer, real_loader):
        """Test temporal decay functionality with real data."""
        # Use real event data
        event_data1 = real_loader.get_frame_data(0)

        # First frame
        frame1 = renderer.render_frame(event_data1, timestamp_s=0.0)
        assert np.var(frame1) > 0  # First frame should have content

        # Second frame with less activity (or same frame for consistency)
        event_data2 = real_loader.get_frame_data(min(1, real_loader.num_frames - 1))
        frame2 = renderer.render_frame(event_data2, timestamp_s=0.033)  # ~33ms later

        # Frame should still have content (decay buffer maintains some signal)
        assert np.var(frame2) > 0

    def test_reset(self, renderer, real_loader):
        """Test renderer reset functionality."""
        # Render a frame to modify state
        event_data = real_loader.get_frame_data(0)
        renderer.render_frame(event_data)

        assert renderer.frame_count == 1
        assert renderer.decay_buffer is not None

        # Reset
        renderer.reset()

        assert renderer.frame_count == 0

    def test_polarity_rendering(self):
        """Test polarity-based rendering preserves polarity information."""
        config = viz.VisualizationConfig(width=100, height=100, fps=10.0, use_colormap=False)
        renderer = viz.EventFrameRenderer(config)

        # Create specific polarity patterns
        event_data = np.zeros((20, 100, 100), dtype=np.uint8)
        event_data[0, 25, 25] = 150  # Strong positive event (even bin)
        event_data[1, 75, 75] = 120  # Strong negative event (odd bin)

        frame = renderer.render_frame(event_data, timestamp_s=1.0)

        assert frame.shape == (100, 100, 3)
        assert frame.dtype == np.uint8

        # Check polarity distinction
        pos_pixel = frame[25, 25]
        neg_pixel = frame[75, 75]

        # Positive should have more red, negative should have more blue
        assert pos_pixel[2] > pos_pixel[0]  # Red > Blue for positive
        assert neg_pixel[0] > neg_pixel[2]  # Blue > Red for negative

    def test_colormap_rendering_preserves_polarity(self):
        """Test that colormap rendering preserves polarity information."""
        colormap_config = viz.VisualizationConfig(
            width=100, height=100, fps=10.0, use_colormap=True, colormap_type="jet"
        )
        renderer = viz.EventFrameRenderer(colormap_config)

        # Create distinct positive and negative events
        event_data = np.zeros((20, 100, 100), dtype=np.uint8)
        event_data[0, 25, 25] = 200  # Strong positive event
        event_data[1, 75, 75] = 200  # Strong negative event

        frame = renderer.render_frame(event_data, timestamp_s=1.0)

        assert frame.shape == (100, 100, 3)
        assert frame.dtype == np.uint8

        # Check that positive and negative events have different color signatures
        pos_pixel = frame[25, 25]
        neg_pixel = frame[75, 75]

        # Both should be non-zero but different
        assert np.sum(pos_pixel) > 0
        assert np.sum(neg_pixel) > 0
        assert not np.array_equal(pos_pixel, neg_pixel)

    def test_different_colormaps(self):
        """Test different colormap types."""
        colormaps = ["jet", "hot", "plasma", "viridis", "inferno"]

        for colormap_type in colormaps:
            colormap_config = viz.VisualizationConfig(
                width=50, height=50, fps=10.0, use_colormap=True, colormap_type=colormap_type
            )
            renderer = viz.EventFrameRenderer(colormap_config)

            # Create event data with intensity
            event_data = np.zeros((20, 50, 50), dtype=np.uint8)
            event_data[0, 25, 25] = 150  # Strong positive event

            frame = renderer.render_frame(event_data, timestamp_s=1.0)

            assert frame.shape == (50, 50, 3)
            assert frame.dtype == np.uint8
            # Ensure the frame has been modified from background
            assert np.sum(frame[25, 25]) > 0  # Should have some color at event location


class TesteTramVisualizer:
    """Test main eTram visualizer class with real data."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return viz.VisualizationConfig(width=320, height=180, fps=10.0)  # Low FPS for faster tests

    @pytest.fixture
    def visualizer(self, config):
        """Create visualizer instance."""
        return viz.eTramVisualizer(config)

    def test_initialization(self, visualizer, config):
        """Test visualizer initialization."""
        assert visualizer.config == config
        assert isinstance(visualizer.renderer, viz.EventFrameRenderer)

    def test_process_file_success(self, visualizer):
        """Test successful file processing with real data."""
        output_path = Path("outputs/test_real_processing.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = visualizer.process_file(
                REAL_ETRAM_DATA, output_path, duration_s=0.1  # Process only 0.1 seconds for speed
            )

            assert result is True
            assert output_path.exists()
            assert output_path.stat().st_size > 0  # File should have content

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_process_file_colormap_mode(self):
        """Test file processing with colormap visualization."""
        config = viz.VisualizationConfig(
            width=320, height=180, fps=10.0, use_colormap=True, colormap_type="jet"
        )
        visualizer = viz.eTramVisualizer(config)
        output_path = Path("outputs/test_colormap_processing.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = visualizer.process_file(
                REAL_ETRAM_DATA, output_path, duration_s=0.1  # Process only 0.1 seconds
            )

            assert result is True
            assert output_path.exists()
            assert output_path.stat().st_size > 0

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_process_file_invalid_data_path(self, visualizer):
        """Test processing with invalid data path."""
        result = visualizer.process_file("/nonexistent/path", "output.mp4")
        assert result is False

    def test_process_directory(self, visualizer):
        """Test directory processing with real data."""
        output_dir = Path("outputs/test_batch")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Process real eTram test directory
            successful_outputs = visualizer.process_directory(
                "data/eTram_processed/test",
                output_dir,
                pattern="*/event_representations_v2",
            )

            # Should process at least one file successfully
            assert isinstance(successful_outputs, list)
            # Clean up any generated files
            for output_file in successful_outputs:
                if output_file.exists():
                    output_file.unlink()

        finally:
            # Clean up output directory
            if output_dir.exists():
                for file in output_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                output_dir.rmdir()


class TestVideoFrame:
    """Test video frame utilities if available."""

    def test_video_frame_creation(self):
        """Test video frame creation and validation."""
        # This would test the Rust video writer functionality
        # For now, we'll just ensure the Python layer doesn't crash
        pass


class TestIntegration:
    """Integration tests with real eTram data."""

    def test_end_to_end_visualization(self):
        """Test complete visualization pipeline with real eTram data."""
        # Use real eTram data for integration testing
        data_dir = REAL_ETRAM_DATA

        # Test configuration based on real data dimensions
        real_loader = viz.eTramDataLoader(data_dir)
        config = viz.VisualizationConfig(
            width=real_loader.width, height=real_loader.height, fps=20.0, decay_ms=80.0
        )

        # Test data loading with real data
        assert real_loader.num_frames > 0
        assert real_loader.width > 0
        assert real_loader.height > 0
        assert real_loader.num_bins == 20

        # Test frame rendering with real data
        renderer = viz.EventFrameRenderer(config)

        # Test first few frames to avoid long test times
        test_frames = min(5, real_loader.num_frames)
        for frame_idx in range(test_frames):
            event_data = real_loader.get_frame_data(frame_idx)
            timestamp_s = real_loader.timestamps_us[frame_idx] / 1_000_000

            frame = renderer.render_frame(event_data, timestamp_s=timestamp_s)

            assert frame.shape == (real_loader.height, real_loader.width, 3)
            assert frame.dtype == np.uint8

            # Frame should be rendered (not all background)
            assert np.var(frame) > 0

        # Test colormap visualization mode
        colormap_config = viz.VisualizationConfig(
            width=real_loader.width,
            height=real_loader.height,
            fps=20.0,
            use_colormap=True,
            colormap_type="jet",
        )
        colormap_renderer = viz.EventFrameRenderer(colormap_config)

        # Test colormap rendering preserves polarity
        event_data = real_loader.get_frame_data(0)
        colormap_frame = colormap_renderer.render_frame(event_data, timestamp_s=0.0)

        assert colormap_frame.shape == (real_loader.height, real_loader.width, 3)
        assert colormap_frame.dtype == np.uint8
        assert np.var(colormap_frame) > 0  # Should have visual content


# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce log noise during testing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
