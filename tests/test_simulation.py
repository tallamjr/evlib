"""Tests for the evlib.simulation module."""

import pytest
import numpy as np
import tempfile
import os


# Test configuration classes
def test_esim_config():
    """Test ESIMConfig class."""
    from evlib.simulation.config import ESIMConfig

    # Test default config
    config = ESIMConfig()
    assert config.positive_threshold == 0.4
    assert config.negative_threshold == 0.4
    assert config.refractory_period_ms == 0.1
    assert config.device == "auto"

    # Test custom config
    config = ESIMConfig(positive_threshold=0.3, negative_threshold=0.5, device="cpu")
    assert config.positive_threshold == 0.3
    assert config.negative_threshold == 0.5
    assert config.device == "cpu"

    # Test validation
    with pytest.raises(ValueError, match="positive_threshold must be positive"):
        ESIMConfig(positive_threshold=-0.1)

    with pytest.raises(ValueError, match="negative_threshold must be positive"):
        ESIMConfig(negative_threshold=0.0)

    # Test from_dict
    config_dict = {
        "positive_threshold": 0.6,
        "negative_threshold": 0.7,
        "device": "cuda",
    }
    config = ESIMConfig.from_dict(config_dict)
    assert config.positive_threshold == 0.6
    assert config.device == "cuda"


def test_video_config():
    """Test VideoConfig class."""
    from evlib.simulation.config import VideoConfig

    # Test default config
    config = VideoConfig()
    assert config.width == 640
    assert config.height == 480
    assert config.fps is None
    assert config.grayscale is True

    # Test custom config
    config = VideoConfig(width=1280, height=720, fps=30.0)
    assert config.width == 1280
    assert config.height == 720
    assert config.fps == 30.0

    # Test validation
    with pytest.raises(ValueError, match="width must be positive"):
        VideoConfig(width=-1)

    with pytest.raises(ValueError, match="start_time must be less than end_time"):
        VideoConfig(start_time=10.0, end_time=5.0)


def test_predefined_configs():
    """Test predefined configuration sets."""
    from evlib.simulation.config import get_esim_config, get_video_config

    # Test ESIM configs
    default_config = get_esim_config("default")
    assert default_config.positive_threshold == 0.4

    sensitive_config = get_esim_config("high_sensitivity")
    assert sensitive_config.positive_threshold == 0.2
    assert sensitive_config.negative_threshold == 0.2

    # Test video configs
    hd_config = get_video_config("hd")
    assert hd_config.width == 1280
    assert hd_config.height == 720

    # Test invalid config name
    with pytest.raises(ValueError, match="Unknown ESIM config"):
        get_esim_config("invalid_config")


def test_dependency_info():
    """Test dependency information function."""
    from evlib.simulation import get_dependency_info

    info = get_dependency_info()
    assert isinstance(info, dict)
    assert "torch" in info
    assert "opencv" in info
    assert "missing_message" in info

    # The values depend on what's installed, so just check types
    assert isinstance(info["torch"], bool)
    assert isinstance(info["opencv"], bool)


# Test ESIM simulator (requires PyTorch)
@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available"),
    reason="PyTorch required for ESIM simulator tests",
)
class TestESIMSimulator:
    """Tests for ESIMSimulator class."""

    def test_simulator_creation(self):
        """Test creating an ESIM simulator."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(device="cpu")  # Force CPU for testing
        simulator = ESIMSimulator(config)

        assert simulator.config == config
        assert not simulator.is_initialized
        assert simulator.device.type == "cpu"

    def test_simulator_reset(self):
        """Test simulator reset functionality."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(device="cpu")
        simulator = ESIMSimulator(config)

        # Reset should work even when not initialized
        simulator.reset()
        assert not simulator.is_initialized

    def test_process_single_frame(self):
        """Test processing a single frame."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(
            device="cpu", positive_threshold=0.1, negative_threshold=0.1
        )
        simulator = ESIMSimulator(config)

        # Create a simple test frame
        frame1 = np.ones((100, 100), dtype=np.uint8) * 128  # Gray frame
        frame2 = np.ones((100, 100), dtype=np.uint8) * 200  # Brighter frame

        # First frame should not generate events (initialization)
        events1 = simulator.process_frame(frame1, 0.0)
        assert len(events1[0]) == 0  # No events from first frame
        assert simulator.is_initialized

        # Second frame should generate events
        events2 = simulator.process_frame(frame2, 0.1)
        x, y, t, p = events2

        # Check that events were generated
        assert len(x) > 0
        assert len(y) == len(x)
        assert len(t) == len(x)
        assert len(p) == len(x)

        # Check data types
        assert x.dtype == np.int64
        assert y.dtype == np.int64
        assert t.dtype == np.float64
        assert p.dtype == np.int64

        # Check value ranges
        assert np.all(x >= 0) and np.all(x < 100)
        assert np.all(y >= 0) and np.all(y < 100)
        assert np.all(t == 0.1)  # All events should have same timestamp
        assert np.all(p == 1)  # All should be positive (brightness increased)

    def test_process_rgb_frame(self):
        """Test processing RGB frames (should convert to grayscale)."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(device="cpu")
        simulator = ESIMSimulator(config)

        # Create RGB frame
        rgb_frame = np.ones((50, 50, 3), dtype=np.uint8) * 128

        # Should work without errors
        events = simulator.process_frame(rgb_frame, 0.0)
        assert len(events[0]) == 0  # First frame, no events expected
        assert simulator.is_initialized

    def test_get_state_info(self):
        """Test getting simulator state information."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(device="cpu")
        simulator = ESIMSimulator(config)

        # Before initialization
        state = simulator.get_state_info()
        assert not state["initialized"]

        # After initialization
        frame = np.ones((50, 50), dtype=np.uint8) * 128
        simulator.process_frame(frame, 0.0)

        state = simulator.get_state_info()
        assert state["initialized"]
        assert "device" in state
        assert "shape" in state
        assert "buffer_stats" in state

    def test_mps_device_support(self):
        """Test MPS device support and dtype compatibility."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator
        import torch

        # Test MPS device selection with auto
        config = ESIMConfig(device="auto")
        simulator = ESIMSimulator(config)

        # Should be MPS if available, otherwise CPU/CUDA
        expected_devices = {"cpu", "cuda", "mps"}
        assert simulator.device.type in expected_devices

        # Test explicit MPS device request
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            config_mps = ESIMConfig(
                device="mps", dtype="float64"
            )  # This should auto-convert
            simulator_mps = ESIMSimulator(config_mps)

            # Should use MPS device
            assert simulator_mps.device.type == "mps"
            # Should auto-convert to float32 for MPS compatibility
            assert simulator_mps._dtype == torch.float32

    def test_device_dtype_compatibility(self):
        """Test device and dtype compatibility handling."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator
        import torch

        # Test CPU with float64 (should work)
        config = ESIMConfig(device="cpu", dtype="float64")
        simulator = ESIMSimulator(config)
        assert simulator._dtype == torch.float64

        # Test CPU with float32 (should work)
        config = ESIMConfig(device="cpu", dtype="float32")
        simulator = ESIMSimulator(config)
        assert simulator._dtype == torch.float32


# Test video processor (requires OpenCV and PyTorch)
@pytest.mark.skipif(
    not pytest.importorskip("cv2", reason="OpenCV not available"),
    reason="OpenCV required for video processor tests",
)
@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available"),
    reason="PyTorch required for video processor tests",
)
class TestVideoProcessor:
    """Tests for VideoToEvents class."""

    def test_video_processor_creation(self):
        """Test creating a video processor."""
        from evlib.simulation.config import ESIMConfig, VideoConfig
        from evlib.simulation.video_processor import VideoToEvents

        esim_config = ESIMConfig(device="cpu")
        video_config = VideoConfig()

        processor = VideoToEvents(esim_config, video_config)
        assert processor.esim_config == esim_config
        assert processor.video_config == video_config

    def test_nonexistent_video_file(self):
        """Test handling of non-existent video file."""
        from evlib.simulation.config import ESIMConfig, VideoConfig
        from evlib.simulation.video_processor import VideoToEvents

        esim_config = ESIMConfig(device="cpu")
        video_config = VideoConfig()
        processor = VideoToEvents(esim_config, video_config)

        with pytest.raises(FileNotFoundError):
            processor.process_video("nonexistent_video.mp4")


def test_convenience_functions():
    """Test convenience functions."""
    # Test with missing dependencies - should not crash
    try:
        from evlib.simulation import video_to_events, create_esim_simulator

        # These functions exist but may not work without dependencies
        assert callable(video_to_events)
        assert callable(create_esim_simulator)
    except ImportError:
        # Functions may not be available without dependencies
        pass


def test_simple_video_to_events():
    """Test simple video to events function."""
    try:
        from evlib.simulation.video_processor import video_to_events_simple

        assert callable(video_to_events_simple)
    except ImportError:
        # Function may not be available without dependencies
        pass


# Integration tests
@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available"),
    reason="Integration tests require PyTorch",
)
@pytest.mark.skipif(
    not pytest.importorskip("cv2", reason="OpenCV not available"),
    reason="Integration tests require OpenCV",
)
class TestSimulationIntegration:
    """Integration tests for the simulation module."""

    def test_full_pipeline_with_synthetic_video(self):
        """Test full pipeline with a synthetic video."""
        import cv2
        from evlib.simulation.config import ESIMConfig, VideoConfig
        from evlib.simulation.video_processor import VideoToEvents

        # Create a synthetic video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            # Create a simple synthetic video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (64, 64))

            # Create frames with changing intensity
            for i in range(10):
                frame = np.ones((64, 64, 3), dtype=np.uint8) * (100 + i * 10)
                out.write(frame)

            out.release()

            # Test processing
            esim_config = ESIMConfig(
                device="cpu", positive_threshold=0.1, negative_threshold=0.1
            )
            video_config = VideoConfig(width=64, height=64)

            processor = VideoToEvents(esim_config, video_config)

            # Get video info
            info = processor.get_video_info(video_path)
            assert info["width"] == 64
            assert info["height"] == 64

            # Process video
            x, y, t, p = processor.process_video(video_path)

            # Should generate some events due to changing intensity
            assert len(x) > 0
            assert len(y) == len(x)
            assert len(t) == len(x)
            assert len(p) == len(x)

        finally:
            # Clean up
            if os.path.exists(video_path):
                os.unlink(video_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
