"""Tests for the evlib.simulation module."""

import os
import tempfile

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# Test configuration classes
def test_esim_config():
    """Test ESIMConfig class."""
    from evlib.simulation.config import ESIMConfig

    config = ESIMConfig()
    assert config.positive_threshold == 0.4
    assert config.negative_threshold == 0.4
    assert config.refractory_period_ms == 0.1
    assert config.device == "auto"

    config = ESIMConfig(positive_threshold=0.3, negative_threshold=0.5, device="cpu")
    assert config.positive_threshold == 0.3
    assert config.negative_threshold == 0.5
    assert config.device == "cpu"

    with pytest.raises(ValueError, match="positive_threshold must be positive"):
        ESIMConfig(positive_threshold=-0.1)

    with pytest.raises(ValueError, match="negative_threshold must be positive"):
        ESIMConfig(negative_threshold=0.0)

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

    config = VideoConfig()
    assert config.width == 640
    assert config.height == 480
    assert config.fps is None
    assert config.grayscale is True

    config = VideoConfig(width=1280, height=720, fps=30.0)
    assert config.width == 1280
    assert config.height == 720
    assert config.fps == 30.0

    with pytest.raises(ValueError, match="width must be positive"):
        VideoConfig(width=-1)

    with pytest.raises(ValueError, match="start_time must be less than end_time"):
        VideoConfig(start_time=10.0, end_time=5.0)


def test_predefined_configs():
    """Test predefined configuration sets."""
    from evlib.simulation.config import get_esim_config, get_video_config

    default_config = get_esim_config("default")
    assert default_config.positive_threshold == 0.4

    sensitive_config = get_esim_config("high_sensitivity")
    assert sensitive_config.positive_threshold == 0.2
    assert sensitive_config.negative_threshold == 0.2

    hd_config = get_video_config("hd")
    assert hd_config.width == 1280
    assert hd_config.height == 720

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

    assert isinstance(info["torch"], bool)
    assert isinstance(info["opencv"], bool)


@pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch required for ESIM simulator tests"
)
class TestESIMSimulator:
    """Tests for ESIMSimulator class."""

    def test_simulator_creation(self):
        """Test creating an ESIM simulator."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(device="cpu")
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

        frame1 = np.ones((100, 100), dtype=np.uint8) * 128
        frame2 = np.ones((100, 100), dtype=np.uint8) * 200

        events1 = simulator.process_frame(frame1, 0.0)
        assert len(events1[0]) == 0
        assert simulator.is_initialized

        events2 = simulator.process_frame(frame2, 0.1)
        x, y, t, p = events2

        assert len(x) > 0
        assert len(y) == len(x)
        assert len(t) == len(x)
        assert len(p) == len(x)

        assert x.dtype == np.int64
        assert y.dtype == np.int64
        assert t.dtype == np.float64
        assert p.dtype == np.int64

        assert np.all(x >= 0) and np.all(x < 100)
        assert np.all(y >= 0) and np.all(y < 100)
        assert np.all(t == 0.1)
        assert np.all(p == 1)

    def test_process_rgb_frame(self):
        """Test processing RGB frames (should convert to grayscale)."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(device="cpu")
        simulator = ESIMSimulator(config)

        rgb_frame = np.ones((50, 50, 3), dtype=np.uint8) * 128

        events = simulator.process_frame(rgb_frame, 0.0)
        assert len(events[0]) == 0
        assert simulator.is_initialized

    def test_get_state_info(self):
        """Test getting simulator state information."""
        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(device="cpu")
        simulator = ESIMSimulator(config)

        state = simulator.get_state_info()
        assert not state["initialized"]

        frame = np.ones((50, 50), dtype=np.uint8) * 128
        simulator.process_frame(frame, 0.0)

        state = simulator.get_state_info()
        assert state["initialized"]
        assert "device" in state
        assert "shape" in state
        assert "buffer_stats" in state

    def test_mps_device_support(self):
        """Test MPS device support and dtype compatibility."""
        import torch

        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(device="auto")
        simulator = ESIMSimulator(config)

        expected_devices = {"cpu", "cuda", "mps"}
        assert simulator.device.type in expected_devices

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            config_mps = ESIMConfig(device="mps", dtype="float64")
            simulator_mps = ESIMSimulator(config_mps)

            assert simulator_mps.device.type == "mps"
            assert simulator_mps._dtype == torch.float32

    def test_device_dtype_compatibility(self):
        """Test device and dtype compatibility handling."""
        import torch

        from evlib.simulation.config import ESIMConfig
        from evlib.simulation.esim import ESIMSimulator

        config = ESIMConfig(device="cpu", dtype="float64")
        simulator = ESIMSimulator(config)
        assert simulator._dtype == torch.float64

        config = ESIMConfig(device="cpu", dtype="float32")
        simulator = ESIMSimulator(config)
        assert simulator._dtype == torch.float32


@pytest.mark.skipif(
    not CV2_AVAILABLE, reason="OpenCV required for video processor tests"
)
@pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch required for video processor tests"
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
    """Test convenience functions exist and are callable."""
    try:
        from evlib.simulation import create_esim_simulator, video_to_events

        assert callable(video_to_events)
        assert callable(create_esim_simulator)
    except ImportError:
        pass


def test_simple_video_to_events():
    """Test simple video to events function exists and is callable."""
    try:
        from evlib.simulation.video_processor import video_to_events_simple

        assert callable(video_to_events_simple)
    except ImportError:
        pass


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Integration tests require PyTorch")
@pytest.mark.skipif(not CV2_AVAILABLE, reason="Integration tests require OpenCV")
class TestSimulationIntegration:
    """Integration tests for the simulation module."""

    def test_full_pipeline_with_synthetic_video(self):
        """Test full pipeline with a synthetic video."""
        import cv2

        from evlib.simulation.config import ESIMConfig, VideoConfig
        from evlib.simulation.video_processor import VideoToEvents

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (64, 64))

            for i in range(10):
                frame = np.ones((64, 64, 3), dtype=np.uint8) * (100 + i * 10)
                out.write(frame)

            out.release()

            esim_config = ESIMConfig(
                device="cpu", positive_threshold=0.1, negative_threshold=0.1
            )
            video_config = VideoConfig(width=64, height=64)

            processor = VideoToEvents(esim_config, video_config)

            info = processor.get_video_info(video_path)
            assert info["width"] == 64
            assert info["height"] == 64

            x, y, t, p = processor.process_video(video_path)

            assert len(x) > 0
            assert len(y) == len(x)
            assert len(t) == len(x)
            assert len(p) == len(x)

        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
