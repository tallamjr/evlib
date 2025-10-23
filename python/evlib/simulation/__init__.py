"""
Event Camera Simulation Module

Provides event camera simulation capabilities with support for various algorithms
including ESIM (Event-based Simulator) for converting video frames to events.
"""

# Try to import optional dependencies
try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False

try:
    import cv2

    _opencv_available = True
except ImportError:
    _opencv_available = False

# Always import configuration
from .config import ESIMConfig, VideoConfig

__all__ = ["ESIMConfig", "VideoConfig"]

# Import classes based on available dependencies
if _opencv_available:
    from .video_processor import VideoToEvents, estimate_event_count

    __all__.extend(["VideoToEvents", "estimate_event_count"])

    # Convenience function for high-level API
    def video_to_events(video_path, esim_config=None, video_config=None):
        """
        Convert video file to events using ESIM algorithm.

        Args:
            video_path (str): Path to input video file
            esim_config (ESIMConfig, optional): ESIM configuration
            video_config (VideoConfig, optional): Video processing configuration

        Returns:
            tuple: (x, y, t, p) arrays containing event data
        """
        processor = VideoToEvents(
            esim_config or ESIMConfig(), video_config or VideoConfig()
        )
        return processor.process_video(video_path)

    __all__.append("video_to_events")

if _torch_available and _opencv_available:
    from .esim import ESIMSimulator

    __all__.append("ESIMSimulator")

    # Factory function
    def create_esim_simulator(config=None):
        """
        Create an ESIM simulator instance.

        Args:
            config (ESIMConfig, optional): Configuration for the simulator

        Returns:
            ESIMSimulator: Configured simulator instance
        """
        return ESIMSimulator(config or ESIMConfig())

    __all__.append("create_esim_simulator")


# Helpful error messages for missing dependencies
def _check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    if not _torch_available:
        missing.append("torch")
    if not _opencv_available:
        missing.append("opencv-python")

    if missing:
        return (
            f"Missing optional dependencies for simulation module: {', '.join(missing)}"
        )
    return None


def get_dependency_info():
    """Get information about available dependencies."""
    return {
        "torch": _torch_available,
        "opencv": _opencv_available,
        "missing_message": _check_dependencies(),
    }
