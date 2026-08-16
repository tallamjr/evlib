"""Event camera simulation: ESIM video-to-events on the Rust kernel (evlib.simulation_rs)."""

try:
    import cv2  # noqa: F401

    _opencv_available = True
except ImportError:
    _opencv_available = False

from .config import (
    ESIM_CONFIGS,
    VIDEO_CONFIGS,
    ESIMConfig,
    VideoConfig,
    get_esim_config,
    get_video_config,
)
from .esim import ESIMSimulator, simulate_frames

__all__ = [
    "ESIMConfig",
    "VideoConfig",
    "ESIM_CONFIGS",
    "VIDEO_CONFIGS",
    "get_esim_config",
    "get_video_config",
    "ESIMSimulator",
    "simulate_frames",
]

# Names gated on OpenCV. Without it, __getattr__ raises a clear install hint
# on first access instead of an AttributeError.
_OPENCV_NAMES = ("VideoToEvents", "estimate_event_count", "video_to_events")
_import_errors = {}
_missing_opencv_error = ImportError("opencv-python is not installed")

if _opencv_available:
    from .video_processor import VideoToEvents, estimate_event_count

    def video_to_events(video_path, esim_config=None, video_config=None):
        """Convert a video file to a Polars event frame with the ESIM kernel."""
        processor = VideoToEvents(
            esim_config or ESIMConfig(), video_config or VideoConfig()
        )
        return processor.process_video(video_path)

    __all__.extend(_OPENCV_NAMES)
else:
    for _name in _OPENCV_NAMES:
        _import_errors[_name] = _missing_opencv_error


def __getattr__(name):
    if name in _import_errors:
        raise ImportError(
            f"evlib.simulation.{name} is unavailable: {_import_errors[name]}. "
            f"Install with: pip install evlib[plot]"
        ) from _import_errors[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _check_dependencies():
    """Return a message naming missing optional dependencies, or None."""
    if not _opencv_available:
        return "Missing optional dependencies for simulation module: opencv-python"
    return None


def get_dependency_info():
    """Report optional dependency availability."""
    return {"opencv": _opencv_available, "missing_message": _check_dependencies()}
