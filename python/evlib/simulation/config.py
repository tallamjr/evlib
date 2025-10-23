"""Configuration classes for event camera simulation."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal


@dataclass
class ESIMConfig:
    """Configuration for the ESIM (Event-based Simulator) algorithm.

    ESIM simulates event cameras by tracking log intensity changes and generating
    events when changes exceed specified thresholds.

    Args:
        positive_threshold: Contrast threshold for positive events (log intensity increase)
        negative_threshold: Contrast threshold for negative events (log intensity decrease)
        refractory_period_ms: Minimum time between events at the same pixel (milliseconds)
        log_floor: Minimum value for log intensity to avoid log(0)
        device: Computing device ("cuda", "cpu", or "auto" for automatic selection)
        dtype: Data type for computations ("float32" or "float64")
        extra_params: Additional algorithm-specific parameters
    """

    positive_threshold: float = 0.4
    negative_threshold: float = 0.4
    refractory_period_ms: float = 0.1
    log_floor: float = 0.001
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto"
    dtype: Literal["float32", "float64"] = "float64"
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.positive_threshold <= 0:
            raise ValueError("positive_threshold must be positive")
        if self.negative_threshold <= 0:
            raise ValueError("negative_threshold must be positive")
        if self.refractory_period_ms < 0:
            raise ValueError("refractory_period_ms must be non-negative")
        if self.log_floor <= 0:
            raise ValueError("log_floor must be positive")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ESIMConfig":
        """Create ESIMConfig from dictionary."""
        extra_params = config_dict.pop("extra_params", {})
        return cls(**config_dict, extra_params=extra_params)


@dataclass
class VideoConfig:
    """Configuration for video processing.

    Args:
        width: Target width for video frames (None to preserve original)
        height: Target height for video frames (None to preserve original)
        fps: Frames per second override (None to use video's native FPS)
        start_time: Start time in seconds (None to start from beginning)
        end_time: End time in seconds (None to process entire video)
        frame_skip: Number of frames to skip between processed frames (0 = process all)
        grayscale: Whether to convert frames to grayscale before processing
        extra_params: Additional video processing parameters
    """

    width: Optional[int] = 640
    height: Optional[int] = 480
    fps: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    frame_skip: int = 0
    grayscale: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.start_time is not None and self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time is not None and self.end_time < 0:
            raise ValueError("end_time must be non-negative")
        if self.start_time is not None and self.end_time is not None:
            if self.start_time >= self.end_time:
                raise ValueError("start_time must be less than end_time")
        if self.frame_skip < 0:
            raise ValueError("frame_skip must be non-negative")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VideoConfig":
        """Create VideoConfig from dictionary."""
        extra_params = config_dict.pop("extra_params", {})
        return cls(**config_dict, extra_params=extra_params)


# Pre-defined configurations for common use cases
ESIM_CONFIGS = {
    "default": ESIMConfig(),
    "high_sensitivity": ESIMConfig(positive_threshold=0.2, negative_threshold=0.2),
    "low_sensitivity": ESIMConfig(positive_threshold=0.8, negative_threshold=0.8),
    "fast": ESIMConfig(refractory_period_ms=0.01, dtype="float32"),
    "accurate": ESIMConfig(dtype="float64", log_floor=0.0001),
    "low_noise": ESIMConfig(
        positive_threshold=0.6, negative_threshold=0.6, refractory_period_ms=1.0
    ),
}

VIDEO_CONFIGS = {
    "default": VideoConfig(),
    "hd": VideoConfig(width=1280, height=720),
    "vga": VideoConfig(width=640, height=480),
    "qvga": VideoConfig(width=320, height=240),
    "fast": VideoConfig(frame_skip=1),  # Process every other frame
    "high_quality": VideoConfig(frame_skip=0, grayscale=False),
}


def get_esim_config(name: str) -> ESIMConfig:
    """Get a pre-defined ESIM configuration by name.

    Args:
        name: Configuration name (e.g., 'default', 'high_sensitivity', 'fast')

    Returns:
        ESIMConfig instance

    Raises:
        ValueError: If configuration name is not found
    """
    if name not in ESIM_CONFIGS:
        available = list(ESIM_CONFIGS.keys())
        raise ValueError(f"Unknown ESIM config '{name}'. Available: {available}")
    return ESIM_CONFIGS[name]


def get_video_config(name: str) -> VideoConfig:
    """Get a pre-defined video configuration by name.

    Args:
        name: Configuration name (e.g., 'default', 'hd', 'fast')

    Returns:
        VideoConfig instance

    Raises:
        ValueError: If configuration name is not found
    """
    if name not in VIDEO_CONFIGS:
        available = list(VIDEO_CONFIGS.keys())
        raise ValueError(f"Unknown video config '{name}'. Available: {available}")
    return VIDEO_CONFIGS[name]
