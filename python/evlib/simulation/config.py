"""Configuration dataclasses for the ESIM simulator and video decode."""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class ESIMConfig:
    """ESIM kernel parameters. Thresholds are log-intensity contrast; refractory in ms."""

    positive_threshold: float = 0.2
    negative_threshold: float = 0.2
    refractory_period_ms: float = 0.0
    log_eps: float = 1e-3
    threshold_sigma: float = 0.0
    seed: int = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"

    def __post_init__(self) -> None:
        if self.positive_threshold <= 0:
            raise ValueError("positive_threshold must be positive")
        if self.negative_threshold <= 0:
            raise ValueError("negative_threshold must be positive")
        if self.refractory_period_ms < 0:
            raise ValueError("refractory_period_ms must be non-negative")
        if self.log_eps <= 0:
            raise ValueError("log_eps must be positive")
        if self.threshold_sigma < 0:
            raise ValueError("threshold_sigma must be non-negative")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'")

    @property
    def refractory_ns(self) -> int:
        return int(round(self.refractory_period_ms * 1e6))

    def kernel_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for evlib.simulation_rs functions."""
        return {
            "c_pos": self.positive_threshold,
            "c_neg": self.negative_threshold,
            "threshold_sigma": self.threshold_sigma,
            "refractory_ns": self.refractory_ns,
            "log_eps": self.log_eps,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ESIMConfig":
        return cls(**config_dict)


@dataclass
class VideoConfig:
    """Video decode settings. None width/height keeps the source size; times in seconds."""

    width: Optional[int] = 640
    height: Optional[int] = 480
    fps: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    frame_skip: int = 0
    grayscale: bool = True

    def __post_init__(self) -> None:
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
        return cls(**config_dict)


ESIM_CONFIGS = {
    "default": ESIMConfig(),
    "high_sensitivity": ESIMConfig(positive_threshold=0.1, negative_threshold=0.1),
    "low_sensitivity": ESIMConfig(positive_threshold=0.5, negative_threshold=0.5),
    "low_noise": ESIMConfig(
        positive_threshold=0.3, negative_threshold=0.3, refractory_period_ms=1.0
    ),
    "mismatch": ESIMConfig(threshold_sigma=0.1),
}

VIDEO_CONFIGS = {
    "default": VideoConfig(),
    "hd": VideoConfig(width=1280, height=720),
    "vga": VideoConfig(width=640, height=480),
    "qvga": VideoConfig(width=320, height=240),
    "fast": VideoConfig(frame_skip=1),
    "high_quality": VideoConfig(frame_skip=0, grayscale=False),
}


def get_esim_config(name: str) -> ESIMConfig:
    """Return a preset ESIMConfig by name; raise ValueError for unknown names."""
    if name not in ESIM_CONFIGS:
        available = list(ESIM_CONFIGS.keys())
        raise ValueError(f"Unknown ESIM config '{name}'. Available: {available}")
    return ESIM_CONFIGS[name]


def get_video_config(name: str) -> VideoConfig:
    """Return a preset VideoConfig by name; raise ValueError for unknown names."""
    if name not in VIDEO_CONFIGS:
        available = list(VIDEO_CONFIGS.keys())
        raise ValueError(f"Unknown video config '{name}'. Available: {available}")
    return VIDEO_CONFIGS[name]
