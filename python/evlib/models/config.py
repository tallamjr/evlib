"""Model configuration classes for evlib models."""

from dataclasses import dataclass, field
from typing import Dict, Any
from enum import Enum


class ModelArchitecture(Enum):
    """Supported model architectures."""

    E2VID_UNET = "e2vid_unet"
    FIRENET = "firenet"
    E2VID_PLUS = "e2vid_plus"
    FIRENET_PLUS = "firenet_plus"
    SPADE_E2VID = "spade_e2vid"
    HYBRID_SPADE_E2VID = "hybrid_spade_e2vid"
    SPADE_E2VID_LITE = "spade_e2vid_lite"
    SSL_E2VID = "ssl_e2vid"


@dataclass
class ModelConfig:
    """Configuration for event-to-video reconstruction models.

    Args:
        in_channels: Number of input channels (default: 5 for voxel grid)
        out_channels: Number of output channels (default: 1 for grayscale)
        base_channels: Base number of channels in the network (default: 64)
        num_layers: Number of layers/blocks in the network (default: 4)
        num_bins: Number of time bins for voxel grid (default: 5)
        use_gpu: Whether to use GPU if available (default: True)
        extra_params: Additional architecture-specific parameters
    """

    in_channels: int = 5
    out_channels: int = 1
    base_channels: int = 32
    num_layers: int = 4
    num_bins: int = 5
    use_gpu: bool = True
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        extra_params = config_dict.pop("extra_params", {})
        return cls(**config_dict, extra_params=extra_params)


@dataclass
class ModelInfo:
    """Information about a pre-trained model.

    Args:
        name: Model name (e.g., "e2vid_unet")
        variant: Model variant (e.g., "base", "plus", "lite")
        architecture: Model architecture type
        description: Model description
        size: Model size in bytes
        url: URL to download the model
        checksum: SHA256 checksum for verification
    """

    name: str
    variant: str
    architecture: ModelArchitecture
    description: str
    size: int
    url: str
    checksum: str

    @property
    def size_mb(self) -> float:
        """Return model size in megabytes."""
        return self.size / (1024 * 1024)


# Pre-defined configurations for common use cases
CONFIGS = {
    "default": ModelConfig(),  # Now matches original E2VID: base_channels=32, num_layers=4
    "lite": ModelConfig(
        base_channels=32, num_layers=3
    ),  # Matches pretrained e2vid-lite.pth
    "high_res": ModelConfig(base_channels=64, num_layers=5),  # Higher capacity variant
    "fast": ModelConfig(base_channels=16, num_layers=3),  # Ultra-fast variant
    "temporal": ModelConfig(num_bins=10, extra_params={"use_lstm": True}),
    "spade": ModelConfig(
        extra_params={"use_skip_connections": True, "spade_layers": [2, 3]}
    ),
    "ssl": ModelConfig(extra_params={"use_momentum": True, "temperature": 0.07}),
}


def get_config(name: str) -> ModelConfig:
    """Get a pre-defined configuration by name.

    Args:
        name: Configuration name (e.g., 'default', 'high_res', 'fast')

    Returns:
        ModelConfig instance

    Raises:
        ValueError: If configuration name is not found
    """
    if name not in CONFIGS:
        raise ValueError(
            f"Configuration '{name}' not found. Available: {list(CONFIGS.keys())}"
        )
    return CONFIGS[name]
