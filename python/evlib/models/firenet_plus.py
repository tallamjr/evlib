"""FireNet+ model implementation with temporal processing."""

import numpy as np
from typing import Union, Tuple, Optional

from .base import BaseModel
from .config import ModelConfig
from .utils import download_model
import evlib


class FireNetPlus(BaseModel):
    """FireNet+: Lightweight temporal event-to-video reconstruction.

    FireNet+ combines the efficiency of FireNet with temporal
    processing capabilities for better dynamic scene reconstruction.

    Example:
        >>> model = FireNetPlus(pretrained=True)
        >>> events = load_events("events.txt")
        >>> frames = model.reconstruct(events, num_frames=5)
    """

    def __init__(self, config: Optional[ModelConfig] = None, pretrained: bool = False):
        """Initialize FireNet+ model.

        Args:
            config: Model configuration
            pretrained: Whether to load pretrained weights
        """
        # FireNet+ uses fewer channels but more time bins
        if config is None:
            config = ModelConfig(base_channels=32, num_bins=8, extra_params={"use_temporal_gates": True})
        super().__init__(config, pretrained)
        self._build_model()

    def _build_model(self):
        """Build the FireNet+ model."""
        self._model_type = "firenet_plus"

    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        model_path = download_model("firenet_plus")
        print(f"Loaded pretrained weights from {model_path}")
        # TODO: Actually load the weights

    def reconstruct(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 5,
    ) -> np.ndarray:
        """Reconstruct frames from events using FireNet+.

        Args:
            events: Event data
            height: Output height
            width: Output width
            num_frames: Number of frames to reconstruct

        Returns:
            Reconstructed frames of shape (num_frames, height, width)
        """
        # Preprocess events
        xs, ys, ts, ps, height, width = self.preprocess_events(events, height, width)

        # Use temporal reconstruction with FireNet+
        frames = evlib.processing.events_to_video_temporal(
            xs,
            ys,
            ts,
            ps,
            height=height,
            width=width,
            num_frames=num_frames,
            model_type=self._model_type,
            num_bins=self.config.num_bins,
        )

        return frames
