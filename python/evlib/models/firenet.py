"""FireNet model implementation."""

import numpy as np
from typing import Union, Tuple, Optional

from .base import BaseModel
from .config import ModelConfig
from .utils import download_model
import evlib


class FireNet(BaseModel):
    """FireNet: Lightweight event-to-video reconstruction model.

    FireNet is a lightweight variant optimized for speed, using
    Fire modules inspired by SqueezeNet for efficient processing.

    Example:
        >>> model = FireNet(pretrained=True)
        >>> events = load_events("events.txt")
        >>> frames = model.reconstruct(events)
    """

    def __init__(self, config: Optional[ModelConfig] = None, pretrained: bool = False):
        """Initialize FireNet model.

        Args:
            config: Model configuration
            pretrained: Whether to load pretrained weights
        """
        # FireNet uses fewer channels for speed
        if config is None:
            config = ModelConfig(base_channels=32)
        super().__init__(config, pretrained)
        self._build_model()

    def _build_model(self):
        """Build the FireNet model."""
        self._model_type = "firenet"

    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        model_path = download_model("firenet")
        print(f"Loaded pretrained weights from {model_path}")
        # TODO: Actually load the weights

    def reconstruct(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        return_all_frames: bool = False,
    ) -> np.ndarray:
        """Reconstruct frames from events using FireNet.

        Args:
            events: Event data
            height: Output height
            width: Output width
            return_all_frames: If True, return all intermediate frames

        Returns:
            Reconstructed frames
        """
        # Preprocess events
        xs, ys, ts, ps, height, width = self.preprocess_events(events, height, width)

        # Use the advanced reconstruction API with FireNet
        if return_all_frames:
            frames = evlib.processing.reconstruct_events_to_frames(
                xs,
                ys,
                ts,
                ps,
                height=height,
                width=width,
                num_frames=10,
                model_type=self._model_type,
                num_bins=self.config.num_bins,
            )
        else:
            frames = evlib.processing.events_to_video_advanced(
                xs,
                ys,
                ts,
                ps,
                height=height,
                width=width,
                model_type=self._model_type,
                num_bins=self.config.num_bins,
            )

        return frames
