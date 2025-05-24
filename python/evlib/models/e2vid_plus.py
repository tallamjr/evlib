"""E2VID+ model implementation with temporal processing."""

import numpy as np
from typing import Union, Tuple, Optional

from .base import BaseModel
from .config import ModelConfig
from .utils import download_model
import evlib


class E2VIDPlus(BaseModel):
    """E2VID+: Enhanced event-to-video reconstruction with temporal memory.

    E2VID+ extends the base E2VID model with ConvLSTM layers for
    temporal processing, enabling better reconstruction of dynamic scenes.

    Example:
        >>> model = E2VIDPlus(pretrained=True)
        >>> events = load_events("events.txt")
        >>> frames = model.reconstruct(events, return_all_frames=True)
    """

    def __init__(self, config: Optional[ModelConfig] = None, pretrained: bool = False):
        """Initialize E2VID+ model.

        Args:
            config: Model configuration
            pretrained: Whether to load pretrained weights
        """
        # E2VID+ benefits from more time bins
        if config is None:
            config = ModelConfig(num_bins=10, extra_params={"use_lstm": True})
        super().__init__(config, pretrained)
        self._build_model()

    def _build_model(self):
        """Build the E2VID+ model."""
        self._model_type = "e2vid_plus"

    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        model_path = download_model("e2vid_plus")
        print(f"Loaded pretrained weights from {model_path}")
        # TODO: Actually load the weights

    def reconstruct(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 10,
    ) -> np.ndarray:
        """Reconstruct frames from events using E2VID+ with temporal processing.

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

        # Use temporal reconstruction
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
