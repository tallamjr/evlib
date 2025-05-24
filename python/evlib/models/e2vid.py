"""E2VID model implementation."""

import numpy as np
from typing import Union, Tuple, Optional

from .base import BaseModel
from .config import ModelConfig
from .utils import download_model
import evlib


class E2VID(BaseModel):
    """E2VID: Event to Video reconstruction model.

    Based on "High Speed and High Dynamic Range Video with an Event Camera"
    by Rebecq et al., CVPR 2019.

    This model uses a UNet architecture to reconstruct intensity frames
    from event data represented as voxel grids.

    Example:
        >>> model = E2VID(pretrained=True)
        >>> events = load_events("events.txt")
        >>> frames = model.reconstruct(events)
    """

    def __init__(self, variant: str = "unet", config: Optional[ModelConfig] = None, pretrained: bool = False):
        """Initialize E2VID model.

        Args:
            variant: Model variant ('unet' or 'onnx')
            config: Model configuration
            pretrained: Whether to load pretrained weights
        """
        super().__init__(config, pretrained)
        self.variant = variant
        self._build_model()

    def _build_model(self):
        """Build the E2VID model."""
        # The actual model is built in Rust
        self._model_type = "unet" if self.variant == "unet" else "onnx"

    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        model_name = f"e2vid_{self.variant}"
        model_path = download_model(model_name)
        print(f"Loaded pretrained weights from {model_path}")
        # TODO: Actually load the weights into the model

    def reconstruct(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        return_all_frames: bool = False,
    ) -> np.ndarray:
        """Reconstruct frames from events.

        Args:
            events: Event data
            height: Output height
            width: Output width
            return_all_frames: If True, return all intermediate frames.
                             If False, return only the final frame.

        Returns:
            Reconstructed frames of shape (height, width) or (num_frames, height, width)
        """
        # Preprocess events
        xs, ys, ts, ps, height, width = self.preprocess_events(events, height, width)

        # Use the advanced reconstruction API
        if return_all_frames:
            frames = evlib.processing.reconstruct_events_to_frames(
                xs,
                ys,
                ts,
                ps,
                height=height,
                width=width,
                num_frames=10,  # Default to 10 frames
                model_type=self._model_type,
                num_bins=self.config.num_bins,
            )
        else:
            # Single frame reconstruction
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

    def __repr__(self) -> str:
        return f"E2VID(variant='{self.variant}', config={self.config}, pretrained={self.pretrained})"
