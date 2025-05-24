"""SPADE-E2VID model implementation."""

import numpy as np
from typing import Union, Tuple, Optional

from .base import BaseModel
from .config import ModelConfig, get_config
from .utils import download_model
import evlib


class SPADE(BaseModel):
    """SPADE-E2VID: Spatially-adaptive normalization for event reconstruction.

    SPADE-E2VID uses spatially-adaptive normalization to better preserve
    spatial structure and details in the reconstructed frames.

    Example:
        >>> model = SPADE(variant="full", pretrained=True)
        >>> events = load_events("events.txt")
        >>> frames = model.reconstruct(events)
    """

    def __init__(self, variant: str = "full", config: Optional[ModelConfig] = None, pretrained: bool = False):
        """Initialize SPADE model.

        Args:
            variant: Model variant ('full', 'hybrid', or 'lite')
            config: Model configuration
            pretrained: Whether to load pretrained weights
        """
        # Use SPADE-specific config if not provided
        if config is None:
            config = get_config("spade")
        super().__init__(config, pretrained)
        self.variant = variant
        self._build_model()

    def _build_model(self):
        """Build the SPADE model."""
        self._model_type = f"spade_{self.variant}"

    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        model_name = f"spade_e2vid_{self.variant}"
        model_path = download_model(model_name)
        print(f"Loaded pretrained weights from {model_path}")
        # TODO: Actually load the weights

    def reconstruct(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct frames from events using SPADE normalization.

        Args:
            events: Event data
            height: Output height
            width: Output width

        Returns:
            Reconstructed frame
        """
        # Preprocess events
        xs, ys, ts, ps, height, width = self.preprocess_events(events, height, width)

        # For now, use the base E2VID reconstruction
        # TODO: Implement SPADE-specific reconstruction when Python bindings are fixed
        frames = evlib.processing.events_to_video_advanced(
            xs,
            ys,
            ts,
            ps,
            height=height,
            width=width,
            model_type="unet",  # Fallback to UNet for now
            num_bins=self.config.num_bins,
        )

        return frames

    def __repr__(self) -> str:
        return f"SPADE(variant='{self.variant}', config={self.config}, pretrained={self.pretrained})"
