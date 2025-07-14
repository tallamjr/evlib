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
        self.variant = variant
        super().__init__(config, pretrained)
        self._build_model()

    def _build_model(self):
        """Build the E2VID model."""
        # The actual model is built in Rust
        if self.variant == "recurrent":
            self._model_type = "recurrent"
        elif self.variant == "unet":
            self._model_type = "unet"
        else:
            self._model_type = "onnx"

    def _detect_model_architecture(self, model_path: str) -> str:
        """Detect model architecture from PyTorch weights.

        Args:
            model_path: Path to the PyTorch model file

        Returns:
            Model architecture type: 'recurrent' or 'unet'
        """
        try:
            import torch

            model_data = torch.load(model_path, map_location="cpu")

            # Check if it's a state dict or has state_dict key
            state_dict = (
                model_data.get("state_dict", model_data) if isinstance(model_data, dict) else model_data
            )

            # Look for recurrent-specific keys
            keys = list(state_dict.keys())
            recurrent_keys = [k for k in keys if "recurrent" in k.lower()]
            lstm_keys = [k for k in keys if "lstm" in k.lower() or "Gates" in k]

            if recurrent_keys or lstm_keys:
                print(
                    f"Detected E2VID Recurrent architecture ({len(recurrent_keys + lstm_keys)} recurrent parameters)"
                )
                return "recurrent"
            else:
                print(f"Detected E2VID UNet architecture ({len(keys)} parameters)")
                return "unet"

        except Exception as e:
            print(f"Could not detect architecture from {model_path}: {e}")
            print(f"   Falling back to variant: {self.variant}")
            return self.variant

    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        model_name = f"e2vid_{self.variant}"
        model_path = download_model(model_name)

        # Detect the actual model architecture from the weights
        detected_arch = self._detect_model_architecture(model_path)

        if detected_arch != self.variant:
            print(f"🔄 Detected architecture '{detected_arch}' differs from variant '{self.variant}'")
            print("   Updating model to use detected architecture")
            self.variant = detected_arch
            self._model_type = detected_arch

        print(f"Loaded pretrained weights from {model_path}")

        # Store the model path for later loading into Rust backend
        self._pretrained_model_path = model_path

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

        # For recurrent models with pretrained weights, use direct Rust backend
        if self._model_type == "recurrent" and hasattr(self, "_pretrained_model_path"):
            try:
                # Use the direct Rust E2Vid backend with loaded weights
                from evlib.evlib_rust.processing import E2Vid

                # Create Rust model and load weights
                rust_model = E2Vid()
                rust_model.load_model_from_file(self._pretrained_model_path)

                # Reconstruct using the Rust backend directly
                frames = rust_model.reconstruct_frame(xs, ys, ts, ps, width, height)

                return frames

            except Exception as e:
                print(f"Direct Rust backend failed: {e}")
                print("   Falling back to processing API")

        # Use the advanced reconstruction API for other cases
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
