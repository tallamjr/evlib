"""Base class for all event-to-video reconstruction models."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

from .config import ModelConfig
import evlib


class BaseModel(ABC):
    """Abstract base class for event-to-video reconstruction models.

    All models should inherit from this class and implement the abstract methods.
    """

    def __init__(self, config: Optional[ModelConfig] = None, pretrained: bool = False):
        """Initialize the model.

        Args:
            config: Model configuration. If None, uses default configuration.
            pretrained: Whether to load pretrained weights.
        """
        self.config = config or ModelConfig()
        self.pretrained = pretrained
        self._model = None

        if pretrained:
            self._load_pretrained_weights()

    @abstractmethod
    def _build_model(self):
        """Build the model architecture."""
        pass

    @abstractmethod
    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        pass

    @abstractmethod
    def reconstruct(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct frames from events.

        Args:
            events: Either a structured array with fields 'x', 'y', 't', 'p',
                   or a tuple of (xs, ys, ts, ps) arrays.
            height: Output image height. If None, inferred from events.
            width: Output image width. If None, inferred from events.

        Returns:
            Reconstructed frames as numpy array of shape (num_frames, height, width)
        """
        pass

    def preprocess_events(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Preprocess events into standard format.

        Args:
            events: Either a structured array or tuple of arrays
            height: Image height
            width: Image width

        Returns:
            Tuple of (xs, ys, ts, ps, height, width)
        """
        if isinstance(events, tuple) and len(events) == 4:
            xs, ys, ts, ps = events
        elif isinstance(events, np.ndarray) and events.dtype.names is not None:
            # Structured array
            xs = events["x"]
            ys = events["y"]
            ts = events["t"]
            ps = events["p"]
        else:
            raise ValueError("Events must be either a structured array or tuple of (x, y, t, p)")

        # Ensure correct dtypes
        xs = np.asarray(xs, dtype=np.int64)
        ys = np.asarray(ys, dtype=np.int64)
        ts = np.asarray(ts, dtype=np.float64)
        ps = np.asarray(ps, dtype=np.int64)

        # Infer dimensions if not provided
        if height is None:
            height = int(np.max(ys)) + 1
        if width is None:
            width = int(np.max(xs)) + 1

        return xs, ys, ts, ps, height, width

    def events_to_voxel_grid(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        ps: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Convert events to voxel grid representation.

        Args:
            xs, ys, ts, ps: Event arrays
            height: Image height
            width: Image width

        Returns:
            Voxel grid of shape (num_bins, height, width)
        """
        return evlib.representations.events_to_voxel_grid(
            xs, ys, ts, ps, self.config.num_bins, (width, height), "count"
        )

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(config={self.config}, pretrained={self.pretrained})"
