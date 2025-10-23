"""Base class for all event-to-video reconstruction models."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

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
        events: Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
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
        events: Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any
        ],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Preprocess events into standard format.

        Args:
            events: Either a structured array, tuple of arrays, or Polars LazyFrame
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
        elif hasattr(events, "collect") or hasattr(events, "columns"):
            # Polars LazyFrame or DataFrame
            try:
                import polars as pl
            except ImportError:
                raise ImportError("Polars is required for LazyFrame/DataFrame input")

            # Convert LazyFrame to DataFrame if needed
            if hasattr(events, "collect"):
                events_df = events.collect()
            else:
                events_df = events

            # Extract arrays
            xs = events_df["x"].to_numpy()
            ys = events_df["y"].to_numpy()

            # Handle timestamp conversion from Duration to seconds
            if events_df["t"].dtype == pl.Duration:
                # Convert from microseconds to seconds
                ts = (
                    events_df["t"].dt.total_microseconds().to_numpy().astype(np.float64)
                    / 1e6
                )
            else:
                ts = events_df["t"].to_numpy().astype(np.float64)

            ps = events_df["polarity"].to_numpy()
        else:
            raise ValueError(
                "Events must be either a structured array, tuple of (x, y, t, p), or Polars LazyFrame/DataFrame"
            )

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
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Polars is required for voxel grid creation")

        # Convert to Polars LazyFrame for voxel grid creation
        events_lf = pl.LazyFrame(
            {
                "x": xs.astype(np.int16),
                "y": ys.astype(np.int16),
                "t": pl.Series((ts * 1e6).astype(np.int64)).cast(
                    pl.Duration(time_unit="us")
                ),
                "polarity": ps.astype(np.int8),
            }
        )

        # Create voxel grid using evlib representations
        voxel_df = evlib.representations.create_voxel_grid(
            events_lf, height, width, self.config.num_bins
        )

        # Convert back to numpy array format (num_bins, height, width)
        voxel_array = np.zeros((self.config.num_bins, height, width), dtype=np.float32)

        # Populate the array from the DataFrame
        if hasattr(voxel_df, "collect"):
            voxel_data = voxel_df.collect()
        else:
            voxel_data = voxel_df

        for row in voxel_data.iter_rows(named=True):
            x = row["x"]
            y = row["y"]
            time_bin = row["time_bin"]
            contribution = row["contribution"]
            if (
                0 <= x < width
                and 0 <= y < height
                and 0 <= time_bin < self.config.num_bins
            ):
                voxel_array[time_bin, y, x] = contribution

        return voxel_array

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(config={self.config}, pretrained={self.pretrained})"
