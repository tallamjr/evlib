"""ESIM simulator: Python wrapper over the Rust kernel (evlib.simulation_rs)."""

from typing import Optional, Tuple

import numpy as np
import polars as pl

import evlib
from .config import ESIMConfig

# ITU-R BT.601 luma weights, applied to RGB order.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _to_grey_u8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = np.rint(frame[..., :3].astype(np.float64) @ _LUMA)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _check_device(config: ESIMConfig) -> None:
    if config.device == "cuda":
        raise NotImplementedError("CUDA backend arrives in a later task")


def _to_dataframe(x, y, t_ns, p) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": pl.Series(x, dtype=pl.Int16),
            "y": pl.Series(y, dtype=pl.Int16),
            "t": pl.Series(t_ns // 1000, dtype=pl.Int64).cast(pl.Duration("us")),
            "polarity": pl.Series(p, dtype=pl.Int8),
        }
    )


def simulate_frames(
    frames: np.ndarray,
    timestamps_ns: np.ndarray,
    config: Optional[ESIMConfig] = None,
    sort: bool = True,
) -> pl.DataFrame:
    """Simulate events from a (T, H, W) uint8 or float32-log stack.

    Returns a DataFrame with x Int16, y Int16, t Duration(us), polarity Int8.
    """
    config = config or ESIMConfig()
    _check_device(config)
    frames = np.ascontiguousarray(frames)
    t = np.ascontiguousarray(np.asarray(timestamps_ns, dtype=np.int64))
    x, y, t_ns, p = evlib.simulation_rs.simulate_frames(
        frames, t, sort=sort, **config.kernel_kwargs()
    )
    return _to_dataframe(x, y, t_ns, p)


class ESIMSimulator:
    """Stateful frame-by-frame simulator with float-second timestamps."""

    def __init__(self, config: ESIMConfig, width: int, height: int):
        _check_device(config)
        self.config = config
        self.width = int(width)
        self.height = int(height)
        self._inner = evlib.simulation_rs.EventSimulator(
            width=self.width, height=self.height, **config.kernel_kwargs()
        )

    def reset(self) -> None:
        self._inner.reset()

    @property
    def is_initialised(self) -> bool:
        return self._inner.is_initialised

    def thresholds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._inner.thresholds()

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One (H, W) or (H, W, 3) RGB frame at `timestamp` seconds; returns (x, y, t_s, p)."""
        grey = _to_grey_u8(frame)
        if grey.shape != (self.height, self.width):
            raise ValueError(
                f"frame shape {grey.shape} does not match (height, width) = "
                f"{(self.height, self.width)}"
            )
        x, y, t_ns, p = self._inner.step(grey, int(round(timestamp * 1e9)))
        return (
            x.astype(np.int64),
            y.astype(np.int64),
            t_ns.astype(np.float64) / 1e9,
            p.astype(np.int64),
        )
