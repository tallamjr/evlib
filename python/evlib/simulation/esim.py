"""ESIM simulator: Python wrapper over the Rust kernel (evlib.simulation_rs)."""

from typing import Optional, Tuple

import numpy as np
import polars as pl

import evlib
from .config import ESIMConfig

# ITU-R BT.601 luma weights, applied to RGB order.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _to_kernel_frame(frame: np.ndarray) -> np.ndarray:
    """uint8 (H, W) or (H, W, 3) RGB -> uint8 grey; float32 (H, W) -> log intensity as-is."""
    frame = np.asarray(frame)
    if frame.dtype == np.float32 and frame.ndim == 2:
        return np.ascontiguousarray(frame)
    if frame.dtype != np.uint8:
        raise TypeError(
            f"frame must be uint8 (H, W) or (H, W, 3), or float32 (H, W) log intensity, "
            f"got {frame.dtype} with shape {frame.shape}"
        )
    if frame.ndim == 3:
        frame = np.rint(frame[..., :3].astype(np.float64) @ _LUMA).astype(np.uint8)
    elif frame.ndim != 2:
        raise TypeError(f"frame must be 2-D or 3-D, got shape {frame.shape}")
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

    def step_ns(
        self, frame: np.ndarray, t_ns: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One frame at `t_ns`; returns the raw kernel arrays (x i16, y i16, t_ns i64, p i8)."""
        kernel_frame = _to_kernel_frame(frame)
        if kernel_frame.shape != (self.height, self.width):
            raise ValueError(
                f"frame shape {kernel_frame.shape} does not match (height, width) = "
                f"{(self.height, self.width)}"
            )
        return self._inner.step(kernel_frame, int(t_ns))

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One (H, W) or (H, W, 3) RGB frame at `timestamp` seconds; returns (x, y, t_s, p)."""
        x, y, t_ns, p = self.step_ns(frame, int(round(timestamp * 1e9)))
        return (
            x.astype(np.int64),
            y.astype(np.int64),
            t_ns.astype(np.float64) / 1e9,
            p.astype(np.int64),
        )
