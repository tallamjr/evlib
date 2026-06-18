"""
Event Representations - Pure Python Implementation

Direct Polars API usage with automatic engine selection and fallbacks.
Provides event-to-representation conversion functions for event camera data.

This module provides exact API replacements for Rust PyO3 functions with identical
function names, signatures, and return types for seamless migration.
"""

import numpy as np
import polars as pl
from typing import Union, Literal

# Use Polars' native type definition
EngineType = Union[Literal["auto", "in-memory", "streaming", "gpu"], pl.GPUEngine]

# Helper type for flexible input handling
EventsInput = Union[pl.LazyFrame, pl.DataFrame]


def _ensure_lazy_frame(events: EventsInput) -> pl.LazyFrame:
    """Convert DataFrame to LazyFrame if needed, otherwise return as-is."""
    if isinstance(events, pl.DataFrame):
        return events.lazy()
    return events


def _collect_with_engine(lazy_frame: pl.LazyFrame, engine: EngineType) -> pl.DataFrame:
    """Safely collect LazyFrame with specified engine."""
    return lazy_frame.collect(engine=engine)


def create_stacked_histogram(
    events: EventsInput,
    height: int,
    width: int,
    bins: int = 10,
    window_duration_ms: float = 50.0,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Generate a stacked-histogram representation with direct Polars engine selection.

    This bins events into fixed-duration windows relative to the global minimum
    timestamp and returns a long-format DataFrame. It is NOT the RVT preprocessing
    representation: for an RVT-identical stacked histogram (the format expected by
    the RVT detection pipeline) use ``evlib.rvt`` instead, e.g.
    ``evlib.rvt.process_sequence`` / ``evlib.rvt.build_sparse_histogram``.

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        height: Sensor height in pixels
        width: Sensor width in pixels
        bins: Number of time bins
        window_duration_ms: Duration of each time window in milliseconds
        engine: Polars engine to use ("auto", "streaming", "gpu", or GPUEngine)

    Returns:
        DataFrame with columns 'time_bin', 'polarity', 'y', 'x', 'count'
    """

    time_window = window_duration_ms * 1000  # Convert to microseconds
    events_lf = _ensure_lazy_frame(events)

    return _collect_with_engine(
        events_lf.with_columns(
            [
                # Convert Duration to microseconds for arithmetic
                pl.col("t").dt.total_microseconds().alias("t_us")
            ]
        )
        .with_columns(
            [
                ((pl.col("t_us") - pl.col("t_us").min()) // time_window)
                .cast(pl.Int32)
                .alias("time_bin")
            ]
        )
        .filter(
            pl.col("x").is_between(0, width - 1)
            & pl.col("y").is_between(0, height - 1)
            & pl.col("time_bin").is_between(0, bins - 1)
        )
        .group_by(["time_bin", "polarity", "y", "x"])
        .agg(pl.len().alias("count"))
        .sort(["time_bin", "polarity", "y", "x"]),
        engine=engine,
    )


def create_voxel_grid(
    events: EventsInput,
    height: int,
    width: int,
    n_time_bins: int = 5,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Generate voxel grid with bilinear interpolation

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        height: Sensor height in pixels
        width: Sensor width in pixels
        n_time_bins: Number of temporal bins
        engine: Polars engine to use

    Returns:
        DataFrame with voxel grid contributions
    """

    events_lf = _ensure_lazy_frame(events)

    # Base frame: sort by time (tonic normalises against the first/last event),
    # cast time to Float64 microseconds (Float32 division diverges from numpy at
    # bin boundaries, a known evlib issue), map polarity to +1/-1, and compute
    # the normalised time, floored bin index and fractional offset. This mirrors
    # tonic's to_voxel_grid_numpy (Zhu et al. 2019 event volume).
    span = (pl.col("t_us").max() - pl.col("t_us").min()).cast(pl.Float64)
    base = (
        events_lf.sort("t")
        .with_columns(
            pl.col("t").dt.total_microseconds().cast(pl.Float64).alias("t_us"),
            # tonic maps p == 0 -> -1; treat any non-positive polarity as -1
            pl.when(pl.col("polarity") > 0)
            .then(pl.lit(1.0))
            .otherwise(pl.lit(-1.0))
            .alias("pol"),
        )
        .with_columns(
            # Guard the degenerate single-timestamp case (max == min): place all
            # mass in bin 0 by defining t_norm = 0.
            pl.when(span > 0)
            .then((pl.col("t_us") - pl.col("t_us").min()) / span * n_time_bins)
            .otherwise(pl.lit(0.0))
            .alias("t_norm")
        )
        .with_columns(
            pl.col("t_norm").floor().alias("t_i"),
        )
        .with_columns(
            pl.col("t_i").cast(pl.Int32).alias("t_i_int"),
            (pl.col("t_norm") - pl.col("t_i")).alias("dt"),
        )
        .filter(
            pl.col("x").is_between(0, width - 1) & pl.col("y").is_between(0, height - 1)
        )
    )

    spatial = [pl.col("x"), pl.col("y")]

    # Left scatter: pol * (1 - dt) into bin t_i, where t_i < n_time_bins.
    left = base.filter(pl.col("t_i_int") < n_time_bins).select(
        *spatial,
        pl.col("t_i_int").alias("time_bin"),
        (pl.col("pol") * (1.0 - pl.col("dt"))).alias("contribution"),
    )

    # Right scatter: pol * dt into bin t_i + 1, where t_i + 1 < n_time_bins.
    right = base.filter((pl.col("t_i_int") + 1) < n_time_bins).select(
        *spatial,
        (pl.col("t_i_int") + 1).alias("time_bin"),
        (pl.col("pol") * pl.col("dt")).alias("contribution"),
    )

    combined = (
        pl.concat([left, right])
        .group_by(["x", "y", "time_bin"])
        .agg(pl.col("contribution").sum().cast(pl.Float64).alias("contribution"))
    )

    return _collect_with_engine(combined, engine=engine)


def voxel_grid(
    events: EventsInput,
    height: int,
    width: int,
    n_time_bins: int = 5,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Alias for create_voxel_grid for backwards compatibility"""
    return create_voxel_grid(events, height, width, n_time_bins, engine)


def create_event_frame(
    events: EventsInput,
    height: int,
    width: int,
    n_time_bins: int = 10,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Generate a tonic-validated event frame (whole-recording, n_time_bins mode).

    This matches tonic's ``to_frame_numpy(n_time_bins=...)`` semantics: the WHOLE
    recording is sliced into ``n_time_bins`` equal-width TIME bins and events are
    counted per ``(polarity, y, x)`` per bin. It is a distinct representation from
    :func:`create_stacked_histogram` (which uses a fixed window duration).

    Boundary handling reproduces tonic's ``SliceByTimeBins`` (overlap=0) exactly:

      * The bin width is integer-floored: ``time_window = (t_max - t_min) //
        n_time_bins``. The bins span only ``[t_min, t_min + n_time_bins *
        time_window]`` (which is ``<= t_max``), so events at or beyond the last
        bin end (``time_bin >= n_time_bins``) are dropped, including the final
        max-time event. This mirrors tonic's ``searchsorted(side='left')``.
      * Bins are left-closed, right-open via ``floor``: an event exactly on a bin
        boundary belongs to the later bin.

    Float64 is used for the binning division because Polars Float32 division
    diverges from numpy at bin edges (a known evlib issue).

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        height: Sensor height in pixels
        width: Sensor width in pixels
        n_time_bins: Number of equal-width temporal bins
        engine: Polars engine to use ("auto", "streaming", "gpu", or GPUEngine)

    Returns:
        Long-format DataFrame with columns ``[time_bin, polarity, y, x, count]``,
        where ``polarity`` is the channel index (0 for negative, 1 for positive).
    """

    events_lf = _ensure_lazy_frame(events)

    # tonic computes the bin width as an INTEGER floor division of the integer
    # microsecond span; replicate via Float64 floor (Float32 diverges at edges).
    span = (pl.col("t_us").max() - pl.col("t_us").min()).cast(pl.Float64)
    time_window = span.floordiv(n_time_bins)

    base = (
        events_lf.with_columns(
            pl.col("t").dt.total_microseconds().cast(pl.Float64).alias("t_us"),
            # Map evlib polarity (-1/1 or 0/1) to channel index: <=0 -> 0, >0 -> 1.
            pl.when(pl.col("polarity") > 0)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("polarity_idx"),
        )
        .with_columns(
            # Degenerate single-timestamp span (time_window == 0): all events in
            # bin 0, matching tonic where every searchsorted slice but the first
            # collapses to empty (idx_start == idx_end) and bin 0 holds everything.
            pl.when(time_window > 0)
            .then(((pl.col("t_us") - pl.col("t_us").min()) / time_window).floor())
            .otherwise(pl.lit(0.0))
            .cast(pl.Int32)
            .alias("time_bin")
        )
        .filter(
            pl.col("x").is_between(0, width - 1)
            & pl.col("y").is_between(0, height - 1)
            & pl.col("time_bin").is_between(0, n_time_bins - 1)
        )
    )

    return _collect_with_engine(
        base.group_by(["time_bin", "polarity_idx", "y", "x"])
        .agg(pl.len().alias("count"))
        .rename({"polarity_idx": "polarity"})
        .sort(["time_bin", "polarity", "y", "x"]),
        engine=engine,
    )


def densify_event_frame(
    df: pl.DataFrame,
    n_time_bins: int,
    n_polarities: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Scatter a long-format event frame DataFrame into a dense array.

    Bridges the long-format output of :func:`create_event_frame` (columns
    ``time_bin, polarity, y, x, count``) to a dense tensor matching tonic's
    ``to_frame_numpy`` output shape ``(n_time_bins, P, H, W)``.

    Args:
        df: DataFrame with columns ``time_bin``, ``polarity`` (channel index),
            ``y``, ``x``, ``count``.
        n_time_bins: Number of temporal bins (size of axis 0).
        n_polarities: Number of polarity channels (size of axis 1).
        height: Sensor height in pixels.
        width: Sensor width in pixels.

    Returns:
        Dense ``(n_time_bins, n_polarities, height, width)`` int64 array.
    """
    dense = np.zeros((n_time_bins, n_polarities, height, width), dtype=np.int64)
    if df.height == 0:
        return dense

    time_bin = df["time_bin"].to_numpy().astype(np.int64)
    polarity = df["polarity"].to_numpy().astype(np.int64)
    y = df["y"].to_numpy().astype(np.int64)
    x = df["x"].to_numpy().astype(np.int64)
    count = df["count"].to_numpy().astype(np.int64)

    flat = dense.reshape(-1)
    indices = (((time_bin * n_polarities) + polarity) * height + y) * width + x
    np.add.at(flat, indices, count)
    return dense


def densify_voxel_grid(
    df: pl.DataFrame,
    n_time_bins: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Scatter a long-format voxel grid DataFrame into a dense array.

    Bridges the long-format output of :func:`create_voxel_grid` (columns
    ``x, y, time_bin, contribution``) to a dense tensor suitable for models and
    for tonic-style validation. Contributions are summed per cell.

    Args:
        df: DataFrame with columns ``x``, ``y``, ``time_bin``, ``contribution``.
        n_time_bins: Number of temporal bins (size of axis 0).
        height: Sensor height in pixels.
        width: Sensor width in pixels.

    Returns:
        Dense ``(n_time_bins, 1, height, width)`` float64 array.
    """
    dense = np.zeros((n_time_bins, 1, height, width), dtype=np.float64)
    if df.height == 0:
        return dense

    time_bin = df["time_bin"].to_numpy().astype(np.int64)
    y = df["y"].to_numpy().astype(np.int64)
    x = df["x"].to_numpy().astype(np.int64)
    contribution = df["contribution"].to_numpy().astype(np.float64)

    flat = dense.reshape(-1)
    indices = ((time_bin * height) + y) * width + x
    np.add.at(flat, indices, contribution)
    return dense


def create_mixed_density_stack(
    events: EventsInput,
    height: int,
    width: int,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Generate mixed density stack representation

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        height: Sensor height in pixels
        width: Sensor width in pixels
        engine: Polars engine to use

    Returns:
        DataFrame with mixed density stack values
    """

    events_lf = _ensure_lazy_frame(events)

    return _collect_with_engine(
        events_lf.with_columns(
            [
                # Convert Duration to microseconds for arithmetic
                pl.col("t").dt.total_microseconds().alias("t_us")
            ]
        )
        .filter(
            pl.col("x").is_between(0, width - 1) & pl.col("y").is_between(0, height - 1)
        )
        .group_by(["x", "y"])
        .agg([pl.col("polarity").sum().alias("polarity_sum"), pl.len().alias("count")]),
        engine=engine,
    )


def time_surface(
    events: EventsInput,
    height: int,
    width: int,
    tau: float = 100000.0,  # microseconds
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Generate time surface representation

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        height: Sensor height in pixels
        width: Sensor width in pixels
        tau: Time constant for exponential decay
        engine: Polars engine to use

    Returns:
        DataFrame with time surface values
    """

    events_lf = _ensure_lazy_frame(events)

    return _collect_with_engine(
        events_lf.sort("t")
        .group_by(["x", "y", "polarity"])
        .agg(
            [pl.col("t").last().alias("last_timestamp"), pl.len().alias("event_count")]
        )
        .with_columns(
            [
                # Exponential decay from last timestamp
                (-(pl.col("last_timestamp").max() - pl.col("last_timestamp")) / tau)
                .exp()
                .alias("surface_value")
            ]
        )
        .filter(
            pl.col("x").is_between(0, width - 1) & pl.col("y").is_between(0, height - 1)
        ),
        engine=engine,
    )


def event_histogram(
    events: EventsInput,
    height: int,
    width: int,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Generate simple event count histogram

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        height: Sensor height in pixels
        width: Sensor width in pixels
        engine: Polars engine to use

    Returns:
        DataFrame with event counts per pixel and polarity
    """

    events_lf = _ensure_lazy_frame(events)

    return _collect_with_engine(
        events_lf.with_columns(
            [
                # Convert Duration to microseconds for arithmetic if needed
                pl.col("t").dt.total_microseconds().alias("t_us")
            ]
        )
        .filter(
            pl.col("x").is_between(0, width - 1) & pl.col("y").is_between(0, height - 1)
        )
        .group_by(["x", "y", "polarity"])
        .agg(
            [
                pl.len().alias("count"),
                pl.col("t_us").sum().alias("polarity_sum"),
            ]  # Mixed density calculation
        ),
        engine=engine,
    )


def preprocess_for_detection(
    events: EventsInput,
    height: int,
    width: int,
    bins: int = 5,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Preprocess events for object detection tasks

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        height: Sensor height in pixels
        width: Sensor width in pixels
        bins: Number of time bins for preprocessing
        engine: Polars engine to use

    Returns:
        DataFrame preprocessed for detection tasks
    """
    return create_stacked_histogram(events, height, width, bins, engine=engine)


def benchmark_vs_rvt(
    events: EventsInput,
    height: int,
    width: int,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Benchmark representation against RVT format

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        height: Sensor height in pixels
        width: Sensor width in pixels
        engine: Polars engine to use

    Returns:
        DataFrame with benchmarking results
    """
    return create_stacked_histogram(events, height, width, bins=20, engine=engine)


# Export all functions
__all__ = [
    "create_stacked_histogram",
    "create_voxel_grid",
    "voxel_grid",
    "densify_voxel_grid",
    "create_event_frame",
    "densify_event_frame",
    "create_mixed_density_stack",
    "time_surface",
    "event_histogram",
    "preprocess_for_detection",
    "benchmark_vs_rvt",
]
