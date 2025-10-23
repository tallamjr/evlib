"""
Event Representations - Pure Python Implementation

Direct Polars API usage with automatic engine selection and fallbacks.
Provides event-to-representation conversion functions for event camera data.

This module provides exact API replacements for Rust PyO3 functions with identical
function names, signatures, and return types for seamless migration.
"""

import polars as pl
from typing import Union, Literal
from polars.lazyframe.engine_config import GPUEngine

# Use Polars' native type definition
EngineType = Union[Literal["auto", "in-memory", "streaming", "gpu"], GPUEngine]

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
    """Generate stacked histogram with direct Polars engine selection

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


def stacked_histogram(
    events: EventsInput,
    height: int,
    width: int,
    bins: int = 10,
    window_duration_ms: float = 50.0,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Alias for create_stacked_histogram for backwards compatibility"""
    return create_stacked_histogram(
        events, height, width, bins, window_duration_ms, engine
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

    return _collect_with_engine(
        events_lf.with_columns(
            [
                # Convert Duration to microseconds for arithmetic
                pl.col("t").dt.total_microseconds().alias("t_us")
            ]
        )
        .with_columns(
            [
                # Normalize time to [0, n_time_bins)
                (
                    (pl.col("t_us") - pl.col("t_us").min())
                    / (pl.col("t_us").max() - pl.col("t_us").min())
                    * n_time_bins
                ).alias("t_norm")
            ]
        )
        .with_columns(
            [
                # Bilinear interpolation weights
                pl.col("t_norm").floor().cast(pl.Int32).alias("t_low"),
                (pl.col("t_norm").floor() + 1).cast(pl.Int32).alias("t_high"),
                (pl.col("t_norm") - pl.col("t_norm").floor()).alias("weight_high"),
                (1.0 - (pl.col("t_norm") - pl.col("t_norm").floor())).alias(
                    "weight_low"
                ),
            ]
        )
        .filter(
            pl.col("x").is_between(0, width - 1)
            & pl.col("y").is_between(0, height - 1)
            & pl.col("t_low").is_between(0, n_time_bins - 1)
        )
        .select(
            [
                pl.col("x"),
                pl.col("y"),
                pl.col("t_low").alias("time_bin"),
                (pl.col("polarity") * pl.col("weight_low")).alias("contribution"),
            ]
        ),
        engine=engine,
    )


def voxel_grid(
    events: EventsInput,
    height: int,
    width: int,
    n_time_bins: int = 5,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Alias for create_voxel_grid for backwards compatibility"""
    return create_voxel_grid(events, height, width, n_time_bins, engine)


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
    "stacked_histogram",
    "create_voxel_grid",
    "voxel_grid",
    "create_mixed_density_stack",
    "time_surface",
    "event_histogram",
    "preprocess_for_detection",
    "benchmark_vs_rvt",
]
