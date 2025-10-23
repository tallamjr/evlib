"""
Event Filtering - Pure Python Implementation

Direct Polars API usage with automatic engine selection and streaming support.
Provides event filtering functions for noise reduction and spatial-temporal filtering.

This module provides exact API replacements for Rust PyO3 functions with identical
function names, signatures, and return types for seamless migration.
"""

import polars as pl
from typing import Union, Literal, Optional, List
from polars.lazyframe.engine_config import GPUEngine
import logging

logger = logging.getLogger(__name__)

# Use Polars' native type definition
EngineType = Union[Literal["auto", "in-memory", "streaming", "gpu"], GPUEngine]

# Helper type for flexible input handling
EventsInput = Union[pl.LazyFrame, pl.DataFrame]


def _ensure_lazy_frame(events: EventsInput) -> pl.LazyFrame:
    """Convert DataFrame to LazyFrame if needed, otherwise return as-is."""
    if isinstance(events, pl.DataFrame):
        return events.lazy()
    return events


def _apply_engine_hint(lazy_frame: pl.LazyFrame, engine: EngineType) -> pl.LazyFrame:
    """
    Apply engine hint to LazyFrame without collecting.

    For filtering operations, we want to maintain lazy evaluation to allow
    efficient chaining of multiple filters. This function just returns the
    LazyFrame with potential optimizations based on the engine parameter.
    """
    # For now, just return the LazyFrame as-is since we want to maintain
    # lazy evaluation for chaining. The engine will be applied when
    # the user finally collects the results.
    return lazy_frame


def _should_use_streaming(events: pl.LazyFrame) -> bool:
    """
    Determine if streaming should be used based on data size.

    This is a heuristic based on the number of events. For very large datasets,
    streaming becomes more memory-efficient.
    """
    try:
        # Quick estimate of data size - if we can't get it cheaply, assume streaming
        row_count = events.select(pl.len()).collect().item()
        # Use streaming for datasets larger than 5M events (similar to Rust backend)
        return row_count > 5_000_000
    except Exception:
        # If we can't determine size, err on the side of streaming
        return True


def filter_by_time(
    events: EventsInput,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    engine: EngineType = "auto",
) -> pl.LazyFrame:
    """
    Filter events by time range with streaming support

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        t_start: Start time in seconds (None for no lower bound)
        t_end: End time in seconds (None for no upper bound)
        engine: Polars engine to use ("auto", "streaming", "gpu", or GPUEngine)

    Returns:
        LazyFrame with events filtered by time range

    Example:
        ```python
        import evlib
        from evlib.filtering import filter_by_time

        events = evlib.load_events("data.h5")

        # Filter to events between 0.1s and 0.5s
        filtered = filter_by_time(events, t_start=0.1, t_end=0.5)

        # Use streaming engine for large datasets
        filtered = filter_by_time(events, t_start=0.1, t_end=0.5, engine="streaming")
        ```
    """
    events_lf = _ensure_lazy_frame(events)

    # Build time filter conditions
    conditions = []

    if t_start is not None:
        # Convert seconds to microseconds for comparison
        t_start_us = t_start * 1_000_000
        conditions.append(pl.col("t").dt.total_microseconds() >= t_start_us)

    if t_end is not None:
        # Convert seconds to microseconds for comparison
        t_end_us = t_end * 1_000_000
        conditions.append(pl.col("t").dt.total_microseconds() <= t_end_us)

    # Apply filters
    if conditions:
        # Combine conditions with AND
        if len(conditions) == 1:
            filter_condition = conditions[0]
        else:
            filter_condition = conditions[0]
            for condition in conditions[1:]:
                filter_condition = filter_condition & condition

        filtered_lf = events_lf.filter(filter_condition)
    else:
        # No filters specified, return original
        filtered_lf = events_lf

    return _apply_engine_hint(filtered_lf, engine)


def filter_by_roi(
    events: EventsInput,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    engine: EngineType = "auto",
) -> pl.LazyFrame:
    """
    Filter events by region of interest (ROI)

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        x_min: Minimum x coordinate (inclusive)
        x_max: Maximum x coordinate (inclusive)
        y_min: Minimum y coordinate (inclusive)
        y_max: Maximum y coordinate (inclusive)
        engine: Polars engine to use

    Returns:
        LazyFrame with events filtered by spatial ROI

    Example:
        ```python
        from evlib.filtering import filter_by_roi

        # Filter to center region
        filtered = filter_by_roi(events, x_min=200, x_max=400, y_min=150, y_max=350)
        ```
    """
    events_lf = _ensure_lazy_frame(events)

    # Apply spatial filter
    filtered_lf = events_lf.filter(
        pl.col("x").is_between(x_min, x_max) & pl.col("y").is_between(y_min, y_max)
    )

    return _apply_engine_hint(filtered_lf, engine)


def filter_by_polarity(
    events: EventsInput, polarity: Union[int, List[int]], engine: EngineType = "auto"
) -> pl.LazyFrame:
    """
    Filter events by polarity value(s)

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        polarity: Polarity value(s) to keep (0/1 or -1/1 depending on data format)
        engine: Polars engine to use

    Returns:
        LazyFrame with events filtered by polarity

    Example:
        ```python
        from evlib.filtering import filter_by_polarity

        # Keep only positive events
        positive_events = filter_by_polarity(events, polarity=1)

        # Keep both positive and negative events (if using other polarities)
        both_events = filter_by_polarity(events, polarity=[1, -1])
        ```
    """
    events_lf = _ensure_lazy_frame(events)

    # Handle single polarity or list of polarities
    if isinstance(polarity, (int, float)):
        polarity_values = [int(polarity)]
    else:
        polarity_values = [int(p) for p in polarity]

    # Apply polarity filter
    filtered_lf = events_lf.filter(pl.col("polarity").is_in(polarity_values))

    return _apply_engine_hint(filtered_lf, engine)


def filter_hot_pixels(
    events: EventsInput,
    threshold_percentile: Optional[float] = None,
    engine: EngineType = "auto",
) -> pl.LazyFrame:
    """
    Remove hot pixels using statistical filtering

    Hot pixels are detected by identifying coordinates that generate significantly
    more events than typical pixels. This implementation uses percentile-based
    thresholding on per-pixel event counts.

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        threshold_percentile: Percentile threshold for hot pixel detection (default: 99.9)
        engine: Polars engine to use

    Returns:
        LazyFrame with hot pixel events removed

    Example:
        ```python
        from evlib.filtering import filter_hot_pixels

        # Remove hot pixels (top 0.1% most active pixels)
        filtered = filter_hot_pixels(events, threshold_percentile=99.9)

        # More aggressive filtering
        filtered = filter_hot_pixels(events, threshold_percentile=95.0)
        ```
    """
    events_lf = _ensure_lazy_frame(events)

    if threshold_percentile is None:
        threshold_percentile = 99.9

    # Calculate per-pixel event counts
    pixel_counts = events_lf.group_by(["x", "y"]).agg(pl.len().alias("event_count"))

    # Calculate threshold based on percentile
    threshold_expr = pixel_counts.select(
        pl.col("event_count").quantile(threshold_percentile / 100.0).alias("threshold")
    )

    # Get threshold value
    threshold_val = threshold_expr.collect().item(0, "threshold")

    # Filter out pixels above threshold
    hot_pixels = pixel_counts.filter(pl.col("event_count") > threshold_val).select(
        ["x", "y"]
    )

    # Anti-join to remove events from hot pixels
    filtered_lf = events_lf.join(hot_pixels, on=["x", "y"], how="anti")

    return _apply_engine_hint(filtered_lf, engine)


def filter_noise(
    events: EventsInput,
    method: Optional[str] = None,
    refractory_period_us: Optional[float] = None,
    engine: EngineType = "auto",
) -> pl.LazyFrame:
    """
    Remove noise events using temporal filtering

    Implements refractory period filtering to remove rapid-fire noise events
    that occur too quickly at the same pixel location.

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        method: Noise filtering method ("refractory" or "correlation", default: "refractory")
        refractory_period_us: Refractory period in microseconds (default: 1000)
        engine: Polars engine to use

    Returns:
        LazyFrame with noise events removed

    Example:
        ```python
        from evlib.filtering import filter_noise

        # Remove events that occur within 1ms of previous event at same pixel
        filtered = filter_noise(events, method="refractory", refractory_period_us=1000)

        # More aggressive filtering
        filtered = filter_noise(events, method="refractory", refractory_period_us=2000)
        ```
    """
    events_lf = _ensure_lazy_frame(events)

    if method is None:
        method = "refractory"

    if refractory_period_us is None:
        refractory_period_us = 1000.0

    if method == "refractory":
        # Sort events by time for temporal operations
        sorted_events = events_lf.sort("t")

        # Add previous timestamp for same pixel
        filtered_lf = (
            sorted_events.with_columns(
                [
                    # Convert duration to microseconds for arithmetic
                    pl.col("t").dt.total_microseconds().alias("t_us")
                ]
            )
            .with_columns(
                [
                    # Get previous timestamp for same pixel coordinate
                    pl.col("t_us").shift(1).over(["x", "y"]).alias("prev_t_us")
                ]
            )
            .with_columns(
                [
                    # Calculate time difference
                    (pl.col("t_us") - pl.col("prev_t_us")).alias("time_diff_us")
                ]
            )
            .filter(
                # Keep events if:
                # 1. First event at pixel (prev_t_us is null), or
                # 2. Time difference is greater than refractory period
                pl.col("prev_t_us").is_null()
                | (pl.col("time_diff_us") > refractory_period_us)
            )
            .drop(["t_us", "prev_t_us", "time_diff_us"])
        )

    elif method == "correlation":
        # Simple temporal correlation filtering
        # For now, implement as refractory with shorter period
        correlation_period = refractory_period_us * 0.5

        sorted_events = events_lf.sort("t")

        filtered_lf = (
            sorted_events.with_columns(
                [pl.col("t").dt.total_microseconds().alias("t_us")]
            )
            .with_columns([pl.col("t_us").shift(1).over(["x", "y"]).alias("prev_t_us")])
            .with_columns(
                [(pl.col("t_us") - pl.col("prev_t_us")).alias("time_diff_us")]
            )
            .filter(
                pl.col("prev_t_us").is_null()
                | (pl.col("time_diff_us") > correlation_period)
            )
            .drop(["t_us", "prev_t_us", "time_diff_us"])
        )

    else:
        raise ValueError(
            f"Unknown noise filtering method: {method}. Use 'refractory' or 'correlation'"
        )

    return _apply_engine_hint(filtered_lf, engine)


# Convenience functions for advanced filtering operations
def filter_multiple_rois(
    events: EventsInput, rois: List[tuple], engine: EngineType = "auto"
) -> pl.LazyFrame:
    """
    Filter events by multiple regions of interest

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        rois: List of (x_min, x_max, y_min, y_max) tuples
        engine: Polars engine to use

    Returns:
        LazyFrame with events from any of the specified ROIs
    """
    events_lf = _ensure_lazy_frame(events)

    if not rois:
        return events_lf

    # Build OR conditions for all ROIs
    roi_conditions = []
    for x_min, x_max, y_min, y_max in rois:
        roi_condition = pl.col("x").is_between(x_min, x_max) & pl.col("y").is_between(
            y_min, y_max
        )
        roi_conditions.append(roi_condition)

    # Combine with OR
    combined_condition = roi_conditions[0]
    for condition in roi_conditions[1:]:
        combined_condition = combined_condition | condition

    filtered_lf = events_lf.filter(combined_condition)

    return _apply_engine_hint(filtered_lf, engine)


def preprocess_events(
    events: EventsInput,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    roi: Optional[tuple] = None,
    polarity: Optional[Union[int, List[int]]] = None,
    remove_hot_pixels: bool = True,
    hot_pixel_threshold: float = 99.9,
    denoise: bool = True,
    refractory_period_us: float = 1000.0,
    engine: EngineType = "auto",
) -> pl.LazyFrame:
    """
    Complete event preprocessing pipeline with all filtering operations

    Args:
        events: LazyFrame or DataFrame with columns 't', 'x', 'y', 'polarity'
        t_start: Start time in seconds (optional)
        t_end: End time in seconds (optional)
        roi: Region of interest as (x_min, x_max, y_min, y_max) (optional)
        polarity: Polarity value(s) to keep (optional)
        remove_hot_pixels: Whether to remove hot pixels
        hot_pixel_threshold: Percentile threshold for hot pixel detection
        denoise: Whether to apply noise filtering
        refractory_period_us: Refractory period for noise filtering
        engine: Polars engine to use

    Returns:
        LazyFrame with fully preprocessed events

    Example:
        ```python
        from evlib.filtering import preprocess_events

        # Complete preprocessing pipeline
        filtered = preprocess_events(
            events,
            t_start=0.1, t_end=0.5,
            roi=(100, 500, 100, 400),
            polarity=1,
            remove_hot_pixels=True,
            denoise=True
        )
        ```
    """
    processed = events

    # Apply filters in optimal order

    # 1. Time filtering (reduces data early)
    if t_start is not None or t_end is not None:
        processed = filter_by_time(
            processed, t_start=t_start, t_end=t_end, engine=engine
        )

    # 2. Spatial filtering (further reduces data)
    if roi is not None:
        x_min, x_max, y_min, y_max = roi
        processed = filter_by_roi(
            processed, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, engine=engine
        )

    # 3. Polarity filtering (simple and fast)
    if polarity is not None:
        processed = filter_by_polarity(processed, polarity=polarity, engine=engine)

    # 4. Hot pixel removal (computationally intensive, do after spatial filtering)
    if remove_hot_pixels:
        processed = filter_hot_pixels(
            processed, threshold_percentile=hot_pixel_threshold, engine=engine
        )

    # 5. Noise filtering (most computationally intensive, do last)
    if denoise:
        processed = filter_noise(
            processed,
            method="refractory",
            refractory_period_us=refractory_period_us,
            engine=engine,
        )

    return processed


# Export all functions
__all__ = [
    "filter_by_time",
    "filter_by_roi",
    "filter_by_polarity",
    "filter_hot_pixels",
    "filter_noise",
    "filter_multiple_rois",
    "preprocess_events",
]
