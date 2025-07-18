"""
Comprehensive event filtering and preprocessing module for evlib.

This module provides high-performance event filtering functionality using Polars
for efficient DataFrame operations. All functions support both file paths and
Polars LazyFrames as input, returning LazyFrames for optimal performance.

Key Features:
- Temporal filtering (time range selection)
- Spatial filtering (region of interest)
- Polarity filtering (positive/negative events)
- Hot pixel removal (statistical outlier detection)
- Noise filtering (refractory period, distance-based)
- High-level preprocessing pipeline
- Progress reporting for large operations
- Handles both 0/1 and -1/1 polarity encodings

Example Usage:
    >>> import evlib
    >>> import evlib.filtering as evf
    >>>
    >>> # Load and filter events
    >>> events = evlib.load_events("data/events.h5")
    >>> filtered = evf.filter_by_time(events, t_start=0.1, t_end=0.5)
    >>> spatial_filter = evf.filter_by_roi(filtered, x_min=100, x_max=500, y_min=100, y_max=400)
    >>>
    >>> # High-level preprocessing
    >>> processed = evf.preprocess_events(
    ...     "data/events.h5",
    ...     t_start=0.1, t_end=0.5,
    ...     roi=(100, 500, 100, 400),
    ...     polarity=1,
    ...     remove_hot_pixels=True,
    ...     remove_noise=True
    ... )
"""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import polars as pl


def _validate_events_input(events: Union[str, Path, pl.LazyFrame]) -> pl.LazyFrame:
    """
    Validate and convert input to LazyFrame.

    Args:
        events: Input events (path or LazyFrame)

    Returns:
        Validated LazyFrame

    Raises:
        ValueError: If input is invalid
        ImportError: If evlib is not available for loading
    """
    if isinstance(events, (str, Path)):
        try:
            import evlib

            return evlib.load_events(str(events))
        except ImportError:
            raise ImportError("evlib is required to load events from file paths")
    elif isinstance(events, pl.LazyFrame):
        return events
    else:
        raise ValueError(f"Invalid input type: {type(events)}. Expected str, Path, or pl.LazyFrame")


def _validate_polarity_encoding(events_lazy: pl.LazyFrame) -> Dict[str, Union[int, float]]:
    """
    Detect and validate polarity encoding (0/1 or -1/1).

    Args:
        events_lazy: LazyFrame containing events

    Returns:
        Dict with polarity encoding information
    """
    # Sample a subset to check polarity values
    sample_events = events_lazy.limit(10000).collect()

    if len(sample_events) == 0:
        return {"encoding": "unknown", "min_val": 0, "max_val": 1}

    unique_polarities = sample_events["polarity"].unique().sort()
    min_pol = unique_polarities.min()
    max_pol = unique_polarities.max()

    if min_pol >= 0 and max_pol <= 1:
        return {"encoding": "0_1", "min_val": 0, "max_val": 1}
    elif min_pol >= -1 and max_pol <= 1:
        return {"encoding": "-1_1", "min_val": -1, "max_val": 1}
    else:
        warnings.warn(f"Unknown polarity encoding: min={min_pol}, max={max_pol}")
        return {"encoding": "unknown", "min_val": min_pol, "max_val": max_pol}


def _report_progress(current: int, total: int, operation: str, start_time: float) -> None:
    """Report progress for long-running operations."""
    if total == 0:
        return

    progress = current / total
    if progress > 0:
        elapsed = time.time() - start_time
        eta = (elapsed / progress) - elapsed
        print(f"{operation}: {current}/{total} ({progress:.1%}) - ETA: {eta:.1f}s")


def filter_by_time(
    events: Union[str, Path, pl.LazyFrame],
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    **load_kwargs,
) -> pl.LazyFrame:
    """
    Filter events by time range.

    Args:
        events: Path to event file or LazyFrame
        t_start: Start time in seconds (None for no lower bound)
        t_end: End time in seconds (None for no upper bound)
        **load_kwargs: Additional arguments passed to evlib.load_events()

    Returns:
        LazyFrame with filtered events

    Example:
        >>> import evlib.filtering as evf
        >>> # Filter events between 0.1 and 0.5 seconds
        >>> filtered = evf.filter_by_time("data/events.h5", t_start=0.1, t_end=0.5)
        >>> print(f"Filtered events: {len(filtered.collect())}")
    """
    events_lazy = _validate_events_input(events)

    # Apply time filters
    conditions = []

    if t_start is not None:
        t_start_us = int(t_start * 1_000_000)  # Convert to microseconds
        conditions.append(pl.col("timestamp") >= pl.duration(microseconds=t_start_us))

    if t_end is not None:
        t_end_us = int(t_end * 1_000_000)  # Convert to microseconds
        conditions.append(pl.col("timestamp") <= pl.duration(microseconds=t_end_us))

    if conditions:
        # Combine conditions with AND
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition & condition

        events_lazy = events_lazy.filter(combined_condition)

    return events_lazy


def filter_by_roi(
    events: Union[str, Path, pl.LazyFrame], x_min: int, x_max: int, y_min: int, y_max: int, **load_kwargs
) -> pl.LazyFrame:
    """
    Filter events by spatial region of interest (ROI).

    Args:
        events: Path to event file or LazyFrame
        x_min: Minimum x coordinate (inclusive)
        x_max: Maximum x coordinate (inclusive)
        y_min: Minimum y coordinate (inclusive)
        y_max: Maximum y coordinate (inclusive)
        **load_kwargs: Additional arguments passed to evlib.load_events()

    Returns:
        LazyFrame with spatially filtered events

    Raises:
        ValueError: If ROI bounds are invalid

    Example:
        >>> import evlib.filtering as evf
        >>> # Filter events in center 400x300 region
        >>> filtered = evf.filter_by_roi(
        ...     "data/events.h5",
        ...     x_min=100, x_max=500,
        ...     y_min=100, y_max=400
        ... )
    """
    # Validate ROI bounds
    if x_min >= x_max or y_min >= y_max:
        raise ValueError(f"Invalid ROI bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

    events_lazy = _validate_events_input(events)

    # Apply spatial filters
    spatial_filter = (
        (pl.col("x") >= x_min) & (pl.col("x") <= x_max) & (pl.col("y") >= y_min) & (pl.col("y") <= y_max)
    )

    return events_lazy.filter(spatial_filter)


def filter_by_polarity(
    events: Union[str, Path, pl.LazyFrame], polarity: Optional[Union[int, List[int]]] = None, **load_kwargs
) -> pl.LazyFrame:
    """
    Filter events by polarity.

    Args:
        events: Path to event file or LazyFrame
        polarity: Polarity value(s) to keep (None for all polarities)
                 Can be single value or list of values
                 Supports both 0/1 and -1/1 encodings
        **load_kwargs: Additional arguments passed to evlib.load_events()

    Returns:
        LazyFrame with polarity-filtered events

    Example:
        >>> import evlib.filtering as evf
        >>> # Keep only positive events
        >>> positive = evf.filter_by_polarity("data/events.h5", polarity=1)
        >>> # Keep both positive and negative (for -1/1 encoding)
        >>> both = evf.filter_by_polarity("data/events.h5", polarity=[-1, 1])
    """
    if polarity is None:
        return _validate_events_input(events)

    events_lazy = _validate_events_input(events)

    # Convert single polarity to list
    if isinstance(polarity, int):
        polarity = [polarity]

    # Apply polarity filter
    polarity_filter = pl.col("polarity").is_in(polarity)

    return events_lazy.filter(polarity_filter)


def filter_hot_pixels(
    events: Union[str, Path, pl.LazyFrame], threshold_percentile: float = 99.9, **load_kwargs
) -> pl.LazyFrame:
    """
    Remove hot pixels based on event count statistics.

    Hot pixels are identified as spatial locations (x,y) that generate
    significantly more events than typical pixels. This function groups
    events by spatial location, counts events per pixel, and removes
    pixels above the specified percentile threshold.

    Args:
        events: Path to event file or LazyFrame
        threshold_percentile: Percentile threshold for hot pixel detection
                             (99.9 means remove pixels with >99.9% event counts)
        **load_kwargs: Additional arguments passed to evlib.load_events()

    Returns:
        LazyFrame with hot pixels removed

    Example:
        >>> import evlib.filtering as evf
        >>> # Remove pixels with >99.9% of event counts
        >>> filtered = evf.filter_hot_pixels("data/events.h5", threshold_percentile=99.9)
        >>> # More aggressive hot pixel removal
        >>> filtered = evf.filter_hot_pixels("data/events.h5", threshold_percentile=99.0)
    """
    events_lazy = _validate_events_input(events)

    print(f"Detecting hot pixels (threshold: {threshold_percentile}th percentile)...")
    start_time = time.time()

    # Group by spatial coordinates and count events
    pixel_counts = events_lazy.group_by(["x", "y"]).agg(pl.len().alias("event_count")).collect()

    if len(pixel_counts) == 0:
        print("No events found for hot pixel detection")
        return events_lazy

    # Calculate threshold
    threshold_count = pixel_counts["event_count"].quantile(threshold_percentile / 100.0)

    # Identify hot pixels
    hot_pixels = pixel_counts.filter(pl.col("event_count") > threshold_count)
    num_hot_pixels = len(hot_pixels)

    if num_hot_pixels == 0:
        print(f"No hot pixels detected (threshold: {threshold_count:.0f} events)")
        return events_lazy

    print(f"Found {num_hot_pixels} hot pixels (threshold: {threshold_count:.0f} events)")

    # Create filter to exclude hot pixels
    hot_pixel_coords = hot_pixels.select(["x", "y"])

    # Use anti-join to remove events from hot pixels
    filtered_events = events_lazy.collect().join(hot_pixel_coords, on=["x", "y"], how="anti").lazy()

    elapsed_time = time.time() - start_time
    print(f"Success: Hot pixel filtering completed in {elapsed_time:.2f}s")

    return filtered_events


def filter_noise(
    events: Union[str, Path, pl.LazyFrame],
    method: str = "refractory",
    refractory_period_us: int = 1000,
    distance_threshold: Optional[int] = None,
    **load_kwargs,
) -> pl.LazyFrame:
    """
    Remove noise events using temporal or spatial filtering.

    Args:
        events: Path to event file or LazyFrame
        method: Noise filtering method ("refractory", "distance")
        refractory_period_us: Refractory period in microseconds for temporal filtering
        distance_threshold: Distance threshold for spatial filtering (not implemented yet)
        **load_kwargs: Additional arguments passed to evlib.load_events()

    Returns:
        LazyFrame with noise events removed

    Example:
        >>> import evlib.filtering as evf
        >>> # Remove events within 1ms refractory period per pixel
        >>> filtered = evf.filter_noise("data/events.h5", method="refractory", refractory_period_us=1000)
    """
    events_lazy = _validate_events_input(events)

    if method == "refractory":
        return _apply_refractory_filter(events_lazy, refractory_period_us)
    elif method == "distance":
        raise NotImplementedError("Distance-based noise filtering not yet implemented")
    else:
        raise ValueError(f"Unknown noise filtering method: {method}")


def _apply_refractory_filter(events_lazy: pl.LazyFrame, refractory_period_us: int) -> pl.LazyFrame:
    """
    Apply refractory period filtering to remove temporal noise.

    For each pixel, events occurring within the refractory period
    of the previous event are considered noise and removed.

    Args:
        events_lazy: LazyFrame containing events
        refractory_period_us: Refractory period in microseconds

    Returns:
        LazyFrame with refractory filtering applied
    """
    print(f"Applying refractory period filter ({refractory_period_us} μs)...")
    start_time = time.time()

    # Collect and sort events by pixel and timestamp
    events_df = events_lazy.collect()

    if len(events_df) == 0:
        print("No events found for refractory filtering")
        return events_lazy

    # Convert timestamp to microseconds for comparison
    events_df = events_df.with_columns([pl.col("timestamp").dt.total_microseconds().alias("timestamp_us")])

    # Sort by pixel coordinates and timestamp
    events_sorted = events_df.sort(["x", "y", "timestamp_us"])

    # Calculate time differences within each pixel
    events_with_diff = events_sorted.with_columns(
        [
            # Calculate time difference from previous event in same pixel
            (pl.col("timestamp_us") - pl.col("timestamp_us").shift(1))
            .over(["x", "y"])
            .alias("time_diff")
        ]
    )

    # Filter events: keep if no previous event or time difference > refractory period
    filtered_events = events_with_diff.filter(
        pl.col("time_diff").is_null() | (pl.col("time_diff") > refractory_period_us)
    )

    # Remove the temporary columns
    result = filtered_events.drop(["timestamp_us", "time_diff"]).lazy()

    elapsed_time = time.time() - start_time
    original_count = len(events_df)
    filtered_count = len(filtered_events)
    removed_count = original_count - filtered_count

    print(f"Success: Refractory filtering completed in {elapsed_time:.2f}s")
    print(f"Success: Removed {removed_count:,} events ({removed_count/original_count:.1%})")

    return result


def preprocess_events(
    events: Union[str, Path, pl.LazyFrame],
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    polarity: Optional[Union[int, List[int]]] = None,
    remove_hot_pixels: bool = True,
    remove_noise: bool = True,
    hot_pixel_threshold: float = 99.9,
    refractory_period_us: int = 1000,
    **kwargs,
) -> pl.LazyFrame:
    """
    High-level event preprocessing pipeline.

    Applies multiple filtering operations in sequence:
    1. Time filtering (if specified)
    2. Spatial filtering (if ROI specified)
    3. Polarity filtering (if specified)
    4. Hot pixel removal (if enabled)
    5. Noise filtering (if enabled)

    Args:
        events: Path to event file or LazyFrame
        t_start: Start time in seconds (None for no lower bound)
        t_end: End time in seconds (None for no upper bound)
        roi: Region of interest as (x_min, x_max, y_min, y_max)
        polarity: Polarity value(s) to keep (None for all)
        remove_hot_pixels: Whether to remove hot pixels
        remove_noise: Whether to apply noise filtering
        hot_pixel_threshold: Percentile threshold for hot pixel detection
        refractory_period_us: Refractory period in microseconds
        **kwargs: Additional arguments passed to individual filters

    Returns:
        LazyFrame with preprocessed events

    Example:
        >>> import evlib.filtering as evf
        >>> # Complete preprocessing pipeline
        >>> processed = evf.preprocess_events(
        ...     "data/events.h5",
        ...     t_start=0.1, t_end=0.5,
        ...     roi=(100, 500, 100, 400),
        ...     polarity=1,
        ...     remove_hot_pixels=True,
        ...     remove_noise=True
        ... )
        >>> print(f"Processed events: {len(processed.collect())}")
    """
    print("=== Event Preprocessing Pipeline ===")
    start_time = time.time()

    # Start with input validation
    events_lazy = _validate_events_input(events)

    # Count initial events
    initial_count = len(events_lazy.collect())
    print(f"Initial events: {initial_count:,}")
    current_events = events_lazy

    # 1. Time filtering
    if t_start is not None or t_end is not None:
        print(f"Applying time filter: {t_start}s to {t_end}s")
        current_events = filter_by_time(current_events, t_start=t_start, t_end=t_end)
        time_filtered_count = len(current_events.collect())
        print(f"After time filtering: {time_filtered_count:,} events")

    # 2. Spatial filtering (ROI)
    if roi is not None:
        x_min, x_max, y_min, y_max = roi
        print(f"Applying ROI filter: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")
        current_events = filter_by_roi(current_events, x_min, x_max, y_min, y_max)
        roi_filtered_count = len(current_events.collect())
        print(f"After ROI filtering: {roi_filtered_count:,} events")

    # 3. Polarity filtering
    if polarity is not None:
        print(f"Applying polarity filter: {polarity}")
        current_events = filter_by_polarity(current_events, polarity=polarity)
        polarity_filtered_count = len(current_events.collect())
        print(f"After polarity filtering: {polarity_filtered_count:,} events")

    # 4. Hot pixel removal
    if remove_hot_pixels:
        print("Applying hot pixel removal...")
        current_events = filter_hot_pixels(current_events, threshold_percentile=hot_pixel_threshold)
        hot_pixel_filtered_count = len(current_events.collect())
        print(f"After hot pixel removal: {hot_pixel_filtered_count:,} events")

    # 5. Noise filtering
    if remove_noise:
        print("Applying noise filtering...")
        current_events = filter_noise(current_events, refractory_period_us=refractory_period_us)
        noise_filtered_count = len(current_events.collect())
        print(f"After noise filtering: {noise_filtered_count:,} events")

    # Final summary
    final_count = len(current_events.collect())
    total_time = time.time() - start_time
    reduction_ratio = (initial_count - final_count) / initial_count if initial_count > 0 else 0

    print("=== Preprocessing Complete ===")
    print(f"Final events: {final_count:,} ({reduction_ratio:.1%} reduction)")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Processing rate: {initial_count/total_time:.0f} events/s")

    return current_events


# Export the main API
__all__ = [
    "filter_by_time",
    "filter_by_roi",
    "filter_by_polarity",
    "filter_hot_pixels",
    "filter_noise",
    "preprocess_events",
]


if __name__ == "__main__":
    # Example usage and testing
    print("evlib.filtering - Comprehensive Event Filtering Module")
    print("=" * 50)
    print()
    print("Available functions:")
    for func_name in __all__:
        print(f"  • {func_name}")
    print()
    print("Example usage:")
    print("  import evlib.filtering as evf")
    print("  # Individual filters")
    print("  filtered = evf.filter_by_time('events.h5', t_start=0.1, t_end=0.5)")
    print("  spatial = evf.filter_by_roi(filtered, x_min=100, x_max=500, y_min=100, y_max=400)")
    print("  clean = evf.filter_hot_pixels(spatial, threshold_percentile=99.9)")
    print()
    print("  # Complete preprocessing pipeline")
    print("  processed = evf.preprocess_events(")
    print("      'events.h5',")
    print("      t_start=0.1, t_end=0.5,")
    print("      roi=(100, 500, 100, 400),")
    print("      polarity=1,")
    print("      remove_hot_pixels=True,")
    print("      remove_noise=True")
    print("  )")
