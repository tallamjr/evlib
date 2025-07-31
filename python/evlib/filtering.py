"""Event filtering functionality for evlib.

This module provides high-level filtering functions that work with both file paths
and Polars LazyFrames.
"""

# Import from parent module
from . import filtering as _rust_filtering
from . import load_events


def filter_by_time(events_or_path, t_start=None, t_end=None):
    """Filter events by time range.

    Args:
        events_or_path: Either a file path (str) or a Polars LazyFrame
        t_start: Start time in seconds (None for no lower bound)
        t_end: End time in seconds (None for no upper bound)

    Returns:
        Polars LazyFrame with filtered events
    """
    if isinstance(events_or_path, str):
        events = load_events(events_or_path)
    else:
        events = events_or_path
    return _rust_filtering.filter_by_time(events, t_start, t_end)


def filter_by_roi(events_or_path, x_min=None, x_max=None, y_min=None, y_max=None):
    """Filter events by region of interest.

    Args:
        events_or_path: Either a file path (str) or a Polars LazyFrame
        x_min, x_max, y_min, y_max: ROI boundaries

    Returns:
        Polars LazyFrame with filtered events
    """
    if isinstance(events_or_path, str):
        events = load_events(events_or_path)
    else:
        events = events_or_path
    return _rust_filtering.filter_by_roi(events, x_min, x_max, y_min, y_max)


def filter_by_polarity(events_or_path, polarity):
    """Filter events by polarity.

    Args:
        events_or_path: Either a file path (str) or a Polars LazyFrame
        polarity: Polarity value to keep (1 or -1)

    Returns:
        Polars LazyFrame with filtered events
    """
    if isinstance(events_or_path, str):
        events = load_events(events_or_path)
    else:
        events = events_or_path
    return _rust_filtering.filter_by_polarity(events, polarity)


def filter_hot_pixels(events_or_path, threshold_percentile=99.9):
    """Filter hot pixels.

    Args:
        events_or_path: Either a file path (str) or a Polars LazyFrame
        threshold_percentile: Percentile threshold for hot pixel detection

    Returns:
        Polars LazyFrame with filtered events
    """
    if isinstance(events_or_path, str):
        events = load_events(events_or_path)
    else:
        events = events_or_path
    return _rust_filtering.filter_hot_pixels(events, threshold_percentile)


def filter_noise(events_or_path, method="refractory", refractory_period_us=1000, hot_pixel_threshold=99.9):
    """Filter noise events.

    Args:
        events_or_path: Either a file path (str) or a Polars LazyFrame
        method: Noise filtering method
        refractory_period_us: Refractory period in microseconds
        hot_pixel_threshold: Hot pixel threshold percentile

    Returns:
        Polars LazyFrame with filtered events
    """
    if isinstance(events_or_path, str):
        events = load_events(events_or_path)
    else:
        events = events_or_path
    return _rust_filtering.filter_noise(events, method, refractory_period_us, hot_pixel_threshold)


def preprocess_events(
    events_or_path,
    t_start=None,
    t_end=None,
    roi=None,
    polarity=None,
    remove_hot_pixels=False,
    remove_noise=False,
):
    """Complete preprocessing pipeline.

    Args:
        events_or_path: Either a file path (str) or a Polars LazyFrame
        t_start: Start time in seconds
        t_end: End time in seconds
        roi: Tuple of (x_min, x_max, y_min, y_max)
        polarity: Polarity to keep (1 or -1)
        remove_hot_pixels: Whether to remove hot pixels
        remove_noise: Whether to apply noise filtering

    Returns:
        Polars LazyFrame with filtered events
    """
    if isinstance(events_or_path, str):
        events = load_events(events_or_path)
    else:
        events = events_or_path

    # Apply filters in sequence
    if t_start is not None or t_end is not None:
        events = _rust_filtering.filter_by_time(events, t_start, t_end)

    if roi is not None:
        x_min, x_max, y_min, y_max = roi
        events = _rust_filtering.filter_by_roi(events, x_min, x_max, y_min, y_max)

    if polarity is not None:
        events = _rust_filtering.filter_by_polarity(events, polarity)

    if remove_hot_pixels:
        events = _rust_filtering.filter_hot_pixels(events)

    if remove_noise:
        events = _rust_filtering.filter_noise(events)

    return events


# Export all functions
__all__ = [
    "filter_by_time",
    "filter_by_roi",
    "filter_by_polarity",
    "filter_hot_pixels",
    "filter_noise",
    "preprocess_events",
]
