"""
Event representations for event camera data.

This module provides functions to create various tensor representations
from event streams, including stacked histograms, voxel grids, and time surfaces.
"""

import numpy as np
from typing import Tuple, Optional, Union


def stacked_histogram(
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    timestamp: np.ndarray,
    bins: int,
    height: int,
    width: int,
    count_cutoff: Optional[int] = None,
    fastmode: bool = True,
) -> np.ndarray:
    """
    Create a stacked histogram representation of event data.
    
    Based on the implementation from the RVT repository, this function creates
    a temporal histogram by dividing the event stream into time bins and
    accumulating events spatially within each bin.
    
    Args:
        x: X coordinates of events (shape: [N])
        y: Y coordinates of events (shape: [N])
        polarity: Polarity of events, should be in {0, 1} (shape: [N])
        timestamp: Timestamps of events (shape: [N])
        bins: Number of temporal bins
        height: Height of the sensor
        width: Width of the sensor
        count_cutoff: Maximum count per bin (default: 255)
        fastmode: If True, use uint8 directly (may overflow). If False, use int16 and clip.
        
    Returns:
        Stacked histogram representation of shape [2*bins, height, width]
        where the first `bins` channels correspond to positive events and
        the next `bins` channels correspond to negative events.
    """
    assert bins >= 1, "Number of bins must be at least 1"
    assert height >= 1, "Height must be at least 1"
    assert width >= 1, "Width must be at least 1"
    assert len(x) == len(y) == len(polarity) == len(timestamp), "All arrays must have same length"
    
    # Set default count cutoff
    if count_cutoff is None:
        count_cutoff = 255
    else:
        assert count_cutoff >= 1, "Count cutoff must be at least 1"
        count_cutoff = min(count_cutoff, 255)
    
    # Choose dtype based on fastmode
    dtype = np.uint8 if fastmode else np.int16
    
    # Initialize representation array: [channels, bins, height, width]
    channels = 2  # positive and negative events
    representation = np.zeros((channels, bins, height, width), dtype=dtype)
    
    # Handle empty events case
    if len(x) == 0:
        return representation.reshape((-1, height, width)).astype(np.uint8)
    
    # Convert polarity to ensure it's in {0, 1}
    polarity = np.asarray(polarity, dtype=np.int32)
    assert polarity.min() >= 0 and polarity.max() <= 1, "Polarity must be in {0, 1}"
    
    # Calculate time bins
    t_min = timestamp[0]
    t_max = timestamp[-1]
    assert t_max >= t_min, "Timestamps must be sorted"
    
    # Normalize timestamps to [0, 1] range
    t_norm = (timestamp - t_min) / max((t_max - t_min), 1e-9)
    
    # Scale to bin range and clamp
    t_scaled = t_norm * bins
    t_idx = np.floor(t_scaled).astype(np.int32)
    t_idx = np.clip(t_idx, 0, bins - 1)
    
    # Convert coordinates to arrays
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)
    
    # Bounds checking
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid_mask]
    y = y[valid_mask]
    polarity = polarity[valid_mask]
    t_idx = t_idx[valid_mask]
    
    # Create linear indices for accumulation
    # Index calculation: x + width * y + height * width * t_idx + bins * height * width * polarity
    indices = (
        x +
        width * y +
        height * width * t_idx +
        bins * height * width * polarity
    )
    
    # Accumulate events
    values = np.ones(len(indices), dtype=dtype)
    np.add.at(representation.ravel(), indices, values)
    
    # Apply count cutoff
    representation = np.clip(representation, 0, count_cutoff)
    
    # Convert to uint8 if not in fastmode
    if not fastmode:
        representation = representation.astype(np.uint8)
    
    # Reshape to merge channels and bins: [2*bins, height, width]
    return representation.reshape((-1, height, width))


def create_voxel_grid(
    x: np.ndarray,
    y: np.ndarray,
    timestamp: np.ndarray,
    polarity: np.ndarray,
    sensor_resolution: Tuple[int, int],
    num_bins: int,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> np.ndarray:
    """
    Create a voxel grid representation from event data.
    
    Args:
        x: X coordinates of events
        y: Y coordinates of events
        timestamp: Timestamps of events
        polarity: Polarity of events (should be in {0, 1} or {-1, 1})
        sensor_resolution: (width, height) of the sensor
        num_bins: Number of temporal bins
        t_min: Minimum timestamp (default: min of timestamp)
        t_max: Maximum timestamp (default: max of timestamp)
        
    Returns:
        Voxel grid of shape [width, height, num_bins]
    """
    width, height = sensor_resolution
    
    if len(x) == 0:
        return np.zeros((width, height, num_bins), dtype=np.float32)
    
    # Set time bounds
    if t_min is None:
        t_min = timestamp.min()
    if t_max is None:
        t_max = timestamp.max()
    
    # Normalize timestamps to [0, 1]
    t_norm = (timestamp - t_min) / max((t_max - t_min), 1e-9)
    
    # Calculate bin indices
    t_scaled = t_norm * num_bins
    t_idx = np.floor(t_scaled).astype(np.int32)
    t_idx = np.clip(t_idx, 0, num_bins - 1)
    
    # Convert polarity to {-1, 1} if needed
    if polarity.min() >= 0:
        polarity = 2 * polarity - 1
    
    # Initialize voxel grid
    voxel_grid = np.zeros((width, height, num_bins), dtype=np.float32)
    
    # Bounds checking
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    t_valid = t_idx[valid_mask]
    p_valid = polarity[valid_mask]
    
    # Accumulate events
    for i in range(len(x_valid)):
        voxel_grid[x_valid[i], y_valid[i], t_valid[i]] += p_valid[i]
    
    return voxel_grid


def create_time_surface(
    x: np.ndarray,
    y: np.ndarray,
    timestamp: np.ndarray,
    polarity: np.ndarray,
    sensor_resolution: Tuple[int, int],
    decay_constant: float = 0.05,
    polarity_separate: bool = False,
) -> np.ndarray:
    """
    Create a time surface representation with exponential decay.
    
    Args:
        x: X coordinates of events
        y: Y coordinates of events
        timestamp: Timestamps of events
        polarity: Polarity of events (should be in {0, 1})
        sensor_resolution: (width, height) of the sensor
        decay_constant: Decay constant for exponential decay
        polarity_separate: If True, create separate surfaces for each polarity
        
    Returns:
        Time surface of shape [height, width] or [2, height, width] if polarity_separate
    """
    width, height = sensor_resolution
    
    if len(x) == 0:
        shape = (2, height, width) if polarity_separate else (height, width)
        return np.zeros(shape, dtype=np.float32)
    
    # Get reference time (latest timestamp)
    t_ref = timestamp.max()
    
    if polarity_separate:
        # Create separate surfaces for positive and negative events
        time_surface = np.zeros((2, height, width), dtype=np.float32)
        
        # Bounds checking
        valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        t_valid = timestamp[valid_mask]
        p_valid = polarity[valid_mask]
        
        # Process each event
        for i in range(len(x_valid)):
            decay_value = np.exp(-(t_ref - t_valid[i]) / decay_constant)
            channel = int(p_valid[i])  # 0 for negative, 1 for positive
            time_surface[channel, y_valid[i], x_valid[i]] = decay_value
    else:
        # Single surface for all events
        time_surface = np.zeros((height, width), dtype=np.float32)
        
        # Bounds checking
        valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        t_valid = timestamp[valid_mask]
        
        # Process each event
        for i in range(len(x_valid)):
            decay_value = np.exp(-(t_ref - t_valid[i]) / decay_constant)
            time_surface[y_valid[i], x_valid[i]] = decay_value
    
    return time_surface


def create_event_histogram(
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    sensor_resolution: Tuple[int, int],
    polarity_separate: bool = False,
) -> np.ndarray:
    """
    Create a spatial histogram of events.
    
    Args:
        x: X coordinates of events
        y: Y coordinates of events
        polarity: Polarity of events (should be in {0, 1})
        sensor_resolution: (width, height) of the sensor
        polarity_separate: If True, create separate histograms for each polarity
        
    Returns:
        Event histogram of shape [height, width] or [2, height, width] if polarity_separate
    """
    width, height = sensor_resolution
    
    if len(x) == 0:
        shape = (2, height, width) if polarity_separate else (height, width)
        return np.zeros(shape, dtype=np.int32)
    
    # Bounds checking
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    p_valid = polarity[valid_mask]
    
    if polarity_separate:
        # Create separate histograms for positive and negative events
        histogram = np.zeros((2, height, width), dtype=np.int32)
        
        # Accumulate events by polarity
        for i in range(len(x_valid)):
            channel = int(p_valid[i])  # 0 for negative, 1 for positive
            histogram[channel, y_valid[i], x_valid[i]] += 1
    else:
        # Single histogram for all events
        histogram = np.zeros((height, width), dtype=np.int32)
        
        # Accumulate all events
        for i in range(len(x_valid)):
            histogram[y_valid[i], x_valid[i]] += 1
    
    return histogram


# Backwards compatibility alias
smooth_voxel = create_voxel_grid