# Filtering API Reference

The filtering module provides comprehensive event filtering functionality using Polars for high-performance DataFrame operations.

## Overview

```python
import evlib.filtering as evf
```

Event filtering is essential for:
- **Temporal analysis**: Selecting events within specific time windows
- **Spatial analysis**: Filtering events by region of interest
- **Noise removal**: Eliminating hot pixels and temporal noise
- **Data preprocessing**: Preparing clean datasets for analysis

## Core Functions

### filter_by_time

Filter events by time range.

**Example Usage:**
```python
import evlib.filtering as evf

# Filter events between 0.1 and 0.5 seconds
filtered = evf.filter_by_time("data/slider_depth/events.txt", t_start=0.1, t_end=0.5)

# Filter with LazyFrame
import evlib
lf = evlib.load_events("data/slider_depth/events.txt")
time_filtered = evf.filter_by_time(lf, t_start=0.1, t_end=0.5)
```

**Parameters:**
- `events` (str|LazyFrame): Path to event file or Polars LazyFrame
- `t_start` (float): Start time in seconds (None for no lower bound)
- `t_end` (float): End time in seconds (None for no upper bound)

**Returns:**
- `LazyFrame`: Filtered events

### filter_by_roi

Filter events by spatial region of interest (ROI).

**Example Usage:**
```python
import evlib.filtering as evf

# Filter events in center region
filtered = evf.filter_by_roi(
    "data/slider_depth/events.txt",
    x_min=100, x_max=500,
    y_min=100, y_max=400
)
```

**Parameters:**
- `events` (str|LazyFrame): Path to event file or Polars LazyFrame
- `x_min` (int): Minimum x coordinate (inclusive)
- `x_max` (int): Maximum x coordinate (inclusive)
- `y_min` (int): Minimum y coordinate (inclusive)
- `y_max` (int): Maximum y coordinate (inclusive)

**Returns:**
- `LazyFrame`: Spatially filtered events

### filter_by_polarity

Filter events by polarity.

**Example Usage:**
```python
import evlib.filtering as evf

# Keep only positive events
positive = evf.filter_by_polarity("data/slider_depth/events.txt", polarity=1)

# Keep both positive and negative (for -1/1 encoding)
both = evf.filter_by_polarity("data/slider_depth/events.txt", polarity=[-1, 1])
```

**Parameters:**
- `events` (str|LazyFrame): Path to event file or Polars LazyFrame
- `polarity` (int|list): Polarity value(s) to keep (None for all)

**Returns:**
- `LazyFrame`: Polarity-filtered events

### filter_hot_pixels

Remove hot pixels based on event count statistics.

**Example Usage:**
```python
import evlib.filtering as evf

# Remove pixels with >99.9% of event counts
filtered = evf.filter_hot_pixels("data/slider_depth/events.txt", threshold_percentile=99.9)

# More aggressive hot pixel removal
filtered = evf.filter_hot_pixels("data/slider_depth/events.txt", threshold_percentile=99.0)
```

**Parameters:**
- `events` (str|LazyFrame): Path to event file or Polars LazyFrame
- `threshold_percentile` (float): Percentile threshold for detection (default: 99.9)

**Returns:**
- `LazyFrame`: Events with hot pixels removed

### filter_noise

Remove noise events using temporal filtering.

**Example Usage:**
```python
import evlib.filtering as evf

# Remove events within 1ms refractory period per pixel
filtered = evf.filter_noise(
    "data/slider_depth/events.txt",
    method="refractory",
    refractory_period_us=1000
)
```

**Parameters:**
- `events` (str|LazyFrame): Path to event file or Polars LazyFrame
- `method` (str): Noise filtering method ("refractory")
- `refractory_period_us` (int): Refractory period in microseconds

**Returns:**
- `LazyFrame`: Events with noise removed

## High-Level API

### preprocess_events

High-level event preprocessing pipeline combining multiple filters.

**Example Usage:**
```python
import evlib.filtering as evf

# Complete preprocessing pipeline
processed = evf.preprocess_events(
    "data/slider_depth/events.txt",
    t_start=0.1, t_end=0.5,
    roi=(100, 500, 100, 400),
    polarity=1,
    remove_hot_pixels=True,
    remove_noise=True,
    hot_pixel_threshold=99.9,
    refractory_period_us=1000
)
```

**Processing Steps:**
1. Time filtering (if specified)
2. Spatial filtering (if ROI specified)
3. Polarity filtering (if specified)
4. Hot pixel removal (if enabled)
5. Noise filtering (if enabled)

**Parameters:**
- `events` (str|LazyFrame): Path to event file or Polars LazyFrame
- `t_start` (float): Start time in seconds (optional)
- `t_end` (float): End time in seconds (optional)
- `roi` (tuple): Region of interest as (x_min, x_max, y_min, y_max) (optional)
- `polarity` (int|list): Polarity value(s) to keep (optional)
- `remove_hot_pixels` (bool): Whether to remove hot pixels (default: True)
- `remove_noise` (bool): Whether to apply noise filtering (default: True)
- `hot_pixel_threshold` (float): Percentile threshold for hot pixels (default: 99.9)
- `refractory_period_us` (int): Refractory period in microseconds (default: 1000)

**Returns:**
- `LazyFrame`: Preprocessed events

## Performance Features

### Polars Integration

All filtering functions leverage Polars for high-performance operations:

- **Lazy evaluation**: Operations are optimized before execution
- **Vectorized processing**: SIMD-optimized operations
- **Memory efficiency**: Minimal memory allocations
- **Scalability**: Handles large datasets efficiently

### Performance Characteristics

| Operation | Performance | Notes |
|-----------|-------------|-------|
| **Time filtering** | 400M+ events/s | Vectorized timestamp comparisons |
| **Spatial filtering** | 350M+ events/s | Efficient coordinate bounds checking |
| **Polarity filtering** | 450M+ events/s | Simple integer comparisons |
| **Hot pixel detection** | Variable | Depends on dataset size and distribution |
| **Noise filtering** | 10M+ events/s | Requires sorting and grouping operations |

### Memory Efficiency

- **Streaming support**: Large files processed in chunks
- **Lazy operations**: Memory usage optimized through query planning
- **Progress reporting**: Real-time feedback for long operations
- **Error handling**: Graceful handling of edge cases

## Examples

### Basic Filtering Chain

```python
import evlib.filtering as evf
import evlib

# Load events
lf = evlib.load_events("data/slider_depth/events.txt")

# Apply filters in sequence
filtered = evf.filter_by_time(lf, t_start=1.0, t_end=2.0)
filtered = evf.filter_by_roi(filtered, x_min=100, x_max=500, y_min=100, y_max=400)
filtered = evf.filter_by_polarity(filtered, polarity=1)

# Collect results
df = filtered.collect()
print(f"Filtered to {len(df):,} events")
```

### Advanced Preprocessing

```python
import evlib.filtering as evf

# Complete preprocessing with all filters
processed = evf.preprocess_events(
    "data/slider_depth/events.txt",
    t_start=0.5, t_end=1.5,
    roi=(200, 400, 150, 350),
    polarity=1,
    remove_hot_pixels=True,
    remove_noise=True,
    hot_pixel_threshold=99.5,
    refractory_period_us=500
)

# Get final results
final_events = processed.collect()
```

### Custom Filtering Pipeline

```python
import evlib.filtering as evf
import evlib
import polars as pl

# Load events
lf = evlib.load_events("data/slider_depth/events.txt")

# Custom filtering with Polars operations
custom_filtered = lf.filter(
    # Time range
    (pl.col("timestamp").dt.total_microseconds() / 1_000_000 >= 0.1) &
    (pl.col("timestamp").dt.total_microseconds() / 1_000_000 <= 0.5) &
    # Spatial bounds
    (pl.col("x") >= 100) & (pl.col("x") <= 500) &
    (pl.col("y") >= 100) & (pl.col("y") <= 400) &
    # Polarity
    (pl.col("polarity") == 1)
)

# Combine with evlib filters
clean_events = evf.filter_hot_pixels(custom_filtered, threshold_percentile=99.0)
```
