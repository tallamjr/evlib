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
import evlib

# Load events and convert to DataFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Convert LazyFrame to DataFrame first

# Filter events between 0.1 and 0.5 seconds
time_filtered_df = evf.filter_by_time(df, t_start=0.1, t_end=0.5)
```

**Parameters:**
- `events` (DataFrame): Polars DataFrame containing event data
- `t_start` (float): Start time in seconds (None for no lower bound)
- `t_end` (float): End time in seconds (None for no upper bound)

**Returns:**
- `DataFrame`: Filtered events

### filter_by_roi

Filter events by spatial region of interest (ROI).

**Example Usage:**
```python
import evlib.filtering as evf
import evlib

# Load events and convert to DataFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Convert LazyFrame to DataFrame first

# Filter events in center region
filtered_df = evf.filter_by_roi(
    df,
    x_min=100, x_max=500,
    y_min=100, y_max=400
)
```

**Parameters:**
- `events` (DataFrame): Polars DataFrame containing event data
- `x_min` (int): Minimum x coordinate (inclusive)
- `x_max` (int): Maximum x coordinate (inclusive)
- `y_min` (int): Minimum y coordinate (inclusive)
- `y_max` (int): Maximum y coordinate (inclusive)

**Returns:**
- `DataFrame`: Spatially filtered events

### filter_by_polarity

Filter events by polarity.

**Example Usage:**
```python
import evlib.filtering as evf
import evlib

# Load events and convert to DataFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Convert LazyFrame to DataFrame first

# Keep only positive events
positive_df = evf.filter_by_polarity(df, polarity=1)

# Keep both positive and negative (for -1/1 encoding)
both_df = evf.filter_by_polarity(df, polarity=[-1, 1])
```

**Parameters:**
- `events` (DataFrame): Polars DataFrame containing event data
- `polarity` (int|list): Polarity value(s) to keep (None for all)

**Returns:**
- `DataFrame`: Polarity-filtered events

### filter_hot_pixels

Remove hot pixels based on event count statistics.

**Example Usage:**
```python
import evlib.filtering as evf
import evlib

# Load events and convert to DataFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Convert LazyFrame to DataFrame first

# Remove pixels with >99.9% of event counts
filtered_df = evf.filter_hot_pixels(df, threshold_percentile=99.9)

# More aggressive hot pixel removal
filtered_df = evf.filter_hot_pixels(df, threshold_percentile=99.0)
```

**Parameters:**
- `events` (DataFrame): Polars DataFrame containing event data
- `threshold_percentile` (float): Percentile threshold for detection (default: 99.9)

**Returns:**
- `DataFrame`: Events with hot pixels removed

### filter_noise

Remove noise events using temporal filtering.

**Example Usage:**
```python
import evlib.filtering as evf
import evlib

# Load events and convert to DataFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Convert LazyFrame to DataFrame first

# Remove events within 1ms refractory period per pixel
filtered_df = evf.filter_noise(
    df,
    method="refractory",
    refractory_period_us=1000
)
```

**Parameters:**
- `events` (DataFrame): Polars DataFrame containing event data
- `method` (str): Noise filtering method ("refractory")
- `refractory_period_us` (int): Refractory period in microseconds

**Returns:**
- `DataFrame`: Events with noise removed

## High-Level API

### Complete Filtering Pipeline

You can combine multiple filters in sequence to create a complete preprocessing pipeline:

**Example Usage:**
```python
import evlib.filtering as evf
import evlib

# Load events and convert to DataFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Convert LazyFrame to DataFrame first

# Apply filters in sequence to create preprocessing pipeline
filtered = evf.filter_by_time(df, t_start=0.1, t_end=0.5)
filtered = evf.filter_by_roi(filtered, x_min=100, x_max=500, y_min=100, y_max=400)
filtered = evf.filter_by_polarity(filtered, polarity=1)
filtered = evf.filter_hot_pixels(filtered, threshold_percentile=99.9)
processed = evf.filter_noise(filtered, method="refractory", refractory_period_us=1000)
```

**Processing Steps:**
1. Time filtering: Select events within time range
2. Spatial filtering: Select events within region of interest
3. Polarity filtering: Keep specified polarity values
4. Hot pixel removal: Remove pixels with excessive event counts
5. Noise filtering: Apply temporal noise reduction

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
| **Time filtering** | 400M+ events/s | Vectorized duration comparisons |
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

# Load events and convert to DataFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Convert LazyFrame to DataFrame first

# Apply filters in sequence
filtered = evf.filter_by_time(df, t_start=1.0, t_end=2.0)
filtered = evf.filter_by_roi(filtered, x_min=100, x_max=500, y_min=100, y_max=400)
filtered = evf.filter_by_polarity(filtered, polarity=1)

print(f"Filtered to {len(filtered):,} events")
```

### Advanced Preprocessing

```python
import evlib.filtering as evf
import evlib

# Load events and convert to DataFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Convert LazyFrame to DataFrame first

# Complete preprocessing with all filters
filtered = evf.filter_by_time(df, t_start=0.5, t_end=1.5)
filtered = evf.filter_by_roi(filtered, x_min=200, x_max=400, y_min=150, y_max=350)
filtered = evf.filter_by_polarity(filtered, polarity=1)
filtered = evf.filter_hot_pixels(filtered, threshold_percentile=99.5)
final_events = evf.filter_noise(filtered, method="refractory", refractory_period_us=500)
```

### Custom Filtering Pipeline

```python
import evlib.filtering as evf
import evlib
import polars as pl

# Load events as LazyFrame for initial filtering
events = evlib.load_events("data/slider_depth/events.txt")

# Custom filtering with Polars operations
custom_filtered_lf = events.filter(
    # Time range
    (pl.col("t").dt.total_microseconds() / 1_000_000 >= 0.1) &
    (pl.col("t").dt.total_microseconds() / 1_000_000 <= 0.5) &
    # Spatial bounds
    (pl.col("x") >= 100) & (pl.col("x") <= 500) &
    (pl.col("y") >= 100) & (pl.col("y") <= 400) &
    # Polarity
    (pl.col("polarity") == 1)
)

# Convert to DataFrame and combine with evlib filters
custom_df = custom_filtered_lf.collect()
clean_events = evf.filter_hot_pixels(custom_df, threshold_percentile=99.0)
```
