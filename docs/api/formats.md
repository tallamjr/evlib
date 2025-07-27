# Formats API Reference

The formats module provides reliable, tested functions for loading and saving event data in various file formats.

## Overview

```python
import evlib
```

The formats module supports:
- **Text files**: Space-separated event data with flexible column mapping
- **HDF5 files**: Hierarchical data format with perfect round-trip compatibility
- **Advanced filtering**: Time windows, spatial bounds, polarity selection

## Core Functions

### load_events

::: evlib.load_events

The primary function for loading event data with comprehensive filtering options.

**Example Usage:**
```python
# High-level API (recommended) - returns Polars LazyFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Time window filtering using evlib.filtering
import evlib.filtering as evf
filtered_events = evf.filter_by_time(evlib.load_events("data/slider_depth/events.txt"), t_start=0.0, t_end=1.0)

# Spatial filtering using preprocessing
processed_events = evf.preprocess_events("data/slider_depth/events.txt", roi=(100, 500, 100, 300))

# Polarity filtering using Polars
import polars as pl
events = evlib.load_events("data/slider_depth/events.txt")
positive_events = events.filter(pl.col('polarity') == 1)

# Data is sorted by timestamp by default when using evlib.load_events
```

**Parameters:**
- `path` (str): Path to the event file
- `t_start` (float, optional): Start time filter (inclusive)
- `t_end` (float, optional): End time filter (inclusive)
- `min_x` (int, optional): Minimum x coordinate (inclusive)
- `max_x` (int, optional): Maximum x coordinate (inclusive)
- `min_y` (int, optional): Minimum y coordinate (inclusive)
- `max_y` (int, optional): Maximum y coordinate (inclusive)
- `polarity` (int, optional): Polarity filter (1 for positive, -1 for negative)
- `sort` (bool): Sort events by timestamp after loading
- `x_col` (int, optional): Column index for x coordinate (0-based)
- `y_col` (int, optional): Column index for y coordinate (0-based)
- `t_col` (int, optional): Column index for timestamp (0-based)
- `p_col` (int, optional): Column index for polarity (0-based)
- `header_lines` (int): Number of header lines to skip

**Returns:**
- `dict`: Dictionary with keys ["x", "y", "timestamp", "polarity"] for Polars integration

### save_events_to_hdf5

::: evlib.save_events_to_hdf5

Save event data to HDF5 format with perfect round-trip compatibility.

**Example Usage:**
```python
import numpy as np

# Prepare event data with correct dtypes
xs = np.array([320, 321, 319], dtype=np.int64)
ys = np.array([240, 241, 239], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3], dtype=np.float64)  # in seconds
ps = np.array([1, -1, 1], dtype=np.int64)  # Use -1 for negative polarity

# Save events to HDF5
output_path = "output_example.h5"
evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, output_path)
print(f"Saved {len(xs)} events to {output_path}")

# Note: HDF5 round-trip currently has type compatibility limitations
# The save function converts to uint16/int8 internally for space efficiency
# To load saved HDF5 files, use the standard evlib.load_events() function
# which handles the type conversions automatically
```

### save_events_to_text

::: evlib.save_events_to_text

Save event data to text format.

**Example Usage:**
```python
import numpy as np

# Prepare event data
xs = np.array([320, 321, 319], dtype=np.int64)
ys = np.array([240, 241, 239], dtype=np.int64)
ts = np.array([0.0001, 0.0002, 0.0003], dtype=np.float64)  # in seconds
ps = np.array([1, 0, 1], dtype=np.int64)

# Save events to text file
evlib.formats.save_events_to_text(xs, ys, ts, ps, "output.txt")
```

## File Format Support

### Text Files

**Standard Format:**
```
# timestamp x y polarity
0.000100 320 240 1
0.000200 321 241 -1
0.000300 319 239 1
```

**Custom Column Mapping:**
```python
# For files with different column order, preprocessing is recommended
# x y polarity timestamp
# 320 240 1 0.000100
# Note: Use evlib.load_events with standard format for best compatibility
events = evlib.load_events("data/slider_depth/events.txt")  # Handles standard formats
df = events.collect()
# Access data: df['x'], df['y'], df['timestamp'], df['polarity']
```

### HDF5 Files

HDF5 files are automatically detected and loaded. The format supports:
- Perfect round-trip compatibility
- Efficient storage and loading
- Multiple dataset structures (auto-detected)

**Supported HDF5 Structures:**
```python
# Structure 1: datasets inside "events" group
# Example paths: /events/t, /events/x, /events/y, /events/p

# Structure 2: separate root datasets
# Example paths: /t, /x, /y, /p
# Alternative names: /timestamps, /x_pos, /y_pos, /polarity
# Short names: /ts, /xs, /ys, /ps
```

## Advanced Features

### Filtering Combinations

All filters can be combined:
```python
# Complex filtering example using high-level API
import evlib.filtering as evf
import polars as pl

# Use preprocessing pipeline for comprehensive filtering
processed_events = evf.preprocess_events(
    "data/slider_depth/events.txt",
    t_start=1.0, t_end=5.0,        # Time window
    roi=(100, 500, 100, 400),      # Spatial bounds (min_x, max_x, min_y, max_y)
    remove_hot_pixels=True
)
# Then filter by polarity using Polars
positive_events = processed_events.filter(pl.col('polarity') == 1)
df = positive_events.collect()
```

### Performance Optimization

```python
import evlib.filtering as evf

# For large files, use filtering for efficiency
filtered_events = evf.filter_by_time(
    evlib.load_events("data/slider_depth/events.txt"),
    t_start=0.1, t_end=1.0  # Only load events in time range
)
df = filtered_events.collect()
```

### Error Handling

```python
try:
    events = evlib.load_events("data/slider_depth/events.txt")
    df = events.collect()
except FileNotFoundError:
    print("File not found")
except OSError as e:
    print(f"Invalid file format: {e}")
```

## Performance Characteristics

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Text loading | ~1.2x Python | Comparable to pandas |
| HDF5 loading | ~0.9x Python | Slight overhead for filtering |
| Large files (>1M events) | ~1.5x faster | Memory efficiency advantage |
| Filtering | Very fast | Applied during loading, not post-processing |

## Best Practices

### File Format Selection
- **Text files**: Human-readable, easy debugging, cross-platform
- **HDF5 files**: Faster loading, smaller file size, metadata support

### Performance Tips
1. **Use HDF5 for large datasets** (>100k events)
2. **Apply filters during loading** rather than post-processing
3. **Use time window filtering** for memory efficiency with large files
4. **Sort only when necessary** (adds processing time)

### Error Prevention
1. **Validate file existence** before loading
2. **Check file format** matches expected structure
3. **Use try-catch blocks** for robust error handling
4. **Verify array shapes** after loading

## Migration from Other Libraries

### From other libraries
```python
# evlib provides a unified API for event data loading
import evlib
import evlib.filtering as evf

# Load events as Polars LazyFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Apply filtering
start_time = 0.1
end_time = 1.0
filtered_events = evf.filter_by_time(evlib.load_events("data/slider_depth/events.txt"), t_start=start_time, t_end=end_time)
df = filtered_events.collect()
```
