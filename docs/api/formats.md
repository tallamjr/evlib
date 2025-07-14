# Formats API Reference

The formats module provides reliable, tested functions for loading and saving event data in various file formats.

## Overview

```python
import evlib.formats
```

The formats module supports:
- **Text files**: Space-separated event data with flexible column mapping
- **HDF5 files**: Hierarchical data format with perfect round-trip compatibility
- **Advanced filtering**: Time windows, spatial bounds, polarity selection

## Core Functions

### load_events

::: evlib.formats.load_events

The primary function for loading event data with comprehensive filtering options.

**Example Usage:**
```python
# Basic loading
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

# Time window filtering
xs, ys, ts, ps = evlib.formats.load_events_filtered("data/slider_depth/events.txt", t_start=0.0,
    t_end=1.0
)

# Spatial filtering
xs, ys, ts, ps = evlib.formats.load_events_filtered("data/slider_depth/events.txt", min_x=100, max_x=500,
    min_y=100, max_y=300
)

# Polarity filtering
positive_events = evlib.formats.load_events_filtered("data/slider_depth/events.txt", polarity=1)
negative_events = evlib.formats.load_events_filtered("data/slider_depth/events.txt", polarity=-1)

# Custom column mapping
xs, ys, ts, ps = evlib.formats.load_events(
    "custom_format.txt",
    x_col=0, y_col=1, p_col=2, t_col=3
)

# Skip header lines
xs, ys, ts, ps = evlib.formats.load_events(
    "events_with_header.txt",
    header_lines=1
)
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
- `tuple`: (x_coordinates, y_coordinates, timestamps, polarities)

### save_events_to_hdf5

::: evlib.formats.save_events_to_hdf5

Save event data to HDF5 format with perfect round-trip compatibility.

**Example Usage:**
```python
# Save events to HDF5
evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, "output.h5")

# Verify round-trip compatibility
xs_loaded, ys_loaded, ts_loaded, ps_loaded = evlib.formats.load_events("output.h5")
assert np.array_equal(xs, xs_loaded)  # Perfect round-trip
```

### save_events_to_text

::: evlib.formats.save_events_to_text

Save event data to text format.

**Example Usage:**
```python
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
# For files with different column order
# x y polarity timestamp
# 320 240 1 0.000100
xs, ys, ts, ps = evlib.formats.load_events(
    "custom.txt",
    x_col=0, y_col=1, p_col=2, t_col=3
)
```

### HDF5 Files

HDF5 files are automatically detected and loaded. The format supports:
- Perfect round-trip compatibility
- Efficient storage and loading
- Multiple dataset structures (auto-detected)

**Supported HDF5 Structures:**
```python
# Structure 1: datasets inside "events" group
/events/t, /events/x, /events/y, /events/p

# Structure 2: separate root datasets
/t, /x, /y, /p
/timestamps, /x_pos, /y_pos, /polarity
/ts, /xs, /ys, /ps
```

## Advanced Features

### Filtering Combinations

All filters can be combined:
```python
# Complex filtering example
xs, ys, ts, ps = evlib.formats.load_events_filtered("data/slider_depth/events.txt", t_start=1.0, t_end=5.0,        # Time window
    min_x=100, max_x=500,          # Spatial bounds
    min_y=100, max_y=400,
    polarity=1,                    # Positive events only
    sort=True                      # Sort by timestamp
)
```

### Performance Optimization

```python
# For large files, use chunked loading
xs, ys, ts, ps = evlib.formats.load_events(
    "massive_dataset.txt",
    chunk_size=100000  # Process in 100k event chunks
)
```

### Error Handling

```python
try:
    xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
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

### From event_utils
```python
# Old: event_utils
import event_utils
events = event_utils.load_events("file.txt")

# New: evlib
import evlib.formats
xs, ys, ts, ps = evlib.formats.load_events("file.txt")
```

### From esim_py
```python
# Old: esim_py
import esim_py
events = esim_py.EventsIterator("file.txt")

# New: evlib (with filtering)
import evlib.formats
xs, ys, ts, ps = evlib.formats.load_events_filtered("file.txt", t_start=start_time,
    t_end=end_time
)
```
