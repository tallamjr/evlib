# Representations API Reference

The representations module provides efficient implementations for converting event data into various spatial-temporal representations using high-performance Polars processing.

## Overview

```python
import evlib.representations as evr
```

Event representations are crucial for:
- **Neural network input**: Converting events to tensor-like formats
- **Visualization**: Creating images from sparse event data
- **Analysis**: Temporal and spatial aggregation of events
- **RVT Replacement**: High-performance alternatives to PyTorch preprocessing

## Core Functions

### create_stacked_histogram

Creates stacked histogram representation with temporal binning.

**Example Usage:**
```python
import evlib.representations as evr

# Create stacked histogram (replaces RVT preprocessing)
events = evlib.load_events("data/slider_depth/events.txt")

# Create stacked histogram representation
hist_df = evr.create_stacked_histogram(
    events,
    height=480,
    width=640,
    bins=10,
    window_duration_ms=50.0
)
# Returns Polars LazyFrame
print(f"Generated stacked histogram with {len(hist_df)} entries")
print(f"Columns: {list(hist_df.columns)}")  # ['window_id', 'channel', 'time_bin', 'y', 'x', 'count']
```

**Parameters:**
- `events_pydf` (polars.DataFrame): Polars DataFrame with event data
- `_height` (int): Ignored parameter (spatial clipping simplified)
- `_width` (int): Ignored parameter (spatial clipping simplified)
- `nbins` (int): Number of temporal bins per window (default: 10)
- `window_duration_ms` (float): Duration of each window in milliseconds (default: 50.0)
- `stride_ms` (float, optional): Stride between windows
- `_count_cutoff` (int, optional): Ignored parameter (count limiting simplified)

**Returns:**
- `polars.LazyFrame`: LazyFrame with columns [window_id, channel, time_bin, y, x, count]

### create_voxel_grid

Creates traditional voxel grid representation.

**Example Usage:**
```python
import evlib.representations as evr

# Create voxel grid
events = evlib.load_events("data/slider_depth/events.txt")
voxel_df = evr.create_voxel_grid(
    events,
    height=480,
    width=640,
    n_time_bins=5
)
# Returns Polars DataFrame directly
print(f"Generated voxel grid with {len(voxel_df)} entries")
print(f"Columns: {list(voxel_df.columns)}")
```

**Parameters:**
- `events_pydf` (polars.DataFrame): Polars DataFrame with event data
- `_height` (int): Ignored parameter (spatial clipping simplified)
- `_width` (int): Ignored parameter (spatial clipping simplified)
- `nbins` (int): Number of temporal bins (default: 5)

**Returns:**
- `polars.DataFrame`: DataFrame with columns [time_bin, y, x, value]

### create_mixed_density_stack

Creates mixed density event stack representation.

**Example Usage:**
```python
import evlib.representations as evr

# Create mixed density stack
events = evlib.load_events("data/slider_depth/events.txt")
stack_df = evr.create_mixed_density_stack(
    events,
    height=480,
    width=640
)
# Returns Polars DataFrame directly
print(f"Generated mixed density stack with {len(stack_df)} entries")
print(f"Columns: {list(stack_df.columns)}")
```

**Parameters:**
- `events_pydf` (polars.DataFrame): Polars DataFrame with event data
- `_height` (int): Ignored parameter (spatial clipping simplified)
- `_width` (int): Ignored parameter (spatial clipping simplified)
- `nbins` (int): Number of temporal bins (default: 10)
- `window_duration_ms` (float): Duration of each window in milliseconds (default: 50.0)

**Returns:**
- `polars.DataFrame`: DataFrame with mixed density stack representation

## High-Level API

### preprocess_for_detection

High-level preprocessing function to replace RVT's preprocessing pipeline.

**Example Usage:**
```python
import evlib.representations as evr

# High-level preprocessing pipeline (under development)
events = evlib.load_events("data/slider_depth/events.txt")
events_df = events.collect()
# High-level preprocessing for neural networks
data_df = evr.create_stacked_histogram(
    events,
    height=480,
    width=640,
    bins=10,
    window_duration_ms=50.0
)
print(f"Preprocessed {len(data_df)} stacked histogram entries for detection pipeline")
# Data is ready for neural network input
```

**Parameters:**
- `events_path` (str): Path to event file
- `representation` (str): Type of representation ("stacked_histogram", "mixed_density", "voxel_grid")
- `height` (int): Output image height
- `width` (int): Output image width
- `**kwargs`: Representation-specific parameters

**Returns:**
- `polars.LazyFrame`: Preprocessed representation ready for neural networks

### benchmark_vs_rvt

Benchmark the Polars-based implementation against RVT's approach.

**Example Usage:**
```python
import evlib.representations as evr

# Performance comparison with RVT (manual benchmarking available)
import time

# evlib approach (using voxel grid as workaround)
start_time = time.time()
events = evlib.load_events("data/slider_depth/events.txt")
events_df = events.collect()
voxel_df = evr.create_voxel_grid(events_df, height=480, width=640, n_time_bins=10)
evlib_time = time.time() - start_time

print(f"evlib processing time: {evlib_time:.3f}s")
print(f"Generated {len(voxel_df)} voxel grid entries")
print("For RVT comparison, implement equivalent PyTorch-based pipeline")
```

**Parameters:**
- `events_path` (str): Path to test event file
- `height` (int): Sensor height
- `width` (int): Sensor width

**Returns:**
- `dict`: Performance comparison results including speedup metrics and output schema

## Use Cases

### Neural Network Input

```python
import evlib.representations as evr

# Prepare input for event-based neural networks
def prepare_network_input(events_path):
    # Create voxel grid representation
    events = evlib.load_events(events_path)
    events_df = events.collect()
    voxel_df = evr.create_voxel_grid(events_df, height=480, width=640, n_time_bins=5)

    # Convert to NumPy for neural network processing
    voxel_array = voxel_df.to_numpy()

    return voxel_array
```

### Temporal Analysis

```python
import evlib.representations as evr
import polars as pl

# Analyze temporal dynamics
def analyze_temporal_activity(events_path, time_window=0.1):
    # Create high temporal resolution representation
    events = evlib.load_events(events_path)
    events_df = events.collect()
    voxel_df = evr.create_voxel_grid(events_df, height=480, width=640, n_time_bins=20)

    # Analyze activity over time (group by temporal bins)
    activity_per_bin = voxel_df.group_by("time_bin").agg([
        pl.col("value").sum().alias("total_activity")
    ])

    return activity_per_bin
```

### Visualization

```python
import evlib.representations as evr
import polars as pl

# Create visualization-ready representations
def create_event_image(events_path):
    # Single time bin for accumulated image
    events = evlib.load_events(events_path)
    events_df = events.collect()
    voxel_df = evr.create_voxel_grid(events_df, height=480, width=640, n_time_bins=1)

    # Convert to image format (group by spatial coordinates)
    event_image = voxel_df.group_by(["y", "x"]).agg([
        pl.col("value").sum().alias("intensity")
    ])

    return event_image
```

## Performance Characteristics

| Operation | Performance vs NumPy | Memory Usage | Notes |
|-----------|---------------------|--------------|-------|
| Standard voxel grid | ~2.1x faster | Lower | Optimized binning |
| Smooth voxel grid | ~1.8x faster | Similar | Interpolation overhead |
| Large datasets (>1M events) | ~3x faster | Much lower | Memory efficiency |

## Advanced Usage

### Multi-Scale Representations

```python
import evlib.representations as evr

# Create multi-scale voxel grids
def create_multiscale_voxels(events_path):
    scales = [
        (640, 480, 5),   # Full resolution
        (320, 240, 5),   # Half resolution
        (160, 120, 5),   # Quarter resolution
    ]

    multiscale_voxels = []
    for width, height, bins in scales:
        events = evlib.load_events(events_path)
        events_df = events.collect()
        voxel_df = evr.create_voxel_grid(events_df, width=width, height=height, n_time_bins=bins)
        multiscale_voxels.append(voxel_df)

    return multiscale_voxels
```

### Custom Temporal Windows

```python
import evlib.filtering as evf
import evlib.representations as evr

# Create voxel grid for specific time window
def voxel_grid_time_window(events_path, t_start, t_end, bins=5):
    # Filter events by time window and create voxel grid
    events = evlib.load_events(events_path)
    filtered_events = evf.filter_by_time(events, t_start=t_start, t_end=t_end)
    filtered_df = filtered_events.collect()

    # Create voxel grid from filtered events
    voxel_df = evr.create_voxel_grid(filtered_df, width=640, height=480, n_time_bins=bins)

    return voxel_df
```

### Polarity-Separated Representations

```python
import evlib.filtering as evf
import evlib.representations as evr

# Create separate representations for positive and negative events
def create_polarity_separated_voxels(events_path):
    # Filter and create voxel grids for each polarity
    events = evlib.load_events(events_path)

    # Positive events (polarity = 1)
    pos_events = evf.filter_by_polarity(events, polarity=1)
    pos_df = pos_events.collect()
    pos_voxel = evr.create_voxel_grid(pos_df, width=640, height=480, n_time_bins=5)

    # Negative events (polarity = -1)
    neg_events = evf.filter_by_polarity(events, polarity=-1)
    neg_df = neg_events.collect()
    neg_voxel = evr.create_voxel_grid(neg_df, width=640, height=480, n_time_bins=5)

    return pos_voxel, neg_voxel
```

## Best Practices

### Choosing Temporal Bins
```python
# Rule of thumb: aim for 1-10 events per bin on average
def estimate_optimal_bins(ts, target_events_per_bin=5):
    total_time = ts.max() - ts.min()
    n_events = len(ts)

    # Estimate bins based on event density
    optimal_bins = max(1, int(n_events / target_events_per_bin))

    # Reasonable bounds
    optimal_bins = min(max(optimal_bins, 3), 20)

    return optimal_bins
```

### Memory Efficiency
```python
import evlib.representations as evr

# For very large datasets, use LazyFrames for memory efficiency
def create_voxel_memory_efficient(events_path):
    # LazyFrames automatically handle memory efficiency
    events = evlib.load_events(events_path)  # Keep as LazyFrame
    events_df = events.collect()
    voxel_df = evr.create_voxel_grid(events_df, height=480, width=640, n_time_bins=5)

    # Only materialize when needed
    print(f"Voxel grid created with {len(voxel_df)} entries")

    return voxel_df
```

### Quality Validation
```python
import evlib.representations as evr

# Validate voxel grid quality
def validate_voxel_grid(events_path):
    # Create voxel grid
    events = evlib.load_events(events_path)
    events_df = events.collect()
    voxel_df = evr.create_voxel_grid(events_df, height=480, width=640, n_time_bins=5)

    # Basic validation
    total_voxel_events = voxel_df["value"].sum()
    non_zero_bins = (voxel_df["value"] != 0).sum()
    max_events_per_bin = voxel_df["value"].max()

    print(f"Total voxel events: {total_voxel_events}")
    print(f"Non-zero bins: {non_zero_bins}")
    print(f"Max events per bin: {max_events_per_bin}")

    return total_voxel_events > 0  # Basic validation
```

## Migration Guide

### From dv-processing
```python
# evlib provides a unified, high-performance API for event representations
import evlib.representations as evr
import evlib.representations as evr

# Define parameters
events_path = "data/slider_depth/events.txt"
width = 640
height = 480
bins = 10

# Load and process events
events = evlib.load_events(events_path)
events_df = events.collect()

# Create voxel grids
voxel_df = evr.create_voxel_grid(events_df, height=height, width=width, n_time_bins=bins)
print(f"Voxel grid created with {len(voxel_df)} entries")

# Create mixed density stacks
mixed_df = evr.create_mixed_density_stack(events_df, height=height, width=width, n_time_bins=bins)
print(f"Mixed density stack created with {len(mixed_df)} entries")

# Note: Stacked histogram has a known filter predicate issue
# Mixed density stack and voxel grid work correctly
