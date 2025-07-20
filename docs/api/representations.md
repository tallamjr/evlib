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
hist_lazy = evr.create_stacked_histogram(
    "data/slider_depth/events.txt",
    height=480, width=640,
    nbins=10, window_duration_ms=50
)
# Returns Polars LazyFrame - call .collect() to materialize
hist_df = hist_lazy.collect()
```

**Parameters:**
- `events` (str|LazyFrame): Path to event file or Polars LazyFrame
- `height` (int): Output image height
- `width` (int): Output image width
- `nbins` (int): Number of temporal bins per window
- `window_duration_ms` (float): Duration of each window in milliseconds
- `stride_ms` (float): Stride between windows (optional)
- `count_cutoff` (int): Maximum count per bin (optional)

**Returns:**
- `polars.LazyFrame`: LazyFrame with columns [window_id, channel, time_bin, y, x, count]

### create_voxel_grid

Creates traditional voxel grid representation.

**Example Usage:**
```python
import evlib.representations as evr

# Create voxel grid
voxel_lazy = evr.create_voxel_grid(
    "data/slider_depth/events.txt",
    height=480, width=640,
    nbins=5
)
# Returns Polars LazyFrame - call .collect() to materialize
voxel_df = voxel_lazy.collect()
```

**Parameters:**
- `events` (str|LazyFrame): Path to event file or Polars LazyFrame
- `height` (int): Output image height
- `width` (int): Output image width
- `nbins` (int): Number of temporal bins

**Returns:**
- `polars.LazyFrame`: LazyFrame with columns [time_bin, y, x, value]

### create_mixed_density_stack

Creates mixed density event stack representation.

**Example Usage:**
```python
import evlib.representations as evr

# Create mixed density stack
stack_lazy = evr.create_mixed_density_stack(
    "data/slider_depth/events.txt",
    height=480, width=640,
    nbins=10, window_duration_ms=50
)
# Returns Polars LazyFrame - call .collect() to materialize
stack_df = stack_lazy.collect()
```

**Parameters:**
- `events` (str|LazyFrame): Path to event file or Polars LazyFrame
- `height` (int): Output image height
- `width` (int): Output image width
- `nbins` (int): Number of temporal bins
- `window_duration_ms` (float): Duration of each window in milliseconds
- `count_cutoff` (int): Maximum absolute value per bin (optional)

**Returns:**
- `polars.LazyFrame`: LazyFrame with columns [window_id, time_bin, y, x, polarity_sum]

## High-Level API

### preprocess_for_detection

High-level preprocessing function to replace RVT's preprocessing pipeline.

**Example Usage:**
```python
import evlib.representations as evr

# Replace RVT preprocessing
data_lazy = evr.preprocess_for_detection(
    "data/slider_depth/events.txt",
    representation="stacked_histogram",
    height=480, width=640,
    nbins=10, window_duration_ms=50
)
# Materialize when needed
data_df = data_lazy.collect()
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

# Benchmark performance
results = evr.benchmark_vs_rvt("data/slider_depth/events.txt", height=480, width=640)
print(f"Speedup: {results['speedup']:.1f}x")
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
    voxel_lazy = evr.create_voxel_grid(events_path, height=480, width=640, nbins=5)
    voxel_df = voxel_lazy.collect()

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
    voxel_lazy = evr.create_voxel_grid(events_path, height=480, width=640, nbins=20)
    voxel_df = voxel_lazy.collect()

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
    voxel_lazy = evr.create_voxel_grid(events_path, height=480, width=640, nbins=1)
    voxel_df = voxel_lazy.collect()

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
        voxel_lazy = evr.create_voxel_grid(events_path, width=width, height=height, nbins=bins)
        voxel_df = voxel_lazy.collect()
        multiscale_voxels.append(voxel_df)

    return multiscale_voxels
```

### Custom Temporal Windows

```python
import evlib.filtering as evf
import evlib.representations as evr

# Create voxel grid for specific time window
def voxel_grid_time_window(events_path, t_start, t_end, bins=5):
    # Filter events to time window using evlib filtering
    filtered_events = evf.filter_by_time(events_path, t_start=t_start, t_end=t_end)

    # Create voxel grid from filtered events
    voxel_lazy = evr.create_voxel_grid(filtered_events, width=640, height=480, nbins=bins)
    voxel_df = voxel_lazy.collect()

    return voxel_df
```

### Polarity-Separated Representations

```python
import evlib.filtering as evf
import evlib.representations as evr

# Create separate representations for positive and negative events
def create_polarity_separated_voxels(events_path):
    # Positive events
    pos_events = evf.filter_by_polarity(events_path, polarity=1)
    pos_voxel_lazy = evr.create_voxel_grid(pos_events, width=640, height=480, nbins=5)
    pos_voxel = pos_voxel_lazy.collect()

    # Negative events
    neg_events = evf.filter_by_polarity(events_path, polarity=0)  # or -1 depending on encoding
    neg_voxel_lazy = evr.create_voxel_grid(neg_events, width=640, height=480, nbins=5)
    neg_voxel = neg_voxel_lazy.collect()

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
    voxel_lazy = evr.create_voxel_grid(events_path, height=480, width=640, nbins=5)

    # Only materialize when needed
    # voxel_df = voxel_lazy.collect()

    return voxel_lazy  # Return lazy for memory efficiency
```

### Quality Validation
```python
import evlib.representations as evr

# Validate voxel grid quality
def validate_voxel_grid(events_path):
    # Create voxel grid
    voxel_lazy = evr.create_voxel_grid(events_path, height=480, width=640, nbins=5)
    voxel_df = voxel_lazy.collect()

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
import evlib
import evlib.representations as evr

# Define parameters
events_path = "data/slider_depth/events.txt"
width = 640
height = 480
bins = 10

# Load events
events = evlib.load_events(events_path)

# Create voxel grids
voxel_lazy = evr.create_voxel_grid(events, width=width, height=height, nbins=bins)
voxel_df = voxel_lazy.collect()

# Create mixed density stacks
mixed_lazy = evr.create_mixed_density_stack(events, width=width, height=height, nbins=bins)
mixed_df = mixed_lazy.collect()

# Create stacked histograms
hist_lazy = evr.create_stacked_histogram(events, width=width, height=height, nbins=bins)
hist_df = hist_lazy.collect()
