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
import evlib
import evlib.representations as evr

# Create stacked histogram (replaces RVT preprocessing)
events = evlib.load_events("data/slider_depth/events.txt")

# Create stacked histogram representation
hist_df = evr.create_stacked_histogram(
    events,  # Can accept LazyFrame or DataFrame
    height=480,
    width=640,
    bins=10,
    window_duration_ms=50.0
)
# Returns Polars DataFrame
print(f"Generated stacked histogram with {len(hist_df)} entries")
print(f"Columns: {list(hist_df.columns)}")  # ['time_bin', 'polarity', 'y', 'x', 'count']
```

**Parameters:**
- `events` (polars.LazyFrame | polars.DataFrame): Polars LazyFrame or DataFrame with event data
- `height` (int): Sensor height in pixels
- `width` (int): Sensor width in pixels
- `bins` (int): Number of temporal bins per window (default: 10)
- `window_duration_ms` (float): Duration of each window in milliseconds (default: 50.0)

**Returns:**
- `polars.DataFrame`: DataFrame with columns `[time_bin, polarity, y, x, count]`

### create_voxel_grid

Creates a voxel grid representation with full bilinear temporal interpolation, matching the Zhu et al. 2019 event volume (the `to_voxel_grid` representation in the tonic reference library). Each event splits its signed polarity (+1 or -1) across the two temporal bins that straddle its normalised timestamp, weighted by the fractional distance to each bin centre. This is the SOTA event volume used by E2VID and related models, not a hard temporal histogram.

The output is in long (sparse) format: one row per non-empty `(x, y, time_bin)` cell, with the summed `contribution`. Use `densify_voxel_grid` to scatter it into a dense `(n_time_bins, 1, H, W)` tensor for model input.

**Example Usage:**
```python
import evlib
import evlib.representations as evr

# Create voxel grid
events = evlib.load_events("data/slider_depth/events.txt")
voxel_df = evr.create_voxel_grid(
    events,  # Can accept LazyFrame or DataFrame
    height=480,
    width=640,
    n_time_bins=5
)
# Returns Polars DataFrame
print(f"Generated voxel grid with {len(voxel_df)} entries")
print(f"Columns: {list(voxel_df.columns)}")  # ['x', 'y', 'time_bin', 'contribution']
```

**Parameters:**
- `events` (polars.LazyFrame | polars.DataFrame): Polars LazyFrame or DataFrame with event data
- `height` (int): Sensor height in pixels
- `width` (int): Sensor width in pixels
- `n_time_bins` (int): Number of temporal bins (default: 5)

**Returns:**
- `polars.DataFrame`: long-format DataFrame with columns `[x, y, time_bin, contribution]`, where `contribution` is the bilinearly interpolated signed weight summed over all events falling in that cell.

### create_event_frame

Creates a tonic-validated event frame, matching tonic's `to_frame_numpy(n_time_bins=...)` semantics. The whole recording is sliced into `n_time_bins` equal-width time bins and events are counted per `(polarity, y, x)` per bin. This differs from `create_stacked_histogram` (which uses a fixed window duration relative to the global minimum timestamp).

Bin boundaries follow tonic's `SliceByTimeBins` (overlap 0) exactly: the bin width is integer-floored, bins are left-closed and right-open, and events at or beyond the final bin end are dropped (including the final max-time event). The output is long format; use `densify_event_frame` to obtain a dense `(n_time_bins, P, H, W)` tensor.

**Example Usage:**
```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
frame_df = evr.create_event_frame(
    events,
    height=480,
    width=640,
    n_time_bins=10,
)
print(f"Columns: {list(frame_df.columns)}")  # ['time_bin', 'polarity', 'y', 'x', 'count']
```

**Parameters:**
- `events` (polars.LazyFrame | polars.DataFrame): Polars LazyFrame or DataFrame with event data
- `height` (int): Sensor height in pixels
- `width` (int): Sensor width in pixels
- `n_time_bins` (int): Number of equal-width temporal bins (default: 10)

**Returns:**
- `polars.DataFrame`: long-format DataFrame with columns `[time_bin, polarity, y, x, count]`, where `polarity` is the channel index (0 for negative, 1 for positive).

### create_time_surface

Creates a tonic-validated HOTS time surface (Lagorce et al. 2016), matching tonic's `to_timesurface_numpy` semantics (overlap 0, `include_incomplete=False`). Events are sliced into fixed `dt` time windows relative to the first event timestamp; each window is left-closed and right-open. tonic maintains, per `(polarity, y, x)`, the most-recent event timestamp that persists across slices and produces `exp(-((i+1)*dt + start_t - memory) / tau)` for slice `i`.

This function performs the slice-local part in Polars and returns, per `(slice, polarity, y, x)`, the maximum event timestamp within that slice (`last_t`). The cross-slice memory persistence and the exponential decay are applied by `densify_time_surface`, because unseen cells must decay to 0 and the decay references a per-slice origin: both are dense numpy operations that stay bit-faithful to tonic.

**Example Usage:**
```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
ts_df = evr.create_time_surface(
    events,
    height=480,
    width=640,
    dt=50000.0,   # slice and decay window, microseconds
    tau=100000.0, # exponential decay time constant, microseconds
)
print(f"Columns: {list(ts_df.columns)}")  # ['slice', 'polarity', 'y', 'x', 'last_t']
```

**Parameters:**
- `events` (polars.LazyFrame | polars.DataFrame): Polars LazyFrame or DataFrame with event data
- `height` (int): Sensor height in pixels
- `width` (int): Sensor width in pixels
- `dt` (float): Time window for slicing and decay reference, in microseconds
- `tau` (float): Exponential decay time constant (applied in `densify_time_surface`)

**Returns:**
- `polars.DataFrame`: long-format DataFrame with columns `[slice, polarity, y, x, last_t]`, where `polarity` is the channel index (0 negative, 1 positive) and `last_t` is the maximum event timestamp (microseconds) within that slice at that pixel and channel.

## Densify Helpers

The `create_voxel_grid`, `create_event_frame` and `create_time_surface` functions return long-format (sparse) Polars DataFrames. The `densify_*` helpers are the bridge from that long format to a dense numpy tensor, which is the model-ready form. The three `create_*` representations and their densifiers are validated to be bit-faithful or numerically faithful against the tonic reference library in `tests/test_representations_conformance.py`.

### densify_voxel_grid

Scatters a `create_voxel_grid` DataFrame into a dense array.

**Parameters:**
- `df` (polars.DataFrame): DataFrame with columns `[x, y, time_bin, contribution]`
- `n_time_bins` (int): Number of temporal bins (size of axis 0)
- `height` (int): Sensor height in pixels
- `width` (int): Sensor width in pixels

**Returns:**
- `numpy.ndarray`: dense `(n_time_bins, 1, height, width)` float64 array.

### densify_event_frame

Scatters a `create_event_frame` DataFrame into a dense array matching tonic's `to_frame_numpy` shape.

**Parameters:**
- `df` (polars.DataFrame): DataFrame with columns `[time_bin, polarity, y, x, count]`
- `n_time_bins` (int): Number of temporal bins (size of axis 0)
- `n_polarities` (int): Number of polarity channels (size of axis 1)
- `height` (int): Sensor height in pixels
- `width` (int): Sensor width in pixels

**Returns:**
- `numpy.ndarray`: dense `(n_time_bins, n_polarities, height, width)` int64 array.

### densify_time_surface

Assembles dense HOTS time surfaces from a `create_time_surface` DataFrame, reproducing tonic's persistent memory (a forward cumulative maximum across the slice axis) before applying the exponential decay.

**Parameters:**
- `df` (polars.DataFrame): DataFrame with columns `[slice, polarity, y, x, last_t]`
- `n_slices` (int): Number of time slices (size of axis 0)
- `n_polarities` (int): Number of polarity channels (size of axis 1)
- `height` (int): Sensor height in pixels
- `width` (int): Sensor width in pixels
- `dt` (float): Time window in microseconds (same value passed to `create_time_surface`)
- `tau` (float): Exponential decay time constant
- `start_t` (float): Timestamp of the first event in microseconds (the decay reference origin)

**Returns:**
- `numpy.ndarray`: dense `(n_slices, n_polarities, height, width)` float64 array.

### create_mixed_density_stack

Creates mixed density event stack representation.

**Example Usage:**
```python
import evlib
import evlib.representations as evr

# Create mixed density stack
events = evlib.load_events("data/slider_depth/events.txt")
stack_df = evr.create_mixed_density_stack(
    events,  # Can accept LazyFrame or DataFrame
    height=480,
    width=640
)
# Returns Polars DataFrame
print(f"Generated mixed density stack with {len(stack_df)} entries")
print(f"Columns: {list(stack_df.columns)}")
```

**Parameters:**
- `events` (polars.LazyFrame | polars.DataFrame): Polars LazyFrame or DataFrame with event data
- `height` (int): Sensor height in pixels
- `width` (int): Sensor width in pixels

**Returns:**
- `polars.DataFrame`: DataFrame with mixed density stack representation

## High-Level API

### preprocess_for_detection

High-level preprocessing function to replace RVT's preprocessing pipeline.

**Example Usage:**
```python
import evlib
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
import evlib
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
import evlib
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
import evlib
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
import evlib
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
import evlib
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
import evlib
import evlib.filtering as evf
import evlib.representations as evr

# Create voxel grid for specific time window
def voxel_grid_time_window(events_path, t_start, t_end, bins=5):
    # Filter events by time window and create voxel grid
    events = evlib.load_events(events_path)
    events_df = events.collect()  # Convert LazyFrame to DataFrame first
    filtered_events = evf.filter_by_time(events_df, t_start=t_start, t_end=t_end)

    # Create voxel grid from filtered events
    voxel_df = evr.create_voxel_grid(filtered_events, width=640, height=480, n_time_bins=bins)

    return voxel_df
```

### Polarity-Separated Representations

```python
import evlib
import evlib.filtering as evf
import evlib.representations as evr

# Create separate representations for positive and negative events
def create_polarity_separated_voxels(events_path):
    # Filter and create voxel grids for each polarity
    events = evlib.load_events(events_path)
    events_df = events.collect()  # Convert LazyFrame to DataFrame first

    # Positive events (polarity = 1)
    pos_events = evf.filter_by_polarity(events_df, polarity=1)
    pos_voxel = evr.create_voxel_grid(pos_events, width=640, height=480, n_time_bins=5)

    # Negative events (polarity = -1)
    neg_events = evf.filter_by_polarity(events_df, polarity=-1)
    neg_voxel = evr.create_voxel_grid(neg_events, width=640, height=480, n_time_bins=5)

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
import evlib
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
import evlib
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
import evlib
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
mixed_df = evr.create_mixed_density_stack(events_df, height=height, width=width)
print(f"Mixed density stack created with {len(mixed_df)} entries")

# Note: Stacked histogram has a known filter predicate issue
# Mixed density stack and voxel grid work correctly
