# Representations API Reference

The representations module provides efficient implementations for converting event data into various spatial-temporal representations.

## Overview

```python
import evlib.representations
```

Event representations are crucial for:
- **Neural network input**: Converting events to tensor-like formats
- **Visualization**: Creating images from sparse event data
- **Analysis**: Temporal and spatial aggregation of events

## Core Functions

### create_voxel_grid

::: evlib.representations.events_to_voxel_grid

Creates a quantized voxel grid representation of events.

**Example Usage:**
```python
# Basic voxel grid
voxel_data, voxel_shape_data, voxel_shape_shape = evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, 5, (640, 480))
# Output shape: (5, 480, 640)

# Normalize event counts
voxel_grid_normalized_data, voxel_grid_normalized_shape = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps,
    width=640,
    height=480,
    bins=5,
    normalize=True
)
```

**Parameters:**
- `xs` (array): X coordinates of events
- `ys` (array): Y coordinates of events
- `ts` (array): Timestamps of events
- `ps` (array): Polarities of events (+1 or -1)
- `width` (int): Output image width
- `height` (int): Output image height
- `bins` (int): Number of temporal bins
- `normalize` (bool): Whether to normalize the output

**Returns:**
- `ndarray`: Voxel grid with shape (bins, height, width)

### create_smooth_voxel_grid

::: evlib.representations.events_to_smooth_voxel_grid

Creates a smooth voxel grid with bilinear interpolation for improved temporal resolution.

**Example Usage:**
```python
# Smooth voxel grid with interpolation
smooth_data, smooth_shape_data, smooth_shape_shape = evlib.representations.events_to_smooth_voxel_grid(xs, ys, ts, ps, 5, (640, 480))
# Improved temporal smoothness compared to standard voxel grid
```

**Key Features:**
- **Bilinear interpolation**: Events contribute to multiple bins
- **Temporal smoothness**: Reduces quantization artifacts
- **Better gradient flow**: Improved for neural network training

## Use Cases

### Neural Network Input

```python
# Prepare input for event-based neural networks
def prepare_network_input(xs, ys, ts, ps):
    # Create smooth voxel grid for better gradients
    voxel_input_data, voxel_input_shape = evlib.representations.events_to_smooth_voxel_grid(xs, ys, ts, ps, 5, (640, 480))

    # Normalize for neural network
    voxel_input = (voxel_input - voxel_input.mean()) / voxel_input.std()

    return voxel_input
```

### Temporal Analysis

```python
# Analyze temporal dynamics
def analyze_temporal_activity(xs, ys, ts, ps, time_window=0.1):
    # Create high temporal resolution representation
    n_bins = int((ts.max() - ts.min()) / time_window)

    temporal_grid_data, temporal_grid_shape = evlib.representations.events_to_voxel_grid(
        xs, ys, ts, ps,
        width=640,
        height=480,
        bins=n_bins
    )

    # Analyze activity over time
    activity_per_bin = temporal_grid.sum(axis=(1, 2))
    return activity_per_bin
```

### Visualization

```python
# Create visualization-ready representations
def create_event_image(xs, ys, ts, ps):
    # Single time bin for accumulated image
    event_image_data, event_image_shape = evlib.representations.events_to_voxel_grid(
        xs, ys, ts, ps,
        width=640,
        height=480,
        bins=1,
        normalize=True
    )[0]  # Take first (only) bin

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
# Create multi-scale voxel grids
def create_multiscale_voxels(xs, ys, ts, ps):
    scales = [
        (640, 480, 5),   # Full resolution
        (320, 240, 5),   # Half resolution
        (160, 120, 5),   # Quarter resolution
    ]

    multiscale_voxels = []
    for width, height, bins in scales:
        # Downsample coordinates
        xs_scaled = (xs * width / 640).astype(int)
        ys_scaled = (ys * height / 480).astype(int)

        voxel_data, voxel_shape = evlib.representations.events_to_voxel_grid(
            xs_scaled, ys_scaled, ts, ps,
            width, height, bins
        )
        multiscale_voxels.append(voxel)

    return multiscale_voxels
```

### Custom Temporal Windows

```python
# Create voxel grid for specific time window
def voxel_grid_time_window(xs, ys, ts, ps, t_start, t_end, bins=5):
    # Filter events to time window
    mask = (ts >= t_start) & (ts <= t_end)
    xs_filtered = xs[mask]
    ys_filtered = ys[mask]
    ts_filtered = ts[mask]
    ps_filtered = ps[mask]

    # Create voxel grid
    voxel_data, voxel_shape = evlib.representations.events_to_voxel_grid(
        xs_filtered, ys_filtered, ts_filtered, ps_filtered,
        width=640, height=480, bins=bins
    )

    return voxel
```

### Polarity-Separated Representations

```python
# Create separate representations for positive and negative events
def create_polarity_separated_voxels(xs, ys, ts, ps):
    # Positive events
    pos_mask = ps > 0
    pos_voxel_data, pos_voxel_shape = evlib.representations.events_to_voxel_grid(xs[pos_mask], ys[pos_mask], ts[pos_mask], ps[pos_mask], 5, (640, 480))

    # Negative events
    neg_mask = ps < 0
    neg_voxel_data, neg_voxel_shape = evlib.representations.events_to_voxel_grid(xs[neg_mask], ys[neg_mask], ts[neg_mask], ps[neg_mask], 5, (640, 480))

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
# For very large datasets, process in chunks
def create_voxel_chunked(xs, ys, ts, ps, chunk_size=100000):
    n_events = len(xs)

    if n_events <= chunk_size:
        return evlib.representations.events_to_voxel_grid(
            xs, ys, ts, ps, 640, 480, 5
        )

    # Process in chunks and accumulate
    accumulated_voxel = np.zeros((5, 480, 640), dtype=np.float32)

    for i in range(0, n_events, chunk_size):
        end_idx = min(i + chunk_size, n_events)
        chunk_voxel_data, chunk_voxel_shape = evlib.representations.events_to_voxel_grid(
            xs[i:end_idx], ys[i:end_idx],
            ts[i:end_idx], ps[i:end_idx],
            640, 480, 5
        )
        accumulated_voxel += chunk_voxel

    return accumulated_voxel
```

### Quality Validation
```python
# Validate voxel grid quality
def validate_voxel_grid(voxel_grid, xs, ys, ts, ps):
    total_events = len(xs)
    voxel_events = voxel_grid.sum()

    # Check if event counts match (approximately)
    event_preservation = voxel_events / total_events

    print(f"Event preservation: {event_preservation:.2%}")
    print(f"Non-zero bins: {(voxel_grid > 0).sum()} / {voxel_grid.size}")
    print(f"Max events per bin: {voxel_grid.max():.0f}")

    return event_preservation > 0.95  # 95% preservation threshold
```

## Migration Guide

### From dv-processing
```python
# Old: dv-processing
import dv_processing as dv
voxel_grid = dv.representations.VoxelGrid(width, height, bins)

# New: evlib
import evlib.representations
voxel_data, voxel_shape_data, voxel_shape_shape = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps, width, height, bins
)
```

### From event_utils
```python
# Old: event_utils
import event_utils
voxel = event_utils.events_to_voxel_grid(events, bins, width, height)

# New: evlib
import evlib.representations
voxel_data, voxel_shape = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps, width, height, bins
)
```
