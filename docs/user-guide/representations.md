# Event Representations

Learn how to convert sparse event data into dense representations suitable for visualization, analysis, and neural networks.

## Overview

Event cameras produce sparse, asynchronous data streams. To work with this data effectively, we often need to convert it into dense representations like images or voxel grids.

evlib provides optimized implementations for the most common event representations:

- **Voxel Grids**: Quantized temporal representations
- **Smooth Voxel Grids**: Interpolated temporal representations
- **Event Images**: Accumulated spatial representations

## Voxel Grids

Voxel grids divide time into discrete bins and accumulate events within each bin, creating a 3D tensor representation.

### Basic Usage

```python
import evlib

# Load events
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

# Create voxel grid
voxel_data, voxel_shape_data, voxel_shape_shape = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps,
    width=640,      # Image width
    height=480,     # Image height
    bins=5          # Number of temporal bins
)

print(f"Voxel grid shape: {voxel_grid.shape}")  # (5, 480, 640)
```

### How Voxel Grids Work

1. **Time Binning**: The event time range is divided into equal bins
2. **Event Assignment**: Each event is assigned to a temporal bin
3. **Spatial Accumulation**: Events accumulate in 2D spatial locations
4. **Polarity Handling**: Positive/negative events contribute +1/-1

```python
# Understanding the time binning
t_min, t_max = ts.min(), ts.max()
duration = t_max - t_min
bin_duration = duration / bins

print(f"Time range: {t_min:.3f} - {t_max:.3f} seconds")
print(f"Duration: {duration:.3f} seconds")
print(f"Bin duration: {bin_duration:.3f} seconds")
```

### Advanced Parameters

```python
# Create voxel grid with normalization
voxel_normalized_data, voxel_normalized_shape = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps,
    width=640,
    height=480,
    bins=5,
    normalize=True  # Normalize by max value
)

# Different temporal bin counts
voxel_high_res_data, voxel_high_res_shape = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps, 640, 480, bins=10   # Higher temporal resolution
)

voxel_low_res_data, voxel_low_res_shape = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps, 640, 480, bins=3    # Lower temporal resolution
)
```

## Smooth Voxel Grids

Smooth voxel grids use bilinear interpolation to reduce temporal quantization artifacts and provide better gradient flow for neural networks.

### Basic Usage

```python
# Create smooth voxel grid
smooth_data, smooth_shape_data, smooth_shape_shape = evlib.representations.events_to_smooth_voxel_grid(xs, ys, ts, ps, 5, (640, 480))

print(f"Smooth voxel shape: {smooth_voxel.shape}")  # (5, 480, 640)
```

### Benefits of Smooth Voxel Grids

1. **Reduced Quantization**: Events contribute to multiple temporal bins
2. **Better Gradients**: Smoother representations for neural network training
3. **Temporal Continuity**: More natural temporal transitions

### Comparison: Regular vs Smooth

```python
import matplotlib.pyplot as plt

# Create both representations
regular_voxel_data, regular_voxel_shape = evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, 5, (640, 480))
smooth_data, smooth_shape_data, smooth_shape_shape = evlib.representations.events_to_smooth_voxel_grid(xs, ys, ts, ps, 5, (640, 480))

# Visualize difference
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    # Regular voxel grid
    axes[0, i].imshow(regular_voxel[i], cmap='RdBu_r')
    axes[0, i].set_title(f'Regular Bin {i}')
    axes[0, i].axis('off')

    # Smooth voxel grid
    axes[1, i].imshow(smooth_voxel[i], cmap='RdBu_r')
    axes[1, i].set_title(f'Smooth Bin {i}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
```

## Event Images

For visualization and simple analysis, you can create accumulated event images.

### Single Accumulated Image

```python
# Create single time bin for accumulated image
event_image_data, event_image_shape = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps,
    width=640,
    height=480,
    bins=1  # Single bin accumulates all events
)[0]  # Take the first (only) bin

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(event_image, cmap='RdBu_r')
plt.title('Accumulated Event Image')
plt.colorbar(label='Event Count')
plt.show()
```

### Polarity-Separated Images

```python
# Separate positive and negative events
pos_mask = ps > 0
neg_mask = ps < 0

# Positive events image
pos_image_data, pos_image_shape = evlib.representations.events_to_voxel_grid(
    xs[pos_mask], ys[pos_mask], ts[pos_mask], ps[pos_mask],
    640, 480, 1
)[0]

# Negative events image
neg_image_data, neg_image_shape = evlib.representations.events_to_voxel_grid(
    xs[neg_mask], ys[neg_mask], ts[neg_mask], ps[neg_mask],
    640, 480, 1
)[0]

# Visualize side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(pos_image, cmap='Reds')
axes[0].set_title('Positive Events (ON)')
axes[0].axis('off')

axes[1].imshow(-neg_image, cmap='Blues')  # Negative for visualization
axes[1].set_title('Negative Events (OFF)')
axes[1].axis('off')

axes[2].imshow(pos_image - neg_image, cmap='RdBu_r')
axes[2].set_title('Combined (Red=ON, Blue=OFF)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## Neural Network Applications

### Preparing Input for Neural Networks

```python
def prepare_network_input(xs, ys, ts, ps, width=640, height=480):
    """Prepare voxel grid input for neural networks"""

    # Create smooth voxel grid for better gradients
    voxel_data, voxel_shape = evlib.representations.events_to_smooth_voxel_grid(
        xs, ys, ts, ps, width, height, bins=5
    )

    # Normalize for stable training
    voxel_mean = voxel.mean()
    voxel_std = voxel.std()
    voxel_normalized = (voxel - voxel_mean) / (voxel_std + 1e-8)

    return voxel_normalized

# Use with PyTorch
import torch

voxel_input = prepare_network_input(xs, ys, ts, ps)
tensor_input = torch.from_numpy(voxel_input).float()
tensor_input = tensor_input.unsqueeze(0)  # Add batch dimension

print(f"Neural network input shape: {tensor_input.shape}")  # (1, 5, 480, 640)
```

### Multi-Scale Representations

```python
def create_multiscale_voxels(xs, ys, ts, ps):
    """Create voxel grids at multiple spatial scales"""

    scales = [
        (640, 480),  # Full resolution
        (320, 240),  # Half resolution
        (160, 120),  # Quarter resolution
    ]

    multiscale_voxels = []

    for width, height in scales:
        # Downsample coordinates
        scale_x = width / 640
        scale_y = height / 480

        xs_scaled = (xs * scale_x).astype(int)
        ys_scaled = (ys * scale_y).astype(int)

        # Clip to bounds
        xs_scaled = np.clip(xs_scaled, 0, width - 1)
        ys_scaled = np.clip(ys_scaled, 0, height - 1)

        # Create voxel grid
        voxel_data, voxel_shape = evlib.representations.events_to_smooth_voxel_grid(
            xs_scaled, ys_scaled, ts, ps, width, height, 5
        )

        multiscale_voxels.append(voxel)

    return multiscale_voxels

# Create multi-scale representation
multi_voxels = create_multiscale_voxels(xs, ys, ts, ps)

for i, voxel in enumerate(multi_voxels):
    print(f"Scale {i}: {voxel.shape}")
```

## Performance Considerations

### Choosing Temporal Bins

The number of temporal bins affects both representation quality and computational cost:

```python
def estimate_optimal_bins(xs, ys, ts, ps, target_events_per_bin=5):
    """Estimate optimal number of temporal bins"""

    n_events = len(xs)
    duration = ts.max() - ts.min()

    # Aim for target events per bin on average
    optimal_bins = max(3, min(20, n_events // target_events_per_bin))

    print(f"Events: {n_events:,}")
    print(f"Duration: {duration:.3f}s")
    print(f"Optimal bins: {optimal_bins}")
    print(f"Events per bin: {n_events // optimal_bins}")

    return optimal_bins

# Use optimal bin count
optimal_bins = estimate_optimal_bins(xs, ys, ts, ps)
voxel_data, voxel_shape = evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, 640, 480, optimal_bins)
```

### Memory Efficiency

For large datasets, process in chunks:

```python
def create_voxel_chunked(xs, ys, ts, ps, width, height, bins, chunk_size=100000):
    """Create voxel grid in chunks for memory efficiency"""

    n_events = len(xs)

    if n_events <= chunk_size:
        return evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, width, height, bins)

    # Initialize accumulator
    voxel_accumulated = np.zeros((bins, height, width), dtype=np.float32)

    # Process in chunks
    for i in range(0, n_events, chunk_size):
        end_idx = min(i + chunk_size, n_events)

        chunk_voxel_data, chunk_voxel_shape = evlib.representations.events_to_voxel_grid(
            xs[i:end_idx], ys[i:end_idx], ts[i:end_idx], ps[i:end_idx],
            width, height, bins
        )

        voxel_accumulated += chunk_voxel

    return voxel_accumulated

# Use for very large datasets
large_voxel = create_voxel_chunked(xs, ys, ts, ps, 640, 480, 5, chunk_size=50000)
```

## Quality Validation

### Validate Representation Quality

```python
def validate_voxel_quality(voxel_grid, xs, ys, ts, ps):
    """Validate that voxel grid preserves event information"""

    total_events = len(xs)
    voxel_events = np.abs(voxel_grid).sum()  # Total accumulated events

    # Check event preservation
    preservation_ratio = voxel_events / total_events

    # Check spatial distribution
    non_zero_pixels = (voxel_grid != 0).sum()
    total_pixels = voxel_grid.shape[1] * voxel_grid.shape[2]
    sparsity = non_zero_pixels / total_pixels

    # Check temporal distribution
    temporal_activity = np.abs(voxel_grid).sum(axis=(1, 2))
    temporal_variance = np.var(temporal_activity)

    print(f"Event preservation: {preservation_ratio:.1%}")
    print(f"Spatial sparsity: {sparsity:.1%}")
    print(f"Temporal variance: {temporal_variance:.2f}")
    print(f"Max events per pixel: {np.abs(voxel_grid).max():.0f}")

    return {
        'preservation': preservation_ratio,
        'sparsity': sparsity,
        'temporal_variance': temporal_variance
    }

# Validate your voxel grid
stats = validate_voxel_quality(voxel_grid, xs, ys, ts, ps)
```

## Best Practices

### 1. Choose the Right Representation

- **Voxel grids**: General purpose, fast
- **Smooth voxel grids**: Neural networks, better quality
- **Event images**: Visualization, simple analysis

### 2. Temporal Resolution Guidelines

- **3-5 bins**: Fast processing, basic temporal info
- **5-10 bins**: Balanced temporal resolution
- **10+ bins**: High temporal resolution, slower processing

### 3. Spatial Resolution

- **Full resolution**: Maximum detail, slower processing
- **Half resolution**: Good balance for most applications
- **Quarter resolution**: Fast processing, less detail

### 4. Normalization

```python
# For neural networks, always normalize
voxel_data, voxel_shape = evlib.representations.events_to_smooth_voxel_grid(xs, ys, ts, ps, 5, (640, 480))
voxel_norm = (voxel - voxel.mean()) / (voxel.std() + 1e-8)

# For visualization, consider clipping outliers
voxel_vis = np.clip(voxel, np.percentile(voxel, 1), np.percentile(voxel, 99))
```

## Next Steps

- [Visualization Guide](visualization.md): Display your representations
- [Neural Networks](models.md): Use representations with deep learning
- [API Reference](../api/representations.md): Detailed function documentation
