# Quick Start

Get up and running with evlib in 5 minutes!

## Basic Event Loading

```python
import evlib

# Load events from a text file
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
print(f"Loaded {len(xs)} events")
```

## Event Filtering

```python
# Load events with time window
xs, ys, ts, ps = evlib.formats.load_events_filtered(
    "data/slider_depth/events.txt",
    t_start=0.0,    # Start time (seconds)
    t_end=1.0       # End time (seconds)
)

# Load events with spatial bounds
xs, ys, ts, ps = evlib.formats.load_events_filtered(
    "data/slider_depth/events.txt",
    min_x=100, max_x=500,  # X coordinate bounds
    min_y=100, max_y=300   # Y coordinate bounds
)

# Load events with polarity filtering
pos_xs, pos_ys, pos_ts, pos_ps = evlib.formats.load_events_filtered(
    "data/slider_depth/events.txt", polarity=1
)   # Positive events only
neg_xs, neg_ys, neg_ts, neg_ps = evlib.formats.load_events_filtered(
    "data/slider_depth/events.txt", polarity=-1
)  # Negative events only
```

## Event Representations

### Voxel Grid
```python
# Create a voxel grid representation
voxel_data, voxel_shape_data, voxel_shape_shape = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps,
    5,              # Number of temporal bins
    (640, 480)      # (width, height)
)
# Shape: (5, 480, 640)
```

### Smooth Voxel Grid
```python
# Create smooth voxel grid with bilinear interpolation
smooth_data, smooth_shape_data, smooth_shape_shape = evlib.representations.events_to_smooth_voxel_grid(
    xs, ys, ts, ps,
    5,              # Number of temporal bins
    (640, 480)      # (width, height)
)
# Improved temporal resolution through interpolation
```

## Event Visualization

### Basic Plotting
```python
import matplotlib.pyplot as plt

# Plot events as scatter plot (using matplotlib directly)
plt.figure(figsize=(10, 6))
plt.scatter(xs[:10000], ys[:10000], c=ps[:10000], cmap='RdBu_r', s=0.1)
plt.title("Event Visualization")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.show()
```

### Terminal Visualization (Ultra-Fast)
```python
# Create image visualization
event_image = evlib.visualization.draw_events_to_image(
    xs[:10000], ys[:10000], ps[:10000], 640, 480
)
# Display the event image
plt.figure(figsize=(10, 6))
plt.imshow(event_image, cmap='gray')
plt.title("Event Image Visualization")
plt.show()
```

## Event Augmentation

```python
# Add spatial transformations
xs_flipped, ys_flipped, ts_aug, ps_aug = evlib.augmentation.flip_events_x(xs, ys, ts, ps, (640, 480))

# Add random noise events
xs_noisy, ys_noisy, ts_noisy, ps_noisy = evlib.augmentation.add_random_events(xs, ys, ts, ps, 1000, (640, 480))
```

## Neural Network Inference

### E2VID Reconstruction
```python
# Download and use E2VID model
model_path = evlib.processing.download_model("e2vid_unet")
reconstructed_frame = evlib.processing.events_to_video(
    xs, ys, ts, ps,
    model_path=model_path,
    width=640,
    height=480
)
```

## File Format Support

### Text Files
```python
# Standard format: timestamp x y polarity
# 0.1 320 240 1
# 0.2 321 241 -1
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
```

### HDF5 Files
```python
# Load events first
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

# Save to HDF5
evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, "output.h5")

# Load from HDF5
xs_h5, ys_h5, ts_h5, ps_h5 = evlib.formats.load_events("output.h5")
```

### Custom Column Mapping
```python
# For files with different column order: x y polarity timestamp
# Note: Column mapping not currently supported in this version
# Use standard format: timestamp x y polarity
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
```

## Performance Tips

### When to Use evlib vs NumPy

SUCCESS: **Use evlib for:**
- Large datasets (>100k events)
- Complex event processing algorithms
- Memory-efficient operations
- Production event processing pipelines

SUCCESS: **Use NumPy for:**
- Simple array operations
- Small datasets (<10k events)
- Rapid prototyping
- Maximum single-operation speed

### Example Performance Comparison
```python
import time
import numpy as np

# Large dataset example
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")  # 1M+ events

# evlib voxel grid (optimized)
start = time.time()
voxel_data, voxel_shape_data, voxel_shape_shape = evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, 5, (640, 480))
time_evlib = time.time() - start

print(f"evlib voxel grid creation: {time_evlib:.3f}s for {len(xs):,} events")
print(f"Voxel grid shape: {voxel_shape}")
```

## Error Handling

```python
try:
    xs, ys, ts, ps = evlib.formats.load_events("nonexistent.txt")
except OSError as e:
    print(f"File error: {e}")

# Example with valid file
try:
    xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
    print(f"Successfully loaded {len(xs)} events")
except OSError as e:
    print(f"Error loading events: {e}")
```

## Next Steps

- [Loading Data Guide](../user-guide/loading-data.md)
- [Visualization Guide](../user-guide/visualization.md)
- [Neural Networks Guide](../user-guide/models.md)
- [Performance Guide](performance.md)
