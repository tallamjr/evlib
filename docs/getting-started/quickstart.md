# Quick Start

Get up and running with evlib in 5 minutes!

## Basic Event Loading

```python
import evlib
import polars as pl

# Load events from a text file (returns Polars LazyFrame)
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Get DataFrame if needed
print(f"Loaded {len(df)} events")

# Access data columns
xs = df['x'].to_numpy()
ys = df['y'].to_numpy()
ts = df['timestamp'].to_numpy()
ps = df['polarity'].to_numpy()
```

## Event Filtering

<!-- NOTE: evlib.filtering module not yet available. Use Polars filtering directly: -->

```python
import polars as pl

# Load events and use Polars filtering
events = evlib.load_events("data/slider_depth/events.txt")

# Time window filtering
filtered_events = events.filter(
    (pl.col('timestamp') >= 0.0) & (pl.col('timestamp') <= 1.0)
)
df = filtered_events.collect()
xs, ys, ts, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['timestamp'].to_numpy(), df['polarity'].to_numpy()

# Spatial bounds filtering
spatial_events = events.filter(
    (pl.col('x') >= 100) & (pl.col('x') <= 500) &
    (pl.col('y') >= 100) & (pl.col('y') <= 300)
)
df = spatial_events.collect()
xs, ys, ts, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['timestamp'].to_numpy(), df['polarity'].to_numpy()

# Polarity filtering
pos_events = events.filter(pl.col('polarity') == 1)  # Positive events only
neg_events = events.filter(pl.col('polarity') == -1)  # Negative events only

# Convert to numpy arrays if needed
pos_df = pos_events.collect()
pos_xs, pos_ys, pos_ts, pos_ps = pos_df['x'].to_numpy(), pos_df['y'].to_numpy(), pos_df['timestamp'].to_numpy(), pos_df['polarity'].to_numpy()
```

## Event Representations

```python
# Basic event data for custom representations
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Access event data for custom processing
print(f"Event data available for custom representations: {len(df)} events")
print(f"Data columns: {df.columns}")
print(f"Data types: {df.dtypes}")

# Note: Advanced representation functions are under development
# Use basic data access for custom implementations
```

## Event Visualization

```python
import numpy as np

# Load events first
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs = df['x'].to_numpy()
ys = df['y'].to_numpy()
ts = df['timestamp'].to_numpy()
ps = df['polarity'].to_numpy()

print(f"Loaded {len(df)} events for visualization")
print(f"Event data shape: x={xs.shape}, y={ys.shape}, t={ts.shape}, p={ps.shape}")

# Note: For actual plotting, install matplotlib and use:
# import matplotlib.pyplot as plt
# plt.scatter(xs[:10000], ys[:10000], c=ps[:10000], cmap='RdBu_r', s=0.1)
# plt.show()
```

## Event Augmentation

```python
import numpy as np

# Load events first
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs = df['x'].to_numpy()
ys = df['y'].to_numpy()
ps = df['polarity'].to_numpy()

# Spatial flip using numpy
xs_flipped = 640 - 1 - xs  # Horizontal flip
ys_flipped = ys.copy()
ps_aug = ps.copy()

print(f"Original events: {len(xs)}")
print(f"Flipped coordinates: x_max={xs_flipped.max()}, x_min={xs_flipped.min()}")

# Note: For timestamp-based augmentation, handle the datetime format properly
# or convert to numeric values first
```

## Neural Network Inference

```python
# Neural network functionality is under development
# Basic data loading for preprocessing:
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
print(f"Loaded {len(df)} events ready for manual preprocessing")
print(f"Event columns: {df.columns}")

# Note: evlib.representations.create_voxel_grid has known issues
# For now, use basic data loading and manual preprocessing
# Neural network models are not currently available
```

## File Format Support

### Text Files
```python
# Standard format: timestamp x y polarity
# 0.1 320 240 1
# 0.2 321 241 -1
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ts, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['timestamp'].to_numpy(), df['polarity'].to_numpy()
```

### HDF5 Files
```python
import numpy as np

# Load events first
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
ts = df['timestamp'].dt.total_seconds().to_numpy().astype(np.float64)

# Save events to HDF5 format
output_path = "quickstart_output.h5"
evlib.save_events_to_hdf5(xs, ys, ts, ps, output_path)

# Load events from HDF5 files:
events_h5 = evlib.load_events(output_path)
df_h5 = events_h5.collect()
xs_h5, ys_h5, ps_h5 = df_h5['x'].to_numpy(), df_h5['y'].to_numpy(), df_h5['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
ts_h5 = df_h5['timestamp'].dt.total_seconds().to_numpy().astype(np.float64)

print("HDF5 round-trip complete!")
```

### Custom Column Mapping
```python
# For files with different column order: x y polarity timestamp
# Note: Column mapping not currently supported in this version
# Use standard format: timestamp x y polarity
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ts, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['timestamp'].to_numpy(), df['polarity'].to_numpy()
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
events = evlib.load_events("data/slider_depth/events.txt")  # 1M+ events
start = time.time()
df = events.collect()
load_time = time.time() - start

print(f"Data loading: {load_time:.3f}s for {len(df):,} events")
print(f"Loading rate: {len(df)/load_time:.0f} events/sec")
print(f"Event columns: {df.columns}")

# Note: Advanced representations have known issues
# Use basic data loading for reliable performance testing
```

## Error Handling

```python
try:
    events = evlib.load_events("nonexistent.txt")
except OSError as e:
    print(f"File error: {e}")

# Example with valid file
try:
    events = evlib.load_events("data/slider_depth/events.txt")
    df = events.collect()
    print(f"Successfully loaded {len(df)} events")
except OSError as e:
    print(f"Error loading events: {e}")
```

## Next Steps

- [Loading Data Guide](../user-guide/loading-data.md)
- [Event Representations Guide](../user-guide/representations.md)
- [Performance Guide](performance.md)
