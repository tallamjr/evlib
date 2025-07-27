# Loading Data

This guide covers everything you need to know about loading event data with evlib.

## Quick Start

```python
import evlib

# Load events as Polars LazyFrame (recommended)
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
print(f"Loaded {len(df)} events")

# Or access DataFrame columns as NumPy arrays
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
ts = df['timestamp'].dt.total_seconds().to_numpy()
print(f"Loaded {len(xs)} events")
```

## Supported File Formats

### Text Files (.txt, .csv)

The most common format for event data:

```
# Standard format: timestamp x y polarity
0.000100 320 240 1
0.000200 321 241 -1
0.000300 319 239 1
```

**Loading text files:**
```python
# High-performance Polars LazyFrame (recommended)
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Or access DataFrame columns as NumPy arrays
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
ts = df['timestamp'].dt.total_seconds().to_numpy()
```

### HDF5 Files (.h5, .hdf5)

Efficient binary format with fast loading:

```python
# HDF5 format provides efficient binary storage
# Load existing HDF5 datasets (e.g., from eTram dataset)
# events = evlib.load_events("data/eTram/h5/val_2/val_night_011_td.h5")
# df = events.collect()

# To save events to HDF5 format:
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs = df['x'].to_numpy().astype(np.int64)
ys = df['y'].to_numpy().astype(np.int64)
ps = df['polarity'].to_numpy().astype(np.int64)
ts = df['timestamp'].dt.total_seconds().to_numpy().astype(np.float64)

# Save to HDF5 for efficient storage
evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, "output.h5")
print(f"Saved {len(xs)} events to HDF5 format")

# Note: The save function uses internal type conversions (uint16/int8)
# For loading, use existing dataset HDF5 files which are fully compatible
```

**HDF5 advantages:**
- 3-5x faster loading than text files
- Smaller file sizes (up to 10x compression)
- Perfect round-trip compatibility
- Metadata support

## Advanced Filtering

### Time Window Filtering

Load only events within a specific time range:

```python
# Load events between 1.0 and 5.0 seconds using filtering
import evlib.filtering as evf
events = evlib.load_events("data/slider_depth/events.txt")
filtered_events = evf.filter_by_time(events, t_start=1.0, t_end=5.0)
df = events.collect()

# Or load with parameters
events = evlib.load_events("data/slider_depth/events.txt", t_start=1.0, t_end=5.0)
```

**Use cases:**
- Processing specific time segments
- Memory-efficient loading of large files
- Temporal analysis of event streams

### Spatial Filtering

Filter events by pixel coordinates:

```python
# Load events in center region only
import evlib.filtering as evf
events = evlib.load_events("data/slider_depth/events.txt")
filtered_events = evf.filter_by_roi(events, x_min=200, x_max=440, y_min=120, y_max=360)
df = filtered_events.collect()

# Or use load_events with parameters
events = evlib.load_events("data/slider_depth/events.txt", min_x=200, max_x=440, min_y=120, max_y=360)
```

**Applications:**
- Region of interest analysis
- Removing border artifacts
- Focus on specific image areas

### Polarity Filtering

Separate positive and negative events:

```python
# Load only positive (ON) events
import evlib.filtering as evf
events = evlib.load_events("data/slider_depth/events.txt")
pos_events = evf.filter_by_polarity(events, polarity=1)
pos_df = pos_events.collect()

# Load only negative (OFF) events
neg_events = evf.filter_by_polarity(events, polarity=0)  # Note: using 0 for negative in this dataset
neg_df = neg_events.collect()
```

**Why use polarity filtering:**
- Analyze ON vs OFF events separately
- Reduce data size for specific analysis
- Implement polarity-specific algorithms

### Combined Filtering

All filters can be combined:

```python
# Complex filtering example using preprocessing pipeline
import evlib.filtering as evf
processed_events = evf.preprocess_events(
    "data/slider_depth/events.txt",
    t_start=2.0, t_end=8.0,      # Time window
    roi=(100, 540, 50, 430),     # Spatial bounds (x_min, x_max, y_min, y_max)
    polarity=1,                  # Positive events only
    remove_hot_pixels=True,
    remove_noise=True
)
df = processed_events.collect()
```

## Custom File Formats

### Different Column Orders

If your files have different column arrangements:

```python
# File format: x y polarity timestamp
# 320 240 1 0.000100
# Note: Column specification may require direct Rust access
# Note: Custom column mapping requires direct format access
# Use standard evlib.load_events for most cases
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
ts = df['timestamp'].dt.total_seconds().to_numpy()
```

### Files with Headers

Skip header lines in your files:

```python
# File with header:
# # timestamp x y polarity
# 0.000100 320 240 1
# Note: Header handling may require direct Rust access
# Note: Header handling requires preprocessing or manual file handling
# Standard evlib.load_events handles most common formats
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
ts = df['timestamp'].dt.total_seconds().to_numpy()
```

### Multiple Headers

```python
# File with multiple header lines
# Note: Header handling may require direct Rust access
# Note: Complex header handling requires preprocessing
# Standard evlib.load_events handles most common formats
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
ts = df['timestamp'].dt.total_seconds().to_numpy()
```

## Performance Optimization

### Memory Management

For very large files, use chunked loading:

```python
# Process large files efficiently with time windows
events = evlib.load_events("data/slider_depth/events.txt", t_start=0.0, t_end=10.0)
df = events.collect()  # Uses optimal Polars engine

# For very large files, use streaming
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Streaming engine for large data
```

### Format Selection

Choose the right format for your needs:

| Format | Best for | Loading Speed | File Size |
|--------|----------|---------------|-----------|
| Text (.txt) | Human readability, debugging | Baseline | Large |
| HDF5 (.h5) | Performance, large datasets | 3-5x faster | 10x smaller |

### Filtering Performance

Apply filters during loading, not after:

```python
# GOOD: Filter during loading using high-level API
import evlib.filtering as evf
events = evlib.load_events("data/slider_depth/events.txt")
filtered_events = evf.filter_by_time(events, t_start=1.0, t_end=2.0)
df = filtered_events.collect()

# GOOD: Use Polars filtering (lazy evaluation)
events = evlib.load_events("data/slider_depth/events.txt")
import polars as pl
filtered = events.filter(
    (pl.col("timestamp") >= 1.0) & (pl.col("timestamp") <= 2.0)
)
df = filtered.collect()

# AVOID: Load all then filter with NumPy
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
ts = df['timestamp'].dt.total_seconds().to_numpy()
# Convert timestamps to seconds for comparison
ts_seconds = ts.astype('float64') / 1e6  # Convert microseconds to seconds
mask = (ts_seconds >= 1.0) & (ts_seconds <= 2.0)
xs, ys, ts, ps = xs[mask], ys[mask], ts[mask], ps[mask]
```

## Error Handling

### Robust Loading

Always handle potential errors:

```python
def load_events_safely(file_path):
    try:
        events = evlib.load_events(file_path)
        df = events.collect()
        print(f"SUCCESS: Successfully loaded {len(df)} events")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return None
    except OSError as e:
        print(f"ERROR: Invalid file format: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return None
```

### Validation

Validate loaded data:

```python
def validate_events(df):
    """Validate event data integrity"""
    assert len(df) > 0, "No events loaded"
    assert all(col in df.columns for col in ['x', 'y', 'timestamp', 'polarity']), "Required columns missing"
    assert (df['x'] >= 0).all(), "X coordinates must be non-negative"
    assert (df['y'] >= 0).all(), "Y coordinates must be non-negative"
    assert df['timestamp'].is_sorted(), "Timestamps must be sorted"
    assert df['polarity'].is_in([1, -1]).all(), "Polarities must be +1 or -1"
    print(f"SUCCESS: Validation passed for {len(df)} events")
```

## Real-World Examples

### Camera Calibration Dataset

```python
# Load slider_depth dataset (1M+ events)
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Basic statistics using Polars
print(f"Events: {len(df):,}")
print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.2f} seconds")
print(f"Resolution: {df['x'].max()+1} x {df['y'].max()+1}")
print(f"Event rate: {len(df)/(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.0f} events/sec")
```

### Temporal Segmentation

```python
# Process video in 0.1 second segments
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
segment_length = 0.1
n_segments = int(duration / segment_length)

for i in range(n_segments):
    t_start = df['timestamp'].min().total_seconds() + i * segment_length
    t_end = t_start + segment_length

    # Load segment using filtering
    segment_events = evlib.filter_by_time("data/slider_depth/events.txt", t_start=t_start, t_end=t_end)
    seg_df = segment_events.collect()

    # Process segment
    if len(seg_df) > 0:
        process_segment(seg_df)
```

### Multi-Resolution Analysis

```python
# Load events at different spatial resolutions
full_res = evlib.load_events("data/slider_depth/events.txt")
full_df = full_res.collect()

# Quarter resolution (downsample by 4)
quarter_res = evlib.filtering.filter_by_roi(evlib.load_events("data/slider_depth/events.txt"), x_min=0, x_max=159, y_min=0, y_max=119)
quarter_df = quarter_res.collect()
```

## Best Practices

### 1. File Organization
```
datasets/
├── raw/
│   ├── experiment1_events.txt
│   └── experiment2_events.txt
├── processed/
│   ├── experiment1_events.h5    # Converted to HDF5
│   └── experiment2_events.h5
└── filtered/
    ├── experiment1_positive.h5  # Polarity filtered
    └── experiment1_time_0_5.h5  # Time filtered
```

### 2. Data Pipeline
```python
def create_data_pipeline(input_file, output_dir):
    # 1. Load and validate
    events = evlib.load_events(input_file)
    df = events.collect()
    validate_events(df)

    # 2. Save as HDF5 for faster future loading
    h5_file = f"{output_dir}/events.h5"
    # Convert timestamps to seconds for saving
    ts_seconds = df['timestamp'].dt.total_seconds().to_numpy()
    evlib.formats.save_events_to_hdf5(
        df['x'].to_numpy(), df['y'].to_numpy(),
        ts_seconds, df['polarity'].to_numpy(),
        h5_file
    )

    # 3. Create filtered versions
    pos_file = f"{output_dir}/positive_events.h5"
    pos_events = evlib.filter_by_polarity(input_file, polarity=1)
    pos_df = pos_events.collect()
    # Convert timestamps to seconds for saving
    pos_ts_seconds = pos_df['timestamp'].dt.total_seconds().to_numpy()
    evlib.formats.save_events_to_hdf5(
        pos_df['x'].to_numpy(), pos_df['y'].to_numpy(),
        pos_ts_seconds, pos_df['polarity'].to_numpy(),
        pos_file
    )

    return h5_file, pos_file
```

### 3. Memory-Efficient Processing
```python
def process_large_dataset(file_path, time_window=1.0):
    """Process large dataset in time windows"""
    import polars as pl

    # Get total duration efficiently
    events = evlib.load_events(file_path)

    # Get min/max timestamps without collecting all data
    time_stats = events.select([
        pl.col("timestamp").min().alias("t_min"),
        pl.col("timestamp").max().alias("t_max")
    ]).collect()

    t_start = time_stats["t_min"][0].total_seconds()
    t_end = time_stats["t_max"][0].total_seconds()

    # Process in time windows
    current_time = t_start
    while current_time < t_end:
        window_end = current_time + time_window

        try:
            window_events = evlib.filter_by_time(
                file_path,
                t_start=current_time,
                t_end=window_end
            )
            df = window_events.collect()

            if len(df) > 0:
                # Process this time window
                result = process_events(df)
                yield current_time, result

        except Exception as e:
            print(f"Error processing window {current_time:.2f}s: {e}")

        current_time = window_end
```

## Troubleshooting

### Common Issues

**Problem**: `FileNotFoundError`
```python
# Solution: Check file path and existence
import os
if os.path.exists("data/slider_depth/events.txt"):
    events = evlib.load_events("data/slider_depth/events.txt")
    df = events.collect()
else:
    print("File does not exist")
```

**Problem**: `OSError: Invalid timestamp`
```python
# Solution: Check file format and column mapping
# Note: Column specification requires direct Rust access
# Note: Column specification requires preprocessing or format handling
# Standard evlib.load_events handles standard formats
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
ts = df['timestamp'].dt.total_seconds().to_numpy()
```

**Problem**: Memory errors with large files
```python
# Solution: Use time window filtering
events = evlib.filtering.filter_by_time(evlib.load_events("data/slider_depth/events.txt"), t_start=0.0, t_end=10.0)
df = events.collect()  # Uses optimal engine for large data
```

## Next Steps

- [Event Representations](representations.md): Convert events to grids and images
