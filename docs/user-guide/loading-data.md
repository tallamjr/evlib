# Loading Data

This guide covers everything you need to know about loading event data with evlib.

## Quick Start

```python
import evlib

# Load events from any supported format
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
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
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
```

### HDF5 Files (.h5, .hdf5)

Efficient binary format with fast loading:

```python
# Load HDF5 file
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.h5")

# Save to HDF5 for faster future loading
evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, "output.h5")
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
# Load events between 1.0 and 5.0 seconds
xs, ys, ts, ps = evlib.formats.load_events_filtered("data/slider_depth/events.txt", t_start=1.0,
    t_end=5.0
)
```

**Use cases:**
- Processing specific time segments
- Memory-efficient loading of large files
- Temporal analysis of event streams

### Spatial Filtering

Filter events by pixel coordinates:

```python
# Load events in center region only
xs, ys, ts, ps = evlib.formats.load_events_filtered("data/slider_depth/events.txt", min_x=200, max_x=440,  # X bounds
    min_y=120, max_y=360   # Y bounds
)
```

**Applications:**
- Region of interest analysis
- Removing border artifacts
- Focus on specific image areas

### Polarity Filtering

Separate positive and negative events:

```python
# Load only positive (ON) events
pos_events = evlib.formats.load_events_filtered("data/slider_depth/events.txt", polarity=1)

# Load only negative (OFF) events
neg_events = evlib.formats.load_events_filtered("data/slider_depth/events.txt", polarity=-1)
```

**Why use polarity filtering:**
- Analyze ON vs OFF events separately
- Reduce data size for specific analysis
- Implement polarity-specific algorithms

### Combined Filtering

All filters can be combined:

```python
# Complex filtering example
xs, ys, ts, ps = evlib.formats.load_events_filtered("data/slider_depth/events.txt", t_start=2.0, t_end=8.0,      # Time window
    min_x=100, max_x=540,        # Spatial bounds
    min_y=50, max_y=430,
    polarity=1,                  # Positive events only
    sort=True                    # Sort by timestamp
)
```

## Custom File Formats

### Different Column Orders

If your files have different column arrangements:

```python
# File format: x y polarity timestamp
# 320 240 1 0.000100
xs, ys, ts, ps = evlib.formats.load_events(
    "custom_format.txt",
    x_col=0, y_col=1, p_col=2, t_col=3
)
```

### Files with Headers

Skip header lines in your files:

```python
# File with header:
# # timestamp x y polarity
# 0.000100 320 240 1
xs, ys, ts, ps = evlib.formats.load_events(
    "events_with_header.txt",
    header_lines=1
)
```

### Multiple Headers

```python
# File with multiple header lines
xs, ys, ts, ps = evlib.formats.load_events(
    "events_complex_header.txt",
    header_lines=3  # Skip first 3 lines
)
```

## Performance Optimization

### Memory Management

For very large files, use chunked loading:

```python
# Process large files in chunks
xs, ys, ts, ps = evlib.formats.load_events(
    "massive_dataset.txt",
    chunk_size=100000  # 100k events per chunk
)
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
# SUCCESS: GOOD: Filter during loading
xs, ys, ts, ps = evlib.formats.load_events_filtered("data/slider_depth/events.txt", t_start=1.0, t_end=2.0
)

# ERROR: AVOID: Load all then filter
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
mask = (ts >= 1.0) & (ts <= 2.0)
xs, ys, ts, ps = xs[mask], ys[mask], ts[mask], ps[mask]
```

## Error Handling

### Robust Loading

Always handle potential errors:

```python
def load_events_safely(file_path):
    try:
        xs, ys, ts, ps = evlib.formats.load_events(file_path)
        print(f"SUCCESS: Successfully loaded {len(xs)} events")
        return xs, ys, ts, ps
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
def validate_events(xs, ys, ts, ps):
    """Validate event data integrity"""
    assert len(xs) == len(ys) == len(ts) == len(ps), "Array lengths must match"
    assert len(xs) > 0, "No events loaded"
    assert np.all(xs >= 0) and np.all(ys >= 0), "Coordinates must be non-negative"
    assert np.all(np.diff(ts) >= 0), "Timestamps must be sorted"
    assert np.all((ps == 1) | (ps == -1)), "Polarities must be +1 or -1"
    print(f"SUCCESS: Validation passed for {len(xs)} events")
```

## Real-World Examples

### Camera Calibration Dataset

```python
# Load slider_depth dataset (1M+ events)
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

# Basic statistics
print(f"Events: {len(xs):,}")
print(f"Duration: {ts.max() - ts.min():.2f} seconds")
print(f"Resolution: {xs.max()+1} x {ys.max()+1}")
print(f"Event rate: {len(xs)/(ts.max()-ts.min()):.0f} events/sec")
```

### Temporal Segmentation

```python
# Process video in 0.1 second segments
duration = ts.max() - ts.min()
segment_length = 0.1
n_segments = int(duration / segment_length)

for i in range(n_segments):
    t_start = ts.min() + i * segment_length
    t_end = t_start + segment_length

    # Load segment
    seg_xs, seg_ys, seg_ts, seg_ps = evlib.formats.load_events_filtered("data/slider_depth/events.txt", t_start=t_start,
        t_end=t_end
    )

    # Process segment
    if len(seg_xs) > 0:
        process_segment(seg_xs, seg_ys, seg_ts, seg_ps)
```

### Multi-Resolution Analysis

```python
# Load events at different spatial resolutions
full_res = evlib.formats.load_events("data/slider_depth/events.txt")

# Quarter resolution (downsample by 4)
quarter_res = evlib.formats.load_events_filtered("data/slider_depth/events.txt", min_x=0, max_x=159,    # 640/4 = 160
    min_y=0, max_y=119     # 480/4 = 120
)
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
    xs, ys, ts, ps = evlib.formats.load_events(input_file)
    validate_events(xs, ys, ts, ps)

    # 2. Save as HDF5 for faster future loading
    h5_file = f"{output_dir}/events.h5"
    evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, h5_file)

    # 3. Create filtered versions
    pos_file = f"{output_dir}/positive_events.h5"
    pos_xs, pos_ys, pos_ts, pos_ps = evlib.formats.load_events(
        input_file, polarity=1
    )
    evlib.formats.save_events_to_hdf5(pos_xs, pos_ys, pos_ts, pos_ps, pos_file)

    return h5_file, pos_file
```

### 3. Memory-Efficient Processing
```python
def process_large_dataset(file_path, time_window=1.0):
    """Process large dataset in time windows"""
    # Get total duration without loading all events
    sample_events = evlib.formats.load_events(file_path, chunk_size=1000)

    # Estimate time range (rough)
    t_start = 0.0
    t_end = sample_events[2].max() * 10  # Rough estimate

    # Process in time windows
    current_time = t_start
    while current_time < t_end:
        window_end = current_time + time_window

        try:
            xs, ys, ts, ps = evlib.formats.load_events(
                file_path,
                t_start=current_time,
                t_end=window_end
            )

            if len(xs) > 0:
                # Process this time window
                result = process_events(xs, ys, ts, ps)
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
    xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
else:
    print("File does not exist")
```

**Problem**: `OSError: Invalid timestamp`
```python
# Solution: Check file format and column mapping
xs, ys, ts, ps = evlib.formats.load_events(
    "data/slider_depth/events.txt",
    t_col=0, x_col=1, y_col=2, p_col=3  # Specify columns
)
```

**Problem**: Memory errors with large files
```python
# Solution: Use time window filtering
xs, ys, ts, ps = evlib.formats.load_events_filtered("large_file.txt", t_start=0.0, t_end=10.0  # Load only first 10 seconds
)
```

## Next Steps

- [Event Representations](representations.md): Convert events to grids and images
- [Visualization](visualization.md): Plot and display event data
- [Neural Networks](models.md): Use events with deep learning models
