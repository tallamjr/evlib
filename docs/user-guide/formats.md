# Event Data Formats

This guide covers all supported event data formats in evlib, including format specifications, compatibility notes, and usage examples.

`evlib.load_events` returns a Polars `LazyFrame` for every format. Call `.collect()` to materialise a `DataFrame`. The timestamp column is named `t` and is a Polars Duration in microseconds, so filter it with `evlib.filtering.filter_by_time` or `pl.col("t").dt.total_microseconds()`, never with raw integer comparisons. `x`/`y` are `Int16` and `polarity` is `Int8` (`-1`/`+1`).

## Supported Formats Overview

| Format | Extension | Status | Use Case |
|--------|-----------|--------|----------|
| **Text** | `.txt`, `.csv` | Production | Human-readable, debugging |
| **HDF5** | `.h5`, `.hdf5` | Production (included in macOS/Linux wheels since 0.13.1) | Large datasets, fast I/O |
| **EVT2** | `.raw` | Production (matches OpenEB exactly) | Prophesee cameras (Gen 1-3) |
| **EVT3** | `.raw`, `.evt3` | Production | Prophesee cameras (Gen 4+) |
| **AEDAT** | `.aedat`, `.aedat4` | Production | iniVation cameras |
| **AER** | `.aer` | Production | Address Event Representation |

Since 0.13.1, the macOS and Linux PyPI wheels statically link HDF5, so HDF5 reading (including Prophesee ECF-compressed data) works out of the box with `pip install evlib`; no system HDF5 or `hdf5plugin` needed. Source builds opt in with `--features hdf5-static` (built from source, no system HDF5) or `--features hdf5` (dynamic, needs a system install); the other formats work without either feature. On Windows, HDF5 support is not available in the native reader; use `h5py` directly for HDF5 I/O there.

## Format Specifications

### Text Format (.txt, .csv)

**Structure:**
```
# Optional header lines
timestamp x y polarity
0.000100 320 240 1
0.000200 321 241 -1
0.000300 319 239 1
```

**On-disk layout** (one event per line): `timestamp x y polarity`, with `timestamp` in seconds. After loading, the in-memory columns are `t` (Duration in microseconds), `x`/`y` (Int16) and `polarity` (Int8, `-1`/`+1`).

**Loading:**
```python
import evlib

# High-level loading (recommended) - returns Polars LazyFrame
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Access DataFrame columns as NumPy arrays.
# t is a Duration column; convert to seconds before treating it as a number.
xs = df['x'].to_numpy()
ys = df['y'].to_numpy()
ps = df['polarity'].to_numpy()
ts = df['t'].dt.total_seconds().to_numpy()
```

**Advantages:**
- Human-readable
- Easy to inspect and debug
- Platform-independent
- No special tools required

**Disadvantages:**
- Large file sizes
- Slow loading for large datasets
- ASCII parsing overhead

### HDF5 Format (.h5, .hdf5)

**Structure:**
```
file.h5
├── events/
│   ├── x (uint16 array)
│   ├── y (uint16 array)
│   ├── t (float64 array)
│   └── p (int8 array)
└── metadata/
    ├── width (int32)
    ├── height (int32)
    └── duration (float64)
```

**Loading:**
```python
import numpy as np

# Load HDF5 file with Polars (recommended)
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Access DataFrame columns as NumPy arrays with correct dtypes
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs = df['x'].to_numpy().astype(np.int64)
ys = df['y'].to_numpy().astype(np.int64)
ps = df['polarity'].to_numpy().astype(np.int64)
# Convert Duration timestamps to seconds (float64)
ts = df['t'].dt.total_seconds().to_numpy().astype(np.float64)

# Save to HDF5
output_path = "formats_output.h5"
evlib.save_events_to_hdf5(xs, ys, ts, ps, output_path)
print(f"Successfully saved {len(xs)} events to {output_path}")
```

**Advantages:**
- 3-5x faster loading than text
- 10x smaller file sizes
- Perfect round-trip compatibility
- Metadata support
- Cross-platform binary format

**Disadvantages:**
- Not available on Windows via the native reader (use `h5py` there)
- Not human-readable
- Source builds need `--features hdf5-static` or `--features hdf5` (the PyPI wheel already includes it)

### EVT2 Format (.raw)

**Status:** Production ready. evlib's EVT2 decode is byte-identical to the OpenEB reference decoder, guarded by an OpenEB conformance gate (`tests/test_openeb_conformance.py`).

**Specification:**
- **Source**: Prophesee cameras (Gen 1-3)
- **Encoding**: Binary, little-endian
- **Resolution**: Up to 1280x720
- **Event Types**: CD (Change Detection) plus the trigger and time-base event types

**Loading:**
```python
import evlib

# EVT2 files are automatically detected.
events = evlib.load_events("data/prophesee/samples/evt2/80_balls.raw")
df = events.collect()
print(f"Loaded {len(df)} events")

# Access columns as NumPy arrays (t is a Duration column).
xs = df['x'].to_numpy()
ys = df['y'].to_numpy()
ps = df['polarity'].to_numpy()
ts = df['t'].dt.total_seconds().to_numpy()
```

### EVT3 Format (.evt3)

**Status:** SUCCESS: Production ready

**Specification:**
- **Source**: Prophesee cameras (Gen 4+)
- **Encoding**: 16-bit binary words, little-endian
- **Header**: Text-based with metadata
- **Event Types**: TIME_HIGH, TIME_LOW, Y_ADDR, X_ADDR, Vector events

**Header Format:**
```
% evt 3.0
% format EVT3;height=H;width=W
% geometry WxH
% camera_integrator_name Prophesee
% generation 4.2
% end
```

**Binary Structure:**
Each event consists of 4 × 16-bit words:
1. **TIME_HIGH**: `(timestamp >> 12) << 4 | 0x8`
2. **TIME_LOW**: `(timestamp & 0xFFF) << 4 | 0x6`
3. **Y_ADDR**: `(y & 0x7FF) << 4 | 0x0`
4. **X_ADDR**: `(polarity_bit << 15) | (x & 0x7FF) << 4 | 0x2`

**Loading** (EVT3 samples are large and gitignored, so this example is not run in CI):
```python notest
import evlib

# EVT3 files are automatically detected.
events = evlib.load_events("data/prophesee/samples/evt3/pedestrians.raw")
df = events.collect()
print(f"Loaded {len(df)} events")
print(f"Columns: {df.columns}")  # ['x', 'y', 't', 'polarity']

# Access columns as NumPy arrays (t is a Duration column).
xs = df['x'].to_numpy()
ys = df['y'].to_numpy()
ps = df['polarity'].to_numpy()
ts = df['t'].dt.total_seconds().to_numpy()
```

**Key Features:**
- SUCCESS: Complete specification compliance
- SUCCESS: Memory-efficient NumPy arrays
- SUCCESS: All event types supported
- SUCCESS: Robust error handling
- SUCCESS: 8/8 tests passing

**Performance:**
- Memory efficient with array-based storage
- Fast batch processing
- Suitable for machine learning pipelines

### AEDAT Format (.aedat)

**Status:** SUCCESS: Production ready

**Specification:**
- **Source**: iniVation cameras (DAVIS, DVS)
- **Encoding**: Binary with headers
- **Version**: AEDAT 2.0 and 3.0 supported

**Loading** (provide your own AEDAT recording):
```python notest
import evlib

events = evlib.load_events("recording.aedat4")
df = events.collect()

xs = df['x'].to_numpy()
ys = df['y'].to_numpy()
ps = df['polarity'].to_numpy()
ts = df['t'].dt.total_seconds().to_numpy()
```

**Features:**
- Address event representation
- Coordinate decoding
- Timestamp reconstruction
- Polarity extraction

### AER Format (.aer)

**Status:** SUCCESS: Production ready

**Specification:**
- **Source**: Address Event Representation
- **Encoding**: Binary address events
- **Compatibility**: Multiple AER variations

**Loading** (provide your own AER recording):
```python notest
import evlib

events = evlib.load_events("recording.aer")
df = events.collect()

xs = df['x'].to_numpy()
ys = df['y'].to_numpy()
ps = df['polarity'].to_numpy()
ts = df['t'].dt.total_seconds().to_numpy()
```

## Format Detection

evlib automatically detects file formats based on content analysis:

```python
# Automatic format detection
format_info = evlib.formats.detect_format("data/slider_depth/events.txt")

print(f"Detected format: {format_info}")

# Format detection also happens automatically when loading
events = evlib.load_events("data/slider_depth/events.txt")  # Auto-detects format
df = events.collect()
```

**Detection Results:**
- **Text files**: High confidence (>0.9)
- **HDF5 files**: High confidence (>0.9)
- **EVT2 files**: High confidence (>0.8)
- **EVT3 files**: High confidence (>0.95)

## Common Issues and Solutions

### Polarity Encoding Mismatch

**Problem**: Real data files often use 0/1 polarity encoding instead of expected -1/1.

**Files Affected:**
- Text files: `0.003811000 96 133 0` (0=negative, 1=positive)
- HDF5 files: May contain 0/1 values
- EVT2 files: Binary encoding varies

**Actual behaviour:** `load_events` does **not** perform automatic polarity conversion. The polarity value in the loaded DataFrame reflects the on-disk encoding for each format:

- **EVT2 (`.raw`)**: polarity is `-1`/`+1` (the Rust reader converts the binary `0`/`1` bit).
- **Text files (e.g. `slider_depth/events.txt`)**: polarity is `0`/`1`, matching the file contents.
- **HDF5 files**: polarity depends on how the file was written.

```python
import evlib
import numpy as np

# Text/slider_depth: polarity is 0/1
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
ps = df['polarity'].to_numpy()
print(f"slider_depth polarities: {np.unique(ps)}")  # [0, 1]

# EVT2: polarity is -1/+1
events = evlib.load_events("data/prophesee/samples/evt2/80_balls.raw")
df = events.collect()
ps = df['polarity'].to_numpy()
print(f"EVT2 polarities: {np.unique(ps)}")  # [-1, 1]
```

If your downstream code requires a specific encoding, convert explicitly:

```python
import evlib
import polars as pl

df = evlib.load_events("data/prophesee/samples/evt2/80_balls.raw").collect()
# Convert 0/1 to -1/+1 (a no-op here since EVT2 already loads as -1/1)
df = df.with_columns(
    pl.when(pl.col("polarity") == 0).then(-1).otherwise(pl.col("polarity")).alias("polarity")
)
```

### HDF5 Dataset Organization

**Problem**: HDF5 files may have different internal dataset structures.

**Solution:**
```python
# Inspect HDF5 structure (example with any HDF5 file)
import h5py
import hdf5plugin

# First create a sample HDF5 file to inspect
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
xs = df['x'].to_numpy().astype(np.int64)
ys = df['y'].to_numpy().astype(np.int64)
ps = df['polarity'].to_numpy().astype(np.int64)
ts = df['t'].dt.total_seconds().to_numpy().astype(np.float64)
evlib.save_events_to_hdf5(xs, ys, ts, ps, "sample.h5")

print("Created HDF5 file with structure:")
print("  /events/x: event x coordinates")
print("  /events/y: event y coordinates")
print("  /events/t: event timestamps")
print("  /events/p: event polarities")

# Now inspect the structure
with h5py.File("sample.h5", "r") as f:
    print("HDF5 structure:")
    f.visititems(print)

    # Check dataset organization
    if "events" in f:
        print("Standard evlib format")
    elif "x" in f and "y" in f:
        print("Flat dataset format")
    else:
        print("Unknown HDF5 organization")
```

## Performance Comparison

### Loading Speed Benchmarks

Based on testing with real data files:

| Format | File Size | Loading Time | Memory Usage | Notes |
|--------|-----------|--------------|--------------|-------|
| Text | 22MB | 2.1s | 180MB | Baseline |
| HDF5 | 6MB | 0.4s | 160MB | 5x faster |
| EVT2 | 526MB | 8.2s | 1.2GB | Large files |
| EVT3 | 45MB | 1.1s | 320MB | Efficient |

**Test Environment:**
- 1M+ events dataset
- Apple M1 Pro
- 16GB RAM
- Python 3.12

### Memory Efficiency

**Event storage sizes:**
- Text: ~180 bytes per event (ASCII overhead)
- HDF5: ~16 bytes per event (binary + compression)
- EVT2: ~8 bytes per event (raw binary)
- EVT3: ~8 bytes per event (packed binary)

**Memory usage after loading:**
- NumPy arrays: ~24 bytes per event
- Event objects: ~120 bytes per event (avoid)

## Best Practices

### 1. Format Selection

```python
# Choose format based on use case
def recommend_format(file_size_mb, use_case):
    if use_case == "debugging":
        return "Text (.txt)"
    elif file_size_mb < 10:
        return "Text (.txt) or HDF5 (.h5)"
    elif file_size_mb < 100:
        return "HDF5 (.h5)"
    else:
        return "HDF5 (.h5) with chunked loading"
```

### 2. Data Conversion Pipeline

```python
def convert_to_hdf5(input_file, output_file):
    """Convert any format to HDF5 for performance"""
    import numpy as np

    # Load with automatic format detection using high-level API
    events = evlib.load_events(input_file)
    df = events.collect()

    # Validate data (note: text format loads polarity as 0/1; EVT2 as -1/+1)
    assert len(df) > 0, "No events loaded"
    assert df['polarity'].is_in([0, 1, -1]).all(), "Invalid polarities"

    # Convert to NumPy for saving (t is a Duration column; convert to seconds)
    xs = df['x'].to_numpy().astype(np.int64)
    ys = df['y'].to_numpy().astype(np.int64)
    ts = df['t'].dt.total_seconds().to_numpy().astype(np.float64)
    ps = df['polarity'].to_numpy().astype(np.int64)

    # Save as HDF5
    evlib.save_events_to_hdf5(xs, ys, ts, ps, output_file)

    # Verify round-trip
    events2 = evlib.load_events(output_file)
    df2 = events2.collect()

    assert len(df) == len(df2), "Event count mismatch"
    print(f"SUCCESS: Converted {len(df)} events to HDF5")
```

### 3. Robust Loading

```python
def load_events_robust(file_path):
    """Load events with comprehensive error handling"""
    try:
        # Try format detection
        format_info = evlib.formats.detect_format(file_path)
        print(f"Detected format: {format_info}")

        # Try high-level API first (handles most cases)
        try:
            events = evlib.load_events(file_path)
            df = events.collect()

            # Validate results
            if len(df) == 0:
                raise ValueError("No events loaded")

            print(f"Successfully loaded {len(df)} events")
            return df

        except Exception as e1:
            print(f"Loading failed: {e1}")
            raise e1

    except Exception as e:
        print(f"ERROR: Failed to load {file_path}: {e}")
        return None
```

### 4. Memory Management

```python
def process_large_file(file_path, time_window=1.0):
    """Process large files in time windows"""
    import polars as pl
    import numpy as np

    # Get duration estimate efficiently
    events = evlib.load_events(file_path)
    time_stats = events.select([
        pl.col("t").min().alias("t_min"),
        pl.col("t").max().alias("t_max")
    ]).collect()

    t_min = time_stats["t_min"][0].total_seconds()
    t_max = time_stats["t_max"][0].total_seconds()
    duration = t_max - t_min

    # Process in time windows using filtering
    results = []
    for t_start in np.arange(0, duration, time_window):
        t_end = min(t_start + time_window, duration)

        # Use the filtering API for time windows
        import evlib.filtering as evf
        events = evlib.load_events(file_path)
        window_events = evf.filter_by_time(
            events, t_start=float(t_start), t_end=float(t_end)
        )

        window_df = window_events.collect()
        if len(window_df) > 0:
            result = process_time_window(window_df)
            results.append(result)

    return results
```

## Future Development

### Planned Improvements

1. **Streaming Support**
   - Chunked reading for very large files
   - Progress reporting
   - Memory-mapped file access

2. **Format Extensions**
   - Direct camera integration
   - Real-time streaming protocols
   - Custom format plugins

### Contributing

To add support for a new format:

1. Implement format detection in `src/ev_formats/format_detector.rs`
2. Add reader implementation in `src/ev_formats/`
3. Create comprehensive tests with real data
4. Update documentation and examples

See the [Contributing Guide](../development/contributing.md) for detailed instructions.

## Summary

evlib provides robust support for multiple event data formats, with automatic format detection and conversion capabilities. All formats are production ready, and EVT2 decode matches the OpenEB reference exactly. The HDF5 format is recommended for performance-critical applications and large datasets; it is included by default in the macOS and Linux wheels, with `--features hdf5-static` and `--features hdf5` available for source builds.

For questions or issues with specific formats, please check the [Testing Documentation](../development/testing.md) or file an issue on GitHub.
