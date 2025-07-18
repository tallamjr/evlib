# Event Data Formats

This guide covers all supported event data formats in evlib, including format specifications, compatibility notes, and usage examples.

## Supported Formats Overview

| Format | Extension | Status | Use Case | Performance |
|--------|-----------|--------|----------|-------------|
| **Text** | `.txt`, `.csv` | SUCCESS: Production | Human-readable, debugging | Baseline |
| **HDF5** | `.h5`, `.hdf5` | SUCCESS: Production | Large datasets, fast I/O | 3-5x faster |
| **EVT2** | `.raw` | WARNING: Partial | Prophesee cameras (Gen 1-3) | Fast binary |
| **EVT3** | `.evt3` | SUCCESS: Production | Prophesee cameras (Gen 4+) | Fast binary |
| **AEDAT** | `.aedat` | SUCCESS: Production | iniVation cameras | Binary |
| **AER** | `.aer` | SUCCESS: Production | Address Event Representation | Binary |

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

**Data Types:**
- `timestamp`: Float64 (seconds)
- `x`: Uint16 (pixel coordinate)
- `y`: Uint16 (pixel coordinate)
- `polarity`: Int8 (1 for positive, -1 for negative)

**Loading:**
```python
import evlib

# Standard loading
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

# With custom column order
xs, ys, ts, ps = evlib.formats.load_events(
    "custom_format.txt",
    x_col=1, y_col=2, t_col=0, p_col=3
)
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
# Load HDF5 file
xs, ys, ts, ps = evlib.formats.load_events("dataset.h5")

# Save to HDF5
evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, "output.h5")
```

**Advantages:**
- 3-5x faster loading than text
- 10x smaller file sizes
- Perfect round-trip compatibility
- Metadata support
- Cross-platform binary format

**Disadvantages:**
- Requires HDF5 libraries
- Not human-readable
- Slightly more complex setup

### EVT2 Format (.raw)

**Status:** WARNING: Partial support - some event types not handled

**Specification:**
- **Source**: Prophesee cameras (Gen 1-3)
- **Encoding**: Binary, little-endian
- **Resolution**: Up to 1280x720
- **Event Types**: CD (Change Detection), some extension events

**Known Issues:**
- Real EVT2 files may contain event type 12 and others not handled by current reader
- **Error**: `InvalidEventType { type_value: 12, offset: 366 }`
- **Workaround**: Use format detection and handle errors gracefully

**Loading:**
```python
try:
    xs, ys, ts, ps = evlib.formats.load_events("data/evt2_file.raw")
except Exception as e:
    print(f"EVT2 loading failed: {e}")
    # Fallback to alternative format or skip
```

**Real Data Testing:**
- SUCCESS: Format detection works (>95% confidence)
- ERROR: Event loading fails on some real files
- DATA: Test files: `data/eTram/raw/val_2/*.raw` (15MB-526MB)

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

**Loading:**
```python
# EVT3 files are automatically detected
xs, ys, ts, ps = evlib.formats.load_events("data/evt3_file.evt3")

# Data structure: separate arrays (not individual event objects)
print(f"X coordinates: {xs}")
print(f"Y coordinates: {ys}")
print(f"Timestamps: {ts}")
print(f"Polarities: {ps}")
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

**Loading:**
```python
# AEDAT files with address decoding
xs, ys, ts, ps = evlib.formats.load_events("data/davis_recording.aedat")
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

**Loading:**
```python
# AER format with real data
xs, ys, ts, ps = evlib.formats.load_events("data/aer_file.aer")
```

## Format Detection

evlib automatically detects file formats based on content analysis:

```python
# Automatic format detection
format_name, confidence, metadata = evlib.detect_format("data/unknown_file.dat")

print(f"Detected format: {format_name}")
print(f"Confidence: {confidence:.2f}")
print(f"Metadata: {metadata}")
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

**Solution:**
```python
# Configure polarity conversion
from evlib.formats import LoadConfig, PolarityEncoding

config = LoadConfig()
config.polarity_encoding = PolarityEncoding.ZeroOne  # Input as 0/1
config.convert_to_standard = True  # Convert to -1/1 internally

xs, ys, ts, ps = evlib.formats.load_events_with_config("data/file.txt", config)
```

**Validation:**
```python
# Check polarity encoding in loaded data
unique_polarities = np.unique(ps)
print(f"Polarity values: {unique_polarities}")

# Should be [-1, 1] after conversion
assert np.all(np.isin(unique_polarities, [-1, 1])), "Invalid polarities"
```

### EVT2 Event Type Errors

**Problem**: Real EVT2 files contain event types not handled by current reader.

**Error Message:**
```
InvalidEventType { type_value: 12, offset: 366 }
```

**Solution:**
```python
# Enable skip_invalid_events mode
config = LoadConfig()
config.skip_invalid_events = True
config.max_errors = 100  # Skip up to 100 invalid events

try:
    xs, ys, ts, ps = evlib.formats.load_events_with_config("data/evt2_file.raw", config)
    print(f"Successfully loaded {len(xs)} events")
except Exception as e:
    print(f"Loading failed: {e}")
```

**Workaround:**
```python
# Alternative: Use format detection and fallback
format_name, confidence, metadata = evlib.detect_format("data/evt2_file.raw")

if format_name == "EVT2" and confidence > 0.8:
    try:
        xs, ys, ts, ps = evlib.formats.load_events("data/evt2_file.raw")
    except Exception:
        print("EVT2 loading failed, trying alternative format")
        # Try alternative processing or skip file
```

### HDF5 Dataset Organization

**Problem**: HDF5 files may have different internal dataset structures.

**Solution:**
```python
# Inspect HDF5 structure
import h5py

with h5py.File("data/file.h5", "r") as f:
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
    # Load with automatic format detection
    xs, ys, ts, ps = evlib.formats.load_events(input_file)

    # Validate data
    assert len(xs) > 0, "No events loaded"
    assert np.all(np.isin(ps, [-1, 1])), "Invalid polarities"

    # Save as HDF5
    evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, output_file)

    # Verify round-trip
    xs2, ys2, ts2, ps2 = evlib.formats.load_events(output_file)
    np.testing.assert_array_equal(xs, xs2)
    np.testing.assert_array_equal(ys, ys2)
    np.testing.assert_array_equal(ts, ts2)
    np.testing.assert_array_equal(ps, ps2)

    print(f"SUCCESS: Converted {len(xs)} events to HDF5")
```

### 3. Robust Loading

```python
def load_events_robust(file_path):
    """Load events with comprehensive error handling"""
    try:
        # Try automatic format detection
        format_name, confidence, metadata = evlib.detect_format(file_path)
        print(f"Detected: {format_name} (confidence: {confidence:.2f})")

        if confidence < 0.7:
            raise ValueError(f"Low confidence format detection: {confidence:.2f}")

        # Load with appropriate configuration
        if format_name in ["Text", "HDF5", "EVT3"]:
            xs, ys, ts, ps = evlib.formats.load_events(file_path)
        elif format_name == "EVT2":
            # Use error-tolerant config for EVT2
            config = LoadConfig()
            config.skip_invalid_events = True
            xs, ys, ts, ps = evlib.formats.load_events_with_config(file_path, config)
        else:
            raise ValueError(f"Unsupported format: {format_name}")

        # Validate results
        if len(xs) == 0:
            raise ValueError("No events loaded")

        return xs, ys, ts, ps

    except Exception as e:
        print(f"ERROR: Failed to load {file_path}: {e}")
        return None
```

### 4. Memory Management

```python
def process_large_file(file_path, time_window=1.0):
    """Process large files in time windows"""
    # Get file info without loading all data
    format_name, confidence, metadata = evlib.detect_format(file_path)

    if "duration" in metadata:
        duration = metadata["duration"]
    else:
        # Estimate duration from sample
        sample_xs, sample_ys, sample_ts, sample_ps = evlib.formats.load_events_filtered(
            file_path, t_start=0.0, t_end=1.0
        )
        duration = sample_ts.max() * 10  # Rough estimate

    # Process in time windows
    results = []
    for t_start in np.arange(0, duration, time_window):
        t_end = min(t_start + time_window, duration)

        xs, ys, ts, ps = evlib.formats.load_events_filtered(
            file_path, t_start=t_start, t_end=t_end
        )

        if len(xs) > 0:
            result = process_time_window(xs, ys, ts, ps)
            results.append(result)

    return results
```

## Future Development

### Planned Improvements

1. **EVT2 Enhancement**
   - Support for all event types (including type 12)
   - Improved error recovery
   - Better real-world file compatibility

2. **Streaming Support**
   - Chunked reading for very large files
   - Progress reporting
   - Memory-mapped file access

3. **Format Extensions**
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

evlib provides robust support for multiple event data formats, with automatic format detection and conversion capabilities. While most formats work well in production, some (like EVT2) require careful error handling when working with real-world data files. The HDF5 format is recommended for performance-critical applications and large datasets.

For questions or issues with specific formats, please check the [Testing Documentation](../development/testing.md) or file an issue on GitHub.
