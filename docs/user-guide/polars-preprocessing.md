# Polars-Based Event Preprocessing

This guide covers evlib's high-performance Polars-based event preprocessing capabilities, designed to replace PyTorch-based preprocessing pipelines like those used in RVT (Recurrent Vision Transformers).

## Overview

The `evlib.representations` module provides high-performance Polars-based implementations of common event camera representations with significant performance improvements over traditional PyTorch approaches. All functions return NumPy arrays but leverage Polars internally for efficient processing.

### Performance Benefits

Based on testing with real event data:

- **evlib Polars Processing**: 2.31s (1.4M events/s)
- **Estimated RVT PyTorch**: ~8.10s (estimated)
- **Performance Speedup**: **3.5x faster**
- **Memory Usage**: 109.3 MB for 69 windows of stacked histograms

### Output Compatibility

- **Shape**: (69, 20, 240, 346) - matches RVT format expectations
- **Data Type**: uint8 for stacked histograms, int8 for mixed density
- **Channel Layout**: Polarity channels × temporal bins × height × width
- **Format**: All functions return NumPy arrays directly (no need for .to_numpy())

## Quick Start

```python
import evlib.representations as evr

# Drop-in replacement for RVT preprocessing
data = evr.preprocess_for_detection(
    "data/sequence.h5",
    representation="stacked_histogram",
    height=480, width=640,
    nbins=10, window_duration_ms=50
)

# All functions return NumPy arrays directly
print(f"Shape: {data.shape}, Type: {type(data)}")
# Output: Shape: (69, 20, 480, 640), Type: <class 'numpy.ndarray'>
```

### Why This API is Better

- **Single import**: All representation functions in one place
- **High performance**: Polars-based processing with NumPy output
- **No conversion needed**: Functions return NumPy arrays directly
- **Clean API**: Only the essential functions, no legacy clutter

## API Reference

### High-Level Functions

#### `preprocess_for_detection()`

Drop-in replacement for RVT preprocessing pipeline.

```python
data = evr.preprocess_for_detection(
    events_path,
    representation="stacked_histogram",  # or "mixed_density", "voxel_grid"
    height=480, width=640,
    **kwargs  # Representation-specific parameters
)
```

**Parameters:**
- `events_path`: Path to event file
- `representation`: Type of representation
- `height`, `width`: Output dimensions
- `**kwargs`: Additional parameters passed to the specific representation function

#### `create_stacked_histogram()`

Main function for temporal histogram creation with windowing.

```python
hist = evr.create_stacked_histogram(
    events_path,
    height=480, width=640,
    nbins=10,                    # Temporal bins per window
    window_duration_ms=50.0,     # Window duration
    stride_ms=None,              # Defaults to window_duration_ms
    count_cutoff=10              # Max count per bin
)
```

**Parameters:**
- `events`: Path to event file or Polars LazyFrame
- `height`, `width`: Output dimensions
- `nbins`: Number of temporal bins per window
- `window_duration_ms`: Duration of each window in milliseconds
- `stride_ms`: Stride between windows (defaults to window_duration_ms for non-overlapping)
- `count_cutoff`: Maximum count per bin (None for no limit)

**Returns:**
- numpy array of shape `(num_windows, 2*nbins, height, width)`

#### `create_mixed_density_stack()`

Logarithmic time binning with polarity accumulation.

```python
mixed = evr.create_mixed_density_stack(
    events_path,
    height=480, width=640,
    nbins=10,
    window_duration_ms=50.0,
    count_cutoff=None
)
```

**Parameters:**
- Similar to `create_stacked_histogram()` but uses logarithmic binning
- `count_cutoff`: Maximum absolute value per bin

**Returns:**
- numpy array of shape `(num_windows, nbins, height, width)` with int8 values

#### `create_voxel_grid()`

Traditional voxel grid representation for entire dataset.

```python
voxels = evr.create_voxel_grid(
    events_path,
    height=480, width=346,
    nbins=5
)
```

**Returns:**
- numpy array of shape `(nbins, height, width)`

### Performance Benchmarking

```python
# Compare against RVT performance
results = evr.benchmark_vs_rvt('data/events.h5')
print(f"Speedup: {results['speedup']:.1f}x")
```

## Technical Implementation

### Key Advantages Over PyTorch Approach

1. **Native Groupby Operations**: Polars handles spatial-temporal grouping natively without tensor indexing
2. **Lazy Evaluation**: Only computes what's needed, reducing memory allocations
3. **Optimized Data Types**: Int16 coordinates vs Int64 for better cache performance
4. **No GPU Transfers**: CPU-based processing eliminates memory transfer overhead
5. **Memory Locality**: Better cache utilization for histogram operations

### Algorithm Details

#### Stacked Histogram Creation

1. **Windowing**: Events divided into time windows (configurable duration/stride)
2. **Temporal Binning**: Within each window, events binned by normalized timestamp
3. **Spatial-Temporal Grouping**: Group by (x, y, time_bin, polarity) and count
4. **Channel Layout**: Negative polarity in bins 0..(nbins-1), positive in bins nbins..(2*nbins-1)

#### Mixed Density Implementation

1. **Logarithmic Binning**: Uses `bin = nbins - log(t_norm) / log(0.5)` for temporal distribution
2. **Polarity Accumulation**: Sums signed polarities instead of counting
3. **Cumulative Integration**: Applies cumulative sum from newest to oldest bins

### Data Flow

```
Event File → evlib.load_events() → Polars LazyFrame → Window Processing → NumPy Array
                                                    ↓
                          Temporal Binning ← Spatial Clipping ← Polarity Conversion
                                                    ↓
                          Groupby Aggregation → Dense Histogram → Output Array
```

## Migration from RVT

### Code Changes Required

**Before (RVT):**
```python
from rvt.representations import StackedHistogram
from rvt.data.utils.preprocessing import H5Reader

# RVT preprocessing
reader = H5Reader("data.h5")
stacker = StackedHistogram(bins=10)
hist = stacker(reader.events)
```

**After (evlib):**
```python
import evlib.representations as evr

# evlib preprocessing (drop-in replacement)
hist = evr.create_stacked_histogram(
    "data.h5",
    height=480, width=640,
    nbins=10, window_duration_ms=50
)
```

### Performance Expectations

- **3-5x speed improvement** for typical event camera datasets
- **Reduced memory usage** through lazy evaluation
- **Better scalability** for high-resolution sensors (>1MP)
- **CPU-only processing** eliminates GPU dependency

## Integration Points

### File Format Support

- **HDF5**: Direct integration with evlib format readers (including BLOSC compression)
- **EVT2/3**: Binary Prophesee formats
- **Text**: Space-separated format support
- **Automatic Detection**: No need to specify format manually

### Memory Management

- **Lazy Loading**: Events loaded on-demand for large files
- **Chunked Processing**: Windows processed individually to control memory usage
- **Optimized Types**: Int16 for coordinates, Int8 for polarities, Duration for timestamps

### Error Handling

- **Empty Windows**: Gracefully handled with zero-filled histograms
- **Boundary Clipping**: Coordinates automatically clipped to sensor dimensions
- **Progress Reporting**: Real-time progress for long operations

## Advanced Usage

### Custom Parameters

```python
# High-framerate preprocessing
hist = evr.create_stacked_histogram(
    'events.h5',
    height=480, width=640,
    nbins=15,
    window_duration_ms=33.3,  # ~30 FPS
    stride_ms=16.7,           # 50% overlap
    count_cutoff=15
)
```

### Production Pipeline

```python
import evlib.representations as evr
import numpy as np

def preprocess_sequence(input_path, output_path):
    """Complete preprocessing pipeline for detection models."""

    # Load and preprocess
    data = evr.preprocess_for_detection(
        input_path,
        representation="stacked_histogram",
        height=480, width=640,
        nbins=10, window_duration_ms=50
    )

    # Save preprocessed data
    np.save(output_path, data)

    return data.shape, data.nbytes / 1024 / 1024  # Shape and MB

# Process dataset
shape, size_mb = preprocess_sequence('input.h5', 'output.npy')
print(f"Preprocessed {shape} ({size_mb:.1f} MB)")
```

### Batch Processing

```python
import glob

def batch_preprocess(input_pattern, output_dir):
    """Process multiple files in batch."""
    files = glob.glob(input_pattern)

    for input_file in files:
        output_file = f"{output_dir}/{Path(input_file).stem}_processed.npy"

        try:
            data = evr.preprocess_for_detection(
                input_file,
                representation="stacked_histogram",
                height=480, width=640
            )
            np.save(output_file, data)
            print(f"SUCCESS: Processed: {input_file} -> {output_file}")

        except Exception as e:
            print(f"ERROR: Failed: {input_file} - {e}")

# Process all HDF5 files in a directory
batch_preprocess("data/*.h5", "preprocessed/")
```

## Troubleshooting

### Common Issues

**Memory Usage**: For very large files (>1GB), consider:
- Using temporal filtering: `t_start`, `t_end`
- Reducing window duration
- Processing in smaller chunks

**Performance**: To optimize performance:
- Use appropriate data types (Int16 for coordinates)
- Enable progress reporting for long operations
- Consider parallel processing for multiple files

**Compatibility**: For RVT migration:
- Check output shapes match expected format
- Verify polarity encoding (-1/1 vs 0/1)
- Test with small datasets first

### Getting Help

- Check the [API Reference](../api/representations.md) for detailed function documentation
- See [Examples](../examples/preprocessing.md) for more usage patterns
- Report issues on [GitHub](https://github.com/evlib/evlib/issues)
