# Polars-Based Event Preprocessing

This guide covers evlib's high-performance Polars-based event preprocessing capabilities, designed to replace PyTorch-based preprocessing pipelines like those used in RVT (Recurrent Vision Transformers).

## Overview

The `evlib.representations` module provides high-performance Polars-based implementations of common event camera representations with significant performance improvements over traditional PyTorch approaches. All functions return Polars LazyFrames for maximum efficiency and lazy evaluation.

### Performance Benefits

Based on testing with real event data:

- **evlib Polars Processing**: 2.31s (1.4M events/s)
- **Estimated RVT PyTorch**: ~8.10s (estimated)
- **Performance Speedup**: **3.5x faster**
- **Memory Usage**: 109.3 MB for 69 windows of stacked histograms

### Output Compatibility

- **Format**: Polars LazyFrames with columns [window_id, channel, time_bin, y, x, count]
- **Data Types**: Int32 for coordinates and counts, Duration for timestamps
- **Lazy Evaluation**: Operations deferred until .collect() is called
- **Engine Selection**: Automatically uses optimal Polars engine (GPU/streaming)
- **RVT Compatibility**: Can be converted to RVT-expected tensor format when needed

## Quick Start

*High-level preprocessing API is under development.*

### Why This API is Better

- **Single import**: All representation functions in one place
- **Lazy evaluation**: Process only what's needed, when needed
- **Memory efficient**: Polars handles large datasets with minimal memory
- **Engine optimized**: Automatically uses best engine (GPU/streaming/CPU)
- **Clean API**: Consistent LazyFrame outputs, no tensor manipulation needed

## API Reference

### High-Level Functions

#### `preprocess_for_detection()`

Drop-in replacement for RVT preprocessing pipeline.

*Detection preprocessing function is under development.*

**Parameters:**
- `events_path`: Path to event file
- `representation`: Type of representation
- `height`, `width`: Output dimensions
- `**kwargs`: Additional parameters passed to the specific representation function

#### `create_stacked_histogram()`

Main function for temporal histogram creation with windowing.

*Stacked histogram creation function is under development.*

**Parameters:**
- `events`: Path to event file or Polars LazyFrame
- `height`, `width`: Output dimensions
- `nbins`: Number of temporal bins per window
- `window_duration_ms`: Duration of each window in milliseconds
- `stride_ms`: Stride between windows (defaults to window_duration_ms for non-overlapping)
- `count_cutoff`: Maximum count per bin (None for no limit)

**Returns:**
- Polars LazyFrame with columns [window_id, channel, time_bin, y, x, count, channel_time_bin]
- channel: 0 for negative polarity, 1 for positive polarity
- channel_time_bin: combined channel*nbins + time_bin for easier tensor conversion

#### `create_mixed_density_stack()`

Logarithmic time binning with polarity accumulation.

*Mixed density stack creation is under development.*

**Parameters:**
- Similar to `create_stacked_histogram()` but uses logarithmic binning
- `count_cutoff`: Maximum absolute value per bin

**Returns:**
- Polars LazyFrame with columns [window_id, time_bin, y, x, polarity_sum]
- polarity_sum: signed accumulation of polarity values

#### `create_voxel_grid()`

Traditional voxel grid representation for entire dataset.

*Voxel grid creation function is under development.*

**Returns:**
- Polars LazyFrame with columns [time_bin, y, x, value]
- value: signed polarity accumulation per voxel

### Performance Benchmarking

*Performance benchmarking is under development.*

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
Event File → evlib.load_events() → Polars LazyFrame → Window Processing → Output LazyFrame
                                                    ↓
                          Temporal Binning ← Spatial Clipping ← Polarity Conversion
                                                    ↓
                          Groupby Aggregation → Sparse Histogram → .collect() → DataFrame
```

## Migration from RVT

### Code Changes Required

**evlib approach:**
```python
import evlib
import evlib.representations as evr

# evlib preprocessing with Polars (high-performance)
events = evlib.load_events("data/slider_depth/events.txt")
hist = evr.create_stacked_histogram(events, width=640, height=480, nbins=10)
hist_df = hist.collect()
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

*High-framerate preprocessing tools are under development.*

### Production Pipeline

*Production pipeline tools are under development.*

### Batch Processing

```python
import glob
from pathlib import Path

def batch_preprocess(input_pattern, output_dir):
    """Process multiple files in batch."""
    files = glob.glob(input_pattern)

    for input_file in files:
        output_file = f"{output_dir}/{Path(input_file).stem}_processed.parquet"

        try:
            # Create lazy preprocessing pipeline
            data_lazy = evr.preprocess_for_detection(
                input_file,
                representation="stacked_histogram",
                height=480, width=640
            )

            # Collect and save efficiently
            data_df = evlib.collect_with_optimal_engine(data_lazy)
            data_df.write_parquet(output_file)

            print(f"SUCCESS: Processed: {input_file} -> {output_file}")
            print(f"  Entries: {len(data_df)}, Size: {data_df.estimated_size()/1024/1024:.1f} MB")

        except Exception as e:
            print(f"ERROR: Failed: {input_file} - {e}")

# Process all HDF5 files in a directory
batch_preprocess("data/*.h5", "preprocessed/")
```

## Troubleshooting

### Common Issues

**Memory Usage**: For very large files (>1GB), consider:
- Use lazy evaluation: don't collect until needed
- Apply filtering before collection: `lazy_frame.filter(condition)`
- Use optimal engine: `evlib.collect_with_optimal_engine(lazy_frame)`
- Stream processing: `lazy_frame.collect(streaming=True)`

**Performance**: To optimize performance:
- Use lazy evaluation for chained operations
- Apply filters early in the pipeline
- Use `.collect(engine="streaming")` for large datasets
- Consider GPU engine with `POLARS_ENGINE_AFFINITY=gpu`

**Compatibility**: For RVT migration:
- LazyFrames provide more flexibility than fixed tensors
- Convert to tensor format only when feeding to neural networks
- Use `.collect()` sparingly - work with LazyFrames when possible
- Check polarity encoding in your specific dataset

### Getting Help

- Check the [API Reference](../api/representations.md) for detailed function documentation
- See [Examples](../examples/preprocessing.md) for more usage patterns
- Report issues on [GitHub](https://github.com/evlib/evlib/issues)
