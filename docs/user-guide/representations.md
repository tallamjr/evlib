# Event Representations

Learn how to convert sparse event data into dense representations suitable for visualization, analysis, and neural networks.

## Overview

Event cameras produce sparse, asynchronous data streams. To work with this data effectively, we often need to convert it into dense representations like images or voxel grids.

evlib provides high-performance Polars-based implementations for common event representations:

- **Stacked Histograms**: Temporal binning with polarity channels (RVT-compatible)
- **Mixed Density Stacks**: Logarithmic time binning with polarity accumulation
- **Voxel Grids**: Traditional quantized temporal representations
- **High-level API**: Easy preprocessing for neural networks

## Stacked Histograms (RVT-Compatible)

Stacked histograms divide time into windows and bins, creating representations compatible with RVT preprocessing pipelines but with much better performance.

### Basic Usage

```python
import evlib
import evlib.representations as evr

# Create stacked histogram (recommended for neural networks)
events = evlib.load_events("data/slider_depth/events.txt")
# Note: Can pass LazyFrame directly - no need to .collect() explicitly

# Use a subset that spans sufficient time for window creation
# In test environments, we may have limited data, so adjust window size
events_df = events.collect()  # Only collect when we need to inspect the data
total_events = len(events_df)
time_span = (events_df['t'].max() - events_df['t'].min()).total_seconds()

# Adjust window duration based on available data
if time_span < 0.1:  # Less than 100ms of data
    window_duration_ms = max(0.001, time_span * 1000 / 4)  # Use 1/4 of available time, min 1μs
    print(f"Using adjusted window duration: {window_duration_ms:.3f}ms for {total_events} events")
else:
    window_duration_ms = 50.0
    print(f"Using standard window duration: {window_duration_ms}ms for {total_events} events")

hist_df = evr.create_stacked_histogram(
    events,  # Pass LazyFrame directly - function handles collection internally
    height=480,
    width=640,
    bins=10,                    # Temporal bins per window
    window_duration_ms=window_duration_ms
)

# Process results
print(f"Generated {len(hist_df)} histogram entries")
print(f"Columns: {list(hist_df.columns)}")  # Stacked histogram columns
```

### How Stacked Histograms Work

1. **Window Creation**: Event stream is divided into overlapping or non-overlapping windows
2. **Temporal Binning**: Each window is divided into equal time bins
3. **Polarity Channels**: Separate channels for positive (1) and negative (0) events
4. **Spatial Accumulation**: Events accumulate in 2D spatial locations
5. **Count Limiting**: Optional cutoff prevents extreme values

### Performance Comparison

```python
import time
import evlib.representations as evr

# Test performance with different bin counts
events = evlib.load_events("data/slider_depth/events.txt")
# Only collect when we need to inspect the data for parameters
events_df_sample = events.collect()

# Use appropriate subset based on available data
total_events = len(events_df_sample)
if total_events > 10000:
    events_subset = events_df_sample.head(10000).lazy()  # Convert back to LazyFrame
    print(f"Using 10k events for performance testing")
else:
    events_subset = events  # Use original LazyFrame
    print(f"Using all {total_events} available events for testing")

# Calculate appropriate window duration
time_range = events_df_sample['t'].max() - events_df_sample['t'].min()
time_span_sec = time_range.total_seconds()

if time_span_sec < 0.1:  # Less than 100ms of data
    window_duration_ms = max(0.001, time_span_sec * 1000 / 4)  # Use 1/4 of available time, min 1μs
else:
    window_duration_ms = 50.0  # Standard 50ms windows

print(f"Time range: {time_span_sec:.6f} seconds")
print(f"Using window duration: {window_duration_ms:.1f}ms")

for nbins in [5, 10, 15]:
    start_time = time.time()
    hist_df = evr.create_stacked_histogram(
        events_subset,  # Pass LazyFrame directly
        height=480, width=640,
        bins=nbins,
        window_duration_ms=window_duration_ms
    )
    duration = time.time() - start_time

    print(f"Bins: {nbins}, Time: {duration:.3f}s, Entries: {len(hist_df)}")

estimated_windows = max(1, int(time_span_sec / (window_duration_ms / 1000)))
print(f"Estimated windows: {estimated_windows}")
```
