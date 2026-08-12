# Polars-Based Event Preprocessing

This guide covers evlib's Polars-based event representations. The functions build event camera representations as Polars expressions, so they run on the CPU Polars engine and on the cudf GPU engine (`engine="gpu"`). The GPU engine uses cudf-polars' own default memory resource, not CUDA managed memory (UVM); see [Performance: GPU memory footprint](../getting-started/performance.md#gpu-memory-footprint) for the measured numbers and when UVM is worth adding.

## Overview

The `evlib.representations` module provides Polars-based implementations of common event camera representations. Each function returns a Polars DataFrame in long format and accepts either a file path or a Polars LazyFrame. `evlib.load_events` itself returns a LazyFrame, so loading, filtering and representation building chain together lazily until the result is collected.

### Architecture

Polars is the query, filter and transform layer. It runs on the CPU, and on the cudf GPU engine (via `engine="gpu"`), which by default allocates through cudf-polars' own pooled memory resource rather than CUDA managed memory (UVM). For the RVT pipeline the stacked-histogram is built by dedicated native scatter-add kernels (Rust, CUDA, Metal), which are the compute layer.

Polars on the GPU is not a free speed-up for a single transfer-bound operation. The GPU win comes from the custom scatter-add kernels, or from compute-heavy workloads. On a single transfer-bound representation call the CPU Polars engine is typically the fastest evlib path.

### Performance

For a benchmark of these representations against tonic (NumPy) on a 20M-event stream, see [Benchmarks](../examples/benchmarks.md). evlib's CPU Polars engine beats tonic on voxel grid (1.35x), event frame (2.9x) and time surface (2.1x). The cudf GPU engine runs all three fully on the GPU, using its default allocator (no UVM needed), and still beats tonic on event frame and time surface, though at that stream size the CPU path is fastest because the operations are transfer-bound.

### Loaded event schema

`load_events` returns a Polars LazyFrame with `t` as a Duration (microseconds), `x` and `y` as Int16, and `polarity` as Int8 (-1 or +1).

```python
import evlib

events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
print(df.schema)
```

<!-- evlib:output -->
<!-- evlib:output:start -->
```text
Schema({'x': Int16, 'y': Int16, 't': Duration(time_unit='us'), 'polarity': Int8})
```
<!-- evlib:output:end -->

## API Reference

### `create_voxel_grid(events, height, width, n_time_bins=5, engine="auto")`

Tonic-validated voxel grid with bilinear temporal interpolation over the whole recording. Returns a DataFrame with columns `x`, `y`, `time_bin`, `contribution` (signed polarity accumulation per voxel).

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
voxel = evr.create_voxel_grid(events, height=480, width=640, n_time_bins=5)
print(f"Voxel grid entries: {len(voxel)}")
```

<!-- evlib:output -->
<!-- evlib:output:start -->
```text
Voxel grid entries: 196836
```
<!-- evlib:output:end -->

### `create_event_frame(events, height, width, n_time_bins=10, engine="auto")`

Matches tonic's `to_frame_numpy(n_time_bins=...)`: the whole recording is sliced into equal-width time bins and events are counted per (polarity, y, x) per bin.

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
frame = evr.create_event_frame(events, height=480, width=640, n_time_bins=10)
print(f"Event frame entries: {len(frame)}")
```

### `create_time_surface(events, height, width, dt, tau, engine="auto")`

Tonic-validated HOTS time surface (Lagorce et al. 2016). `dt` is the slice window and `tau` the decay constant, both in microseconds.

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
surface = evr.create_time_surface(
    events, height=480, width=640, dt=10000.0, tau=50000.0
)
print(f"Time surface entries: {len(surface)}")
```

### `create_stacked_histogram(events, height, width, bins=10, window_duration_ms=50.0, engine="auto")`

Bins events into fixed-duration windows and counts per (time_bin, polarity, y, x). Returns a DataFrame with columns `time_bin`, `polarity`, `y`, `x`, `count`. This is a general representation; for the RVT-compatible stacked histogram, use `evlib.rvt` (see below).

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
hist = evr.create_stacked_histogram(
    events, height=480, width=640, bins=10, window_duration_ms=50.0
)
print(f"Stacked histogram entries: {len(hist)}")
```

### RVT-compatible preprocessing

`evlib.rvt.process_sequence` reproduces RVT's stacked-histogram preprocessing. Its `backend` argument has four options:

- `"polars"`: Polars, CPU or cudf GPU (via `engine="gpu"`).
- `"rust"`: Rust dense scatter-add, CPU.
- `"cuda"`: custom CUDA scatter-add kernel, NVIDIA GPU.
- `"metal"`: Metal scatter-add kernel, Apple Silicon GPU.

The native kernels are also exposed directly as `evlib.representations_rs.stacked_histogram_dense`, `stacked_histogram_dense_cuda` and `stacked_histogram_dense_metal`. On the gen4_1mpx validation set the CUDA backend reaches parity-plus with the RVT torch-GPU reference and the Rust CPU backend is 1.32x faster than the RVT torch-CPU reference; see [Benchmarks](../examples/benchmarks.md).

## Selecting the engine

Pass `engine="auto"` (the default) for CPU Polars, or `engine="gpu"` for the cudf GPU engine, which uses cudf-polars' default memory resource. `engine="gpu"` requires `cudf-polars` and a CUDA-capable GPU; it will error if neither is available.

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
voxel_gpu = evr.create_voxel_grid(events, height=480, width=640, n_time_bins=5, engine="gpu")
print(f"Voxel grid entries (GPU): {len(voxel_gpu)}")
```

## Filtering

Use `evlib.filtering` to filter events before building a representation. The helpers operate on the LazyFrame so the work stays lazy and GPU-collectable.

```python
import evlib
import evlib.filtering as evf

events = evlib.load_events("data/slider_depth/events.txt")

windowed = evf.filter_by_time(events, t_start=0.1, t_end=0.5)
roi = evf.filter_by_roi(events, x_min=100, x_max=500, y_min=50, y_max=400)
positive = evf.filter_by_polarity(events, polarity=1)

print(f"Windowed events: {len(windowed.collect()):,}")
print(f"ROI events: {len(roi.collect()):,}")
print(f"Positive events: {len(positive.collect()):,}")
```

<!-- evlib:output -->
<!-- evlib:output:start -->
```text
Windowed events: 119,988
ROI events: 473,427
Positive events: 447,783
```
<!-- evlib:output:end -->

## Integration points

### File format support

- HDF5 (including ECF and BLOSC compression)
- EVT2 and EVT3 binary Prophesee formats
- Text (space-separated)
- Automatic format detection, so the format does not need to be specified manually

### Memory management

- Lazy loading, so events are read on demand for large files.
- Optimised types: Int16 coordinates, Int8 polarity, Duration timestamps.
- The cudf GPU engine's default allocator is enough for every evlib dataset tested to date (measured peak was about 3GB on a 62.96M-event file on a 24GB GPU); see [Performance: GPU memory footprint](../getting-started/performance.md#gpu-memory-footprint) for the numbers, and for the CUDA managed memory (UVM) recipe to use if a workload, or a co-tenant on the GPU, ever exceeds available VRAM.

## Batch processing

```python
import evlib
import evlib.representations as evr
import glob
from pathlib import Path

def batch_preprocess(input_pattern, output_dir):
    for input_file in glob.glob(input_pattern):
        output_file = f"{output_dir}/{Path(input_file).stem}_voxel.parquet"
        events = evlib.load_events(input_file)
        voxel = evr.create_voxel_grid(events, height=480, width=640, n_time_bins=5)
        voxel.write_parquet(output_file)
        print(f"Processed {input_file} -> {output_file} ({len(voxel)} entries)")
```

## Troubleshooting

Memory usage on very large files: keep the pipeline lazy and apply filters before collecting. On GPU, the default cudf allocator has held up to about 3GB peak on the largest real file tested (62.96M events); if you ever exceed available VRAM, see [Performance: GPU memory footprint](../getting-started/performance.md#gpu-memory-footprint) for the CUDA managed memory (UVM) recipe.

Performance: use the default CPU engine for single transfer-bound representation calls, and `engine="gpu"` for compute-heavy workloads or the largest files. Apply filters early so less data reaches the aggregation.

## Getting help

- See [Benchmarks](../examples/benchmarks.md) for measured performance and the benchmark harnesses.
- Report issues on [GitHub](https://github.com/tallamjr/evlib/issues).
