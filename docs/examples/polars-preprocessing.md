# Polars Preprocessing Examples

This document provides examples for using evlib's Polars-based event representations. The functions build event camera representations as Polars expressions, so they run on the CPU Polars engine, on the cudf GPU engine, and on CUDA managed memory (UVM) for workloads larger than VRAM.

## Overview

The `evlib.representations` module provides Polars-based implementations of common event camera representations. Each function returns a Polars DataFrame (long format) and accepts a Polars LazyFrame or DataFrame as input (not a file path; load with `evlib.load_events` first). Internally the work is expressed as Polars operations, so the engine is selectable via the `engine` argument.

### How it executes

1. Native groupby and aggregation rather than tensor indexing.
2. Lazy evaluation that defers work until the result is collected.
3. Optimised data types (Int16 coordinates, Int8 polarity) for cache locality.
4. A selectable engine: `engine="auto"` for CPU Polars, `engine="gpu"` for cudf with CUDA managed memory.

For a benchmark of these representations against tonic (NumPy), see [Benchmarks](benchmarks.md). On a single transfer-bound stream the CPU Polars engine is usually the fastest evlib path; the GPU path is for compute-heavy or larger-than-VRAM workloads.

## Available test datasets

- `data/slider_depth/events.txt`: text format (about 22MB, 1.1M events)
- `data/eTram/h5/`: HDF5 format eTram samples

## Loaded event schema

`load_events` returns a Polars LazyFrame. The schema is `t` as a Duration (microseconds), `x` and `y` as Int16, and `polarity` as Int8 (-1 or +1).

```python
import evlib

events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
print(df.schema)
print(f"Loaded {len(df):,} events")
```

## Voxel grid

`create_voxel_grid` builds a tonic-validated voxel grid with bilinear temporal interpolation over the whole recording.

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
voxel = evr.create_voxel_grid(events, height=480, width=640, n_time_bins=5)
print(f"Voxel grid entries: {len(voxel)}")
print(voxel.columns)
```

To run it on the GPU, pass `engine="gpu"`:

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
voxel_gpu = evr.create_voxel_grid(events, height=480, width=640, n_time_bins=5, engine="gpu")
print(f"Voxel grid entries (GPU): {len(voxel_gpu)}")
```

## Event frame

`create_event_frame` matches tonic's `to_frame_numpy(n_time_bins=...)`: the whole recording is sliced into equal-width time bins and events are counted per (polarity, y, x) per bin.

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
frame = evr.create_event_frame(events, height=480, width=640, n_time_bins=10)
print(f"Event frame entries: {len(frame)}")
```

## Time surface

`create_time_surface` builds a tonic-validated HOTS time surface (Lagorce et al. 2016). `dt` is the slice window and `tau` the decay constant, both in microseconds.

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
surface = evr.create_time_surface(
    events, height=480, width=640, dt=10000.0, tau=50000.0
)
print(f"Time surface entries: {len(surface)}")
```

## Stacked histogram

`create_stacked_histogram` bins events into fixed-duration windows and counts per (time_bin, polarity, y, x). This is a general representation; for the RVT-identical stacked histogram expected by the RVT detection pipeline, use `evlib.rvt` (see the RVT preprocessing section below).

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")
hist = evr.create_stacked_histogram(
    events, height=480, width=640, bins=10, window_duration_ms=50.0
)
print(f"Stacked histogram entries: {len(hist)}")
```

## RVT-identical preprocessing

For the exact RVT stacked-histogram preprocessing, use `evlib.rvt.process_sequence`. It has a `backend` argument with four options: `"polars"` (CPU or cudf GPU), `"rust"` (Rust dense scatter-add, CPU), `"cuda"` (custom CUDA scatter-add kernel, NVIDIA GPU), and `"metal"` (Metal scatter-add kernel, Apple Silicon GPU). The native kernels are also exposed directly as `evlib.representations_rs.stacked_histogram_dense`, `stacked_histogram_dense_cuda` and `stacked_histogram_dense_metal`.

On the gen4_1mpx validation set the CUDA backend reaches parity-plus with the RVT torch-GPU reference (283.6s versus 286.3s) and the Rust CPU backend is 1.32x faster than the RVT torch-CPU reference. See [Benchmarks](benchmarks.md) for the full table and plots.

## Filtering

Use `evlib.filtering` to filter events before building a representation. The helpers operate on the LazyFrame so the work stays lazy and GPU-collectable.

```python
import evlib
import evlib.filtering as evf

events = evlib.load_events("data/slider_depth/events.txt")

positive = evf.filter_by_polarity(events, polarity=1)
roi = evf.filter_by_roi(events, x_min=100, x_max=500, y_min=50, y_max=400)

print(f"Positive events: {len(positive.collect()):,}")
print(f"ROI events: {len(roi.collect()):,}")
```

## Batch processing

Process multiple files by looping over a glob pattern and writing each result to Parquet.

```python
import evlib
import evlib.representations as evr
import glob
from pathlib import Path

def batch_voxel_grids(input_pattern, output_dir):
    for input_file in glob.glob(input_pattern):
        output_file = f"{output_dir}/{Path(input_file).stem}_voxel.parquet"
        events = evlib.load_events(input_file)
        voxel = evr.create_voxel_grid(events, height=480, width=640, n_time_bins=5)
        voxel.write_parquet(output_file)
        print(f"Processed {input_file} -> {output_file} ({len(voxel)} entries)")
```

## Next steps

1. Benchmark with your own data using the harnesses described in [Benchmarks](benchmarks.md).
2. Tune `n_time_bins`, `window_duration_ms`, `dt` and `tau` for your use case.
3. Use `engine="gpu"` for compute-heavy or larger-than-VRAM workloads, and the default CPU engine for single transfer-bound calls.
