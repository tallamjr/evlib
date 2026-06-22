# Benchmarks

This page documents evlib's two committed benchmark suites: the RVT preprocessing pipeline and the representation comparison against tonic. Both run each measurement in a fresh subprocess so peak resident memory is an unambiguous per-process figure.

## The four compute backends

The RVT stacked-histogram is built by dedicated native scatter-add kernels. `evlib.rvt.process_sequence(..., backend=...)` selects between them:

- `"polars"`: Polars, CPU or cudf GPU (via `engine="gpu"`).
- `"rust"`: Rust dense scatter-add, CPU.
- `"cuda"`: custom CUDA scatter-add kernel, NVIDIA GPU.
- `"metal"`: Metal scatter-add kernel, Apple Silicon GPU.

The kernels are exposed directly as `evlib.representations_rs.stacked_histogram_dense`, `stacked_histogram_dense_cuda` and `stacked_histogram_dense_metal`.

Polars is the query, filter and transform layer (CPU, cudf GPU engine, and CUDA managed memory for workloads larger than VRAM). The native scatter-add kernels are the compute layer. Polars on the GPU is not a free speed-up for a single transfer-bound operation; the custom scatter-add kernels are where the GPU wins, and the CUDA-versus-RVT-GPU result is parity-plus because the shared HDF5 read dominates the largest sequences.

## RVT preprocessing pipeline

Validation run on the gen4_1mpx set (18 sequences, single pass) on an RTX 4090. Every output was asserted bit-identical to the RVT torch reference, except a roughly 1e-10 float-binning boundary quirk on three sequences.

| Pipeline | Total time (18 sequences) | Relative to RVT |
|----------|---------------------------|-----------------|
| evlib CUDA (custom kernel) | 283.6s | 1.01x faster than RVT torch-GPU (parity, edging ahead); 1.88x faster than RVT torch-CPU |
| RVT torch-GPU (reference) | 286.3s | baseline (GPU) |
| evlib Rust-CPU | 406.2s | 1.32x faster than RVT torch-CPU |
| RVT torch-CPU (reference) | 534.2s | baseline (CPU) |

Plots: `benchmarks/out/rvt_final_time.png` (wall-clock) and `benchmarks/out/rvt_final_memory.png` (peak resident memory).

### Running the RVT benchmark

```bash
python -m benchmarks.bench_rvt_dataset \
    --orig-root ~/datasets/gen4_1mpx_original/val \
    --ref-root  ~/datasets/gen4_1mpx_processed_RVT/gen4/val \
    --backends evlib_gpu evlib_cpu evlib_rust rvt_gpu rvt_cpu
```

The harness drives the raw to processed RVT stacked-histogram preprocessing across every sequence of a split, runs each (backend, sequence) in a fresh subprocess, and asserts every output is bit-identical to that sequence's committed RVT reference before keeping its timing. A non-identical output is a hard failure.

## Representations versus tonic

For the general representation surface (voxel grid, event frame, time surface), tonic (pure NumPy) is the natural baseline. The harness loads one real event stream once and feeds the identical events to both libraries. The numbers below are from a 20M-event eTram stream.

| Representation | evlib Polars CPU | tonic (NumPy) | Speedup |
|----------------|------------------|---------------|---------|
| voxel_grid | 0.62s | 0.84s | 1.35x |
| event_frame | 0.32s | 0.92s | 2.9x |
| time_surface | 0.26s | 0.55s | 2.1x |

Plot: `benchmarks/out/tonic_bench_time.png`.

evlib's cudf GPU plus UVM path runs all three fully on the GPU. At this stream size the operations are transfer-bound, so CPU Polars is the fastest evlib path; the GPU path still beats tonic on event_frame and time_surface.

### Running the tonic benchmark

```bash
python -m benchmarks.bench_tonic \
    --raw ~/datasets/eTram/raw/test_1/test_day_001.raw \
    --n-events 30000000
```

This runs each (operation, backend) in a fresh subprocess across tonic, evlib CPU Polars, and evlib GPU (cudf with UVM). evlib's outputs are already validated against tonic, so this harness measures speed and memory, not correctness.

## Reproducing the representation timings yourself

The representation functions return Polars DataFrames and accept a Polars LazyFrame or DataFrame (not a file path).

```python
import time
import evlib
import evlib.representations as evr

events = evlib.load_events("data/slider_depth/events.txt")

start = time.time()
voxel = evr.create_voxel_grid(events, height=480, width=640, n_time_bins=5)
print(f"voxel_grid: {time.time() - start:.3f}s, {len(voxel)} entries")

events = evlib.load_events("data/slider_depth/events.txt")
start = time.time()
frame = evr.create_event_frame(events, height=480, width=640, n_time_bins=10)
print(f"event_frame: {time.time() - start:.3f}s, {len(frame)} entries")
```

To run on the GPU, pass `engine="gpu"` to any representation function; on a single transfer-bound stream the CPU engine (`engine="auto"`) is typically faster.

## Loaded event schema

`load_events` returns a Polars LazyFrame with `t` as a Duration (microseconds), `x` and `y` as Int16, and `polarity` as Int8 (-1 or +1).

```python
import evlib

events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
print(df.schema)
print(f"Loaded {len(df):,} events")
```
