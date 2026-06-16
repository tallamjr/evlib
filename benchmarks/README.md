# evlib vs RVT preprocessing benchmark

This benchmark compares evlib's RVT-preprocessing pipeline against RVT's own torch
implementation on the same task: turning the raw Gen4 1Mpx validation recording
(`moorea_2019-02-21_000_td_2257500000_2317500000`, roughly 540 million events) into the
stacked-histogram event representation used by RVT.

Run it with:

```bash
.venv/bin/python -m benchmarks.bench_rvt_pipeline --repeats 3
```

## What is compared

Four pipeline variants, all producing the identical output tensor of shape
`(1198, 20, 360, 640)` in `uint8`:

1. **evlib rust (dense scatter-add)**: `evlib.rvt.process_sequence(backend="rust")`. Reads
   the raw h5 directly (no Parquet conversion), corrects time to non-decreasing, computes
   per-window slices with `np.searchsorted`, and hands each batch of windows to a Rust
   function that scatter-adds event counts straight into a preallocated dense buffer per
   window (the analogue of torch's `tensor.put_(accumulate=True)`), clips each pixel to the
   count cutoff and casts to `uint8`. This is the fast path.
2. **evlib streaming (full)**: `evlib.rvt.process_sequence(engine="streaming")`, the default
   Polars backend. This is the real end-to-end pipeline. It includes evlib's one-time
   conversion of the raw h5 into a sorted Parquet file, then the windowed Polars build of
   the representation.
3. **evlib build only**: the same Polars build path, but reading from a pre-built Parquet
   so the one-time h5 to Parquet conversion is excluded. This isolates the per-run build
   cost from the conversion that only needs to happen once per recording.
4. **RVT torch (reference)**: RVT's genuine code, reusing the modules under `lib/RVT`
   (`H5Reader` with its numba cum-max time correction, `StackedHistogram` with
   `count_cutoff=10, fastmode=True`, and `downsample_ev_repr` nearest-exact 0.5). RVT
   reads slices straight from the raw h5 and builds dense torch tensors per window.

## Methodology

- **Same raw input** for all variants: the committed Gen4 val h5.
- **Same reference grid**: every variant windows over the committed reference
  `timestamps_us.npy` (1198 window-end timestamps), so the windowing is identical. The
  RVT reference is fed this grid directly rather than recomputing it from labels, which
  is the only adaptation made to its code.
- **Bit-identical outputs verified**: before any timing, each variant is run once and its
  output asserted `np.array_equal` to the committed reference output across all 1198
  windows. The run aborts if any variant is not bit-identical, so we never benchmark a
  broken build. All four variants pass.
- **Timing**: each variant runs in a fresh subprocess, `--repeats 3` times. The bar shows
  the median with min..max error bars. The reported time is the pipeline body wall-clock
  (excluding interpreter and import start-up, which is identical across variants).
- **Peak memory**: each subprocess reports its own peak resident set size via
  `resource.getrusage(RUSAGE_SELF).ru_maxrss`, normalised to bytes per platform (bytes on
  macOS, KiB on Linux). This captures native Polars and torch buffers that `tracemalloc`
  would miss.

## Headline numbers

Measured on this machine (macOS, CPU), medians over 3 repeats:

| pipeline | median time | peak memory |
| --- | --- | --- |
| evlib rust (dense scatter-add, raw h5) | 15.6 s | 7.38 GB |
| evlib streaming (full) | 58.5 s | 4.29 GB |
| evlib build only | 28.2 s | 3.28 GB |
| RVT torch (reference) | 23.4 s | 6.40 GB |

- **Time**: the **Rust dense scatter-add backend is about 1.5x faster than RVT's torch
  reference** (15.6 s vs 23.4 s median) on CPU, and roughly 3.7x faster than the full
  Polars streaming pipeline. It wins by skipping the h5 to Parquet conversion entirely
  (it reads the raw h5 directly) and by replacing the Polars hash group-by with a flat
  scatter-add into a preallocated dense buffer, which is the same algorithm torch uses but
  without torch's per-window tensor allocation overhead.
- **Memory**: the Rust backend uses **1.15x more** peak memory than RVT (7.38 GB vs
  6.40 GB). It holds one int64 copy of the global corrected time array (~4.3 GB for
  540 M events) for the global `searchsorted`, which is the dominant fixed cost; the
  per-batch event slices and dense output buffers are small and do not grow with sequence
  length, so memory is bounded. This is an honest trade: the Rust path buys a 1.5x speedup
  at a modest memory premium over torch. The Polars backends still use about 1.5x less
  peak memory than both (4.29 GB full, 3.28 GB build-only) by streaming windowed Parquet
  batches, at the cost of being slower.
- The default backend remains the Polars streaming path; the Rust scatter-add backend is
  opt-in via `backend="rust"`.

See `out/rvt_pipeline_time.png` and `out/rvt_pipeline_memory.png` for the charts, and
`out/rvt_pipeline_bench.md` for the regenerated min/median/max table.

## GPU note

evlib's build path also accepts `engine="streaming"` backed by the cudf-polars GPU
engine. That GPU path is coded but is validated separately on CUDA hardware; the numbers
above are CPU-only. The expectation is that the GPU engine shifts the time comparison in
evlib's favour, but that claim is deliberately left unmade here until it is measured on
CUDA.
