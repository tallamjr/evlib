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

Three pipeline variants, all producing the identical output tensor of shape
`(1198, 20, 360, 640)` in `uint8`:

1. **evlib streaming (full)**: `evlib.rvt.process_sequence(engine="streaming")`. This is
   the real end-to-end pipeline. It includes evlib's one-time conversion of the raw h5
   into a sorted Parquet file, then the windowed Polars build of the representation.
2. **evlib build only**: the same Polars build path, but reading from a pre-built Parquet
   so the one-time h5 to Parquet conversion is excluded. This isolates the per-run build
   cost from the conversion that only needs to happen once per recording.
3. **RVT torch (reference)**: RVT's genuine code, reusing the modules under `lib/RVT`
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
  broken build. All three variants pass.
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
| evlib streaming (full) | 58.5 s | 4.29 GB |
| evlib build only | 28.7 s | 3.37 GB |
| RVT torch (reference) | 22.2 s | 6.40 GB |

- **Memory**: evlib uses about **1.5x less peak memory** than RVT (4.29 GB vs 6.40 GB
  full, 3.37 GB build-only). RVT loads the entire event-time array into RAM for its
  numba time correction and materialises dense torch tensors per window, whereas evlib
  streams windowed batches with Polars predicate pushdown.
- **Time**: on CPU, RVT's torch reference is currently **faster** in wall-clock terms.
  The evlib full pipeline is about 2.6x slower because it pays a one-time h5 to Parquet
  conversion (roughly 30 s of the 58.5 s); the conversion-free build-only stage is about
  1.3x slower than RVT. This is an honest result, not a regression to hide: evlib's
  advantage on this CPU run is memory footprint and a reusable Parquet artefact, not raw
  speed.

See `out/rvt_pipeline_time.png` and `out/rvt_pipeline_memory.png` for the charts, and
`out/rvt_pipeline_bench.md` for the regenerated min/median/max table.

## GPU note

evlib's build path also accepts `engine="streaming"` backed by the cudf-polars GPU
engine. That GPU path is coded but is validated separately on CUDA hardware; the numbers
above are CPU-only. The expectation is that the GPU engine shifts the time comparison in
evlib's favour, but that claim is deliberately left unmade here until it is measured on
CUDA.
