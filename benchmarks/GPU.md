# Validating the GPU (cudf-polars) path

The evlib RVT pipeline forwards an `engine` argument straight to Polars'
`LazyFrame.collect(engine=...)`, so the GPU engine is selected with `engine="gpu"`
(or a `pl.GPUEngine(...)` instance). This works end to end:

```
evlib-rvt-preprocess --in-h5 <raw.h5> --grid-npy <timestamps_us.npy> \
    --dataset gen4 --height 720 --width 1280 --engine gpu
```

On a host without CUDA / cudf-polars (such as this macOS laptop) Polars transparently
falls back to the CPU engine and produces identical output, so the GPU path cannot be
benchmarked here. It must be validated on an NVIDIA CUDA machine.

## Why this needs real validation, not assumption

cudf-polars accelerates a query by translating the Polars plan to a GPU plan and
**falling back to the CPU engine for any operation it does not support** (per the
RAPIDS cudf-polars docs: it "supports most core Polars expressions ... falling back to
the CPU engine transparently if not supported").

Our query is a histogram build, not a typical relational workload. Its window assignment
uses two `join_asof` joins plus an `explode`, and the per-window time normalisation uses
a window expression (`.over("window_id")`). These advanced operations are **not confirmed
GPU-supported** by cudf-polars. If they fall back to CPU, the GPU buys little or nothing
for this specific query. So the first validation step is to measure GPU coverage, not to
assume a speedup.

There is also a deeper point from the CPU benchmark (see `README.md`): RVT's torch path is
a dense scatter-add, which is algorithmically cheaper than our Polars hash group-by. A GPU
group-by may or may not beat a GPU scatter-add (torch on CUDA). The honest expectation is
"measure it", not "GPU will obviously win".

## Step-by-step on a CUDA machine

1. Install a cudf-polars build matching the CUDA toolkit and the installed Polars version
   (1.30.0 here). Follow the current RAPIDS install matrix at
   https://docs.rapids.ai/install , e.g. for CUDA 12:
   ```
   pip install --extra-index-url=https://pypi.nvidia.com cudf-polars-cu12
   ```
   Verify: `python -c "import cudf_polars; print('cudf-polars ok')"`.

2. Confirm bit-identity on GPU first (correctness before speed). Run the acceptance test
   with the engine overridden to gpu, or run `process_sequence(..., engine="gpu")` and
   `np.array_equal` the output against the reference `event_representations_ds2_nearest.h5`.
   The GPU output MUST be bit-identical; if it is not, stop and investigate before timing.

3. Measure GPU coverage (the critical check). Run with verbose fallback reporting:
   ```
   POLARS_VERBOSE=1 evlib-rvt-preprocess ... --engine gpu 2>&1 | grep -i "fallback\|gpu\|cpu"
   ```
   If `join_asof`, `explode`, or the `over` window expression report a CPU fallback, the
   GPU is not actually accelerating those stages. Note exactly which stages run on GPU.

4. Benchmark. Extend `benchmarks/bench_rvt_pipeline.py` to add a third variant that runs
   `process_sequence(..., engine="gpu")` (the harness already measures wall-clock and peak
   RSS per variant in a subprocess; add a `gpu` entry alongside `streaming` and
   `build only`). Compare against the CPU `streaming` and `RVT torch` bars. Record the
   numbers; do not infer them.

## If the GPU falls back (likely for join_asof / explode / over)

To actually exploit the GPU, the window-assignment and per-window normalisation would need
restructuring into GPU-supported operations (e.g. replacing the `join_asof` + `explode`
window assignment with an integer-arithmetic window id and the `.over()` min/max with a
`group_by` + join of per-window bounds). That restructuring must keep the output
bit-identical (the acceptance gate is the arbiter) and should only be written and merged on
a machine where it can actually be run and verified, per the project's rule against
unvalidated code.
