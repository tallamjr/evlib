# GPU (cudf-polars) validation

The evlib RVT pipeline forwards an `engine` argument straight to Polars'
`LazyFrame.collect(engine=...)`, so the GPU engine is selected with `engine="gpu"`
(or a `pl.GPUEngine(...)` instance):

```
evlib-rvt-preprocess --in-h5 <raw.h5> --grid-npy <timestamps_us.npy> \
    --dataset gen4 --height 720 --width 1280 --engine gpu
```

This was validated on an NVIDIA RTX 4090 (24GB VRAM). The results below are real
measurements, not projections.

## What was validated

- **Correctness.** `engine="gpu"` and the default CPU `.collect()` produce numerically
  identical results, checked on a small file (slider_depth, about 53k rows) and on the
  largest real file available, `events_gpu_torch.h5` (62,961,273 events, 818.5MB raw).
- **Default memory behaviour.** `engine="gpu"` allocates through cudf-polars' own default
  memory resource; it does not use CUDA managed memory (UVM) by default. The `engine`
  argument passes straight through to `LazyFrame.collect(engine=engine)` (see
  `python/evlib/representations.py:29-31`).
- **Measured VRAM footprint.** A realistic query, `filter_by_polarity` followed by
  `create_stacked_histogram`, on the 818.5MB/62.96M-event file peaked at about 3GB: roughly
  3.7x the raw data size, comfortably inside the RTX 4090's 24GB.
- **No realistic evlib dataset needs UVM.** At that overhead, reaching the 24GB card limit
  needs roughly 500M+ events in a single query. The largest real file available is about 8x
  smaller than that threshold; even eTram, the largest dataset evlib's own docs mention (up
  to 1.6GB raw, about 123M events), is about 4x smaller and would use only about 6GB peak
  VRAM.

evlib's on-wire schema is exactly 13 bytes/event (`x`: Int16, `y`: Int16, `t`: Duration(us)
as i64, `polarity`: Int8), confirmed empirically: 818.5MB / 62,961,273 events = 13.00
bytes/event exactly.

## The RVT preprocessing query on GPU

The RVT window-assignment query is a histogram build, not a typical relational workload: it
uses two `join_asof` joins plus an `explode` for window assignment, and a window expression
(`.over("window_id")`) for per-window time normalisation. cudf-polars falls back to the CPU
transparently for any operation it does not support (per the RAPIDS cudf-polars docs), so
this specific query's GPU coverage is a separate, still-open question from the general
correctness and memory results above, which were measured on `filter_by_polarity` plus
`create_stacked_histogram`, not on the RVT `join_asof`/`explode`/`over` path.

In practice this gap does not block RVT preprocessing on NVIDIA hardware: `backend="cuda"`,
the dedicated custom CUDA scatter-add kernel, is the validated and recommended path there.
It reaches parity-plus with RVT's own torch-GPU reference (283.6s versus 286.3s on the
gen4_1mpx validation set, bit-identical output; see `docs/getting-started/performance.md`).
`engine="gpu"` on the Polars backend remains available for the RVT pipeline's own query, and
would be worth measuring for anyone restructuring that query, but it is not the path evlib
ships or recommends for RVT preprocessing today.

## If a workload, or a co-tenant on the GPU, exceeds VRAM

UVM was validated as a forward-looking, synthetic test, not because any real evlib dataset
needs it. Without UVM, evlib-shaped GPU queries above available headroom either fail with a
CUDA out-of-memory error, or degrade badly: 100%+ GPU utilisation and a stall of 4 or more
minutes with no result, observed at about 503M events. Both `rmm.mr.ManagedMemoryResource()`
(basic UVM) and a pooled-plus-prefetch variant succeeded at every tested size; the
pooled-plus-prefetch variant was 1.46-1.75x faster than basic UVM in these tests
(directionally consistent with, though larger than, the roughly 1.2-1.3x published
elsewhere).

A real, observed reason to want UVM: not evlib's own data size, but a shared or
multi-tenant GPU. During this testing, a concurrent unrelated job was using about 20.5GB of
the 24GB card, leaving the default GPU path only about 3.5GB of real headroom.

```python notest
import polars as pl
import rmm
import rmm.mr as mr

# Basic UVM: works, but pages fault frequently (measured 1.46-1.75x slower
# than the pooled+prefetch variant below).
managed = mr.ManagedMemoryResource()
mr.set_current_device_resource(managed)
engine = pl.GPUEngine(memory_resource=managed)

# Recommended: pooled allocation on top of managed memory, with prefetch
# hints to the CUDA driver, to reduce page-fault overhead.
managed = mr.ManagedMemoryResource()
pool = mr.PoolMemoryResource(managed)
prefetch = mr.PrefetchResourceAdaptor(pool)
mr.set_current_device_resource(prefetch)
engine = pl.GPUEngine(memory_resource=prefetch)

result = lazy_frame.collect(engine=engine)
```

## Related documentation

See `docs/getting-started/performance.md` for the RVT torch-GPU comparison, the full
backend timing table, and the GPU memory footprint section, and
`docs/development/architecture.md` for how the query layer (Polars/cudf) and compute layer
(native scatter-add kernels) split.
