# Processing API Reference

All event processing in evlib is pure Python Polars. Filtering and representations are lazy Polars queries, so they run on the CPU streaming engine or on the GPU via cudf-polars (`collect(engine="gpu")`) without code changes. Deep learning models live in Python/PyTorch under `evlib.models`.

There is no Rust neural-network backend, and no ONNX, `tch`, or candle bindings: that machinery was removed. Models are Python only.

## `evlib.filtering`

Lazy event filters that accept and return a Polars `LazyFrame` (a `DataFrame` is also accepted as input). Inputs are expected to have columns `[x, y, t, polarity]` as produced by `evlib.load_events`, where `t` is a Duration in microseconds. Every function takes an `engine` argument (`"auto"`, `"streaming"`, `"gpu"`, or a Polars `GPUEngine`).

```python
def filter_by_time(events, t_start=None, t_end=None, engine="auto") -> pl.LazyFrame
def filter_by_roi(events, x_min, x_max, y_min, y_max, engine="auto") -> pl.LazyFrame
def filter_by_polarity(events, polarity, engine="auto") -> pl.LazyFrame
def filter_hot_pixels(events, threshold_percentile=99.9, engine="auto") -> pl.LazyFrame
def filter_noise(events, method="refractory", refractory_period_us=1000, engine="auto") -> pl.LazyFrame
def filter_multiple_rois(events, rois, engine="auto") -> pl.LazyFrame
def preprocess_events(events, ...) -> pl.LazyFrame
```

`t_start`/`t_end` are in **seconds** (converted to microseconds internally). `filter_by_polarity` matches the stored `-1`/`+1` encoding.

```python
import evlib
from evlib.filtering import filter_by_time, filter_by_roi, filter_by_polarity

events = evlib.load_events("data/prophesee/samples/evt2/80_balls.raw")

# Chain lazy filters, then collect once
filtered = filter_by_polarity(
    filter_by_roi(
        filter_by_time(events, t_start=0.1, t_end=0.5),
        x_min=100, x_max=500, y_min=50, y_max=400,
    ),
    polarity=1,
)

df = filtered.collect(engine="streaming")
print(f"{len(df):,} events after filtering")
```

You can also write the filters directly as Polars expressions instead of calling the helpers:

```python
import polars as pl

filtered = events.filter(
    (pl.col("t").dt.total_microseconds() / 1_000_000).is_between(0.1, 0.5)
    & pl.col("x").is_between(100, 500)
    & (pl.col("polarity") == 1)
)
```

## `evlib.representations`

Event-to-representation conversion, also pure Python Polars. The main builders:

```python
def create_stacked_histogram(events, height, width, bins=10, window_duration_ms=50.0, ...) -> pl.LazyFrame
def create_voxel_grid(events, height, width, n_time_bins=5, engine="auto") -> pl.LazyFrame
def create_mixed_density_stack(events, height, width, ...) -> pl.LazyFrame
def preprocess_for_detection(events, height, width, bins=5, engine="auto") -> pl.LazyFrame
```

Note the parameter names: `create_stacked_histogram` takes `bins` and `window_duration_ms`, while `create_voxel_grid` takes `n_time_bins`.

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data/prophesee/samples/evt2/80_balls.raw")

# Stacked histogram (RVT-style input)
hist = evr.create_stacked_histogram(
    events, height=720, width=1280, bins=5, window_duration_ms=50.0
)

# Voxel grid
voxel = evr.create_voxel_grid(events, height=720, width=1280, n_time_bins=5)
```

For the RVT-identical preprocessing pipeline, use `evlib.rvt.process_sequence(...)`. It offers four interchangeable backends via `backend=`:

- `"polars"`: Polars on the CPU, or on the cudf GPU engine when you pass an `engine=` of `"gpu"` or a `pl.GPUEngine(...)`.
- `"rust"`: Rust dense scatter-add on the CPU.
- `"cuda"`: a custom CUDA scatter-add kernel on an NVIDIA GPU; it loads the nvcc-built `librvt_scatter.so` via the `EVLIB_CUDA_LIB` environment variable.
- `"metal"`: a Metal scatter-add kernel on Apple Silicon; build it with `CC=clang maturin develop --features metal`.

The native kernels behind these backends are exposed directly as `evlib.representations_rs.stacked_histogram_dense` (CPU), `stacked_histogram_dense_cuda`, and `stacked_histogram_dense_metal`.

Note that `evlib.rvt` and `evlib.representations.create_stacked_histogram` compute different quantities: use `evlib.rvt` when you need bit-identical RVT preprocessing.

## `evlib.models`

Deep learning models are implemented in Python with PyTorch and ship with real pretrained weights:

- **E2VID**: event-to-video reconstruction (`evlib.models.E2VID`).
- **RVT**: Recurrent Vision Transformer for detection (`evlib.models.RVT`), with the supporting YOLOX FPN and head components.

```python
from evlib.models import E2VID  # requires `pip install evlib[torch]`
```

These require PyTorch (`pip install evlib[torch]`). If PyTorch is not installed, `evlib.models` imports with a warning and exposes only `ModelConfig`. See the [representations API](representations.md) and the `python/evlib/models/` source for configuration and usage details.
