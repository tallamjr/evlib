# Core API Reference

The core loading API is `evlib.load_events`. It returns a single Polars `LazyFrame`, not separate NumPy arrays. Everything downstream (filtering, representations, models) operates on that frame.

## `evlib.load_events`

```python
def load_events(
    path,
    t_start=None,   # inclusive lower time bound, seconds
    t_end=None,     # inclusive upper time bound, seconds
    min_x=None, max_x=None,   # inclusive spatial bounds
    min_y=None, max_y=None,
    polarity=None,  # keep only events with this polarity value
    sort=True,      # sort by timestamp after filtering
) -> pl.LazyFrame
```

The Rust loader detects the format, decodes the file, and builds the frame. Any time, spatial, or polarity bounds you pass are applied here as Polars expressions, so loading and filtering fuse into one lazy, GPU-collectable query. With `sort=True` (the default) the result is sorted by timestamp.

### Returned frame

A Polars `LazyFrame` with four columns:

| Column     | Type                 | Notes                                            |
|------------|----------------------|--------------------------------------------------|
| `x`        | integer              | Pixel x coordinate                               |
| `y`        | integer              | Pixel y coordinate                               |
| `t`        | Duration(microseconds) | Timestamp; use `pl.col("t").dt.total_microseconds()` to get an integer |
| `polarity` | integer              | Encoding depends on the source format; see below |

`t` is a Polars Duration in microseconds, **not** float seconds. To work in seconds, divide the microsecond count: `pl.col("t").dt.total_microseconds() / 1_000_000`. `load_events` does **not** convert polarity encoding: the value reflects the on-disk format. Text files typically load as `0`/`1`; EVT2, EVT3, and AEDAT typically load as `-1`/`+1`. See [Polarity Encoding Mismatch](../user-guide/formats.md#polarity-encoding-mismatch) in the formats guide for the per-format table.

Because the result is a `LazyFrame`, nothing is computed until you `collect()`. Collect with a selectable engine: `"streaming"` for large CPU datasets, or `"gpu"` where cudf-polars and CUDA are available.

### Example

```python
import evlib
import polars as pl

# Lazy load with built-in filters fused into the query
events = evlib.load_events(
    "data/prophesee/samples/evt2/80_balls.raw",
    t_start=24.0, t_end=25.0,   # seconds (file spans ~23.6s-29.9s)
    polarity=1,
)

# Add more Polars expressions lazily, then collect once
df = (
    events
    .filter(pl.col("x").is_between(100, 500))
    .collect(engine="streaming")
)

print(f"Loaded {len(df):,} events")
print(f"Resolution: {df['x'].max()} x {df['y'].max()}")
print(f"Duration:   {df['t'].max() - df['t'].min()}")

# Convert timestamps to seconds when needed
seconds = df.select(
    (pl.col("t").dt.total_microseconds() / 1_000_000).alias("t_seconds")
)
```

## Related functions

- `evlib.detect_format(path)` and `evlib.get_format_description(...)`: automatic format detection.
- `evlib.save_events_to_text(...)` and `evlib.save_events_to_hdf5(...)`: writing events back out (HDF5 save uses the Rust `hdf5` feature on Linux/macOS, falling back to `h5py` elsewhere).

See the [formats reference](formats.md) for the full list of supported input formats, and the [processing reference](processing.md) for filtering and representations.
