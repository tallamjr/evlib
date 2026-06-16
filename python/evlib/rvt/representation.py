"""Polars stacked-histogram builder. Produces a sparse count table identical to RVT's dense
StackedHistogram, with the 2x nearest-exact downsample folded into a coordinate filter.

Engine selection
----------------
The ``engine`` argument is forwarded to ``LazyFrame.collect(engine=...)``. Accepted values:
``"auto"``, ``"in-memory"``, ``"streaming"``, ``"gpu"`` (string) or a ``pl.GPUEngine(...)``
instance. GPU and streaming are mutually exclusive in Polars: the GPU (cudf-polars) backend
does not stream. Requesting the GPU engine on a host without CUDA / cudf-polars is safe; Polars
transparently falls back to the default CPU engine and produces identical output, so no explicit
fallback handling is needed here."""

from typing import Union

import numpy as np
import polars as pl

from evlib.rvt.downsample import selected_source_indices

EngineType = str  # "auto" | "in-memory" | "streaming" | "gpu"


def build_sparse_histogram(
    events: Union[pl.DataFrame, pl.LazyFrame],
    ev_repr_timestamps_us: np.ndarray,
    delta_t_us: int,
    nbins: int,
    count_cutoff: int,
    height: int,
    width: int,
    downsample_by_2: bool,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    lf = events.lazy() if isinstance(events, pl.DataFrame) else events

    # --- window assignment ---
    # RVT slices each window i from the globally sorted event array as
    #   [searchsorted(t, T_i - delta_t, "left"), searchsorted(t, T_i, "right"))
    # which keeps every event with  T_i - delta_t <= t <= T_i  (both ends inclusive).
    # When two consecutive grid timestamps are exactly delta_t apart, an event sitting on
    # that shared boundary therefore belongs to BOTH windows. We must replicate this
    # duplication, so an event is assigned to the contiguous range of windows
    #   lo = first i with T_i >= t      (forward as-of on t)
    #   hi = last  i with T_i <= t + dt (backward as-of on t + dt)
    # and the range [lo, hi] is exploded out. For almost all events lo == hi (single
    # window); the duplicate only appears at a shared boundary.
    grid = np.asarray(ev_repr_timestamps_us, dtype=np.int64)
    grid_ids = np.arange(len(grid), dtype=np.int64)
    lo_lf = pl.DataFrame({"lo": grid_ids, "T_lo": grid}).lazy().sort("T_lo")
    hi_lf = pl.DataFrame({"hi": grid_ids, "T_hi": grid}).lazy().sort("T_hi")
    lf = lf.sort("t")
    lf = lf.join_asof(lo_lf, left_on="t", right_on="T_lo", strategy="forward")
    lf = lf.with_columns((pl.col("t") + delta_t_us).alias("_t_hi"))
    lf = lf.join_asof(hi_lf, left_on="_t_hi", right_on="T_hi", strategy="backward")
    lf = lf.filter(
        pl.col("lo").is_not_null()
        & pl.col("hi").is_not_null()
        & (pl.col("lo") <= pl.col("hi"))
    )
    lf = lf.with_columns(
        pl.int_ranges(pl.col("lo"), pl.col("hi") + 1).alias("window_id")
    ).explode("window_id")

    # --- per-window time binning (MUST happen before the downsample fold) ---
    # RVT computes t0 = time[0], t1 = time[-1] over ALL events in the window slice, builds the
    # full-resolution histogram, and only THEN downsamples. So the normalization range t0/t1 must
    # be taken over the complete event set. If we applied the downsample coordinate filter first,
    # the window's first/last event would be dropped whenever it lands on a non-selected pixel,
    # changing t0/t1 and shifting every event's bin (observed as off-by-one-bin, same total count).
    #
    # Binning is done in Float64, not Float32: torch's int64 true-divide yields a correctly-rounded
    # float32, but Polars float32 division uses an approximate/SIMD reciprocal that rounds the wrong
    # way at exact bin boundaries (e.g. 5000/50000*10, true value 1.0, floors to 0 instead of 1).
    # Float64 is accurate and reproduces torch's result bit-for-bit on this data.
    t0 = pl.col("t").min().over("window_id")
    t1 = pl.col("t").max().over("window_id")
    denom = (t1 - t0).clip(lower_bound=1).cast(pl.Float64)
    t_norm = ((pl.col("t") - t0).cast(pl.Float64) / denom) * pl.lit(
        nbins, dtype=pl.Float64
    )
    t_idx = t_norm.floor().cast(pl.Int32).clip(upper_bound=nbins - 1)
    lf = lf.with_columns(t_idx.alias("t_idx")).with_columns(
        (
            pl.col("p").cast(pl.Int32).clip(lower_bound=0) * nbins + pl.col("t_idx")
        ).alias("channel")
    )

    # --- downsample fold (gen4): nearest-exact gather applied as a coordinate filter ---
    if downsample_by_2:
        rows = selected_source_indices(height, height // 2)
        cols = selected_source_indices(width, width // 2)
        row_lut = pl.DataFrame(
            {
                "y": np.asarray(rows, dtype=np.int64),
                "y_out": np.arange(len(rows), dtype=np.int64),
            }
        ).lazy()
        col_lut = pl.DataFrame(
            {
                "x": np.asarray(cols, dtype=np.int64),
                "x_out": np.arange(len(cols), dtype=np.int64),
            }
        ).lazy()
        lf = (
            lf.with_columns(pl.col("y").cast(pl.Int64), pl.col("x").cast(pl.Int64))
            .join(row_lut, on="y", how="inner")
            .join(col_lut, on="x", how="inner")
            .drop("y", "x")
            .rename({"y_out": "y", "x_out": "x"})
        )
        out_h, out_w = height // 2, width // 2
    else:
        out_h, out_w = height, width

    # --- aggregate counts per (window, channel, output pixel) and clip to count_cutoff ---
    lf = (
        lf.filter(
            pl.col("x").cast(pl.Int64).is_between(0, out_w - 1)
            & pl.col("y").cast(pl.Int64).is_between(0, out_h - 1)
        )
        .group_by(["window_id", "channel", "y", "x"])
        .agg(pl.len().alias("count"))
        .with_columns(pl.col("count").clip(upper_bound=count_cutoff).cast(pl.UInt32))
        .sort(["window_id", "channel", "y", "x"])
    )

    return lf.collect(engine=engine)
