"""Polars stacked-histogram builder. Produces a sparse count table identical to RVT's dense
StackedHistogram, with the 2x nearest-exact downsample folded into a coordinate filter.

Engine selection
----------------
The ``engine`` argument is forwarded to ``LazyFrame.collect(engine=...)``. Accepted values:
``"auto"``, ``"in-memory"``, ``"streaming"``, ``"gpu"`` (string) or a ``pl.GPUEngine(...)``
instance. GPU and streaming are mutually exclusive in Polars: the GPU (cudf-polars) backend
does not stream. Requesting the GPU engine on a host without CUDA / cudf-polars is safe; Polars
transparently falls back to the default CPU engine and produces the same output, so no explicit
fallback handling is needed here.

Two window-assignment front-ends share one binning/aggregation core (``_bin_downsample_aggregate``):

* :func:`build_sparse_histogram` cross-joins events with a (small) window-end grid and filters
  the RVT membership predicate. Used for bounded per-batch work; GPU-supported.
* :func:`build_sparse_histogram_assigned` takes events already tagged with ``window_id`` (assigned
  upstream via ``np.searchsorted``), so a whole sequence can be aggregated in one large GPU pass
  without the cross-join blow-up.
"""

from typing import Union

import numpy as np
import polars as pl

from evlib.rvt.downsample import selected_source_indices

EngineType = str  # "auto" | "in-memory" | "streaming" | "gpu"


def _bin_downsample_aggregate(
    lf: pl.LazyFrame,
    *,
    delta_t_us: int,
    nbins: int,
    count_cutoff: int,
    height: int,
    width: int,
    downsample_by_2: bool,
) -> pl.LazyFrame:
    """Bit-identity-critical core: per-window time binning -> downsample fold -> count aggregate.

    ``lf`` must already carry the columns ``t, x, y, p, window_id`` (each event-window membership
    is one row, boundary duplicates included). Shared by both window-assignment front-ends so the
    binning math has a single source of truth.
    """
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
    # clip()s are written as when/then because the cudf-polars GPU engine has no `clip`
    # unary. Each is exactly equivalent to the clip it replaces (same dtype), so the output
    # matches torch RVT exactly. denom keeps the Int64 div-by-zero guard (span == 0
    # for a single-timestamp window).
    _span = t1 - t0
    denom = (
        pl.when(_span < 1)
        .then(pl.lit(1, dtype=pl.Int64))
        .otherwise(_span)
        .cast(pl.Float64)
    )
    t_norm = ((pl.col("t") - t0).cast(pl.Float64) / denom) * pl.lit(
        nbins, dtype=pl.Float64
    )
    _idx = t_norm.floor().cast(pl.Int32)
    t_idx = (
        pl.when(_idx > nbins - 1)
        .then(pl.lit(nbins - 1, dtype=pl.Int32))
        .otherwise(_idx)
    )
    _p = pl.col("p").cast(pl.Int32)
    lf = lf.with_columns(t_idx.alias("t_idx")).with_columns(
        (
            pl.when(_p < 0).then(pl.lit(0, dtype=pl.Int32)).otherwise(_p) * nbins
            + pl.col("t_idx")
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
        .with_columns(
            pl.when(pl.col("count") > count_cutoff)
            .then(pl.lit(count_cutoff, dtype=pl.UInt32))
            .otherwise(pl.col("count"))
            .cast(pl.UInt32)
            .alias(
                "count"
            )  # when/then does not preserve the input column name (clip did)
        )
        .sort(["window_id", "channel", "y", "x"])
    )
    return lf


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

    # --- window assignment (cross-join + membership filter) ---
    # RVT slices each window i from the globally sorted event array as
    #   [searchsorted(t, T_i - delta_t, "left"), searchsorted(t, T_i, "right"))
    # which keeps every event with  T_i - delta_t <= t <= T_i  (both ends inclusive).
    # When two consecutive grid timestamps are exactly delta_t apart, an event sitting on
    # that shared boundary therefore belongs to BOTH windows.
    #
    # We reproduce that exactly by cross-joining each (already window-bounded) batch of events
    # with the batch's small window-end grid and keeping the membership predicate. The set of
    # windows kept for an event is precisely { i : t <= T_i <= t + delta_t }, the same as the
    # forward/backward as-of range the previous implementation exploded, and the shared-boundary
    # event naturally matches two rows. Crucially, cross-join + filter are supported by the
    # cudf-polars GPU engine, whereas join_asof and int_ranges are not, so the whole query now
    # runs on the GPU instead of silently falling back to CPU. (For a whole-sequence single pass
    # without the cross-join blow-up, use build_sparse_histogram_assigned with searchsorted ids.)
    grid = np.asarray(ev_repr_timestamps_us, dtype=np.int64)
    grid_lf = pl.DataFrame(
        {"window_id": np.arange(len(grid), dtype=np.int64), "T": grid}
    ).lazy()
    lf = (
        lf.join(grid_lf, how="cross")
        .filter(
            (pl.col("t") >= pl.col("T") - delta_t_us) & (pl.col("t") <= pl.col("T"))
        )
        .drop("T")
    )

    lf = _bin_downsample_aggregate(
        lf,
        delta_t_us=delta_t_us,
        nbins=nbins,
        count_cutoff=count_cutoff,
        height=height,
        width=width,
        downsample_by_2=downsample_by_2,
    )
    return lf.collect(engine=engine)


def build_sparse_histogram_assigned(
    events_with_window_id: Union[pl.DataFrame, pl.LazyFrame],
    delta_t_us: int,
    nbins: int,
    count_cutoff: int,
    height: int,
    width: int,
    downsample_by_2: bool,
    engine: EngineType = "auto",
) -> pl.DataFrame:
    """Aggregate events already tagged with ``window_id`` (one row per event-window membership).

    Window assignment is done upstream with ``np.searchsorted`` (O(n_events)), so a whole sequence
    (or a large multi-window batch) can be aggregated in a single GPU pass without the per-window
    cross-join blow-up. Output matches :func:`build_sparse_histogram` exactly.
    """
    lf = (
        events_with_window_id.lazy()
        if isinstance(events_with_window_id, pl.DataFrame)
        else events_with_window_id
    )
    lf = _bin_downsample_aggregate(
        lf,
        delta_t_us=delta_t_us,
        nbins=nbins,
        count_cutoff=count_cutoff,
        height=height,
        width=width,
        downsample_by_2=downsample_by_2,
    )
    return lf.collect(engine=engine)
