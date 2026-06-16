"""Polars stacked-histogram builder. Produces a sparse count table identical to RVT's dense
StackedHistogram, with the 2x nearest-exact downsample folded into a coordinate filter."""

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

    # --- window assignment: window_id = first grid index with T_end >= t; keep if t >= T_end - dt ---
    grid = np.asarray(ev_repr_timestamps_us, dtype=np.int64)
    grid_lf = pl.DataFrame(
        {"window_id": np.arange(len(grid), dtype=np.int64), "T_end": grid}
    ).lazy()
    # forward as-of join: attach the first T_end >= t
    lf = lf.sort("t").join_asof(
        grid_lf.sort("T_end"), left_on="t", right_on="T_end", strategy="forward"
    )
    lf = lf.filter(
        pl.col("T_end").is_not_null() & (pl.col("t") >= (pl.col("T_end") - delta_t_us))
    )

    # --- downsample fold (gen4) ---
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

    # --- per-window float32 time binning ---
    # torch computes: t_norm = (t - t0).float() / max(t1 - t0, 1); * nbins; floor; clamp to nbins-1.
    # The true-divide of an int64 tensor by an int produces float32 in torch, so we cast the
    # numerator to Float32 BEFORE the division and keep the denominator Float32 too.
    t0 = pl.col("t").min().over("window_id")
    t1 = pl.col("t").max().over("window_id")
    denom = (t1 - t0).clip(lower_bound=1).cast(pl.Float32)
    t_norm = ((pl.col("t") - t0).cast(pl.Float32) / denom) * pl.lit(
        nbins, dtype=pl.Float32
    )
    t_idx = t_norm.floor().cast(pl.Int32).clip(upper_bound=nbins - 1)

    lf = (
        lf.filter(
            pl.col("x").cast(pl.Int64).is_between(0, out_w - 1)
            & pl.col("y").cast(pl.Int64).is_between(0, out_h - 1)
        )
        .with_columns(t_idx.alias("t_idx"))
        .with_columns(
            (
                pl.col("p").cast(pl.Int32).clip(lower_bound=0) * nbins + pl.col("t_idx")
            ).alias("channel")
        )
        .group_by(["window_id", "channel", "y", "x"])
        .agg(pl.len().alias("count"))
        .with_columns(pl.col("count").clip(upper_bound=count_cutoff).cast(pl.UInt32))
        .sort(["window_id", "channel", "y", "x"])
    )

    return lf.collect(engine=engine)
