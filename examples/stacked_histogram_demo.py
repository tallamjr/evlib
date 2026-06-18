#!/usr/bin/env python3
"""
Demonstration of the stacked histogram representation in evlib.

`create_stacked_histogram` returns a long-format Polars DataFrame with columns
[time_bin, polarity, y, x, count]. This script loads tracked data, builds the
histogram, then densifies a couple of (time_bin, polarity) slices into 2-D numpy
arrays purely for plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import evlib

DATA_FILE = "data/slider_depth/events.txt"
HEIGHT = 180  # slider_depth sensor height (y in [0, 179])
WIDTH = 240  # slider_depth sensor width  (x in [0, 239])


def densify_slice(hist: pl.DataFrame, time_bin: int, polarity: int) -> np.ndarray:
    """Scatter a single (time_bin, polarity) slice into a dense [HEIGHT, WIDTH] array."""
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    sub = hist.filter(
        (pl.col("time_bin") == time_bin) & (pl.col("polarity") == polarity)
    )
    ys = sub["y"].to_numpy()
    xs = sub["x"].to_numpy()
    counts = sub["count"].to_numpy()
    frame[ys, xs] = counts
    return frame


def demonstrate_stacked_histogram():
    """Build a stacked histogram from tracked data and visualise a few slices."""
    print("Stacked Histogram Demonstration")
    print("=" * 40)

    # Load events as a LazyFrame [x, y, t, polarity].
    events = evlib.load_events(DATA_FILE)
    df = events.collect()
    print(f"Loaded {df.height:,} events from {DATA_FILE}")
    print(
        f"Spatial range: x=[{df['x'].min()}, {df['x'].max()}], "
        f"y=[{df['y'].min()}, {df['y'].max()}]"
    )

    bins = 8
    hist = evlib.create_stacked_histogram(
        events,
        height=HEIGHT,
        width=WIDTH,
        bins=bins,
        window_duration_ms=50.0,
    )

    print(f"\nStacked histogram: {hist.height:,} non-zero cells")
    print(f"Columns: {hist.columns}")
    print(f"Temporal bins: {sorted(hist['time_bin'].unique().to_list())}")
    print(f"Polarities: {sorted(hist['polarity'].unique().to_list())}")
    print(f"Count range: [{hist['count'].min()}, {hist['count'].max()}]")

    vmax = float(hist["count"].max())
    polarities = sorted(hist["polarity"].unique().to_list())
    pos_pol, neg_pol = polarities[-1], polarities[0]

    # Visualise the first few temporal bins for both polarities.
    max_bins_to_show = min(4, bins)
    fig, axes = plt.subplots(2, max_bins_to_show, figsize=(16, 8))
    fig.suptitle("Stacked Histogram Visualisation", fontsize=16)

    for i in range(max_bins_to_show):
        pos_slice = densify_slice(hist, time_bin=i, polarity=pos_pol)
        axes[0, i].imshow(pos_slice, cmap="Reds", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"Positive\nTime Bin {i}")
        axes[0, i].axis("off")

        neg_slice = densify_slice(hist, time_bin=i, polarity=neg_pol)
        axes[1, i].imshow(neg_slice, cmap="Blues", vmin=0, vmax=vmax)
        axes[1, i].set_title(f"Negative\nTime Bin {i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = "/tmp/stacked_histogram_demo.png"
    plt.savefig(out_path)
    plt.close()
    print(f"\nSaved figure to {out_path}")

    # Compare with the voxel grid representation [x, y, time_bin, contribution].
    voxel_grid = evlib.create_voxel_grid(
        events, height=HEIGHT, width=WIDTH, n_time_bins=bins
    )
    print(f"\nVoxel grid: {voxel_grid.height:,} cells, columns {voxel_grid.columns}")
    print(
        f"Contribution range: "
        f"[{voxel_grid['contribution'].min():.3f}, {voxel_grid['contribution'].max():.3f}]"
    )

    print("\nDemonstration complete!")


if __name__ == "__main__":
    demonstrate_stacked_histogram()
