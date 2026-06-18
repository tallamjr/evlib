#!/usr/bin/env python3
"""
End-to-end evlib pipeline: load -> filter -> represent -> visualise.

This is the canonical "getting started" script. It runs entirely on tracked data
(data/slider_depth/events.txt) and demonstrates the four core stages:

  1. Load     events into a Polars LazyFrame [x, y, t, polarity].
  2. Filter   with evlib.filtering (a time window and a spatial ROI).
  3. Represent the filtered events as a stacked histogram (long-format DataFrame).
  4. Visualise one densified window slice and save it as a PNG.

Run headless with:
    MPLBACKEND=Agg .venv/bin/python examples/end_to_end_pipeline.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import evlib
import evlib.filtering as evf

# slider_depth sensor geometry: x in [0, 239], y in [0, 179].
DATA_FILE = "data/slider_depth/events.txt"
HEIGHT = 180
WIDTH = 240
OUTPUT_PNG = "examples/end_to_end_pipeline.png"


def densify_window(hist: pl.DataFrame, time_bin: int) -> np.ndarray:
    """Densify one temporal bin into a signed [HEIGHT, WIDTH] image.

    Positive-polarity counts add, negative-polarity counts subtract, so the
    resulting image shows the net activity for that slice.
    """
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    sub = hist.filter(pl.col("time_bin") == time_bin)
    for row in sub.iter_rows(named=True):
        sign = 1.0 if row["polarity"] > 0 else -1.0
        frame[row["y"], row["x"]] += sign * row["count"]
    return frame


def main() -> None:
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(
            f"{DATA_FILE} not found. This example needs the tracked slider_depth data."
        )

    # Stage 1: load.
    print("1. Load")
    events = evlib.load_events(DATA_FILE)
    n_raw = events.collect().height
    print(f"   loaded {n_raw:,} events as a LazyFrame {dict(events.collect_schema())}")

    # Stage 2: filter. Keep a 1 second window, then a central spatial ROI.
    print("2. Filter")
    filtered = evf.filter_by_time(events, t_start=0.5, t_end=1.5)
    filtered = evf.filter_by_roi(filtered, x_min=40, x_max=200, y_min=20, y_max=160)
    n_filtered = filtered.collect().height
    print(
        f"   time window [0.5, 1.5] s and ROI x=[40, 200], y=[20, 160] "
        f"-> {n_filtered:,} events ({100 * n_filtered / n_raw:.1f}% kept)"
    )

    # Stage 3: represent as a stacked histogram (long-format DataFrame).
    print("3. Represent")
    bins = 6
    hist = evlib.create_stacked_histogram(
        filtered,
        height=HEIGHT,
        width=WIDTH,
        bins=bins,
        window_duration_ms=50.0,
    )
    print(
        f"   stacked histogram: {hist.height:,} non-zero cells, "
        f"{hist['time_bin'].n_unique()} temporal bins, columns {hist.columns}"
    )

    # Stage 4: visualise the densified temporal bins for one window.
    print("4. Visualise")
    n_show = min(3, bins)
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]
    fig.suptitle("End-to-end pipeline: net activity per temporal bin", fontsize=14)

    limit = float(hist["count"].max())
    for i in range(n_show):
        image = densify_window(hist, time_bin=i)
        axes[i].imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[i].set_title(f"Time bin {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=100)
    plt.close()
    print(f"   saved figure to {OUTPUT_PNG}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
