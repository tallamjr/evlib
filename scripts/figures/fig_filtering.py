"""Generate the hot-pixel/noise filtering before-after figure for the docs.

Left panel: raw event counts per pixel, hot pixels visible as bright points.
Right panel: the same sequence after `filter_hot_pixels` + `filter_noise`.

Usage:
    .venv/bin/python scripts/figures/fig_filtering.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# House style: Tahoma for all figure text.
matplotlib.rcParams["font.family"] = "Tahoma"

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import evlib
import evlib.filtering as evf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "data" / "slider_depth" / "events.txt"
OUTPUT_PATH = REPO_ROOT / "docs" / "images" / "fig_filtering.png"


def _pixel_counts(events: pl.DataFrame, height: int, width: int) -> np.ndarray:
    counts = np.zeros((height, width), np.int64)
    np.add.at(counts, (events["y"].to_numpy(), events["x"].to_numpy()), 1)
    return counts


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"required fixture missing: {DATA_PATH}")

    raw = evlib.load_events(str(DATA_PATH))
    raw_events = raw.collect(engine="streaming")
    width = int(raw_events["x"].max()) + 1
    height = int(raw_events["y"].max()) + 1

    filtered = evf.filter_noise(evf.filter_hot_pixels(raw), method="refractory")
    filtered_events = filtered.collect(engine="streaming")

    raw_counts = _pixel_counts(raw_events, height, width)
    filtered_counts = _pixel_counts(filtered_events, height, width)
    vmax = float(np.percentile(raw_counts[raw_counts > 0], 99))

    fig, (ax_raw, ax_filtered) = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
    for ax, counts, title, n in (
        (ax_raw, raw_counts, "Raw events", len(raw_events)),
        (
            ax_filtered,
            filtered_counts,
            "filter_hot_pixels + filter_noise(refractory)",
            len(filtered_events),
        ),
    ):
        ax.imshow(counts, cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.02,
            0.02,
            f"{n:,} events",
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            va="bottom",
            ha="left",
        )

    fig.suptitle("evlib.filtering: hot-pixel and refractory noise removal", fontsize=12)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, facecolor="white")
    plt.close(fig)
    print(
        f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(raw_events):,} -> {len(filtered_events):,} events)"
    )


if __name__ == "__main__":
    main()
