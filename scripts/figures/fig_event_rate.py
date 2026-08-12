"""Generate the events-per-millisecond figure for the README/docs.

Shows the bursty (non-uniform) event rate of the slider_depth sequence, the
motivation for lazy/streaming processing over fixed-rate frame processing.

Usage:
    .venv/bin/python scripts/figures/fig_event_rate.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# House style: Tahoma for all figure text.
matplotlib.rcParams["font.family"] = "Tahoma"

import matplotlib.pyplot as plt
import polars as pl

import evlib

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "data" / "slider_depth" / "events.txt"
OUTPUT_PATH = REPO_ROOT / "docs" / "images" / "fig_event_rate.png"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"required fixture missing: {DATA_PATH}")

    events = evlib.load_events(str(DATA_PATH)).collect(engine="streaming")
    per_ms = (
        events.with_columns((pl.col("t").dt.total_microseconds() // 1000).alias("ms"))
        .group_by("ms")
        .len()
        .sort("ms")
    )

    duration_ms = int(per_ms["ms"].max() - per_ms["ms"].min())

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    ax.plot(per_ms["ms"], per_ms["len"], color="#3a8bff", linewidth=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Events per ms")
    ax.set_title(
        f"Event rate over {duration_ms}ms ({len(events):,} events, slider_depth): "
        "bursty, not uniform"
    )
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({len(events):,} events)")


if __name__ == "__main__":
    main()
