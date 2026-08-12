"""Generate the Polars-vs-naive-struct memory bar chart for the docs.

Compares evlib's Polars on-wire layout (`x:Int16 + y:Int16 + t:Duration(i64)
+ polarity:Int8`) against a naive per-event struct. The Polars figure is
computed from real loaded data (`df.estimated_size() / len(df)`); the naive
figure (110 bytes/event: float64 x/y/t + int32 polarity + object overhead)
is the reference value already quoted in docs/index.md.

Usage:
    .venv/bin/python scripts/figures/fig_memory.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# House style: Tahoma for all figure text.
matplotlib.rcParams["font.family"] = "Tahoma"

import matplotlib.pyplot as plt

import evlib

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "data" / "slider_depth" / "events.txt"
OUTPUT_PATH = REPO_ROOT / "docs" / "images" / "fig_memory.png"

NAIVE_BYTES_PER_EVENT = 110.0


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"required fixture missing: {DATA_PATH}")

    events = evlib.load_events(str(DATA_PATH)).collect(engine="streaming")
    polars_bytes_per_event = events.estimated_size() / len(events)

    labels = [
        "Naive struct\n(float64 x/y/t + int32 polarity)",
        "evlib Polars\n(Int16/Int16/Duration(i64)/Int8)",
    ]
    values = [NAIVE_BYTES_PER_EVENT, polars_bytes_per_event]
    colors = ["#888888", "#3a8bff"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{value:.1f} B",
            ha="center",
            fontsize=10,
        )
    ax.set_ylabel("Bytes per event")
    ax.set_title(
        f"evlib is {NAIVE_BYTES_PER_EVENT / polars_bytes_per_event:.1f}x smaller than a naive "
        f"struct (slider_depth, {len(events):,} events)"
    )
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, facecolor="white")
    plt.close(fig)
    print(
        f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)} ({polars_bytes_per_event:.1f} bytes/event)"
    )


if __name__ == "__main__":
    main()
