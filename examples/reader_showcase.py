#!/usr/bin/env python3
"""
Showcase evlib's reader across the tracked data formats.

`evlib.load_events` auto-detects the format and returns a Polars LazyFrame with
columns [x, y, t, polarity], where `t` is a microsecond Duration and `polarity`
is encoded per the source file (text slider_depth uses 0/1, EVT2 uses -1/1).

This script reads the two formats tracked in the repo:
  - DAVIS text:      data/slider_depth/events.txt
  - Prophesee EVT2:  data/prophesee/samples/evt2/80_balls.raw

For each, it prints the schema, event count, coordinate and time ranges, and the
polarity balance using Polars only (no numpy-tuple unpacking).
"""

from pathlib import Path

import polars as pl

import evlib

DATASETS = [
    ("DAVIS text (slider_depth)", "data/slider_depth/events.txt"),
    ("Prophesee EVT2 (80_balls)", "data/prophesee/samples/evt2/80_balls.raw"),
]


def show_reader(name: str, file_path: str) -> None:
    """Load one file and report its schema and summary statistics via Polars."""
    print("=" * 60)
    print(f"{name}")
    print(f"Loading: {file_path}")
    print("=" * 60)

    # Lazily load; collect once for the summary so we only scan the file once.
    events = evlib.load_events(file_path)
    print(f"LazyFrame schema: {dict(events.collect_schema())}")

    df = events.collect()
    n_events = df.height
    print(f"Events: {n_events:,}")

    # Coordinate ranges.
    print(
        f"Spatial range: x=[{df['x'].min()}, {df['x'].max()}], "
        f"y=[{df['y'].min()}, {df['y'].max()}]"
    )

    # Time range and span (Duration -> seconds for readability).
    t_min = df["t"].min()
    t_max = df["t"].max()
    span_s = (df["t"].max() - df["t"].min()).total_seconds()
    print(f"Time range: {t_min} to {t_max} (span {span_s:.3f} s)")
    if span_s > 0:
        print(f"Mean event rate: {n_events / span_s:,.0f} events/s")

    # Polarity balance via a Polars group-by.
    balance = df.group_by("polarity").agg(pl.len().alias("count")).sort("polarity")
    print("Polarity balance:")
    for row in balance.iter_rows(named=True):
        share = 100.0 * row["count"] / n_events
        print(f"  polarity {row['polarity']:>2}: {row['count']:>12,}  ({share:5.1f}%)")
    print()


def main() -> None:
    """Run the reader showcase over every tracked dataset that is present."""
    print("evlib Reader Showcase (tracked data)")
    print()

    for name, file_path in DATASETS:
        if Path(file_path).exists():
            show_reader(name, file_path)
        else:
            print(f"Skipping {name}: {file_path} not found\n")

    print("=" * 60)
    print("Done. evlib.load_events auto-detects text and EVT2 formats and")
    print("returns a uniform Polars LazyFrame [x, y, t, polarity].")


if __name__ == "__main__":
    main()
