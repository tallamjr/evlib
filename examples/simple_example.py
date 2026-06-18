#!/usr/bin/env python3
"""
Simple example: load events and create a stacked histogram with an RVT-style config.

`create_stacked_histogram` takes an in-memory LazyFrame/DataFrame (not a path) and
returns a long-format Polars DataFrame with columns [time_bin, polarity, y, x, count].
"""

import evlib


def simple_example():
    """Load tracked data and build a stacked histogram with an RVT-style config."""

    data_file = "data/slider_depth/events.txt"
    print(f"Loading {data_file} ...")

    # load_events returns a Polars LazyFrame [x, y, t, polarity]
    events = evlib.load_events(data_file)

    # RVT-style configuration: 10 temporal bins over 50 ms windows.
    print("Creating stacked histogram ...")
    stacked_hist = evlib.create_stacked_histogram(
        events,
        height=180,  # slider_depth sensor height (y in [0, 179])
        width=240,  # slider_depth sensor width  (x in [0, 239])
        bins=10,  # RVT standard: 10 temporal bins per window
        window_duration_ms=50.0,  # RVT standard: 50 ms windows
    )

    # The result is a long-format Polars DataFrame, not a dense numpy array.
    n_rows = stacked_hist.height
    n_bins = stacked_hist["time_bin"].n_unique()
    polarities = sorted(stacked_hist["polarity"].unique().to_list())
    total_count = stacked_hist["count"].sum()

    print(f"SUCCESS: stacked histogram has {n_rows:,} non-zero cells")
    print(f"  - columns: {stacked_hist.columns}")
    print(f"  - temporal bins: {n_bins}")
    print(f"  - polarities: {polarities}")
    print(f"  - total event count across cells: {total_count:,}")

    return stacked_hist


if __name__ == "__main__":
    histogram = simple_example()
