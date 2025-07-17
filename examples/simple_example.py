#!/usr/bin/env python3
"""
Simple example: Load data and create stacked histogram with RVT configuration
"""

import evlib.representations as evr


def simple_example():
    """Simple example matching RVT's typical configuration."""

    # Load data and create stacked histogram with RVT's standard config
    print("Loading data and creating stacked histogram...")

    # RVT's standard configuration
    stacked_hist = evr.create_stacked_histogram(
        "data/slider_depth/events.txt",  # Event file path
        height=240,  # Sensor height
        width=346,  # Sensor width
        nbins=10,  # RVT standard: 10 temporal bins
        window_duration_ms=50.0,  # RVT standard: 50ms windows
        count_cutoff=10,  # RVT standard: count cutoff of 10
    )

    print(f"✓ Created stacked histogram with shape: {stacked_hist.shape}")
    print(f"  - {stacked_hist.shape[0]} time windows")
    print(f"  - {stacked_hist.shape[1]} channels (20 = 10 bins × 2 polarities)")
    print(f"  - {stacked_hist.shape[2]}×{stacked_hist.shape[3]} spatial resolution")
    print(f"  - Data type: {stacked_hist.dtype}")
    print(f"  - Memory usage: {stacked_hist.nbytes / 1024 / 1024:.1f} MB")

    return stacked_hist


if __name__ == "__main__":
    histogram = simple_example()
