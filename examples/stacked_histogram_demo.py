#!/usr/bin/env python3
"""
Demonstration of stacked histogram functionality in evlib.

This script shows how to use the stacked histogram representation
to process event camera data with temporal binning.
"""

import matplotlib.pyplot as plt
import numpy as np

import evlib


def load_sample_data():
    """Load sample event data from the slider_depth dataset."""
    filename = "data/slider_depth/events.txt"

    try:
        # Use evlib to load the data properly
        df = evlib.load_events(filename).collect()

        # Extract arrays from the DataFrame
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        pol = df["polarity"].to_numpy()
        timestamp = df["timestamp"].cast(float).to_numpy() / 1_000_000  # Convert to seconds

        print(f"Successfully loaded {len(x)} events from {filename}")
        return x, y, pol, timestamp

    except FileNotFoundError:
        print(f"Real data file not found: {filename}")
        print("Please ensure the data directory is available")
        raise


def demonstrate_stacked_histogram():
    """Demonstrate stacked histogram functionality."""
    print("Stacked Histogram Demonstration")
    print("=" * 40)

    # Load data
    filename = "data/slider_depth/events.txt"
    x, y, pol, timestamp = load_sample_data()

    # Use subset of data for demonstration
    num_events = min(5000, len(x))
    x = x[:num_events]
    y = y[:num_events]
    pol = pol[:num_events]
    timestamp = timestamp[:num_events]

    print(f"Processing {num_events} events")
    print(f"Time range: {timestamp[0]:.6f} to {timestamp[-1]:.6f} seconds")
    print(f"Spatial range: x=[{x.min()}, {x.max()}], y=[{y.min()}, {y.max()}]")

    # Create stacked histogram
    bins = 8
    height = 240
    width = 346

    stacked_hist = evlib.create_stacked_histogram(
        filename, height=height, width=width, nbins=bins, window_duration_ms=50.0, count_cutoff=255
    )

    print(f"\nStacked histogram shape: {stacked_hist.shape}")
    print(f"Expected shape: [2*{bins}, {height}, {width}] = [{2*bins}, {height}, {width}]")
    print(f"Value range: [{stacked_hist.min()}, {stacked_hist.max()}]")

    # Visualize results - show first 4 bins for clarity
    max_bins_to_show = min(4, bins)
    fig, axes = plt.subplots(2, max_bins_to_show, figsize=(16, 8))
    fig.suptitle("Stacked Histogram Visualization", fontsize=16)

    # Use the first window for visualization
    window_idx = 0

    # Show positive events for each time bin
    for i in range(max_bins_to_show):
        ax = axes[0, i]
        pos_slice = stacked_hist[window_idx, i, :, :]
        ax.imshow(pos_slice, cmap="Reds", vmin=0, vmax=stacked_hist.max())
        ax.set_title(f"Positive Events\nTime Bin {i}")
        ax.axis("off")

    # Show negative events for each time bin
    for i in range(max_bins_to_show):
        ax = axes[1, i]
        neg_slice = stacked_hist[window_idx, bins + i, :, :]
        ax.imshow(neg_slice, cmap="Blues", vmin=0, vmax=stacked_hist.max())
        ax.set_title(f"Negative Events\nTime Bin {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("/tmp/stacked_histogram_demo.png")
    plt.close()

    # Compare with other representations
    print("\nComparing with other representations:")

    # Voxel grid
    voxel_grid = evlib.create_voxel_grid(filename, height=height, width=width, nbins=bins)
    print(f"Voxel grid shape: {voxel_grid.shape}")
    print(f"Voxel grid value range: [{voxel_grid.min():.3f}, {voxel_grid.max():.3f}]")

    # Simple event histogram (manual implementation since no evlib function exists)
    event_hist = np.zeros((2, height, width), dtype=np.int32)
    for i in range(num_events):
        event_hist[pol[i], y[i], x[i]] += 1
    print(f"Event histogram shape: {event_hist.shape}")
    print(f"Event histogram value range: [{event_hist.min()}, {event_hist.max()}]")

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Representation Comparison", fontsize=16)

    # Stacked histogram middle bin - First window, middle time bin
    middle_window = 0
    middle_time_bin = bins // 2
    axes[0, 0].imshow(stacked_hist[middle_window, middle_time_bin, :, :], cmap="Reds")
    axes[0, 0].set_title("Stacked Histogram\n(Middle Time Bin, Positive)")
    axes[0, 0].axis("off")

    # Voxel grid middle bin
    axes[0, 1].imshow(voxel_grid[bins // 2, :, :].T, cmap="RdBu_r")
    axes[0, 1].set_title("Voxel Grid\n(Middle Time Bin)")
    axes[0, 1].axis("off")

    # Stacked histogram negative events - First window, middle time bin + bins offset
    axes[1, 0].imshow(stacked_hist[middle_window, bins + middle_time_bin, :, :], cmap="Blues")
    axes[1, 0].set_title("Stacked Histogram\n(Middle Time Bin, Negative)")
    axes[1, 0].axis("off")

    # Event histogram positive
    axes[1, 1].imshow(event_hist[1], cmap="viridis")
    axes[1, 1].set_title("Event Histogram\n(Positive Events)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("/tmp/representation_comparison.png")
    plt.close()

    print("\nDemonstration complete!")


if __name__ == "__main__":
    demonstrate_stacked_histogram()
