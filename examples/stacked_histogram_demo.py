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
    filename = "../data/slider_depth/events.txt"

    try:
        with open(filename, "r") as f:
            lines = f.readlines()

        timestamp = []
        x = []
        y = []
        pol = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 4:
                timestamp.append(float(parts[0]))
                x.append(int(parts[1]))
                y.append(int(parts[2]))
                pol.append(int(parts[3]))

        return np.array(x), np.array(y), np.array(pol), np.array(timestamp)

    except FileNotFoundError:
        print(f"Real data file not found: {filename}")
        print("Using synthetic data instead")
        return generate_synthetic_data()


def generate_synthetic_data():
    """Generate synthetic event data for demonstration."""
    num_events = 1000

    # Create random events
    x = np.random.randint(0, 240, num_events)
    y = np.random.randint(0, 180, num_events)
    pol = np.random.choice([0, 1], num_events)
    timestamp = np.sort(np.random.uniform(0, 1, num_events))

    return x, y, pol, timestamp


def demonstrate_stacked_histogram():
    """Demonstrate stacked histogram functionality."""
    print("Stacked Histogram Demonstration")
    print("=" * 40)

    # Load data
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
    height = 180
    width = 240

    stacked_hist = evlib.stacked_histogram(
        x, y, pol, timestamp, bins=bins, height=height, width=width, count_cutoff=255, fastmode=True
    )

    print(f"\nStacked histogram shape: {stacked_hist.shape}")
    print(f"Expected shape: [2*{bins}, {height}, {width}] = [{2*bins}, {height}, {width}]")
    print(f"Value range: [{stacked_hist.min()}, {stacked_hist.max()}]")

    # Visualize results - show first 4 bins for clarity
    max_bins_to_show = min(4, bins)
    fig, axes = plt.subplots(2, max_bins_to_show, figsize=(16, 8))
    fig.suptitle("Stacked Histogram Visualization", fontsize=16)

    # Show positive events for each time bin
    for i in range(max_bins_to_show):
        ax = axes[0, i]
        pos_slice = stacked_hist[i, :, :]
        im = ax.imshow(pos_slice, cmap="Reds", vmin=0, vmax=stacked_hist.max())
        ax.set_title(f"Positive Events\nTime Bin {i}")
        ax.axis("off")

    # Show negative events for each time bin
    for i in range(max_bins_to_show):
        ax = axes[1, i]
        neg_slice = stacked_hist[bins + i, :, :]
        im = ax.imshow(neg_slice, cmap="Blues", vmin=0, vmax=stacked_hist.max())
        ax.set_title(f"Negative Events\nTime Bin {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("/tmp/stacked_histogram_demo.png")
    plt.close()

    # Compare with other representations
    print("\nComparing with other representations:")

    # Voxel grid
    voxel_grid = evlib.create_voxel_grid(
        x, y, timestamp, pol, sensor_resolution=(width, height), num_bins=bins
    )
    print(f"Voxel grid shape: {voxel_grid.shape}")
    print(f"Voxel grid value range: [{voxel_grid.min():.3f}, {voxel_grid.max():.3f}]")

    # Time surface
    time_surface = evlib.create_time_surface(
        x, y, timestamp, pol, sensor_resolution=(width, height), polarity_separate=True
    )
    print(f"Time surface shape: {time_surface.shape}")
    print(f"Time surface value range: [{time_surface.min():.3f}, {time_surface.max():.3f}]")

    # Event histogram
    event_hist = evlib.create_event_histogram(
        x, y, pol, sensor_resolution=(width, height), polarity_separate=True
    )
    print(f"Event histogram shape: {event_hist.shape}")
    print(f"Event histogram value range: [{event_hist.min()}, {event_hist.max()}]")

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Representation Comparison", fontsize=16)

    # Stacked histogram middle bin
    axes[0, 0].imshow(stacked_hist[bins // 2, :, :], cmap="Reds")
    axes[0, 0].set_title("Stacked Histogram\n(Middle Time Bin, Positive)")
    axes[0, 0].axis("off")

    # Voxel grid middle bin
    axes[0, 1].imshow(voxel_grid[:, :, bins // 2].T, cmap="RdBu_r")
    axes[0, 1].set_title("Voxel Grid\n(Middle Time Bin)")
    axes[0, 1].axis("off")

    # Time surface positive
    axes[1, 0].imshow(time_surface[1], cmap="Reds")
    axes[1, 0].set_title("Time Surface\n(Positive Events)")
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
