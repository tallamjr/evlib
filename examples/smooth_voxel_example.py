#!/usr/bin/env python3
"""
Smooth Voxel Grid Example - Demonstrating the smooth voxel grid representation

This example demonstrates:
1. Creating synthetic event data
2. Converting events to smooth voxel grid representations using different interpolation methods
3. Visualizing the differences between standard and smooth voxel grids
"""
import matplotlib.pyplot as plt
import numpy as np

import evlib


def create_circular_motion_events(num_events=1000, radius=30, center_x=64, center_y=64):
    """Create synthetic events in a circular motion pattern"""
    xs = np.zeros(num_events, dtype=np.int64)
    ys = np.zeros(num_events, dtype=np.int64)
    ts = np.zeros(num_events, dtype=np.float64)
    ps = np.zeros(num_events, dtype=np.int64)

    for i in range(num_events):
        t = i / num_events
        angle = t * 2 * np.pi

        # Add some noise to the radius to make it less perfect
        curr_radius = radius + np.random.normal(0, 2)

        x = center_x + curr_radius * np.cos(angle)
        y = center_y + curr_radius * np.sin(angle)

        xs[i] = round(x)
        ys[i] = round(y)
        ts[i] = t
        ps[i] = 1 if i % 2 == 0 else -1

    return xs, ys, ts, ps


def visualize_voxel_grid_slices(voxel_grid, title="Voxel Grid Slices", figsize=(15, 10)):
    """Visualize a voxel grid as a series of time-bin slices"""
    n_bins = voxel_grid.shape[0]
    fig, axes = plt.subplots(1, n_bins, figsize=figsize)

    # Normalize for better visualization
    vmin = np.min(voxel_grid)
    vmax = np.max(voxel_grid) if np.max(voxel_grid) > 0 else 1.0

    for i in range(n_bins):
        ax = axes[i] if n_bins > 1 else axes
        img = ax.imshow(voxel_grid[i], cmap="plasma", vmin=vmin, vmax=vmax)
        ax.set_title(f"Bin {i}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img, cax=cbar_ax)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust for the colorbar
    return fig


def visualize_voxel_grid_3d(voxel_grid, title="3D Voxel Grid", threshold=0.01):
    """Visualize a voxel grid as a 3D scatter plot with a threshold for visibility"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n_bins, height, width = voxel_grid.shape

    # Get indices of elements above threshold
    z, y, x = np.where(voxel_grid > threshold * np.max(voxel_grid))
    values = voxel_grid[z, y, x]

    # Normalize values for point size
    values_norm = values / np.max(values) if np.max(values) > 0 else values

    # Plot the points
    scatter = ax.scatter(x, y, z, c=z, s=50 * values_norm, cmap="viridis", alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time Bin")
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Time Bin")

    return fig


def compare_voxel_methods(events, resolution=(128, 128), num_bins=5):
    """Compare standard voxel grid with smooth voxel grid using different interpolation methods"""
    xs, ys, ts, ps = events

    # Create standard voxel grid using wrapper function
    std_voxel_data, std_shape = evlib.representations.events_to_voxel_grid(
        xs, ys, ts, ps, num_bins, resolution, "count"
    )
    std_voxel_grid = std_voxel_data.reshape(std_shape)
    print(f"Standard voxel grid shape: {std_voxel_grid.shape}, sum: {np.sum(std_voxel_grid)}")

    # Create smooth voxel grids with different interpolation methods using wrapper functions
    smooth_tri_data, smooth_tri_shape = evlib.representations.events_to_smooth_voxel_grid(
        xs, ys, ts, ps, num_bins, resolution, "trilinear"
    )
    smooth_trilinear = smooth_tri_data.reshape(smooth_tri_shape)
    print(f"Trilinear voxel grid shape: {smooth_trilinear.shape}, sum: {np.sum(smooth_trilinear)}")

    smooth_bil_data, smooth_bil_shape = evlib.representations.events_to_smooth_voxel_grid(
        xs, ys, ts, ps, num_bins, resolution, "bilinear"
    )
    smooth_bilinear = smooth_bil_data.reshape(smooth_bil_shape)
    print(f"Bilinear voxel grid shape: {smooth_bilinear.shape}, sum: {np.sum(smooth_bilinear)}")

    smooth_temp_data, smooth_temp_shape = evlib.representations.events_to_smooth_voxel_grid(
        xs, ys, ts, ps, num_bins, resolution, "temporal"
    )
    smooth_temporal = smooth_temp_data.reshape(smooth_temp_shape)
    print(f"Temporal voxel grid shape: {smooth_temporal.shape}, sum: {np.sum(smooth_temporal)}")

    # Return all grids for possible visualization if needed
    return {
        "standard": std_voxel_grid,
        "trilinear": smooth_trilinear,
        "bilinear": smooth_bilinear,
        "temporal": smooth_temporal,
    }


def main():
    """Example demonstrating smooth voxel grid representation"""
    print("Smooth Voxel Grid Example")

    # Create synthetic events
    print("Creating synthetic events...")
    events = create_circular_motion_events(num_events=200)

    try:
        # Compare voxel methods
        print("Comparing voxel grid methods...")
        results = compare_voxel_methods(events, resolution=(128, 128), num_bins=3)

        # Print summary of each method
        for method, grid in results.items():
            print(f"{method.capitalize()} method: shape={grid.shape}, max={np.max(grid)}, min={np.min(grid)}")

    except Exception as e:
        print(f"Error: {e}")
        print("Note: This example requires the smooth voxel grid implementation.")

    print("Example completed!")


if __name__ == "__main__":
    main()
