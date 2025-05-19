#!/usr/bin/env python3
"""
evlib Showcase - Demonstrating the full functionality of the event camera utilities

This example demonstrates:
1. Creating and manipulating event data
2. Applying various augmentations (noise, rotations, etc.)
3. Creating event representations (voxel grids, timestamp images)
4. Performing contrast maximization for motion estimation
5. Visualizing events and results
"""
import matplotlib.pyplot as plt
import numpy as np

import evlib


def create_synthetic_events(num_events, pattern="spiral"):
    """Create synthetic events for demonstration purposes"""
    width, height = 128, 128
    center_x, center_y = width // 2, height // 2
    xs = np.zeros(num_events, dtype=np.int64)
    ys = np.zeros(num_events, dtype=np.int64)
    ts = np.zeros(num_events, dtype=np.float64)
    ps = np.zeros(num_events, dtype=np.int64)

    if pattern == "spiral":
        for i in range(num_events):
            t = i / num_events
            angle = t * 4 * np.pi
            radius = t * 40

            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            xs[i] = round(x)
            ys[i] = round(y)
            ts[i] = t
            ps[i] = 1 if i % 2 == 0 else -1
    else:  # random pattern
        xs = np.random.randint(0, width, num_events, dtype=np.int64)
        ys = np.random.randint(0, height, num_events, dtype=np.int64)
        ts = np.sort(np.random.random(num_events))
        ps = np.random.choice([-1, 1], num_events, dtype=np.int64)

    return xs, ys, ts, ps


def visualize_events(event_sets, titles, figsize=(20, 5)):
    """Visualize multiple sets of events side by side"""
    fig, axes = plt.subplots(1, len(event_sets), figsize=figsize)

    if len(event_sets) == 1:
        axes = [axes]

    for i, ((xs, ys, ts, ps), title) in enumerate(zip(event_sets, titles)):
        # Use polarity for coloring (red=1, blue=-1)
        colors = ["r" if p > 0 else "b" for p in ps]

        axes[i].scatter(xs, ys, c=colors, s=5, alpha=0.7)
        axes[i].set_title(title)
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].grid(True)
        axes[i].set_aspect("equal")

    plt.tight_layout()
    return fig


def visualize_voxel_grid(voxel_grid, title="Voxel Grid"):
    """Visualize a voxel grid as a 3D scatter plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Process the voxel grid to get indices of non-zero elements
    n_bins, height, width = voxel_grid.shape
    nonzero_indices = np.nonzero(voxel_grid)
    t_indices, y_indices, x_indices = nonzero_indices
    values = voxel_grid[nonzero_indices]

    # Normalize values for color intensity
    values_norm = values / np.max(values) if np.max(values) > 0 else values

    # Create a colormap (time bins determine color)
    t_norm = t_indices / max(1, n_bins - 1)

    # Plot non-zero voxels as scatter points
    scatter = ax.scatter(
        x_indices,
        y_indices,
        t_indices,
        c=t_norm,
        s=5 + 30 * values_norm,
        cmap="viridis",
        alpha=0.7,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time Bin")
    ax.set_title(title)

    # Add a colorbar to show the time mapping
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Normalized Time")

    return fig


def main():
    print("evlib showcase example")

    # Create synthetic events
    num_events = 500
    xs, ys, ts, ps = create_synthetic_events(num_events)
    print(f"Created {num_events} synthetic events")

    # Display original events
    visualize_events([(xs, ys, ts, ps)], ["Original Events"])
    plt.show()

    # Apply augmentations
    print("\nApplying augmentations...")

    # Add random events
    xs_rand, ys_rand, ts_rand, ps_rand = evlib.add_random_events(xs, ys, ts, ps, to_add=200)
    print(f"Added random events. New count: {len(xs_rand)}")

    # Add correlated events
    xs_corr, ys_corr, ts_corr, ps_corr = evlib.add_correlated_events(
        xs, ys, ts, ps, to_add=150, xy_std=3.0, ts_std=0.01
    )
    print(f"Added correlated events. New count: {len(xs_corr)}")

    # Rotate events
    sensor_resolution = (128, 128)
    xs_rot, ys_rot, theta, center = evlib.rotate_events(
        xs,
        ys,
        ts,
        ps,
        sensor_resolution=sensor_resolution,
        theta_radians=np.radians(45),
    )
    print("Rotated events by 45 degrees")

    # Visualize augmented events
    visualize_events(
        [
            (xs, ys, ts, ps),
            (xs_rand, ys_rand, ts_rand, ps_rand),
            (xs_corr, ys_corr, ts_corr, ps_corr),
            (xs_rot, ys_rot, ts, ps),
        ],
        [
            "Original Events",
            "With Random Events",
            "With Correlated Events",
            "Rotated Events",
        ],
    )
    plt.show()

    # Create representations
    print("\nCreating representations...")

    # Check the module functionality and use appropriately
    # Create voxel grid
    try:
        # Flatten the arrays for the voxel grid function
        xs_flat = xs.flatten() if hasattr(xs, "flatten") else xs
        ys_flat = ys.flatten() if hasattr(ys, "flatten") else ys
        ts_flat = ts.flatten() if hasattr(ts, "flatten") else ts
        ps_flat = ps.flatten() if hasattr(ps, "flatten") else ps

        # Convert to voxel grid using evlib function
        voxel_grid = evlib.representations.events_to_voxel_grid_py(
            xs_flat, ys_flat, ts_flat, ps_flat, 5, sensor_resolution, "count"
        )
    except Exception as e:
        print(f"Note: Falling back to simple representation: {e}")
        # Simple implementation for demonstration
        voxel_grid = np.zeros((5, 128, 128))
        for i in range(5):
            for x, y, t, p in zip(xs, ys, ts, ps):
                bin_idx = min(int(t * 5), 4)
                if bin_idx == i and 0 <= x < 128 and 0 <= y < 128:
                    voxel_grid[i, int(y), int(x)] += 1
    print(f"Created voxel grid with shape: {voxel_grid.shape}")

    # Visualize voxel grid
    visualize_voxel_grid(voxel_grid)
    plt.show()

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
