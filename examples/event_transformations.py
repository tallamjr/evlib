#!/usr/bin/env python3
"""
Event transformations example with evlib

This example demonstrates:
1. Flipping events along x and y axes
2. Rotating events
3. Clipping events to bounds
4. Visualizing the transformations
"""
import numpy as np
import matplotlib.pyplot as plt
import evlib


def visualize_transformations(event_sets, titles, figsize=(12, 10)):
    """
    Visualize multiple transformed event sets

    Args:
        event_sets: List of (xs, ys, ts, ps) tuples
        titles: List of subplot titles
        figsize: Figure size tuple
    """
    n = len(event_sets)
    rows = (n + 1) // 2  # Calculate rows needed
    cols = min(n, 2)  # Maximum 2 columns

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if n == 1:
        axes = np.array([axes])

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, ((xs, ys, ts, ps), title) in enumerate(zip(event_sets, titles)):
        row, col = i // cols, i % cols
        ax = axes[row, col]

        # Use polarity for coloring (red=1, blue=-1)
        colors = ["r" if p > 0 else "b" for p in ps]

        ax.scatter(xs, ys, c=colors, s=30, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        # Set consistent limits for better comparison
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()


def main():
    print("evlib event transformations example")

    # Create sample event data
    xs = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64)
    ys = np.array([15, 25, 35, 45, 55, 65, 75], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
    ps = np.array([1, -1, 1, -1, 1, -1, 1], dtype=np.int64)

    # Set the sensor resolution
    sensor_resolution = (100, 100)  # (height, width)

    # 1. Flip events along x-axis
    # Flip events along x axis
    flipped_x_xs, flipped_x_ys, flipped_x_ts, flipped_x_ps = evlib.flip_events_x(
        xs, ys, ts, ps, sensor_resolution
    )

    # 2. Flip events along y-axis
    # Flip events along y axis
    flipped_y_xs, flipped_y_ys, flipped_y_ts, flipped_y_ps = evlib.flip_events_y(
        xs, ys, ts, ps, sensor_resolution
    )

    # 3. Rotate events by 45 degrees
    theta_radians = np.pi / 4  # 45 degrees
    center_of_rotation = (50, 50)  # Center of rotation
    # Rotate events
    rotated_xs, rotated_ys, theta_returned, center_returned = evlib.rotate_events(
        xs,
        ys,
        ts,
        ps,
        sensor_resolution=sensor_resolution,
        theta_radians=theta_radians,
        center_of_rotation=center_of_rotation,
    )

    # 4. Clip events to bounds
    bounds = [30, 70, 30, 70]  # [min_y, max_y, min_x, max_x]
    # Clip events to bounds
    clipped_xs, clipped_ys, clipped_ts, clipped_ps = evlib.clip_events_to_bounds(xs, ys, ts, ps, bounds)

    # Visualize all transformations
    visualize_transformations(
        [
            (xs, ys, ts, ps),
            (flipped_x_xs, flipped_x_ys, ts, ps),
            (flipped_y_xs, flipped_y_ys, ts, ps),
            (rotated_xs, rotated_ys, ts, ps),
            (clipped_xs, clipped_ys, clipped_ts, clipped_ps),
        ],
        [
            "Original Events",
            "Flipped X",
            "Flipped Y",
            "Rotated 45°",
            "Clipped to Bounds",
        ],
    )


if __name__ == "__main__":
    main()
