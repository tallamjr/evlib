#!/usr/bin/env python3
"""
Generate and visualize synthetic DVS event data with evlib

This example demonstrates:
1. Creating synthetic DVS event data from moving patterns
2. Applying various transformations to the events
3. Visualizing events in 3D space (x, y, t)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import evlib


def generate_moving_bar_events(width=128, height=128, num_frames=10):
    """
    Generate synthetic DVS events from a moving horizontal bar pattern

    Args:
        width: Width of the simulated sensor
        height: Height of the simulated sensor
        num_frames: Number of frames in the animation

    Returns:
        xs, ys, ts, ps: Event arrays
    """
    events = []
    timestamps = np.linspace(0, 1, num_frames)

    # Create moving horizontal bar
    for i, t in enumerate(timestamps):
        y_pos = int(height * (i / num_frames))

        # Generate positive events at leading edge
        for x in range(width):
            if i > 0 and y_pos > 0:
                events.append((x, y_pos, t, 1))  # Positive events (brightness increase)
                events.append((x, y_pos - 1, t, -1))  # Negative events (brightness decrease)

    # Convert to numpy arrays
    xs = np.array([e[0] for e in events], dtype=np.int64)
    ys = np.array([e[1] for e in events], dtype=np.int64)
    ts = np.array([e[2] for e in events], dtype=np.float64)
    ps = np.array([e[3] for e in events], dtype=np.int64)

    return xs, ys, ts, ps


def plot_events_3d(xs, ys, ts, ps, title, fig=None, ax=None):
    """
    Create 3D visualization of DVS events over time

    Args:
        xs, ys, ts, ps: Event arrays
        title: Plot title
        fig, ax: Optional figure and axis objects
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Color by polarity
    colors = ["r" if p > 0 else "b" for p in ps]

    # Plot 3D points
    ax.scatter(xs, ys, ts, c=colors, s=5, alpha=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time")
    ax.set_title(title)

    return fig, ax


def display_events_as_frames(xs, ys, ts, ps, width, height, num_frames=10):
    """
    Display events as an animation of accumulated frames

    Args:
        xs, ys, ts, ps: Event arrays
        width, height: Dimensions of the sensor
        num_frames: Number of frames to split events into
    """
    # Create time bins
    t_min, t_max = ts.min(), ts.max()
    time_bins = np.linspace(t_min, t_max, num_frames + 1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize empty frame
    frame = np.zeros((height, width, 3), dtype=np.float32)
    im = ax.imshow(frame, animated=True)
    ax.set_title("DVS Events Visualization")

    def update(frame_idx):
        # Clear frame
        frame = np.zeros((height, width, 3), dtype=np.float32)

        # Get events for this time bin
        t_start = time_bins[frame_idx]
        t_end = time_bins[frame_idx + 1]

        mask = (ts >= t_start) & (ts < t_end)
        frame_xs = xs[mask]
        frame_ys = ys[mask]
        frame_ps = ps[mask]

        # Map events to the frame
        for x, y, p in zip(frame_xs, frame_ys, frame_ps):
            if 0 <= x < width and 0 <= y < height:
                if p > 0:
                    frame[y, x] = [1, 0, 0]  # Red for positive
                else:
                    frame[y, x] = [0, 0, 1]  # Blue for negative

        im.set_array(frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=200)
    plt.show()

    return ani


def main():
    print("evlib synthetic DVS data example")

    # Generate sample DVS events
    width, height = 128, 128
    xs, ys, ts, ps = generate_moving_bar_events(width, height, 20)
    print(f"Generated {len(xs)} events")

    # Apply correlated noise
    new_xs, new_ys, new_ts, new_ps = evlib.add_correlated_events(
        xs,
        ys,
        ts,
        ps,
        to_add=len(xs) // 4,  # Add 25% more events
        xy_std=1.0,  # Low spatial spread
        ts_std=0.01,  # Low temporal spread
    )

    sensor_resolution = (height, width)

    # Flip along both axes (rotate 180 degrees)
    flipped_xs, flipped_ys, _, _ = evlib.flip_events_x(new_xs, new_ys, new_ts, new_ps, sensor_resolution)
    flipped_xs, flipped_ys, flipped_ts, flipped_ps = evlib.flip_events_y(
        flipped_xs, flipped_ys, new_ts, new_ps, sensor_resolution
    )

    # Plot original and transformed events
    fig = plt.figure(figsize=(15, 5))

    # Original events
    ax1 = fig.add_subplot(131, projection="3d")
    plot_events_3d(xs, ys, ts, ps, "Original DVS Events", fig, ax1)

    # Events with correlated noise
    ax2 = fig.add_subplot(132, projection="3d")
    plot_events_3d(new_xs, new_ys, new_ts, new_ps, "Events with Correlated Noise", fig, ax2)

    # 180° rotated events
    ax3 = fig.add_subplot(133, projection="3d")
    plot_events_3d(flipped_xs, flipped_ys, flipped_ts, flipped_ps, "180° Rotated Events", fig, ax3)

    plt.tight_layout()
    plt.show()

    # Display events as frames
    print("Displaying events as an animation...")
    display_events_as_frames(new_xs, new_ys, new_ts, new_ps, width, height)


if __name__ == "__main__":
    main()
