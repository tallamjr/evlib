#!/usr/bin/env python3
"""
Event augmentation examples with evlib

This example demonstrates:
1. Adding random events
2. Adding correlated events
3. Removing events
4. Visualizing events with matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
import evlib


def visualize_events(event_sets, titles, figsize=(15, 5)):
    """
    Visualize multiple sets of events side by side

    Args:
        event_sets: List of (xs, ys, ts, ps) tuples
        titles: List of subplot titles
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, len(event_sets), figsize=figsize)

    if len(event_sets) == 1:
        axes = [axes]

    for i, ((xs, ys, ts, ps), title) in enumerate(zip(event_sets, titles)):
        # Use polarity for coloring (red=1, blue=-1)
        colors = ["r" if p > 0 else "b" for p in ps]

        axes[i].scatter(xs, ys, c=colors, s=30, alpha=0.7)
        axes[i].set_title(title)
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("evlib event augmentation example")

    # Create sample event data
    xs = np.array([50, 60, 70, 80, 90], dtype=np.int64)
    ys = np.array([50, 60, 70, 80, 90], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    ps = np.array([1, -1, 1, -1, 1], dtype=np.int64)

    # 1. Add random events
    to_add = 15
    # Add random events
    new_xs, new_ys, new_ts, new_ps = evlib.add_random_events(xs, ys, ts, ps, to_add)
    print(f"Original events: {len(xs)}, After adding random events: {len(new_xs)}")

    # 2. Add correlated events
    to_add = 10
    xy_std = 2.0  # Standard deviation for x,y coordinates
    ts_std = 0.005  # Standard deviation for timestamps

    # Add correlated events
    corr_xs, corr_ys, corr_ts, corr_ps = evlib.add_correlated_events(
        xs, ys, ts, ps, to_add, xy_std=xy_std, ts_std=ts_std
    )
    print(f"Original events: {len(xs)}, After adding correlated events: {len(corr_xs)}")

    # 3. Remove events
    to_remove = 2
    # Remove events
    rem_xs, rem_ys, rem_ts, rem_ps = evlib.remove_events(xs, ys, ts, ps, to_remove)
    print(f"Original events: {len(xs)}, After removing events: {len(rem_xs)}")

    # Visualize the results
    visualize_events(
        [
            (xs, ys, ts, ps),
            (new_xs, new_ys, new_ts, new_ps),
            (corr_xs, corr_ys, corr_ts, corr_ps),
            (rem_xs, rem_ys, rem_ts, rem_ps),
        ],
        [
            "Original Events",
            "After Random Events",
            "After Correlated Events",
            "After Removing Events",
        ],
    )


if __name__ == "__main__":
    main()
