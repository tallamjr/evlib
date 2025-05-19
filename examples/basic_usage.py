#!/usr/bin/env python3
"""
Basic usage example of evlib - Event camera utilities

This example demonstrates how to:
1. Create and manipulate event data
2. Convert events to block representation
3. Use the hello_world test function
"""
import numpy as np
import evlib


def main():
    print("evlib basic usage example")

    # Create example event data
    xs = np.array([10, 20, 30, 40], dtype=np.int64)
    ys = np.array([50, 60, 70, 80], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    print(f"Event data shape: x={xs.shape}, y={ys.shape}, t={ts.shape}, p={ps.shape}")

    # Convert to block representation
    block = evlib.core.events_to_block_py(xs, ys, ts, ps)
    print(f"Block representation shape: {block.shape}")
    print("Block format (x,y,t,p):")
    print(block)

    # Add more events
    added_xs, added_ys, added_ts, added_ps = evlib.augmentation.add_random_events_py(xs, ys, ts, ps, to_add=3)

    print("\nAfter adding random events:")
    print(f"Original count: {len(xs)}, New count: {len(added_xs)}")

    # Print events as a table
    print("\nEvents table (x, y, t, p):")
    for x, y, t, p in zip(added_xs, added_ys, added_ts, added_ps):
        print(f"{x:3d}, {y:3d}, {t:.3f}, {p:2d}")


if __name__ == "__main__":
    main()
