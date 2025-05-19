#!/usr/bin/env python3
"""
Test the voxel grid function
"""
import numpy as np
import evlib

# Create example data
xs = np.array([10, 20, 30], dtype=np.int64)
ys = np.array([40, 50, 60], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3], dtype=np.float64)
ps = np.array([1, -1, 1], dtype=np.int64)

try:
    # Try with all the parameters explicit
    result = evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, 3, (100, 100), "count")
    print("Success:", result.shape)
except Exception as e:
    print("Error:", e)
