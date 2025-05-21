# \!/usr/bin/env python3
"""
Test script for the wrapper functions
"""
import numpy as np
import evlib

# Create event data (1D arrays)
xs = np.array([10, 20, 30, 40, 50], dtype=np.int64)
ys = np.array([15, 25, 35, 45, 55], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1], dtype=np.int64)

# Convert events to voxel grid
num_bins = 5
resolution = (100, 100)  # (width, height)

# Test standard voxel grid wrapper
print("Testing standard voxel grid wrapper...")
try:
    voxel_grid = evlib.create_voxel_grid(xs, ys, ts, ps, num_bins, resolution, "count")
    print(f"Standard voxel grid shape: {voxel_grid.shape}")
    print(f"Max value: {np.max(voxel_grid)}")
    print(f"Sum: {np.sum(voxel_grid)}")
except Exception as e:
    print(f"Error with standard voxel grid wrapper: {e}")

# Test smooth voxel grid wrapper
print("\nTesting smooth voxel grid wrapper...")
try:
    # Test with fixed parameters
    interp = "trilinear"
    smooth_voxel_grid = evlib.create_smooth_voxel_grid(xs, ys, ts, ps, num_bins, resolution, interp)
    print(f"Smooth voxel grid shape: {smooth_voxel_grid.shape}")
    print(f"Max value: {np.max(smooth_voxel_grid)}")
    print(f"Sum: {np.sum(smooth_voxel_grid)}")
except Exception as e:
    print(f"Error with smooth voxel grid wrapper: {e}")

# Test different interpolation methods
print("\nTesting different interpolation methods:")
for method in ["trilinear", "bilinear", "temporal"]:
    try:
        grid = evlib.create_smooth_voxel_grid(xs, ys, ts, ps, num_bins, resolution, method)
        print(f"Method {method}: Shape {grid.shape}, Sum {np.sum(grid)}")
    except Exception as e:
        print(f"Error with method {method}: {e}")
