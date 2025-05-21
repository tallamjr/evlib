# This file is required for Python to recognize this directory as a package
# Import the actual functionality from the Rust extension module
# ruff: noqa: F403
from evlib.evlib import *

import evlib.evlib as evlib_rust

# Import numpy for reshaping
import numpy as np


# Define wrapper functions for the voxel grid representations
def create_voxel_grid(xs, ys, ts, ps, num_bins, resolution=(None, None), method="count"):
    """
    Create a voxel grid representation from event data.

    Args:
        xs: Array of x coordinates
        ys: Array of y coordinates
        ts: Array of timestamps
        ps: Array of polarities
        num_bins: Number of time bins for the voxel grid
        resolution: Tuple of (width, height) for the output grid
        method: Method to accumulate events ("count", "polarity", "binary")

    Returns:
        A 3D numpy array with shape (num_bins, height, width)
    """
    try:
        # Try the direct approach (returns tuple of (flat_array, shape))
        # Access through the explicit import
        result = evlib_rust.evlib_rust.representations.events_to_voxel_grid(
            xs, ys, ts, ps, num_bins, resolution, method
        )

        if isinstance(result, tuple) and len(result) == 2:
            # Unpack and reshape
            flat_data, shape = result
            return flat_data.reshape(shape)
        else:
            # Already in correct shape
            return result
    except Exception as e:
        print(f"Warning: Error creating voxel grid: {e}")
        print("Using fallback method")

        # Fallback implementation using numpy
        if resolution[0] is None or resolution[1] is None:
            # Determine resolution
            width = int(max(xs)) + 1 if len(xs) > 0 else 1
            height = int(max(ys)) + 1 if len(ys) > 0 else 1
        else:
            width, height = resolution

        # Initialize grid
        grid = np.zeros((num_bins, height, width), dtype=np.float32)

        if len(ts) == 0:
            return grid

        # Determine time range
        t_min, t_max = min(ts), max(ts)
        t_range = t_max - t_min if t_max > t_min else 1.0

        # Process each event
        for x, y, t, p in zip(xs, ys, ts, ps):
            if 0 <= x < width and 0 <= y < height:
                # Calculate normalized timestamp
                t_norm = (t - t_min) / t_range

                # Determine bin
                bin_idx = min(int(t_norm * num_bins), num_bins - 1)

                # Update grid based on method
                if method == "binary":
                    grid[bin_idx, y, x] = 1.0
                elif method == "polarity":
                    grid[bin_idx, y, x] += float(p)
                else:  # "count"
                    grid[bin_idx, y, x] += 1.0

        return grid


def create_smooth_voxel_grid(xs, ys, ts, ps, num_bins, resolution=(None, None), interpolation="trilinear"):
    """
    Create a smooth voxel grid representation from event data.

    Args:
        xs: Array of x coordinates
        ys: Array of y coordinates
        ts: Array of timestamps
        ps: Array of polarities
        num_bins: Number of time bins for the voxel grid
        resolution: Tuple of (width, height) for the output grid
        interpolation: Interpolation method ("trilinear", "bilinear", "temporal")

    Returns:
        A 3D numpy array with shape (num_bins, height, width)
    """
    try:
        # Try the direct approach (returns tuple of (flat_array, shape))
        # Access through the explicit import
        result = evlib_rust.evlib_rust.representations.events_to_smooth_voxel_grid(
            xs, ys, ts, ps, num_bins, resolution, interpolation
        )

        if isinstance(result, tuple) and len(result) == 2:
            # Unpack and reshape
            flat_data, shape = result
            return flat_data.reshape(shape)
        else:
            # Already in correct shape
            return result
    except Exception as e:
        print(f"Warning: Error creating smooth voxel grid: {e}")
        print("Using fallback method with standard voxel grid (no interpolation)")

        # Fall back to regular voxel grid
        return create_voxel_grid(xs, ys, ts, ps, num_bins, resolution, "count")
