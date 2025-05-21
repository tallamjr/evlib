#!/usr/bin/env python3
"""
Test the smooth voxel grid function
"""
import numpy as np
import evlib

# Use wrapper function instead of direct access
create_smooth_voxel_grid = evlib.create_smooth_voxel_grid


def test_events_to_smooth_voxel_grid_py():
    """Test creating a smooth voxel grid from events"""
    # Create sample event data with non-integer positions for interpolation test
    xs = np.array([10, 20, 30, 40], dtype=np.int64)
    ys = np.array([15, 25, 35, 45], dtype=np.int64)
    ts = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    num_bins = 5
    resolution = (50, 50)  # (width, height)

    # Create voxel grid with default (trilinear) interpolation
    voxel_grid = create_smooth_voxel_grid(xs, ys, ts, ps, num_bins, resolution)

    # Check shape and type
    assert voxel_grid.shape == (num_bins, resolution[1], resolution[0])
    assert voxel_grid.dtype == np.float32

    # Check that the voxel grid contains non-zero elements
    assert np.sum(voxel_grid) > 0


def test_events_to_smooth_voxel_grid_py_empty():
    """Test creating a smooth voxel grid from empty events"""
    # Create empty event data
    xs = np.array([], dtype=np.int64)
    ys = np.array([], dtype=np.int64)
    ts = np.array([], dtype=np.float64)
    ps = np.array([], dtype=np.int64)

    num_bins = 3
    resolution = (50, 50)  # (width, height)

    # Create voxel grid
    voxel_grid = create_smooth_voxel_grid(xs, ys, ts, ps, num_bins, resolution)

    # Check shape and type
    assert voxel_grid.shape == (num_bins, resolution[1], resolution[0])
    assert voxel_grid.dtype == np.float32

    # Check that the voxel grid is all zeros
    assert np.all(voxel_grid == 0)


def test_different_interpolation_methods():
    """Test different interpolation methods for smooth voxel grid"""
    # Create sample event data
    xs = np.array([10, 20, 30, 40], dtype=np.int64)
    ys = np.array([15, 25, 35, 45], dtype=np.int64)
    ts = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    num_bins = 5
    resolution = (50, 50)  # (width, height)

    # Test different interpolation methods
    interpolation_methods = ["trilinear", "bilinear", "temporal"]

    for method in interpolation_methods:
        # Create smooth voxel grid
        voxel_grid = create_smooth_voxel_grid(xs, ys, ts, ps, num_bins, resolution, method)

        # Check shape and type
        assert voxel_grid.shape == (num_bins, resolution[1], resolution[0])
        assert voxel_grid.dtype == np.float32

        # Sum of all voxel values should be non-zero
        assert np.sum(voxel_grid) != 0


if __name__ == "__main__":
    # Run the tests manually if this file is executed
    test_events_to_smooth_voxel_grid_py()
    test_events_to_smooth_voxel_grid_py_empty()
    test_different_interpolation_methods()
    print("All tests passed!")
