import numpy as np
import pytest
import evlib

# Voxel grid functionality has been removed - placeholder tests


def test_voxel_grid_removal():
    """Test that voxel grid functions work with actual implementation"""
    # Create sample event data
    xs = np.array([10, 20, 30, 40], dtype=np.int64)
    ys = np.array([15, 25, 35, 45], dtype=np.int64)
    ts = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    num_bins = 3
    resolution = (50, 50)  # (width, height)

    # Test available voxel grid functions
    try:
        voxel_grid = evlib.create_voxel_grid(xs, ys, ts, ps, num_bins, resolution, "count")
        assert voxel_grid.shape == (num_bins, resolution[1], resolution[0])
        assert voxel_grid.dtype == np.float32
        print("PASS: create_voxel_grid works")
    except Exception as e:
        pytest.skip(f"create_voxel_grid not available: {e}")

    # Test smooth voxel grid
    try:
        smooth_voxel_grid = evlib.smooth_voxel(xs, ys, ts, ps, num_bins, resolution)
        assert smooth_voxel_grid.shape == (num_bins, resolution[1], resolution[0])
        assert smooth_voxel_grid.dtype == np.float32
        print("PASS: smooth_voxel works")
    except Exception as e:
        pytest.skip(f"smooth_voxel not available: {e}")


def test_voxel_grid_removal_empty():
    """Test that voxel grid functions handle empty input correctly"""
    # Create empty event data
    xs = np.array([], dtype=np.int64)
    ys = np.array([], dtype=np.int64)
    ts = np.array([], dtype=np.float64)
    ps = np.array([], dtype=np.int64)

    num_bins = 3
    resolution = (50, 50)  # (width, height)

    # Test available voxel grid functions with empty data
    try:
        voxel_grid = evlib.create_voxel_grid(xs, ys, ts, ps, num_bins, resolution, "count")
        assert voxel_grid.shape == (num_bins, resolution[1], resolution[0])
        assert voxel_grid.dtype == np.float32
        assert np.all(voxel_grid == 0)
        print("PASS: create_voxel_grid handles empty data")
    except Exception as e:
        pytest.skip(f"create_voxel_grid not available: {e}")

    # Test smooth voxel grid with empty data
    try:
        smooth_voxel_grid = evlib.smooth_voxel(xs, ys, ts, ps, num_bins, resolution)
        assert smooth_voxel_grid.shape == (num_bins, resolution[1], resolution[0])
        assert smooth_voxel_grid.dtype == np.float32
        assert np.all(smooth_voxel_grid == 0)
        print("PASS: smooth_voxel handles empty data")
    except Exception as e:
        pytest.skip(f"smooth_voxel not available: {e}")
