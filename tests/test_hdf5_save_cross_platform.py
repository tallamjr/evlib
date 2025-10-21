"""
Test HDF5 save functionality across platforms.

This test verifies that HDF5 save works on:
- Linux/macOS with Rust hdf5-metno (when HDF5 feature enabled)
- Windows with Python h5py fallback
"""

import tempfile
import os
import numpy as np
import pytest


def test_hdf5_save_python_fallback():
    """Test HDF5 save functionality (uses h5py fallback when Rust unavailable)."""
    h5py = pytest.importorskip("h5py")
    import evlib

    # Create test data
    n_events = 1000
    xs = np.random.randint(0, 640, n_events, dtype=np.int64)
    ys = np.random.randint(0, 480, n_events, dtype=np.int64)
    ts = np.sort(np.random.uniform(0, 1.0, n_events))
    ps = np.random.choice([-1, 1], n_events).astype(np.int64)

    # Save to temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Use the public save function (uses Python fallback when Rust not available)
        evlib.save_events_to_hdf5(xs, ys, ts, ps, tmp_path)

        # Verify file was created
        assert os.path.exists(tmp_path)

        # Verify contents using h5py
        with h5py.File(tmp_path, "r") as f:
            assert "events" in f
            grp = f["events"]

            # Check datasets exist
            assert "xs" in grp
            assert "ys" in grp
            assert "ts" in grp
            assert "ps" in grp

            # Verify data
            np.testing.assert_array_equal(grp["xs"][:], xs.astype(np.uint16))
            np.testing.assert_array_equal(grp["ys"][:], ys.astype(np.uint16))
            np.testing.assert_array_almost_equal(grp["ts"][:], ts)
            np.testing.assert_array_equal(grp["ps"][:], ps.astype(np.int8))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_hdf5_save_auto_fallback():
    """Test that save_events_to_hdf5 automatically chooses correct implementation."""
    h5py = pytest.importorskip("h5py")
    import evlib

    # Create test data
    n_events = 500
    xs = np.random.randint(0, 640, n_events, dtype=np.int64)
    ys = np.random.randint(0, 480, n_events, dtype=np.int64)
    ts = np.sort(np.random.uniform(0, 1.0, n_events))
    ps = np.random.choice([-1, 1], n_events).astype(np.int64)

    # Save using the auto-fallback function
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # This should work on all platforms
        evlib.save_events_to_hdf5(xs, ys, ts, ps, tmp_path)

        # Verify file exists
        assert os.path.exists(tmp_path)

        # Verify contents
        with h5py.File(tmp_path, "r") as f:
            assert "events" in f
            grp = f["events"]

            # Check all datasets present
            assert set(grp.keys()) == {"xs", "ys", "ts", "ps"}

            # Verify data integrity
            assert len(grp["xs"]) == n_events
            assert len(grp["ys"]) == n_events
            assert len(grp["ts"]) == n_events
            assert len(grp["ps"]) == n_events

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_hdf5_save_validation():
    """Test that save_events_to_hdf5 validates array lengths."""
    pytest.importorskip("h5py")
    import evlib

    # Create mismatched arrays
    xs = np.array([1, 2, 3], dtype=np.int64)
    ys = np.array([1, 2], dtype=np.int64)  # Wrong length
    ts = np.array([0.1, 0.2, 0.3])
    ps = np.array([-1, 1, -1], dtype=np.int64)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Should raise ValueError for mismatched lengths
        with pytest.raises(ValueError, match="Arrays must have the same length"):
            evlib.save_events_to_hdf5(xs, ys, ts, ps, tmp_path)

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_hdf5_save_roundtrip():
    """Test saving and loading HDF5 files."""
    h5py = pytest.importorskip("h5py")
    import evlib

    # Create test data
    n_events = 2000
    xs = np.random.randint(0, 1280, n_events, dtype=np.int64)
    ys = np.random.randint(0, 720, n_events, dtype=np.int64)
    ts = np.sort(np.random.uniform(0, 2.0, n_events))
    ps = np.random.choice([-1, 1], n_events).astype(np.int64)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save events
        evlib.save_events_to_hdf5(xs, ys, ts, ps, tmp_path)

        # Load back using h5py
        with h5py.File(tmp_path, "r") as f:
            grp = f["events"]
            loaded_xs = grp["xs"][:]
            loaded_ys = grp["ys"][:]
            loaded_ts = grp["ts"][:]
            loaded_ps = grp["ps"][:]

        # Verify roundtrip
        np.testing.assert_array_equal(loaded_xs, xs.astype(np.uint16))
        np.testing.assert_array_equal(loaded_ys, ys.astype(np.uint16))
        np.testing.assert_array_almost_equal(loaded_ts, ts)
        np.testing.assert_array_equal(loaded_ps, ps.astype(np.int8))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
