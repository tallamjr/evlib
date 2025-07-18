"""
Test HDF5 loading functionality with various file formats.
Note: HDF5 save functionality is not tested as it's not part of the core use case.
"""

import numpy as np
import tempfile
import os
import evlib


def test_hdf5_load_various_formats():
    """Test that the load function can handle various HDF5 formats including legacy ones"""
    # Create sample event data
    num_events = 100
    xs = np.random.randint(0, 100, num_events, dtype=np.int64)
    ys = np.random.randint(0, 100, num_events, dtype=np.int64)
    ts = np.sort(np.random.random(num_events).astype(np.float64))
    ps = np.random.choice([-1, 1], num_events).astype(np.int64)

    # Test 1: Legacy format (separate datasets in root with different names)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name

    try:
        # Create legacy format manually
        import h5py

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("t", data=ts)
            f.create_dataset("x", data=xs.astype(np.uint16))
            f.create_dataset("y", data=ys.astype(np.uint16))
            f.create_dataset("p", data=ps.astype(np.int8))

        # Load and verify using main API
        df = evlib.load_events(hdf5_path).collect()
        loaded_xs = df["x"].to_numpy()
        loaded_ys = df["y"].to_numpy()
        loaded_ts = df["timestamp"].cast(float).to_numpy() / 1_000_000  # Convert Duration[us] to seconds
        loaded_ps = df["polarity"].to_numpy()

        assert np.array_equal(xs, loaded_xs), "Legacy format: X coordinates do not match"
        assert np.array_equal(ys, loaded_ys), "Legacy format: Y coordinates do not match"
        assert np.allclose(ts, loaded_ts), "Legacy format: Timestamps do not match"
        assert np.array_equal(ps, loaded_ps), "Legacy format: Polarities do not match"

    finally:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

    # Test 2: Alternative naming convention
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name

    try:
        # Create alternative format manually
        import h5py

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("timestamps", data=ts)
            f.create_dataset("x_pos", data=xs.astype(np.uint16))
            f.create_dataset("y_pos", data=ys.astype(np.uint16))
            f.create_dataset("polarity", data=ps.astype(np.int8))

        # Load and verify using main API
        df = evlib.load_events(hdf5_path).collect()
        loaded_xs = df["x"].to_numpy()
        loaded_ys = df["y"].to_numpy()
        loaded_ts = df["timestamp"].cast(float).to_numpy() / 1_000_000  # Convert Duration[us] to seconds
        loaded_ps = df["polarity"].to_numpy()

        assert np.array_equal(xs, loaded_xs), "Alternative format: X coordinates do not match"
        assert np.array_equal(ys, loaded_ys), "Alternative format: Y coordinates do not match"
        assert np.allclose(ts, loaded_ts), "Alternative format: Timestamps do not match"
        assert np.array_equal(ps, loaded_ps), "Alternative format: Polarities do not match"

    finally:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

    # Test 3: Events group format (common in real datasets)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name

    try:
        # Create events group format manually
        import h5py

        with h5py.File(hdf5_path, "w") as f:
            events_group = f.create_group("events")
            events_group.create_dataset("ts", data=ts)
            events_group.create_dataset("xs", data=xs.astype(np.uint16))
            events_group.create_dataset("ys", data=ys.astype(np.uint16))
            events_group.create_dataset("ps", data=ps.astype(np.int8))

        # Load and verify using main API
        df = evlib.load_events(hdf5_path).collect()
        loaded_xs = df["x"].to_numpy()
        loaded_ys = df["y"].to_numpy()
        loaded_ts = df["timestamp"].cast(float).to_numpy() / 1_000_000  # Convert Duration[us] to seconds
        loaded_ps = df["polarity"].to_numpy()

        assert np.array_equal(xs, loaded_xs), "Events group format: X coordinates do not match"
        assert np.array_equal(ys, loaded_ys), "Events group format: Y coordinates do not match"
        assert np.allclose(ts, loaded_ts), "Events group format: Timestamps do not match"
        assert np.array_equal(ps, loaded_ps), "Events group format: Polarities do not match"

    finally:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)


def test_hdf5_empty_file():
    """Test HDF5 loading with empty datasets"""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name

    try:
        # Create empty HDF5 file
        import h5py

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("t", data=np.array([], dtype=np.float64))
            f.create_dataset("x", data=np.array([], dtype=np.uint16))
            f.create_dataset("y", data=np.array([], dtype=np.uint16))
            f.create_dataset("p", data=np.array([], dtype=np.int8))

        # Load and verify
        df = evlib.load_events(hdf5_path).collect()
        loaded_xs = df["x"].to_numpy()
        loaded_ys = df["y"].to_numpy()
        loaded_ts = df["timestamp"].cast(float).to_numpy() / 1_000_000
        loaded_ps = df["polarity"].to_numpy()

        assert len(loaded_xs) == 0, "Expected empty X coordinates"
        assert len(loaded_ys) == 0, "Expected empty Y coordinates"
        assert len(loaded_ts) == 0, "Expected empty timestamps"
        assert len(loaded_ps) == 0, "Expected empty polarities"

    finally:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)


if __name__ == "__main__":
    test_hdf5_load_various_formats()
    test_hdf5_empty_file()
    print("All HDF5 loading tests passed!")
