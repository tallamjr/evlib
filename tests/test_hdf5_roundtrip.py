"""
Test HDF5 save/load round-trip functionality to ensure data integrity.
"""

import numpy as np
import pytest
import tempfile
import os
import evlib


def test_hdf5_save_load_roundtrip():
    """Test that HDF5 save/load maintains data integrity"""
    # Create sample event data
    num_events = 1000
    xs = np.random.randint(0, 100, num_events, dtype=np.int64)
    ys = np.random.randint(0, 100, num_events, dtype=np.int64)
    ts = np.sort(np.random.random(num_events).astype(np.float64))
    ps = np.random.choice([-1, 1], num_events).astype(np.int64)
    
    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name
    
    try:
        # Save events to HDF5
        evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, hdf5_path)
        
        # Load events back
        loaded_xs, loaded_ys, loaded_ts, loaded_ps = evlib.formats.load_events(hdf5_path)
        
        # Check that data matches exactly
        assert np.array_equal(xs, loaded_xs), "X coordinates do not match"
        assert np.array_equal(ys, loaded_ys), "Y coordinates do not match"
        assert np.allclose(ts, loaded_ts), "Timestamps do not match"
        assert np.array_equal(ps, loaded_ps), "Polarities do not match"
        
        # Check shapes are preserved
        assert xs.shape == loaded_xs.shape, "X shape mismatch"
        assert ys.shape == loaded_ys.shape, "Y shape mismatch"
        assert ts.shape == loaded_ts.shape, "Timestamp shape mismatch"
        assert ps.shape == loaded_ps.shape, "Polarity shape mismatch"
        
    finally:
        # Clean up
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)


def test_hdf5_save_load_empty_data():
    """Test HDF5 save/load with empty data"""
    # Create empty event data
    xs = np.array([], dtype=np.int64)
    ys = np.array([], dtype=np.int64)
    ts = np.array([], dtype=np.float64)
    ps = np.array([], dtype=np.int64)
    
    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name
    
    try:
        # Save empty events to HDF5
        evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, hdf5_path)
        
        # Load events back
        loaded_xs, loaded_ys, loaded_ts, loaded_ps = evlib.formats.load_events(hdf5_path)
        
        # Check that empty data is preserved
        assert len(loaded_xs) == 0, "Expected empty X coordinates"
        assert len(loaded_ys) == 0, "Expected empty Y coordinates"
        assert len(loaded_ts) == 0, "Expected empty timestamps"
        assert len(loaded_ps) == 0, "Expected empty polarities"
        
    finally:
        # Clean up
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)


def test_hdf5_save_load_boundary_values():
    """Test HDF5 save/load with boundary values"""
    # Create event data with boundary values
    xs = np.array([0, 65535], dtype=np.int64)  # uint16 boundaries
    ys = np.array([0, 65535], dtype=np.int64)  # uint16 boundaries
    ts = np.array([0.0, 1e12], dtype=np.float64)  # Large timestamp range
    ps = np.array([-1, 1], dtype=np.int64)  # Polarity boundaries
    
    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name
    
    try:
        # Save events to HDF5
        evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, hdf5_path)
        
        # Load events back
        loaded_xs, loaded_ys, loaded_ts, loaded_ps = evlib.formats.load_events(hdf5_path)
        
        # Check that boundary values are preserved
        assert np.array_equal(xs, loaded_xs), "Boundary X coordinates do not match"
        assert np.array_equal(ys, loaded_ys), "Boundary Y coordinates do not match"
        assert np.allclose(ts, loaded_ts), "Boundary timestamps do not match"
        assert np.array_equal(ps, loaded_ps), "Boundary polarities do not match"
        
    finally:
        # Clean up
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)


def test_hdf5_file_structure():
    """Test that HDF5 file structure is as expected"""
    # Create sample event data
    num_events = 10
    xs = np.random.randint(0, 100, num_events, dtype=np.int64)
    ys = np.random.randint(0, 100, num_events, dtype=np.int64)
    ts = np.sort(np.random.random(num_events).astype(np.float64))
    ps = np.random.choice([-1, 1], num_events).astype(np.int64)
    
    # Create temporary HDF5 file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name
    
    try:
        # Save events to HDF5
        evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, hdf5_path)
        
        # Check file structure using h5py
        import h5py
        with h5py.File(hdf5_path, 'r') as f:
            # Check that 'events' group exists
            assert 'events' in f, "Missing 'events' group in HDF5 file"
            events_group = f['events']
            
            # Check that all required datasets exist
            assert 'xs' in events_group, "Missing 'xs' dataset in events group"
            assert 'ys' in events_group, "Missing 'ys' dataset in events group"
            assert 'ts' in events_group, "Missing 'ts' dataset in events group"
            assert 'ps' in events_group, "Missing 'ps' dataset in events group"
            
            # Check data types
            assert events_group['xs'].dtype == 'uint16', "Incorrect xs dtype"
            assert events_group['ys'].dtype == 'uint16', "Incorrect ys dtype"
            assert events_group['ts'].dtype == 'float64', "Incorrect ts dtype"
            assert events_group['ps'].dtype == 'int8', "Incorrect ps dtype"
            
            # Check shapes
            assert events_group['xs'].shape == (num_events,), "Incorrect xs shape"
            assert events_group['ys'].shape == (num_events,), "Incorrect ys shape"
            assert events_group['ts'].shape == (num_events,), "Incorrect ts shape"
            assert events_group['ps'].shape == (num_events,), "Incorrect ps shape"
        
    finally:
        # Clean up
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)


def test_hdf5_load_various_formats():
    """Test that the load function can handle various HDF5 formats including legacy ones"""
    # Create sample event data
    num_events = 100
    xs = np.random.randint(0, 100, num_events, dtype=np.int64)
    ys = np.random.randint(0, 100, num_events, dtype=np.int64)
    ts = np.sort(np.random.random(num_events).astype(np.float64))
    ps = np.random.choice([-1, 1], num_events).astype(np.int64)
    
    # Test 1: Current format (events group with xs, ys, ts, ps datasets)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name
    
    try:
        # Save using our save function
        evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, hdf5_path)
        
        # Load back and verify
        loaded_xs, loaded_ys, loaded_ts, loaded_ps = evlib.formats.load_events(hdf5_path)
        assert np.array_equal(xs, loaded_xs), "Current format: X coordinates do not match"
        assert np.array_equal(ys, loaded_ys), "Current format: Y coordinates do not match"
        assert np.allclose(ts, loaded_ts), "Current format: Timestamps do not match"
        assert np.array_equal(ps, loaded_ps), "Current format: Polarities do not match"
        
    finally:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
    
    # Test 2: Legacy format (separate datasets in root with different names)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name
    
    try:
        # Create legacy format manually
        import h5py
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('t', data=ts)
            f.create_dataset('x', data=xs.astype(np.uint16))
            f.create_dataset('y', data=ys.astype(np.uint16))
            f.create_dataset('p', data=ps.astype(np.int8))
        
        # Load and verify
        loaded_xs, loaded_ys, loaded_ts, loaded_ps = evlib.formats.load_events(hdf5_path)
        assert np.array_equal(xs, loaded_xs), "Legacy format: X coordinates do not match"
        assert np.array_equal(ys, loaded_ys), "Legacy format: Y coordinates do not match"
        assert np.allclose(ts, loaded_ts), "Legacy format: Timestamps do not match"
        assert np.array_equal(ps, loaded_ps), "Legacy format: Polarities do not match"
        
    finally:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
    
    # Test 3: Alternative naming convention
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf5_path = tmp_file.name
    
    try:
        # Create alternative format manually
        import h5py
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('timestamps', data=ts)
            f.create_dataset('x_pos', data=xs.astype(np.uint16))
            f.create_dataset('y_pos', data=ys.astype(np.uint16))
            f.create_dataset('polarity', data=ps.astype(np.int8))
        
        # Load and verify
        loaded_xs, loaded_ys, loaded_ts, loaded_ps = evlib.formats.load_events(hdf5_path)
        assert np.array_equal(xs, loaded_xs), "Alternative format: X coordinates do not match"
        assert np.array_equal(ys, loaded_ys), "Alternative format: Y coordinates do not match"
        assert np.allclose(ts, loaded_ts), "Alternative format: Timestamps do not match"
        assert np.array_equal(ps, loaded_ps), "Alternative format: Polarities do not match"
        
    finally:
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)


if __name__ == "__main__":
    test_hdf5_save_load_roundtrip()
    test_hdf5_save_load_empty_data()
    test_hdf5_save_load_boundary_values()
    test_hdf5_file_structure()
    test_hdf5_load_various_formats()
    print("All HDF5 tests passed!")