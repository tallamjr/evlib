"""Test event representations functionality that actually exists."""

import numpy as np
import pytest
import evlib
import evlib.representations as evr
import tempfile
import os


def test_create_stacked_histogram():
    """Test create_stacked_histogram function with real data."""
    # Create sample event data using evlib's expected format
    x = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55], dtype=np.int64)
    # Use microsecond timestamps to avoid conversion issues
    t = np.array([100000, 200000, 300000, 400000, 500000], dtype=np.float64)
    p = np.array([1, -1, 1, -1, 1], dtype=np.int64)

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()  # Convert LazyFrame to DataFrame

            # The function expects 't' column, which is what we have
            import polars as pl

            # Ensure we have the expected column format
            assert "t" in events_df.columns, f"Expected 't' column, got: {events_df.columns}"

            # Test create_stacked_histogram_py - expects DataFrame, returns DataFrame
            hist_df = evr.create_stacked_histogram_py(events_df, 64, 64, nbins=5, window_duration_ms=100)

            # Validate output is DataFrame

            assert isinstance(hist_df, pl.DataFrame)

            # Should have columns: [window_id, channel, time_bin, y, x, count, channel_time_bin]
            expected_columns = ["window_id", "channel", "time_bin", "y", "x", "count", "channel_time_bin"]
            assert all(col in hist_df.columns for col in expected_columns)

            # Check data types (Rust uses efficient types)
            assert hist_df["window_id"].dtype == pl.Int64
            assert hist_df["channel"].dtype == pl.Int8  # Polarity is -1/1, fits in Int8
            assert hist_df["time_bin"].dtype == pl.Int16  # Time bins are small numbers, fits in Int16
            assert hist_df["y"].dtype == pl.Int16  # Coordinates fit in Int16
            assert hist_df["x"].dtype == pl.Int16  # Coordinates fit in Int16
            assert hist_df["count"].dtype == pl.UInt32

            print(f"Success: create_stacked_histogram returned DataFrame with {len(hist_df)} rows")

        except Exception as e:
            # Function exists but has implementation issues - should fail
            pytest.fail(f"create_stacked_histogram should work but failed: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_voxel_grid():
    """Test create_voxel_grid function with real data."""
    # Create sample event data with proper temporal spacing
    x = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55], dtype=np.int64)
    # Use microsecond timestamps to avoid NaN conversion issues
    t = np.array([100000, 200000, 300000, 400000, 500000], dtype=np.float64)
    p = np.array([1, -1, 1, -1, 1], dtype=np.int64)

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()  # Convert LazyFrame to DataFrame

            # The function expects 't' column, which is what we have
            import polars as pl

            # Ensure we have the expected column format
            assert "t" in events_df.columns, f"Expected 't' column, got: {events_df.columns}"

            # Test create_voxel_grid_py - expects DataFrame, returns DataFrame
            voxel_df = evr.create_voxel_grid_py(events_df, 64, 64, nbins=5)

            # Validate output is DataFrame

            assert isinstance(voxel_df, pl.DataFrame)

            # Should have columns: [time_bin, y, x, value]
            expected_columns = ["time_bin", "y", "x", "value"]
            assert all(col in voxel_df.columns for col in expected_columns)

            # Check data types (Rust uses efficient types)
            assert voxel_df["time_bin"].dtype == pl.Int16  # Time bins are small numbers, fits in Int16
            assert voxel_df["y"].dtype == pl.Int16  # Coordinates fit in Int16
            assert voxel_df["x"].dtype == pl.Int16  # Coordinates fit in Int16
            assert voxel_df["value"].dtype == pl.Int32  # Implementation returns Int32 for value

            print(f"Success: create_voxel_grid returned DataFrame with {len(voxel_df)} rows")

        except Exception as e:
            # Function exists but has implementation issues - should fail
            pytest.fail(f"create_voxel_grid should work but failed: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_mixed_density_stack():
    """Test create_mixed_density_stack function with real data."""
    # Create sample event data using evlib's expected format
    x = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55], dtype=np.int64)
    # Use microsecond timestamps to avoid conversion issues
    t = np.array([100000, 200000, 300000, 400000, 500000], dtype=np.float64)
    p = np.array([1, -1, 1, -1, 1], dtype=np.int64)

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()  # Convert LazyFrame to DataFrame

            # The function expects 't' column, which is what we have
            import polars as pl

            # Ensure we have the expected column format
            assert "t" in events_df.columns, f"Expected 't' column, got: {events_df.columns}"

            # Test create_mixed_density_stack_py - expects DataFrame, returns DataFrame
            mixed_df = evr.create_mixed_density_stack_py(events_df, 64, 64, nbins=10, window_duration_ms=50.0)

            # Validate output is DataFrame
            assert isinstance(mixed_df, pl.DataFrame)

            # Should have columns: [window_id, time_bin, y, x, polarity_sum]
            expected_columns = ["window_id", "time_bin", "y", "x", "polarity_sum"]
            assert all(col in mixed_df.columns for col in expected_columns)

            # Check data types
            assert mixed_df["window_id"].dtype == pl.Int64
            assert mixed_df["time_bin"].dtype == pl.Int16  # Time bins are small numbers, fits in Int16
            assert mixed_df["y"].dtype == pl.Int16  # Coordinates fit in Int16
            assert mixed_df["x"].dtype == pl.Int16  # Coordinates fit in Int16
            assert mixed_df["polarity_sum"].dtype == pl.Int64  # Sum of polarities

            print(f"Success: create_mixed_density_stack returned DataFrame with {len(mixed_df)} rows")

        except Exception as e:
            # Function exists but has implementation issues - should fail
            pytest.fail(f"create_mixed_density_stack should work but failed: {e}")
        finally:
            os.unlink(tmp.name)
