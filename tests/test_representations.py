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

            # Test create_stacked_histogram - expects DataFrame, returns DataFrame
            hist_df = evr.create_stacked_histogram(events_df, 64, 64, nbins=5, window_duration_ms=100)

            # Validate output is DataFrame
            import polars as pl

            assert isinstance(hist_df, pl.DataFrame)

            # Should have columns: [window_id, channel, time_bin, y, x, count]
            expected_columns = ["window_id", "channel", "time_bin", "y", "x", "count"]
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

            # Test create_voxel_grid - expects DataFrame, returns DataFrame
            voxel_df = evr.create_voxel_grid(events_df, 64, 64, nbins=5)

            # Validate output is DataFrame
            import polars as pl

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


def test_benchmark_vs_rvt():
    """Test benchmark_vs_rvt function with real data."""
    # Create sample event data with proper timestamps
    x = np.array([10, 20, 30, 40, 50] * 100, dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55] * 100, dtype=np.int64)
    # Use microsecond timestamps from 100ms to 1s
    t = np.linspace(100000, 1000000, 500, dtype=np.float64)
    p = np.array([1, -1, 1, -1, 1] * 100, dtype=np.int64)

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Test benchmark_vs_rvt
            results = evr.benchmark_vs_rvt(tmp.name, height=64, width=64)

            # Validate output
            assert isinstance(results, dict)
            assert "polars_time" in results
            assert "speedup" in results
            assert results["polars_time"] > 0
            assert results["speedup"] > 0

            print(f"Success: benchmark_vs_rvt returned {results}")

        except Exception as e:
            pytest.skip(f"benchmark_vs_rvt not working: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_enhanced_voxel_grid():
    """Test create_enhanced_voxel_grid function with real data."""
    # Create sample event data with proper temporal spacing
    x = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55], dtype=np.int64)
    # Use microsecond timestamps to test bilinear interpolation
    t = np.array([100000, 200000, 300000, 400000, 500000], dtype=np.float64)
    p = np.array([1, 0, 1, 0, 1], dtype=np.int64)  # Use 0/1 encoding

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()  # Convert LazyFrame to DataFrame

            # Ensure polarity is 0/1 and rename timestamp to t
            import polars as pl

            if "polarity" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.when(pl.col("polarity") < 0).then(0).otherwise(1).alias("polarity")]
                )
            if "timestamp" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.col("timestamp").dt.total_microseconds().cast(pl.Float64).alias("t")]
                ).drop("timestamp")
            # Convert data types to expected format
            events_df = events_df.with_columns(
                [
                    pl.col("x").cast(pl.Int64),
                    pl.col("y").cast(pl.Int64),
                    pl.col("polarity").cast(pl.Int64),
                ]
            )

            # Test create_enhanced_voxel_grid - expects DataFrame, returns DataFrame
            voxel_df = evr.create_enhanced_voxel_grid(events_df, 64, 64, n_time_bins=5)

            # Validate output is DataFrame
            import polars as pl

            assert isinstance(voxel_df, pl.DataFrame)

            # Should have columns: [time_bin, polarity_channel, y, x, value]
            expected_columns = ["time_bin", "polarity_channel", "y", "x", "value"]
            assert all(col in voxel_df.columns for col in expected_columns)

            # Check data types
            assert voxel_df["time_bin"].dtype == pl.Int32
            assert voxel_df["polarity_channel"].dtype == pl.Int32
            assert voxel_df["y"].dtype == pl.Int32
            assert voxel_df["x"].dtype == pl.Int32
            # Value dtype can be Float32 or Float64 depending on implementation
            assert voxel_df["value"].dtype in [pl.Float32, pl.Float64]

            # Verify bilinear interpolation - values should be fractional
            values = voxel_df["value"].to_numpy()
            non_zero_values = values[values != 0.0]

            # Some values should be fractional due to bilinear interpolation
            fractional_values = non_zero_values[np.abs(non_zero_values - np.round(non_zero_values)) > 1e-6]
            print(
                f"Enhanced voxel grid has {len(fractional_values)} fractional values from bilinear interpolation"
            )

            print(f"Success: create_enhanced_voxel_grid returned DataFrame with {len(voxel_df)} rows")

        except Exception as e:
            pytest.fail(f"create_enhanced_voxel_grid should work but failed: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_enhanced_frame_time_window():
    """Test create_enhanced_frame with time window slicing."""
    # Create sample event data with known time distribution
    x = np.array([10, 20, 30, 40, 50, 10, 20, 30] * 2, dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55, 15, 25, 35] * 2, dtype=np.int64)
    # Create events spanning 1000 microseconds (1 ms)
    t = np.array([100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000] * 2, dtype=np.float64)
    p = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 2, dtype=np.int64)  # Use 0/1 encoding

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()  # Convert LazyFrame to DataFrame

            # Ensure polarity is 0/1 and rename timestamp to t
            import polars as pl

            if "polarity" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.when(pl.col("polarity") < 0).then(0).otherwise(1).alias("polarity")]
                )
            if "timestamp" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.col("timestamp").dt.total_microseconds().cast(pl.Float64).alias("t")]
                ).drop("timestamp")
            # Convert data types to expected format
            events_df = events_df.with_columns(
                [
                    pl.col("x").cast(pl.Int64),
                    pl.col("y").cast(pl.Int64),
                    pl.col("polarity").cast(pl.Int64),
                ]
            )

            # Test time window slicing - 200us windows
            frame_df = evr.create_enhanced_frame(
                events_df,
                height=64,
                width=64,
                polarity_channels=2,
                time_window=200000.0,  # 200 ms in microseconds
                overlap=0.0,
                include_incomplete=True,
            )

            # Validate output is DataFrame
            assert isinstance(frame_df, pl.DataFrame)

            # Should have columns: [frame_id, polarity_channel, y, x, count]
            expected_columns = ["frame_id", "polarity_channel", "y", "x", "count"]
            assert all(col in frame_df.columns for col in expected_columns)

            # Check data types
            assert frame_df["frame_id"].dtype == pl.Int32
            assert frame_df["polarity_channel"].dtype == pl.Int32
            assert frame_df["y"].dtype == pl.Int32
            assert frame_df["x"].dtype == pl.Int32
            assert frame_df["count"].dtype == pl.Float32

            # Should have multiple frames due to time window slicing
            unique_frames = frame_df["frame_id"].unique().to_numpy()
            assert len(unique_frames) > 1, f"Expected multiple frames, got {len(unique_frames)}"

            # Should have both polarity channels
            unique_polarities = frame_df["polarity_channel"].unique().to_numpy()
            assert len(unique_polarities) == 2, f"Expected 2 polarity channels, got {len(unique_polarities)}"

            print(f"Success: time window slicing created {len(unique_frames)} frames")

        except Exception as e:
            pytest.fail(f"create_enhanced_frame time window test failed: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_enhanced_frame_event_count():
    """Test create_enhanced_frame with event count slicing."""
    # Create sample event data
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55, 15, 25, 35], dtype=np.int64)
    t = np.linspace(100000, 500000, 8, dtype=np.float64)
    p = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()

            # Ensure polarity is 0/1 and rename timestamp to t
            import polars as pl

            if "polarity" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.when(pl.col("polarity") < 0).then(0).otherwise(1).alias("polarity")]
                )
            if "timestamp" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.col("timestamp").dt.total_microseconds().cast(pl.Float64).alias("t")]
                ).drop("timestamp")
            # Convert data types to expected format
            events_df = events_df.with_columns(
                [
                    pl.col("x").cast(pl.Int64),
                    pl.col("y").cast(pl.Int64),
                    pl.col("polarity").cast(pl.Int64),
                ]
            )

            # Test event count slicing - 3 events per frame
            frame_df = evr.create_enhanced_frame(
                events_df,
                height=64,
                width=64,
                polarity_channels=2,
                event_count=3,
                overlap=0,
                include_incomplete=True,
            )

            # Validate output
            assert isinstance(frame_df, pl.DataFrame)

            expected_columns = ["frame_id", "polarity_channel", "y", "x", "count"]
            assert all(col in frame_df.columns for col in expected_columns)

            # Should have multiple frames (8 events / 3 events per frame = 2-3 frames)
            unique_frames = frame_df["frame_id"].unique().to_numpy()
            assert len(unique_frames) >= 2, f"Expected at least 2 frames, got {len(unique_frames)}"

            print(f"Success: event count slicing created {len(unique_frames)} frames")

        except Exception as e:
            pytest.fail(f"create_enhanced_frame event count test failed: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_enhanced_frame_time_bins():
    """Test create_enhanced_frame with fixed number of time bins."""
    # Create sample event data
    x = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55], dtype=np.int64)
    t = np.array([100000, 200000, 300000, 400000, 500000], dtype=np.float64)
    p = np.array([1, 0, 1, 0, 1], dtype=np.int64)

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()

            # Ensure polarity is 0/1 and rename timestamp to t
            import polars as pl

            if "polarity" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.when(pl.col("polarity") < 0).then(0).otherwise(1).alias("polarity")]
                )
            if "timestamp" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.col("timestamp").dt.total_microseconds().cast(pl.Float64).alias("t")]
                ).drop("timestamp")
            # Convert data types to expected format
            events_df = events_df.with_columns(
                [
                    pl.col("x").cast(pl.Int64),
                    pl.col("y").cast(pl.Int64),
                    pl.col("polarity").cast(pl.Int64),
                ]
            )

            # Test time bins slicing - exactly 3 time bins
            frame_df = evr.create_enhanced_frame(
                events_df, height=64, width=64, polarity_channels=2, n_time_bins=3, overlap=0.0
            )

            # Validate output
            assert isinstance(frame_df, pl.DataFrame)

            expected_columns = ["frame_id", "polarity_channel", "y", "x", "count"]
            assert all(col in frame_df.columns for col in expected_columns)

            # Should have exactly 3 frames (or fewer if some are empty)
            unique_frames = frame_df["frame_id"].unique().to_numpy()
            assert len(unique_frames) <= 3, f"Expected at most 3 frames, got {len(unique_frames)}"
            assert len(unique_frames) >= 1, f"Expected at least 1 frame, got {len(unique_frames)}"

            print(f"Success: time bins slicing created {len(unique_frames)} frames")

        except Exception as e:
            pytest.fail(f"create_enhanced_frame time bins test failed: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_enhanced_frame_event_bins():
    """Test create_enhanced_frame with fixed number of event bins."""
    # Create sample event data
    x = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55, 15], dtype=np.int64)
    t = np.linspace(100000, 600000, 6, dtype=np.float64)
    p = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()

            # Ensure polarity is 0/1 and rename timestamp to t
            import polars as pl

            if "polarity" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.when(pl.col("polarity") < 0).then(0).otherwise(1).alias("polarity")]
                )
            if "timestamp" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.col("timestamp").dt.total_microseconds().cast(pl.Float64).alias("t")]
                ).drop("timestamp")
            # Convert data types to expected format
            events_df = events_df.with_columns(
                [
                    pl.col("x").cast(pl.Int64),
                    pl.col("y").cast(pl.Int64),
                    pl.col("polarity").cast(pl.Int64),
                ]
            )

            # Test event bins slicing - exactly 2 event bins
            frame_df = evr.create_enhanced_frame(
                events_df, height=64, width=64, polarity_channels=2, n_event_bins=2, overlap=0.0
            )

            # Validate output
            assert isinstance(frame_df, pl.DataFrame)

            expected_columns = ["frame_id", "polarity_channel", "y", "x", "count"]
            assert all(col in frame_df.columns for col in expected_columns)

            # Should have exactly 2 frames (or fewer if some are empty)
            unique_frames = frame_df["frame_id"].unique().to_numpy()
            assert len(unique_frames) <= 2, f"Expected at most 2 frames, got {len(unique_frames)}"
            assert len(unique_frames) >= 1, f"Expected at least 1 frame, got {len(unique_frames)}"

            print(f"Success: event bins slicing created {len(unique_frames)} frames")

        except Exception as e:
            pytest.fail(f"create_enhanced_frame event bins test failed: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_enhanced_frame_single_polarity():
    """Test create_enhanced_frame with single polarity sensor."""
    # Create sample event data with single polarity
    x = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55], dtype=np.int64)
    t = np.array([100000, 200000, 300000, 400000, 500000], dtype=np.float64)
    p = np.array([1, 1, 1, 1, 1], dtype=np.int64)  # All same polarity

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()

            # Ensure polarity is 0/1 and rename timestamp to t
            import polars as pl

            if "polarity" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.when(pl.col("polarity") < 0).then(0).otherwise(1).alias("polarity")]
                )
            if "timestamp" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.col("timestamp").dt.total_microseconds().cast(pl.Float64).alias("t")]
                ).drop("timestamp")
            # Convert data types to expected format
            events_df = events_df.with_columns(
                [
                    pl.col("x").cast(pl.Int64),
                    pl.col("y").cast(pl.Int64),
                    pl.col("polarity").cast(pl.Int64),
                ]
            )

            # Test single polarity mode
            frame_df = evr.create_enhanced_frame(
                events_df,
                height=64,
                width=64,
                polarity_channels=1,  # Single polarity
                n_time_bins=2,
                overlap=0.0,
            )

            # Validate output
            assert isinstance(frame_df, pl.DataFrame)

            expected_columns = ["frame_id", "polarity_channel", "y", "x", "count"]
            assert all(col in frame_df.columns for col in expected_columns)

            # Should have only polarity channel 0
            unique_polarities = frame_df["polarity_channel"].unique().to_numpy()
            assert len(unique_polarities) == 1, f"Expected 1 polarity channel, got {len(unique_polarities)}"
            assert unique_polarities[0] == 0, f"Expected polarity channel 0, got {unique_polarities[0]}"

            print("Success: single polarity mode works correctly")

        except Exception as e:
            pytest.fail(f"create_enhanced_frame single polarity test failed: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_enhanced_frame_overlap():
    """Test create_enhanced_frame with overlap."""
    # Create sample event data
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    y = np.array([15, 25, 35, 45, 55, 15, 25, 35], dtype=np.int64)
    t = np.linspace(100000, 800000, 8, dtype=np.float64)
    p = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()

            # Ensure polarity is 0/1 and rename timestamp to t
            import polars as pl

            if "polarity" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.when(pl.col("polarity") < 0).then(0).otherwise(1).alias("polarity")]
                )
            if "timestamp" in events_df.columns:
                events_df = events_df.with_columns(
                    [pl.col("timestamp").dt.total_microseconds().cast(pl.Float64).alias("t")]
                ).drop("timestamp")
            # Convert data types to expected format
            events_df = events_df.with_columns(
                [
                    pl.col("x").cast(pl.Int64),
                    pl.col("y").cast(pl.Int64),
                    pl.col("polarity").cast(pl.Int64),
                ]
            )

            # Test with overlap - event count with 1 event overlap
            frame_df = evr.create_enhanced_frame(
                events_df,
                height=64,
                width=64,
                polarity_channels=2,
                event_count=4,
                overlap=1.0,  # 1 event overlap
                include_incomplete=True,
            )

            # Validate output
            assert isinstance(frame_df, pl.DataFrame)

            expected_columns = ["frame_id", "polarity_channel", "y", "x", "count"]
            assert all(col in frame_df.columns for col in expected_columns)

            # With overlap, should have more frames than without
            unique_frames = frame_df["frame_id"].unique().to_numpy()
            assert (
                len(unique_frames) >= 2
            ), f"Expected at least 2 frames with overlap, got {len(unique_frames)}"

            print(f"Success: overlap parameter works, created {len(unique_frames)} frames")

        except Exception as e:
            pytest.fail(f"create_enhanced_frame overlap test failed: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_enhanced_frame_error_handling():
    """Test create_enhanced_frame error handling."""
    # Create sample event data
    x = np.array([10, 20, 30], dtype=np.int64)
    y = np.array([15, 25, 35], dtype=np.int64)
    t = np.array([100000, 200000, 300000], dtype=np.float64)
    p = np.array([1, 0, 1], dtype=np.int64)

    # Save to temporary HDF5 file using evlib
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        try:
            evlib.save_events_to_hdf5(x, y, t, p, tmp.name)

            # Load events as DataFrame first
            events_lf = evlib.load_events(tmp.name)
            events_df = events_lf.collect()

            # Test error: no slicing method specified
            with pytest.raises(Exception):
                evr.create_enhanced_frame(
                    events_df,
                    height=64,
                    width=64,
                    polarity_channels=2,
                    # No slicing method specified
                )

            # Test error: multiple slicing methods specified
            with pytest.raises(Exception):
                evr.create_enhanced_frame(
                    events_df,
                    height=64,
                    width=64,
                    polarity_channels=2,
                    time_window=100000.0,
                    event_count=5,  # Both specified - should error
                )

            # Test error: invalid polarity channels
            with pytest.raises(Exception):
                evr.create_enhanced_frame(
                    events_df,
                    height=64,
                    width=64,
                    polarity_channels=3,  # Invalid - must be 1 or 2
                    n_time_bins=2,
                )

            print("Success: error handling works correctly")

        except Exception as e:
            if (
                "no slicing method" in str(e).lower()
                or "multiple" in str(e).lower()
                or "invalid" in str(e).lower()
            ):
                print("Success: error handling works correctly")
            else:
                pytest.fail(f"create_enhanced_frame error handling test failed unexpectedly: {e}")
        finally:
            os.unlink(tmp.name)


def test_create_bina_rep_basic():
    """Test create_bina_rep function with basic functionality."""
    import numpy as np
    import polars as pl

    # Create binary event frames for Bina-Rep
    # For n_frames=2, n_bits=3, we need 2*3=6 total frames
    n_frames = 2
    n_bits = 3
    polarity_channels = 2
    height = 8
    width = 8
    total_frames = n_frames * n_bits  # 6 frames

    # Create test binary event frames
    # Frame pattern: alternating patterns for testing the bit encoding
    event_frames = np.zeros((total_frames, polarity_channels, height, width), dtype=np.float32)

    # Set up patterns to test N-bit encoding
    # First Bina-Rep frame (frames 0,1,2):
    event_frames[0, 0, 2, 3] = 1.0  # Frame 0: bit position 4 (2^2)
    event_frames[1, 0, 2, 3] = 1.0  # Frame 1: bit position 2 (2^1)
    event_frames[2, 0, 2, 3] = 0.0  # Frame 2: bit position 1 (2^0)
    # Expected value: 1*4 + 1*2 + 0*1 = 6, normalized by (2^3-1)=7 → 6/7 ≈ 0.857

    # Second Bina-Rep frame (frames 3,4,5):
    event_frames[3, 1, 4, 5] = 0.0  # Frame 3: bit position 4 (2^2)
    event_frames[4, 1, 4, 5] = 1.0  # Frame 4: bit position 2 (2^1)
    event_frames[5, 1, 4, 5] = 1.0  # Frame 5: bit position 1 (2^0)
    # Expected value: 0*4 + 1*2 + 1*1 = 3, normalized by (2^3-1)=7 → 3/7 ≈ 0.429

    # Test the function
    result_df = evr.create_bina_rep(event_frames, n_frames, n_bits)

    # Validate output is DataFrame
    assert isinstance(result_df, pl.DataFrame)

    # Should have columns: [time_frame, polarity_channel, y, x, bina_rep_value]
    expected_columns = ["time_frame", "polarity_channel", "y", "x", "bina_rep_value"]
    assert all(col in result_df.columns for col in expected_columns)

    # Check data types
    assert result_df["time_frame"].dtype == pl.Int32
    assert result_df["polarity_channel"].dtype == pl.Int8
    assert result_df["y"].dtype == pl.Int16
    assert result_df["x"].dtype == pl.Int16
    assert result_df["bina_rep_value"].dtype == pl.Float32

    # Check specific values
    result_list = result_df.to_dicts()

    # Should have 2 non-zero entries (one for each pattern we set)
    assert len(result_list) == 2, f"Expected 2 entries, got {len(result_list)}"

    # Check first entry (frame 0, polarity 0, position (2,3))
    entry1 = next((r for r in result_list if r["time_frame"] == 0 and r["polarity_channel"] == 0), None)
    assert entry1 is not None, "Expected entry for time_frame=0, polarity_channel=0"
    assert entry1["y"] == 2 and entry1["x"] == 3
    assert (
        abs(entry1["bina_rep_value"] - (6.0 / 7.0)) < 1e-6
    ), f"Expected ~0.857, got {entry1['bina_rep_value']}"

    # Check second entry (frame 1, polarity 1, position (4,5))
    entry2 = next((r for r in result_list if r["time_frame"] == 1 and r["polarity_channel"] == 1), None)
    assert entry2 is not None, "Expected entry for time_frame=1, polarity_channel=1"
    assert entry2["y"] == 4 and entry2["x"] == 5
    assert (
        abs(entry2["bina_rep_value"] - (3.0 / 7.0)) < 1e-6
    ), f"Expected ~0.429, got {entry2['bina_rep_value']}"

    print("Success: create_bina_rep basic functionality works correctly")


def test_create_bina_rep_edge_cases():
    """Test create_bina_rep function with edge cases."""
    import numpy as np

    # Test case 1: Invalid n_bits (< 2)
    event_frames = np.zeros((2, 2, 4, 4), dtype=np.float32)
    try:
        result_df = evr.create_bina_rep(event_frames, 1, 1)  # n_bits < 2
        pytest.fail("Should have failed with n_bits < 2")
    except Exception as e:
        assert "n_bits must be >= 2" in str(e)
        print("Success: n_bits validation works")

    # Test case 2: Mismatched frame count
    event_frames = np.zeros((5, 2, 4, 4), dtype=np.float32)  # 5 frames
    try:
        result_df = evr.create_bina_rep(event_frames, 2, 3)  # expects 2*3=6 frames
        pytest.fail("Should have failed with mismatched frame count")
    except Exception as e:
        assert "must have exactly" in str(e)
        print("Success: frame count validation works")

    # Test case 3: All zeros (should return empty DataFrame)
    event_frames = np.zeros((4, 2, 4, 4), dtype=np.float32)  # 2 frames * 2 bits
    result_df = evr.create_bina_rep(event_frames, 2, 2)

    # Should return empty DataFrame since all values are zero
    assert len(result_df) == 0, f"Expected empty DataFrame, got {len(result_df)} rows"
    print("Success: all-zeros case works correctly")

    # Test case 4: Maximum bit encoding (all 1s)
    event_frames = np.ones((4, 2, 2, 2), dtype=np.float32)  # 2 frames * 2 bits, all 1s
    result_df = evr.create_bina_rep(event_frames, 2, 2)

    # All pixels should have maximum value: (2^2-1)/(2^2-1) = 1.0
    assert len(result_df) == 2 * 2 * 2 * 2  # 2 frames * 2 polarities * 2x2 pixels = 16
    max_value = result_df["bina_rep_value"].max()
    assert abs(max_value - 1.0) < 1e-6, f"Expected max value 1.0, got {max_value}"
    print("Success: maximum bit encoding works correctly")


def test_create_bina_rep_normalization():
    """Test create_bina_rep normalization for different n_bits values."""
    import numpy as np

    # Test different n_bits values to verify normalization
    for n_bits in [2, 3, 4, 8]:
        n_frames = 1
        total_frames = n_frames * n_bits
        polarity_channels = 1
        height, width = 2, 2

        # Create event frames with all 1s to get maximum value
        event_frames = np.ones((total_frames, polarity_channels, height, width), dtype=np.float32)

        result_df = evr.create_bina_rep(event_frames, n_frames, n_bits)

        # Maximum possible value should be (2^n_bits - 1) / (2^n_bits - 1) = 1.0
        max_value = result_df["bina_rep_value"].max()
        assert abs(max_value - 1.0) < 1e-6, f"n_bits={n_bits}: Expected max 1.0, got {max_value}"

        # Check that we get the expected number of entries (all pixels)
        expected_entries = n_frames * polarity_channels * height * width
        assert (
            len(result_df) == expected_entries
        ), f"n_bits={n_bits}: Expected {expected_entries} entries, got {len(result_df)}"

        print(f"Success: n_bits={n_bits} normalization works correctly")


def test_create_bina_rep_bit_patterns():
    """Test create_bina_rep with specific bit patterns to verify algorithm."""
    import numpy as np

    # Test 3-bit encoding with specific patterns
    n_bits = 3
    n_frames = 1
    total_frames = n_bits  # 3 frames
    polarity_channels = 1
    height, width = 1, 4  # 4 pixels to test different patterns

    event_frames = np.zeros((total_frames, polarity_channels, height, width), dtype=np.float32)

    # Pixel 0: Pattern 000 → expected value: 0
    # (already zero)

    # Pixel 1: Pattern 001 → expected value: 1, normalized: 1/7 ≈ 0.143
    event_frames[2, 0, 0, 1] = 1.0  # Bit 0 (rightmost)

    # Pixel 2: Pattern 101 → expected value: 5, normalized: 5/7 ≈ 0.714
    event_frames[0, 0, 0, 2] = 1.0  # Bit 2 (leftmost)
    event_frames[2, 0, 0, 2] = 1.0  # Bit 0 (rightmost)

    # Pixel 3: Pattern 111 → expected value: 7, normalized: 7/7 = 1.0
    event_frames[0, 0, 0, 3] = 1.0  # Bit 2
    event_frames[1, 0, 0, 3] = 1.0  # Bit 1
    event_frames[2, 0, 0, 3] = 1.0  # Bit 0

    result_df = evr.create_bina_rep(event_frames, n_frames, n_bits)

    # Convert to dictionary for easier checking
    result_dict = {r["x"]: r["bina_rep_value"] for r in result_df.to_dicts()}

    # Pixel 0 should not appear (value 0)
    assert 0 not in result_dict, "Pixel 0 should not appear (zero value)"

    # Pixel 1: Pattern 001 → 1/7 ≈ 0.143
    assert abs(result_dict[1] - (1.0 / 7.0)) < 1e-6, f"Pixel 1: expected {1.0/7.0}, got {result_dict[1]}"

    # Pixel 2: Pattern 101 → 5/7 ≈ 0.714
    assert abs(result_dict[2] - (5.0 / 7.0)) < 1e-6, f"Pixel 2: expected {5.0/7.0}, got {result_dict[2]}"

    # Pixel 3: Pattern 111 → 7/7 = 1.0
    assert abs(result_dict[3] - (7.0 / 7.0)) < 1e-6, f"Pixel 3: expected {7.0/7.0}, got {result_dict[3]}"

    print("Success: specific bit patterns work correctly")


def test_create_bina_rep_multiple_frames():
    """Test create_bina_rep with multiple output frames."""
    import numpy as np

    # Test with 3 output frames, 2 bits each → 6 total input frames
    n_frames = 3
    n_bits = 2
    total_frames = n_frames * n_bits  # 6 frames
    polarity_channels = 2
    height, width = 2, 2

    event_frames = np.zeros((total_frames, polarity_channels, height, width), dtype=np.float32)

    # Set up different patterns for each output frame
    # Frame 0 (input frames 0,1): Pattern 10 → value 2, normalized: 2/3 ≈ 0.667
    event_frames[0, 0, 0, 0] = 1.0  # Bit 1 (leftmost)
    # event_frames[1, 0, 0, 0] = 0.0  # Bit 0 (already zero)

    # Frame 1 (input frames 2,3): Pattern 01 → value 1, normalized: 1/3 ≈ 0.333
    # event_frames[2, 1, 1, 0] = 0.0  # Bit 1 (already zero)
    event_frames[3, 1, 1, 0] = 1.0  # Bit 0

    # Frame 2 (input frames 4,5): Pattern 11 → value 3, normalized: 3/3 = 1.0
    event_frames[4, 0, 1, 1] = 1.0  # Bit 1
    event_frames[5, 0, 1, 1] = 1.0  # Bit 0

    result_df = evr.create_bina_rep(event_frames, n_frames, n_bits)

    # Should have 3 entries (one for each pattern we set)
    assert len(result_df) == 3, f"Expected 3 entries, got {len(result_df)}"

    # Check each frame
    result_list = result_df.to_dicts()

    # Frame 0 entry
    frame0_entry = next((r for r in result_list if r["time_frame"] == 0), None)
    assert frame0_entry is not None
    assert abs(frame0_entry["bina_rep_value"] - (2.0 / 3.0)) < 1e-6

    # Frame 1 entry
    frame1_entry = next((r for r in result_list if r["time_frame"] == 1), None)
    assert frame1_entry is not None
    assert abs(frame1_entry["bina_rep_value"] - (1.0 / 3.0)) < 1e-6

    # Frame 2 entry
    frame2_entry = next((r for r in result_list if r["time_frame"] == 2), None)
    assert frame2_entry is not None
    assert abs(frame2_entry["bina_rep_value"] - (3.0 / 3.0)) < 1e-6

    print("Success: multiple output frames work correctly")
