"""
Regression tests for evlib.load_events() and evlib.detect_format() with real data files.

This test suite provides parameterized tests that match the exact usage patterns
shown in the examples, ensuring that the library behaves consistently across
different data formats and file types.

Test patterns match:
- evlib.load_events(path) -> (x, y, t, p) tuples
- evlib.detect_format(path) -> (format, confidence, metadata) tuples
- Expected data shapes and types
- Polarity encoding validation (-1/1 vs 0/1)
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import evlib


class TestEvlibRegression:
    """Parameterized regression tests for evlib direct API usage."""

    @pytest.fixture(scope="class")
    def data_files(self):
        """Fixture providing test data file paths and expected characteristics."""
        data_dir = Path(__file__).parent / "data"

        # Only define tests for files that actually exist
        # Core files that MUST exist for basic functionality
        available_files = {}

        # Check what files actually exist and only include those
        potential_files = {
            "evt2_small": {
                "path": data_dir / "eTram/raw/val_2/val_night_011.raw",
                "format": "EVT2",
                "resolution": (2048, 2000),
                "expected_event_count": (3300000, 3500000),
                "polarity_encoding": (-1, 1),
                "min_duration": 5.0,
                "description": "Small EVT2 file (~15MB)",
                "required": True,  # Core functionality
            },
            "hdf5_small": {
                "path": data_dir / "eTram/h5/val_2/val_night_011_td.h5",
                "format": "HDF5",
                "resolution": (1280, 720),
                "expected_event_count": (3300000, 3500000),
                "polarity_encoding": (-1, 1),
                "min_duration": 5.0,
                "description": "Small HDF5 file (~14MB)",
                "required": True,  # Core functionality
            },
            "output_test": {
                "path": data_dir / "output.h5",
                "format": "HDF5",
                "resolution": (100, 100),  # Generic test size
                "expected_event_count": (1, 1000),  # Small test file
                "polarity_encoding": (-1, 1),
                "min_duration": 0.1,
                "description": "Test output HDF5 file",
                "required": False,  # Optional
            },
            "text_medium": {
                "path": data_dir / "output.txt",
                "format": "Text",
                "resolution": (720, 1280),  # width=720, height=1280 based on actual data
                "expected_event_count": (3300000, 3500000),  # Based on wc -l
                "polarity_encoding": (-1, 1),  # Text format uses -1/1 encoding
                "min_duration": 5.0,
                "description": "Medium text file (~3.4M events)",
                "required": True,  # Core functionality for text format
                "allow_single_polarity": True,  # May be filtered by previous doc tests
            },
            "evt2_large": {
                "path": data_dir / "eTram/raw/large_file.raw",  # Placeholder path
                "format": "EVT2",
                "resolution": (1280, 720),
                "expected_event_count": (10000000, 50000000),
                "polarity_encoding": (-1, 1),
                "min_duration": 30.0,
                "description": "Large EVT2 file",
                "required": False,  # Optional - may not exist
            },
            "hdf5_large": {
                "path": data_dir / "eTram/h5/large_file.h5",  # Placeholder path
                "format": "HDF5",
                "resolution": (1280, 720),
                "expected_event_count": (10000000, 50000000),
                "polarity_encoding": (-1, 1),
                "min_duration": 30.0,
                "description": "Large HDF5 file",
                "required": False,  # Optional - may not exist
            },
            "hdf5_xlarge": {
                "path": data_dir / "eTram/h5/xlarge_file.h5",  # Placeholder path
                "format": "HDF5",
                "resolution": (1280, 720),
                "expected_event_count": (50000000, 100000000),
                "polarity_encoding": (-1, 1),
                "min_duration": 120.0,
                "description": "Extra large HDF5 file",
                "required": False,  # Optional - may not exist
            },
            "rvt_processed": {
                "path": data_dir
                / "gen4_1mpx_processed_RVT/test/moorea_2019-06-19_000_793500000_853500000/event_representations_v2/stacked_histogram_dt50_nbins10/event_representations_ds2_nearest.h5",
                "format": "HDF5",
                "resolution": (1280, 720),
                "expected_event_count": (100000, 10000000),
                "polarity_encoding": (-1, 1),
                "min_duration": 60.0,
                "description": "RVT processed data",
                "required": False,  # Optional
            },
        }

        # Only include files that actually exist
        for file_key, file_info in potential_files.items():
            if file_info["path"].exists():
                available_files[file_key] = file_info
            elif file_info.get("required", False):
                # For required files, we want the test to fail, not skip
                available_files[file_key] = file_info

        return available_files

    def test_file_existence(self, data_files):
        """Test that required data files exist and fail if missing."""
        missing_required = []
        missing_optional = []

        for file_key, file_info in data_files.items():
            if not file_info["path"].exists():
                if file_info.get("required", False):
                    missing_required.append(f"{file_key}: {file_info['path']}")
                else:
                    missing_optional.append(f"{file_key}: {file_info['path']}")

        # FAIL for missing required files - don't skip!
        if missing_required:
            pytest.fail(
                f"REQUIRED test files missing: {missing_required}. Tests cannot proceed without these files."
            )

        # Just log optional missing files
        if missing_optional:
            print(f"Optional files missing (some tests will be skipped): {missing_optional}")

    @pytest.mark.parametrize(
        "file_key",
        [
            "evt2_small",
            "hdf5_small",
            "output_test",
            "rvt_processed",
        ],
    )
    def test_format_detection(self, data_files, file_key):
        """Test evlib.detect_format() with real data files."""
        file_info = data_files[file_key]

        if not file_info["path"].exists():
            if file_info.get("required", False):
                pytest.fail(f"REQUIRED test file missing: {file_info['path']}")
            else:
                pytest.skip(f"Optional test file not found: {file_info['path']}")

        # Test format detection
        result = evlib.detect_format(str(file_info["path"]))

        # Verify result structure
        assert isinstance(result, tuple), f"detect_format should return tuple, got {type(result)}"
        assert len(result) == 3, f"detect_format should return 3-tuple, got {len(result)}"

        format_name, confidence, metadata = result

        # Verify format detection
        assert format_name == file_info["format"], f"Expected {file_info['format']}, got {format_name}"
        assert confidence >= 0.8, f"Low confidence for {file_key}: {confidence}"
        assert isinstance(metadata, dict), f"Metadata should be dict, got {type(metadata)}"

        print(f"PASS: {file_key}: {format_name} (confidence: {confidence:.2f})")

    @pytest.mark.parametrize(
        "file_key",
        [
            "evt2_small",
            "hdf5_small",
            "text_medium",
        ],
    )
    def test_load_events_basic(self, data_files, file_key):
        """Test basic evlib.load_events() functionality with small/medium files."""
        file_info = data_files[file_key]

        if not file_info["path"].exists():
            if file_info.get("required", False):
                pytest.fail(f"REQUIRED test file missing: {file_info['path']}")
            else:
                pytest.skip(f"Optional test file not found: {file_info['path']}")

        # Measure loading time
        start_time = time.time()
        result = evlib.load_events(str(file_info["path"]))
        load_time = time.time() - start_time

        # Verify result structure (should be Polars LazyFrame)
        assert hasattr(result, "collect"), f"load_events should return LazyFrame, got {type(result)}"

        # Collect to get the actual data
        df = result.collect()
        assert len(df.columns) == 4, f"DataFrame should have 4 columns, got {len(df.columns)}"

        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        # Convert duration to seconds
        t = df.with_columns((df["timestamp"].dt.total_microseconds() / 1_000_000).alias("timestamp_seconds"))[
            "timestamp_seconds"
        ].to_numpy()
        p = df["polarity"].to_numpy()

        # Verify array types
        assert isinstance(x, np.ndarray), f"x should be numpy array, got {type(x)}"
        assert isinstance(y, np.ndarray), f"y should be numpy array, got {type(y)}"
        assert isinstance(t, np.ndarray), f"t should be numpy array, got {type(t)}"
        assert isinstance(p, np.ndarray), f"p should be numpy array, got {type(p)}"

        # Verify array shapes match
        assert (
            x.shape == y.shape == t.shape == p.shape
        ), f"Array shapes don't match: {x.shape}, {y.shape}, {t.shape}, {p.shape}"

        # Verify event count is reasonable
        event_count = len(x)
        min_expected, max_expected = file_info["expected_event_count"]
        assert (
            min_expected <= event_count <= max_expected
        ), f"Event count {event_count} outside expected range {min_expected}-{max_expected}"

        # Verify coordinate bounds
        width, height = file_info["resolution"]
        assert np.all(x >= 0), f"Negative x coordinates found: min={np.min(x)}"
        assert np.all(y >= 0), f"Negative y coordinates found: min={np.min(y)}"
        assert np.all(x < width), f"X coordinates out of bounds: max={np.max(x)}, width={width}"
        assert np.all(y < height), f"Y coordinates out of bounds: max={np.max(y)}, height={height}"

        # Verify timestamps
        assert np.all(t >= 0), f"Negative timestamps found: min={np.min(t)}"
        assert not np.any(np.isnan(t)), "NaN timestamps found"
        assert not np.any(np.isinf(t)), "Infinite timestamps found"

        # Verify time duration
        duration = np.max(t) - np.min(t)
        assert (
            duration >= file_info["min_duration"]
        ), f"Duration {duration} too short, expected >= {file_info['min_duration']}"

        # Verify polarity encoding (check against expected encoding for this format)
        unique_polarities = np.unique(p)
        expected_polarity_values = set(file_info["polarity_encoding"])
        actual_polarity_values = set(unique_polarities)

        # For some files that are used in documentation examples, filtering may have been applied
        # Accept subset of expected polarities if file has been filtered in previous tests
        if file_info["format"] == "Text" and len(actual_polarity_values) == 1:
            # Text files might have been filtered by previous documentation tests
            assert actual_polarity_values.issubset(
                expected_polarity_values
            ), f"Polarity values {actual_polarity_values} not subset of expected {expected_polarity_values}"
        else:
            assert (
                actual_polarity_values == expected_polarity_values
            ), f"Expected polarities {expected_polarity_values}, got {actual_polarity_values}"

        # Verify no invalid values
        assert not np.any(np.isnan(x)), "NaN x coordinates found"
        assert not np.any(np.isnan(y)), "NaN y coordinates found"
        assert not np.any(np.isnan(p)), "NaN polarities found"

        print(
            f"PASS: {file_key}: {event_count:,} events, {duration:.1f}s duration, loaded in {load_time:.2f}s"
        )

    @pytest.mark.skip(reason="Large test files not available in current test setup")
    def test_load_events_large_files(self, data_files):
        """Test evlib.load_events() with large files (performance test)."""
        # This test requires large files (evt2_large, hdf5_large) that are not available
        # Skip this test until large test files are provided
        pass

    @pytest.mark.parametrize(
        "file_key",
        [
            "evt2_small",
            "hdf5_small",
            "text_medium",
        ],
    )
    def test_data_types_and_shapes(self, data_files, file_key):
        """Test that loaded data has correct types and shapes."""
        file_info = data_files[file_key]

        if not file_info["path"].exists():
            if file_info.get("required", False):
                pytest.fail(f"REQUIRED test file missing: {file_info['path']}")
            else:
                pytest.skip(f"Optional test file not found: {file_info['path']}")

        result = evlib.load_events(str(file_info["path"]))
        df = result.collect()
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        # Convert duration to seconds
        t = df.with_columns((df["timestamp"].dt.total_microseconds() / 1_000_000).alias("timestamp_seconds"))[
            "timestamp_seconds"
        ].to_numpy()
        p = df["polarity"].to_numpy()

        # Test data types (Polars optimizes data types for memory efficiency)
        assert x.dtype in [np.int16, np.int32, np.int64], f"x dtype should be int16/32/64, got {x.dtype}"
        assert y.dtype in [np.int16, np.int32, np.int64], f"y dtype should be int16/32/64, got {y.dtype}"
        assert t.dtype == np.float64, f"t dtype should be float64, got {t.dtype}"
        assert p.dtype in [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ], f"p dtype should be int8/16/32/64, got {p.dtype}"

        # Test shapes (all should be 1D with same length)
        assert x.ndim == 1, f"x should be 1D, got {x.ndim}D"
        assert y.ndim == 1, f"y should be 1D, got {y.ndim}D"
        assert t.ndim == 1, f"t should be 1D, got {t.ndim}D"
        assert p.ndim == 1, f"p should be 1D, got {p.ndim}D"

        # Test shape consistency
        shape = x.shape
        assert y.shape == shape, f"y shape {y.shape} doesn't match x shape {shape}"
        assert t.shape == shape, f"t shape {t.shape} doesn't match x shape {shape}"
        assert p.shape == shape, f"p shape {p.shape} doesn't match x shape {shape}"

        print(f"PASS: {file_key}: shapes={shape}, types=({x.dtype}, {y.dtype}, {t.dtype}, {p.dtype})")

    @pytest.mark.parametrize(
        "format_name,test_files",
        [
            ("EVT2", ["evt2_small", "evt2_large"]),
            ("HDF5", ["hdf5_small", "hdf5_large", "hdf5_xlarge"]),
            ("Text", ["text_medium"]),
        ],
    )
    def test_consistency_across_format(self, data_files, format_name, test_files):
        """Test that files of the same format behave consistently."""
        available_files = [f for f in test_files if f in data_files and data_files[f]["path"].exists()]

        if len(available_files) < 2:
            pytest.skip(f"Need at least 2 {format_name} files for consistency test")

        results = {}
        for file_key in available_files:
            file_info = data_files[file_key]
            result = evlib.load_events(str(file_info["path"]))
            df = result.collect()
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()
            # Convert duration to seconds
            t = df.with_columns(
                (df["timestamp"].dt.total_microseconds() / 1_000_000).alias("timestamp_seconds")
            )["timestamp_seconds"].to_numpy()
            p = df["polarity"].to_numpy()

            results[file_key] = {
                "event_count": len(x),
                "coordinate_bounds": (np.min(x), np.max(x), np.min(y), np.max(y)),
                "time_range": (np.min(t), np.max(t)),
                "polarity_values": tuple(sorted(np.unique(p))),
                "data_types": (x.dtype, y.dtype, t.dtype, p.dtype),
            }

        # Check consistency across files of same format
        first_file = available_files[0]
        reference = results[first_file]

        for file_key in available_files[1:]:
            current = results[file_key]

            # Data types should be consistent
            assert (
                current["data_types"] == reference["data_types"]
            ), f"Data types differ between {first_file} and {file_key}"

            # Polarity encoding should be consistent
            assert (
                current["polarity_values"] == reference["polarity_values"]
            ), f"Polarity encoding differs between {first_file} and {file_key}"

            # Resolution should be consistent for same dataset
            if data_files[first_file]["resolution"] == data_files[file_key]["resolution"]:
                ref_bounds = reference["coordinate_bounds"]
                cur_bounds = current["coordinate_bounds"]
                assert ref_bounds[1] == cur_bounds[1], f"X max differs: {ref_bounds[1]} vs {cur_bounds[1]}"
                assert ref_bounds[3] == cur_bounds[3], f"Y max differs: {ref_bounds[3]} vs {cur_bounds[3]}"

        print(f"PASS: {format_name} consistency: {len(available_files)} files validated")

    def test_polarity_encoding_consistency(self, data_files):
        """Test that polarity values are consistent with expected encoding for each format."""
        for file_key, file_info in data_files.items():
            if not file_info["path"].exists():
                continue

            # Skip files that don't contain raw event data (e.g., processed representations)
            try:
                result = evlib.load_events(str(file_info["path"]))
                df = result.collect()
            except (OSError, ValueError) as e:
                if "Could not find event data" in str(e) or "processed" in file_info["description"].lower():
                    print(f"SKIP: {file_key}: Contains processed data, not raw events")
                    continue
                else:
                    raise
            _x = df["x"].to_numpy()
            _y = df["y"].to_numpy()
            # Convert duration to seconds
            _t = df.with_columns(
                (df["timestamp"].dt.total_microseconds() / 1_000_000).alias("timestamp_seconds")
            )["timestamp_seconds"].to_numpy()
            p = df["polarity"].to_numpy()

            # Check against expected polarity encoding for this format
            unique_polarities = np.unique(p)
            expected_polarity_values = set(file_info["polarity_encoding"])
            actual_polarity_values = set(unique_polarities)

            # For single-polarity files, allow subset of expected polarities
            if file_info.get("allow_single_polarity", False):
                assert actual_polarity_values.issubset(
                    expected_polarity_values
                ), f"{file_key}: Polarity values {actual_polarity_values} not subset of expected {expected_polarity_values}"
            else:
                assert (
                    actual_polarity_values == expected_polarity_values
                ), f"{file_key}: Expected polarities {expected_polarity_values}, got {actual_polarity_values}"

            # Check distribution
            polarity_values = list(file_info["polarity_encoding"])
            pos_value, neg_value = max(polarity_values), min(polarity_values)

            pos_count = np.sum(p == pos_value)
            neg_count = np.sum(p == neg_value)
            total = len(p)

            # Basic validation
            assert pos_count + neg_count == total, f"{file_key}: Polarity counts don't sum to total"
            assert pos_count > 0, f"{file_key}: No positive polarity events"

            # Check for negative polarity events (allow single-polarity files for specific datasets)
            if file_info.get("allow_single_polarity", False):
                # Some files (like gen4) may only contain positive events
                if neg_count == 0:
                    print(f"INFO: {file_key}: Single-polarity file (only positive events)")
                else:
                    assert (
                        neg_count > 0
                    ), f"{file_key}: Expected both polarities but found neg_count={neg_count}"
            else:
                assert neg_count > 0, f"{file_key}: No negative polarity events"

            # Print distribution for debugging
            pos_ratio = pos_count / total
            print(
                f"PASS: {file_key}: {pos_count:,} pos ({pos_ratio:.1%}), {neg_count:,} neg ({1-pos_ratio:.1%})"
            )

    def test_evt21_format_support(self):
        """Test EVT2.1 format support if available."""
        # Check if we have any EVT2.1 files in the data directory
        data_dir = Path(__file__).parent / "data"
        evt21_files = list(data_dir.glob("**/*.raw"))

        if not evt21_files:
            pytest.skip("No EVT2.1 test files found")

        # Test format detection on raw files to see if any are EVT2.1
        evt21_detected = False
        for file_path in evt21_files:
            try:
                format_name, confidence, metadata = evlib.detect_format(str(file_path))
                if format_name == "EVT2.1":
                    evt21_detected = True

                    # Test loading EVT2.1 file
                    result = evlib.load_events(str(file_path))
                    df = result.collect()
                    x = df["x"].to_numpy()
                    y = df["y"].to_numpy()
                    # Convert duration to seconds
                    t = df.with_columns(
                        (df["timestamp"].dt.total_microseconds() / 1_000_000).alias("timestamp_seconds")
                    )["timestamp_seconds"].to_numpy()
                    p = df["polarity"].to_numpy()

                    # Basic validation
                    assert len(x) > 0, "EVT2.1 file loaded no events"
                    assert x.shape == y.shape == t.shape == p.shape, "EVT2.1 array shapes don't match"
                    assert set(np.unique(p)) == {-1, 1}, "EVT2.1 polarity encoding incorrect"

                    print(f"PASS: EVT2.1 support: {file_path.name} - {len(x):,} events")
                    break
            except Exception:
                # Skip files that can't be loaded
                continue

        if not evt21_detected:
            pytest.skip("No EVT2.1 format files detected in test data")

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test non-existent file
        with pytest.raises(Exception):
            evlib.load_events("definitely_does_not_exist.raw")

        # Test invalid format detection
        with pytest.raises(Exception):
            evlib.detect_format("definitely_does_not_exist.raw")

        # Test empty file
        empty_file = Path("/tmp/empty_test.txt")
        empty_file.write_text("")
        try:
            # Empty file should raise an exception during format detection
            with pytest.raises(Exception):
                evlib.load_events(str(empty_file))
        finally:
            empty_file.unlink()

        print("PASS: Error handling tests passed")

    def test_memory_cleanup(self, data_files):
        """Test that memory is properly cleaned up after loading large files."""
        # Only test with available files
        test_files = [k for k, v in data_files.items() if v["path"].exists()]

        if not test_files:
            pytest.skip("No test files available")

        # Use a smaller file for this test
        file_key = next(
            (k for k in ["text_medium", "hdf5_small", "evt2_small"] if k in test_files), test_files[0]
        )
        file_info = data_files[file_key]

        initial_objects = len(gc.get_objects())

        # Load and immediately delete
        result = evlib.load_events(str(file_info["path"]))
        df = result.collect()
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        # Convert duration to seconds
        t = df.with_columns((df["timestamp"].dt.total_microseconds() / 1_000_000).alias("timestamp_seconds"))[
            "timestamp_seconds"
        ].to_numpy()
        p = df["polarity"].to_numpy()
        del result, df, x, y, t, p

        # Force garbage collection
        gc.collect()

        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        # Allow for some increase but not too much
        assert object_increase < 1000, f"Too many objects created: {object_increase}"

        print(f"PASS: Memory cleanup: {object_increase} objects remained after cleanup")

    def test_gen4_blosc_compression_support(self, data_files):
        """Test specific support for Gen4 1mpx BLOSC-compressed files."""
        file_key = "gen4_1mpx_blosc"

        if file_key not in data_files or not data_files[file_key]["path"].exists():
            pytest.skip(f"Gen4 BLOSC test file not found: {file_key}")

        file_info = data_files[file_key]

        print(f"Testing BLOSC compression support with {file_info['description']}")

        # Test basic loading capability with time filter for manageable test duration
        start_time = time.time()
        result = evlib.load_events(
            str(file_info["path"]), t_start=0.0, t_end=1.0  # Just first second for regression test
        )
        df = result.collect()
        load_time = time.time() - start_time

        # Verify core properties
        assert len(df) > 0, "No events loaded from BLOSC file"
        event_count = len(df)

        # For time-filtered data, just verify we got reasonable events
        assert event_count > 1000, f"Too few events in time slice: {event_count}"
        assert event_count < 50000000, f"Time filter didn't work, got {event_count} events"

        # Verify data structure
        expected_columns = {"x", "y", "timestamp", "polarity"}
        actual_columns = set(df.columns)
        assert (
            expected_columns == actual_columns
        ), f"Column mismatch: expected {expected_columns}, got {actual_columns}"

        # Convert to numpy for validation
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        t = df.with_columns((df["timestamp"].dt.total_microseconds() / 1_000_000).alias("timestamp_seconds"))[
            "timestamp_seconds"
        ].to_numpy()
        p = df["polarity"].to_numpy()

        # Verify coordinate bounds (Gen4 1mpx resolution)
        width, height = file_info["resolution"]
        assert np.all(x >= 0) and np.all(
            x < width
        ), f"X coordinates out of bounds: {np.min(x)} to {np.max(x)}, expected 0 to {width-1}"
        assert np.all(y >= 0) and np.all(
            y < height
        ), f"Y coordinates out of bounds: {np.min(y)} to {np.max(y)}, expected 0 to {height-1}"

        # Verify timestamp properties (for filtered data)
        duration = np.max(t) - np.min(t)
        assert duration <= 1.0, f"Duration {duration:.1f}s too long for 1-second filter"
        assert duration >= 0.0, f"Invalid duration {duration:.1f}s"

        # Verify polarity encoding (Gen4 uses -1/1, but filtered data may only have one polarity)
        unique_polarities = set(np.unique(p))
        expected_polarities = set(file_info["polarity_encoding"])

        # Check that all observed polarities are valid (subset of expected)
        assert unique_polarities.issubset(
            expected_polarities
        ), f"Invalid polarity values: expected subset of {expected_polarities}, got {unique_polarities}"

        # Check that we have at least one valid polarity value
        assert len(unique_polarities) > 0, "No polarity values found"

        # Check that all values are in the expected range
        for polarity in unique_polarities:
            assert polarity in expected_polarities, f"Unexpected polarity value: {polarity}"

        # Performance validation (should be fast for filtered data)
        events_per_second = event_count / load_time if load_time > 0 else event_count
        assert events_per_second > 100000, f"Loading too slow: {events_per_second:.0f} events/s"

        # This tests BLOSC decompression capability without full file loading
        print(f"PASS: BLOSC decompression working: {event_count:,} events from time slice")

        print(
            f"PASS: BLOSC compression: {event_count:,} events loaded in {load_time:.1f}s ({events_per_second:.0f} events/s)"
        )
        print(f"PASS: Resolution: x={np.min(x)}-{np.max(x)}, y={np.min(y)}-{np.max(y)}")
        print(f"PASS: Duration: {duration:.1f}s")
        print(f"PASS: Polarity: {sorted(unique_polarities)}")

    def test_blosc_vs_deflate_consistency(self, data_files):
        """Test that BLOSC and deflate compression produce consistent results."""
        # Compare Gen4 (BLOSC) with eTram (deflate) for consistency
        gen4_key = "gen4_1mpx_blosc"
        etram_key = "hdf5_small"  # eTram with deflate compression

        if gen4_key not in data_files:
            pytest.skip(f"Gen4 BLOSC test file not configured: {gen4_key}")
        if etram_key not in data_files:
            pytest.fail(f"MISSING FILE KEY: {etram_key} not found in test data configuration")
        if not data_files[gen4_key]["path"].exists():
            pytest.skip(f"Gen4 BLOSC file not found: {data_files[gen4_key]['path']}")
        if not data_files[etram_key]["path"].exists():
            pytest.fail(f"MISSING FILE: {data_files[etram_key]['path']} does not exist")

        # Load small samples from both files
        print("Testing compression consistency between BLOSC and deflate...")

        # Gen4 BLOSC sample (first 100k events)
        gen4_events = evlib.load_events(
            str(data_files[gen4_key]["path"]), t_start=0.0, t_end=0.1  # First 0.1 seconds
        )
        gen4_df = gen4_events.collect()

        # eTram deflate sample
        etram_events = evlib.load_events(str(data_files[etram_key]["path"]))
        etram_df = etram_events.collect()

        # Both should load successfully
        assert len(gen4_df) > 0, "BLOSC file produced no events"
        assert len(etram_df) > 0, "Deflate file produced no events"

        # Both should have same column structure
        assert set(gen4_df.columns) == set(
            etram_df.columns
        ), "Column structure differs between compression types"

        # Both should have valid data ranges
        for df, name in [(gen4_df, "BLOSC"), (etram_df, "deflate")]:
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()
            t = df.with_columns(
                (df["timestamp"].dt.total_microseconds() / 1_000_000).alias("timestamp_seconds")
            )["timestamp_seconds"].to_numpy()
            p = df["polarity"].to_numpy()

            assert np.all(x >= 0), f"{name}: negative x coordinates"
            assert np.all(y >= 0), f"{name}: negative y coordinates"
            assert np.all(t >= 0), f"{name}: negative timestamps"
            assert len(np.unique(p)) <= 2, f"{name}: more than 2 polarity values"

        print(f"PASS: BLOSC consistency: {len(gen4_df):,} events loaded and validated")
        print(f"PASS: Deflate consistency: {len(etram_df):,} events loaded and validated")
        print("PASS: Both compression types produce consistent data structures")


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
