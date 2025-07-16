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

import pytest
import numpy as np
import sys
from pathlib import Path
import time
import gc

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import evlib


class TestEvlibRegression:
    """Parameterized regression tests for evlib direct API usage."""

    @pytest.fixture(scope="class")
    def data_files(self):
        """Fixture providing test data file paths and expected characteristics."""
        data_dir = Path(__file__).parent.parent / "data"

        return {
            "evt2_small": {
                "path": data_dir / "eTram/raw/val_2/val_night_011.raw",
                "format": "EVT2",
                "resolution": (2048, 2000),  # Actual sensor resolution from data
                "expected_event_count": (2700000, 2800000),  # Actual: 2774556
                "polarity_encoding": (-1, 1),  # EVT2 correctly converts to -1/1
                "min_duration": 8000.0,  # Actual: 8612.691s
                "description": "Small EVT2 file (~15MB)",
            },
            "evt2_large": {
                "path": data_dir / "eTram/raw/val_2/val_night_007.raw",
                "format": "EVT2",
                "resolution": (2048, 2000),  # Same sensor as small file
                "expected_event_count": (20000000, 30000000),  # Approximate range
                "polarity_encoding": (-1, 1),
                "min_duration": 200.0,
                "description": "Large EVT2 file (~526MB)",
            },
            "hdf5_small": {
                "path": data_dir / "eTram/h5/val_2/val_night_011_td.h5",
                "format": "HDF5",
                "resolution": (1280, 720),  # Actual: 1279x719 max
                "expected_event_count": (3300000, 3500000),  # Actual: 3397511
                "polarity_encoding": (0, 1),  # HDF5 uses 0/1 encoding
                "min_duration": 5000000.0,  # Actual: 5091880.000s (microseconds?)
                "description": "Small HDF5 file (~14MB)",
            },
            "hdf5_large": {
                "path": data_dir / "eTram/h5/val_2/val_night_007_td.h5",
                "format": "HDF5",
                "resolution": (1280, 720),
                "expected_event_count": (25000000, 35000000),
                "polarity_encoding": (0, 1),
                "min_duration": 200.0,
                "description": "Large HDF5 file (~456MB)",
            },
            "text_medium": {
                "path": data_dir / "slider_depth/events.txt",
                "format": "Text",
                "resolution": (240, 180),  # Actual: 239x179 max
                "expected_event_count": (1070000, 1080000),  # Actual: 1078541
                "polarity_encoding": (0, 1),  # Text files use 0/1 encoding
                "min_duration": 3.0,  # Actual: 3.400s
                "description": "Text file (~22MB)",
            },
            "hdf5_xlarge": {
                "path": data_dir / "original/front/seq01.h5",
                "format": "HDF5",
                "resolution": (346, 240),
                "expected_event_count": (200000000, 300000000),
                "polarity_encoding": (0, 1),
                "min_duration": 20000.0,
                "description": "Extra large HDF5 file (~1.6GB)",
            },
            "hdf5_seq02": {
                "path": data_dir / "original/front/seq02.h5",
                "format": "HDF5",
                "resolution": (346, 240),
                "expected_event_count": (280000000, 290000000),
                "polarity_encoding": (0, 1),
                "min_duration": 20000.0,
                "description": "Extra large HDF5 file seq02",
            },
        }

    def test_file_existence(self, data_files):
        """Test that all expected data files exist."""
        missing_files = []
        for file_key, file_info in data_files.items():
            if not file_info["path"].exists():
                missing_files.append(f"{file_key}: {file_info['path']}")

        if missing_files:
            pytest.skip(f"Missing test files: {missing_files}")

    @pytest.mark.parametrize(
        "file_key",
        [
            "evt2_small",
            "evt2_large",
            "hdf5_small",
            "hdf5_large",
            "text_medium",
            "hdf5_xlarge",
            "hdf5_seq02",
        ],
    )
    def test_format_detection(self, data_files, file_key):
        """Test evlib.detect_format() with real data files."""
        file_info = data_files[file_key]

        if not file_info["path"].exists():
            pytest.skip(f"Test file not found: {file_info['path']}")

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

        print(f"✓ {file_key}: {format_name} (confidence: {confidence:.2f})")

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
            pytest.skip(f"Test file not found: {file_info['path']}")

        # Measure loading time
        start_time = time.time()
        result = evlib.load_events(str(file_info["path"]))
        load_time = time.time() - start_time

        # Verify result structure
        assert isinstance(result, tuple), f"load_events should return tuple, got {type(result)}"
        assert len(result) == 4, f"load_events should return 4-tuple, got {len(result)}"

        x, y, t, p = result

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
        assert (
            set(unique_polarities) == expected_polarity_values
        ), f"Expected polarities {expected_polarity_values}, got {set(unique_polarities)}"

        # Verify no invalid values
        assert not np.any(np.isnan(x)), "NaN x coordinates found"
        assert not np.any(np.isnan(y)), "NaN y coordinates found"
        assert not np.any(np.isnan(p)), "NaN polarities found"

        print(f"✓ {file_key}: {event_count:,} events, {duration:.1f}s duration, loaded in {load_time:.2f}s")

    @pytest.mark.parametrize(
        "file_key",
        [
            "evt2_large",
            "hdf5_large",
        ],
    )
    def test_load_events_large_files(self, data_files, file_key):
        """Test evlib.load_events() with large files (performance test)."""
        file_info = data_files[file_key]

        if not file_info["path"].exists():
            pytest.skip(f"Test file not found: {file_info['path']}")

        # Measure loading time and memory
        start_time = time.time()
        result = evlib.load_events(str(file_info["path"]))
        load_time = time.time() - start_time

        x, y, t, p = result
        event_count = len(x)

        # Performance assertions
        events_per_second = event_count / load_time
        assert events_per_second > 1000000, f"Loading too slow: {events_per_second:.0f} events/s"

        # Memory efficiency check (rough estimate)
        estimated_memory_per_event = 32  # bytes (conservative estimate)
        estimated_memory_mb = (event_count * estimated_memory_per_event) / (1024 * 1024)
        assert estimated_memory_mb < 5000, f"Estimated memory usage too high: {estimated_memory_mb:.1f}MB"

        # Basic validation
        assert len(x) > 0, "No events loaded"
        assert x.shape == y.shape == t.shape == p.shape, "Array shapes don't match"

        print(f"✓ {file_key}: {event_count:,} events in {load_time:.1f}s ({events_per_second:.0f} events/s)")

    @pytest.mark.parametrize(
        "file_key",
        [
            "text_medium",
            "hdf5_small",
        ],
    )
    def test_load_events_with_filtering(self, data_files, file_key):
        """Test evlib.load_events() with temporal and spatial filtering."""
        file_info = data_files[file_key]

        if not file_info["path"].exists():
            pytest.skip(f"Test file not found: {file_info['path']}")

        # Load full dataset first
        x_full, y_full, t_full, p_full = evlib.load_events(str(file_info["path"]))
        full_count = len(x_full)

        # Test temporal filtering
        t_min, t_max = np.min(t_full), np.max(t_full)
        t_range = t_max - t_min
        t_start = t_min + t_range * 0.3
        t_end = t_max - t_range * 0.3

        x_time, y_time, t_time, p_time = evlib.load_events(
            str(file_info["path"]), t_start=t_start, t_end=t_end
        )

        # Verify temporal filtering
        assert len(x_time) < full_count, "Temporal filtering didn't reduce event count"
        assert len(x_time) > 0, "Temporal filtering removed all events"
        assert np.all(t_time >= t_start), "Temporal filtering failed (start bound)"
        assert np.all(t_time <= t_end), "Temporal filtering failed (end bound)"

        # Test spatial filtering
        width, height = file_info["resolution"]
        x_center, y_center = width // 2, height // 2
        roi_size = min(width, height) // 4

        x_spatial, y_spatial, t_spatial, p_spatial = evlib.load_events(
            str(file_info["path"]),
            min_x=x_center - roi_size,
            max_x=x_center + roi_size,
            min_y=y_center - roi_size,
            max_y=y_center + roi_size,
        )

        # Verify spatial filtering
        assert len(x_spatial) < full_count, "Spatial filtering didn't reduce event count"
        assert len(x_spatial) > 0, "Spatial filtering removed all events"
        assert np.all(x_spatial >= x_center - roi_size), "Spatial filtering failed (x min)"
        assert np.all(x_spatial <= x_center + roi_size), "Spatial filtering failed (x max)"
        assert np.all(y_spatial >= y_center - roi_size), "Spatial filtering failed (y min)"
        assert np.all(y_spatial <= y_center + roi_size), "Spatial filtering failed (y max)"

        # Test polarity filtering
        x_pos, y_pos, t_pos, p_pos = evlib.load_events(str(file_info["path"]), polarity=1)

        # Verify polarity filtering
        assert len(x_pos) < full_count, "Polarity filtering didn't reduce event count"
        assert len(x_pos) > 0, "Polarity filtering removed all events"
        assert np.all(p_pos == 1), "Polarity filtering failed"

        print(
            f"✓ {file_key} filtering: full={full_count:,}, time={len(x_time):,}, spatial={len(x_spatial):,}, polarity={len(x_pos):,}"
        )

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
            pytest.skip(f"Test file not found: {file_info['path']}")

        x, y, t, p = evlib.load_events(str(file_info["path"]))

        # Test data types (should match your examples)
        assert x.dtype == np.int64, f"x dtype should be int64, got {x.dtype}"
        assert y.dtype == np.int64, f"y dtype should be int64, got {y.dtype}"
        assert t.dtype == np.float64, f"t dtype should be float64, got {t.dtype}"
        assert p.dtype == np.int64, f"p dtype should be int64, got {p.dtype}"

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

        print(f"✓ {file_key}: shapes={shape}, types=(int64, int64, float64, int64)")

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
        available_files = [f for f in test_files if data_files[f]["path"].exists()]

        if len(available_files) < 2:
            pytest.skip(f"Need at least 2 {format_name} files for consistency test")

        results = {}
        for file_key in available_files:
            file_info = data_files[file_key]
            x, y, t, p = evlib.load_events(str(file_info["path"]))

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

        print(f"✓ {format_name} consistency: {len(available_files)} files validated")

    def test_polarity_encoding_consistency(self, data_files):
        """Test that polarity values are consistent with expected encoding for each format."""
        for file_key, file_info in data_files.items():
            if not file_info["path"].exists():
                continue

            x, y, t, p = evlib.load_events(str(file_info["path"]))

            # Check against expected polarity encoding for this format
            unique_polarities = np.unique(p)
            expected_polarity_values = set(file_info["polarity_encoding"])
            assert (
                set(unique_polarities) == expected_polarity_values
            ), f"{file_key}: Expected polarities {expected_polarity_values}, got {set(unique_polarities)}"

            # Check distribution
            polarity_values = list(file_info["polarity_encoding"])
            pos_value, neg_value = max(polarity_values), min(polarity_values)

            pos_count = np.sum(p == pos_value)
            neg_count = np.sum(p == neg_value)
            total = len(p)

            assert pos_count + neg_count == total, f"{file_key}: Polarity counts don't sum to total"
            assert pos_count > 0, f"{file_key}: No positive polarity events"
            assert neg_count > 0, f"{file_key}: No negative polarity events"

            # Print distribution for debugging
            pos_ratio = pos_count / total
            print(f"✓ {file_key}: {pos_count:,} pos ({pos_ratio:.1%}), {neg_count:,} neg ({1-pos_ratio:.1%})")

    def test_evt21_format_support(self):
        """Test EVT2.1 format support if available."""
        # Check if we have any EVT2.1 files in the data directory
        data_dir = Path(__file__).parent.parent / "data"
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
                    x, y, t, p = evlib.load_events(str(file_path))

                    # Basic validation
                    assert len(x) > 0, "EVT2.1 file loaded no events"
                    assert x.shape == y.shape == t.shape == p.shape, "EVT2.1 array shapes don't match"
                    assert set(np.unique(p)) == {-1, 1}, "EVT2.1 polarity encoding incorrect"

                    print(f"✓ EVT2.1 support: {file_path.name} - {len(x):,} events")
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
            result = evlib.load_events(str(empty_file))
            x, y, t, p = result
            assert len(x) == 0, "Empty file should return empty arrays"
        finally:
            empty_file.unlink()

        print("✓ Error handling tests passed")

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
        x, y, t, p = evlib.load_events(str(file_info["path"]))
        del x, y, t, p

        # Force garbage collection
        gc.collect()

        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        # Allow for some increase but not too much
        assert object_increase < 1000, f"Too many objects created: {object_increase}"

        print(f"✓ Memory cleanup: {object_increase} objects remained after cleanup")


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
