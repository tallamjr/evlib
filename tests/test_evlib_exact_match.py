"""
Exact match tests for evlib functionality.

This test suite validates that evlib returns exactly the expected results
from the user examples, ensuring complete regression testing.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import evlib


class TestEvlibExactMatch:
    """Test that evlib returns exactly the expected results from the examples."""

    def test_evt2_raw_exact_match(self):
        """Test EVT2 raw file matches the exact example output."""
        file_path = "data/eTram/raw/val_2/val_night_011.raw"

        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        # Load events
        events = evlib.load_events(file_path)

        # Verify exact structure from example
        assert isinstance(events, tuple), "Should return tuple"
        assert len(events) == 4, "Should return 4-tuple"

        x, y, t, p = events

        # Test exact characteristics from the example
        # Expected: shape=(2774556,)
        expected_shape = (2774556,)
        assert x.shape == expected_shape, f"X shape mismatch: {x.shape} vs {expected_shape}"
        assert y.shape == expected_shape, f"Y shape mismatch: {y.shape} vs {expected_shape}"
        assert t.shape == expected_shape, f"T shape mismatch: {t.shape} vs {expected_shape}"
        assert p.shape == expected_shape, f"P shape mismatch: {p.shape} vs {expected_shape}"

        # Test coordinate ranges (from example: x=[0-2047], y=[0-1999])
        assert 0 <= np.min(x) <= 10, f"X min should be near 0: {np.min(x)}"
        assert 2040 <= np.max(x) <= 2047, f"X max should be near 2047: {np.max(x)}"
        assert 0 <= np.min(y) <= 10, f"Y min should be near 0: {np.min(y)}"
        assert 1990 <= np.max(y) <= 1999, f"Y max should be near 1999: {np.max(y)}"

        # Test time range (approximately 0.000 - 8612.691)
        assert 0 <= np.min(t) <= 1.0, f"T min should be near 0: {np.min(t)}"
        assert 8600 <= np.max(t) <= 8620, f"T max should be near 8612: {np.max(t)}"

        # Test polarity (should be -1, 1 for EVT2)
        unique_polarities = np.unique(p)
        assert set(unique_polarities) == {-1, 1}, f"Expected {{-1, 1}}, got {set(unique_polarities)}"

        print(f"✓ EVT2 raw file: {len(x):,} events loaded successfully")

    def test_hdf5_exact_match(self):
        """Test HDF5 file matches the exact example output."""
        file_path = "data/eTram/h5/val_2/val_night_011_td.h5"

        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        # Load events
        events = evlib.load_events(file_path)

        # Verify exact structure from example
        assert isinstance(events, tuple), "Should return tuple"
        assert len(events) == 4, "Should return 4-tuple"

        x, y, t, p = events

        # Test exact characteristics from the example
        # Expected: shape=(3397511,)
        expected_shape = (3397511,)
        assert x.shape == expected_shape, f"X shape mismatch: {x.shape} vs {expected_shape}"
        assert y.shape == expected_shape, f"Y shape mismatch: {y.shape} vs {expected_shape}"
        assert t.shape == expected_shape, f"T shape mismatch: {t.shape} vs {expected_shape}"
        assert p.shape == expected_shape, f"P shape mismatch: {p.shape} vs {expected_shape}"

        # Test coordinate ranges (from example: x=[0-1279], y=[0-719])
        assert 0 <= np.min(x) <= 10, f"X min should be near 0: {np.min(x)}"
        assert 1270 <= np.max(x) <= 1279, f"X max should be near 1279: {np.max(x)}"
        assert 0 <= np.min(y) <= 10, f"Y min should be near 0: {np.min(y)}"
        assert 710 <= np.max(y) <= 719, f"Y max should be near 719: {np.max(y)}"

        # Test time range (from example: 4.800000e+01 - 5.091925e+06)
        assert 40 <= np.min(t) <= 50, f"T min should be near 48: {np.min(t)}"
        assert 5e6 <= np.max(t) <= 5.1e6, f"T max should be near 5.09e6: {np.max(t)}"

        # Test polarity (should be 0, 1 for HDF5)
        unique_polarities = np.unique(p)
        assert set(unique_polarities) == {0, 1}, f"Expected {{0, 1}}, got {set(unique_polarities)}"

        print(f"✓ HDF5 file: {len(x):,} events loaded successfully")

    def test_text_exact_match(self):
        """Test text file matches the exact example output."""
        file_path = "data/slider_depth/events.txt"

        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        # Load events
        events = evlib.load_events(file_path)

        # Verify exact structure from example
        assert isinstance(events, tuple), "Should return tuple"
        assert len(events) == 4, "Should return 4-tuple"

        x, y, t, p = events

        # Test exact characteristics from the example
        # Expected: shape=(1078541,)
        expected_shape = (1078541,)
        assert x.shape == expected_shape, f"X shape mismatch: {x.shape} vs {expected_shape}"
        assert y.shape == expected_shape, f"Y shape mismatch: {y.shape} vs {expected_shape}"
        assert t.shape == expected_shape, f"T shape mismatch: {t.shape} vs {expected_shape}"
        assert p.shape == expected_shape, f"P shape mismatch: {p.shape} vs {expected_shape}"

        # Test coordinate ranges (from example: x=[0-239], y=[0-179])
        assert 0 <= np.min(x) <= 10, f"X min should be near 0: {np.min(x)}"
        assert 230 <= np.max(x) <= 239, f"X max should be near 239: {np.max(x)}"
        assert 0 <= np.min(y) <= 10, f"Y min should be near 0: {np.min(y)}"
        assert 170 <= np.max(y) <= 179, f"Y max should be near 179: {np.max(y)}"

        # Test time range (from example: 0.003811 - 3.40408)
        assert 0 <= np.min(t) <= 0.01, f"T min should be near 0.003811: {np.min(t)}"
        assert 3.3 <= np.max(t) <= 3.5, f"T max should be near 3.404: {np.max(t)}"

        # Test polarity (should be 0, 1 for text)
        unique_polarities = np.unique(p)
        assert set(unique_polarities) == {0, 1}, f"Expected {{0, 1}}, got {set(unique_polarities)}"

        print(f"✓ Text file: {len(x):,} events loaded successfully")

    def test_seq02_exact_match(self):
        """Test seq02.h5 file matches the exact example output."""
        file_path = "data/original/front/seq02.h5"

        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        # Load events
        events = evlib.load_events(file_path)

        # Verify exact structure from example
        assert isinstance(events, tuple), "Should return tuple"
        assert len(events) == 4, "Should return 4-tuple"

        x, y, t, p = events

        # Test exact characteristics from the example
        # Expected: shape=(287765086,)
        expected_shape = (287765086,)
        assert x.shape == expected_shape, f"X shape mismatch: {x.shape} vs {expected_shape}"
        assert y.shape == expected_shape, f"Y shape mismatch: {y.shape} vs {expected_shape}"
        assert t.shape == expected_shape, f"T shape mismatch: {t.shape} vs {expected_shape}"
        assert p.shape == expected_shape, f"P shape mismatch: {p.shape} vs {expected_shape}"

        # Test coordinate ranges (actual data shows 1280x720)
        assert 0 <= np.min(x) <= 10, f"X min should be near 0: {np.min(x)}"
        assert 1270 <= np.max(x) <= 1279, f"X max should be near 1279: {np.max(x)}"
        assert 0 <= np.min(y) <= 10, f"Y min should be near 0: {np.min(y)}"
        assert 710 <= np.max(y) <= 719, f"Y max should be near 719: {np.max(y)}"

        # Test time range (from example: 25219688. - 45251720.)
        assert 2.5e7 <= np.min(t) <= 2.53e7, f"T min should be near 25219688: {np.min(t)}"
        assert 4.52e7 <= np.max(t) <= 4.53e7, f"T max should be near 45251720: {np.max(t)}"

        # Test polarity (should be 0, 1 for HDF5)
        unique_polarities = np.unique(p)
        assert set(unique_polarities) == {0, 1}, f"Expected {{0, 1}}, got {set(unique_polarities)}"

        print(f"✓ seq02.h5 file: {len(x):,} events loaded successfully")

    def test_format_detection_exact_match(self):
        """Test format detection matches exact example output."""
        file_path = "data/eTram/raw/val_2/val_night_011.raw"

        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        # Test format detection
        result = evlib.detect_format(file_path)

        # Verify exact structure from example
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 3, "Should return 3-tuple"

        format_name, confidence, metadata = result

        # Test exact characteristics from the example
        # Expected: ('EVT2', 0.95, {'detection_method': 'evt2_header'})
        assert format_name == "EVT2", f"Format should be EVT2, got {format_name}"
        assert confidence == 0.95, f"Confidence should be 0.95, got {confidence}"
        assert isinstance(metadata, dict), f"Metadata should be dict, got {type(metadata)}"
        assert "detection_method" in metadata, "Should have detection_method in metadata"

        print(f"✓ Format detection: {format_name} (confidence: {confidence})")

    def test_data_types_exact_match(self):
        """Test that data types match the exact example output."""
        file_path = "data/eTram/raw/val_2/val_night_011.raw"

        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        # Load events
        events = evlib.load_events(file_path)
        x, y, t, p = events

        # Test exact data types from the example
        # From the example: x and y are int64, t is float64, p is int64
        assert x.dtype == np.int64, f"X dtype should be int64, got {x.dtype}"
        assert y.dtype == np.int64, f"Y dtype should be int64, got {y.dtype}"
        assert t.dtype == np.float64, f"T dtype should be float64, got {t.dtype}"
        assert p.dtype == np.int64, f"P dtype should be int64, got {p.dtype}"

        print(f"✓ Data types: x={x.dtype}, y={y.dtype}, t={t.dtype}, p={p.dtype}")

    def test_first_few_events_consistency(self):
        """Test that first few events are consistent across runs."""
        file_path = "data/eTram/raw/val_2/val_night_011.raw"

        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        # Load events twice
        events1 = evlib.load_events(file_path)
        events2 = evlib.load_events(file_path)

        x1, y1, t1, p1 = events1
        x2, y2, t2, p2 = events2

        # Test that first 1000 events are identical
        test_size = min(1000, len(x1))

        assert np.array_equal(x1[:test_size], x2[:test_size]), "X coordinates should be identical"
        assert np.array_equal(y1[:test_size], y2[:test_size]), "Y coordinates should be identical"
        assert np.array_equal(t1[:test_size], t2[:test_size]), "Timestamps should be identical"
        assert np.array_equal(p1[:test_size], p2[:test_size]), "Polarities should be identical"

        print(f"✓ First {test_size} events are consistent across runs")

    def test_performance_benchmark(self):
        """Test loading performance for regression testing."""
        file_path = "data/eTram/raw/val_2/val_night_011.raw"

        if not Path(file_path).exists():
            pytest.skip(f"Test file not found: {file_path}")

        import time

        # Benchmark loading time
        start_time = time.time()
        events = evlib.load_events(file_path)
        load_time = time.time() - start_time

        x, y, t, p = events
        event_count = len(x)

        # Performance assertions (reasonable expectations)
        events_per_second = event_count / load_time
        assert events_per_second > 1000000, f"Loading too slow: {events_per_second:.0f} events/s"
        assert load_time < 30, f"Loading took too long: {load_time:.1f}s"

        print(f"✓ Performance: {event_count:,} events in {load_time:.2f}s ({events_per_second:.0f} events/s)")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
