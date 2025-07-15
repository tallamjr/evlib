"""
Comprehensive test suite for validating format readers against real data files.

This test suite validates:
1. EVT2/EVT3 readers against eTram data
2. HDF5 readers against eTram H5 data
3. Text format readers against slider_depth data
4. Format detection accuracy
5. Performance and memory usage
6. Data integrity and consistency
"""

import os
import sys
import time
import pytest
import numpy as np
from pathlib import Path
import psutil
import gc
from typing import Dict, List, Tuple, Optional

# Add the project root to the path so we can import evlib
sys.path.insert(0, str(Path(__file__).parent.parent))

import evlib


class TestRealDataFormats:
    """Test suite for real data format validation."""

    def setup_method(self):
        """Setup method called before each test."""
        self.data_dir = Path(__file__).parent.parent / "data"
        self.results = {}
        
        # Memory tracking
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Test files
        self.test_files = {
            "evt2_small": self.data_dir / "eTram/raw/val_2/val_night_011.raw",  # 15MB
            "evt2_large": self.data_dir / "eTram/raw/val_2/val_night_007.raw",  # 526MB
            "hdf5_small": self.data_dir / "eTram/h5/val_2/val_night_011_td.h5",  # 14MB
            "hdf5_large": self.data_dir / "eTram/h5/val_2/val_night_007_td.h5",  # 456MB
            "text_medium": self.data_dir / "slider_depth/events.txt",  # 22MB
            "hdf5_original": self.data_dir / "original/front/seq01.h5",  # 1.6GB
        }
        
        # Expected sensor resolutions
        self.expected_resolutions = {
            "evt2_small": (1280, 720),
            "evt2_large": (1280, 720),
            "hdf5_small": (1280, 720),
            "hdf5_large": (1280, 720),
            "text_medium": (346, 240),  # DAVIS sensor
            "hdf5_original": (346, 240),
        }

    def teardown_method(self):
        """Cleanup after each test."""
        # Force garbage collection
        gc.collect()
        
        # Check for memory leaks
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.initial_memory
        
        if memory_increase > 100:  # More than 100MB increase
            print(f"WARNING: Memory increased by {memory_increase:.1f}MB during test")

    def measure_performance(self, func, *args, **kwargs) -> Tuple[any, float, float]:
        """Measure function performance and memory usage."""
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        return result, execution_time, memory_usage

    def validate_events(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, p: np.ndarray, 
                       file_key: str) -> Dict[str, any]:
        """Validate event data integrity."""
        results = {
            "event_count": len(x),
            "coordinate_bounds": {
                "x_min": int(np.min(x)),
                "x_max": int(np.max(x)),
                "y_min": int(np.min(y)),
                "y_max": int(np.max(y)),
            },
            "time_range": {
                "t_start": float(np.min(t)),
                "t_end": float(np.max(t)),
                "duration": float(np.max(t) - np.min(t)),
            },
            "polarity_distribution": {
                "positive": int(np.sum(p == 1)),
                "negative": int(np.sum(p == -1)),
                "total": len(p),
            },
            "data_integrity": {
                "has_nan": bool(np.any(np.isnan(t))),
                "has_inf": bool(np.any(np.isinf(t))),
                "sorted_by_time": bool(np.all(t[:-1] <= t[1:])),
            }
        }
        
        # Check coordinate bounds against expected sensor resolution
        if file_key in self.expected_resolutions:
            expected_width, expected_height = self.expected_resolutions[file_key]
            results["coordinate_validation"] = {
                "x_within_bounds": bool(np.all((x >= 0) & (x < expected_width))),
                "y_within_bounds": bool(np.all((y >= 0) & (y < expected_height))),
                "expected_resolution": (expected_width, expected_height),
            }
        
        # Check polarity encoding
        valid_polarities = np.all(np.isin(p, [-1, 1]))
        results["polarity_validation"] = {
            "valid_encoding": bool(valid_polarities),
            "unique_values": list(np.unique(p)),
        }
        
        return results

    def test_evt2_format_detection(self):
        """Test format detection for EVT2 files."""
        if not self.test_files["evt2_small"].exists():
            pytest.skip("EVT2 test file not found")
        
        # Test format detection
        format_info = evlib.formats.detect_format(str(self.test_files["evt2_small"]))
        format_name, confidence, metadata = format_info
        
        assert format_name == "EVT2", f"Expected EVT2, got {format_name}"
        assert confidence > 0.8, f"Low confidence: {confidence}"
        assert "sensor_resolution" in metadata

    def test_evt2_small_file_loading(self):
        """Test loading small EVT2 file."""
        if not self.test_files["evt2_small"].exists():
            pytest.skip("EVT2 small test file not found")
        
        # Load events and measure performance
        load_func = lambda: evlib.formats.load_events(str(self.test_files["evt2_small"]))
        (x, y, t, p), execution_time, memory_usage = self.measure_performance(load_func)
        
        # Validate results
        validation_results = self.validate_events(x, y, t, p, "evt2_small")
        
        # Store results for later analysis
        self.results["evt2_small"] = {
            "validation": validation_results,
            "performance": {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "events_per_second": validation_results["event_count"] / execution_time,
            }
        }
        
        # Basic assertions
        assert len(x) > 0, "No events loaded"
        assert len(x) == len(y) == len(t) == len(p), "Array length mismatch"
        assert validation_results["coordinate_validation"]["x_within_bounds"], "X coordinates out of bounds"
        assert validation_results["coordinate_validation"]["y_within_bounds"], "Y coordinates out of bounds"
        assert validation_results["polarity_validation"]["valid_encoding"], "Invalid polarity encoding"
        assert not validation_results["data_integrity"]["has_nan"], "NaN values in timestamps"
        assert not validation_results["data_integrity"]["has_inf"], "Inf values in timestamps"
        
        print(f"EVT2 small file: {validation_results['event_count']} events in {execution_time:.2f}s")

    def test_evt2_large_file_loading(self):
        """Test loading large EVT2 file."""
        if not self.test_files["evt2_large"].exists():
            pytest.skip("EVT2 large test file not found")
        
        # Load events and measure performance
        load_func = lambda: evlib.formats.load_events(str(self.test_files["evt2_large"]))
        (x, y, t, p), execution_time, memory_usage = self.measure_performance(load_func)
        
        # Validate results
        validation_results = self.validate_events(x, y, t, p, "evt2_large")
        
        # Store results
        self.results["evt2_large"] = {
            "validation": validation_results,
            "performance": {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "events_per_second": validation_results["event_count"] / execution_time,
            }
        }
        
        # Basic assertions
        assert len(x) > 0, "No events loaded"
        assert len(x) == len(y) == len(t) == len(p), "Array length mismatch"
        assert validation_results["coordinate_validation"]["x_within_bounds"], "X coordinates out of bounds"
        assert validation_results["coordinate_validation"]["y_within_bounds"], "Y coordinates out of bounds"
        assert validation_results["polarity_validation"]["valid_encoding"], "Invalid polarity encoding"
        
        print(f"EVT2 large file: {validation_results['event_count']} events in {execution_time:.2f}s")

    def test_hdf5_format_detection(self):
        """Test format detection for HDF5 files."""
        if not self.test_files["hdf5_small"].exists():
            pytest.skip("HDF5 test file not found")
        
        # Test format detection
        format_info = evlib.formats.detect_format(str(self.test_files["hdf5_small"]))
        format_name, confidence, metadata = format_info
        
        assert format_name == "HDF5", f"Expected HDF5, got {format_name}"
        assert confidence > 0.8, f"Low confidence: {confidence}"

    def test_hdf5_small_file_loading(self):
        """Test loading small HDF5 file."""
        if not self.test_files["hdf5_small"].exists():
            pytest.skip("HDF5 small test file not found")
        
        # Load events and measure performance
        load_func = lambda: evlib.formats.load_events(str(self.test_files["hdf5_small"]))
        (x, y, t, p), execution_time, memory_usage = self.measure_performance(load_func)
        
        # Validate results
        validation_results = self.validate_events(x, y, t, p, "hdf5_small")
        
        # Store results
        self.results["hdf5_small"] = {
            "validation": validation_results,
            "performance": {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "events_per_second": validation_results["event_count"] / execution_time,
            }
        }
        
        # Basic assertions
        assert len(x) > 0, "No events loaded"
        assert len(x) == len(y) == len(t) == len(p), "Array length mismatch"
        assert validation_results["coordinate_validation"]["x_within_bounds"], "X coordinates out of bounds"
        assert validation_results["coordinate_validation"]["y_within_bounds"], "Y coordinates out of bounds"
        assert validation_results["polarity_validation"]["valid_encoding"], "Invalid polarity encoding"
        
        print(f"HDF5 small file: {validation_results['event_count']} events in {execution_time:.2f}s")

    def test_hdf5_large_file_loading(self):
        """Test loading large HDF5 file."""
        if not self.test_files["hdf5_large"].exists():
            pytest.skip("HDF5 large test file not found")
        
        # Load events and measure performance
        load_func = lambda: evlib.formats.load_events(str(self.test_files["hdf5_large"]))
        (x, y, t, p), execution_time, memory_usage = self.measure_performance(load_func)
        
        # Validate results
        validation_results = self.validate_events(x, y, t, p, "hdf5_large")
        
        # Store results
        self.results["hdf5_large"] = {
            "validation": validation_results,
            "performance": {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "events_per_second": validation_results["event_count"] / execution_time,
            }
        }
        
        # Basic assertions
        assert len(x) > 0, "No events loaded"
        assert len(x) == len(y) == len(t) == len(p), "Array length mismatch"
        assert validation_results["coordinate_validation"]["x_within_bounds"], "X coordinates out of bounds"
        assert validation_results["coordinate_validation"]["y_within_bounds"], "Y coordinates out of bounds"
        assert validation_results["polarity_validation"]["valid_encoding"], "Invalid polarity encoding"
        
        print(f"HDF5 large file: {validation_results['event_count']} events in {execution_time:.2f}s")

    def test_text_format_detection(self):
        """Test format detection for text files."""
        if not self.test_files["text_medium"].exists():
            pytest.skip("Text test file not found")
        
        # Test format detection
        format_info = evlib.formats.detect_format(str(self.test_files["text_medium"]))
        format_name, confidence, metadata = format_info
        
        assert format_name == "Text", f"Expected Text, got {format_name}"
        assert confidence > 0.8, f"Low confidence: {confidence}"

    def test_text_file_loading(self):
        """Test loading text file."""
        if not self.test_files["text_medium"].exists():
            pytest.skip("Text test file not found")
        
        # Load events and measure performance
        load_func = lambda: evlib.formats.load_events(str(self.test_files["text_medium"]))
        (x, y, t, p), execution_time, memory_usage = self.measure_performance(load_func)
        
        # Validate results
        validation_results = self.validate_events(x, y, t, p, "text_medium")
        
        # Store results
        self.results["text_medium"] = {
            "validation": validation_results,
            "performance": {
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "events_per_second": validation_results["event_count"] / execution_time,
            }
        }
        
        # Basic assertions
        assert len(x) > 0, "No events loaded"
        assert len(x) == len(y) == len(t) == len(p), "Array length mismatch"
        assert validation_results["coordinate_validation"]["x_within_bounds"], "X coordinates out of bounds"
        assert validation_results["coordinate_validation"]["y_within_bounds"], "Y coordinates out of bounds"
        assert validation_results["polarity_validation"]["valid_encoding"], "Invalid polarity encoding"
        
        print(f"Text file: {validation_results['event_count']} events in {execution_time:.2f}s")

    def test_polarity_encoding_conversion(self):
        """Test polarity encoding conversion from 0/1 to -1/1."""
        if not self.test_files["text_medium"].exists():
            pytest.skip("Text test file not found")
        
        # Load with default polarity encoding (should handle 0/1 -> -1/1 conversion)
        x, y, t, p = evlib.formats.load_events(str(self.test_files["text_medium"]))
        
        # Check that polarities are correctly encoded as -1/1
        unique_polarities = np.unique(p)
        assert set(unique_polarities) == {-1, 1}, f"Expected {{-1, 1}}, got {set(unique_polarities)}"
        
        # Count distribution
        pos_count = np.sum(p == 1)
        neg_count = np.sum(p == -1)
        total_count = len(p)
        
        assert pos_count + neg_count == total_count, "Polarity counts don't sum to total"
        assert pos_count > 0, "No positive polarity events"
        assert neg_count > 0, "No negative polarity events"
        
        print(f"Polarity distribution: {pos_count} positive, {neg_count} negative")

    def test_data_consistency_evt2_vs_hdf5(self):
        """Test data consistency between EVT2 and HDF5 versions of same file."""
        evt2_file = self.test_files["evt2_small"]
        hdf5_file = self.test_files["hdf5_small"]
        
        if not (evt2_file.exists() and hdf5_file.exists()):
            pytest.skip("Both EVT2 and HDF5 files needed for comparison")
        
        # Load both files
        x_evt2, y_evt2, t_evt2, p_evt2 = evlib.formats.load_events(str(evt2_file))
        x_hdf5, y_hdf5, t_hdf5, p_hdf5 = evlib.formats.load_events(str(hdf5_file))
        
        # Basic comparison
        evt2_count = len(x_evt2)
        hdf5_count = len(x_hdf5)
        
        # Allow for small differences due to format conversion
        count_diff = abs(evt2_count - hdf5_count)
        max_allowed_diff = max(evt2_count, hdf5_count) * 0.01  # 1% tolerance
        
        assert count_diff <= max_allowed_diff, f"Event count difference too large: {count_diff}"
        
        # Compare coordinate bounds
        evt2_x_range = (np.min(x_evt2), np.max(x_evt2))
        hdf5_x_range = (np.min(x_hdf5), np.max(x_hdf5))
        evt2_y_range = (np.min(y_evt2), np.max(y_evt2))
        hdf5_y_range = (np.min(y_hdf5), np.max(y_hdf5))
        
        assert evt2_x_range == hdf5_x_range, f"X range mismatch: {evt2_x_range} vs {hdf5_x_range}"
        assert evt2_y_range == hdf5_y_range, f"Y range mismatch: {evt2_y_range} vs {hdf5_y_range}"
        
        # Compare time ranges
        evt2_t_range = (np.min(t_evt2), np.max(t_evt2))
        hdf5_t_range = (np.min(t_hdf5), np.max(t_hdf5))
        
        # Allow for small time differences due to precision
        time_diff = abs(evt2_t_range[1] - hdf5_t_range[1])
        assert time_diff < 1.0, f"Time range difference too large: {time_diff}"
        
        print(f"Consistency check: EVT2 {evt2_count} events, HDF5 {hdf5_count} events")

    def test_filtering_functionality(self):
        """Test filtering functionality with real data."""
        if not self.test_files["text_medium"].exists():
            pytest.skip("Text test file not found")
        
        # Load full dataset
        x_full, y_full, t_full, p_full = evlib.formats.load_events(str(self.test_files["text_medium"]))
        
        # Test time filtering
        t_start = np.min(t_full) + 0.1
        t_end = np.max(t_full) - 0.1
        x_time, y_time, t_time, p_time = evlib.formats.load_events(
            str(self.test_files["text_medium"]),
            t_start=t_start,
            t_end=t_end
        )
        
        assert len(x_time) < len(x_full), "Time filtering didn't reduce event count"
        assert np.all(t_time >= t_start), "Time filtering failed (start)"
        assert np.all(t_time <= t_end), "Time filtering failed (end)"
        
        # Test spatial filtering
        x_center = (np.min(x_full) + np.max(x_full)) // 2
        y_center = (np.min(y_full) + np.max(y_full)) // 2
        x_spatial, y_spatial, t_spatial, p_spatial = evlib.formats.load_events(
            str(self.test_files["text_medium"]),
            min_x=x_center - 50,
            max_x=x_center + 50,
            min_y=y_center - 50,
            max_y=y_center + 50
        )
        
        assert len(x_spatial) < len(x_full), "Spatial filtering didn't reduce event count"
        assert np.all(x_spatial >= x_center - 50), "Spatial filtering failed (x min)"
        assert np.all(x_spatial <= x_center + 50), "Spatial filtering failed (x max)"
        assert np.all(y_spatial >= y_center - 50), "Spatial filtering failed (y min)"
        assert np.all(y_spatial <= y_center + 50), "Spatial filtering failed (y max)"
        
        # Test polarity filtering
        x_pos, y_pos, t_pos, p_pos = evlib.formats.load_events(
            str(self.test_files["text_medium"]),
            polarity=1
        )
        
        assert len(x_pos) < len(x_full), "Polarity filtering didn't reduce event count"
        assert np.all(p_pos == 1), "Polarity filtering failed"
        
        print(f"Filtering tests: full {len(x_full)}, time {len(x_time)}, spatial {len(x_spatial)}, polarity {len(x_pos)}")

    def test_memory_efficiency(self):
        """Test memory efficiency with large files."""
        if not self.test_files["evt2_large"].exists():
            pytest.skip("Large EVT2 file not found")
        
        # Get initial memory usage
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Load large file
        x, y, t, p = evlib.formats.load_events(str(self.test_files["evt2_large"]))
        
        # Get peak memory usage
        peak_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        # Calculate memory efficiency
        event_count = len(x)
        bytes_per_event = (memory_used * 1024 * 1024) / event_count
        
        # Each event should use roughly 24 bytes (8 bytes for timestamp, 2 for x, 2 for y, 1 for polarity)
        # Plus Python overhead, so let's allow up to 100 bytes per event
        assert bytes_per_event < 100, f"Memory usage too high: {bytes_per_event:.1f} bytes per event"
        
        print(f"Memory efficiency: {memory_used:.1f}MB for {event_count} events ({bytes_per_event:.1f} bytes/event)")
        
        # Clean up
        del x, y, t, p
        gc.collect()

    def test_error_handling(self):
        """Test error handling with malformed files."""
        # Test with non-existent file
        with pytest.raises(Exception):
            evlib.formats.load_events("non_existent_file.txt")
        
        # Test with empty file
        empty_file = self.data_dir / "empty_test.txt"
        empty_file.write_text("")
        try:
            x, y, t, p = evlib.formats.load_events(str(empty_file))
            assert len(x) == 0, "Empty file should return empty arrays"
        finally:
            empty_file.unlink()
        
        # Test with malformed text file
        malformed_file = self.data_dir / "malformed_test.txt"
        malformed_file.write_text("not a valid event format\n")
        try:
            with pytest.raises(Exception):
                evlib.formats.load_events(str(malformed_file))
        finally:
            malformed_file.unlink()

    def test_performance_summary(self):
        """Generate performance summary from all tests."""
        if not self.results:
            pytest.skip("No performance data available")
        
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        for file_key, data in self.results.items():
            validation = data["validation"]
            performance = data["performance"]
            
            print(f"\n{file_key.upper()}:")
            print(f"  Events: {validation['event_count']:,}")
            print(f"  Time: {performance['execution_time']:.2f}s")
            print(f"  Memory: {performance['memory_usage']:.1f}MB")
            print(f"  Rate: {performance['events_per_second']:,.0f} events/s")
            print(f"  Coordinates: ({validation['coordinate_bounds']['x_min']},{validation['coordinate_bounds']['y_min']}) to ({validation['coordinate_bounds']['x_max']},{validation['coordinate_bounds']['y_max']})")
            print(f"  Duration: {validation['time_range']['duration']:.3f}s")
            print(f"  Polarity: {validation['polarity_distribution']['positive']} pos, {validation['polarity_distribution']['negative']} neg")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])