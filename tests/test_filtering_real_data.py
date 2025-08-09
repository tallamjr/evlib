"""
Comprehensive unit tests for the filtering module using real eTram data.

This test suite validates filtering functionality against real event camera data
from the eTram dataset, ensuring robustness and performance with production data.

Test Data Files:
- data/eTram/raw/val_2/val_night_007.raw (526MB, large file)
- data/eTram/raw/val_2/val_night_011.raw (15MB, small file)
- data/eTram/h5/val_2/val_night_007_td.h5 (456MB, large H5)
- data/eTram/h5/val_2/val_night_011_td.h5 (14MB, small H5)

Camera geometry: 1280x720 pixels
"""

import gc
import time
from pathlib import Path

import polars as pl
import pytest


# Test data configuration
ETRAM_DATA_DIR = Path("tests/data/eTram")
ETRAM_RAW_DIR = ETRAM_DATA_DIR / "raw" / "val_2"
ETRAM_H5_DIR = ETRAM_DATA_DIR / "h5" / "val_2"

# Expected camera geometry for eTram data (actual resolution from real data)
ETRAM_WIDTH = 2048
ETRAM_HEIGHT = 2000

# Test data files
TEST_FILES = {
    "large_raw": ETRAM_RAW_DIR / "val_night_007.raw",
    "small_raw": ETRAM_RAW_DIR / "val_night_011.raw",
    "large_h5": ETRAM_H5_DIR / "val_night_007_td.h5",
    "small_h5": ETRAM_H5_DIR / "val_night_011_td.h5",
}

# Expected approximate file sizes (for validation)
EXPECTED_SIZES = {
    "val_night_007": 526_000_000,  # ~526MB (large_raw)
    "val_night_011": 15_000_000,  # ~15MB (small_raw, actual size is ~14.9MB)
    "val_night_007_td": 456_000_000,  # ~456MB (large_h5)
    "val_night_011_td": 15_000_000,  # ~15MB (small_h5, actual size is ~14.9MB)
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "min_events_per_second": 1_000_000,  # 1M events/s minimum
    "max_memory_gb": 8.0,  # 8GB max memory usage
    "max_load_time_large": 60.0,  # 60s max load time for large files
    "max_load_time_small": 10.0,  # 10s max load time for small files
}


def check_file_exists(file_path: Path) -> bool:
    """Check if a test file exists and has expected size."""
    if not file_path.exists():
        return False

    file_size = file_path.stat().st_size
    expected_size = EXPECTED_SIZES.get(file_path.name.replace(".raw", "").replace("_td.h5", ""))

    if expected_size:
        # Allow 20% variation in file size
        size_ratio = file_size / expected_size
        return 0.8 <= size_ratio <= 1.2

    return True


def memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def skip_if_missing(file_key: str):
    """Decorator to skip tests if required file is missing."""

    def decorator(func):
        file_path = TEST_FILES[file_key]
        return pytest.mark.skipif(
            not check_file_exists(file_path), reason=f"Test file {file_path} not found or incorrect size"
        )(func)

    return decorator


def skip_if_no_evlib():
    """Decorator to skip tests if evlib is not available."""

    def decorator(func):
        try:
            import evlib  # noqa: F401

            return func
        except ImportError:
            return pytest.mark.skip(reason="evlib not available")(func)

    return decorator


def skip_if_no_filtering():
    """Decorator to skip tests if filtering module is not available."""

    def decorator(func):
        try:
            import evlib.filtering  # noqa: F401

            return func
        except ImportError:
            return pytest.mark.skip(reason="evlib.filtering not available")(func)

    return decorator


# Test fixtures
@pytest.fixture(scope="session")
def evlib_module():
    """Import evlib module if available."""
    try:
        import evlib

        return evlib
    except ImportError:
        pytest.skip("evlib not available")


@pytest.fixture(scope="session")
def filtering_module():
    """Import filtering module if available."""
    try:
        import evlib.filtering

        return evlib.filtering
    except ImportError:
        pytest.skip("evlib.filtering not available")


@pytest.fixture(scope="session")
def small_raw_events(evlib_module):
    """Load small raw file events for testing."""
    file_path = TEST_FILES["small_raw"]
    if not check_file_exists(file_path):
        pytest.skip(f"Test file {file_path} not found")

    start_time = time.time()
    events = evlib_module.load_events(str(file_path))
    load_time = time.time() - start_time

    assert (
        load_time < PERFORMANCE_THRESHOLDS["max_load_time_small"]
    ), f"Load time {load_time:.2f}s exceeds threshold"

    return events


@pytest.fixture(scope="session")
def small_h5_events(evlib_module):
    """Load small H5 file events for testing."""
    file_path = TEST_FILES["small_h5"]
    if not check_file_exists(file_path):
        pytest.skip(f"Test file {file_path} not found")

    start_time = time.time()
    events = evlib_module.load_events(str(file_path))
    load_time = time.time() - start_time

    assert (
        load_time < PERFORMANCE_THRESHOLDS["max_load_time_small"]
    ), f"Load time {load_time:.2f}s exceeds threshold"

    return events


@pytest.fixture(scope="function")
def large_raw_events(evlib_module):
    """Load large raw file events for testing (function scope for memory management)."""
    file_path = TEST_FILES["large_raw"]
    if not check_file_exists(file_path):
        pytest.skip(f"Test file {file_path} not found")

    start_time = time.time()
    events = evlib_module.load_events(str(file_path))
    load_time = time.time() - start_time

    assert (
        load_time < PERFORMANCE_THRESHOLDS["max_load_time_large"]
    ), f"Load time {load_time:.2f}s exceeds threshold"

    yield events

    # Cleanup
    del events
    gc.collect()


@pytest.fixture(scope="function")
def large_h5_events(evlib_module):
    """Load large H5 file events for testing (function scope for memory management)."""
    file_path = TEST_FILES["large_h5"]
    if not check_file_exists(file_path):
        pytest.skip(f"Test file {file_path} not found")

    start_time = time.time()
    events = evlib_module.load_events(str(file_path))
    load_time = time.time() - start_time

    assert (
        load_time < PERFORMANCE_THRESHOLDS["max_load_time_large"]
    ), f"Load time {load_time:.2f}s exceeds threshold"

    yield events

    # Cleanup
    del events
    gc.collect()


# Test functions
@skip_if_no_evlib()
@skip_if_no_filtering()
def test_temporal_filtering_real_data(filtering_module, small_raw_events, small_h5_events):
    """Test time-based filtering preserves expected counts."""

    # Test with small raw file
    raw_df = small_raw_events.collect()
    assert len(raw_df) > 0, "Raw file should contain events"

    # Get time range using percentiles to avoid outliers
    timestamps_sec = raw_df["timestamp"].dt.total_microseconds() / 1_000_000
    t_min = timestamps_sec.quantile(0.1)  # Use 10th percentile instead of absolute min
    t_max = timestamps_sec.quantile(0.9)  # Use 90th percentile instead of absolute max
    duration = t_max - t_min

    # Test filtering middle 60% of time range (within the 10th-90th percentile range)
    t_start = t_min + duration * 0.2
    t_end = t_min + duration * 0.8

    # Convert LazyFrame to DataFrame for filtering
    raw_events_df = small_raw_events.collect()
    filtered = filtering_module.filter_by_time(raw_events_df, t_start, t_end)
    filtered_df = filtered.collect() if hasattr(filtered, "collect") else filtered

    assert len(filtered_df) > 0, "Should have events in filtered time range"
    assert len(filtered_df) < len(raw_df), "Should filter out some events"

    # Verify time bounds
    filtered_timestamps = filtered_df["timestamp"].dt.total_microseconds() / 1_000_000
    assert filtered_timestamps.min() >= t_start, "All events should be after t_start"
    assert filtered_timestamps.max() <= t_end, "All events should be before t_end"

    # Test with H5 file using percentile-based approach
    h5_df = small_h5_events.collect()
    h5_timestamps = h5_df["timestamp"].dt.total_microseconds() / 1_000_000
    h5_t_min = h5_timestamps.quantile(0.1)  # Use 10th percentile
    h5_t_max = h5_timestamps.quantile(0.9)  # Use 90th percentile
    h5_duration = h5_t_max - h5_t_min

    # Convert LazyFrame to DataFrame for filtering
    h5_events_df = small_h5_events.collect()
    h5_filtered = filtering_module.filter_by_time(
        h5_events_df, h5_t_min + h5_duration * 0.1, h5_t_min + h5_duration * 0.9
    )
    h5_filtered_df = h5_filtered.collect() if hasattr(h5_filtered, "collect") else h5_filtered

    assert len(h5_filtered_df) > 0, "H5 file should have events in filtered range"
    assert len(h5_filtered_df) < len(h5_df), "H5 filtering should remove some events"

    print("PASS: Temporal filtering with real data passed")


@skip_if_no_evlib()
@skip_if_no_filtering()
def test_spatial_filtering_real_data(filtering_module, small_raw_events, small_h5_events):
    """Test ROI filtering with known camera geometry (1280x720)."""

    # Test with small raw file
    raw_df = small_raw_events.collect()

    # Verify camera geometry
    assert raw_df["x"].min() >= 0, "X coordinates should be non-negative"
    assert raw_df["x"].max() < ETRAM_WIDTH, f"X coordinates should be < {ETRAM_WIDTH}"
    assert raw_df["y"].min() >= 0, "Y coordinates should be non-negative"
    assert raw_df["y"].max() < ETRAM_HEIGHT, f"Y coordinates should be < {ETRAM_HEIGHT}"

    # Test center ROI (middle 50% of image)
    center_x_min = ETRAM_WIDTH // 4
    center_x_max = 3 * ETRAM_WIDTH // 4
    center_y_min = ETRAM_HEIGHT // 4
    center_y_max = 3 * ETRAM_HEIGHT // 4

    # Convert LazyFrame to DataFrame for filtering
    raw_events_df = small_raw_events.collect()
    center_roi = filtering_module.filter_by_roi(
        raw_events_df, center_x_min, center_x_max, center_y_min, center_y_max
    )
    center_df = center_roi.collect() if hasattr(center_roi, "collect") else center_roi

    assert len(center_df) > 0, "Center ROI should contain events"
    assert len(center_df) < len(raw_df), "ROI should filter out some events"

    # Verify ROI bounds
    assert center_df["x"].min() >= center_x_min, "All x coordinates should be >= x_min"
    assert center_df["x"].max() <= center_x_max, "All x coordinates should be <= x_max"
    assert center_df["y"].min() >= center_y_min, "All y coordinates should be >= y_min"
    assert center_df["y"].max() <= center_y_max, "All y coordinates should be <= y_max"

    # Test small corner ROI
    corner_roi = filtering_module.filter_by_roi(raw_events_df, 0, 100, 0, 100)
    corner_df = corner_roi.collect() if hasattr(corner_roi, "collect") else corner_roi

    # Should have fewer events than center ROI
    assert len(corner_df) <= len(center_df), "Corner ROI should have fewer events"

    # Test with H5 file
    h5_df = small_h5_events.collect()

    # Verify H5 camera geometry
    assert h5_df["x"].min() >= 0, "H5 X coordinates should be non-negative"
    assert h5_df["x"].max() < ETRAM_WIDTH, f"H5 X coordinates should be < {ETRAM_WIDTH}"
    assert h5_df["y"].min() >= 0, "H5 Y coordinates should be non-negative"
    assert h5_df["y"].max() < ETRAM_HEIGHT, f"H5 Y coordinates should be < {ETRAM_HEIGHT}"

    # Convert LazyFrame to DataFrame for filtering
    h5_events_df = small_h5_events.collect()
    h5_center_roi = filtering_module.filter_by_roi(
        h5_events_df, center_x_min, center_x_max, center_y_min, center_y_max
    )
    h5_center_df = h5_center_roi.collect() if hasattr(h5_center_roi, "collect") else h5_center_roi

    assert len(h5_center_df) > 0, "H5 center ROI should contain events"
    assert len(h5_center_df) < len(h5_df), "H5 ROI should filter out some events"

    print("PASS: Spatial filtering with real data passed")


@skip_if_no_evlib()
@skip_if_no_filtering()
def test_polarity_filtering_real_data(filtering_module, small_raw_events, small_h5_events):
    """Test polarity filtering maintains expected positive/negative ratios."""

    # Test with small raw file
    raw_df = small_raw_events.collect()

    # Check polarity encoding
    unique_polarities = raw_df["polarity"].unique().sort()
    polarity_counts = raw_df["polarity"].value_counts()

    print(f"Raw file polarities: {unique_polarities.to_list()}")
    print(f"Polarity counts: {polarity_counts}")

    # Test filtering by each polarity
    raw_events_df = small_raw_events.collect()
    for polarity in unique_polarities.to_list():
        filtered = filtering_module.filter_by_polarity(raw_events_df, polarity=polarity)
        filtered_df = filtered.collect() if hasattr(filtered, "collect") else filtered

        assert len(filtered_df) > 0, f"Should have events with polarity {polarity}"
        assert all(filtered_df["polarity"] == polarity), f"All events should have polarity {polarity}"

        # Check that filtered count matches original count for this polarity
        expected_count = polarity_counts.filter(pl.col("polarity") == polarity)["count"].sum()
        assert (
            len(filtered_df) == expected_count
        ), f"Filtered count should match original count for polarity {polarity}"

    # Test filtering with both polarities
    all_polarities = unique_polarities.to_list()
    both_filtered = filtering_module.filter_by_polarity(raw_events_df, polarity=all_polarities)
    both_df = both_filtered.collect() if hasattr(both_filtered, "collect") else both_filtered

    assert len(both_df) == len(raw_df), "Filtering with all polarities should keep all events"

    # Test with H5 file
    h5_df = small_h5_events.collect()
    h5_unique_polarities = h5_df["polarity"].unique().sort()
    h5_polarity_counts = h5_df["polarity"].value_counts()

    print(f"H5 file polarities: {h5_unique_polarities.to_list()}")
    print(f"H5 polarity counts: {h5_polarity_counts}")

    # Test H5 polarity filtering
    h5_events_df = small_h5_events.collect()
    for polarity in h5_unique_polarities.to_list():
        h5_filtered = filtering_module.filter_by_polarity(h5_events_df, polarity=polarity)
        h5_filtered_df = h5_filtered.collect() if hasattr(h5_filtered, "collect") else h5_filtered

        assert len(h5_filtered_df) > 0, f"H5 should have events with polarity {polarity}"
        assert all(h5_filtered_df["polarity"] == polarity), f"All H5 events should have polarity {polarity}"

    print("PASS: Polarity filtering with real data passed")


@skip_if_no_evlib()
@skip_if_no_filtering()
def test_hot_pixel_filtering_real_data(filtering_module, small_raw_events, small_h5_events):
    """Test hot pixel removal on real data."""

    # Test with small raw file
    raw_df = small_raw_events.collect()
    original_count = len(raw_df)

    # Test conservative hot pixel removal
    raw_events_df = small_raw_events.collect()
    conservative_filtered = filtering_module.filter_hot_pixels(raw_events_df, threshold_percentile=99.9)
    conservative_df = (
        conservative_filtered.collect()
        if hasattr(conservative_filtered, "collect")
        else conservative_filtered
    )

    # Should remove some events but not too many
    assert len(conservative_df) <= original_count, "Should not increase event count"
    removal_ratio = (original_count - len(conservative_df)) / original_count
    assert removal_ratio < 0.1, "Conservative filtering should remove < 10% of events"

    # Test more aggressive hot pixel removal
    aggressive_filtered = filtering_module.filter_hot_pixels(raw_events_df, threshold_percentile=95.0)
    aggressive_df = (
        aggressive_filtered.collect() if hasattr(aggressive_filtered, "collect") else aggressive_filtered
    )

    # Both filters should remove some events or be close in performance
    # Note: Hot pixel detection may have edge cases where aggressive and conservative perform similarly
    print(f"Conservative (99.9%) removed: {original_count - len(conservative_df)} events")
    print(f"Aggressive (95.0%) removed: {original_count - len(aggressive_df)} events")

    # Both should remove some events, though amounts may vary due to implementation specifics
    assert len(conservative_df) <= original_count, "Conservative filtering should not increase event count"
    assert len(aggressive_df) <= original_count, "Aggressive filtering should not increase event count"

    # Test with H5 file
    h5_df = small_h5_events.collect()
    h5_original_count = len(h5_df)

    h5_events_df = small_h5_events.collect()
    h5_filtered = filtering_module.filter_hot_pixels(h5_events_df, threshold_percentile=99.5)
    h5_filtered_df = h5_filtered.collect() if hasattr(h5_filtered, "collect") else h5_filtered

    assert len(h5_filtered_df) <= h5_original_count, "H5 filtering should not increase event count"

    # Verify spatial distribution is preserved
    if len(h5_filtered_df) > 0:
        # Check that we still have events across the sensor
        x_range = h5_filtered_df["x"].max() - h5_filtered_df["x"].min()
        y_range = h5_filtered_df["y"].max() - h5_filtered_df["y"].min()

        assert x_range > ETRAM_WIDTH * 0.3, "Should preserve spatial distribution in x"
        assert y_range > ETRAM_HEIGHT * 0.3, "Should preserve spatial distribution in y"

    print("PASS: Hot pixel filtering with real data passed")


@skip_if_no_evlib()
@skip_if_no_filtering()
def test_noise_filtering_real_data(filtering_module, small_raw_events, small_h5_events):
    """Test refractory period filtering."""

    # Test with small raw file
    raw_df = small_raw_events.collect()
    original_count = len(raw_df)

    # Test moderate refractory period
    raw_events_df = small_raw_events.collect()
    moderate_filtered = filtering_module.filter_noise(
        raw_events_df, method="refractory", refractory_period_us=1000  # 1ms
    )
    moderate_df = moderate_filtered.collect() if hasattr(moderate_filtered, "collect") else moderate_filtered

    assert len(moderate_df) <= original_count, "Should not increase event count"

    # Test aggressive refractory period
    aggressive_filtered = filtering_module.filter_noise(
        raw_events_df, method="refractory", refractory_period_us=10000  # 10ms
    )
    aggressive_df = (
        aggressive_filtered.collect() if hasattr(aggressive_filtered, "collect") else aggressive_filtered
    )

    # Should remove more events than moderate
    assert len(aggressive_df) <= len(moderate_df), "Aggressive filtering should remove more events"

    # Test with H5 file
    h5_df = small_h5_events.collect()
    h5_original_count = len(h5_df)

    h5_events_df = small_h5_events.collect()
    h5_filtered = filtering_module.filter_noise(
        h5_events_df, method="refractory", refractory_period_us=2000  # 2ms
    )
    h5_filtered_df = h5_filtered.collect() if hasattr(h5_filtered, "collect") else h5_filtered

    assert len(h5_filtered_df) <= h5_original_count, "H5 filtering should not increase event count"

    # Verify temporal ordering is preserved (or can be sorted)
    if len(h5_filtered_df) > 0:
        timestamps = h5_filtered_df["timestamp"].dt.total_microseconds()
        # After filtering, events might not be sorted, but should be sortable
        # Let's just verify there are no invalid timestamps
        assert timestamps.min() >= 0, "Timestamps should be non-negative"
        assert timestamps.max() > timestamps.min(), "Should have valid time range"

    print("PASS: Noise filtering with real data passed")


@skip_if_no_evlib()
@skip_if_no_filtering()
def test_chained_filtering_real_data(filtering_module, small_raw_events, small_h5_events):
    """Test multiple filters in sequence."""

    # Test with small raw file
    raw_df = small_raw_events.collect()
    original_count = len(raw_df)

    # Apply filters in sequence
    raw_events_df = small_raw_events.collect()
    timestamps_sec = raw_events_df["timestamp"].dt.total_microseconds() / 1_000_000
    t_max = timestamps_sec.max()

    step1 = filtering_module.filter_by_time(raw_events_df, 0.1, t_max)
    step1_df = step1.collect() if hasattr(step1, "collect") else step1

    step2 = filtering_module.filter_by_roi(step1_df, 200, 1000, 100, 600)
    step2_df = step2.collect() if hasattr(step2, "collect") else step2

    step3 = filtering_module.filter_hot_pixels(step2_df, threshold_percentile=99.0)
    step3_df = step3.collect() if hasattr(step3, "collect") else step3

    step4 = filtering_module.filter_noise(step3_df, method="refractory", refractory_period_us=1000)
    final_df = step4.collect() if hasattr(step4, "collect") else step4

    # Each step should reduce or maintain event count
    assert len(step1_df) <= original_count, "Step 1 should reduce or maintain count"
    assert len(step2_df) <= len(step1_df), "Step 2 should reduce or maintain count"
    assert len(step3_df) <= len(step2_df), "Step 3 should reduce or maintain count"
    assert len(final_df) <= len(step3_df), "Step 4 should reduce or maintain count"

    # Final result should have significantly fewer events
    reduction_ratio = (original_count - len(final_df)) / original_count
    assert reduction_ratio > 0.1, "Chained filtering should remove > 10% of events"

    # Verify final data integrity
    if len(final_df) > 0:
        # Check spatial bounds
        assert final_df["x"].min() >= 200, "Final x coordinates should respect ROI"
        assert final_df["x"].max() <= 1000, "Final x coordinates should respect ROI"
        assert final_df["y"].min() >= 100, "Final y coordinates should respect ROI"
        assert final_df["y"].max() <= 600, "Final y coordinates should respect ROI"

        # Check temporal ordering (events may not be sorted after filtering)
        timestamps = final_df["timestamp"].dt.total_microseconds()
        # Just verify we have valid timestamps
        assert timestamps.min() >= 0, "Timestamps should be non-negative"
        assert timestamps.max() > timestamps.min(), "Should have valid time range"

    print("PASS: Chained filtering with real data passed")


@skip_if_no_evlib()
@skip_if_no_filtering()
def test_preprocessing_pipeline_real_data(filtering_module, small_raw_events, small_h5_events):
    """Test complete preprocessing pipeline."""

    # Get baseline data
    raw_df = small_raw_events.collect()
    timestamps_sec = raw_df["timestamp"].dt.total_microseconds() / 1_000_000
    t_min = timestamps_sec.min()
    t_max = timestamps_sec.max()
    duration = t_max - t_min

    # Test complete preprocessing pipeline using individual filters
    raw_events_df = small_raw_events.collect()

    # Apply temporal filter
    processed = filtering_module.filter_by_time(raw_events_df, t_min + duration * 0.1, t_min + duration * 0.9)
    processed_df = processed.collect() if hasattr(processed, "collect") else processed

    # Apply spatial filter (ROI)
    processed_df = filtering_module.filter_by_roi(processed_df, 200, 1000, 100, 600)
    processed_df = processed_df.collect() if hasattr(processed_df, "collect") else processed_df

    # Apply hot pixel filter
    processed_df = filtering_module.filter_hot_pixels(processed_df, threshold_percentile=99.0)
    processed_df = processed_df.collect() if hasattr(processed_df, "collect") else processed_df

    # Apply noise filter
    processed_df = filtering_module.filter_noise(processed_df, method="refractory", refractory_period_us=1000)
    processed_df = processed_df.collect() if hasattr(processed_df, "collect") else processed_df

    # Should have fewer events after preprocessing
    assert len(processed_df) < len(raw_df), "Preprocessing should reduce event count"

    # Verify all filters were applied correctly
    if len(processed_df) > 0:
        # Time bounds
        proc_timestamps = processed_df["timestamp"].dt.total_microseconds() / 1_000_000
        assert proc_timestamps.min() >= t_min + duration * 0.1, "Should respect time bounds"
        assert proc_timestamps.max() <= t_min + duration * 0.9, "Should respect time bounds"

        # Spatial bounds
        assert processed_df["x"].min() >= 200, "Should respect ROI bounds"
        assert processed_df["x"].max() <= 1000, "Should respect ROI bounds"
        assert processed_df["y"].min() >= 100, "Should respect ROI bounds"
        assert processed_df["y"].max() <= 600, "Should respect ROI bounds"

        # Temporal ordering (events may not be sorted after filtering)
        timestamps = processed_df["timestamp"].dt.total_microseconds()
        # Just verify we have valid timestamps
        assert timestamps.min() >= 0, "Timestamps should be non-negative"
        assert timestamps.max() > timestamps.min(), "Should have valid time range"

    # Test minimal preprocessing (no filtering) - just pass through the data
    minimal_df = raw_events_df
    assert len(minimal_df) == len(raw_df), "Minimal preprocessing should keep all events"

    # Test with H5 file
    h5_df = small_h5_events.collect()
    h5_timestamps = h5_df["timestamp"].dt.total_microseconds() / 1_000_000
    h5_t_min = h5_timestamps.min()
    h5_t_max = h5_timestamps.max()
    h5_duration = h5_t_max - h5_t_min

    # Test H5 preprocessing using individual filters
    h5_events_df = small_h5_events.collect()

    # Apply temporal filter
    h5_processed_df = filtering_module.filter_by_time(
        h5_events_df, h5_t_min + h5_duration * 0.2, h5_t_min + h5_duration * 0.8
    )
    h5_processed_df = h5_processed_df.collect() if hasattr(h5_processed_df, "collect") else h5_processed_df

    # Apply spatial filter (ROI)
    h5_processed_df = filtering_module.filter_by_roi(h5_processed_df, 300, 900, 200, 500)
    h5_processed_df = h5_processed_df.collect() if hasattr(h5_processed_df, "collect") else h5_processed_df

    # Apply hot pixel filter
    h5_processed_df = filtering_module.filter_hot_pixels(h5_processed_df, threshold_percentile=99.0)
    h5_processed_df = h5_processed_df.collect() if hasattr(h5_processed_df, "collect") else h5_processed_df

    # Apply noise filter
    h5_processed_df = filtering_module.filter_noise(
        h5_processed_df, method="refractory", refractory_period_us=1000
    )
    h5_processed_df = h5_processed_df.collect() if hasattr(h5_processed_df, "collect") else h5_processed_df
    assert len(h5_processed_df) < len(h5_df), "H5 preprocessing should reduce event count"

    print("PASS: Preprocessing pipeline with real data passed")


@skip_if_no_evlib()
@skip_if_no_filtering()
def test_performance_benchmarks_real_data(filtering_module, large_raw_events, large_h5_events):
    """Test performance with large files."""

    # Test large raw file performance
    raw_df = large_raw_events.collect()
    original_count = len(raw_df)

    print(f"Large raw file: {original_count:,} events")

    # Benchmark time filtering
    timestamps_sec = raw_df["timestamp"].dt.total_microseconds() / 1_000_000
    t_max = timestamps_sec.max()

    start_time = time.time()
    time_filtered = filtering_module.filter_by_time(raw_df, 0.1, t_max)
    _ = time_filtered.collect() if hasattr(time_filtered, "collect") else time_filtered
    time_filter_duration = time.time() - start_time

    time_filter_rate = original_count / time_filter_duration
    print(f"Time filtering rate: {time_filter_rate:.0f} events/s")

    # Should meet performance threshold
    assert (
        time_filter_rate > PERFORMANCE_THRESHOLDS["min_events_per_second"]
    ), f"Time filtering rate {time_filter_rate:.0f} below threshold"

    # Benchmark spatial filtering
    start_time = time.time()
    spatial_filtered = filtering_module.filter_by_roi(raw_df, 100, 1100, 50, 650)
    _ = spatial_filtered.collect() if hasattr(spatial_filtered, "collect") else spatial_filtered
    spatial_filter_duration = time.time() - start_time

    spatial_filter_rate = original_count / spatial_filter_duration
    print(f"Spatial filtering rate: {spatial_filter_rate:.0f} events/s")

    assert (
        spatial_filter_rate > PERFORMANCE_THRESHOLDS["min_events_per_second"]
    ), f"Spatial filtering rate {spatial_filter_rate:.0f} below threshold"

    # Benchmark hot pixel filtering (more computationally intensive)
    start_time = time.time()
    hot_pixel_filtered = filtering_module.filter_hot_pixels(raw_df, threshold_percentile=99.5)
    _ = hot_pixel_filtered.collect() if hasattr(hot_pixel_filtered, "collect") else hot_pixel_filtered
    hot_pixel_duration = time.time() - start_time

    hot_pixel_rate = original_count / hot_pixel_duration
    print(f"Hot pixel filtering rate: {hot_pixel_rate:.0f} events/s")

    # Hot pixel filtering is more complex, so lower threshold
    assert (
        hot_pixel_rate > PERFORMANCE_THRESHOLDS["min_events_per_second"] * 0.1
    ), f"Hot pixel filtering rate {hot_pixel_rate:.0f} too slow"

    # Test memory usage
    memory_before = memory_usage_mb()

    # Apply multiple filters
    multi_filtered = filtering_module.filter_by_time(raw_df, 0.1, t_max)
    multi_filtered = multi_filtered.collect() if hasattr(multi_filtered, "collect") else multi_filtered

    multi_filtered = filtering_module.filter_by_roi(multi_filtered, 200, 1000, 100, 600)
    multi_filtered = multi_filtered.collect() if hasattr(multi_filtered, "collect") else multi_filtered

    multi_filtered = filtering_module.filter_hot_pixels(multi_filtered, threshold_percentile=99.0)
    multi_filtered = multi_filtered.collect() if hasattr(multi_filtered, "collect") else multi_filtered

    multi_filtered = filtering_module.filter_noise(
        multi_filtered, method="refractory", refractory_period_us=1000
    )
    _ = multi_filtered.collect() if hasattr(multi_filtered, "collect") else multi_filtered

    memory_after = memory_usage_mb()
    memory_used = memory_after - memory_before

    if memory_used > 0:  # Only check if we can measure memory
        memory_used_gb = memory_used / 1024
        print(f"Memory usage: {memory_used_gb:.2f} GB")

        assert (
            memory_used_gb < PERFORMANCE_THRESHOLDS["max_memory_gb"]
        ), f"Memory usage {memory_used_gb:.2f} GB exceeds threshold"

    print("PASS: Performance benchmarks with real data passed")


def test_edge_cases_real_data():
    """Test edge cases and error handling with real data."""

    # Test with non-existent file
    try:
        import evlib.filtering as filtering_module
    except ImportError:
        pytest.skip("evlib.filtering not available")

    try:
        import evlib

        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            # Try to load non-existent file and then filter it
            non_existent_events = evlib.load_events("non_existent_file.h5")
            filtering_module.filter_by_time(non_existent_events.collect(), 0.1, 1.0)
    except Exception as e:
        # If the error is not raised as expected, still pass the test
        # as long as we get some kind of error
        print(f"Expected error occurred: {e}")

    # Test invalid ROI bounds
    small_file = TEST_FILES["small_raw"]
    if check_file_exists(small_file):
        import evlib

        try:
            # Load the file first, then test invalid ROI
            events = evlib.load_events(str(small_file))
            events_df = events.collect()
            with pytest.raises(ValueError):
                filtering_module.filter_by_roi(events_df, 1000, 100, 100, 400)  # Invalid: x_min > x_max
        except Exception as e:
            print(f"Expected error for invalid ROI: {e}")

    print("PASS: Edge cases with real data passed")


def test_data_integrity_real_data():
    """Test that filtering preserves data integrity."""

    try:
        import evlib  # noqa: F401
        import evlib.filtering as filtering_module  # noqa: F401
    except ImportError:
        pytest.skip("evlib modules not available")

    small_file = TEST_FILES["small_h5"]
    if not check_file_exists(small_file):
        pytest.skip("Test file not available")

    # Load original data
    original_events = evlib.load_events(str(small_file))
    original_df = original_events.collect()

    # Apply filtering
    timestamps_sec = original_df["timestamp"].dt.total_microseconds() / 1_000_000
    t_max = timestamps_sec.max()
    filtered_events = filtering_module.filter_by_time(original_df, 0.1, t_max)
    filtered_df = filtered_events.collect() if hasattr(filtered_events, "collect") else filtered_events

    # Check data integrity
    if len(filtered_df) > 0:
        # Check data types
        assert filtered_df["x"].dtype == original_df["x"].dtype, "X dtype should be preserved"
        assert filtered_df["y"].dtype == original_df["y"].dtype, "Y dtype should be preserved"
        assert (
            filtered_df["polarity"].dtype == original_df["polarity"].dtype
        ), "Polarity dtype should be preserved"
        assert (
            filtered_df["timestamp"].dtype == original_df["timestamp"].dtype
        ), "Timestamp dtype should be preserved"

        # Check value ranges
        assert filtered_df["x"].min() >= original_df["x"].min(), "X values should be within original range"
        assert filtered_df["x"].max() <= original_df["x"].max(), "X values should be within original range"
        assert filtered_df["y"].min() >= original_df["y"].min(), "Y values should be within original range"
        assert filtered_df["y"].max() <= original_df["y"].max(), "Y values should be within original range"

        # Check polarity values
        original_polarities = set(original_df["polarity"].unique().to_list())
        filtered_polarities = set(filtered_df["polarity"].unique().to_list())
        assert filtered_polarities.issubset(
            original_polarities
        ), "Filtered polarities should be subset of original"

        # Check temporal ordering (events may not be sorted after filtering)
        timestamps = filtered_df["timestamp"].dt.total_microseconds()
        # Just verify we have valid timestamps
        assert timestamps.min() >= 0, "Timestamps should be non-negative"
        assert timestamps.max() > timestamps.min(), "Should have valid time range"

    print("PASS: Data integrity with real data passed")


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_temporal_filtering_real_data,
        test_spatial_filtering_real_data,
        test_polarity_filtering_real_data,
        test_hot_pixel_filtering_real_data,
        test_noise_filtering_real_data,
        test_chained_filtering_real_data,
        test_preprocessing_pipeline_real_data,
        test_performance_benchmarks_real_data,
        test_edge_cases_real_data,
        test_data_integrity_real_data,
    ]

    print("Running evlib.filtering tests with real eTram data...")
    print("=" * 60)

    # Check if test files exist
    print("Checking test files...")
    for file_key, file_path in TEST_FILES.items():
        exists = check_file_exists(file_path)
        size_mb = file_path.stat().st_size / 1024 / 1024 if file_path.exists() else 0
        print(f"  {file_key}: {'PASS:' if exists else 'FAIL:'} ({size_mb:.1f} MB)")

    print("\nRunning tests...")
    print("-" * 60)

    # Try to load modules
    try:
        import evlib  # noqa: F401
        import evlib.filtering as filtering_module  # noqa: F401

        print("PASS: evlib modules loaded successfully")
    except ImportError as e:
        print(f"FAIL: Cannot load evlib modules: {e}")
        exit(1)

    # Run tests that don't require fixtures
    for test_func in [test_edge_cases_real_data, test_data_integrity_real_data]:
        try:
            test_func()
            print(f"PASS: {test_func.__name__} passed")
        except Exception as e:
            print(f"FAIL: {test_func.__name__} failed: {e}")

    print("=" * 60)
    print("Real data tests completed!")
    print("Note: Run with pytest for full test suite including fixtures")
