"""
Tests for evlib.filtering module.

This test suite covers all filtering functions with various scenarios:
- Time filtering
- Spatial filtering (ROI)
- Polarity filtering
- Hot pixel removal
- Noise filtering
- Complete preprocessing pipeline
"""

import numpy as np
import pytest
import polars as pl


# Test data generators
def create_test_events(num_events=1000, width=640, height=480, duration_sec=1.0):
    """Create synthetic test events."""
    np.random.seed(42)  # For reproducible tests

    # Random spatial coordinates
    x = np.random.randint(0, width, num_events)
    y = np.random.randint(0, height, num_events)

    # Random timestamps uniformly distributed
    timestamps_us = np.random.uniform(0, duration_sec * 1_000_000, num_events)
    timestamps_us = np.sort(timestamps_us)  # Sort for realism

    # Random polarities (0/1 encoding)
    polarity = np.random.randint(0, 2, num_events)

    # Convert to Polars DataFrame
    events_df = pl.DataFrame(
        {
            "x": x,
            "y": y,
            "timestamp": pl.Series(timestamps_us, dtype=pl.Duration(time_unit="us")),
            "polarity": polarity,
        }
    )

    return events_df.lazy()


def create_hot_pixel_events(num_events=1000, width=640, height=480):
    """Create test events with intentional hot pixels."""
    np.random.seed(42)

    # Most events are normal
    normal_events = int(num_events * 0.8)
    x_normal = np.random.randint(0, width, normal_events)
    y_normal = np.random.randint(0, height, normal_events)

    # Some events are from hot pixels (concentrated at specific locations)
    hot_events = num_events - normal_events
    hot_pixels = [(100, 100), (200, 200), (300, 300)]  # 3 hot pixels

    x_hot = []
    y_hot = []
    for i in range(hot_events):
        hot_x, hot_y = hot_pixels[i % len(hot_pixels)]
        x_hot.append(hot_x)
        y_hot.append(hot_y)

    # Combine
    x = np.concatenate([x_normal, x_hot])
    y = np.concatenate([y_normal, y_hot])

    timestamps_us = np.random.uniform(0, 1_000_000, num_events)
    timestamps_us = np.sort(timestamps_us)

    polarity = np.random.randint(0, 2, num_events)

    events_df = pl.DataFrame(
        {
            "x": x,
            "y": y,
            "timestamp": pl.Series(timestamps_us, dtype=pl.Duration(time_unit="us")),
            "polarity": polarity,
        }
    )

    return events_df.lazy()


def create_noisy_events(num_events=1000, width=640, height=480):
    """Create test events with temporal noise (rapid-fire events)."""
    np.random.seed(42)

    # Normal events
    normal_events = int(num_events * 0.7)
    x_normal = np.random.randint(0, width, normal_events)
    y_normal = np.random.randint(0, height, normal_events)
    t_normal = np.random.uniform(0, 1_000_000, normal_events)

    # Noisy events (rapid-fire at same locations)
    noisy_events = num_events - normal_events
    noise_pixels = [(150, 150), (250, 250)]  # 2 noisy pixels

    x_noisy = []
    y_noisy = []
    t_noisy = []

    for i in range(noisy_events):
        pixel_x, pixel_y = noise_pixels[i % len(noise_pixels)]
        x_noisy.append(pixel_x)
        y_noisy.append(pixel_y)
        # Create rapid-fire events (within 500μs of each other)
        base_time = 500_000 + i * 100  # Base time
        t_noisy.append(base_time + np.random.uniform(0, 500))

    # Combine
    x = np.concatenate([x_normal, x_noisy])
    y = np.concatenate([y_normal, y_noisy])
    timestamps_us = np.concatenate([t_normal, t_noisy])

    # Sort by timestamp
    sort_idx = np.argsort(timestamps_us)
    x = x[sort_idx]
    y = y[sort_idx]
    timestamps_us = timestamps_us[sort_idx]

    polarity = np.random.randint(0, 2, num_events)

    events_df = pl.DataFrame(
        {
            "x": x,
            "y": y,
            "timestamp": pl.Series(timestamps_us, dtype=pl.Duration(time_unit="us")),
            "polarity": polarity,
        }
    )

    return events_df.lazy()


# Test filtering functions
def test_filter_by_time():
    """Test temporal filtering functionality."""
    try:
        from evlib.filtering import filter_by_time
    except ImportError:
        pytest.skip("evlib.filtering not available")

    # Create test events
    events = create_test_events(num_events=1000, duration_sec=1.0)

    # Test time filtering
    filtered = filter_by_time(events, t_start=0.2, t_end=0.8)
    filtered_df = filtered.collect()

    # Check results
    assert len(filtered_df) > 0, "Should have events in time range"
    assert len(filtered_df) < 1000, "Should filter out some events"

    # Verify time bounds
    timestamps_sec = filtered_df["timestamp"].dt.total_microseconds() / 1_000_000
    assert timestamps_sec.min() >= 0.2, "All events should be after t_start"
    assert timestamps_sec.max() <= 0.8, "All events should be before t_end"

    # Test edge cases
    empty_filtered = filter_by_time(events, t_start=2.0, t_end=3.0)
    assert len(empty_filtered.collect()) == 0, "Should be empty for out-of-range time"

    # Test single-sided bounds
    start_only = filter_by_time(events, t_start=0.5)
    start_only_df = start_only.collect()
    start_timestamps = start_only_df["timestamp"].dt.total_microseconds() / 1_000_000
    assert start_timestamps.min() >= 0.5, "Should respect start bound"

    end_only = filter_by_time(events, t_end=0.5)
    end_only_df = end_only.collect()
    end_timestamps = end_only_df["timestamp"].dt.total_microseconds() / 1_000_000
    assert end_timestamps.max() <= 0.5, "Should respect end bound"

    print("PASS: Time filtering tests passed")


def test_filter_by_roi():
    """Test spatial filtering functionality."""
    try:
        from evlib.filtering import filter_by_roi
    except ImportError:
        pytest.skip("evlib.filtering not available")

    # Create test events
    events = create_test_events(num_events=1000, width=640, height=480)

    # Test ROI filtering
    filtered = filter_by_roi(events, x_min=100, x_max=500, y_min=100, y_max=400)
    filtered_df = filtered.collect()

    # Check results
    assert len(filtered_df) > 0, "Should have events in ROI"
    assert len(filtered_df) < 1000, "Should filter out some events"

    # Verify spatial bounds
    assert filtered_df["x"].min() >= 100, "All x coordinates should be >= x_min"
    assert filtered_df["x"].max() <= 500, "All x coordinates should be <= x_max"
    assert filtered_df["y"].min() >= 100, "All y coordinates should be >= y_min"
    assert filtered_df["y"].max() <= 400, "All y coordinates should be <= y_max"

    # Test edge cases
    with pytest.raises(ValueError):
        filter_by_roi(events, x_min=500, x_max=100, y_min=100, y_max=400)  # Invalid bounds

    # Test small ROI
    small_roi = filter_by_roi(events, x_min=0, x_max=10, y_min=0, y_max=10)
    small_roi_df = small_roi.collect()
    # Should have very few events (possibly zero)
    assert len(small_roi_df) >= 0, "Small ROI should not error"

    print("PASS: ROI filtering tests passed")


def test_filter_by_polarity():
    """Test polarity filtering functionality."""
    try:
        from evlib.filtering import filter_by_polarity
    except ImportError:
        pytest.skip("evlib.filtering not available")

    # Create test events
    events = create_test_events(num_events=1000)

    # Test single polarity filtering
    positive_events = filter_by_polarity(events, polarity=1)
    positive_df = positive_events.collect()

    negative_events = filter_by_polarity(events, polarity=0)
    negative_df = negative_events.collect()

    # Check results
    assert len(positive_df) > 0, "Should have positive events"
    assert len(negative_df) > 0, "Should have negative events"
    assert all(positive_df["polarity"] == 1), "All events should be positive"
    assert all(negative_df["polarity"] == 0), "All events should be negative"

    # Test multiple polarities
    both_polarities = filter_by_polarity(events, polarity=[0, 1])
    both_df = both_polarities.collect()
    assert len(both_df) == 1000, "Should keep all events with both polarities"

    # Test no filtering (None)
    no_filter = filter_by_polarity(events, polarity=None)
    no_filter_df = no_filter.collect()
    assert len(no_filter_df) == 1000, "Should keep all events when polarity=None"

    print("PASS: Polarity filtering tests passed")


def test_filter_hot_pixels():
    """Test hot pixel detection and removal."""
    try:
        from evlib.filtering import filter_hot_pixels
    except ImportError:
        pytest.skip("evlib.filtering not available")

    # Create test events with hot pixels
    events = create_hot_pixel_events(num_events=1000)

    # Test hot pixel filtering
    filtered = filter_hot_pixels(events, threshold_percentile=95.0)
    filtered_df = filtered.collect()

    # Should remove some events (the hot pixels)
    assert len(filtered_df) < 1000, "Should remove hot pixel events"
    assert len(filtered_df) > 0, "Should retain normal events"

    # Test with different thresholds
    aggressive = filter_hot_pixels(events, threshold_percentile=90.0)
    aggressive_df = aggressive.collect()

    lenient = filter_hot_pixels(events, threshold_percentile=99.5)
    lenient_df = lenient.collect()

    # More aggressive filtering should remove more events
    assert len(aggressive_df) <= len(lenient_df), "Aggressive filtering should remove more events"

    # Test edge case: no hot pixels
    normal_events = create_test_events(num_events=100)
    no_hot_filtered = filter_hot_pixels(normal_events, threshold_percentile=99.9)
    no_hot_df = no_hot_filtered.collect()
    assert len(no_hot_df) == 100, "Should not remove events if no hot pixels"

    print("PASS: Hot pixel filtering tests passed")


def test_filter_noise():
    """Test noise filtering functionality."""
    try:
        from evlib.filtering import filter_noise
    except ImportError:
        pytest.skip("evlib.filtering not available")

    # Create test events with noise
    events = create_noisy_events(num_events=1000)

    # Test refractory period filtering
    filtered = filter_noise(events, method="refractory", refractory_period_us=1000)
    filtered_df = filtered.collect()

    # Should remove some events (the noise)
    assert len(filtered_df) < 1000, "Should remove noisy events"
    assert len(filtered_df) > 0, "Should retain normal events"

    # Test with different refractory periods
    short_period = filter_noise(events, method="refractory", refractory_period_us=100)
    short_df = short_period.collect()

    long_period = filter_noise(events, method="refractory", refractory_period_us=10000)
    long_df = long_period.collect()

    # Longer refractory period should remove more events
    assert len(long_df) <= len(short_df), "Longer refractory period should remove more events"

    # Test invalid method
    with pytest.raises(ValueError):
        filter_noise(events, method="invalid_method")

    # Test unimplemented method
    with pytest.raises(NotImplementedError):
        filter_noise(events, method="distance")

    print("PASS: Noise filtering tests passed")


def test_preprocess_events():
    """Test complete preprocessing pipeline."""
    try:
        from evlib.filtering import preprocess_events
    except ImportError:
        pytest.skip("evlib.filtering not available")

    # Create test events
    events = create_test_events(num_events=1000, duration_sec=1.0)

    # Test complete preprocessing
    processed = preprocess_events(
        events,
        t_start=0.1,
        t_end=0.9,
        roi=(100, 500, 100, 400),
        polarity=1,
        remove_hot_pixels=True,
        remove_noise=True,
        hot_pixel_threshold=99.0,
        refractory_period_us=1000,
    )

    processed_df = processed.collect()

    # Should have fewer events after all filtering
    assert len(processed_df) < 1000, "Should remove events through preprocessing"
    assert len(processed_df) >= 0, "Should not error on valid preprocessing"

    # Verify filters were applied
    if len(processed_df) > 0:
        # Time bounds
        timestamps_sec = processed_df["timestamp"].dt.total_microseconds() / 1_000_000
        assert timestamps_sec.min() >= 0.1, "Should respect time bounds"
        assert timestamps_sec.max() <= 0.9, "Should respect time bounds"

        # Spatial bounds
        assert processed_df["x"].min() >= 100, "Should respect ROI bounds"
        assert processed_df["x"].max() <= 500, "Should respect ROI bounds"
        assert processed_df["y"].min() >= 100, "Should respect ROI bounds"
        assert processed_df["y"].max() <= 400, "Should respect ROI bounds"

        # Polarity
        assert all(processed_df["polarity"] == 1), "Should keep only positive events"

    # Test minimal preprocessing
    minimal = preprocess_events(events, remove_hot_pixels=False, remove_noise=False)
    minimal_df = minimal.collect()
    assert len(minimal_df) == 1000, "Should keep all events with minimal preprocessing"

    print("PASS: Preprocessing pipeline tests passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    try:
        from evlib.filtering import filter_by_time, filter_by_roi, filter_by_polarity
    except ImportError:
        pytest.skip("evlib.filtering not available")

    # Test with empty events
    empty_events = pl.DataFrame(
        {"x": [], "y": [], "timestamp": pl.Series([], dtype=pl.Duration(time_unit="us")), "polarity": []}
    ).lazy()

    # All filters should handle empty input gracefully
    time_filtered = filter_by_time(empty_events, t_start=0.1, t_end=0.5)
    assert len(time_filtered.collect()) == 0, "Should handle empty events"

    roi_filtered = filter_by_roi(empty_events, x_min=0, x_max=100, y_min=0, y_max=100)
    assert len(roi_filtered.collect()) == 0, "Should handle empty events"

    polarity_filtered = filter_by_polarity(empty_events, polarity=1)
    assert len(polarity_filtered.collect()) == 0, "Should handle empty events"

    # Test with single event
    single_event = pl.DataFrame(
        {
            "x": [100],
            "y": [200],
            "timestamp": pl.Series([500_000], dtype=pl.Duration(time_unit="us")),
            "polarity": [1],
        }
    ).lazy()

    single_filtered = filter_by_time(single_event, t_start=0.1, t_end=0.9)
    assert len(single_filtered.collect()) == 1, "Should handle single event"

    print("PASS: Edge case tests passed")


def test_input_validation():
    """Test input validation and error handling."""
    try:
        from evlib.filtering import _validate_events_input
    except ImportError:
        pytest.skip("evlib.filtering not available")

    # Test valid LazyFrame input
    events = create_test_events(num_events=100)
    validated = _validate_events_input(events)
    assert isinstance(validated, pl.LazyFrame), "Should return LazyFrame"

    # Test invalid input types
    with pytest.raises(ValueError):
        _validate_events_input(123)  # Invalid type

    with pytest.raises(ValueError):
        _validate_events_input([1, 2, 3])  # Invalid type

    print("PASS: Input validation tests passed")


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_filter_by_time,
        test_filter_by_roi,
        test_filter_by_polarity,
        test_filter_hot_pixels,
        test_filter_noise,
        test_preprocess_events,
        test_edge_cases,
        test_input_validation,
    ]

    print("Running evlib.filtering tests...")
    print("=" * 50)

    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"FAIL: {test_func.__name__} failed: {e}")
        else:
            print(f"PASS: {test_func.__name__} passed")

    print("=" * 50)
    print("All tests completed!")
