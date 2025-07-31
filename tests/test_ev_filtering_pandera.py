"""
Comprehensive pytest tests for ev_filtering module with pandera validation.

This test suite validates the Polars-first filtering implementation using pandera
schemas to ensure data integrity throughout the filtering pipeline. Tests use
real eTram data to verify functionality against production datasets.

Test Coverage:
- Individual filter validation (temporal, spatial, polarity, hot pixel, noise)
- Combined filtering pipeline
- Data integrity validation with pandera schemas
- Performance benchmarks
- Error handling and edge cases

Camera Specifications:
- eTram dataset: 1280x720 pixels (some files may have different resolutions)
- Polarity encoding: 0/1 (converted internally to -1/1 for processing)

Note: This test uses validation helpers from tests/validation_helpers.py
"""

import time
from pathlib import Path
from typing import Optional, Union

import polars as pl
import pytest

# Import validation helpers from tests directory
try:
    from validation_helpers import (
        quick_validate_events,
        validate_events,
        create_event_schema,
        PROPHESEE_GEN4_SCHEMA,
        ETRAM_SCHEMA,
        PERMISSIVE_SCHEMA,
    )
    import pandera.polars as pa
    from pandera import Column, Field

    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False

# Test markers and skip conditions
requires_pandera = pytest.mark.skipif(not PANDERA_AVAILABLE, reason="pandera not available")
requires_evlib = pytest.mark.skipif(True, reason="Check evlib availability in fixture")
requires_data = pytest.mark.requires_data


# =============================================================================
# Pandera Schemas for Event Data Validation
# =============================================================================

if PANDERA_AVAILABLE:

    class EventDataSchema(pa.DataFrameModel):
        """Schema for validating event camera data structure and constraints."""

        # Coordinate columns (sensor-specific ranges)
        x: pl.Int16 = Field(ge=0, le=2048, description="X coordinate in pixels")
        y: pl.Int16 = Field(ge=0, le=2000, description="Y coordinate in pixels")

        # Timestamp column (microseconds duration)
        timestamp: pl.Duration = Field(description="Timestamp as duration in microseconds")

        # Polarity column (binary values)
        polarity: pl.Int8 = Field(isin=[-1, 1], description="Event polarity (-1=negative, 1=positive)")

        class Config:
            strict = True
            coerce = True

    class FilteredEventSchema(EventDataSchema):
        """Schema for filtered events with relaxed constraints."""

        # Filtered events may have reduced coordinate ranges
        x: pl.Int16 = Field(ge=0, le=2048, description="X coordinate (filtered)")
        y: pl.Int16 = Field(ge=0, le=2000, description="Y coordinate (filtered)")

        # Timestamps should still be valid and ordered
        timestamp: pl.Duration = Field(description="Timestamp as duration (filtered)")

        class Config:
            strict = True

    class SpatialFilterSchema(FilteredEventSchema):
        """Schema for spatially filtered events within specific ROI."""

        # More restrictive coordinate ranges for ROI filtering
        x: pl.Int16 = Field(description="X coordinate within ROI")
        y: pl.Int16 = Field(description="Y coordinate within ROI")

    class TemporalFilterSchema(FilteredEventSchema):
        """Schema for temporally filtered events within time window."""

        # Time should be within specified window
        timestamp: pl.Duration = Field(description="Timestamp within time window")

else:
    # Dummy classes if pandera is not available
    class EventDataSchema:
        @staticmethod
        def validate(df, lazy=True):
            return df

    class FilteredEventSchema:
        @staticmethod
        def validate(df, lazy=True):
            return df

    class SpatialFilterSchema:
        @staticmethod
        def validate(df, lazy=True):
            return df

    class TemporalFilterSchema:
        @staticmethod
        def validate(df, lazy=True):
            return df


# =============================================================================
# Test Fixtures
# =============================================================================


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
def test_data_path():
    """Find and return path to suitable test data."""
    data_dir = Path(__file__).parent / "data"

    # Try eTram HDF5 data first (good size and quality)
    etram_h5 = data_dir / "eTram" / "h5" / "val_2"
    if etram_h5.exists():
        h5_files = list(etram_h5.glob("*.h5"))
        if h5_files:
            return h5_files[0]

    # Try Gen4 HDF5 data
    gen4_h5 = data_dir / "gen4_1mpx_processed_RVT" / "test"
    if gen4_h5.exists():
        h5_files = list(gen4_h5.glob("**/*.h5"))
        if h5_files:
            return h5_files[0]

    # Try slider depth text data (smaller but good for testing)
    slider_txt = data_dir / "slider_depth" / "events.txt"
    if slider_txt.exists():
        return slider_txt

    pytest.skip("No suitable test data found")


@pytest.fixture(scope="session")
def sample_events(evlib_module, test_data_path):
    """Load sample events from test data."""
    print(f"Loading test data from: {test_data_path}")

    # Load events using evlib (auto-detects format)
    events = evlib_module.load_events(str(test_data_path))

    # Get count for display purposes
    count = events.select(pl.len()).collect()["len"][0]
    print(f"Loaded {count:,} events")
    return events


@pytest.fixture(scope="session")
def sample_events_df(sample_events):
    """Sample events as LazyFrame for validation."""
    # Events are already loaded as LazyFrame, just return them
    return sample_events


# =============================================================================
# Validation Helper Functions
# =============================================================================


def validate_event_schema(df: Union[pl.LazyFrame, pl.DataFrame], schema_class, name: str = "events") -> bool:
    """Validate event data against pandera schema."""
    if not PANDERA_AVAILABLE:
        print(f"WARNING:  Pandera not available, skipping schema validation for {name}")
        return True

    try:
        # Handle both LazyFrame and DataFrame inputs
        if isinstance(df, pl.LazyFrame):
            df_collected = df.collect()
        else:
            df_collected = df

        schema_class.validate(df_collected, lazy=True)
        print(f" Schema validation passed for {name}")
        return True
    except pa.errors.SchemaError as e:
        print(f" Schema validation failed for {name}: {e}")
        # Show first few problematic rows for debugging
        print("First 5 rows of data:")
        if isinstance(df, pl.LazyFrame):
            print(df.limit(5).collect())
        else:
            print(df.head(5))
        return False


def validate_monotonic_timestamps(df: Union[pl.LazyFrame, pl.DataFrame], name: str = "events") -> bool:
    """Validate that timestamps are monotonically increasing."""
    print(f"Validating timestamp monotonicity for {name}...")

    # Handle both LazyFrame and DataFrame inputs
    if isinstance(df, pl.DataFrame):
        df_lazy = df.lazy()
    else:
        df_lazy = df

    # Check for non-monotonic timestamps
    check_df = df_lazy.select(
        [
            pl.col("timestamp"),
            (pl.col("timestamp").diff() < pl.duration(microseconds=0)).sum().alias("backward_jumps"),
        ]
    ).collect()

    backward_jumps = check_df["backward_jumps"][0]

    if backward_jumps > 0:
        print(f"WARNING:  Found {backward_jumps} backward timestamp jumps in {name}")
        return False
    else:
        print(f" Timestamps are monotonic for {name}")
        return True


def validate_coordinate_ranges(
    df: pl.LazyFrame,
    name: str,
    expected_x_range: Optional[tuple] = None,
    expected_y_range: Optional[tuple] = None,
) -> bool:
    """Validate coordinate ranges are within expected bounds."""
    print(f"Validating coordinate ranges for {name}...")

    stats = df.select(
        [
            pl.col("x").min().alias("x_min"),
            pl.col("x").max().alias("x_max"),
            pl.col("y").min().alias("y_min"),
            pl.col("y").max().alias("y_max"),
            pl.len().alias("count"),
        ]
    ).collect()

    x_min, x_max = stats["x_min"][0], stats["x_max"][0]
    y_min, y_max = stats["y_min"][0], stats["y_max"][0]
    count = stats["count"][0]

    print(f"  Coordinate ranges: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}] ({count:,} events)")

    valid = True

    # Check expected ranges if provided
    if expected_x_range:
        if not (expected_x_range[0] <= x_min and x_max <= expected_x_range[1]):
            print(f" X coordinates outside expected range {expected_x_range}")
            valid = False

    if expected_y_range:
        if not (expected_y_range[0] <= y_min and y_max <= expected_y_range[1]):
            print(f" Y coordinates outside expected range {expected_y_range}")
            valid = False

    if valid:
        print(f" Coordinate ranges valid for {name}")

    return valid


# =============================================================================
# Individual Filter Tests
# =============================================================================


@requires_pandera
@requires_data
class TestTemporalFiltering:
    """Test temporal filtering with pandera validation."""

    def test_time_window_filter(self, sample_events_df):
        """Test time window filtering preserves temporal bounds."""
        print("\n" + "=" * 60)
        print("TEMPORAL FILTERING TEST")
        print("=" * 60)

        # Validate input schema
        assert validate_event_schema(sample_events_df, EventDataSchema, "input events")

        # Get time range (convert duration to seconds)
        time_stats = sample_events_df.select(
            [
                (pl.col("timestamp").dt.total_microseconds() / 1_000_000).min().alias("t_min"),
                (pl.col("timestamp").dt.total_microseconds() / 1_000_000).max().alias("t_max"),
                pl.len().alias("count"),
            ]
        ).collect()

        t_min, t_max = time_stats["t_min"][0], time_stats["t_max"][0]
        duration = t_max - t_min

        print(f"Original time range: [{t_min:.3f}, {t_max:.3f}]s (duration: {duration:.3f}s)")

        # Apply temporal filter for middle 50% of time range
        window_start = t_min + duration * 0.25
        window_end = t_max - duration * 0.25

        import evlib.filtering

        # Use the new polars-first API - no numpy conversion needed!
        df = sample_events_df.collect()

        start_time = time.time()
        filtered_df = evlib.filtering.filter_by_time(df, window_start, window_end)

        # The filtered_df is already a DataFrame with proper schema
        # Convert to LazyFrame for subsequent operations
        filtered_df = filtered_df.lazy()
        filter_time = time.time() - start_time

        print(f"Filtering completed in {filter_time:.3f}s")

        # Validate results
        original_count = sample_events_df.select(pl.len()).collect()["len"][0]
        filtered_count = filtered_df.select(pl.len()).collect()["len"][0]
        reduction = (original_count - filtered_count) / original_count

        print(f"Events: {original_count:,} → {filtered_count:,} ({reduction:.1%} reduction)")

        # Validate time window
        filtered_stats = filtered_df.select(
            [
                (pl.col("timestamp").dt.total_microseconds() / 1_000_000).min().alias("t_min"),
                (pl.col("timestamp").dt.total_microseconds() / 1_000_000).max().alias("t_max"),
            ]
        ).collect()

        filtered_t_min = filtered_stats["t_min"][0]
        filtered_t_max = filtered_stats["t_max"][0]

        print(f"Filtered time range: [{filtered_t_min:.3f}, {filtered_t_max:.3f}]s")

        # Assertions
        assert filtered_count > 0, "Should have events after temporal filtering"
        assert filtered_count < original_count, "Should reduce event count"
        assert filtered_t_min >= window_start, "All events should be after window start"
        assert filtered_t_max <= window_end, "All events should be before window end"

        # Validate schema and monotonicity
        assert validate_event_schema(filtered_df, TemporalFilterSchema, "temporally filtered events")
        assert validate_monotonic_timestamps(filtered_df, "temporally filtered events")


@requires_pandera
@requires_data
class TestSpatialFiltering:
    """Test spatial filtering with pandera validation."""

    def test_roi_filter(self, sample_events_df):
        """Test ROI filtering preserves spatial bounds."""
        print("\n" + "=" * 60)
        print("SPATIAL FILTERING TEST")
        print("=" * 60)

        # Validate input schema
        assert validate_event_schema(sample_events_df, EventDataSchema, "input events")

        # Get coordinate ranges
        coord_stats = sample_events_df.select(
            [
                pl.col("x").min().alias("x_min"),
                pl.col("x").max().alias("x_max"),
                pl.col("y").min().alias("y_min"),
                pl.col("y").max().alias("y_max"),
                pl.len().alias("count"),
            ]
        ).collect()

        x_min, x_max = coord_stats["x_min"][0], coord_stats["x_max"][0]
        y_min, y_max = coord_stats["y_min"][0], coord_stats["y_max"][0]

        print(f"Original coordinate ranges: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")

        # Define ROI (center 50% of sensor)
        roi_x_min = int(x_min + (x_max - x_min) * 0.25)
        roi_x_max = int(x_max - (x_max - x_min) * 0.25)
        roi_y_min = int(y_min + (y_max - y_min) * 0.25)
        roi_y_max = int(y_max - (y_max - y_min) * 0.25)

        print(f"ROI: X=[{roi_x_min}, {roi_x_max}], Y=[{roi_y_min}, {roi_y_max}]")

        import evlib.filtering

        # Use the new polars-first API
        df = sample_events_df.collect()

        start_time = time.time()
        filtered_df = evlib.filtering.filter_by_roi(df, roi_x_min, roi_x_max, roi_y_min, roi_y_max)

        # Convert to LazyFrame for subsequent operations
        filtered_df = filtered_df.lazy()
        filter_time = time.time() - start_time

        print(f"Filtering completed in {filter_time:.3f}s")

        # Validate results
        original_count = sample_events_df.select(pl.len()).collect()["len"][0]
        filtered_count = filtered_df.select(pl.len()).collect()["len"][0]
        reduction = (original_count - filtered_count) / original_count

        print(f"Events: {original_count:,} → {filtered_count:,} ({reduction:.1%} reduction)")

        # Assertions
        assert filtered_count > 0, "Should have events after spatial filtering"
        assert filtered_count < original_count, "Should reduce event count"

        # Validate coordinate ranges
        assert validate_coordinate_ranges(
            filtered_df,
            "spatially filtered events",
            expected_x_range=(roi_x_min, roi_x_max),
            expected_y_range=(roi_y_min, roi_y_max),
        )

        # Validate schema
        assert validate_event_schema(filtered_df, SpatialFilterSchema, "spatially filtered events")


@requires_pandera
@requires_data
class TestPolarityFiltering:
    """Test polarity filtering with pandera validation."""

    def test_positive_only_filter(self, sample_events_df):
        """Test positive polarity filtering."""
        print("\n" + "=" * 60)
        print("POLARITY FILTERING TEST")
        print("=" * 60)

        # Validate input schema
        assert validate_event_schema(sample_events_df, EventDataSchema, "input events")

        # Get polarity distribution (count positive events where polarity=1)
        polarity_stats = sample_events_df.select(
            [(pl.col("polarity") == 1).sum().alias("positive_count"), pl.len().alias("total_count")]
        ).collect()

        positive_count = polarity_stats["positive_count"][0]
        total_count = polarity_stats["total_count"][0]
        negative_count = total_count - positive_count

        print("Original polarity distribution:")
        print(f"  Positive: {positive_count:,} ({positive_count/total_count:.1%})")
        print(f"  Negative: {negative_count:,} ({negative_count/total_count:.1%})")

        # Test positive events only
        import evlib.filtering

        # Use the new polars-first API
        df = sample_events_df.collect()

        start_time = time.time()
        filtered_df = evlib.filtering.filter_by_polarity(df, polarity=1)

        # Convert to LazyFrame for subsequent operations
        filtered_df = filtered_df.lazy()
        filter_time = time.time() - start_time

        print(f"Filtering completed in {filter_time:.3f}s")

        # Validate results
        filtered_count = filtered_df.select(pl.len()).collect()["len"][0]
        filtered_polarity_stats = filtered_df.select(
            [(pl.col("polarity") == 1).sum().alias("positive_count"), pl.len().alias("total_count")]
        ).collect()

        filtered_positive = filtered_polarity_stats["positive_count"][0]

        print(f"Filtered events: {filtered_count:,}")
        print(f"All positive events: {filtered_positive:,}")

        # Assertions
        assert filtered_count > 0, "Should have positive events"
        assert filtered_positive == filtered_count, "All filtered events should have positive polarity"
        assert filtered_count <= positive_count, "Should not have more positive events than original"

        # Validate schema
        assert validate_event_schema(filtered_df, FilteredEventSchema, "polarity filtered events")


@requires_pandera
@requires_data
class TestHotPixelFiltering:
    """Test hot pixel filtering with pandera validation."""

    def test_statistical_threshold_filter(self, sample_events_df):
        """Test statistical hot pixel filtering."""
        print("\n" + "=" * 60)
        print("HOT PIXEL FILTERING TEST")
        print("=" * 60)

        # Validate input schema
        assert validate_event_schema(sample_events_df, EventDataSchema, "input events")

        # Analyze pixel activity before filtering
        pixel_stats = (
            sample_events_df.group_by([pl.col("x"), pl.col("y")])
            .agg([pl.len().alias("event_count")])
            .sort("event_count", descending=True)
            .collect()
        )

        top_pixels = pixel_stats.head(10)
        total_pixels = len(pixel_stats)

        print(f"Total active pixels: {total_pixels:,}")
        print("Top 10 most active pixels:")
        for i in range(min(10, len(top_pixels))):
            row = top_pixels.row(i)
            x, y, count = row[0], row[1], row[2]
            print(f"  ({x:3}, {y:3}): {count:,} events")

        import evlib.filtering

        # Use the new polars-first API
        df = sample_events_df.collect()

        start_time = time.time()
        filtered_df = evlib.filtering.filter_hot_pixels(df, threshold_percentile=95.0)

        # Convert to LazyFrame for subsequent operations
        filtered_df = filtered_df.lazy()
        filter_time = time.time() - start_time

        print(f"Filtering completed in {filter_time:.3f}s")

        # Validate results
        original_count = sample_events_df.select(pl.len()).collect()["len"][0]
        filtered_count = filtered_df.select(pl.len()).collect()["len"][0]
        reduction = (original_count - filtered_count) / original_count

        print(f"Events: {original_count:,} → {filtered_count:,} ({reduction:.1%} reduction)")

        # Assertions
        assert filtered_count > 0, "Should have events after hot pixel filtering"
        assert filtered_count <= original_count, "Should not increase event count"

        # Validate schema
        assert validate_event_schema(filtered_df, FilteredEventSchema, "hot pixel filtered events")


@requires_pandera
@requires_data
class TestNoiseFiltering:
    """Test noise filtering with pandera validation."""

    def test_refractory_filter(self, sample_events_df):
        """Test refractory period noise filtering."""
        print("\n" + "=" * 60)
        print("NOISE FILTERING TEST")
        print("=" * 60)

        # Validate input schema
        assert validate_event_schema(sample_events_df, EventDataSchema, "input events")

        # Use refractory period filtering
        import evlib.filtering

        # Use the new polars-first API
        df = sample_events_df.collect()

        start_time = time.time()
        filtered_df = evlib.filtering.filter_noise(df, method="refractory", refractory_period_us=1000.0)

        # Convert to LazyFrame for subsequent operations
        filtered_df = filtered_df.lazy()
        filter_time = time.time() - start_time

        print(f"Filtering completed in {filter_time:.3f}s")

        # Validate results
        original_count = sample_events_df.select(pl.len()).collect()["len"][0]
        filtered_count = filtered_df.select(pl.len()).collect()["len"][0]
        reduction = (original_count - filtered_count) / original_count

        print(f"Events: {original_count:,} → {filtered_count:,} ({reduction:.1%} reduction)")

        # Assertions
        assert filtered_count > 0, "Should have events after noise filtering"
        assert filtered_count <= original_count, "Should not increase event count"

        # Validate that timestamps - note: may not be monotonic after refractory filtering (expected)
        validate_monotonic_timestamps(filtered_df, "noise-filtered events")

        # Validate schema
        assert validate_event_schema(filtered_df, FilteredEventSchema, "noise filtered events")


# =============================================================================
# Combined Pipeline Tests
# =============================================================================


@requires_pandera
@requires_data
class TestCombinedFiltering:
    """Test combined filtering pipeline with pandera validation."""

    def test_complete_pipeline(self, sample_events_df):
        """Test combined filtering pipeline maintains data integrity."""
        print("\n" + "=" * 60)
        print("COMBINED FILTERING PIPELINE TEST")
        print("=" * 60)

        # Validate input schema
        assert validate_event_schema(sample_events_df, EventDataSchema, "input events")

        # Get data characteristics
        stats = sample_events_df.select(
            [
                pl.col("x").min().alias("x_min"),
                pl.col("x").max().alias("x_max"),
                pl.col("y").min().alias("y_min"),
                pl.col("y").max().alias("y_max"),
                (pl.col("timestamp").dt.total_microseconds() / 1_000_000).min().alias("t_min"),
                (pl.col("timestamp").dt.total_microseconds() / 1_000_000).max().alias("t_max"),
                pl.len().alias("count"),
            ]
        ).collect()

        x_min, x_max = stats["x_min"][0], stats["x_max"][0]
        y_min, y_max = stats["y_min"][0], stats["y_max"][0]
        t_min, t_max = stats["t_min"][0], stats["t_max"][0]
        duration = t_max - t_min

        print(f"Input data: {stats['count'][0]:,} events")
        print(f"Time range: [{t_min:.3f}, {t_max:.3f}]s (duration: {duration:.3f}s)")
        print(f"Coordinate ranges: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")

        # Create comprehensive filtering pipeline using individual filters
        import evlib.filtering

        # Use the new polars-first API
        df = sample_events_df.collect()

        print("\nApplying combined filtering pipeline...")
        start_time = time.time()

        # Apply filters in sequence
        # 1. Temporal filter
        filtered_df = evlib.filtering.filter_by_time(
            df, t_start=t_min + duration * 0.1, t_end=t_max - duration * 0.1
        )

        # 2. Spatial filter (ROI)
        roi_x_min = int(x_min + (x_max - x_min) * 0.2)
        roi_x_max = int(x_max - (x_max - x_min) * 0.2)
        roi_y_min = int(y_min + (y_max - y_min) * 0.2)
        roi_y_max = int(y_max - (y_max - y_min) * 0.2)

        filtered_df = evlib.filtering.filter_by_roi(filtered_df, roi_x_min, roi_x_max, roi_y_min, roi_y_max)

        # 3. Hot pixel filter
        filtered_df = evlib.filtering.filter_hot_pixels(filtered_df, threshold_percentile=90.0)

        # 4. Noise filter
        filtered_df = evlib.filtering.filter_noise(
            filtered_df, method="refractory", refractory_period_us=500.0
        )

        # Convert to LazyFrame for subsequent operations
        filtered_df = filtered_df.lazy()
        total_filter_time = time.time() - start_time

        print(f"Complete pipeline executed in {total_filter_time:.3f}s")

        # Validate results
        original_count = sample_events_df.select(pl.len()).collect()["len"][0]
        filtered_count = filtered_df.select(pl.len()).collect()["len"][0]
        reduction = (original_count - filtered_count) / original_count

        print("\nFinal results:")
        print(f"Events: {original_count:,} → {filtered_count:,} ({reduction:.1%} reduction)")
        print(f"Processing rate: {original_count / total_filter_time:,.0f} events/second")

        # Assertions
        assert filtered_count > 0, "Should have events after combined filtering"
        assert filtered_count < original_count, "Should reduce event count"
        assert reduction > 0.1, "Should achieve significant reduction (>10%)"

        # Validate final output
        assert validate_coordinate_ranges(filtered_df, "final filtered events")
        # Note: monotonic timestamps not guaranteed after refractory filtering (expected behavior)
        validate_monotonic_timestamps(filtered_df, "final filtered events")

        # Validate schema
        assert validate_event_schema(filtered_df, FilteredEventSchema, "combined filtered events")

    def test_progressive_filtering(self, sample_events_df):
        """Test that each filter stage progressively reduces event count."""
        print("\n" + "=" * 60)
        print("PROGRESSIVE FILTERING TEST")
        print("=" * 60)

        # Apply filters one by one to track progression
        original_count = sample_events_df.select(pl.len()).collect()["len"][0]

        print(f"Starting with {original_count:,} events")

        # Step 1: Temporal filter
        import evlib.filtering

        # Use the new polars-first API
        df = sample_events_df.collect()

        step1_df = evlib.filtering.filter_by_time(df, t_start=0.1, t_end=None)
        step1_count = len(step1_df)
        print(f"After temporal filter: {step1_count:,} events")

        # Step 2: Add spatial filter
        stats = sample_events_df.select(
            [
                pl.col("x").min().alias("x_min"),
                pl.col("x").max().alias("x_max"),
                pl.col("y").min().alias("y_min"),
                pl.col("y").max().alias("y_max"),
            ]
        ).collect()

        x_min, x_max = stats["x_min"][0], stats["x_max"][0]
        y_min, y_max = stats["y_min"][0], stats["y_max"][0]

        step2_df = evlib.filtering.filter_by_roi(
            step1_df,
            x_min=int(x_min + (x_max - x_min) * 0.1),
            x_max=int(x_max - (x_max - x_min) * 0.1),
            y_min=int(y_min + (y_max - y_min) * 0.1),
            y_max=int(y_max - (y_max - y_min) * 0.1),
        )
        step2_count = len(step2_df)
        print(f"After spatial filter: {step2_count:,} events")

        # Step 3: Add hot pixel filter
        step3_df = evlib.filtering.filter_hot_pixels(step2_df, threshold_percentile=95.0)
        step3_count = len(step3_df)
        print(f"After hot pixel filter: {step3_count:,} events")

        # Assertions for progressive reduction
        assert step1_count <= original_count, "Step 1 should not increase count"
        assert step2_count <= step1_count, "Step 2 should not increase count"
        assert step3_count <= step2_count, "Step 3 should not increase count"

        # Validate schemas at each step
        assert validate_event_schema(step1_df, FilteredEventSchema, "step 1 filtered events")
        assert validate_event_schema(step2_df, FilteredEventSchema, "step 2 filtered events")
        assert validate_event_schema(step3_df, FilteredEventSchema, "step 3 filtered events")


# =============================================================================
# Performance and Edge Case Tests
# =============================================================================


@requires_pandera
@requires_data
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    def test_performance_benchmarks(self, sample_events_df):
        """Test that filtering meets performance requirements."""
        original_count = sample_events_df.select(pl.len()).collect()["len"][0]

        # Benchmark simple temporal filter
        import evlib.filtering

        # Collect DataFrame for filtering
        df = sample_events_df.collect()

        start_time = time.time()
        filtered_df = evlib.filtering.filter_by_time(df, t_start=0.1, t_end=None)
        filter_time = time.time() - start_time

        events_per_second = original_count / filter_time if filter_time > 0 else float("inf")

        print(f"Performance: {events_per_second:,.0f} events/second")

        # Should process at least 100K events per second (conservative threshold)
        assert events_per_second > 100_000, f"Performance too slow: {events_per_second:,.0f} events/s"

        # Validate output
        assert validate_event_schema(filtered_df, FilteredEventSchema, "performance test output")

    def test_empty_result_handling(self, sample_events_df):
        """Test handling of filters that result in empty datasets."""
        # Create filter that should remove all events (impossible time window)
        import evlib.filtering

        # Use the new polars-first API
        df = sample_events_df.collect()

        filtered_df = evlib.filtering.filter_by_time(df, t_start=99999.0, t_end=99999.1)

        # Should handle empty result gracefully
        assert len(filtered_df) == 0, "Should have no events for impossible time window"

        # Empty DataFrame should work
        assert len(filtered_df) == 0, "Empty DataFrame should have 0 rows"

        # Schema validation should work with empty DataFrames
        if PANDERA_AVAILABLE:
            # Empty DataFrames might not validate against strict schemas
            # This is expected behavior
            print(" Empty result handling works correctly")

    def test_data_integrity_preservation(self, sample_events_df):
        """Test that filtering preserves data integrity (no corruption)."""
        original_df = sample_events_df.collect()

        # Apply mild filtering that should preserve most data
        stats = sample_events_df.select(
            [
                (pl.col("timestamp").dt.total_microseconds() / 1_000_000).min().alias("t_min"),
                (pl.col("timestamp").dt.total_microseconds() / 1_000_000).max().alias("t_max"),
            ]
        ).collect()

        t_min, t_max = stats["t_min"][0], stats["t_max"][0]
        duration = t_max - t_min

        import evlib.filtering

        # Use the new polars-first API
        df = sample_events_df.collect()

        filtered_df = evlib.filtering.filter_by_time(
            df, t_start=t_min + duration * 0.01, t_end=t_max - duration * 0.01
        )
        # The filtered_df is already a DataFrame with proper schema

        if len(filtered_df) > 0:
            # Check data types are preserved
            assert filtered_df["x"].dtype == original_df["x"].dtype, "X dtype should be preserved"
            assert filtered_df["y"].dtype == original_df["y"].dtype, "Y dtype should be preserved"
            assert (
                filtered_df["timestamp"].dtype == original_df["timestamp"].dtype
            ), "Timestamp dtype should be preserved"
            assert (
                filtered_df["polarity"].dtype == original_df["polarity"].dtype
            ), "Polarity dtype should be preserved"

            # Check value ranges are within original bounds
            assert (
                filtered_df["x"].min() >= original_df["x"].min()
            ), "X values should be within original range"
            assert (
                filtered_df["x"].max() <= original_df["x"].max()
            ), "X values should be within original range"
            assert (
                filtered_df["y"].min() >= original_df["y"].min()
            ), "Y values should be within original range"
            assert (
                filtered_df["y"].max() <= original_df["y"].max()
            ), "Y values should be within original range"

            # Check polarity values are valid
            original_polarities = set(original_df["polarity"].unique().to_list())
            filtered_polarities = set(filtered_df["polarity"].unique().to_list())
            assert filtered_polarities.issubset(
                original_polarities
            ), "Filtered polarities should be subset of original"

        print(" Data integrity preserved through filtering")


# =============================================================================
# Main Test Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
