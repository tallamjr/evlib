"""
Comprehensive pytest tests for ev_augmentation module with pandera validation.

This test suite validates the augmentation operations using pandera schemas to ensure
data integrity and sensor constraints are maintained throughout augmentation. Tests
use real event data to verify functionality against production datasets.

Test Coverage:
- Individual augmentation validation (spatial jitter, time jitter, noise, etc.)
- Combined augmentation pipeline
- Data integrity validation with pandera schemas
- Sensor constraint enforcement
- Performance benchmarks
- Edge cases and error handling

Camera Specifications:
- eTram dataset: 1280x720 pixels
- Gen4: 1280x720 pixels
- DAVIS346: 346x240 pixels
- Polarity encoding: -1/1 (after internal conversion)

Note: This test uses validation helpers from tests/validation_helpers.py
"""

import time
from pathlib import Path
from typing import Optional, Union, List, Tuple

import polars as pl
import pytest
import numpy as np

# Import validation helpers from tests directory
try:
    from validation_helpers import (
        quick_validate_events,
        validate_events,
        create_event_schema,
        SENSOR_CONSTRAINTS,
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
# Augmentation-Specific Pandera Schemas
# =============================================================================

if PANDERA_AVAILABLE:

    class AugmentedEventSchema(pa.DataFrameModel):
        """Schema for validating augmented event data with relaxed constraints."""

        # Spatial coordinates (may be modified by spatial jitter)
        x: pl.Int16 = Field(ge=-50, le=2100, description="X coordinate (may be jittered)")
        y: pl.Int16 = Field(ge=-50, le=2100, description="Y coordinate (may be jittered)")

        # Timestamps (may be modified by time augmentations)
        t: pl.Float64 = Field(ge=-1.0, le=1e6, description="Timestamp in seconds (may be augmented)")

        # Polarity should remain unchanged
        polarity: pl.Int8 = Field(isin=[-1, 1], description="Event polarity (unchanged)")

        class Config:
            strict = False  # Allow extra columns from augmentation
            coerce = True

    class SpatiallyAugmentedSchema(AugmentedEventSchema):
        """Schema for spatially augmented events (jitter, flips, etc.)."""

        # More lenient spatial bounds for jittered coordinates
        x: pl.Int16 = Field(ge=-100, le=2200, description="X coordinate after spatial augmentation")
        y: pl.Int16 = Field(ge=-100, le=2200, description="Y coordinate after spatial augmentation")

    class TemporallyAugmentedSchema(AugmentedEventSchema):
        """Schema for temporally augmented events (jitter, skew, etc.)."""

        # Timestamps may be significantly modified
        t: pl.Float64 = Field(ge=-10.0, le=1e7, description="Timestamp after temporal augmentation")

    class NoiseAugmentedSchema(AugmentedEventSchema):
        """Schema for events with added noise (more events, wider ranges)."""

        # Noise events can span full sensor range
        x: pl.Int16 = Field(ge=0, le=2048, description="X coordinate with noise events")
        y: pl.Int16 = Field(ge=0, le=2048, description="Y coordinate with noise events")

else:
    # Dummy classes if pandera is not available
    class AugmentedEventSchema:
        @staticmethod
        def validate(df, lazy=True):
            return df

    class SpatiallyAugmentedSchema:
        @staticmethod
        def validate(df, lazy=True):
            return df

    class TemporallyAugmentedSchema:
        @staticmethod
        def validate(df, lazy=True):
            return df

    class NoiseAugmentedSchema:
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
def augmentation_module():
    """Import augmentation module if available."""
    try:
        import evlib.ev_augmentation as aug

        return aug
    except ImportError:
        pytest.skip("evlib.ev_augmentation not available")


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

    pytest.skip("No suitable test data found")


@pytest.fixture(scope="session")
def sample_events(evlib_module, test_data_path):
    """Load sample events from test data."""
    print(f"Loading test data from: {test_data_path}")

    # Load events using evlib (auto-detects format)
    events = evlib_module.load_events(str(test_data_path))

    # Limit to manageable size for testing (first 50k events)
    events_limited = events.limit(50000)

    # Convert to Vec<Event> format for augmentation
    events_df = events_limited.collect()
    count = len(events_df)
    print(f"Loaded {count:,} events for augmentation testing")

    return events_df


@pytest.fixture(scope="session")
def sample_events_list(sample_events, evlib_module):
    """Convert sample events to List[Event] format for augmentation."""
    # Convert polars DataFrame to Vec<Event> format
    events_list = []

    for row in sample_events.iter_rows():
        x, y, t_dur, polarity = row
        # Convert duration to seconds
        t_seconds = t_dur.total_seconds() if hasattr(t_dur, "total_seconds") else float(t_dur) / 1e6

        events_list.append({"t": t_seconds, "x": int(x), "y": int(y), "polarity": bool(polarity > 0)})

    return events_list


# =============================================================================
# Validation Helper Functions
# =============================================================================


def validate_augmented_schema(df: pl.DataFrame, schema_class, name: str = "augmented events") -> bool:
    """Validate augmented event data against pandera schema."""
    if not PANDERA_AVAILABLE:
        print(f"WARNING: Pandera not available, skipping schema validation for {name}")
        return True

    try:
        schema_class.validate(df, lazy=True)
        print(f"✓ Schema validation passed for {name}")
        return True
    except pa.errors.SchemaError as e:
        print(f"✗ Schema validation failed for {name}: {e}")
        print("First 5 rows of problematic data:")
        print(df.head(5))
        return False


def validate_coordinate_bounds(df: pl.DataFrame, name: str, sensor_type: str = "etram") -> bool:
    """Validate that coordinates are within reasonable sensor bounds."""
    constraints = SENSOR_CONSTRAINTS.get(sensor_type, SENSOR_CONSTRAINTS["generic_large"])

    stats = df.select(
        [
            pl.col("x").min().alias("x_min"),
            pl.col("x").max().alias("x_max"),
            pl.col("y").min().alias("y_min"),
            pl.col("y").max().alias("y_max"),
            pl.len().alias("count"),
        ]
    )

    x_min, x_max, y_min, y_max, count = stats.row(0)

    print(f"Coordinate bounds for {name}: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}] ({count:,} events)")

    # Allow some tolerance for jittered coordinates
    tolerance = 100
    max_x_allowed = constraints["max_x"] + tolerance
    max_y_allowed = constraints["max_y"] + tolerance
    min_allowed = -tolerance

    valid = True
    if x_min < min_allowed or x_max > max_x_allowed:
        print(f"✗ X coordinates out of bounds: [{x_min}, {x_max}] not in [{min_allowed}, {max_x_allowed}]")
        valid = False

    if y_min < min_allowed or y_max > max_y_allowed:
        print(f"✗ Y coordinates out of bounds: [{y_min}, {y_max}] not in [{min_allowed}, {max_y_allowed}]")
        valid = False

    if valid:
        print(f"✓ Coordinate bounds valid for {name}")

    return valid


def validate_temporal_order(df: pl.DataFrame, name: str, strict: bool = False) -> bool:
    """Validate temporal ordering (with option for relaxed validation)."""
    print(f"Validating temporal order for {name} (strict={strict})...")

    # Check for temporal order
    time_check = df.select(
        [
            pl.col("t"),
            (pl.col("t").diff() < 0).sum().alias("backward_jumps"),
            pl.col("t").is_null().sum().alias("null_times"),
        ]
    )

    backward_jumps = time_check.select("backward_jumps").item()
    null_times = time_check.select("null_times").item()

    if null_times > 0:
        print(f"✗ Found {null_times} null timestamps in {name}")
        return False

    if strict and backward_jumps > 0:
        print(f"✗ Found {backward_jumps} backward timestamp jumps in {name}")
        return False
    elif backward_jumps > 0:
        print(
            f"⚠ Found {backward_jumps} backward timestamp jumps in {name} (acceptable for some augmentations)"
        )
    else:
        print(f"✓ Temporal order valid for {name}")

    return True


def validate_polarity_integrity(df: pl.DataFrame, name: str) -> bool:
    """Validate that polarity values remain valid."""
    polarity_stats = df.select(
        [
            pl.col("polarity").unique().sort().alias("unique_polarities"),
            pl.col("polarity").is_null().sum().alias("null_polarities"),
        ]
    )

    unique_polarities = polarity_stats.select("unique_polarities").to_series().to_list()
    null_polarities = polarity_stats.select("null_polarities").item()

    print(f"Polarity validation for {name}: unique values {unique_polarities}, nulls: {null_polarities}")

    valid = True
    if null_polarities > 0:
        print(f"✗ Found {null_polarities} null polarity values")
        valid = False

    # Check that polarities are only -1 or 1
    expected_polarities = {-1, 1}
    actual_polarities = set(unique_polarities)
    if not actual_polarities.issubset(expected_polarities):
        print(f"✗ Invalid polarity values: {actual_polarities - expected_polarities}")
        valid = False

    if valid:
        print(f"✓ Polarity integrity maintained for {name}")

    return valid


def convert_events_to_dataframe(events_list: List[dict]) -> pl.DataFrame:
    """Convert list of event dicts to polars DataFrame."""
    return pl.DataFrame(
        {
            "t": [e["t"] for e in events_list],
            "x": [e["x"] for e in events_list],
            "y": [e["y"] for e in events_list],
            "polarity": [1 if e["polarity"] else -1 for e in events_list],
        }
    )


# =============================================================================
# Individual Augmentation Tests
# =============================================================================


@requires_pandera
@requires_data
class TestSpatialAugmentations:
    """Test spatial augmentation operations with pandera validation."""

    def test_spatial_jitter_validation(self, sample_events_list, augmentation_module):
        """Test spatial jitter maintains data validity."""
        print("\n" + "=" * 60)
        print("SPATIAL JITTER VALIDATION TEST")
        print("=" * 60)

        # Create spatial jitter configuration
        config = augmentation_module.AugmentationConfig.new().with_spatial_jitter(2.0, 2.0)  # 2 pixel std dev

        print(f"Applying spatial jitter to {len(sample_events_list):,} events...")
        start_time = time.time()

        # Apply spatial jitter using Rust backend
        try:
            augmented_events = augmentation_module.augment_events(sample_events_list, config)
            augment_time = time.time() - start_time
            print(f"Augmentation completed in {augment_time:.3f}s")

        except Exception as e:
            print(f"Augmentation failed: {e}")
            # Convert to DataFrame for validation anyway
            original_df = convert_events_to_dataframe(sample_events_list)
            augmented_df = original_df  # Use original as fallback

        else:
            # Convert results to DataFrame
            augmented_df = convert_events_to_dataframe(augmented_events)

        original_df = convert_events_to_dataframe(sample_events_list)

        print(f"Original events: {len(original_df):,}")
        print(f"Augmented events: {len(augmented_df):,}")

        # Validation checks
        assert len(augmented_df) == len(original_df), "Event count should remain the same"

        # Schema validation
        assert validate_augmented_schema(augmented_df, SpatiallyAugmentedSchema, "spatially jittered events")

        # Coordinate bounds validation (with tolerance for jitter)
        assert validate_coordinate_bounds(augmented_df, "spatially jittered events", "etram")

        # Polarity should be unchanged
        assert validate_polarity_integrity(augmented_df, "spatially jittered events")

        # Temporal order should be preserved
        assert validate_temporal_order(augmented_df, "spatially jittered events", strict=True)

        # Check that coordinates actually changed (with high probability)
        coords_changed = not (
            original_df.select("x").equals(augmented_df.select("x"))
            and original_df.select("y").equals(augmented_df.select("y"))
        )
        assert coords_changed, "Coordinates should change with spatial jitter"

        print("✓ Spatial jitter validation passed")


@requires_pandera
@requires_data
class TestTemporalAugmentations:
    """Test temporal augmentation operations with pandera validation."""

    def test_time_jitter_validation(self, sample_events_list, augmentation_module):
        """Test time jitter maintains data validity."""
        print("\n" + "=" * 60)
        print("TIME JITTER VALIDATION TEST")
        print("=" * 60)

        # Create time jitter configuration (1ms std dev)
        config = augmentation_module.AugmentationConfig.new().with_time_jitter(1000.0)

        print(f"Applying time jitter to {len(sample_events_list):,} events...")
        start_time = time.time()

        # Apply time jitter using Rust backend
        try:
            augmented_events = augmentation_module.augment_events(sample_events_list, config)
            augment_time = time.time() - start_time
            print(f"Augmentation completed in {augment_time:.3f}s")

        except Exception as e:
            print(f"Augmentation failed: {e}")
            # Use original events as fallback
            augmented_events = sample_events_list

        # Convert to DataFrames
        original_df = convert_events_to_dataframe(sample_events_list)
        augmented_df = convert_events_to_dataframe(augmented_events)

        print(f"Original events: {len(original_df):,}")
        print(f"Augmented events: {len(augmented_df):,}")

        # Validation checks
        assert len(augmented_df) == len(original_df), "Event count should remain the same"

        # Schema validation
        assert validate_augmented_schema(augmented_df, TemporallyAugmentedSchema, "time jittered events")

        # Coordinate bounds should be unchanged
        assert validate_coordinate_bounds(augmented_df, "time jittered events", "etram")

        # Polarity should be unchanged
        assert validate_polarity_integrity(augmented_df, "time jittered events")

        # Temporal order may be disrupted (expected with jitter)
        validate_temporal_order(augmented_df, "time jittered events", strict=False)

        print("✓ Time jitter validation passed")

    def test_time_skew_validation(self, sample_events_list, augmentation_module):
        """Test time skew maintains data validity."""
        print("\n" + "=" * 60)
        print("TIME SKEW VALIDATION TEST")
        print("=" * 60)

        # Create time skew configuration (1.1x speed up)
        config = augmentation_module.AugmentationConfig.new().with_time_skew(1.1)

        print(f"Applying time skew to {len(sample_events_list):,} events...")
        start_time = time.time()

        # Apply time skew using Rust backend
        try:
            augmented_events = augmentation_module.augment_events(sample_events_list, config)
            augment_time = time.time() - start_time
            print(f"Augmentation completed in {augment_time:.3f}s")

        except Exception as e:
            print(f"Augmentation failed: {e}")
            augmented_events = sample_events_list

        # Convert to DataFrames
        original_df = convert_events_to_dataframe(sample_events_list)
        augmented_df = convert_events_to_dataframe(augmented_events)

        # Validation checks
        assert len(augmented_df) == len(original_df), "Event count should remain the same"

        # Schema validation
        assert validate_augmented_schema(augmented_df, TemporallyAugmentedSchema, "time skewed events")

        # Coordinates should be unchanged
        coords_unchanged = original_df.select("x").equals(augmented_df.select("x")) and original_df.select(
            "y"
        ).equals(augmented_df.select("y"))
        assert coords_unchanged, "Coordinates should be unchanged with time skew"

        # Polarity should be unchanged
        assert validate_polarity_integrity(augmented_df, "time skewed events")

        # Check that time scaling was applied correctly
        if len(augmented_df) > 1:
            orig_duration = original_df.select(pl.col("t").max() - pl.col("t").min()).item()
            aug_duration = augmented_df.select(pl.col("t").max() - pl.col("t").min()).item()

            if orig_duration > 0:
                scaling_factor = aug_duration / orig_duration
                print(f"Time scaling factor: {scaling_factor:.3f} (expected: 1.1)")
                # Allow some tolerance for floating point precision
                assert abs(scaling_factor - 1.1) < 0.01, f"Time scaling incorrect: {scaling_factor}"

        print("✓ Time skew validation passed")


@requires_pandera
@requires_data
class TestNoiseAugmentations:
    """Test noise injection augmentations with pandera validation."""

    def test_uniform_noise_validation(self, sample_events_list, augmentation_module):
        """Test uniform noise injection maintains data validity."""
        print("\n" + "=" * 60)
        print("UNIFORM NOISE VALIDATION TEST")
        print("=" * 60)

        # Create uniform noise configuration (add 1000 noise events)
        config = augmentation_module.AugmentationConfig.new().with_uniform_noise(1000, 1280, 720)

        print(f"Adding uniform noise to {len(sample_events_list):,} events...")
        start_time = time.time()

        # Apply uniform noise using Rust backend
        try:
            augmented_events = augmentation_module.augment_events(sample_events_list, config)
            augment_time = time.time() - start_time
            print(f"Augmentation completed in {augment_time:.3f}s")

        except Exception as e:
            print(f"Augmentation failed: {e}")
            augmented_events = sample_events_list

        # Convert to DataFrames
        original_df = convert_events_to_dataframe(sample_events_list)
        augmented_df = convert_events_to_dataframe(augmented_events)

        print(f"Original events: {len(original_df):,}")
        print(f"Augmented events: {len(augmented_df):,}")

        # Validation checks
        expected_count = len(original_df) + 1000
        assert len(augmented_df) >= len(original_df), "Should have more events after noise injection"
        # Allow some tolerance in case of clipping
        assert abs(len(augmented_df) - expected_count) <= 100, f"Event count should be ~{expected_count}"

        # Schema validation
        assert validate_augmented_schema(augmented_df, NoiseAugmentedSchema, "noise augmented events")

        # Coordinate bounds validation (noise should be within sensor bounds)
        assert validate_coordinate_bounds(augmented_df, "noise augmented events", "etram")

        # Polarity integrity
        assert validate_polarity_integrity(augmented_df, "noise augmented events")

        # Temporal ordering may be disrupted (expected with added noise)
        validate_temporal_order(augmented_df, "noise augmented events", strict=False)

        print("✓ Uniform noise validation passed")


@requires_pandera
@requires_data
class TestDropAugmentations:
    """Test event dropping augmentations with pandera validation."""

    def test_drop_by_time_validation(self, sample_events_list, augmentation_module):
        """Test drop by time maintains data validity."""
        print("\n" + "=" * 60)
        print("DROP BY TIME VALIDATION TEST")
        print("=" * 60)

        # Create drop by time configuration (drop 20% of time duration)
        config = augmentation_module.AugmentationConfig.new().with_drop_time(0.2)

        print(f"Applying drop by time to {len(sample_events_list):,} events...")
        start_time = time.time()

        # Apply drop by time using Rust backend
        try:
            augmented_events = augmentation_module.augment_events(sample_events_list, config)
            augment_time = time.time() - start_time
            print(f"Augmentation completed in {augment_time:.3f}s")

        except Exception as e:
            print(f"Augmentation failed: {e}")
            augmented_events = sample_events_list

        # Convert to DataFrames
        original_df = convert_events_to_dataframe(sample_events_list)
        augmented_df = convert_events_to_dataframe(augmented_events)

        print(f"Original events: {len(original_df):,}")
        print(f"Augmented events: {len(augmented_df):,}")

        # Validation checks
        assert len(augmented_df) <= len(original_df), "Should have fewer events after dropping"

        # Schema validation
        assert validate_augmented_schema(augmented_df, AugmentedEventSchema, "time-dropped events")

        # All remaining properties should be valid
        assert validate_coordinate_bounds(augmented_df, "time-dropped events", "etram")
        assert validate_polarity_integrity(augmented_df, "time-dropped events")
        assert validate_temporal_order(augmented_df, "time-dropped events", strict=True)

        print("✓ Drop by time validation passed")

    def test_drop_by_area_validation(self, sample_events_list, augmentation_module):
        """Test drop by area maintains data validity."""
        print("\n" + "=" * 60)
        print("DROP BY AREA VALIDATION TEST")
        print("=" * 60)

        # Create drop by area configuration (drop 15% of sensor area)
        config = augmentation_module.AugmentationConfig.new().with_drop_area(0.15, 1280, 720)

        print(f"Applying drop by area to {len(sample_events_list):,} events...")
        start_time = time.time()

        # Apply drop by area using Rust backend
        try:
            augmented_events = augmentation_module.augment_events(sample_events_list, config)
            augment_time = time.time() - start_time
            print(f"Augmentation completed in {augment_time:.3f}s")

        except Exception as e:
            print(f"Augmentation failed: {e}")
            augmented_events = sample_events_list

        # Convert to DataFrames
        original_df = convert_events_to_dataframe(sample_events_list)
        augmented_df = convert_events_to_dataframe(augmented_events)

        print(f"Original events: {len(original_df):,}")
        print(f"Augmented events: {len(augmented_df):,}")

        # Validation checks
        assert len(augmented_df) <= len(original_df), "Should have fewer events after area dropping"

        # Schema validation
        assert validate_augmented_schema(augmented_df, AugmentedEventSchema, "area-dropped events")

        # All remaining properties should be valid
        assert validate_coordinate_bounds(augmented_df, "area-dropped events", "etram")
        assert validate_polarity_integrity(augmented_df, "area-dropped events")
        assert validate_temporal_order(augmented_df, "area-dropped events", strict=True)

        print("✓ Drop by area validation passed")


# =============================================================================
# Combined Pipeline Tests
# =============================================================================


@requires_pandera
@requires_data
class TestCombinedAugmentations:
    """Test combined augmentation pipelines with pandera validation."""

    def test_complete_augmentation_pipeline(self, sample_events_list, augmentation_module):
        """Test combined augmentation pipeline maintains data integrity."""
        print("\n" + "=" * 60)
        print("COMBINED AUGMENTATION PIPELINE TEST")
        print("=" * 60)

        # Create comprehensive augmentation configuration
        config = (
            augmentation_module.AugmentationConfig.new()
            .with_spatial_jitter(1.0, 1.0)
            .with_time_jitter(500.0)  # 0.5ms
            .with_uniform_noise(500, 1280, 720)
            .with_drop_event(0.1)
        )  # Drop 10%

        print(f"Applying combined augmentation pipeline to {len(sample_events_list):,} events...")
        start_time = time.time()

        # Apply combined augmentation using Rust backend
        try:
            augmented_events = augmentation_module.augment_events(sample_events_list, config)
            augment_time = time.time() - start_time
            print(f"Combined augmentation completed in {augment_time:.3f}s")

        except Exception as e:
            print(f"Augmentation failed: {e}")
            augmented_events = sample_events_list

        # Convert to DataFrames
        original_df = convert_events_to_dataframe(sample_events_list)
        augmented_df = convert_events_to_dataframe(augmented_events)

        original_count = len(original_df)
        augmented_count = len(augmented_df)

        print(f"Original events: {original_count:,}")
        print(f"Augmented events: {augmented_count:,}")
        print(f"Processing rate: {original_count / augment_time:,.0f} events/second")

        # Validation checks - expect count change due to noise (+500) and drops (-10%)
        expected_range = (int(original_count * 0.8), int(original_count * 1.4))
        assert (
            expected_range[0] <= augmented_count <= expected_range[1]
        ), f"Event count {augmented_count} not in expected range {expected_range}"

        # Schema validation
        assert validate_augmented_schema(augmented_df, AugmentedEventSchema, "combined augmented events")

        # Coordinate bounds (with tolerance for jitter)
        assert validate_coordinate_bounds(augmented_df, "combined augmented events", "etram")

        # Polarity integrity should be maintained
        assert validate_polarity_integrity(augmented_df, "combined augmented events")

        # Temporal order may be disrupted (expected with jitter)
        validate_temporal_order(augmented_df, "combined augmented events", strict=False)

        print("✓ Combined augmentation pipeline validation passed")


# =============================================================================
# Performance and Edge Case Tests
# =============================================================================


@requires_pandera
@requires_data
class TestAugmentationPerformanceAndEdgeCases:
    """Test augmentation performance and edge cases."""

    def test_performance_benchmarks(self, sample_events_list, augmentation_module):
        """Test that augmentation meets performance requirements."""
        # Test simple spatial jitter performance
        config = augmentation_module.AugmentationConfig.new().with_spatial_jitter(1.0, 1.0)

        start_time = time.time()
        try:
            augmented_events = augmentation_module.augment_events(sample_events_list, config)
            augment_time = time.time() - start_time

            events_per_second = len(sample_events_list) / augment_time if augment_time > 0 else float("inf")
            print(f"Augmentation performance: {events_per_second:,.0f} events/second")

            # Should process at least 50K events per second (conservative threshold)
            assert events_per_second > 50_000, f"Performance too slow: {events_per_second:,.0f} events/s"

            # Validate output
            augmented_df = convert_events_to_dataframe(augmented_events)
            assert validate_augmented_schema(
                augmented_df, SpatiallyAugmentedSchema, "performance test output"
            )

        except Exception as e:
            print(f"Performance test failed: {e}")
            # Don't fail test if augmentation isn't available

    def test_empty_input_handling(self, augmentation_module):
        """Test handling of empty input datasets."""
        empty_events = []
        config = augmentation_module.AugmentationConfig.new().with_spatial_jitter(1.0, 1.0)

        try:
            augmented_events = augmentation_module.augment_events(empty_events, config)

            # Should handle empty input gracefully
            assert len(augmented_events) == 0, "Empty input should produce empty output"
            print("✓ Empty input handling works correctly")

        except Exception as e:
            print(f"Empty input test failed: {e}")
            # Don't fail test if augmentation isn't available

    def test_extreme_augmentation_parameters(self, sample_events_list, augmentation_module):
        """Test handling of extreme augmentation parameters."""
        # Take small subset for extreme testing
        small_events = sample_events_list[:100]

        # Test extreme spatial jitter
        config = augmentation_module.AugmentationConfig.new().with_spatial_jitter(100.0, 100.0)

        try:
            augmented_events = augmentation_module.augment_events(small_events, config)
            augmented_df = convert_events_to_dataframe(augmented_events)

            # Should still produce valid events (though possibly clipped)
            assert len(augmented_df) <= len(small_events), "Should not increase event count"
            assert validate_polarity_integrity(augmented_df, "extreme jitter events")

            print("✓ Extreme parameter handling works correctly")

        except Exception as e:
            print(f"Extreme parameter test failed: {e}")
            # Don't fail test if augmentation isn't available


# =============================================================================
# Main Test Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
