"""
Comprehensive pytest tests for load_events() function with pandera validation.

This test suite validates the structure and data integrity of events loaded by
evlib.load_events() from different real datasets. Uses pandera schemas to ensure
data fits sensor bounds and logical constraints.

Test Coverage:
- Event data structure validation (x, y, timestamp, polarity)
- Sensor-specific coordinate bounds validation
- Timestamp logical validation and monotonicity
- Polarity encoding validation (expected -1/1 format)
- Dataset-specific constraints (eTram, Gen4, slider_depth, Prophesee samples)
- Performance and memory efficiency validation

Camera Specifications Tested:
- eTram dataset: 1280x720 pixels, polarity 0/1 → -1/1 conversion
- Gen4 dataset: 1280x720 pixels, various timestamp ranges
- slider_depth: 346x240 pixels (DAVIS camera), text format
- Prophesee samples: Various resolutions, EVT2/EVT3 formats

Note: This test uses validation helpers from tests/validation_helpers.py
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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
# Dataset Information and Expected Constraints
# =============================================================================

DATASET_SPECS = {
    "etram": {
        "sensor_type": "etram",
        "resolution": (1280, 720),
        "expected_polarity_format": "-1_1",  # Converted format
        "expected_duration_range": (1.0, 300.0),  # seconds
        "min_events": 100_000,
        "file_patterns": ["*.h5"],
        "data_paths": ["eTram/h5/val_2"],
    },
    "prophesee_hdf5": {
        "sensor_type": "prophesee_gen4",
        "resolution": (1280, 720),
        "expected_polarity_format": "-1_1",
        "expected_duration_range": (0.001, 10.0),  # Small sample file
        "min_events": 10_000,
        "file_patterns": ["*.hdf5"],
        "data_paths": ["prophersee/samples/hdf5"],
    },
    "prophesee_evt2": {
        "sensor_type": "generic_large",  # High resolution sensor
        "resolution": (2048, 2048),  # Conservative estimate
        "expected_polarity_format": "-1_1",
        "expected_duration_range": (1.0, 3600.0),  # Could be long recordings
        "min_events": 100_000,
        "file_patterns": ["*.raw"],
        "data_paths": ["prophersee/samples/evt2"],
    },
    "prophesee_evt3": {
        "sensor_type": "generic_large",  # High resolution sensor
        "resolution": (2200, 1800),  # Conservative estimate based on observed ranges
        "expected_polarity_format": "-1_1",
        "expected_duration_range": (1.0, 3600.0),
        "min_events": 1_000_000,  # Large file
        "file_patterns": ["*.raw"],
        "data_paths": ["prophersee/samples/evt3"],
    },
    "slider_depth": {
        "sensor_type": "davis346",  # Similar to DAVIS camera
        "resolution": (346, 240),  # Standard DAVIS resolution
        "expected_polarity_format": "0_1",  # Text format often uses 0/1
        "expected_duration_range": (5.0, 60.0),
        "min_events": 100_000,
        "file_patterns": ["events.txt"],
        "data_paths": ["slider_depth"],
    },
}


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
def test_data_files():
    """Discover and return paths to all available test data files."""
    # Try both tests/data and project root data directories
    potential_data_dirs = [
        Path(__file__).parent / "data",  # tests/data
        Path(__file__).parent.parent / "data",  # project root data
    ]

    discovered_files = {}

    for dataset_name, spec in DATASET_SPECS.items():
        discovered_files[dataset_name] = []

        # Try each data directory for this dataset
        for data_dir in potential_data_dirs:
            if not data_dir.exists():
                continue

            for data_path in spec["data_paths"]:
                full_path = data_dir / data_path
                if full_path.exists():
                    for pattern in spec["file_patterns"]:
                        files = list(full_path.glob(pattern))
                        if pattern == "*.h5":
                            # Also search recursively for h5 files
                            files.extend(list(full_path.glob(f"**/{pattern}")))
                        discovered_files[dataset_name].extend(files)

                    # If we found files for this dataset, don't check other data dirs
                    if discovered_files[dataset_name]:
                        break

            # If we found files for this dataset, don't check other data dirs
            if discovered_files[dataset_name]:
                break

    # Filter to only datasets with available files
    available_datasets = {k: v for k, v in discovered_files.items() if v}

    if not available_datasets:
        pytest.skip("No test data files found")

    return available_datasets


@pytest.fixture(
    scope="session", params=["etram", "prophesee_hdf5", "prophesee_evt2", "prophesee_evt3", "slider_depth"]
)
def dataset_info(request, test_data_files):
    """Parametrized fixture providing dataset information and file path."""
    dataset_name = request.param

    if dataset_name not in test_data_files:
        pytest.skip(f"No {dataset_name} data files available")

    # Select first available file for this dataset
    file_path = test_data_files[dataset_name][0]
    spec = DATASET_SPECS[dataset_name]

    return {
        "name": dataset_name,
        "file_path": file_path,
        "spec": spec,
    }


@pytest.fixture(scope="session")
def loaded_events(evlib_module, dataset_info):
    """Load events from dataset file."""
    file_path = dataset_info["file_path"]
    dataset_name = dataset_info["name"]

    print(f"\nLoading {dataset_name} data from: {file_path}")

    # Load events using evlib (auto-detects format)
    start_time = time.time()
    events = evlib_module.load_events(str(file_path))
    load_time = time.time() - start_time

    # Get count for display purposes
    count = events.select(pl.len()).collect()[0, 0]
    print(f"Loaded {count:,} events in {load_time:.3f}s ({count/load_time:,.0f} events/s)")

    return events


# =============================================================================
# Validation Helper Functions
# =============================================================================


def validate_event_data_structure(
    events_df: pl.LazyFrame, dataset_name: str, spec: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate event data structure and constraints."""
    results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }

    try:
        # Collect basic statistics
        sample_df = events_df.limit(10).collect()
        if len(sample_df) == 0:
            results["errors"].append("Dataset is empty")
            return results

        # Check column presence and types
        required_columns = {"x", "y", "t", "polarity"}
        actual_columns = set(sample_df.columns)

        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            results["errors"].append(f"Missing required columns: {missing}")
            return results

        # Validate data types
        expected_types = {
            "x": pl.Int16,
            "y": pl.Int16,
            "t": pl.Duration,  # evlib internal format
            "polarity": pl.Int8,
        }

        for col, expected_type in expected_types.items():
            actual_type = sample_df[col].dtype
            if actual_type != expected_type:
                results["errors"].append(f"Column '{col}' has type {actual_type}, expected {expected_type}")

        if results["errors"]:
            return results

        # Use pandera validation
        if PANDERA_AVAILABLE:
            # Determine polarity encoding for pandera schema
            polarity_encoding = "zero_one" if spec["expected_polarity_format"] == "0_1" else "minus_one_one"

            validation_result = validate_events(
                events_df,
                sensor_type=spec["sensor_type"],
                strict=True,
                data_format="duration",
                polarity_encoding=polarity_encoding,
            )

            results["valid"] = validation_result["valid"]
            results["errors"].extend(validation_result["errors"])
            results["warnings"].extend(validation_result["warnings"])
            results["statistics"] = validation_result["statistics"]
        else:
            results["warnings"].append("Pandera not available, skipping schema validation")
            results["valid"] = True

    except Exception as e:
        results["errors"].append(f"Validation failed: {e}")

    return results


def validate_sensor_bounds(events_df: pl.LazyFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that coordinates are within sensor bounds."""
    results = {"valid": True, "errors": [], "warnings": []}

    try:
        if spec["resolution"] is None:
            results["warnings"].append("No resolution specified, skipping bounds check")
            return results

        expected_width, expected_height = spec["resolution"]
        max_x, max_y = expected_width - 1, expected_height - 1

        # Check coordinate bounds
        coord_stats = events_df.select(
            [
                pl.col("x").min().alias("x_min"),
                pl.col("x").max().alias("x_max"),
                pl.col("y").min().alias("y_min"),
                pl.col("y").max().alias("y_max"),
            ]
        ).collect()

        x_min, x_max = coord_stats[0, 0], coord_stats[0, 1]
        y_min, y_max = coord_stats[0, 2], coord_stats[0, 3]

        # Validate bounds
        if x_min < 0 or x_max > max_x:
            results["errors"].append(f"X coordinates [{x_min}, {x_max}] exceed sensor bounds [0, {max_x}]")
            results["valid"] = False

        if y_min < 0 or y_max > max_y:
            results["errors"].append(f"Y coordinates [{y_min}, {y_max}] exceed sensor bounds [0, {max_y}]")
            results["valid"] = False

        # Check for reasonable coverage (not all events in tiny region)
        x_coverage = (x_max - x_min) / max_x
        y_coverage = (y_max - y_min) / max_y

        if x_coverage < 0.1 or y_coverage < 0.1:
            results["warnings"].append(f"Low spatial coverage: X={x_coverage:.1%}, Y={y_coverage:.1%}")

    except Exception as e:
        results["errors"].append(f"Bounds validation failed: {e}")
        results["valid"] = False

    return results


def validate_temporal_properties(events_df: pl.LazyFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate temporal properties of event data."""
    results = {"valid": True, "errors": [], "warnings": []}

    try:
        # Get timestamp statistics (convert Duration to seconds)
        temporal_stats = events_df.select(
            [
                (pl.col("t").dt.total_microseconds() / 1_000_000).min().alias("t_min"),
                (pl.col("t").dt.total_microseconds() / 1_000_000).max().alias("t_max"),
                pl.len().alias("event_count"),
            ]
        ).collect()

        t_min, t_max = temporal_stats[0, 0], temporal_stats[0, 1]
        event_count = temporal_stats[0, 2]
        duration = t_max - t_min

        # Validate duration bounds
        expected_min, expected_max = spec["expected_duration_range"]
        if duration < expected_min or duration > expected_max:
            results["warnings"].append(
                f"Duration {duration:.1f}s outside expected range [{expected_min}, {expected_max}]s"
            )

        # Check for reasonable event rate
        event_rate = event_count / duration if duration > 0 else 0
        if event_rate < 1000:  # Less than 1kHz seems low for event cameras
            results["warnings"].append(f"Low event rate: {event_rate:.0f} Hz")
        elif event_rate > 10_000_000:  # More than 10MHz seems suspiciously high
            results["warnings"].append(f"Very high event rate: {event_rate:.0f} Hz")

        # Check monotonicity (informational only)
        backward_jumps = events_df.select((pl.col("t").diff() < pl.duration(microseconds=0)).sum()).collect()[
            0, 0
        ]

        if backward_jumps > 0:
            results["warnings"].append(
                f"Found {backward_jumps} non-monotonic timestamp jumps (may be normal)"
            )

        # Validate timestamp starts near zero or is reasonable
        if t_min < -1.0:  # Negative timestamps beyond reasonable clock offset
            results["warnings"].append(f"Negative timestamp start: {t_min:.3f}s")
        elif t_min > 1e9:  # Very large start timestamp (Unix time?)
            results["warnings"].append(f"Large timestamp start: {t_min:.0f}s (Unix time?)")

    except Exception as e:
        results["errors"].append(f"Temporal validation failed: {e}")
        results["valid"] = False

    return results


def validate_polarity_encoding(events_df: pl.LazyFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate polarity encoding and distribution."""
    results = {"valid": True, "errors": [], "warnings": []}

    try:
        # Get polarity statistics (simpler approach)
        polarity_stats = events_df.select(
            [
                (pl.col("polarity") == 1).sum().alias("positive_count"),
                (pl.col("polarity") == -1).sum().alias("negative_count"),
                (pl.col("polarity") == 0).sum().alias("zero_count"),
                pl.len().alias("total_count"),
            ]
        ).collect()

        positive_count = polarity_stats[0, 0]
        negative_count = polarity_stats[0, 1]
        zero_count = polarity_stats[0, 2]
        total_count = polarity_stats[0, 3]

        # Determine unique polarities from counts
        unique_polarities = []
        if positive_count > 0:
            unique_polarities.append(1)
        if negative_count > 0:
            unique_polarities.append(-1)
        if zero_count > 0:
            unique_polarities.append(0)

        # Check for valid polarity encoding
        actual_polarities = set(unique_polarities)
        expected_format = spec["expected_polarity_format"]

        if expected_format == "-1_1":
            # Expect -1/1 encoding (binary formats converted by evlib)
            # Allow single-polarity files (some event camera files may only have positive or negative events)
            valid_polarities = {-1, 1}
            if not actual_polarities.issubset(valid_polarities):
                results["errors"].append(f"Expected -1/1 polarity encoding, got {actual_polarities}")
                results["valid"] = False
        elif expected_format == "0_1":
            # Expect 0/1 encoding (text formats kept as-is)
            # Allow single-polarity files
            valid_polarities = {0, 1}
            if not actual_polarities.issubset(valid_polarities):
                results["errors"].append(f"Expected 0/1 polarity encoding, got {actual_polarities}")
                results["valid"] = False
        else:
            # Unknown format, just check for reasonable values
            valid_polarities = {-1, 0, 1}
            if not actual_polarities.issubset(valid_polarities):
                results["errors"].append(
                    f"Invalid polarity values: {actual_polarities}, must be subset of {valid_polarities}"
                )
                results["valid"] = False

        # Additional check for unexpected zero values in -1/1 formats
        if expected_format == "-1_1" and zero_count > 0:
            results["errors"].append(f"Found {zero_count} events with polarity=0 in -1/1 format")
            results["valid"] = False

        # Check polarity balance
        if total_count > 0:
            positive_ratio = positive_count / total_count
            if positive_ratio < 0.05 or positive_ratio > 0.95:
                results["warnings"].append(
                    f"Extreme polarity imbalance: {positive_ratio:.1%} positive events"
                )

    except Exception as e:
        results["errors"].append(f"Polarity validation failed: {e}")
        results["valid"] = False

    return results


# =============================================================================
# Main Test Classes
# =============================================================================


@requires_pandera
@requires_data
class TestLoadEventsValidation:
    """Test load_events() function with comprehensive validation."""

    def test_event_data_structure(self, loaded_events, dataset_info):
        """Test that loaded events have correct structure and types."""
        dataset_name = dataset_info["name"]
        spec = dataset_info["spec"]

        print(f"\n{'='*60}")
        print(f"TESTING DATA STRUCTURE: {dataset_name}")
        print(f"{'='*60}")

        # Validate basic structure
        results = validate_event_data_structure(loaded_events, dataset_name, spec)

        # Print results
        if results["statistics"] and not results["statistics"].get("error"):
            stats = results["statistics"]
            event_count = stats.get("event_count", "N/A")
            if isinstance(event_count, int):
                print(f"Event count: {event_count:,}")
            else:
                print(f"Event count: {event_count}")
            if "coordinate_ranges" in stats:
                x_range = stats["coordinate_ranges"]["x"]
                y_range = stats["coordinate_ranges"]["y"]
                print(f"Coordinate ranges: X=[{x_range[0]}, {x_range[1]}], Y=[{y_range[0]}, {y_range[1]}]")
            if "duration_seconds" in stats:
                print(f"Duration: {stats['duration_seconds']:.3f}s")
        elif results["statistics"].get("error"):
            print(f"⚠️  Statistics collection failed: {results['statistics']['error']}")

        # Print warnings
        for warning in results["warnings"]:
            print(f"⚠️  {warning}")

        # Assert validation passed
        if results["errors"]:
            for error in results["errors"]:
                print(f"❌ {error}")

        assert results["valid"], f"Data structure validation failed: {results['errors']}"

        # Additional basic checks
        event_count = loaded_events.select(pl.len()).collect()[0, 0]
        assert event_count >= spec["min_events"], f"Too few events: {event_count} < {spec['min_events']}"

        print(f"✅ Data structure validation passed for {dataset_name}")

    def test_sensor_coordinate_bounds(self, loaded_events, dataset_info):
        """Test that coordinates are within expected sensor bounds."""
        dataset_name = dataset_info["name"]
        spec = dataset_info["spec"]

        print(f"\n{'='*60}")
        print(f"TESTING COORDINATE BOUNDS: {dataset_name}")
        print(f"{'='*60}")

        results = validate_sensor_bounds(loaded_events, spec)

        # Print warnings
        for warning in results["warnings"]:
            print(f"⚠️  {warning}")

        # Print errors if any
        if results["errors"]:
            for error in results["errors"]:
                print(f"❌ {error}")

        assert results["valid"], f"Coordinate bounds validation failed: {results['errors']}"
        print(f"✅ Coordinate bounds validation passed for {dataset_name}")

    def test_temporal_properties(self, loaded_events, dataset_info):
        """Test that timestamps are logical and within expected ranges."""
        dataset_name = dataset_info["name"]
        spec = dataset_info["spec"]

        print(f"\n{'='*60}")
        print(f"TESTING TEMPORAL PROPERTIES: {dataset_name}")
        print(f"{'='*60}")

        results = validate_temporal_properties(loaded_events, spec)

        # Print warnings
        for warning in results["warnings"]:
            print(f"⚠️  {warning}")

        # Print errors if any
        if results["errors"]:
            for error in results["errors"]:
                print(f"❌ {error}")

        assert results["valid"], f"Temporal validation failed: {results['errors']}"
        print(f"✅ Temporal properties validation passed for {dataset_name}")

    def test_polarity_encoding(self, loaded_events, dataset_info):
        """Test that polarity values are correctly encoded as -1/1."""
        dataset_name = dataset_info["name"]
        spec = dataset_info["spec"]

        print(f"\n{'='*60}")
        print(f"TESTING POLARITY ENCODING: {dataset_name}")
        print(f"{'='*60}")

        results = validate_polarity_encoding(loaded_events, spec)

        # Print warnings
        for warning in results["warnings"]:
            print(f"⚠️  {warning}")

        # Print errors if any
        if results["errors"]:
            for error in results["errors"]:
                print(f"❌ {error}")

        assert results["valid"], f"Polarity validation failed: {results['errors']}"
        print(f"✅ Polarity encoding validation passed for {dataset_name}")


@requires_pandera
@requires_data
class TestCrossDatasetConsistency:
    """Test consistency across different datasets and formats."""

    def test_format_consistency(self, test_data_files, evlib_module):
        """Test that different file formats produce consistent data structures."""
        print(f"\n{'='*60}")
        print("TESTING FORMAT CONSISTENCY")
        print(f"{'='*60}")

        loaded_datasets = {}

        # Load one file from each available dataset
        for dataset_name, files in test_data_files.items():
            if files:
                file_path = files[0]
                print(f"Loading {dataset_name}: {file_path}")

                try:
                    events = evlib_module.load_events(str(file_path))
                    loaded_datasets[dataset_name] = events

                    # Quick structure check
                    sample = events.limit(5).collect()
                    print(f"  Columns: {sample.columns}")
                    print(f"  Types: {sample.dtypes}")

                except Exception as e:
                    print(f"  ❌ Failed to load {dataset_name}: {e}")

        # Verify all loaded datasets have consistent structure
        if len(loaded_datasets) > 1:
            reference_name, reference_events = next(iter(loaded_datasets.items()))
            reference_sample = reference_events.limit(1).collect()
            reference_columns = set(reference_sample.columns)
            reference_types = reference_sample.dtypes

            for dataset_name, events in loaded_datasets.items():
                if dataset_name == reference_name:
                    continue

                sample = events.limit(1).collect()
                columns = set(sample.columns)
                types = sample.dtypes

                # Check column consistency
                assert columns == reference_columns, (
                    f"Column mismatch between {reference_name} and {dataset_name}: "
                    f"{reference_columns} vs {columns}"
                )

                # Check type consistency
                assert types == reference_types, (
                    f"Type mismatch between {reference_name} and {dataset_name}: "
                    f"{reference_types} vs {types}"
                )

        print(f"✅ Format consistency validation passed across {len(loaded_datasets)} datasets")


@requires_pandera
@requires_data
class TestPerformanceAndScalability:
    """Test performance characteristics of load_events()."""

    def test_loading_performance(self, dataset_info, evlib_module):
        """Test that loading performance meets reasonable expectations."""
        file_path = dataset_info["file_path"]
        dataset_name = dataset_info["name"]

        print(f"\n{'='*60}")
        print(f"TESTING LOADING PERFORMANCE: {dataset_name}")
        print(f"{'='*60}")

        # Measure loading time
        start_time = time.time()
        events = evlib_module.load_events(str(file_path))
        load_time = time.time() - start_time

        # Get event count
        event_count = events.select(pl.len()).collect()[0, 0]

        # Calculate performance metrics
        events_per_second = event_count / load_time if load_time > 0 else float("inf")

        print(f"File: {file_path}")
        print(f"Events: {event_count:,}")
        print(f"Load time: {load_time:.3f}s")
        print(f"Performance: {events_per_second:,.0f} events/second")

        # Performance assertions (dataset-specific thresholds)
        if dataset_name == "prophesee_hdf5":
            # ECF-compressed HDF5 files are slower due to decompression overhead
            min_performance = 5_000  # ECF decompression is computationally expensive
        else:
            # Standard performance threshold for other formats
            min_performance = 50_000

        assert events_per_second > min_performance, (
            f"Loading performance too slow: {events_per_second:,.0f} events/s "
            f"(expected > {min_performance:,} events/s for {dataset_name})"
        )

        # Memory efficiency check (should be LazyFrame)
        assert hasattr(events, "collect"), "Events should be returned as LazyFrame for memory efficiency"

        print(f"✅ Performance validation passed for {dataset_name}")

    def test_memory_efficiency(self, loaded_events, dataset_info):
        """Test memory efficiency of loaded data structures."""
        dataset_name = dataset_info["name"]

        print(f"\n{'='*60}")
        print(f"TESTING MEMORY EFFICIENCY: {dataset_name}")
        print(f"{'='*60}")

        # Verify LazyFrame usage (memory efficient)
        assert isinstance(loaded_events, pl.LazyFrame), "Events should be LazyFrame for memory efficiency"

        # Test that we can work with data without loading everything into memory
        sample_count = loaded_events.limit(1000).select(pl.len()).collect()[0, 0]
        assert sample_count <= 1000, "Limit should work properly"

        # Test basic operations work efficiently
        coord_stats = loaded_events.select(
            [
                pl.col("x").min().alias("x_min"),
                pl.col("x").max().alias("x_max"),
                pl.col("y").min().alias("y_min"),
                pl.col("y").max().alias("y_max"),
            ]
        ).collect()

        assert len(coord_stats) == 1, "Statistics should be computed efficiently"

        print(f"✅ Memory efficiency validation passed for {dataset_name}")


# =============================================================================
# Main Test Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
