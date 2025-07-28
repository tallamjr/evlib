"""
Demonstration of event data validation with Pandera.

This module shows how to use the validation schemas to ensure
event data quality and catch issues early in the processing pipeline.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, Optional
from .validation import (
    validate_events,
    quick_validate_events,
    PROPHESEE_GEN4_SCHEMA,
    DAVIS_SCHEMA,
    PERMISSIVE_SCHEMA,
)


def create_test_events(
    num_events: int = 1000, sensor_type: str = "prophesee_gen4", add_garbage: bool = False
) -> pl.LazyFrame:
    """
    Create synthetic test event data for validation testing.

    Args:
        num_events: Number of events to generate
        sensor_type: Type of sensor to simulate
        add_garbage: Whether to add some invalid data for testing

    Returns:
        Polars LazyFrame with test event data
    """

    # Sensor specifications
    sensor_specs = {
        "prophesee_gen4": {"width": 1280, "height": 720},
        "davis346": {"width": 346, "height": 240},
        "generic": {"width": 640, "height": 480},
    }

    spec = sensor_specs.get(sensor_type, sensor_specs["generic"])

    # Generate valid events
    np.random.seed(42)  # Reproducible

    x_coords = np.random.randint(0, spec["width"], num_events)
    y_coords = np.random.randint(0, spec["height"], num_events)
    polarities = np.random.choice([-1, 1], num_events)

    # Generate monotonic timestamps
    base_time = 1000.0  # Start at 1 second
    time_deltas = np.random.exponential(0.001, num_events)  # ~1ms average
    timestamps = base_time + np.cumsum(time_deltas)

    if add_garbage:
        # Add some invalid data to test validation
        garbage_count = min(10, num_events // 10)
        garbage_indices = np.random.choice(num_events, size=garbage_count, replace=False)

        # Invalid coordinates (use first third of garbage indices)
        coord_count = len(garbage_indices) // 3
        if coord_count > 0:
            x_coords[garbage_indices[:coord_count]] = spec["width"] + 1000  # Way outside bounds
            y_coords[garbage_indices[:coord_count]] = spec["height"] + 1000

        # Invalid polarities (use middle third)
        pol_start = coord_count
        pol_end = min(pol_start + coord_count, len(garbage_indices))
        if pol_end > pol_start:
            polarities[garbage_indices[pol_start:pol_end]] = np.random.randint(10, 100, pol_end - pol_start)

        # Invalid timestamps (use remaining indices)
        ts_start = pol_end
        if ts_start < len(garbage_indices):
            timestamps[garbage_indices[ts_start:]] = -999999  # Negative timestamps

    # Create DataFrame
    events_df = pl.LazyFrame(
        {
            "x": x_coords.astype(np.int32),
            "y": y_coords.astype(np.int32),
            "timestamp": timestamps,
            "polarity": polarities.astype(np.int32),
        }
    )

    return events_df


def demo_validation() -> Dict[str, Any]:
    """
    Demonstrate event data validation with various scenarios.

    Returns:
        Dictionary with demonstration results
    """

    results = {}

    print("Event Data Validation Demo")
    print("=" * 50)

    # Test 1: Valid Prophesee Gen4 data
    print("\n1. Testing valid Prophesee Gen4 data...")
    valid_events = create_test_events(1000, "prophesee_gen4", add_garbage=False)
    validation_result = validate_events(valid_events, sensor_type="prophesee_gen4", strict=True)

    print(f"   Valid: {validation_result['valid']}")
    if validation_result["statistics"]:
        stats = validation_result["statistics"]
        print(f"   Event count: {stats['event_count']}")
        print(f"   X range: {stats['coordinate_ranges']['x']}")
        print(f"   Y range: {stats['coordinate_ranges']['y']}")
        print(f"   Time range: {stats['timestamp_range'][0]:.3f} - {stats['timestamp_range'][1]:.3f}s")
        print(f"   Polarities: {stats['unique_polarities']}")

    results["valid_gen4"] = validation_result

    # Test 2: Data with garbage values
    print("\n2. Testing data with garbage values...")
    invalid_events = create_test_events(1000, "prophesee_gen4", add_garbage=True)
    validation_result = validate_events(invalid_events, sensor_type="prophesee_gen4", strict=True)

    print(f"   Valid: {validation_result['valid']}")
    if validation_result["errors"]:
        print(f"   Errors found: {len(validation_result['errors'])}")
        for error in validation_result["errors"][:2]:  # Show first 2 errors
            print(f"      - {error['type']}: {error['message'][:100]}...")

    results["invalid_gen4"] = validation_result

    # Test 3: Quick validation
    print("\n3. Testing quick validation...")
    quick_valid = quick_validate_events(valid_events)
    quick_invalid = quick_validate_events(invalid_events)

    print(f"   Quick valid check: {quick_valid}")
    print(f"   Quick invalid check: {quick_invalid}")

    results["quick_validation"] = {"valid_data": quick_valid, "invalid_data": quick_invalid}

    # Test 4: Different sensor types
    print("\n4. Testing different sensor constraints...")
    davis_events = create_test_events(500, "davis346", add_garbage=False)

    # Test with correct sensor type (should pass)
    davis_validation = validate_events(davis_events, sensor_type="davis346", strict=True)
    print(f"   DAVIS with DAVIS schema: {davis_validation['valid']}")

    # Test with wrong sensor type (should fail due to coordinate constraints)
    prophesee_validation = validate_events(davis_events, sensor_type="prophesee_gen4", strict=True)
    print(f"   DAVIS with Prophesee schema: {prophesee_validation['valid']}")

    results["sensor_constraints"] = {
        "davis_with_davis_schema": davis_validation["valid"],
        "davis_with_prophesee_schema": prophesee_validation["valid"],
    }

    print("\nValidation demonstration complete!")
    return results


def validate_loaded_events(
    events_df: pl.LazyFrame, sensor_type: str = "generic_large", verbose: bool = True
) -> bool:
    """
    Validate loaded event data and report results.

    Args:
        events_df: LazyFrame with event data
        sensor_type: Expected sensor type
        verbose: Whether to print detailed results

    Returns:
        True if validation passes, False otherwise
    """

    if verbose:
        print(f"Validating events for {sensor_type} sensor...")

    # First try quick validation
    quick_result = quick_validate_events(events_df)

    if not quick_result:
        if verbose:
            print("Quick validation failed - data contains obvious errors")
        return False

    # Then try strict validation
    validation_result = validate_events(events_df, sensor_type=sensor_type, strict=True)

    if verbose:
        if validation_result["valid"]:
            print("Validation passed!")
            if validation_result["statistics"]:
                stats = validation_result["statistics"]
                print(f"   Events: {stats['event_count']:,}")
                print(
                    f"   Coordinates: x={stats['coordinate_ranges']['x']}, y={stats['coordinate_ranges']['y']}"
                )
                print(f"   Polarities: {stats['unique_polarities']}")
                time_range = stats["timestamp_range"]
                duration = time_range[1] - time_range[0]
                print(f"   Duration: {duration:.3f}s ({time_range[0]:.3f} to {time_range[1]:.3f})")
        else:
            print("Validation failed!")
            for error in validation_result["errors"]:
                print(f"   {error['type']}: {error['message']}")

    return validation_result["valid"]


if __name__ == "__main__":
    # Run the demonstration
    demo_results = demo_validation()
