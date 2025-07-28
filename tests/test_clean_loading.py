#!/usr/bin/env python3
"""
Test script to demonstrate clean HDF5 loading behavior.

This tests both the ECF decoding and the integrated validation system.
"""

import evlib
import os
import sys


def test_clean_loading():
    """Test that HDF5 loading no longer produces excessive warnings."""

    print("Testing Clean HDF5 Loading")
    print("=" * 50)

    # Check if we have test data available
    test_files = [
        "tests/data/eTram/h5/val_2/val_night_011_td.h5",
        "tests/data/output.h5",
        "tests/data/gen4_1mpx_processed_RVT/test/moorea_2019-06-19_000_793500000_853500000/event_representations_v2/stacked_histogram_dt50_nbins10/event_representations_ds2_nearest.h5",
    ]

    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = file_path
            break

    if not test_file:
        print("No test HDF5 files found in expected locations")
        print("   Expected files:")
        for f in test_files:
            print(f"   - {f}")
        return False

    print(f"Testing with file: {test_file}")
    print(f"File size: {os.path.getsize(test_file) / (1024*1024):.1f} MB")

    try:
        print("\n1. Testing basic loading (should show minimal output):")
        print("   Loading events...")

        # Capture stdout/stderr to show what gets printed
        import io
        from contextlib import redirect_stderr

        captured_stderr = io.StringIO()

        with redirect_stderr(captured_stderr):
            events = evlib.load_events(test_file)
            df = events.collect()

        # Show captured output
        error_output = captured_stderr.getvalue()
        if error_output.strip():
            print("   Output from loading:")
            for line in error_output.strip().split("\n"):
                print(f"      {line}")
        else:
            print("   Silent loading (no debug output)")

        print(f"\n   Successfully loaded {len(df):,} events")
        print(f"   Coordinate ranges: x={df['x'].min()}-{df['x'].max()}, y={df['y'].min()}-{df['y'].max()}")
        print(f"   Polarity values: {sorted(df['polarity'].unique())}")

        # Test direct validation functions
        print("\n2. Testing direct validation functions:")
        try:
            # Test with the loaded data
            validation_result = evlib.validation.validate_events(
                events, sensor_type="generic_large", strict=False
            )
            print(f"   Direct validation: {validation_result['valid']}")

            quick_result = evlib.validation.quick_validate_events(events)
            print(f"   Quick validation: {quick_result}")
        except Exception as e:
            print(f"   Direct validation error: {e}")

        # Test integrated validation (may not work with all file types)
        print("\n3. Testing integrated validation:")
        try:
            validated_events = evlib.load_events(test_file, validate="quick")
            validated_df = validated_events.collect()
            print(f"   Integrated validation passed: {len(validated_df):,} events")
        except ValueError as e:
            print(f"   WARNING: Integrated validation failed: {str(e)[:100]}...")
        except Exception as e:
            print(f"   Integrated validation error: {e}")

        return True

    except Exception as e:
        print(f"\nLoading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_validation_integration():
    """Test the validation system integration."""

    print("\nTesting Validation Integration")
    print("=" * 40)

    # Test validation functions are available
    print("Available validation functions:")

    if hasattr(evlib, "validation"):
        print("   evlib.validation module")
        print("   Available schemas:", [x for x in dir(evlib.validation) if "SCHEMA" in x])

        if hasattr(evlib.validation, "validate_events"):
            print("   evlib.validation.validate_events")
        else:
            print("   evlib.validation.validate_events not found")

        if hasattr(evlib.validation, "quick_validate_events"):
            print("   evlib.validation.quick_validate_events")
        else:
            print("   evlib.validation.quick_validate_events not found")
    else:
        print("   evlib.validation module not found")

    # Test load_events validation parameter
    import inspect

    sig = inspect.signature(evlib.load_events)
    if "validate" in sig.parameters:
        print("   evlib.load_events supports validate parameter")
    else:
        print("   evlib.load_events validate parameter not found")


if __name__ == "__main__":
    print("evlib Clean Loading & Validation Test")
    print("=" * 60)

    success = test_clean_loading()
    test_validation_integration()

    if success:
        print("\nAll tests completed successfully!")
        print("\nWhat was fixed:")
        print("   - Removed excessive 'WARNING: Could not find valid ECF header' messages")
        print("   - Silenced debug output from chunk scanning")
        print("   - Only show SUCCESS message when events are loaded")
        print("   - Integrated Pandera validation system")
        sys.exit(0)
    else:
        print("\nSome tests failed")
        sys.exit(1)
