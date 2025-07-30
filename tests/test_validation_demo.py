"""
Test validation helpers.

This test shows how to use the validation helpers from validation_helpers.py
for testing event data quality.
"""

import polars as pl
import pytest
from pathlib import Path


def find_test_data() -> Path:
    """Find suitable test data file."""
    data_dir = Path(__file__).parent / "data"

    # Try eTram HDF5 data first
    etram_h5 = data_dir / "eTram" / "h5" / "val_2"
    if etram_h5.exists():
        h5_files = list(etram_h5.glob("*.h5"))
        if h5_files:
            return h5_files[0]

    # Try slider depth text data
    slider_txt = data_dir / "slider_depth" / "events.txt"
    if slider_txt.exists():
        return slider_txt

    pytest.skip("No suitable test data found")


def test_validation_helpers_basic():
    """Test basic validation helper functionality."""
    try:
        # Import validation helpers
        from validation_helpers import quick_validate_events, validate_events

        # Import evlib
        import evlib

        # Load test data
        test_file = find_test_data()
        events = evlib.load_events(str(test_file)).limit(1000)

        # Test quick validation
        is_valid = quick_validate_events(events)
        assert isinstance(is_valid, bool)

        # Test detailed validation
        result = validate_events(events, sensor_type="etram")
        assert isinstance(result, dict)
        assert "valid" in result

        print(f"Quick validation: {'PASSED' if is_valid else 'FAILED'}")
        print(f"Detailed validation: {'PASSED' if result['valid'] else 'FAILED'}")

    except ImportError:
        pytest.skip("Validation helpers not available (pandera not installed)")


def test_validation_schemas():
    """Test that validation schemas can be created."""
    try:
        from validation_helpers import create_event_schema, ETRAM_SCHEMA

        # Test schema creation
        schema = create_event_schema("etram", data_format="duration")
        assert schema is not None

        # Test pre-defined schema
        assert ETRAM_SCHEMA is not None

        print("✓ Validation schemas working correctly")

    except ImportError:
        pytest.skip("Validation helpers not available (pandera not installed)")


if __name__ == "__main__":
    # Run the tests when executed directly
    test_validation_helpers_basic()
    test_validation_schemas()
    print("✓ All validation helper tests passed!")
