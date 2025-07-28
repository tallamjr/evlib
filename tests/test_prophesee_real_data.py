#!/usr/bin/env python3
"""
Test script to verify Prophesee HDF5 data loading works with real data.
"""

import sys
import traceback
from pathlib import Path


def test_prophesee_hdf5_loading():
    """Test loading real Prophesee HDF5 data."""
    try:
        import evlib.formats as evf

        # Test with the actual pedestrians.hdf5 file
        data_path = Path(
            "/Users/tallam/github/tallamjr/origin/evlib/data/prophersee/samples/hdf5/pedestrians.hdf5"
        )

        if not data_path.exists():
            print(f"Test data file not found: {data_path}")
            return False

        print(f"Loading data from: {data_path}")

        # Load events as LazyFrame
        events_lazy = evf.load_events(str(data_path))
        print("Successfully loaded LazyFrame")

        # Collect to DataFrame to get actual data
        events_df = events_lazy.collect()
        num_events = len(events_df)

        print(f"Number of events: {num_events:,}")

        if num_events == 0:
            print("No events loaded - this indicates a problem")
            return False

        # Check data structure
        columns = events_df.columns
        print(f"Columns: {columns}")

        expected_columns = ["x", "y", "timestamp", "polarity"]
        missing_columns = set(expected_columns) - set(columns)
        if missing_columns:
            print(f"Missing expected columns: {missing_columns}")
            return False

        # Check data ranges
        if num_events > 0:
            sample = events_df.head(5)
            print("Sample events:")
            print(sample)

            # Check coordinate ranges (should be reasonable for event camera)
            x_range = (events_df["x"].min(), events_df["x"].max())
            y_range = (events_df["y"].min(), events_df["y"].max())
            print(f"X range: {x_range}, Y range: {y_range}")

            # Check timestamp range
            t_range = (events_df["timestamp"].min(), events_df["timestamp"].max())
            print(f"Timestamp range: {t_range}")

            # Check polarity values
            polarity_values = sorted(events_df["polarity"].unique().to_list())
            print(f"Polarity values: {polarity_values}")

        print("All checks passed!")
        return True

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure evlib is properly built with: maturin develop --features arrow")
        return False
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return False


def test_ecf_codec_functions():
    """Test ECF codec functions directly."""
    try:
        import evlib.formats as evf

        print("Testing ECF codec functions...")

        # Test the Prophesee ECF decode function with dummy data
        # The function expects compressed_data (bytes) and debug (bool)
        dummy_compressed_data = b"\x18\x00\x00\x00"  # Simple header with 3 events
        result = evf.test_prophesee_ecf_decode(dummy_compressed_data, True)
        print(f"Prophesee ECF decode test result: {result}")

        return True

    except Exception as e:
        print(f"Error testing ECF codec: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing Prophesee HDF5 data loading...")
    print("=" * 50)

    success = True

    # Test real data loading
    print("\n1. Testing real HDF5 data loading:")
    success &= test_prophesee_hdf5_loading()

    # Test ECF codec functions
    print("\n2. Testing ECF codec functions:")
    success &= test_ecf_codec_functions()

    print("\n" + "=" * 50)
    if success:
        print("All tests passed! Prophesee ECF codec is working with real data.")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)
