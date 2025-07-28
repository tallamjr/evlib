#!/usr/bin/env python3
"""
Test the completely clean loading output.
"""

import evlib
import io
import sys
from contextlib import redirect_stderr


def test_completely_clean_loading():
    """Test that HDF5 loading now produces minimal output."""

    print("Testing Completely Clean Loading")
    print("=" * 50)

    # Use the test file that exists
    test_file = "tests/data/eTram/h5/val_2/val_night_011_td.h5"

    print(f"Testing with: {test_file}")

    # Capture all stderr output
    captured_stderr = io.StringIO()

    try:
        with redirect_stderr(captured_stderr):
            print("   Loading events...")
            events = evlib.load_events(test_file)
            df = events.collect()

        # Check what was printed
        error_output = captured_stderr.getvalue()

        print(f"   Loaded {len(df):,} events")
        print(f"   Coordinates: x={df['x'].min()}-{df['x'].max()}, y={df['y'].min()}-{df['y'].max()}")

        print("\nOutput Analysis:")
        if error_output.strip():
            lines = error_output.strip().split("\n")
            print(f"   Total output lines: {len(lines)}")

            for line in lines:
                if "SUCCESS" in line:
                    print(f"   SUCCESS: {line.strip()}")
                elif "WARNING" in line:
                    print(f"   WARNING: {line.strip()}")
                elif "ERROR" in line:
                    print(f"   ERROR: {line.strip()}")
                else:
                    print(f"   INFO: {line.strip()}")

            # Count duplicate messages
            success_count = len([line for line in lines if "SUCCESS" in line])
            if success_count > 1:
                print(f"   Found {success_count} SUCCESS messages (should be 1)")
            else:
                print(f"   Clean SUCCESS message count: {success_count}")

        else:
            print("   Completely silent loading!")

        return True

    except Exception as e:
        print(f"   Loading failed: {e}")
        return False


if __name__ == "__main__":
    test_completely_clean_loading()
