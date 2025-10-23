#!/usr/bin/env python3
"""
Test script to verify the packed coordinate decoding fix.
Creates synthetic ECF data that matches the problematic format:
- Y-bits = 0 (constant Y coordinates)
- X-bits = 1 (1 bit per X coordinate)
- 16,384 events
"""

import struct
import sys
import os

# Add current directory to Python path so we can import evlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_packed_ecf_data():
    """Create synthetic ECF data with packed coordinates (Y-bits=0, X-bits=1)"""

    # ECF header (32-bit integer with bit-packed fields)
    num_events = 16384
    ys_xs_and_ps_packed = True  # Bit 1: use packed coordinates
    xs_and_ps_packed = False  # Bit 0: not just X/P packed

    # Pack the header
    header = (num_events << 2) | (int(ys_xs_and_ps_packed) << 1) | int(xs_and_ps_packed)

    # Base timestamp
    base_timestamp = 7800

    # Coordinate encoding byte: Y-bits=0, X-bits=1
    coord_bits = 0x01  # Y-bits (upper 4 bits) = 0, X-bits (lower 4 bits) = 1

    print("Creating ECF data:")
    print(f"  Header: 0x{header:08x}")
    print(f"  Events: {num_events}")
    print(f"  Base timestamp: {base_timestamp}")
    print(f"  Coord bits: 0x{coord_bits:02x} (Y-bits=0, X-bits=1)")

    # Start building the ECF data
    ecf_data = bytearray()

    # Header (4 bytes)
    ecf_data.extend(struct.pack("<I", header))

    # Base timestamp (8 bytes)
    ecf_data.extend(struct.pack("<q", base_timestamp))

    # Coordinate encoding byte (1 byte)
    ecf_data.append(coord_bits)

    # Create packed coordinate data
    # With X-bits=1 and 1 bit for polarity, we have 2 bits per event
    # 16,384 events * 2 bits = 32,768 bits = 4,096 bytes
    bits_per_event = 1 + 1  # 1 bit for X, 1 bit for polarity
    total_bits = num_events * bits_per_event
    coordinate_bytes = (total_bits + 7) // 8  # Round up to nearest byte

    print(f"  Coordinate data: {coordinate_bytes} bytes ({total_bits} bits)")

    # Create some pattern in the packed data
    coord_data = bytearray(coordinate_bytes)
    for i in range(coordinate_bytes):
        # Create a pattern that alternates bits
        coord_data[i] = 0xAA if i % 2 == 0 else 0x55

    ecf_data.extend(coord_data)

    # Add timestamp delta encoding
    # Delta bits = 8 (for demo)
    delta_bits = 8
    ecf_data.append(delta_bits)

    # Add timestamp deltas (8 bits each)
    for i in range(num_events):
        delta = (i % 200) + 1  # Varying deltas 1-200
        ecf_data.append(delta)

    print(f"  Total ECF data size: {len(ecf_data)} bytes")
    return bytes(ecf_data)


def test_ecf_decoding():
    """Test our ECF decoder with the synthetic packed data"""

    print("\n" + "=" * 60)
    print("Testing ECF Decoder with Packed Coordinates")
    print("=" * 60)

    # Create test data
    ecf_data = create_packed_ecf_data()

    print(f"\nECF data (first 32 bytes): {ecf_data[:32].hex()}")

    # Try to import and test our ECF decoder directly
    try:
        # Import the Rust module
        import evlib

        formats = evlib.formats

        # Check if we have the test_prophesee_ecf_decode function
        if hasattr(formats, "test_prophesee_ecf_decode"):
            print("\n✓ Found test_prophesee_ecf_decode function")

            # Test decoding
            print("\nDecoding ECF data...")
            result = formats.test_prophesee_ecf_decode(ecf_data, debug=True)

            print(f"\n✓ Successfully decoded {len(result)} events")

            # Show first few events
            for i in range(min(10, len(result))):
                x, y, p, t = result[i]
                print(f"  Event {i}: x={x}, y={y}, p={p}, t={t}")

            # Check if we're getting real coordinates instead of placeholders
            x_values = [event[0] for event in result[:100]]  # First 100 X coordinates
            y_values = [event[1] for event in result[:100]]  # First 100 Y coordinates

            print("\nCoordinate analysis (first 100 events):")
            print(f"  X range: {min(x_values)} - {max(x_values)}")
            print(f"  Y range: {min(y_values)} - {max(y_values)}")
            print(f"  Unique X values: {len(set(x_values))}")
            print(f"  Unique Y values: {len(set(y_values))}")

            if len(set(x_values)) == 1 and len(set(y_values)) == 1:
                print(
                    "  ⚠️  All coordinates are the same - this suggests placeholder data"
                )
            elif len(set(x_values)) <= 2 and len(set(y_values)) <= 2:
                print(
                    "  ✓ Limited coordinate values - this matches X-bits=1, Y-bits=0 encoding"
                )
            else:
                print("  ✓ Variable coordinates - successfully decoded real data")

        else:
            print("\n⚠️  test_prophesee_ecf_decode function not found")
            print(
                "Available functions:",
                [attr for attr in dir(formats) if not attr.startswith("_")],
            )

    except ImportError as e:
        print(f"\n❌ Failed to import evlib.formats: {e}")
    except Exception as e:
        print(f"\n❌ Error during ECF decoding: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run the test"""
    print("ECF Packed Coordinate Decoding Test")
    print("Tests Y-bits=0, X-bits=1 format with 16,384 events\n")

    test_ecf_decoding()

    print("\n" + "=" * 60)
    print("Test complete.")


if __name__ == "__main__":
    main()
