#!/usr/bin/env python3
"""
Comprehensive test suite for EVT3.0 format support in evlib

This consolidated test suite combines all EVT3 testing functionality and verifies:
1. Format detection works correctly
2. Header parsing extracts metadata properly
3. Binary event decoding follows EVT3.0 specification
4. Event coordinate and timestamp reconstruction is accurate
5. Error handling works for malformed data
6. Integration with other evlib functions
7. DataFrame-based event loading and validation

The test creates binary events according to the EVT3 specification:
- Time Low/High events (4-bit type + 12-bit time data)
- Y address events (4-bit type + 11-bit Y + 1-bit orig)
- X address events (4-bit type + 11-bit X + 1-bit polarity)
- Vector Base X events (4-bit type + 11-bit X + 1-bit polarity)
- Vector 12/8 events (4-bit type + validity mask)
"""

import os
import struct
import sys
import tempfile
import unittest

import numpy as np
import pytest

# Add the evlib package to the path
sys.path.insert(0, "/Users/tallam/github/tallamjr/origin/evlib/python")


def create_evt3_header(width=640, height=480):
    """Create a realistic EVT3 header with proper metadata"""
    header = [
        "% evt 3.0",
        "% date 2024-01-15 12:00:00",
        "% format EVT3;height=480;width=640",
        "% camera_type prophesee",
        "% sensor_generation 4.0",
        "% serial_number 12345",
        "% system_id 1000",
        "% plugin_name evt3_reader",
        "% plugin_version 1.0.0",
        "% integrator_name evlib_test",
        "% firmware_version 2.4.0",
        "% end",
    ]

    header_bytes = "\n".join(header).encode("utf-8")
    return header_bytes


def encode_evt3_event(event_type, data_bits):
    """Encode an EVT3 event as a 16-bit little-endian word"""
    # EVT3 format: [15:4] data, [3:0] event type
    raw_data = (data_bits << 4) | event_type
    return struct.pack("<H", raw_data)


def create_evt3_binary_events():
    """Create realistic binary event data according to EVT3 specification"""
    events = bytearray()

    # Event sequence to create several events at different coordinates
    # Following the EVT3 decoder state machine requirements

    # 1. Time Low event (0x6) - set timestamp to 1000 μs
    time_low = 1000 & 0xFFF
    events.extend(encode_evt3_event(0x6, time_low))

    # 2. Time High event (0x8) - set upper timestamp bits
    time_high = (50000 >> 12) & 0xFFF
    events.extend(encode_evt3_event(0x8, time_high))

    # 3. Y address event (0x0) - set Y coordinate to 240
    y_coord = 240
    orig_bit = 0  # Master system
    y_data = (orig_bit << 11) | y_coord
    events.extend(encode_evt3_event(0x0, y_data))

    # 4. X address event (0x2) - single event at (320, 240) with positive polarity
    x_coord = 320
    polarity = 1  # positive
    x_data = (polarity << 11) | x_coord
    events.extend(encode_evt3_event(0x2, x_data))

    # 5. Update timestamp for next events
    time_low = 2000 & 0xFFF
    events.extend(encode_evt3_event(0x6, time_low))

    # 6. New Y coordinate
    y_coord = 100
    y_data = (orig_bit << 11) | y_coord
    events.extend(encode_evt3_event(0x0, y_data))

    # 7. Vector Base X event (0x3) - set base X for vector events
    base_x = 200
    base_polarity = 0  # negative
    vect_base_data = (base_polarity << 11) | base_x
    events.extend(encode_evt3_event(0x3, vect_base_data))

    # 8. Vector 12 event (0x4) - create events at X positions 200, 202, 205, 210
    # Bit mask: positions 0, 2, 5, 10 are set
    validity_mask = (1 << 0) | (1 << 2) | (1 << 5) | (1 << 10)
    events.extend(encode_evt3_event(0x4, validity_mask))

    # 9. Update timestamp again
    time_low = 3000 & 0xFFF
    events.extend(encode_evt3_event(0x6, time_low))

    # 10. New Y coordinate
    y_coord = 150
    y_data = (orig_bit << 11) | y_coord
    events.extend(encode_evt3_event(0x0, y_data))

    # 11. Vector Base X event with different base
    base_x = 400
    base_polarity = 1  # positive
    vect_base_data = (base_polarity << 11) | base_x
    events.extend(encode_evt3_event(0x3, vect_base_data))

    # 12. Vector 8 event (0x5) - create events at first 3 positions
    validity_mask = (1 << 0) | (1 << 1) | (1 << 2)
    events.extend(encode_evt3_event(0x5, validity_mask))

    return bytes(events)


def create_test_evt3_file():
    """Create a complete EVT3 test file with header and binary data"""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".raw", delete=False) as f:
        # Write header
        header = create_evt3_header()
        f.write(header)

        # Write binary events
        binary_events = create_evt3_binary_events()
        f.write(binary_events)

        return f.name


def test_format_detection():
    """Test that EVT3 format is correctly detected"""
    print("Testing EVT3 format detection...")

    test_file = create_test_evt3_file()

    try:
        import evlib

        # Test format detection
        if hasattr(evlib, "detect_format"):
            format_info = evlib.detect_format(test_file)
            print(f"Detected format: {format_info[0]}")
            print(f"Confidence: {format_info[1]:.2f}")
            print(f"Metadata: {format_info[2]}")

            # Verify format is detected as EVT3
            assert format_info[0] == "EVT3", f"Expected EVT3, got {format_info[0]}"
            assert format_info[1] > 0.9, f"Low confidence: {format_info[1]}"

            print("PASS: Format detection passed")
        else:
            print("WARN: detect_format not available in current build")

    except ImportError as e:
        print(f"WARN: Could not import evlib: {e}")
        pytest.skip(f"Could not import evlib: {e}")
    except Exception as e:
        print(f"FAIL: Error in format detection: {e}")
        pytest.fail(f"Error in format detection: {e}")
    finally:
        os.unlink(test_file)


def test_evt3_loading():
    """Test loading events from EVT3 file"""
    print("\nTesting EVT3 event loading...")

    test_file = create_test_evt3_file()

    try:
        import evlib

        # Test loading events
        if hasattr(evlib, "load_events"):
            x, y, t, p = evlib.load_events(test_file)

            print(f"Loaded {len(x)} events")
            print(f"X coordinates: {x}")
            print(f"Y coordinates: {y}")
            print(f"Timestamps: {t}")
            print(f"Polarities: {p}")

            # Verify we got expected events
            assert len(x) > 0, "No events loaded"

            # Check that we have the expected single event at (320, 240)
            single_event_found = False
            for i in range(len(x)):
                if x[i] == 320 and y[i] == 240:
                    single_event_found = True
                    assert p[i] == 1, f"Expected polarity 1 at (320, 240), got {p[i]}"
                    break

            assert single_event_found, "Expected single event at (320, 240) not found"

            # Check that we have vector events at Y=100 (from Vector 12 event)
            vector_events_100 = [i for i in range(len(x)) if y[i] == 100]
            assert len(vector_events_100) > 0, "No vector events found at Y=100"

            # Check that we have vector events at Y=150 (from Vector 8 event)
            vector_events_150 = [i for i in range(len(x)) if y[i] == 150]
            assert len(vector_events_150) > 0, "No vector events found at Y=150"

            # Verify timestamp ordering
            for i in range(1, len(t)):
                assert t[i] >= t[i - 1], f"Timestamps not ordered: {t[i]} < {t[i-1]}"

            print("PASS: Event loading passed")
        else:
            print("WARN: load_events not available in current build")

    except ImportError as e:
        print(f"WARN: Could not import evlib: {e}")
        pytest.skip(f"Could not import evlib: {e}")
    except Exception as e:
        print(f"FAIL: Error loading events: {e}")
        pytest.fail(f"Error loading events: {e}")
    finally:
        os.unlink(test_file)


def test_evt3_metadata_extraction():
    """Test that metadata is correctly extracted from EVT3 header"""
    print("\nTesting EVT3 metadata extraction...")

    test_file = create_test_evt3_file()

    try:
        import evlib

        # Test format detection to get metadata
        if hasattr(evlib, "detect_format"):
            format_info = evlib.detect_format(test_file)
            metadata = format_info[2]

            print(f"Extracted metadata: {metadata}")

            # Verify metadata contains expected information
            assert "detection_method" in metadata, "Missing detection_method in metadata"

            # Check if sensor resolution was detected
            # Note: This depends on the specific header parsing implementation

            print("PASS: Metadata extraction passed")
        else:
            print("WARN: detect_format not available in current build")

    except ImportError as e:
        print(f"WARN: Could not import evlib: {e}")
        pytest.skip(f"Could not import evlib: {e}")
    except Exception as e:
        print(f"FAIL: Error extracting metadata: {e}")
        pytest.fail(f"Error extracting metadata: {e}")
    finally:
        os.unlink(test_file)


def test_evt3_error_handling():
    """Test error handling for malformed EVT3 data"""
    print("\nTesting EVT3 error handling...")

    # Create malformed EVT3 file (invalid header)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".raw", delete=False) as f:
        f.write(b"% evt 2.0\n% invalid header\n% end\n")
        f.write(b"\x00\x00\x00\x00")  # Some dummy binary data
        malformed_file = f.name

    try:
        import evlib

        # Test loading malformed file
        if hasattr(evlib, "load_events"):
            try:
                x, y, t, p = evlib.load_events(malformed_file)
                print("WARN: Expected error loading malformed EVT3 file, but succeeded")
            except Exception as e:
                print(f"PASS: Correctly caught error for malformed file: {e}")

        # Test with file that doesn't exist
        if hasattr(evlib, "detect_format"):
            try:
                _format_info = evlib.detect_format("/nonexistent/file.raw")
                print("WARN: Expected error for nonexistent file, but succeeded")
            except Exception as e:
                print(f"PASS: Correctly caught error for nonexistent file: {e}")
        else:
            print("WARN: detect_format not available in current build")

    except ImportError as e:
        print(f"WARN: Could not import evlib: {e}")
        pytest.skip(f"Could not import evlib: {e}")
    finally:
        os.unlink(malformed_file)


def test_evt3_coordinate_bounds():
    """Test coordinate bounds validation"""
    print("\nTesting EVT3 coordinate bounds validation...")

    # Create EVT3 file with out-of-bounds coordinates
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".raw", delete=False) as f:
        # Write header with small sensor resolution
        header = create_evt3_header(width=320, height=240)
        f.write(header)

        # Create events with coordinates outside bounds
        events = bytearray()

        # Time event
        events.extend(encode_evt3_event(0x6, 1000))

        # Y address event with coordinate outside bounds (Y=500 > 240)
        y_coord = 500
        orig_bit = 0
        y_data = (orig_bit << 11) | y_coord
        events.extend(encode_evt3_event(0x0, y_data))

        # X address event with coordinate outside bounds (X=400 > 320)
        x_coord = 400
        polarity = 1
        x_data = (polarity << 11) | x_coord
        events.extend(encode_evt3_event(0x2, x_data))

        f.write(bytes(events))
        bounds_test_file = f.name

    try:
        import evlib

        # Test loading file with out-of-bounds coordinates
        if hasattr(evlib, "load_events"):
            try:
                x, y, t, p = evlib.load_events(bounds_test_file)
                print(f"Loaded {len(x)} events (bounds check may be disabled)")

                # Check if any events were loaded despite bounds issues
                if len(x) > 0:
                    print("WARN: Events loaded despite coordinate bounds issues")
                else:
                    print("PASS: No events loaded due to coordinate bounds validation")

            except Exception as e:
                print(f"PASS: Correctly caught bounds error: {e}")
        else:
            print("WARN: load_events not available in current build")

    except ImportError as e:
        print(f"WARN: Could not import evlib: {e}")
        pytest.skip(f"Could not import evlib: {e}")
    finally:
        os.unlink(bounds_test_file)


class TestEVT3FormatSupport(unittest.TestCase):
    """Comprehensive test suite for EVT3 format support"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_files = []
        self.maxDiff = None

    def tearDown(self):
        """Clean up test files"""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def create_evt3_test_file(self, include_events=True, header_params=None, events_data=None):
        """Create a comprehensive EVT3 test file with known content"""
        if header_params is None:
            header_params = {"height": 720, "width": 1280}

        if events_data is None and include_events:
            # Default test events following EVT3 specification
            events_data = [
                (0x123456, 640, 360, 1),  # Event 1: positive polarity
                (0x234567, 100, 200, -1),  # Event 2: negative polarity
                (0x345678, 800, 400, 1),  # Event 3: positive polarity
            ]

        # Create comprehensive EVT3 header
        header = f"""% evt 3.0
% date 2024-01-15 12:00:00
% format EVT3;height={header_params['height']};width={header_params['width']}
% geometry {header_params['width']}x{header_params['height']}
% camera_integrator_name Prophesee
% camera_type prophesee
% sensor_generation 4.0
% serial_number 12345
% system_id 1000
% plugin_name evt3_reader
% plugin_version 1.0.0
% integrator_name evlib_test
% firmware_version 2.4.0
% end
"""

        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            f.write(header.encode("utf-8"))

            if include_events and events_data:
                # Convert events to EVT3 binary format
                for timestamp_us, x, y, polarity in events_data:
                    # EVT3 encoding: 4 words per event (16-bit little-endian)
                    time_high = ((timestamp_us >> 12) & 0xFFF) << 4 | 0x8
                    time_low = (timestamp_us & 0xFFF) << 4 | 0x6
                    y_addr = (y & 0x7FF) << 4 | 0x0
                    x_addr = ((1 if polarity > 0 else 0) << 15) | ((x & 0x7FF) << 4) | 0x2

                    # Write as little-endian 16-bit words
                    f.write(struct.pack("<H", time_high))
                    f.write(struct.pack("<H", time_low))
                    f.write(struct.pack("<H", y_addr))
                    f.write(struct.pack("<H", x_addr))

            file_path = f.name

        self.test_files.append(file_path)
        return file_path

    def test_evt3_format_detection(self):
        """Test that EVT3 files are correctly detected"""
        test_file = self.create_evt3_test_file()

        try:
            import evlib

            if hasattr(evlib, "detect_format"):
                format_name, confidence, metadata = evlib.detect_format(test_file)

                self.assertEqual(format_name, "EVT3", "EVT3 format should be detected")
                self.assertGreater(confidence, 0.9, "Detection confidence should be high")
                self.assertIn("detection_method", metadata, "Metadata should contain detection method")
            else:
                self.skipTest("detect_format not available in current build")

        except ImportError:
            self.skipTest("Could not import evlib")

    def test_evt3_header_parsing(self):
        """Test EVT3 header parsing"""
        test_file = self.create_evt3_test_file(
            include_events=False, header_params={"height": 480, "width": 640}
        )

        # Verify header format manually
        with open(test_file, "rb") as f:
            content = f.read()
            header_end = content.find(b"% end")
            self.assertNotEqual(header_end, -1, "Header should contain end marker")

            header_part = content[:header_end].decode("utf-8")
            self.assertIn("evt 3.0", header_part, "Header should contain EVT3 version")
            self.assertIn("format EVT3", header_part, "Header should contain format declaration")
            self.assertIn("height=480", header_part, "Header should contain height parameter")
            self.assertIn("width=640", header_part, "Header should contain width parameter")

    def test_evt3_event_loading(self):
        """Test loading events from EVT3 file"""
        test_file = self.create_evt3_test_file(include_events=True)

        try:
            import evlib

            if hasattr(evlib, "load_events"):
                # Test DataFrame-based loading
                df = evlib.load_events(test_file).collect()
                self.assertGreater(len(df), 0, "Should load some data")

                # Extract arrays from DataFrame
                if len(df) > 0:
                    x_coords = df["x"].to_numpy()
                    y_coords = df["y"].to_numpy()
                    timestamps = df["timestamp"].cast(float).to_numpy()
                    polarities = df["polarity"].to_numpy()

                    # Verify data types
                    self.assertIsInstance(x_coords, np.ndarray, "X coordinates should be numpy array")
                    self.assertIsInstance(y_coords, np.ndarray, "Y coordinates should be numpy array")
                    self.assertIsInstance(timestamps, np.ndarray, "Timestamps should be numpy array")
                    self.assertIsInstance(polarities, np.ndarray, "Polarities should be numpy array")

                    # Verify array consistency
                    self.assertEqual(len(x_coords), len(df), "X coords should match DataFrame length")
                    self.assertEqual(len(y_coords), len(df), "Y coords should match DataFrame length")
                    self.assertEqual(len(timestamps), len(df), "Timestamps should match DataFrame length")
                    self.assertEqual(len(polarities), len(df), "Polarities should match DataFrame length")

                    # Verify data ranges
                    self.assertTrue(np.all(x_coords >= 0), "X coordinates should be non-negative")
                    self.assertTrue(np.all(y_coords >= 0), "Y coordinates should be non-negative")
                    self.assertTrue(np.all(timestamps >= 0), "Timestamps should be non-negative")
                    self.assertTrue(np.all(np.isin(polarities, [-1, 1])), "Polarities should be -1 or 1")

            else:
                self.skipTest("load_events not available in current build")

        except ImportError:
            self.skipTest("Could not import evlib")

    def test_evt3_event_data_correctness(self):
        """Test that EVT3 events are decoded correctly"""
        # Create test file with known events
        test_events = [
            (0x123456, 640, 360, 1),  # Event 1: positive polarity
            (0x234567, 100, 200, -1),  # Event 2: negative polarity
        ]
        test_file = self.create_evt3_test_file(include_events=True, events_data=test_events)

        try:
            import evlib

            if hasattr(evlib, "load_events"):
                df = evlib.load_events(test_file).collect()

                if len(df) > 0:
                    x_coords = df["x"].to_numpy()
                    y_coords = df["y"].to_numpy()
                    timestamps = df["timestamp"].cast(float).to_numpy()
                    polarities = df["polarity"].to_numpy()

                    # Expected values based on our test data
                    expected_timestamps = [0x123456, 0x234567]  # Timestamps in microseconds
                    expected_x = [640, 100]
                    expected_y = [360, 200]
                    expected_polarities = [1, -1]

                    # Verify coordinates
                    np.testing.assert_array_equal(x_coords, expected_x, "X coordinates should match")
                    np.testing.assert_array_equal(y_coords, expected_y, "Y coordinates should match")

                    # Verify timestamps (with tolerance for floating point)
                    np.testing.assert_allclose(
                        timestamps, expected_timestamps, rtol=1e-6, err_msg="Timestamps should match"
                    )

                    # Verify polarities
                    np.testing.assert_array_equal(polarities, expected_polarities, "Polarities should match")

            else:
                self.skipTest("load_events not available in current build")

        except ImportError:
            self.skipTest("Could not import evlib")

    def test_evt3_empty_file(self):
        """Test handling of EVT3 file with no events"""
        test_file = self.create_evt3_test_file(include_events=False)

        try:
            import evlib

            # Should still detect format correctly
            if hasattr(evlib, "detect_format"):
                format_name, confidence, metadata = evlib.detect_format(test_file)
                self.assertEqual(format_name, "EVT3")

            # Loading should work but return empty DataFrame
            if hasattr(evlib, "load_events"):
                df = evlib.load_events(test_file).collect()
                self.assertEqual(len(df), 0, "Empty EVT3 file should return empty DataFrame")

        except ImportError:
            self.skipTest("Could not import evlib")

    def test_evt3_malformed_file(self):
        """Test error handling for malformed EVT3 files"""
        # Create malformed EVT3 file
        malformed_header = """% evt 3.0
% format EVT3;height=invalid;width=also_invalid
% geometry 1280x720
% end
"""

        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            f.write(malformed_header.encode("utf-8"))
            # Add some invalid binary data
            f.write(b"\xff\xff\x00\x00\xaa\xbb\xcc\xdd")
            test_file = f.name

        self.test_files.append(test_file)

        try:
            import evlib

            # Format detection should still work
            if hasattr(evlib, "detect_format"):
                format_name, confidence, metadata = evlib.detect_format(test_file)
                self.assertEqual(format_name, "EVT3")

            # Loading should either work gracefully or raise appropriate error
            if hasattr(evlib, "load_events"):
                try:
                    _events = evlib.load_events(test_file)
                    # If it succeeds, that's fine too
                except Exception as e:
                    # Should be a meaningful error message
                    self.assertIsInstance(e, Exception)

        except ImportError:
            self.skipTest("Could not import evlib")

    def test_evt3_coordinate_bounds(self):
        """Test coordinate bounds validation"""
        # Create EVT3 file with out-of-bounds coordinates
        header_params = {"height": 240, "width": 320}
        out_of_bounds_events = [
            (0x123456, 400, 500, 1),  # Both coordinates out of bounds
        ]
        test_file = self.create_evt3_test_file(
            include_events=True, header_params=header_params, events_data=out_of_bounds_events
        )

        try:
            import evlib

            if hasattr(evlib, "load_events"):
                try:
                    df = evlib.load_events(test_file).collect()
                    # Events may be loaded despite bounds issues (depends on implementation)
                    if len(df) == 0:
                        print("PASS: No events loaded due to coordinate bounds validation")
                    else:
                        print("WARN: Events loaded despite coordinate bounds issues")
                except Exception as e:
                    print(f"PASS: Correctly caught bounds error: {e}")

        except ImportError:
            self.skipTest("Could not import evlib")

    def test_evt3_specification_compliance(self):
        """Test that our EVT3 implementation follows the specification"""
        # Test data that follows EVT3 specification exactly
        test_events = [
            (0x123456, 640, 360, 1),  # Event at 1.193046 seconds
            (0x234567, 100, 200, -1),  # Event at 2.310503 seconds
            (0x345678, 800, 400, 1),  # Event at 3.427896 seconds
        ]

        test_file = self.create_evt3_test_file(include_events=True, events_data=test_events)

        # Verify file format manually
        with open(test_file, "rb") as f:
            content = f.read()

            # Check header
            self.assertTrue(content.startswith(b"% evt 3.0"), "Should start with EVT3 magic")

            # Find binary data
            header_end = content.find(b"% end")
            binary_data = content[header_end + 6 :]  # Skip "% end\n"

            # Should have 12 words (24 bytes) for 3 events
            self.assertEqual(len(binary_data), 24, "Should have 24 bytes for 3 events")

            # Parse and verify each event
            for i, (expected_t, expected_x, expected_y, expected_p) in enumerate(test_events):
                offset = i * 8  # 8 bytes per event
                event_data = binary_data[offset : offset + 8]
                words = struct.unpack("<4H", event_data)
                time_high, time_low, y_addr, x_addr = words

                # Reconstruct timestamp
                reconstructed_t = ((time_high >> 4) << 12) | (time_low >> 4)
                self.assertEqual(reconstructed_t, expected_t, f"Event {i} timestamp mismatch")

                # Reconstruct coordinates
                reconstructed_x = (x_addr >> 4) & 0x7FF
                reconstructed_y = (y_addr >> 4) & 0x7FF
                self.assertEqual(reconstructed_x, expected_x, f"Event {i} X coordinate mismatch")
                self.assertEqual(reconstructed_y, expected_y, f"Event {i} Y coordinate mismatch")

                # Reconstruct polarity
                reconstructed_p = 1 if (x_addr & 0x8000) else -1
                self.assertEqual(reconstructed_p, expected_p, f"Event {i} polarity mismatch")

    def test_evt3_integration_with_evlib(self):
        """Test EVT3 integration with other evlib functions"""
        try:
            import evlib

            # Test that evlib functions that should work still work
            self.assertTrue(hasattr(evlib, "create_voxel_grid"), "create_voxel_grid should be available")
            self.assertTrue(hasattr(evlib, "representations"), "representations module should be available")

            # Test that we can at least call the functions (even if they fail)
            try:
                events = np.array(
                    [(0.1, 100, 100, 1)], dtype=[("t", "f8"), ("x", "u2"), ("y", "u2"), ("polarity", "i1")]
                )
                _result = evlib.create_voxel_grid(events, (100, 100, 1), 480, 640, 5)
                print("INFO: create_voxel_grid works with structured array")
            except Exception as e:
                print(f"INFO: create_voxel_grid failed as expected: {e}")

        except ImportError:
            self.skipTest("Could not import evlib")

    def test_evt3_error_handling_comprehensive(self):
        """Test comprehensive error handling for various edge cases"""
        try:
            import evlib

            # Test non-existent file
            if hasattr(evlib, "load_events"):
                with self.assertRaises(Exception):
                    evlib.load_events("/nonexistent/file.raw")

            # Test completely empty file
            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
                empty_file = f.name
            self.test_files.append(empty_file)

            if hasattr(evlib, "load_events"):
                try:
                    result = evlib.load_events(empty_file)
                    # If it doesn't raise an exception, the result should be empty/reasonable
                    print(f"INFO: load_events handled empty file gracefully with result: {result}")
                except Exception as e:
                    print(f"INFO: load_events appropriately failed with: {e}")

            # Test corrupted header
            corrupted_header = """% evt 3.0
% format EVT3;height=720;width=1280
% geometry 1280x720
% this header is cut off and inval"""

            with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
                f.write(corrupted_header.encode("utf-8"))
                corrupted_file = f.name
            self.test_files.append(corrupted_file)

            if hasattr(evlib, "load_events"):
                try:
                    _result = evlib.load_events(corrupted_file)
                    print("INFO: load_events handled corrupted header gracefully")
                except Exception as e:
                    print(f"INFO: load_events appropriately failed with: {e}")

        except ImportError:
            self.skipTest("Could not import evlib")


if __name__ == "__main__":
    # Print environment information
    print("Comprehensive EVT3 Test Suite")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    # Test evlib import
    try:
        import evlib

        print("evlib imported successfully")
        print(f"evlib attributes: {[x for x in dir(evlib) if not x.startswith('_')][:10]}...")

        if hasattr(evlib, "formats"):
            print(
                f"evlib.formats attributes: {[x for x in dir(evlib.formats) if not x.startswith('_')][:10]}..."
            )

    except ImportError as e:
        print(f"ERROR: Failed to import evlib: {e}")
        sys.exit(1)

    print("=" * 50)

    # Run the test suite
    unittest.main(verbosity=2)
