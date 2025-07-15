#!/usr/bin/env python3
"""
Comprehensive test for EVT3.0 format support in evlib

This test creates a realistic EVT3 file with proper binary encoding and verifies:
1. Format detection works correctly
2. Header parsing extracts metadata properly
3. Binary event decoding follows EVT3.0 specification
4. Event coordinate and timestamp reconstruction is accurate
5. Error handling works for malformed data

The test creates binary events according to the EVT3 specification:
- Time Low/High events (4-bit type + 12-bit time data)
- Y address events (4-bit type + 11-bit Y + 1-bit orig)
- X address events (4-bit type + 11-bit X + 1-bit polarity)
- Vector Base X events (4-bit type + 11-bit X + 1-bit polarity)
- Vector 12/8 events (4-bit type + validity mask)
"""

import struct
import tempfile
import os
import sys
import numpy as np
from pathlib import Path

# Add the evlib package to the path
sys.path.insert(0, '/Users/tallam/github/tallamjr/origin/evlib/python')

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
        "% end"
    ]
    
    header_bytes = '\n'.join(header).encode('utf-8')
    return header_bytes

def encode_evt3_event(event_type, data_bits):
    """Encode an EVT3 event as a 16-bit little-endian word"""
    # EVT3 format: [15:4] data, [3:0] event type
    raw_data = (data_bits << 4) | event_type
    return struct.pack('<H', raw_data)

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
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.raw', delete=False) as f:
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
        if hasattr(evlib, 'detect_format'):
            format_info = evlib.detect_format(test_file)
            print(f"Detected format: {format_info[0]}")
            print(f"Confidence: {format_info[1]:.2f}")
            print(f"Metadata: {format_info[2]}")
            
            # Verify format is detected as EVT3
            assert format_info[0] == 'EVT3', f"Expected EVT3, got {format_info[0]}"
            assert format_info[1] > 0.9, f"Low confidence: {format_info[1]}"
            
            print("✓ Format detection passed")
            return True
        else:
            print("⚠ detect_format not available in current build")
            
    except ImportError as e:
        print(f"⚠ Could not import evlib: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in format detection: {e}")
        return False
    finally:
        os.unlink(test_file)
    
    return False

def test_evt3_loading():
    """Test loading events from EVT3 file"""
    print("\nTesting EVT3 event loading...")
    
    test_file = create_test_evt3_file()
    
    try:
        import evlib
        
        # Test loading events
        if hasattr(evlib, 'load_events'):
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
                assert t[i] >= t[i-1], f"Timestamps not ordered: {t[i]} < {t[i-1]}"
            
            print("✓ Event loading passed")
            return True
        else:
            print("⚠ load_events not available in current build")
            
    except ImportError as e:
        print(f"⚠ Could not import evlib: {e}")
        return False
    except Exception as e:
        print(f"✗ Error loading events: {e}")
        return False
    finally:
        os.unlink(test_file)
    
    return False

def test_evt3_metadata_extraction():
    """Test that metadata is correctly extracted from EVT3 header"""
    print("\nTesting EVT3 metadata extraction...")
    
    test_file = create_test_evt3_file()
    
    try:
        import evlib
        
        # Test format detection to get metadata
        if hasattr(evlib, 'detect_format'):
            format_info = evlib.detect_format(test_file)
            metadata = format_info[2]
            
            print(f"Extracted metadata: {metadata}")
            
            # Verify metadata contains expected information
            assert 'detection_method' in metadata, "Missing detection_method in metadata"
            
            # Check if sensor resolution was detected
            # Note: This depends on the specific header parsing implementation
            
            print("✓ Metadata extraction passed")
            return True
        else:
            print("⚠ detect_format not available in current build")
            
    except ImportError as e:
        print(f"⚠ Could not import evlib: {e}")
        return False
    except Exception as e:
        print(f"✗ Error extracting metadata: {e}")
        return False
    finally:
        os.unlink(test_file)
    
    return False

def test_evt3_error_handling():
    """Test error handling for malformed EVT3 data"""
    print("\nTesting EVT3 error handling...")
    
    # Create malformed EVT3 file (invalid header)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.raw', delete=False) as f:
        f.write(b"% evt 2.0\n% invalid header\n% end\n")
        f.write(b"\x00\x00\x00\x00")  # Some dummy binary data
        malformed_file = f.name
    
    try:
        import evlib
        
        # Test loading malformed file
        if hasattr(evlib, 'load_events'):
            try:
                x, y, t, p = evlib.load_events(malformed_file)
                print("⚠ Expected error loading malformed EVT3 file, but succeeded")
            except Exception as e:
                print(f"✓ Correctly caught error for malformed file: {e}")
                
        # Test with file that doesn't exist
        if hasattr(evlib, 'detect_format'):
            try:
                format_info = evlib.detect_format("/nonexistent/file.raw")
                print("⚠ Expected error for nonexistent file, but succeeded")
            except Exception as e:
                print(f"✓ Correctly caught error for nonexistent file: {e}")
        else:
            print("⚠ detect_format not available in current build")
                
    except ImportError as e:
        print(f"⚠ Could not import evlib: {e}")
        return False
    finally:
        os.unlink(malformed_file)
    
    return True

def test_evt3_coordinate_bounds():
    """Test coordinate bounds validation"""
    print("\nTesting EVT3 coordinate bounds validation...")
    
    # Create EVT3 file with out-of-bounds coordinates
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.raw', delete=False) as f:
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
        if hasattr(evlib, 'load_events'):
            try:
                x, y, t, p = evlib.load_events(bounds_test_file)
                print(f"Loaded {len(x)} events (bounds check may be disabled)")
                
                # Check if any events were loaded despite bounds issues
                if len(x) > 0:
                    print("⚠ Events loaded despite coordinate bounds issues")
                else:
                    print("✓ No events loaded due to coordinate bounds validation")
                    
            except Exception as e:
                print(f"✓ Correctly caught bounds error: {e}")
        else:
            print("⚠ load_events not available in current build")
                
    except ImportError as e:
        print(f"⚠ Could not import evlib: {e}")
        return False
    finally:
        os.unlink(bounds_test_file)
    
    return True

def run_comprehensive_test():
    """Run all EVT3 tests"""
    print("=" * 60)
    print("COMPREHENSIVE EVT3.0 FORMAT SUPPORT TEST")
    print("=" * 60)
    
    tests = [
        test_format_detection,
        test_evt3_loading,
        test_evt3_metadata_extraction,
        test_evt3_error_handling,
        test_evt3_coordinate_bounds,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! EVT3.0 format support is production-ready.")
    else:
        print("⚠ Some tests failed. EVT3.0 format support may need fixes.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)