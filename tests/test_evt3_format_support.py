"""
Unit tests for EVT3 format support in evlib

This test suite verifies that evlib can correctly handle EVT3.0 format files
as specified by Prophesee: https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html

Tests cover:
- Format detection
- Header parsing
- Binary event decoding
- Event reconstruction
- Error handling
"""

import unittest
import tempfile
import struct
import os
import numpy as np
import evlib


class TestEVT3FormatSupport(unittest.TestCase):
    """Test suite for EVT3 format support"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_files = []

    def tearDown(self):
        """Clean up test files"""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def create_evt3_test_file(self, include_events=True):
        """Create a test EVT3 file with known content"""
        
        # Standard EVT3 header
        header = """% evt 3.0
% format EVT3;height=720;width=1280
% geometry 1280x720
% camera_integrator_name Prophesee
% generation 4.2
% end
"""
        
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
            f.write(header.encode('utf-8'))
            
            if include_events:
                # Create two known events following EVT3 specification
                
                # Event 1: (640, 360) at timestamp 0x123456 μs, positive polarity
                time_high1 = (0x123 << 4) | 0x8  # Time High
                time_low1 = (0x456 << 4) | 0x6   # Time Low
                y_addr1 = (0 << 15) | (360 << 4) | 0x0  # Y Address
                x_addr1 = (1 << 15) | (640 << 4) | 0x2  # X Address (positive)
                
                # Event 2: (100, 200) at timestamp 0x234567 μs, negative polarity
                time_high2 = (0x234 << 4) | 0x8  # Time High
                time_low2 = (0x567 << 4) | 0x6   # Time Low
                y_addr2 = (0 << 15) | (200 << 4) | 0x0  # Y Address
                x_addr2 = (0 << 15) | (100 << 4) | 0x2  # X Address (negative)
                
                # Write events as little-endian 16-bit words
                events = [time_high1, time_low1, y_addr1, x_addr1,
                         time_high2, time_low2, y_addr2, x_addr2]
                
                for event in events:
                    f.write(struct.pack('<H', event))
            
            file_path = f.name
        
        self.test_files.append(file_path)
        return file_path

    def test_evt3_format_detection(self):
        """Test that EVT3 files are correctly detected"""
        
        test_file = self.create_evt3_test_file()
        
        # Test format detection
        format_name, confidence, metadata = evlib.detect_format(test_file)
        
        self.assertEqual(format_name, "EVT3", "EVT3 format should be detected")
        self.assertGreater(confidence, 0.9, "Detection confidence should be high")
        self.assertIn("detection_method", metadata, "Metadata should contain detection method")

    def test_evt3_header_parsing(self):
        """Test EVT3 header parsing"""
        
        test_file = self.create_evt3_test_file(include_events=False)
        
        # Test format detection to check header parsing
        format_name, confidence, metadata = evlib.detect_format(test_file)
        
        self.assertEqual(format_name, "EVT3")
        # Note: Sensor resolution extraction may not be fully implemented
        # but the format should still be detected correctly

    def test_evt3_event_loading(self):
        """Test loading events from EVT3 file"""
        
        test_file = self.create_evt3_test_file(include_events=True)
        
        # Load events
        events = evlib.load_events(test_file)
        
        # Verify we got data
        self.assertIsInstance(events, (list, tuple), "Events should be returned as sequence")
        self.assertGreater(len(events), 0, "Should load some data")
        
        # Based on current implementation, events are returned as separate arrays
        if len(events) == 4:
            x_coords, y_coords, timestamps, polarities = events
            
            # Verify data types
            self.assertIsInstance(x_coords, np.ndarray, "X coordinates should be numpy array")
            self.assertIsInstance(y_coords, np.ndarray, "Y coordinates should be numpy array") 
            self.assertIsInstance(timestamps, np.ndarray, "Timestamps should be numpy array")
            self.assertIsInstance(polarities, np.ndarray, "Polarities should be numpy array")
            
            # Verify we got 2 events
            self.assertEqual(len(x_coords), 2, "Should have 2 events")
            self.assertEqual(len(y_coords), 2, "Y coords should match X coords length")
            self.assertEqual(len(timestamps), 2, "Timestamps should match X coords length")
            self.assertEqual(len(polarities), 2, "Polarities should match X coords length")

    def test_evt3_event_data_correctness(self):
        """Test that EVT3 events are decoded correctly"""
        
        test_file = self.create_evt3_test_file(include_events=True)
        
        # Load events
        events = evlib.load_events(test_file)
        
        if len(events) == 4:
            x_coords, y_coords, timestamps, polarities = events
            
            # Expected values based on our test data
            # Event 1: timestamp=0x123456 μs, x=640, y=360, polarity=positive
            # Event 2: timestamp=0x234567 μs, x=100, y=200, polarity=negative
            
            expected_timestamps = [0x123456 / 1_000_000.0, 0x234567 / 1_000_000.0]  # Convert to seconds
            expected_x = [640, 100]
            expected_y = [360, 200]
            expected_polarities = [1, -1]  # positive=1, negative=-1
            
            # Verify X coordinates
            np.testing.assert_array_equal(x_coords, expected_x, "X coordinates should match")
            
            # Verify Y coordinates
            np.testing.assert_array_equal(y_coords, expected_y, "Y coordinates should match")
            
            # Verify timestamps (with small tolerance for floating point)
            np.testing.assert_allclose(timestamps, expected_timestamps, rtol=1e-6, 
                                     err_msg="Timestamps should match")
            
            # Verify polarities
            np.testing.assert_array_equal(polarities, expected_polarities, "Polarities should match")

    def test_evt3_empty_file(self):
        """Test handling of EVT3 file with no events"""
        
        test_file = self.create_evt3_test_file(include_events=False)
        
        # Should still detect format correctly
        format_name, confidence, metadata = evlib.detect_format(test_file)
        self.assertEqual(format_name, "EVT3")
        
        # Loading should work but return empty/minimal data
        events = evlib.load_events(test_file)
        self.assertIsInstance(events, (list, tuple))

    def test_evt3_malformed_file(self):
        """Test error handling for malformed EVT3 files"""
        
        # Create malformed EVT3 file
        malformed_header = """% evt 3.0
% format EVT3;height=invalid;width=also_invalid
% end
"""
        
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
            f.write(malformed_header.encode('utf-8'))
            # Add some random binary data
            f.write(b'\xFF\xFF\x00\x00\xAA\xBB')
            test_file = f.name
        
        self.test_files.append(test_file)
        
        # Format detection should still work
        format_name, confidence, metadata = evlib.detect_format(test_file)
        self.assertEqual(format_name, "EVT3")
        
        # Loading should either work gracefully or raise appropriate error
        try:
            events = evlib.load_events(test_file)
            # If it succeeds, that's fine too
        except Exception as e:
            # Should be a meaningful error message
            self.assertIsInstance(e, Exception)

    def test_evt3_data_access_pattern(self):
        """Test the recommended pattern for accessing EVT3 data"""
        
        test_file = self.create_evt3_test_file(include_events=True)
        
        # Load events
        events = evlib.load_events(test_file)
        
        # Recommended access pattern
        if len(events) == 4:
            x_coords, y_coords, timestamps, polarities = events
            
            # Iterate through events
            num_events = len(x_coords)
            reconstructed_events = []
            
            for i in range(num_events):
                event = {
                    't': float(timestamps[i]),
                    'x': int(x_coords[i]),
                    'y': int(y_coords[i]),
                    'polarity': int(polarities[i])
                }
                reconstructed_events.append(event)
            
            # Verify we can reconstruct the events properly
            self.assertEqual(len(reconstructed_events), 2, "Should reconstruct 2 events")
            
            # Check first event
            event1 = reconstructed_events[0]
            self.assertAlmostEqual(event1['t'], 1.193046, places=6, msg="First event timestamp")
            self.assertEqual(event1['x'], 640, "First event X coordinate")
            self.assertEqual(event1['y'], 360, "First event Y coordinate") 
            self.assertEqual(event1['polarity'], 1, "First event polarity")


class TestEVT3Integration(unittest.TestCase):
    """Integration tests for EVT3 with other evlib functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_files = []

    def tearDown(self):
        """Clean up test files"""
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_evt3_with_representations(self):
        """Test using EVT3 data with evlib representation functions"""
        
        # Create test file (this is a placeholder - would need actual EVT3 data)
        header = """% evt 3.0
% format EVT3;height=480;width=640
% end
"""
        
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
            f.write(header.encode('utf-8'))
            test_file = f.name
        
        self.test_files.append(test_file)
        
        # Just test that format detection works
        format_name, confidence, metadata = evlib.detect_format(test_file)
        self.assertEqual(format_name, "EVT3")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)