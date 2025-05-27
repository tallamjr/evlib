#!/usr/bin/env python3
"""
Test script for real-time event streaming Python bindings

This script tests that the Python bindings for real-time event streaming
work correctly without requiring an actual webcam.
"""

import sys
import traceback


def test_import():
    """Test that evlib can be imported"""
    try:
        import evlib  # noqa: F401

        print("✓ evlib imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import evlib: {e}")
        return False


def test_realtime_availability():
    """Test if real-time streaming is available"""
    try:
        import evlib

        available = evlib.simulation.is_realtime_available()

        if available:
            print("✓ Real-time streaming is available (GStreamer support detected)")
        else:
            print("⚠ Real-time streaming is not available (GStreamer support not detected)")

        return True
    except Exception as e:
        print(f"✗ Failed to check real-time availability: {e}")
        traceback.print_exc()
        return False


def test_realtime_config():
    """Test creating a real-time stream configuration"""
    try:
        import evlib

        PyRealtimeStreamConfig = evlib.simulation.PyRealtimeStreamConfig

        # Test default config
        config = PyRealtimeStreamConfig()
        print(f"✓ Default config created: {config}")

        # Test custom config
        config = PyRealtimeStreamConfig(
            target_fps=15.0,
            contrast_threshold=0.2,
            device_id=0,
            resolution=(320, 240),
            max_buffer_size=5000,
            auto_adjust_fps=True,
        )
        print(f"✓ Custom config created: {config}")

        # Test property access
        assert config.target_fps == 15.0
        assert config.contrast_threshold == 0.2
        assert config.device_id == 0
        assert config.resolution == (320, 240)
        assert config.max_buffer_size == 5000
        assert config.auto_adjust_fps

        # Test property modification
        config.target_fps = 25.0
        config.contrast_threshold = 0.25
        assert config.target_fps == 25.0
        assert config.contrast_threshold == 0.25

        print("✓ Config properties work correctly")
        return True

    except ImportError:
        print("⚠ Real-time streaming classes not available (expected without GStreamer)")
        return True
    except Exception as e:
        print(f"✗ Failed to test real-time config: {e}")
        traceback.print_exc()
        return False


def test_stream_creation():
    """Test creating a real-time event stream (without starting it)"""
    try:
        import evlib

        PyRealtimeStreamConfig = evlib.simulation.PyRealtimeStreamConfig
        PyRealtimeEventStream = evlib.simulation.PyRealtimeEventStream

        # Create config
        config = PyRealtimeStreamConfig(
            target_fps=10.0, contrast_threshold=0.3, device_id=0, resolution=(160, 120), auto_adjust_fps=False
        )

        # Create stream (this should work even without webcam)
        stream = PyRealtimeEventStream(config)
        print(f"✓ Real-time stream created: {stream}")

        # Test that stream is not streaming initially
        assert not stream.is_streaming()
        print("✓ Stream correctly reports not streaming initially")

        # Test getting empty events
        xs, ys, ts, ps = stream.get_events(max_count=10)
        assert len(xs) == 0 and len(ys) == 0 and len(ts) == 0 and len(ps) == 0
        print("✓ Empty event retrieval works")

        # Test getting stats
        stats = stream.get_stats()
        print(f"✓ Initial stats: {stats}")
        assert stats.frames_processed == 0
        assert stats.events_generated == 0

        # Test reset (should work even when not streaming)
        stream.reset()
        print("✓ Stream reset works")

        return True

    except ImportError:
        print("⚠ Real-time streaming classes not available (expected without GStreamer)")
        return True
    except Exception as e:
        print(f"✗ Failed to test stream creation: {e}")
        traceback.print_exc()
        return False


def test_convenience_function():
    """Test the convenience function for creating streams"""
    try:
        import evlib

        create_realtime_stream_py = evlib.simulation.create_realtime_stream_py

        # This should work even without webcam, just creating the object
        stream = create_realtime_stream_py(
            target_fps=20.0, contrast_threshold=0.18, device_id=0, max_buffer_size=8000, resolution=(640, 480)
        )

        print(f"✓ Convenience function created stream: {stream}")
        return True

    except ImportError:
        print("⚠ Real-time streaming functions not available (expected without GStreamer)")
        return True
    except RuntimeError as e:
        if "GStreamer" in str(e):
            print("⚠ GStreamer not available, which is expected")
            return True
        else:
            print(f"✗ Unexpected runtime error: {e}")
            return False
    except Exception as e:
        print(f"✗ Failed to test convenience function: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Testing Real-time Event Streaming Python Bindings")
    print("=" * 55)

    tests = [
        ("Import Test", test_import),
        ("Real-time Availability", test_realtime_availability),
        ("Configuration Test", test_realtime_config),
        ("Stream Creation Test", test_stream_creation),
        ("Convenience Function Test", test_convenience_function),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        if test_func():
            passed += 1
        else:
            print("This test failed!")

    print("\n" + "=" * 55)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Real-time bindings are working correctly.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
