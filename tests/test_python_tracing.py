#!/usr/bin/env python3
"""
Test Python tracing integration for evlib
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import evlib

    print(f"✓ evlib imported successfully, version: {evlib.__version__}")
except ImportError as e:
    print(f"✗ Failed to import evlib: {e}")
    sys.exit(1)


def test_python_tracing_functions():
    """Test that tracing configuration functions are available from Python"""
    print("\n=== Testing Python Tracing Functions ===")

    # Test that tracing functions are available
    assert hasattr(evlib, "tracing_config"), "tracing_config module should be available"
    assert hasattr(evlib.tracing_config, "init"), "init function should be available"
    assert hasattr(evlib.tracing_config, "init_debug"), "init_debug function should be available"
    assert hasattr(evlib.tracing_config, "init_with_filter"), "init_with_filter function should be available"
    assert hasattr(evlib.tracing_config, "init_production"), "init_production function should be available"
    assert hasattr(evlib.tracing_config, "init_development"), "init_development function should be available"

    print("✓ All tracing configuration functions are available")


def test_tracing_initialization():
    """Test that tracing can be initialized from Python"""
    print("\n=== Testing Tracing Initialization ===")

    # Test that we can call the init function without errors
    try:
        evlib.tracing_config.init()
        print("✓ evlib.tracing_config.init() completed without error")
    except Exception as e:
        print(
            f"⚠ evlib.tracing_config.init() may have failed (might be expected if already initialized): {e}"
        )

    # Test custom filter
    try:
        evlib.tracing_config.init_with_filter("evlib=info")
        print("✓ evlib.tracing_config.init_with_filter() completed without error")
    except Exception as e:
        print(
            f"⚠ evlib.tracing_config.init_with_filter() may have failed (might be expected if already initialized): {e}"
        )


def test_event_loading_with_tracing():
    """Test event loading with tracing enabled"""
    print("\n=== Testing Event Loading with Tracing ===")

    # Initialize tracing
    try:
        evlib.tracing_config.init_debug()
        print("✓ Debug tracing initialized")
    except Exception as e:
        print(f"⚠ Debug tracing initialization may have failed: {e}")

    # Test loading events from the test data
    test_file = "data/slider_depth/events.txt"

    if not os.path.exists(test_file):
        print(f"⚠ Test file {test_file} not found, skipping actual loading test")
        return

    try:
        # Test format detection
        format_result = evlib.detect_format(test_file)
        print(f"✓ Format detection: {format_result}")

        # Test event loading
        events = evlib.load_events(test_file)
        if hasattr(events, "collect"):
            events_list = events.collect()
            print(f"✓ Loaded {len(events_list)} events with tracing enabled")
        else:
            print(f"✓ Events loaded (type: {type(events)}) with tracing enabled")

    except Exception as e:
        print(f"✗ Event loading failed: {e}")
        import traceback

        traceback.print_exc()


def test_different_configuration_functions():
    """Test different tracing configuration functions"""
    print("\n=== Testing Different Configuration Functions ===")

    configurations = [
        ("production", evlib.tracing_config.init_production),
        ("development", evlib.tracing_config.init_development),
    ]

    for name, func in configurations:
        try:
            func()
            print(f"✓ {name} configuration completed without error")
        except Exception as e:
            print(f"⚠ {name} configuration may have failed (might be expected if already initialized): {e}")


def main():
    """Run all tests"""
    print("Starting evlib Python tracing tests...")

    test_python_tracing_functions()
    test_tracing_initialization()
    test_event_loading_with_tracing()
    test_different_configuration_functions()

    print("\n=== Test Summary ===")
    print("Python tracing integration tests completed!")
    print("Note: Some initialization warnings are expected since tracing")
    print("can only be initialized once per process.")


if __name__ == "__main__":
    main()
