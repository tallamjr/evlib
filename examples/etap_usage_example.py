#!/usr/bin/env python3
"""
ETAP Usage Example

This example demonstrates how to use the configurable ETAP integration
with evlib after the hardcoded path dependencies have been removed.
"""

import numpy as np

# Example of setting ETAP_PATH environment variable
# You can do this in your shell or programmatically:
# export ETAP_PATH=/path/to/your/ETAP/repository


def setup_etap_environment():
    """
    Example of how to set up ETAP environment programmatically.
    This should be done before importing evlib.etap_integration.
    """
    # Option 1: Set environment variable
    # etap_path = "/path/to/your/ETAP/repository"
    # os.environ['ETAP_PATH'] = etap_path

    # Option 2: Place ETAP in a standard location
    # The integration will automatically check:
    # - ~/git/ETAP
    # - ~/github/ETAP
    # - ~/src/ETAP
    # - /opt/ETAP
    # - /usr/local/share/ETAP

    print("ETAP environment setup example:")
    print("1. Set ETAP_PATH environment variable:")
    print("   export ETAP_PATH=/path/to/ETAP")
    print("2. Or place ETAP in a standard location like ~/git/ETAP")


def create_sample_events():
    """Create sample event data for testing."""
    # Generate some sample events
    np.random.seed(42)
    num_events = 1000

    xs = np.random.randint(0, 640, num_events)
    ys = np.random.randint(0, 480, num_events)
    ts = np.sort(np.random.uniform(0, 1.0, num_events))
    ps = np.random.choice([0, 1], num_events)

    return xs, ys, ts, ps


def main():
    """Main example function."""
    print("ETAP Integration Usage Example")
    print("=" * 40)

    # Show environment setup
    setup_etap_environment()
    print()

    try:
        # Import ETAP integration
        from evlib.etap_integration import get_etap_status, create_etap_tracker
        import evlib.tracking

        # Check status
        status = get_etap_status()
        print("ETAP Status:")
        print(f"  PyTorch available: {status['torch_available']}")
        print(f"  ETAP available: {status['etap_available']}")
        print(f"  Fully functional: {status['fully_functional']}")
        print()

        if not status["fully_functional"]:
            print("❌ ETAP not fully available. Please check installation.")
            return

        # Example usage with proper model path
        model_path = "/path/to/your/etap_model.pth"

        print("Example tracker creation:")
        print(f"  model_path = '{model_path}'")
        print("  tracker = create_etap_tracker(model_path)")
        print()

        # Note: This will fail unless you have a real model file
        print("Note: To actually create a tracker, you need:")
        print("1. A trained ETAP model file (.pth)")
        print("2. ETAP repository in a standard location or ETAP_PATH set")
        print()

        # Create sample data
        xs, ys, ts, ps = create_sample_events()

        # Create sample query points
        query_points = [
            evlib.tracking.PyQueryPoint(0, 320, 240),  # Center of image
            evlib.tracking.PyQueryPoint(0, 100, 100),  # Top-left region
        ]

        print("Sample usage (requires real model file):")
        print("```python")
        print("# Create tracker with explicit model path")
        print("tracker = create_etap_tracker('/path/to/etap_model.pth')")
        print()
        print("# Track points")
        print("results = tracker.track_points(")
        print("    (xs, ys, ts, ps),")
        print("    query_points,")
        print("    resolution=(640, 480)")
        print(")")
        print()
        print("# Access results")
        print("for track_id, track_result in results.items():")
        print("    coords = track_result.coords")
        print("    visibility = track_result.visibility")
        print("    print(f'Track {track_id}: {len(coords)} points')")
        print("```")

    except ImportError as e:
        print(f"❌ Failed to import ETAP integration: {e}")
        print("Make sure evlib is properly installed.")

    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
