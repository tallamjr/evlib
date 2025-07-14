"""Test streaming event processing pipeline."""

from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import importlib.util

    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
except ImportError:
    TORCH_AVAILABLE = False

EVLIB_AVAILABLE = False
try:
    import evlib

    EVLIB_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")
def test_evlib_streaming_integration():
    """Test integration with evlib streaming module."""
    import numpy as np

    # Skip if streaming module is not available
    if not hasattr(evlib, 'streaming'):
        pytest.skip("streaming module not available")
    
    # Create streaming configuration
    config = evlib.streaming.create_streaming_config()
    assert config is not None
    print(f"Created streaming config: {config}")

    # Test PyStreamingProcessor
    processor = evlib.streaming.PyStreamingProcessor(config)
    assert not processor.is_ready()  # No model loaded yet

    # Test buffer status
    buffer_len, buffer_util = processor.get_buffer_status()
    assert buffer_len == 0
    assert buffer_util == 0.0

    # Create test events
    n_events = 1000
    xs = np.random.randint(0, 240, n_events, dtype=np.int64)
    ys = np.random.randint(0, 180, n_events, dtype=np.int64)
    ts = np.sort(np.random.uniform(0, 1.0, n_events))
    ps = np.random.choice([-1, 1], n_events).astype(np.int64)

    # Process events (without model)
    result = processor.process_events(xs, ys, ts, ps)
    # Should work even without model (returns voxel representation)
    if result is not None:
        assert result.shape == (180, 240)  # height x width
        print(f"✓ Processing without model successful, output shape: {result.shape}")

    # Test statistics
    stats = processor.get_stats()
    assert stats.total_events_processed >= 0
    print(f"✓ Stats: {stats}")

    # Test functional interface
    result2 = evlib.streaming.process_events_streaming(xs[:100], ys[:100], ts[:100], ps[:100])
    if result2 is not None:
        print(f"✓ Functional interface successful, output shape: {result2.shape}")

    # Test EventStream
    stream = evlib.streaming.PyEventStream(config)
    assert not stream.is_running()

    # Start stream (will fail without model, but that's expected)
    try:
        stream.start()
        assert False, "Should fail without model"
    except Exception as e:
        assert "Model must be loaded" in str(e)
        print("✓ Expected error when starting stream without model")

    print("Streaming integration test completed successfully!")


if __name__ == "__main__":
    test_evlib_streaming_integration()
    print("All streaming tests passed!")
