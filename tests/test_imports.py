import evlib


def test_evlib_import():
    """Test that evlib can be imported."""
    assert hasattr(evlib, "__version__")


def test_evlib_formats():
    """Test that evlib.formats submodule exists and has expected attributes."""
    assert hasattr(evlib, "formats")
    assert hasattr(evlib.formats, "load_events")
    assert callable(evlib.formats.load_events)

    # Test function attributes
    func = evlib.formats.load_events
    assert hasattr(func, "__name__")
    assert func.__name__ == "load_events"
