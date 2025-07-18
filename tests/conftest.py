"""
Pytest configuration and fixtures for evlib tests.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "docs: marks tests as documentation tests")
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "matplotlib: marks tests that use matplotlib")
    config.addinivalue_line("markers", "requires_data: marks tests that require test data files")


@pytest.fixture(scope="session")
def evlib_available():
    """Check if evlib is available for testing."""
    try:
        import evlib

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        pytest.skip("Test data directory not found")
    return data_dir


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_data_files(temp_dir):
    """Create mock data files for testing."""
    # Create a simple events.txt file
    events_file = temp_dir / "events.txt"
    events_file.write_text(
        """# timestamp x y polarity
0.000100 320 240 1
0.000200 321 241 -1
0.000300 319 239 1
0.000400 322 242 1
0.000500 318 238 -1
"""
    )

    # Create directory structure
    slider_depth_dir = temp_dir / "slider_depth"
    slider_depth_dir.mkdir(parents=True)
    (slider_depth_dir / "events.txt").write_text(events_file.read_text())

    return temp_dir


@pytest.fixture(autouse=True)
def skip_if_no_evlib(request, evlib_available):
    """Skip tests that require evlib if it's not available."""
    if request.node.get_closest_marker("requires_evlib") and not evlib_available:
        pytest.skip("evlib not available")


@pytest.fixture(autouse=True)
def skip_slow_tests(request):
    """Skip slow tests unless explicitly requested."""
    if request.node.get_closest_marker("slow") and not request.config.getoption("--run-slow"):
        pytest.skip("slow test skipped (use --run-slow to run)")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--run-integration", action="store_true", default=False, help="run integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip logic."""
    # Add markers based on test names and content
    for item in items:
        # Add docs marker to all tests in docs directory
        if "docs" in str(item.fspath):
            item.add_marker(pytest.mark.docs)

        # Add integration marker to integration tests
        if "integration" in item.name.lower() or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add slow marker to potentially slow tests
        if any(keyword in item.name.lower() for keyword in ["download", "large", "benchmark"]):
            item.add_marker(pytest.mark.slow)

        # Add matplotlib marker to tests that use matplotlib
        if "matplotlib" in item.name.lower() or "plt" in str(
            item.function.__code__.co_names if hasattr(item, "function") else []
        ):
            item.add_marker(pytest.mark.matplotlib)

    # Skip integration tests unless explicitly requested
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="integration test skipped (use --run-integration to run)")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def matplotlib_backend():
    """Set matplotlib backend for testing."""
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    return matplotlib
