"""
Configuration for pytest-markdown-docs testing.

This file provides fixtures and configuration for testing code examples
found in the documentation.
"""

import pytest
import shutil
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))


# Mock evlib for testing
class MockEvlib:
    """Mock evlib module for testing documentation examples."""

    def __init__(self):
        self.formats = MockFormats()
        self.representations = MockRepresentations()
        self.visualization = MockVisualization()


class MockFormats:
    """Mock evlib.formats module."""

    def load_events(self, file_path, **kwargs):
        """Mock load_events function."""
        import numpy as np

        n_events = 10
        xs = np.random.randint(0, 640, n_events)
        ys = np.random.randint(0, 480, n_events)
        ts = np.linspace(0, 1, n_events)
        ps = np.random.choice([-1, 1], n_events)
        return xs, ys, ts, ps

    def load_events_filtered(self, file_path, **kwargs):
        """Mock load_events_filtered function."""
        return self.load_events(file_path, **kwargs)

    def save_events_to_hdf5(self, xs, ys, ts, ps, file_path):
        """Mock save_events_to_hdf5 function."""
        pass


class MockRepresentations:
    """Mock evlib.representations module."""

    def events_to_voxel_grid(self, xs, ys, ts, ps, n_bins, shape):
        """Mock events_to_voxel_grid function."""
        import numpy as np

        h, w = shape
        voxel_data = np.random.rand(n_bins, h, w).astype(np.float32)
        voxel_shape_data = (n_bins, h, w)
        voxel_shape_shape = (n_bins, h, w)
        return voxel_data, voxel_shape_data, voxel_shape_shape


class MockVisualization:
    """Mock evlib.visualization module."""

    def draw_events_to_image(self, xs, ys, ps, width, height):
        """Mock draw_events_to_image function."""
        import numpy as np

        return np.random.rand(height, width)


# Global namespace for code blocks
_global_namespace = {}


@pytest.fixture(autouse=True, scope="session")
def setup_global_namespace():
    """Set up global namespace that persists across code blocks."""
    global _global_namespace

    # Import common modules
    import numpy as np
    import time

    # Set up matplotlib
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Mock show and figure
        def mock_show(*args, **kwargs):
            pass

        class MockFigure:
            def __init__(self, *args, **kwargs):
                pass

            def show(self):
                pass

        def mock_figure(*args, **kwargs):
            return MockFigure(*args, **kwargs)

        plt.show = mock_show
        plt.figure = mock_figure

        _global_namespace["plt"] = plt
        _global_namespace["matplotlib"] = matplotlib

    except ImportError:
        pass

    # Add evlib (mock or real)
    try:
        import evlib

        _global_namespace["evlib"] = evlib
    except ImportError:
        _global_namespace["evlib"] = MockEvlib()

    # Add other common modules
    _global_namespace["np"] = np
    _global_namespace["numpy"] = np
    _global_namespace["time"] = time

    # Run documentation examples inside a contained scratch directory so any
    # files they write (output.h5, events.parquet, ...) land under
    # tests/.output/ instead of the repository root. Read-only resource
    # directories are symlinked in so the examples' relative "data/..." paths
    # still resolve.
    project_root = Path(__file__).parent.parent
    scratch = project_root / "tests" / ".output" / "docs"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    for resource in ("data", "benchmarks", "examples"):
        src = project_root / resource
        if src.exists():
            (scratch / resource).symlink_to(src, target_is_directory=True)

    original_cwd = os.getcwd()
    os.chdir(scratch)
    _global_namespace["_original_cwd"] = original_cwd

    yield _global_namespace

    # Cleanup
    os.chdir(original_cwd)


@pytest.fixture(autouse=True)
def inject_global_namespace(setup_global_namespace):
    """Inject global namespace into test execution."""
    import builtins

    # Store original globals
    original_globals = getattr(builtins, "__dict__", {}).copy()

    # Inject our namespace
    for name, value in setup_global_namespace.items():
        if not name.startswith("_"):
            setattr(builtins, name, value)

    yield

    # Restore original globals
    for name in list(builtins.__dict__.keys()):
        if name in setup_global_namespace and not name.startswith("_"):
            if name in original_globals:
                setattr(builtins, name, original_globals[name])
            else:
                delattr(builtins, name)


def pytest_configure(config):
    """Configure pytest for markdown docs testing."""
    # Add custom markers
    config.addinivalue_line("markers", "docs: marks tests as documentation tests")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line(
        "markers", "requires_data: marks tests requiring data files"
    )
    config.addinivalue_line("markers", "requires_evlib: marks tests requiring evlib")
    config.addinivalue_line(
        "markers", "requires_matplotlib: marks tests requiring matplotlib"
    )


def pytest_runtest_setup(item):
    """Set up individual test runs."""
    # Mark all tests in docs as docs tests
    if "docs" in str(item.fspath):
        item.add_marker(pytest.mark.docs)

    # Inject global namespace into test
    if hasattr(item, "obj") and hasattr(item.obj, "__globals__"):
        item.obj.__globals__.update(_global_namespace)
