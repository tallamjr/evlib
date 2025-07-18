"""
Configuration for pytest-markdown-docs testing.

This file provides fixtures and configuration for testing code examples
found in the documentation.
"""

import pytest
import sys
import os
import tempfile
import shutil
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
        self.augmentation = MockAugmentation()
        self.processing = MockProcessing()


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

    def events_to_smooth_voxel_grid(self, xs, ys, ts, ps, n_bins, shape):
        """Mock events_to_smooth_voxel_grid function."""
        return self.events_to_voxel_grid(xs, ys, ts, ps, n_bins, shape)


class MockVisualization:
    """Mock evlib.visualization module."""

    def draw_events_to_image(self, xs, ys, ps, width, height):
        """Mock draw_events_to_image function."""
        import numpy as np

        return np.random.rand(height, width)


class MockAugmentation:
    """Mock evlib.augmentation module."""

    def flip_events_x(self, xs, ys, ts, ps, shape):
        """Mock flip_events_x function."""

        xs_flipped = shape[0] - xs
        return xs_flipped, ys, ts, ps

    def add_random_events(self, xs, ys, ts, ps, n_events, shape):
        """Mock add_random_events function."""
        import numpy as np

        n_original = len(xs)
        new_xs = np.concatenate([xs, np.random.randint(0, shape[0], n_events)])
        new_ys = np.concatenate([ys, np.random.randint(0, shape[1], n_events)])
        new_ts = np.concatenate([ts, np.random.uniform(ts.min(), ts.max(), n_events)])
        new_ps = np.concatenate([ps, np.random.choice([-1, 1], n_events)])
        return new_xs, new_ys, new_ts, new_ps


class MockProcessing:
    """Mock evlib.processing module."""

    def download_model(self, model_name):
        """Mock download_model function."""
        return f"/mock/path/to/{model_name}"

    def events_to_video(self, xs, ys, ts, ps, model_path, width, height):
        """Mock events_to_video function."""
        import numpy as np

        return np.random.rand(height, width, 3)


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

    # Create mock data directory
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True)

    # Create slider_depth directory
    slider_depth_dir = data_dir / "slider_depth"
    slider_depth_dir.mkdir(parents=True)

    # Create mock events.txt file
    events_file = slider_depth_dir / "events.txt"
    events_content = """# Mock events file for testing
# timestamp x y polarity
0.000100 320 240 1
0.000200 321 241 -1
0.000300 319 239 1
0.000400 322 242 1
0.000500 318 238 -1
"""
    events_file.write_text(events_content)

    # Store temp dir for cleanup
    _global_namespace["_temp_dir"] = temp_dir

    # Change to temp directory
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    _global_namespace["_original_cwd"] = original_cwd

    yield _global_namespace

    # Cleanup
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)


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
    config.addinivalue_line("markers", "requires_data: marks tests requiring data files")
    config.addinivalue_line("markers", "requires_evlib: marks tests requiring evlib")
    config.addinivalue_line("markers", "requires_matplotlib: marks tests requiring matplotlib")


def pytest_runtest_setup(item):
    """Set up individual test runs."""
    # Mark all tests in docs as docs tests
    if "docs" in str(item.fspath):
        item.add_marker(pytest.mark.docs)

    # Inject global namespace into test
    if hasattr(item, "obj") and hasattr(item.obj, "__globals__"):
        item.obj.__globals__.update(_global_namespace)
