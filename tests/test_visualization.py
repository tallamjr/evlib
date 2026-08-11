"""
Tests for evlib visualization functionality.

Replaces the previous 24-test suite, which never ran: its module-level skip
pointed at `data/eTram_processed/test/test_day_010`, a path that exists
nowhere in the repo or CI, and several assertions referenced attributes that
do not exist on the real classes (`renderer.decay_buffer`, `renderer.reset()`
return value). The wrong `requires_hdf5` mark also gated on the Rust
`--features hdf5` build, which this module does not need: it reads HDF5 via
h5py, not the Rust extension.

This suite runs against the tracked fixture
`tests/data_fixtures/mini_seq/event_representations_v2/
stacked_histogram_dt50_nbins10/event_representations_ds2_nearest.h5`
(shape (6, 20, 8, 12) uint8), loaded by direct file path since the fixture's
directory layout (`dt50_nbins10`, no `=` signs) does not match
`eTramDataLoader`'s directory-search pattern
(`dt=50_nbins=10`).
"""

from pathlib import Path

import numpy as np
import pytest

try:
    import cv2  # noqa: F401
    import h5py  # noqa: F401
    import evlib.visualization as viz
except ImportError as e:
    pytest.skip(f"evlib[plot] dependencies not available: {e}", allow_module_level=True)

FIXTURE_H5 = Path(
    "tests/data_fixtures/mini_seq/event_representations_v2/"
    "stacked_histogram_dt50_nbins10/event_representations_ds2_nearest.h5"
)


def test_fixture_is_present():
    """Fixture is tracked in git; a missing file means the checkout is broken."""
    assert FIXTURE_H5.exists(), f"tracked fixture missing: {FIXTURE_H5}"


class TestVisualizationConfig:
    def test_default_config(self):
        config = viz.VisualizationConfig()

        assert config.width == 640
        assert config.height == 360
        assert config.fps == 30.0
        assert config.positive_color == (0, 0, 255)
        assert config.negative_color == (255, 128, 0)
        assert config.background_color == (200, 180, 150)
        assert config.decay_ms == 100.0
        assert config.show_stats is True
        assert config.codec == "mp4v"

    def test_frame_duration_calculation(self):
        """frame_duration_ms is derived from fps in __post_init__, not the field default."""
        config = viz.VisualizationConfig(fps=60.0)
        assert abs(config.frame_duration_ms - 16.667) < 0.01


class TesteTramDataLoader:
    def test_load_metadata_from_direct_path(self):
        loader = viz.eTramDataLoader(FIXTURE_H5)

        assert loader.h5_file_path == FIXTURE_H5
        assert loader.data_format == "representation"
        assert loader.num_frames == 6
        assert loader.num_bins == 20
        assert loader.height == 8
        assert loader.width == 12
        assert loader.dtype == np.uint8
        assert len(loader.timestamps_us) == loader.num_frames
        assert loader.end_time_s > loader.start_time_s

    def test_get_frame_data(self):
        loader = viz.eTramDataLoader(FIXTURE_H5)

        frame_data = loader.get_frame_data(0)
        assert frame_data.shape == (20, 8, 12)
        assert frame_data.dtype == np.uint8

        with pytest.raises(ValueError, match="out of range"):
            loader.get_frame_data(-1)

        with pytest.raises(ValueError, match="out of range"):
            loader.get_frame_data(loader.num_frames)

    def test_get_frame_range(self):
        loader = viz.eTramDataLoader(FIXTURE_H5)

        frame_data = loader.get_frame_range(0, 3)
        assert frame_data.shape == (3, 20, 8, 12)

        with pytest.raises(ValueError, match="Invalid frame range"):
            loader.get_frame_range(-1, 3)

        with pytest.raises(ValueError, match="Invalid frame range"):
            loader.get_frame_range(3, 1)

    def test_missing_h5_file(self):
        with pytest.raises(FileNotFoundError, match="No HDF5 file found"):
            viz.eTramDataLoader("/nonexistent/path")


class TestEventFrameRenderer:
    def test_render_frame_exact_bgr_polarity(self):
        """First render_frame call has no decay blend (previous_frame is None), so
        colours are exact: _render_polarity_frame hardcodes pure BGR red/blue,
        it does not use config.positive_color/negative_color."""
        config = viz.VisualizationConfig(width=4, height=4, decay_ms=100.0)
        renderer = viz.EventFrameRenderer(config)

        event_data = np.zeros((4, 4, 4), dtype=np.uint8)
        event_data[0, 1, 1] = 5  # even bin -> positive
        event_data[1, 2, 2] = 5  # odd bin -> negative

        frame = renderer.render_frame(event_data, timestamp_s=0.0)

        assert frame.dtype == np.uint8
        assert tuple(int(c) for c in frame[1, 1]) == (0, 0, 255)
        assert tuple(int(c) for c in frame[2, 2]) == (255, 0, 0)
        assert tuple(int(c) for c in frame[0, 0]) == (200, 180, 150)
        assert renderer.frame_count == 1

    def test_colormap_rendering_smoke(self):
        loader = viz.eTramDataLoader(FIXTURE_H5)
        config = viz.VisualizationConfig(
            width=loader.width,
            height=loader.height,
            use_colormap=True,
            colormap_type="jet",
        )
        renderer = viz.EventFrameRenderer(config)

        frame = renderer.render_frame(loader.get_frame_data(0), timestamp_s=0.0)

        assert frame.shape == (loader.height, loader.width, 3)
        assert frame.dtype == np.uint8

    def test_render_frame_with_fixture(self):
        loader = viz.eTramDataLoader(FIXTURE_H5)
        config = viz.VisualizationConfig(
            width=loader.width, height=loader.height, fps=20.0
        )
        renderer = viz.EventFrameRenderer(config)

        frame = renderer.render_frame(loader.get_frame_data(0), timestamp_s=0.0)

        assert frame.shape == (8, 12, 3)
        assert frame.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
