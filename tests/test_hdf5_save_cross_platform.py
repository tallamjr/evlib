"""
Test HDF5 save functionality across platforms.

This file tests ``evlib.save_events_to_hdf5``, which is a pure-Python h5py wrapper.
It does NOT require the Rust HDF5 feature (``--features hdf5``); it only requires
h5py to be installed. The module skips if h5py is absent.

Real event data is loaded from ``data/slider_depth/events.txt`` (a tracked text-format
recording, always present in CI) so no synthetic arrays are fabricated. All output
files are written under pytest's ``tmp_path`` so nothing lands in the repository.
"""

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import evlib

ROOT = Path(__file__).resolve().parents[1]
SLIDER_DEPTH_EVENTS = ROOT / "data/slider_depth/events.txt"


def _load_slider_events(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the first *n* events from slider_depth as numpy arrays.

    Returns (xs, ys, ts, ps) where:
    - xs, ys: int64 pixel coordinates
    - ts: float64 timestamps in seconds
    - ps: int64 polarity values (0 or 1 as stored in the text file)
    """
    df = evlib.load_events(str(SLIDER_DEPTH_EVENTS)).head(n).collect()
    xs = df["x"].to_numpy().astype(np.int64)
    ys = df["y"].to_numpy().astype(np.int64)
    ts = df["t"].dt.total_microseconds().to_numpy() / 1_000_000.0
    ps = df["polarity"].to_numpy().astype(np.int64)
    return xs, ys, ts, ps


def test_hdf5_save_python_fallback(tmp_path):
    """HDF5 save uses the h5py path and writes all four datasets correctly."""
    xs, ys, ts, ps = _load_slider_events(1000)
    out = tmp_path / "events.h5"

    evlib.save_events_to_hdf5(xs, ys, ts, ps, str(out))
    assert out.exists()

    with h5py.File(out, "r") as f:
        assert "events" in f
        grp = f["events"]

        assert "xs" in grp
        assert "ys" in grp
        assert "ts" in grp
        assert "ps" in grp

        np.testing.assert_array_equal(grp["xs"][:], xs.astype(np.uint16))
        np.testing.assert_array_equal(grp["ys"][:], ys.astype(np.uint16))
        np.testing.assert_array_almost_equal(grp["ts"][:], ts)
        np.testing.assert_array_equal(grp["ps"][:], ps.astype(np.int8))


def test_hdf5_save_auto_fallback(tmp_path):
    """save_events_to_hdf5 writes all four datasets and preserves event count."""
    xs, ys, ts, ps = _load_slider_events(500)
    out = tmp_path / "events.h5"

    evlib.save_events_to_hdf5(xs, ys, ts, ps, str(out))
    assert out.exists()

    with h5py.File(out, "r") as f:
        assert "events" in f
        grp = f["events"]

        assert set(grp.keys()) == {"xs", "ys", "ts", "ps"}

        assert len(grp["xs"]) == len(xs)
        assert len(grp["ys"]) == len(ys)
        assert len(grp["ts"]) == len(ts)
        assert len(grp["ps"]) == len(ps)


def test_hdf5_save_validation(tmp_path):
    """save_events_to_hdf5 raises ValueError when array lengths differ."""
    xs = np.array([1, 2, 3], dtype=np.int64)
    ys = np.array([1, 2], dtype=np.int64)  # deliberately wrong length
    ts = np.array([0.1, 0.2, 0.3])
    ps = np.array([-1, 1, -1], dtype=np.int64)
    out = tmp_path / "events.h5"

    with pytest.raises(ValueError, match="Arrays must have the same length"):
        evlib.save_events_to_hdf5(xs, ys, ts, ps, str(out))


def test_hdf5_save_roundtrip(tmp_path):
    """Saving and reloading events via h5py produces bit-identical arrays."""
    xs, ys, ts, ps = _load_slider_events(2000)
    out = tmp_path / "events.h5"

    evlib.save_events_to_hdf5(xs, ys, ts, ps, str(out))

    with h5py.File(out, "r") as f:
        grp = f["events"]
        loaded_xs = grp["xs"][:]
        loaded_ys = grp["ys"][:]
        loaded_ts = grp["ts"][:]
        loaded_ps = grp["ps"][:]

    np.testing.assert_array_equal(loaded_xs, xs.astype(np.uint16))
    np.testing.assert_array_equal(loaded_ys, ys.astype(np.uint16))
    np.testing.assert_array_almost_equal(loaded_ts, ts)
    np.testing.assert_array_equal(loaded_ps, ps.astype(np.int8))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
