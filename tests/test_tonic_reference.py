"""Prove the ported tonic voxel-grid oracle is faithful to real tonic.

Loads the REAL tonic ``to_voxel_grid.py`` by file path (it imports only numpy,
so it can be loaded standalone without tonic's package ``__init__``) and asserts
the local port in ``tests/conformance/tonic_reference.py`` is bit-identical on a
synthetic structured event array.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tests.conformance.tonic_reference import tonic_voxel_grid

REAL_TONIC_PATH = (
    Path(__file__).resolve().parents[1]
    / "lib"
    / "tonic"
    / "tonic"
    / "functional"
    / "to_voxel_grid.py"
)


def _load_real_tonic():
    spec = importlib.util.spec_from_file_location("real_to_voxel_grid", REAL_TONIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_events(n=2000, width=64, height=48, seed=7):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(
        [("x", np.int64), ("y", np.int64), ("t", np.float64), ("p", np.int64)]
    )
    events = np.zeros(n, dtype=dtype)
    events["x"] = rng.integers(0, width, n)
    events["y"] = rng.integers(0, height, n)
    # monotonically increasing timestamps (tonic assumes sorted by t)
    events["t"] = np.sort(rng.uniform(0, 1_000_000, n))
    events["p"] = rng.integers(0, 2, n)  # 0/1 polarity, exercises the 0 -> -1 map
    return events


@pytest.mark.skipif(
    not REAL_TONIC_PATH.exists(), reason="real tonic source not present"
)
def test_port_matches_real_tonic_bit_identical():
    width, height, n_time_bins = 64, 48, 5
    events = _synthetic_events(width=width, height=height)

    real = _load_real_tonic()
    # real tonic mutates the p field in place; give each its own copy
    ref = real.to_voxel_grid_numpy(
        events.copy(), (width, height, 2), n_time_bins=n_time_bins
    )
    port = tonic_voxel_grid(events.copy(), (width, height, 2), n_time_bins=n_time_bins)

    assert ref.shape == port.shape == (n_time_bins, 1, height, width)
    assert np.array_equal(ref, port), "ported oracle diverges from real tonic"
