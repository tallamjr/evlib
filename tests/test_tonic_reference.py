"""Prove the ported tonic voxel-grid oracle is faithful to real tonic.

Loads the REAL tonic ``to_voxel_grid.py`` by file path (it imports only numpy,
so it can be loaded standalone without tonic's package ``__init__``) and asserts
the local port in ``tests/conformance/tonic_reference.py`` is bit-identical on a
synthetic structured event array.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conformance.tonic_reference import tonic_frame, tonic_voxel_grid

_TONIC_ROOT = Path(__file__).resolve().parents[1] / "lib" / "tonic" / "tonic"

REAL_TONIC_PATH = _TONIC_ROOT / "functional" / "to_voxel_grid.py"
REAL_SLICERS_PATH = _TONIC_ROOT / "slicers.py"
REAL_TO_FRAME_PATH = _TONIC_ROOT / "functional" / "to_frame.py"


def _load_real_tonic():
    spec = importlib.util.spec_from_file_location("real_to_voxel_grid", REAL_TONIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_real_to_frame():
    """Load real tonic ``to_frame_numpy`` by path.

    ``to_frame.py`` does ``from tonic.slicers import ...``, so we first load the
    standalone ``slicers.py`` (numpy + typing_extensions only) and register it in
    ``sys.modules`` as ``tonic.slicers`` so the import resolves without pulling
    in tonic's package ``__init__`` (which needs scipy/librosa).
    """
    slicers_spec = importlib.util.spec_from_file_location(
        "tonic.slicers", REAL_SLICERS_PATH
    )
    slicers_mod = importlib.util.module_from_spec(slicers_spec)
    sys.modules["tonic.slicers"] = slicers_mod
    slicers_spec.loader.exec_module(slicers_mod)

    frame_spec = importlib.util.spec_from_file_location(
        "real_to_frame", REAL_TO_FRAME_PATH
    )
    frame_mod = importlib.util.module_from_spec(frame_spec)
    frame_spec.loader.exec_module(frame_mod)
    return frame_mod


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


@pytest.mark.skipif(
    not (REAL_SLICERS_PATH.exists() and REAL_TO_FRAME_PATH.exists()),
    reason="real tonic source not present",
)
def test_frame_port_matches_real_tonic_bit_identical():
    width, height, n_time_bins = 64, 48, 5
    events = _synthetic_events(width=width, height=height)

    real = _load_real_to_frame()
    # to_frame_numpy returns (T, P, H, W) int16; copy because it may touch p.
    ref = real.to_frame_numpy(
        events.copy(), (width, height, 2), n_time_bins=n_time_bins
    )
    port = tonic_frame(events.copy(), (width, height, 2), n_time_bins=n_time_bins)

    assert port.shape == (n_time_bins, 2, height, width)
    assert ref.shape == port.shape
    assert np.array_equal(ref.astype(np.int64), port), (
        "ported frame oracle diverges from real tonic"
    )


def test_frame_port_matches_hand_computed():
    """Cross-check the boundary handling against a hand-computed expectation.

    Six events over t in [0, 100] us, n_time_bins=2, sensor 3x2 (WxH), P=2.

    time_window = (100 - 0) // 2 = 50, stride = 50.
    bin 0: t in [0, 50)   -> events at t=0, 10, 49
    bin 1: t in [50, 100) -> events at t=50, 70
    DROPPED: t=100 (>= t0 + 2*50 = 100; searchsorted-left puts it past bin 1).
    """
    dtype = np.dtype(
        [("x", np.int64), ("y", np.int64), ("t", np.float64), ("p", np.int64)]
    )
    events = np.array(
        [
            (0, 0, 0, 1),  # bin 0, pol+ -> ch1
            (0, 0, 10, 0),  # bin 0, pol0 -> ch0
            (1, 1, 49, 1),  # bin 0, pol+ -> ch1
            (2, 0, 50, 0),  # bin 1, pol0 -> ch0
            (2, 1, 70, 1),  # bin 1, pol+ -> ch1
            (0, 1, 100, 1),  # DROPPED (boundary, searchsorted-left)
        ],
        dtype=dtype,
    )

    expected = np.zeros((2, 2, 2, 3), dtype=np.int64)  # (T, P, H, W)
    expected[0, 1, 0, 0] = 1  # t=0
    expected[0, 0, 0, 0] = 1  # t=10
    expected[0, 1, 1, 1] = 1  # t=49
    expected[1, 0, 0, 2] = 1  # t=50
    expected[1, 1, 1, 2] = 1  # t=70

    port = tonic_frame(events, (3, 2, 2), n_time_bins=2)
    assert np.array_equal(port, expected)
