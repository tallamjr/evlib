"""Memoisation and pickle-safety tests for EvlibStreamSource global time."""

import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

import h5py  # noqa: E402

from evlib.data.sources import EvlibStreamSource  # noqa: E402

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"
REPR_NAME = "stacked_histogram_dt50_nbins10"


def _make_seq_dir(tmp_path: Path) -> Path:
    """Copy mini_seq into tmp_path and add the non-ds2 representation h5.

    ``EvlibStreamSource`` is built with ``downsample_by_2=False`` so the dense
    kernel emits full-resolution ``[20, 8, 12]`` windows; that flag also drives
    the label source, which reads ``event_representations.h5``. mini_seq ships
    only the ds2 variant, so we copy the tracked fixture and provide the full-res
    name as a duplicate (same on-disk array, only used for shape and labels).
    """
    seq_dir = tmp_path / "mini_seq"
    shutil.copytree(FIX, seq_dir)
    repr_dir = seq_dir / "event_representations_v2" / REPR_NAME
    shutil.copy(
        repr_dir / "event_representations_ds2_nearest.h5",
        repr_dir / "event_representations.h5",
    )
    return seq_dir


def _make_raw_h5(tmp_path: Path) -> Path:
    """Write a tiny real raw-events h5 whose times fall inside the grid range.

    The mini_seq grid is ``arange(6) * 50000`` so events span ``[0, 250000]``.
    Coordinates stay inside the 8x12 sensor used by the fixture.
    """
    path = tmp_path / "raw_events.h5"
    t = np.array(
        [0, 10_000, 49_000, 60_000, 120_000, 150_000, 199_000, 240_000],
        dtype=np.uint32,
    )
    x = np.array([0, 3, 11, 5, 7, 2, 9, 1], dtype=np.uint16)
    y = np.array([0, 1, 7, 2, 4, 6, 3, 5], dtype=np.uint16)
    p = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8)
    with h5py.File(str(path), "w") as f:
        grp = f.create_group("events")
        grp.create_dataset("t", data=t)
        grp.create_dataset("x", data=x)
        grp.create_dataset("y", data=y)
        grp.create_dataset("p", data=p)
    return path


def _make_source(raw_h5: Path, seq_dir: Path) -> EvlibStreamSource:
    return EvlibStreamSource(
        raw_h5=raw_h5,
        seq_dir=seq_dir,
        downsample_by_2=False,
        height=8,
        width=12,
        nbins=10,
        dataset_group="events",
    )


def _windows_equal(a, b) -> bool:
    ev_a, lab_a = a
    ev_b, lab_b = b
    if len(ev_a) != len(ev_b):
        return False
    for ta, tb in zip(ev_a, ev_b):
        if not torch.equal(ta, tb):
            return False
    if len(lab_a) != len(lab_b):
        return False
    for la, lb in zip(lab_a, lab_b):
        if la is None or lb is None:
            if la is not lb:
                return False
        elif not torch.equal(la, lb):
            return False
    return True


def test_global_time_built_once_across_calls(tmp_path, monkeypatch):
    raw_h5 = _make_raw_h5(tmp_path)
    seq_dir = _make_seq_dir(tmp_path)
    src = _make_source(raw_h5, seq_dir)

    calls = {"n": 0}
    real_accumulate = np.maximum.accumulate

    def counting_accumulate(*args, **kwargs):
        calls["n"] += 1
        return real_accumulate(*args, **kwargs)

    monkeypatch.setattr(np.maximum, "accumulate", counting_accumulate)

    src.read_windows(0, 6)
    src.read_windows(0, 6)

    assert calls["n"] == 1


def test_memoisation_preserves_output(tmp_path):
    raw_h5 = _make_raw_h5(tmp_path)
    seq_dir = _make_seq_dir(tmp_path)

    fresh = _make_source(raw_h5, seq_dir)
    reference = fresh.read_windows(0, 6)

    src = _make_source(raw_h5, seq_dir)
    first = src.read_windows(0, 6)
    second = src.read_windows(0, 6)

    assert _windows_equal(first, reference)
    assert _windows_equal(second, reference)


def test_out_of_bounds_raises_without_building_cache(tmp_path):
    # An out-of-range read_windows must validate against the cheap grid and
    # raise BEFORE _ensure_time() does the heavy global-time build, so the
    # ~2 GB read is never triggered on a misconfigured call. On a fresh source
    # _t_full stays None after the ValueError.
    raw_h5 = _make_raw_h5(tmp_path)
    seq_dir = _make_seq_dir(tmp_path)
    src = _make_source(raw_h5, seq_dir)

    n_windows = src.window_count()  # 6 for mini_seq
    assert src._t_full is None

    with pytest.raises(ValueError):
        src.read_windows(0, n_windows + 1)
    assert src._t_full is None

    with pytest.raises(ValueError):
        src.read_windows(-1, 2)
    assert src._t_full is None

    with pytest.raises(ValueError):
        src.read_windows(3, 3)
    assert src._t_full is None


def test_pickle_drops_big_array_and_rebuilds(tmp_path):
    raw_h5 = _make_raw_h5(tmp_path)
    seq_dir = _make_seq_dir(tmp_path)
    src = _make_source(raw_h5, seq_dir)
    reference = src.read_windows(0, 6)

    assert src._t_full is not None

    restored = pickle.loads(pickle.dumps(src))
    assert restored._t_full is None
    assert restored._starts is None
    assert restored._ends is None

    rebuilt = restored.read_windows(0, 6)
    assert restored._t_full is not None
    assert _windows_equal(rebuilt, reference)
