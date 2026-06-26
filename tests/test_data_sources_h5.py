import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from evlib.data.sources import (
    PreprocessedH5Source,
    _nbins_from_repr_name,
    _scale_label_rows_to_ds2,
)

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"

_LABEL_DTYPE = np.dtype(
    [
        ("t", "<f4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("class_id", "<u4"),
        ("class_confidence", "<f4"),
    ]
)


def _make_rows(boxes):
    rows = np.zeros(len(boxes), dtype=_LABEL_DTYPE)
    for i, (x, y, w, h) in enumerate(boxes):
        rows[i] = (0.0, x, y, w, h, 0, 1.0)
    return rows


def test_scale_label_rows_to_ds2_halves_and_recomputes():
    # Full-resolution box well inside a 720x1280 sensor; repr is ds2 (360x640).
    rows = _make_rows([(100.0, 200.0, 40.0, 60.0)])
    out = _scale_label_rows_to_ds2(rows, repr_h=360, repr_w=640)
    # x,y halved; w,h recomputed from halved corners (here no clamping applies).
    np.testing.assert_allclose(out["x"], [50.0])
    np.testing.assert_allclose(out["y"], [100.0])
    np.testing.assert_allclose(out["w"], [20.0])
    np.testing.assert_allclose(out["h"], [30.0])


def test_scale_label_rows_to_ds2_clamps_to_repr_bounds():
    # A box whose halved far corner exceeds the ds2 bounds must clamp to W-1/H-1.
    rows = _make_rows([(1270.0, 710.0, 20.0, 20.0)])
    out = _scale_label_rows_to_ds2(rows, repr_h=360, repr_w=640)
    # (x+w)*0.5 = 645 -> clamp to 639 ; (y+h)*0.5 = 365 -> clamp to 359.
    np.testing.assert_allclose(out["x"], [635.0])
    np.testing.assert_allclose(out["y"], [355.0])
    np.testing.assert_allclose(out["w"], [639.0 - 635.0])
    np.testing.assert_allclose(out["h"], [359.0 - 355.0])


def test_scale_label_rows_to_ds2_drops_flat_boxes():
    # A degenerate full-res box that halves to w<=0/h<=0 is dropped.
    rows = _make_rows([(100.0, 100.0, 40.0, 60.0), (10.0, 10.0, 0.0, 0.0)])
    out = _scale_label_rows_to_ds2(rows, repr_h=360, repr_w=640)
    assert out.shape == (1,)
    np.testing.assert_allclose(out["x"], [50.0])


def test_nbins_parses_both_repr_name_forms():
    # evlib-native on-disk form (no '=')
    assert _nbins_from_repr_name("stacked_histogram_dt50_nbins10") == 10
    # upstream-RVT on-disk form (with '='), used by RVT_REPR_DIR_NAME and real
    # gen4/eTram data
    assert _nbins_from_repr_name("stacked_histogram_dt=50_nbins=10") == 10


def test_window_count_and_shape():
    src = PreprocessedH5Source(FIX)
    assert src.window_count() == 6
    ev, labels = src.read_windows(0, 6)
    assert len(ev) == 6 and len(labels) == 6
    assert ev[0].shape == (20, 8, 12) and ev[0].dtype == torch.uint8


def test_label_alignment():
    src = PreprocessedH5Source(FIX)
    _, labels = src.read_windows(0, 6)
    # boxes only on repr indices 1 (1 box) and 4 (2 boxes); rest None
    assert labels[0] is None and labels[2] is None and labels[5] is None
    assert labels[1].shape == (1, 5)
    assert labels[4].shape == (2, 5)


def test_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        PreprocessedH5Source(FIX.parent / "does_not_exist").window_count()


def test_no_persistent_handle_after_construction():
    src = PreprocessedH5Source(FIX)
    assert src._h5 is None and src._data is None
    src.window_count()
    # window_count reads metadata then closes the file: no live handle remains.
    assert src._h5 is None and src._data is None


def test_picklable_after_window_count():
    src = PreprocessedH5Source(FIX)
    assert src.window_count() == 6
    restored = pickle.loads(pickle.dumps(src))
    # The unpickled source holds no live handle and still reads correctly.
    assert restored._h5 is None and restored._data is None
    assert restored.window_count() == 6
    ev, labels = restored.read_windows(0, 6)
    assert len(ev) == 6 and len(labels) == 6


def test_data_handle_opened_only_on_read():
    src = PreprocessedH5Source(FIX)
    src.window_count()
    assert src._data is None
    src.read_windows(0, 2)
    assert src._h5 is not None and src._data is not None


def _write_tiny_seq(root, h, w, downsample_by_2, box_xywh):
    """Write a minimal RVT-layout sequence with one full-res box on repr index 0."""
    import h5py
    import hdf5plugin

    repr_name = "stacked_histogram_dt50_nbins10"
    repr_dir = root / "event_representations_v2" / repr_name
    lab_dir = root / "labels_v2"
    repr_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    n, c = 1, 20
    data = np.zeros((n, c, h, w), dtype=np.uint8)
    h5_name = (
        "event_representations_ds2_nearest.h5"
        if downsample_by_2
        else "event_representations.h5"
    )
    with h5py.File(repr_dir / h5_name, "w") as f:
        f.create_dataset("data", data=data, **hdf5plugin.Blosc())
    np.save(repr_dir / "timestamps_us.npy", np.array([50_000], dtype=np.int64))
    np.save(repr_dir / "objframe_idx_2_repr_idx.npy", np.array([0], dtype=np.int64))

    rows = _make_rows([box_xywh])
    np.savez(
        lab_dir / "labels.npz",
        labels=rows,
        objframe_idx_2_label_idx=np.array([0], dtype=np.int64),
    )
    np.save(lab_dir / "timestamps_us.npy", np.array([50_000], dtype=np.int64))


def test_read_window_gt_scales_to_ds2_when_downsampling(tmp_path):
    # Full-resolution box stored on disk; ds2 repr is 360x640.
    root = tmp_path / "ds2_seq"
    _write_tiny_seq(
        root, h=360, w=640, downsample_by_2=True, box_xywh=(100.0, 200.0, 40.0, 60.0)
    )
    src = PreprocessedH5Source(root, downsample_by_2=True)
    gt = src.read_window_gt(0)
    assert gt is not None and gt.shape == (1,)
    # box halved into ds2 space.
    np.testing.assert_allclose(gt["x"], [50.0])
    np.testing.assert_allclose(gt["y"], [100.0])
    np.testing.assert_allclose(gt["w"], [20.0])
    np.testing.assert_allclose(gt["h"], [30.0])


def test_read_window_gt_unscaled_when_not_downsampling(tmp_path):
    # Same full-resolution box, full-res repr (720x1280): returned verbatim.
    root = tmp_path / "full_seq"
    _write_tiny_seq(
        root, h=720, w=1280, downsample_by_2=False, box_xywh=(100.0, 200.0, 40.0, 60.0)
    )
    src = PreprocessedH5Source(root, downsample_by_2=False)
    gt = src.read_window_gt(0)
    assert gt is not None and gt.shape == (1,)
    np.testing.assert_allclose(gt["x"], [100.0])
    np.testing.assert_allclose(gt["y"], [200.0])
    np.testing.assert_allclose(gt["w"], [40.0])
    np.testing.assert_allclose(gt["h"], [60.0])
