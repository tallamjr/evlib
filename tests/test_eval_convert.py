"""Tests for evlib RVT detection / GT -> Prophesee BBOX_DTYPE converters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evlib.eval.convert import gt_rows_to_prophesee, preds_to_prophesee
from evlib.eval.prophesee import BBOX_DTYPE

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"


def test_preds_to_prophesee_corner_to_wh_and_fields():
    # yolox postprocess row: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    pred = np.array(
        [
            [10.0, 20.0, 30.0, 50.0, 0.9, 0.4, 2.0],
            [0.0, 0.0, 5.0, 8.0, 0.5, 0.8, 1.0],
        ],
        dtype=np.float32,
    )
    out = preds_to_prophesee(pred, frame_t=12345)
    assert out.dtype == BBOX_DTYPE
    assert out.shape == (2,)
    # top-left corner x,y preserved; w,h derived from corners.
    np.testing.assert_allclose(out["x"], [10.0, 0.0])
    np.testing.assert_allclose(out["y"], [20.0, 0.0])
    np.testing.assert_allclose(out["w"], [20.0, 5.0])
    np.testing.assert_allclose(out["h"], [30.0, 8.0])
    np.testing.assert_array_equal(out["class_id"], [2, 1])
    # class_confidence must be the CLASS conf alone (column 5), NOT obj*cls.
    np.testing.assert_allclose(out["class_confidence"], [0.4, 0.8])
    np.testing.assert_array_equal(out["t"], [12345, 12345])
    np.testing.assert_array_equal(out["track_id"], [0, 0])


def test_preds_to_prophesee_class_conf_not_obj_times_cls():
    pred = np.array([[1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.0]], dtype=np.float32)
    out = preds_to_prophesee(pred, frame_t=0)
    # If the implementation wrongly used obj*cls it would be 0.30.
    assert out["class_confidence"][0] == pytest.approx(0.6)


def test_preds_to_prophesee_accepts_torch_tensor():
    torch = pytest.importorskip("torch")
    pred = torch.tensor([[10.0, 20.0, 30.0, 50.0, 0.9, 0.4, 2.0]], dtype=torch.float32)
    out = preds_to_prophesee(pred, frame_t=7)
    np.testing.assert_allclose(out["w"], [20.0])
    np.testing.assert_allclose(out["class_confidence"], [0.4])
    assert out["t"][0] == 7


def test_preds_to_prophesee_none_and_empty():
    empty_none = preds_to_prophesee(None, frame_t=99)
    assert empty_none.dtype == BBOX_DTYPE
    assert empty_none.shape == (0,)

    empty_arr = preds_to_prophesee(np.zeros((0, 7), dtype=np.float32), frame_t=99)
    assert empty_arr.dtype == BBOX_DTYPE
    assert empty_arr.shape == (0,)


def _gt_rows_with_track_id():
    dtype = np.dtype(
        [
            ("t", "<f4"),
            ("x", "<f4"),
            ("y", "<f4"),
            ("w", "<f4"),
            ("h", "<f4"),
            ("class_id", "<u4"),
            ("class_confidence", "<f4"),
            ("track_id", "<u4"),
        ]
    )
    rows = np.zeros((2,), dtype=dtype)
    rows["t"] = [1000.0, 1000.0]
    rows["x"] = [4.0, 7.0]
    rows["y"] = [5.0, 8.0]
    rows["w"] = [10.0, 11.0]
    rows["h"] = [12.0, 13.0]
    rows["class_id"] = [0, 2]
    rows["class_confidence"] = [1.0, 1.0]
    rows["track_id"] = [3, 9]
    return rows


def _gt_rows_without_track_id():
    dtype = np.dtype(
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
    rows = np.zeros((2,), dtype=dtype)
    rows["t"] = [2000.0, 2000.0]
    rows["x"] = [1.0, 2.0]
    rows["y"] = [3.0, 4.0]
    rows["w"] = [5.0, 6.0]
    rows["h"] = [7.0, 8.0]
    rows["class_id"] = [1, 0]
    rows["class_confidence"] = [1.0, 1.0]
    return rows


def test_gt_rows_to_prophesee_with_track_id():
    rows = _gt_rows_with_track_id()
    out = gt_rows_to_prophesee(rows)
    assert out.dtype == BBOX_DTYPE
    assert out.shape == (2,)
    np.testing.assert_array_equal(out["t"], [1000, 1000])
    np.testing.assert_allclose(out["x"], [4.0, 7.0])
    np.testing.assert_allclose(out["y"], [5.0, 8.0])
    np.testing.assert_allclose(out["w"], [10.0, 11.0])
    np.testing.assert_allclose(out["h"], [12.0, 13.0])
    np.testing.assert_array_equal(out["class_id"], [0, 2])
    np.testing.assert_array_equal(out["track_id"], [3, 9])
    np.testing.assert_allclose(out["class_confidence"], [1.0, 1.0])


def test_gt_rows_to_prophesee_without_track_id_defaults_zero():
    rows = _gt_rows_without_track_id()
    out = gt_rows_to_prophesee(rows)
    assert out.dtype == BBOX_DTYPE
    # top-left coords preserved exactly.
    np.testing.assert_allclose(out["x"], [1.0, 2.0])
    np.testing.assert_allclose(out["y"], [3.0, 4.0])
    np.testing.assert_allclose(out["w"], [5.0, 6.0])
    np.testing.assert_allclose(out["h"], [7.0, 8.0])
    np.testing.assert_array_equal(out["t"], [2000, 2000])
    np.testing.assert_array_equal(out["class_id"], [1, 0])
    # track_id absent on disk -> defaults to 0.
    np.testing.assert_array_equal(out["track_id"], [0, 0])


def test_gt_rows_to_prophesee_frame_t_consistency():
    rows = _gt_rows_without_track_id()
    out = gt_rows_to_prophesee(rows, frame_t=2000)
    np.testing.assert_array_equal(out["t"], [2000, 2000])
    with pytest.raises(ValueError):
        gt_rows_to_prophesee(rows, frame_t=9999)


def test_gt_rows_to_prophesee_empty():
    rows = _gt_rows_without_track_id()[:0]
    out = gt_rows_to_prophesee(rows)
    assert out.dtype == BBOX_DTYPE
    assert out.shape == (0,)


# --- per-window GT accessor on the source ---

pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from evlib.data.sources import PreprocessedH5Source  # noqa: E402


def test_read_window_gt_returns_on_disk_topleft_and_timestamps():
    src = PreprocessedH5Source(FIX)
    # repr windows with object frames: 1 (1 box), 4 (2 boxes); rest have none.
    gt1 = src.read_window_gt(1)
    assert gt1 is not None
    assert gt1.shape == (1,)
    np.testing.assert_allclose(gt1["x"], [1.0])
    np.testing.assert_allclose(gt1["y"], [1.0])
    np.testing.assert_allclose(gt1["w"], [2.0])
    np.testing.assert_allclose(gt1["h"], [2.0])
    np.testing.assert_array_equal(gt1["class_id"], [0])
    # on-disk timestamps preserved (NOT centre-converted yolox labels).
    np.testing.assert_array_equal(gt1["t"], [0])

    gt4 = src.read_window_gt(4)
    assert gt4 is not None
    assert gt4.shape == (2,)
    np.testing.assert_allclose(gt4["x"], [3.0, 5.0])
    np.testing.assert_allclose(gt4["y"], [3.0, 1.0])
    np.testing.assert_allclose(gt4["w"], [2.0, 1.0])
    np.testing.assert_allclose(gt4["h"], [2.0, 1.0])
    np.testing.assert_array_equal(gt4["class_id"], [1, 0])


def test_read_window_gt_none_for_label_less_windows():
    src = PreprocessedH5Source(FIX)
    for idx in (0, 2, 3, 5):
        assert src.read_window_gt(idx) is None


def test_read_window_gt_matches_prophesee_converter():
    src = PreprocessedH5Source(FIX)
    rows = src.read_window_gt(4)
    proph = gt_rows_to_prophesee(rows)
    assert proph.dtype == BBOX_DTYPE
    np.testing.assert_allclose(proph["x"], [3.0, 5.0])
    np.testing.assert_array_equal(proph["track_id"], [0, 0])
    np.testing.assert_array_equal(proph["t"], [0, 0])


def test_read_window_gt_does_not_change_read_windows():
    src = PreprocessedH5Source(FIX)
    _, labels = src.read_windows(0, 6)
    # read_windows still yields centre-yolox tensors unchanged.
    assert labels[0] is None and labels[2] is None and labels[5] is None
    assert labels[1].shape == (1, 5)
    assert labels[4].shape == (2, 5)
