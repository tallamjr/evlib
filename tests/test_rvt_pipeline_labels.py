"""--labels-npy must write the RVT label artifacts via evlib.data.label_preprocess
(2026-08-08 P3 finding: the phantom evlib.rvt.labels import made it a silent no-op)."""

from pathlib import Path

import numpy as np
import pytest

from evlib.data.label_preprocess import (
    BBOX_DTYPE,
    apply_filters,
    build_objframes_and_grid,
)
from evlib.rvt import pipeline


@pytest.fixture()
def bbox_npy(tmp_path):
    # Dense 60 Hz raw labels (median diff 16667 us) as the gen4 cadence check
    # requires; the frame walker then selects ~10 Hz object frames from them.
    # Boxes are large enough to pass every filter (class_id 0 is kept by
    # keep_classes_gen4).
    n = 30
    raw = np.zeros(n, dtype=BBOX_DTYPE)
    raw["t"] = 100_000 + 16_667 * np.arange(n, dtype=np.uint64)
    raw["x"] = 100.0
    raw["y"] = 100.0
    raw["w"] = 120.0
    raw["h"] = 120.0
    raw["class_id"] = 0
    raw["class_confidence"] = 1.0
    raw["track_id"] = np.arange(n, dtype=np.uint32)
    path = tmp_path / "seq_bbox.npy"
    np.save(path, raw)
    return path


def _expected(bbox_path, split="val"):
    filtered = apply_filters(
        np.load(bbox_path), dataset="gen4", split=split, height=720, width=1280
    )
    return build_objframes_and_grid(filtered, dataset="gen4")


def test_write_labels_writes_rvt_artifacts(tmp_path, bbox_npy):
    expected = _expected(bbox_npy)
    out_dir = tmp_path / "out"
    pipeline._write_labels(
        bbox_npy, "val", "gen4", out_dir, expected.ev_repr_timestamps_us_end, 720, 1280
    )

    labels_npz = np.load(out_dir / "labels_v2" / "labels.npz")
    np.testing.assert_array_equal(labels_npz["labels"], expected.labels)
    np.testing.assert_array_equal(
        labels_npz["objframe_idx_2_label_idx"], expected.objframe_idx_2_label_idx
    )
    repr_dir = out_dir / "event_representations_v2" / pipeline.REPR_NAME
    np.testing.assert_array_equal(
        np.load(repr_dir / "objframe_idx_2_repr_idx.npy"),
        expected.objframe_idx_2_repr_idx,
    )


def test_write_labels_rejects_mismatched_grid(tmp_path, bbox_npy):
    expected = _expected(bbox_npy)
    with pytest.raises(ValueError, match="grid"):
        pipeline._write_labels(
            bbox_npy,
            "val",
            "gen4",
            tmp_path / "out",
            expected.ev_repr_timestamps_us_end + 1,
            720,
            1280,
        )


def test_write_labels_none_is_noop(tmp_path):
    pipeline._write_labels(
        None, "val", "gen4", tmp_path / "out", np.array([]), 720, 1280
    )
    assert not (tmp_path / "out").exists()


REAL_BBOX = (
    Path(__file__).parent.parent
    / "data"
    / "gen4_label_preprocess"
    / "moorea_2019-06-17_test_02_000_3172500000_3232500000"
    / "moorea_2019-06-17_test_02_000_3172500000_3232500000_bbox.npy"
)


def test_write_labels_real_gen4_bbox(tmp_path):
    # Real data check over a real gen4 recording's raw bbox file. The file is
    # gitignored, so skip when absent (CI); it runs locally.
    if not REAL_BBOX.exists():
        pytest.skip("gitignored real bbox file not present")
    expected = _expected(REAL_BBOX, split="test")
    out_dir = tmp_path / "out"
    pipeline._write_labels(
        REAL_BBOX,
        "test",
        "gen4",
        out_dir,
        expected.ev_repr_timestamps_us_end,
        720,
        1280,
    )
    labels_npz = np.load(out_dir / "labels_v2" / "labels.npz")
    assert len(labels_npz["labels"]) == len(expected.labels) > 0
