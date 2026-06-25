"""TDD tests for RVT bbox filter chain (label preprocessing, task B1).

Small real structured arrays of the exact on-disk 8-field dtype exercise each
filter's known-kept/known-dropped members and the boundary cases from the brief.
"""

from __future__ import annotations

import numpy as np
import pytest

from evlib.data.label_preprocess import (
    BBOX_DTYPE,
    NoLabelsError,
    apply_filters,
    conservative_size_filter,
    crop_to_fov,
    keep_classes_gen4,
    prophesee_size_filter,
    read_raw_bbox,
    remove_faulty_huge_bbox,
)

GEN4_HEIGHT = 720
GEN4_WIDTH = 1280


def make_bbox(rows) -> np.ndarray:
    """Build a structured bbox array of the exact on-disk dtype from row tuples.

    Each row is (t, x, y, w, h, class_id, class_confidence, track_id).
    """
    return np.array(rows, dtype=BBOX_DTYPE)


def test_bbox_dtype_field_order_and_types():
    expected = [
        ("t", "<u8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("class_id", "|u1"),
        ("class_confidence", "<f4"),
        ("track_id", "<u4"),
    ]
    assert BBOX_DTYPE.descr == expected


# --- read_raw_bbox -----------------------------------------------------------


def test_read_raw_bbox_roundtrip(tmp_path):
    arr = make_bbox(
        [
            (10, 1.0, 2.0, 30.0, 40.0, 0, 0.9, 7),
            (20, 5.0, 6.0, 50.0, 60.0, 2, 0.8, 8),
        ]
    )
    path = tmp_path / "seq_bbox.npy"
    np.save(path, arr)
    loaded = read_raw_bbox(path)
    assert loaded.dtype == BBOX_DTYPE
    assert np.array_equal(loaded, arr)


def test_read_raw_bbox_rejects_wrong_fields(tmp_path):
    wrong = np.array(
        [(1, 2.0, 3.0)],
        dtype=[("t", "<u8"), ("x", "<f4"), ("y", "<f4")],
    )
    path = tmp_path / "bad_bbox.npy"
    np.save(path, wrong)
    with pytest.raises(ValueError) as exc:
        read_raw_bbox(path)
    assert str(path) in str(exc.value)


# --- keep_classes_gen4 -------------------------------------------------------


def test_keep_classes_gen4_keeps_0_1_2_drops_3_to_6():
    labels = make_bbox(
        [(1, 0.0, 0.0, 10.0, 10.0, cls, 1.0, cls) for cls in (0, 1, 2, 3, 4, 5, 6)]
    )
    kept = keep_classes_gen4(labels)
    assert sorted(kept["class_id"].tolist()) == [0, 1, 2]
    # track_id carried through on the kept rows.
    assert sorted(kept["track_id"].tolist()) == [0, 1, 2]


# --- crop_to_fov -------------------------------------------------------------


def test_crop_to_fov_clamps_negative_x_and_recomputes_w():
    # x=-10, w=50 -> x clamped to 0; right edge = -10 + 50 = 40 -> w = 40.
    labels = make_bbox([(1, -10.0, 5.0, 50.0, 30.0, 0, 0.5, 11)])
    cropped = crop_to_fov(labels, GEN4_HEIGHT, GEN4_WIDTH)
    assert cropped.shape[0] == 1
    assert cropped["x"][0] == 0.0
    assert cropped["w"][0] == 40.0
    assert cropped["y"][0] == 5.0
    assert cropped["h"][0] == 30.0
    assert cropped["track_id"][0] == 11


def test_crop_to_fov_clamps_right_edge_to_width_minus_one():
    # Right edge well past the frame: x=1270, w=100 -> right clamped to W-1=1279.
    labels = make_bbox([(1, 1270.0, 10.0, 100.0, 30.0, 1, 0.5, 12)])
    cropped = crop_to_fov(labels, GEN4_HEIGHT, GEN4_WIDTH)
    assert cropped.shape[0] == 1
    assert cropped["x"][0] == 1270.0
    # 1279 - 1270 = 9, NOT 1280 - 1270 = 10 (W-1 bound matters).
    assert cropped["w"][0] == 9.0


def test_crop_to_fov_clamps_bottom_edge_to_height_minus_one():
    labels = make_bbox([(1, 10.0, 710.0, 30.0, 100.0, 1, 0.5, 13)])
    cropped = crop_to_fov(labels, GEN4_HEIGHT, GEN4_WIDTH)
    assert cropped.shape[0] == 1
    # 719 - 710 = 9, NOT 720 - 710 = 10.
    assert cropped["h"][0] == 9.0


def test_crop_to_fov_drops_box_fully_outside():
    # Entirely left of the frame: x=-100, w=50 -> right edge -50, both clamp to 0.
    labels = make_bbox(
        [
            (1, -100.0, 5.0, 50.0, 30.0, 0, 0.5, 21),
            (2, 100.0, 5.0, 30.0, 30.0, 0, 0.5, 22),
        ]
    )
    cropped = crop_to_fov(labels, GEN4_HEIGHT, GEN4_WIDTH)
    assert cropped.shape[0] == 1
    assert cropped["track_id"][0] == 22


# --- conservative_size_filter ------------------------------------------------


def test_conservative_size_filter_boundary_at_five():
    labels = make_bbox(
        [
            (1, 0.0, 0.0, 5.0, 5.0, 0, 0.5, 31),  # exactly 5 -> kept
            (2, 0.0, 0.0, 4.999, 10.0, 0, 0.5, 32),  # w < 5 -> dropped
            (3, 0.0, 0.0, 10.0, 4.999, 0, 0.5, 33),  # h < 5 -> dropped
            (4, 0.0, 0.0, 6.0, 6.0, 0, 0.5, 34),  # both > 5 -> kept
        ]
    )
    kept = conservative_size_filter(labels)
    assert sorted(kept["track_id"].tolist()) == [31, 34]


# --- prophesee_size_filter (gen1 branch, diag/side) --------------------------


def test_prophesee_size_filter_gen4_diag_and_side():
    # gen4: min_box_diag=60, min_box_side=20.
    labels = make_bbox(
        [
            (1, 0.0, 0.0, 20.0, 60.0, 0, 0.5, 41),  # side ok, diag sqrt(400+3600)=63 ok
            (2, 0.0, 0.0, 19.0, 60.0, 0, 0.5, 42),  # side < 20 -> dropped
            (3, 0.0, 0.0, 30.0, 30.0, 0, 0.5, 43),  # diag sqrt(1800)=42 < 60 -> dropped
            (4, 0.0, 0.0, 50.0, 50.0, 0, 0.5, 44),  # both ok
        ]
    )
    kept = prophesee_size_filter(labels, "gen4")
    assert sorted(kept["track_id"].tolist()) == [41, 44]


# --- remove_faulty_huge_bbox -------------------------------------------------


def test_remove_faulty_huge_bbox_drops_above_threshold():
    max_width = (9 * GEN4_WIDTH) // 10  # 1152
    labels = make_bbox(
        [
            (1, 0.0, 0.0, float(max_width), 10.0, 0, 0.5, 51),  # == max -> kept
            (2, 0.0, 0.0, float(max_width + 1), 10.0, 0, 0.5, 52),  # > max -> dropped
        ]
    )
    kept = remove_faulty_huge_bbox(labels, GEN4_WIDTH)
    assert sorted(kept["track_id"].tolist()) == [51]


# --- apply_filters: split-conditional faulty filter --------------------------


def _huge_box_array():
    # w = 1200 > (9*1280)//10 = 1152 -> only dropped on train.
    return make_bbox(
        [
            (1, 0.0, 0.0, 1200.0, 30.0, 0, 0.5, 61),
            (2, 0.0, 0.0, 50.0, 50.0, 1, 0.5, 62),
        ]
    )


def test_apply_filters_faulty_drops_on_train():
    kept = apply_filters(
        _huge_box_array(),
        dataset="gen4",
        split="train",
        height=GEN4_HEIGHT,
        width=GEN4_WIDTH,
    )
    assert kept["track_id"].tolist() == [62]


def test_apply_filters_faulty_kept_on_val():
    kept = apply_filters(
        _huge_box_array(),
        dataset="gen4",
        split="val",
        height=GEN4_HEIGHT,
        width=GEN4_WIDTH,
    )
    assert sorted(kept["track_id"].tolist()) == [61, 62]


# --- apply_filters: full chain on a mixed array ------------------------------


def test_apply_filters_full_chain_mixed_array():
    labels = make_bbox(
        [
            # class 4 (traffic sign) -> dropped by class filter
            (1, 100.0, 100.0, 50.0, 50.0, 4, 0.5, 71),
            # negative x clamped, then large enough -> kept
            (2, -10.0, 5.0, 60.0, 60.0, 0, 0.9, 72),
            # tiny box (w<5 after no crop) -> dropped by conservative size
            (3, 200.0, 200.0, 4.0, 100.0, 1, 0.8, 73),
            # class 2 car, valid size -> kept
            (4, 300.0, 300.0, 40.0, 40.0, 2, 0.7, 74),
        ]
    )
    kept = apply_filters(
        labels,
        dataset="gen4",
        split="val",
        height=GEN4_HEIGHT,
        width=GEN4_WIDTH,
    )
    # Order preserved: row 72 then 74.
    assert kept["track_id"].tolist() == [72, 74]
    # crop applied to row 72: x=-10 -> 0, right edge -10+60=50 -> w=50.
    row72 = kept[kept["track_id"] == 72][0]
    assert row72["x"] == 0.0
    assert row72["w"] == 50.0
    # class_confidence carried through.
    assert row72["class_confidence"] == np.float32(0.9)
    assert kept[kept["track_id"] == 74][0]["class_confidence"] == np.float32(0.7)


def test_apply_filters_raises_when_all_removed():
    # Single class-4 box -> removed by class filter -> zero boxes.
    labels = make_bbox([(1, 100.0, 100.0, 50.0, 50.0, 4, 0.5, 81)])
    with pytest.raises(NoLabelsError):
        apply_filters(
            labels,
            dataset="gen4",
            split="val",
            height=GEN4_HEIGHT,
            width=GEN4_WIDTH,
        )


def test_apply_filters_psee_branch_uses_prophesee_size():
    # With apply_psee_bbox_filter=True the prophesee (diag/side) filter runs.
    # Box w=10,h=10: passes conservative (>=5) but fails gen4 side (>=20).
    labels = make_bbox([(1, 100.0, 100.0, 10.0, 10.0, 0, 0.5, 91)])
    with pytest.raises(NoLabelsError):
        apply_filters(
            labels,
            dataset="gen4",
            split="val",
            height=GEN4_HEIGHT,
            width=GEN4_WIDTH,
            apply_psee_bbox_filter=True,
        )
