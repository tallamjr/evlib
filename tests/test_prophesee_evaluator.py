"""Tests for the ported Prophesee detection mAP evaluator.

These exercise the evaluator core with synthetic structured-array ground truth
and detections (real numpy arrays of BBOX_DTYPE, not mocks). The boxes are sized
and timestamped to survive the gen4 box filters (min_box_diag=60, min_box_side=20,
skip_ts=5e5) so the resulting AP reflects the matching logic rather than being
trivially zeroed by filtering.
"""

import numpy as np
import pytest

from evlib.eval.prophesee import BBOX_DTYPE, GEN4_CLASSES, PropheseeEvaluator


# gen4 image dimensions (1280x720).
IMG_HEIGHT = 720
IMG_WIDTH = 1280

# Timestamp comfortably past the 0.5s (5e5 us) skip threshold.
LATE_TS = 1_000_000


def _make_box(t, x, y, w, h, class_id, confidence):
    box = np.zeros((1,), dtype=BBOX_DTYPE)
    box["t"] = t
    box["x"] = x
    box["y"] = y
    box["w"] = w
    box["h"] = h
    box["class_id"] = class_id
    box["track_id"] = 0
    box["class_confidence"] = confidence
    return box


def test_gen4_classes_exposed():
    assert GEN4_CLASSES == ("pedestrian", "two-wheeler", "car")


def test_perfect_match_yields_high_ap():
    # GT and prediction are the same box (same class, same location), so a single
    # perfect match over a single class/frame should give AP near 1.0.
    gt = _make_box(LATE_TS, x=100, y=100, w=120, h=120, class_id=0, confidence=1.0)
    dt = _make_box(LATE_TS, x=100, y=100, w=120, h=120, class_id=0, confidence=0.99)

    evaluator = PropheseeEvaluator(dataset="gen4", downsample_by_2=False)
    evaluator.add_labels([gt])
    evaluator.add_predictions([dt])
    metrics = evaluator.evaluate_buffer(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

    assert metrics is not None
    assert metrics["AP"] >= 0.9
    assert metrics["AP_50"] >= 0.9


def test_miss_yields_low_ap():
    # The prediction is far from the GT and in a different class, so there is no
    # match -> AP should be zero.
    gt = _make_box(LATE_TS, x=100, y=100, w=120, h=120, class_id=0, confidence=1.0)
    dt = _make_box(LATE_TS, x=900, y=500, w=120, h=120, class_id=2, confidence=0.99)

    evaluator = PropheseeEvaluator(dataset="gen4", downsample_by_2=False)
    evaluator.add_labels([gt])
    evaluator.add_predictions([dt])
    metrics = evaluator.evaluate_buffer(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

    assert metrics is not None
    assert metrics["AP"] == 0.0
    assert metrics["AP_50"] == 0.0


def test_evaluate_buffer_asserts_on_length_mismatch():
    gt = _make_box(LATE_TS, x=100, y=100, w=120, h=120, class_id=0, confidence=1.0)
    dt_a = _make_box(LATE_TS, x=100, y=100, w=120, h=120, class_id=0, confidence=0.99)
    dt_b = _make_box(LATE_TS, x=200, y=200, w=120, h=120, class_id=0, confidence=0.99)

    evaluator = PropheseeEvaluator(dataset="gen4", downsample_by_2=False)
    evaluator.add_labels([gt])
    evaluator.add_predictions([dt_a, dt_b])

    with pytest.raises(AssertionError):
        evaluator.evaluate_buffer(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
