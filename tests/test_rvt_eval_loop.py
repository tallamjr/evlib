"""Loop-level tests for the RVT gen4 sequence-eval harness.

These exercise ``evlib.eval.rvt_eval.evaluate_rvt_gen4`` end to end with a real
``RVT(variant="tiny", num_classes=3)`` model (random init is fine; the loop, not
accuracy, is under test) and the real ``PropheseeEvaluator``. No mocks of the
model or evaluator are used.

A tiny in-memory ``ReprSource`` stand-in supplies backbone-valid windows: the
tracked ``mini_seq`` fixture is 8x12, too small for the real backbone (input must
be divisible by 32 and reduce to a final stage divisible by 7), so we synthesise
224x224 windows in small counts. Some windows carry synthetic top-left GT boxes
with on-disk timestamps and some do not, which lets us assert that only
GT-bearing frames are scored.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch

from evlib.data import LABEL_NPZ_FIELDS

pytestmark = pytest.mark.slow

# Backbone-valid spatial size: divisible by 32 and the final stage (224/32=7) is
# divisible by 7 for the MaxViT window/grid attention.
WINDOW_H = 224
WINDOW_W = 224
NBINS = 10
CHANNELS = 2 * NBINS

# The harness pads bottom/right to a backbone multiple; 224 is already a multiple
# of 64 so the padded HW equals the window HW for this synthetic source.
PADDED_H = WINDOW_H
PADDED_W = WINDOW_W

METRIC_KEYS = {"AP", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L"}

# Window indices (within a source) that carry GT, with the on-disk timestamp the
# rows are stamped with. A frame index of None means no GT (must not be scored).
DELTA_T_US = 50_000


def _make_gt_rows(window_idx: int) -> np.ndarray:
    """Build a few synthetic top-left GT boxes for one window.

    Boxes are large enough to clear the gen4 box filters (min_box_diag=30,
    min_box_side=10) and are stamped with a strictly increasing per-window
    timestamp above the skip_ts=0.5e6 cutoff so the evaluator keeps them.
    """
    dtype = np.dtype(
        {
            "names": list(LABEL_NPZ_FIELDS),
            "formats": ["<i8", "<f4", "<f4", "<f4", "<f4", "<u4", "<f4", "<u4"],
        }
    )
    t_us = 1_000_000 + window_idx * DELTA_T_US
    boxes = [
        (t_us, 20.0, 30.0, 60.0, 50.0, 0, 1.0, 0),
        (t_us, 120.0, 80.0, 40.0, 70.0, 1, 1.0, 0),
    ]
    rows = np.zeros((len(boxes),), dtype=dtype)
    for i, (t, x, y, w, h, c, conf, tid) in enumerate(boxes):
        rows[i] = (t, x, y, w, h, c, conf, tid)
    return rows


class SyntheticSource:
    """In-memory ReprSource: backbone-valid windows + per-window top-left GT.

    Implements the ``ReprSource`` protocol (``window_count``/``read_windows``)
    plus the Task-2 ``read_window_gt`` accessor. ``gt_windows`` lists which window
    indices carry GT; the rest return ``None`` from ``read_window_gt`` and an
    all-``None`` label list from ``read_windows``.
    """

    def __init__(self, n_windows: int, gt_windows: List[int], seed: int = 0) -> None:
        self._n = n_windows
        self._gt_windows = set(gt_windows)
        rng = np.random.default_rng(seed)
        # Deterministic uint8 windows so a repeated run is bit-identical.
        self._data = rng.integers(
            0, 4, size=(n_windows, CHANNELS, WINDOW_H, WINDOW_W), dtype=np.uint8
        )

    def window_count(self) -> int:
        return self._n

    def read_windows(
        self, lo: int, hi: int
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        if lo < 0 or hi > self._n or lo >= hi:
            raise ValueError(f"range [{lo},{hi}) out of bounds for {self._n}")
        ev = [
            torch.from_numpy(np.ascontiguousarray(self._data[i])) for i in range(lo, hi)
        ]
        labels: List[Optional[torch.Tensor]] = []
        for repr_i in range(lo, hi):
            rows = self.read_window_gt(repr_i)
            if rows is None:
                labels.append(None)
            else:
                from evlib.data.labels import boxes_to_yolox

                labels.append(boxes_to_yolox(rows))
        return ev, labels

    def read_window_gt(self, repr_idx: int) -> Optional[np.ndarray]:
        if repr_idx in self._gt_windows:
            return _make_gt_rows(repr_idx)
        return None


def _make_model():
    from evlib.models.rvt import RVT

    torch.manual_seed(0)
    model = RVT(variant="tiny", num_classes=3)
    model.eval()
    return model


def test_evaluate_rvt_gen4_returns_metric_dict():
    """The harness runs end to end and returns the COCO metric keys."""
    from evlib.eval.rvt_eval import evaluate_rvt_gen4

    model = _make_model()
    # Two short sequences (separate sources => is_first_sample fires twice).
    sources = [
        SyntheticSource(n_windows=4, gt_windows=[1, 3], seed=1),
        SyntheticSource(n_windows=3, gt_windows=[0, 2], seed=2),
    ]
    metrics = evaluate_rvt_gen4(
        model,
        sources,
        sequence_length=2,
        batch_size=1,
        device="cpu",
        padded_hw=(PADDED_H, PADDED_W),
    )
    assert isinstance(metrics, dict)
    assert METRIC_KEYS.issubset(metrics.keys())


def test_only_gt_bearing_frames_are_scored():
    """Frames without GT are never added to the evaluator buffer."""
    from evlib.eval import rvt_eval

    model = _make_model()
    # 5 windows, only two carry GT.
    gt_windows = [1, 4]
    source = SyntheticSource(n_windows=5, gt_windows=gt_windows, seed=3)

    added_label_frames: List[int] = []
    added_pred_frames: List[int] = []

    real_add_labels = rvt_eval.PropheseeEvaluator.add_labels
    real_add_predictions = rvt_eval.PropheseeEvaluator.add_predictions

    def counting_add_labels(self, labels):
        added_label_frames.append(len(labels))
        return real_add_labels(self, labels)

    def counting_add_predictions(self, predictions):
        added_pred_frames.append(len(predictions))
        return real_add_predictions(self, predictions)

    rvt_eval.PropheseeEvaluator.add_labels = counting_add_labels
    rvt_eval.PropheseeEvaluator.add_predictions = counting_add_predictions
    try:
        rvt_eval.evaluate_rvt_gen4(
            model,
            [source],
            sequence_length=2,
            batch_size=1,
            device="cpu",
            padded_hw=(PADDED_H, PADDED_W),
        )
    finally:
        rvt_eval.PropheseeEvaluator.add_labels = real_add_labels
        rvt_eval.PropheseeEvaluator.add_predictions = real_add_predictions

    # Exactly the GT-bearing windows are scored, no padded/empty frames.
    assert sum(added_label_frames) == len(gt_windows)
    assert sum(added_pred_frames) == len(gt_windows)


def test_state_reset_makes_runs_deterministic():
    """is_first_sample state reset makes a repeated run bit-identical."""
    from evlib.eval.rvt_eval import evaluate_rvt_gen4

    def run_once():
        model = _make_model()
        source = SyntheticSource(n_windows=4, gt_windows=[1, 3], seed=7)
        return evaluate_rvt_gen4(
            model,
            [source],
            sequence_length=2,
            batch_size=1,
            device="cpu",
            padded_hw=(PADDED_H, PADDED_W),
        )

    first = run_once()
    second = run_once()
    assert first["AP"] == second["AP"]
    assert first["AP_50"] == second["AP_50"]
