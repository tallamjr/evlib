"""RVT bbox filter chain for label preprocessing (build-side, task B1).

Reads a raw Prophesee ``*_bbox.npy`` structured array and applies RVT's exact
bbox filter chain, producing a filtered structured array. The arithmetic mirrors
``lib/RVT/scripts/genx/preprocess_dataset.py`` precisely (the right/bottom crop
bounds use ``W-1``/``H-1``) so the later byte-identical gate (B3) holds.

The ``track_id`` and ``class_confidence`` fields are carried through every
filter, matching the real on-disk schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

# Verified on-disk schema from the staged real data. The field order is
# load-bearing: B2/B3 write ``labels.npz`` from arrays of exactly this dtype.
BBOX_DTYPE = np.dtype(
    [
        ("t", "<u8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("class_id", "u1"),
        ("class_confidence", "<f4"),
        ("track_id", "<u4"),
    ]
)

LABEL_NPZ_FIELDS = tuple(BBOX_DTYPE.names)

# gen1/gen4 frame geometry (RVT preprocess_dataset.py lines 58-59).
_DATASET_HEIGHT = {"gen1": 240, "gen4": 720}
_DATASET_WIDTH = {"gen1": 304, "gen4": 1280}


class NoLabelsError(Exception):
    """Raised when a sequence filters down to zero boxes.

    Mirrors RVT's ``NoLabelsException`` intent: zero surviving labels is a named
    failure, never a silently returned empty array.
    """


def read_raw_bbox(path: Union[str, Path]) -> np.ndarray:
    """Load a raw structured ``*_bbox.npy`` and validate its fields.

    Raises ``ValueError`` naming the path if the array is not structured with
    exactly the expected fields. Does not coerce silently.
    """
    path = Path(path)
    labels = np.load(str(path))
    if labels.dtype.names is None:
        raise ValueError(
            f"{path} is not a structured bbox array (got dtype {labels.dtype})"
        )
    actual = tuple(labels.dtype.names)
    if actual != LABEL_NPZ_FIELDS:
        raise ValueError(
            f"{path} has unexpected bbox fields {actual}; expected {LABEL_NPZ_FIELDS}"
        )
    return labels


def keep_classes_gen4(labels: np.ndarray) -> np.ndarray:
    """Keep gen4 classes pedestrian/two-wheeler/car (``class_id <= 2``).

    Drops truck, bus, traffic sign, traffic light. gen4 only.
    (RVT ``prophesee_remove_labels_filter_gen4``, lines 263-271.)
    """
    keep = labels["class_id"] <= 2
    return labels[keep]


def crop_to_fov(labels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Clamp box edges to the FOV, recompute w/h, drop degenerate boxes.

    Matches RVT ``crop_to_fov_filter`` (lines 232-260): clamp left/top/right/
    bottom edges to ``[0, W-1]`` / ``[0, H-1]``, recompute w/h from the clamped
    edges, then drop boxes with ``w <= 0`` or ``h <= 0``. Operates on a copy so
    the caller's array is not mutated.
    """
    labels = labels.copy()
    x_left = labels["x"]
    y_top = labels["y"]
    x_right = x_left + labels["w"]
    y_bottom = y_top + labels["h"]

    x_left_cropped = np.clip(x_left, a_min=0, a_max=width - 1)
    y_top_cropped = np.clip(y_top, a_min=0, a_max=height - 1)
    x_right_cropped = np.clip(x_right, a_min=0, a_max=width - 1)
    y_bottom_cropped = np.clip(y_bottom, a_min=0, a_max=height - 1)

    w_cropped = x_right_cropped - x_left_cropped
    h_cropped = y_bottom_cropped - y_top_cropped
    assert np.all(w_cropped >= 0)
    assert np.all(h_cropped >= 0)

    labels["x"] = x_left_cropped
    labels["y"] = y_top_cropped
    labels["w"] = w_cropped
    labels["h"] = h_cropped

    keep = (labels["w"] > 0) & (labels["h"] > 0)
    return labels[keep]


def conservative_size_filter(labels: np.ndarray) -> np.ndarray:
    """Keep boxes with ``w >= 5 AND h >= 5`` (RVT ``conservative_bbox_filter``).

    Used when ``apply_psee_bbox_filter`` is False (the gen4 config).
    """
    min_box_side = 5
    side_ok = (labels["w"] >= min_box_side) & (labels["h"] >= min_box_side)
    return labels[side_ok]


def prophesee_size_filter(labels: np.ndarray, dataset: str) -> np.ndarray:
    """Prophesee diag/side size filter (RVT ``prophesee_bbox_filter``).

    Keeps boxes whose diagonal is at least ``min_box_diag`` and whose width and
    height each reach ``min_box_side``. gen4: diag 60, side 20; gen1: 30, 10.
    Used when ``apply_psee_bbox_filter`` is True (the gen1 branch).
    """
    if dataset not in {"gen1", "gen4"}:
        raise ValueError(f"unknown dataset {dataset!r}")
    min_box_diag = 60 if dataset == "gen4" else 30
    min_box_side = 20 if dataset == "gen4" else 10

    w_lbl = labels["w"]
    h_lbl = labels["h"]
    diag_ok = w_lbl**2 + h_lbl**2 >= min_box_diag**2
    side_ok = (w_lbl >= min_box_side) & (h_lbl >= min_box_side)
    return labels[diag_ok & side_ok]


def remove_faulty_huge_bbox(labels: np.ndarray, width: int) -> np.ndarray:
    """Drop boxes wider than ``(9 * width) // 10`` (RVT ``remove_faulty_huge``).

    These span the frame horizontally without covering an object. TRAIN-split
    only; the caller gates by split.
    """
    max_width = (9 * width) // 10
    side_ok = labels["w"] <= max_width
    return labels[side_ok]


def apply_filters(
    labels: np.ndarray,
    *,
    dataset: str = "gen4",
    split: str,
    height: int,
    width: int,
    apply_psee_bbox_filter: bool = False,
    apply_faulty_bbox_filter: bool = True,
) -> np.ndarray:
    """Run RVT's filter chain in order, returning the surviving boxes.

    Order (RVT ``apply_filters``, lines 274-288):
    1. gen4 class removal (gen4 only),
    2. crop-to-FOV,
    3. size filter: prophesee diag/side if ``apply_psee_bbox_filter`` else
       conservative (gen4 uses conservative),
    4. faulty-huge removal, TRAIN split only and only if
       ``apply_faulty_bbox_filter``.

    Raises ``NoLabelsError`` if zero boxes survive (never returns empty silently).
    """
    if dataset not in {"gen1", "gen4"}:
        raise ValueError(f"unknown dataset {dataset!r}")

    if dataset == "gen4":
        labels = keep_classes_gen4(labels)
    labels = crop_to_fov(labels, height, width)
    if apply_psee_bbox_filter:
        labels = prophesee_size_filter(labels, dataset)
    else:
        labels = conservative_size_filter(labels)
    if split == "train" and apply_faulty_bbox_filter:
        labels = remove_faulty_huge_bbox(labels, width)

    if labels.shape[0] == 0:
        raise NoLabelsError(
            f"all boxes removed by the filter chain (dataset={dataset}, split={split})"
        )
    return labels
