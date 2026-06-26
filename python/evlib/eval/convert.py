"""Convert evlib RVT detections and on-disk GT labels to Prophesee BBOX_DTYPE.

The Prophesee evaluator (``evlib.eval.prophesee``) consumes structured arrays of
``BBOX_DTYPE`` with top-left-corner boxes. Two converters bridge to it:

- ``preds_to_prophesee``: a yolox ``postprocess`` ``[N, 7]`` detection block for one
  frame (columns ``x1, y1, x2, y2, obj_conf, class_conf, class_pred``) -> BBOX rows,
  with corner coords turned into top-left ``x, y`` + ``w, h`` and the CLASS confidence
  alone used for ``class_confidence`` (matching the RVT reference ``to_prophesee``).
- ``gt_rows_to_prophesee``: on-disk ``labels.npz`` structured rows (already top-left,
  already carrying ``t``) -> BBOX rows, defaulting ``track_id`` to 0 when the on-disk
  schema omits it.

Reference: ``lib/ssms_event_cameras/RVT/utils/evaluation/prophesee/io/box_loading.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from evlib.eval.prophesee import BBOX_DTYPE


def preds_to_prophesee(yolox_pred, frame_t: int) -> np.ndarray:
    """Convert one frame's yolox detections to a ``BBOX_DTYPE`` array.

    ``yolox_pred`` is the evlib ``postprocess`` output for a single image: an
    ``[N, 7]`` array/tensor with columns ``(x1, y1, x2, y2, obj_conf, class_conf,
    class_pred)`` in corner coordinates, or ``None`` when there are no detections.

    The mapping mirrors the RVT reference: ``x = x1``, ``y = y1``,
    ``w = x2 - x1``, ``h = y2 - y1``, ``class_id = class_pred``,
    ``class_confidence = class_conf`` (the class confidence ALONE, not
    ``obj_conf * class_conf``), ``t = frame_t``, ``track_id = 0``.
    """
    num_pred = 0 if yolox_pred is None else int(yolox_pred.shape[0])
    out = np.zeros((num_pred,), dtype=BBOX_DTYPE)
    if num_pred == 0:
        return out

    preds = _to_numpy(yolox_pred)
    if preds.ndim != 2 or preds.shape[1] != 7:
        raise ValueError(f"expected yolox_pred of shape [N, 7], got {preds.shape}")

    out["t"] = np.full((num_pred,), frame_t, dtype=BBOX_DTYPE["t"])
    out["x"] = preds[:, 0].astype(BBOX_DTYPE["x"])
    out["y"] = preds[:, 1].astype(BBOX_DTYPE["y"])
    out["w"] = (preds[:, 2] - preds[:, 0]).astype(BBOX_DTYPE["w"])
    out["h"] = (preds[:, 3] - preds[:, 1]).astype(BBOX_DTYPE["h"])
    out["class_id"] = preds[:, 6].astype(BBOX_DTYPE["class_id"])
    # Column 5 is the class confidence alone; the obj confidence (column 4) is
    # deliberately not multiplied in, matching the RVT reference to_prophesee.
    out["class_confidence"] = preds[:, 5].astype(BBOX_DTYPE["class_confidence"])
    out["track_id"] = np.zeros((num_pred,), dtype=BBOX_DTYPE["track_id"])
    return out


def gt_rows_to_prophesee(
    structured_rows: np.ndarray, frame_t: Optional[int] = None
) -> np.ndarray:
    """Convert on-disk ``labels.npz`` GT rows to a ``BBOX_DTYPE`` array.

    The rows already carry ``t`` and top-left ``x, y, w, h`` plus ``class_id``;
    they are mapped field-by-field, NOT re-derived from the centre-yolox training
    format. The real gen4 schema includes ``track_id``; the tracked ``mini_seq``
    fixture omits it, in which case ``track_id`` defaults to 0.

    If ``frame_t`` is supplied it must agree with the rows' on-disk ``t`` (raising
    ``ValueError`` otherwise) and is used as the output timestamp.
    """
    names = structured_rows.dtype.names
    if names is None:
        raise ValueError("structured_rows must be a structured numpy array")

    out = np.zeros((len(structured_rows),), dtype=BBOX_DTYPE)
    for name in ("t", "x", "y", "w", "h", "class_id", "class_confidence"):
        if name not in names:
            raise ValueError(f"GT rows missing required field {name!r}")
        out[name] = structured_rows[name].astype(BBOX_DTYPE[name])

    if "track_id" in names:
        out["track_id"] = structured_rows["track_id"].astype(BBOX_DTYPE["track_id"])
    # else: track_id stays at its zero-fill default.

    if frame_t is not None and len(structured_rows) > 0:
        unique_t = np.unique(out["t"])
        if unique_t.size != 1 or int(unique_t.item()) != int(frame_t):
            raise ValueError(
                f"frame_t={frame_t} disagrees with on-disk GT timestamps "
                f"{unique_t.tolist()}"
            )
        out["t"] = np.full((len(out),), frame_t, dtype=BBOX_DTYPE["t"])
    return out


def _to_numpy(arr) -> np.ndarray:
    """Return ``arr`` as a numpy array, detaching a torch tensor if needed."""
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)
