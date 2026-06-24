"""Read-side object-label handling: RVT structured array -> yolox box tensors."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

LABEL_FIELDS = ("t", "x", "y", "w", "h", "class_id", "class_confidence")


def boxes_to_yolox(frame_boxes: np.ndarray) -> Optional[torch.Tensor]:
    """Convert one frame's structured boxes to a [num_boxes, 5] float32 tensor.

    Row layout is yolox: [class_id, cx, cy, w, h], with cx/cy the box centre
    (RVT stores top-left x/y). Returns None when the frame has no boxes.
    """
    if frame_boxes.shape[0] == 0:
        return None
    x = frame_boxes["x"].astype(np.float32)
    y = frame_boxes["y"].astype(np.float32)
    w = frame_boxes["w"].astype(np.float32)
    h = frame_boxes["h"].astype(np.float32)
    cls = frame_boxes["class_id"].astype(np.float32)
    out = np.stack([cls, x + w / 2.0, y + h / 2.0, w, h], axis=1)
    return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))
