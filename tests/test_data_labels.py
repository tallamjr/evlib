import numpy as np
import torch
from evlib.data.labels import boxes_to_yolox, LABEL_FIELDS


def _struct(rows):
    dt = np.dtype(
        [(f, np.float32 if f != "class_id" else np.int32) for f in LABEL_FIELDS]
    )
    arr = np.zeros(len(rows), dtype=dt)
    for i, r in enumerate(rows):
        for f in LABEL_FIELDS:
            arr[f][i] = r[f]
    return arr


def test_empty_frame_returns_none():
    assert boxes_to_yolox(_struct([])) is None


def test_box_centre_conversion():
    arr = _struct(
        [
            {
                "t": 0,
                "x": 10,
                "y": 20,
                "w": 4,
                "h": 6,
                "class_id": 2,
                "class_confidence": 1.0,
            }
        ]
    )
    out = boxes_to_yolox(arr)
    assert out.shape == (1, 5)
    assert out.dtype == torch.float32
    # [class_id, cx, cy, w, h] = [2, 12, 23, 4, 6]
    assert torch.allclose(out[0], torch.tensor([2.0, 12.0, 23.0, 4.0, 6.0]))
