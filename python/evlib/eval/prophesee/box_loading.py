"""Structured box dtype used by the Prophesee detection evaluator.

Ported from the RVT reference
(utils/evaluation/prophesee/io/box_loading.py). Only the ObjectLabels-free
parts are ported here: the BBOX_DTYPE structured dtype and the
backward-compatible reformat helper. The to_prophesee / loaded_label_to_prophesee
converters depend on data.genx_utils.labels.ObjectLabels and are deliberately
left out (they belong to the format-converter task).

Copyright: (c) 2019-2020 Prophesee
"""

from __future__ import print_function

import numpy as np

BBOX_DTYPE = np.dtype(
    {
        "names": ["t", "x", "y", "w", "h", "class_id", "track_id", "class_confidence"],
        "formats": ["<i8", "<f4", "<f4", "<f4", "<f4", "<u4", "<u4", "<f4"],
        "offsets": [0, 8, 12, 16, 20, 24, 28, 32],
        "itemsize": 40,
    }
)


def reformat_boxes(boxes):
    """ReFormat boxes according to new rule.

    This allows to be backward-compatible with imerit annotation.
        't' = 'ts'
        'class_confidence' = 'confidence'
    """
    if "t" not in boxes.dtype.names or "class_confidence" not in boxes.dtype.names:
        new = np.zeros((len(boxes),), dtype=BBOX_DTYPE)
        for name in boxes.dtype.names:
            if name == "ts":
                new["t"] = boxes[name]
            elif name == "confidence":
                new["class_confidence"] = boxes[name]
            else:
                new[name] = boxes[name]
        return new
    else:
        return boxes
