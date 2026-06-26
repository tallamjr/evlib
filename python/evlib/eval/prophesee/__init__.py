"""Prophesee detection mAP evaluator (ported from the RVT reference).

Public surface:
- PropheseeEvaluator: buffered evaluator for gen1/gen4 detections.
- BBOX_DTYPE: structured numpy dtype for top-left-corner boxes.
- GEN1_CLASSES / GEN4_CLASSES: per-camera class tuples used for scoring.
- evaluate_list / evaluate_detection / filter_boxes: lower-level entry points.
"""

from .box_filtering import filter_boxes
from .box_loading import BBOX_DTYPE, reformat_boxes
from .coco_eval import evaluate_detection
from .evaluation import evaluate_list
from .evaluator import PropheseeEvaluator

GEN1_CLASSES = ("car", "pedestrian")
GEN4_CLASSES = ("pedestrian", "two-wheeler", "car")

__all__ = [
    "PropheseeEvaluator",
    "BBOX_DTYPE",
    "GEN1_CLASSES",
    "GEN4_CLASSES",
    "evaluate_list",
    "evaluate_detection",
    "filter_boxes",
    "reformat_boxes",
]
