"""The sequence/batch contract shared by every dataset and collate function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


class DataKey:
    EV_REPR = "ev_repr"
    OBJLABELS_SEQ = "objlabels_seq"
    IS_FIRST_SAMPLE = "is_first_sample"
    IS_PADDED_MASK = "is_padded_mask"
    WORKER_ID = "worker_id"


@dataclass
class SequenceSample:
    """One training sequence: T ordered window tensors + aligned per-window labels."""

    ev_repr: List[torch.Tensor]  # each [C, H, W], uint8 or float32
    labels: List[Optional[torch.Tensor]]  # each [num_boxes, 5] (yolox) or None
    is_first_sample: bool
    is_padded_mask: List[bool]

    def __post_init__(self) -> None:
        n = len(self.ev_repr)
        if not (n == len(self.labels) == len(self.is_padded_mask)):
            raise ValueError(
                f"length mismatch: ev_repr={n}, labels={len(self.labels)}, "
                f"is_padded_mask={len(self.is_padded_mask)}"
            )
