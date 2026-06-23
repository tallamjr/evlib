"""Collate functions turning SequenceSample lists into model-ready batch dicts."""

from __future__ import annotations

from typing import List

import torch

from evlib.data.sequence import DataKey, SequenceSample


def custom_collate_random(samples: List[SequenceSample]) -> dict:
    if not samples:
        raise ValueError("empty batch")
    T = len(samples[0].ev_repr)
    if any(len(s.ev_repr) != T for s in samples):
        raise ValueError("all samples in a batch must share the same sequence length T")

    ev_repr = [torch.stack([s.ev_repr[t] for s in samples], dim=0) for t in range(T)]
    labels = [[s.labels[t] for s in samples] for t in range(T)]
    is_first = [s.is_first_sample for s in samples]
    padded = torch.tensor(
        [[s.is_padded_mask[t] for s in samples] for t in range(T)], dtype=torch.bool
    )
    return {
        DataKey.EV_REPR: ev_repr,
        DataKey.OBJLABELS_SEQ: labels,
        DataKey.IS_FIRST_SAMPLE: is_first,
        DataKey.IS_PADDED_MASK: padded,
    }
