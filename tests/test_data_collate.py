import torch
import pytest
from evlib.data.sequence import SequenceSample, DataKey
from evlib.data.collate import custom_collate_random


def _mk(is_first):
    return SequenceSample(
        ev_repr=[torch.ones(2, 4, 4), torch.ones(2, 4, 4)],
        labels=[None, torch.zeros(1, 5)],
        is_first_sample=is_first,
        is_padded_mask=[False, False],
    )


def test_collate_shapes():
    batch = custom_collate_random([_mk(True), _mk(True)])
    ev = batch[DataKey.EV_REPR]
    assert len(ev) == 2  # T
    assert ev[0].shape == (2, 2, 4, 4)  # [B, C, H, W]
    assert batch[DataKey.IS_FIRST_SAMPLE] == [True, True]
    assert batch[DataKey.IS_PADDED_MASK].shape == (2, 2)  # [T, B]
    assert (
        len(batch[DataKey.OBJLABELS_SEQ]) == 2
        and len(batch[DataKey.OBJLABELS_SEQ][0]) == 2
    )


def test_collate_rejects_ragged_T():
    a = _mk(True)
    b = SequenceSample([torch.ones(2, 4, 4)], [None], True, [False])
    with pytest.raises(ValueError):
        custom_collate_random([a, b])
