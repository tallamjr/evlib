import torch
import pytest
from evlib.data.sequence import SequenceSample, DataKey
from evlib.data.collate import custom_collate_random


def _mk(is_first, is_padded_mask=None):
    return SequenceSample(
        ev_repr=[torch.ones(2, 4, 4), torch.ones(2, 4, 4)],
        labels=[None, torch.zeros(1, 5)],
        is_first_sample=is_first,
        is_padded_mask=is_padded_mask if is_padded_mask is not None else [False, False],
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


def test_collate_padded_mask_is_t_by_b_orientation():
    # Asymmetric T=2, B=3 so a transposed mask would have shape (3, 2) and fail.
    # Mark a single known padded cell at sequence step t=1 of the middle sample
    # (b=1) and assert it lands at [t=1][b=1] in the [T, B] mask.
    samples = [
        _mk(True, is_padded_mask=[False, False]),
        _mk(True, is_padded_mask=[False, True]),
        _mk(True, is_padded_mask=[False, False]),
    ]
    batch = custom_collate_random(samples)
    mask = batch[DataKey.IS_PADDED_MASK]
    assert mask.shape == (2, 3)  # [T, B]
    assert bool(mask[1][1]) is True
    # Every other cell stays False, proving the orientation is not transposed.
    assert mask.sum().item() == 1


def test_collate_attaches_worker_id():
    # Collate runs in-process in tests (no DataLoader worker), so worker id is 0.
    batch = custom_collate_random([_mk(True), _mk(True)])
    assert DataKey.WORKER_ID in batch
    assert batch[DataKey.WORKER_ID] == 0


def test_collate_rejects_ragged_T():
    a = _mk(True)
    b = SequenceSample([torch.ones(2, 4, 4)], [None], True, [False])
    with pytest.raises(ValueError):
        custom_collate_random([a, b])
