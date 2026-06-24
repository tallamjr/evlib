import torch
import pytest
from evlib.data.sequence import SequenceSample, DataKey


def test_sequence_sample_lengths_must_match():
    with pytest.raises(ValueError):
        SequenceSample(
            ev_repr=[torch.zeros(2, 4, 4)],
            labels=[None, None],
            is_first_sample=True,
            is_padded_mask=[False],
        )


def test_sequence_sample_ok_and_keys():
    s = SequenceSample(
        ev_repr=[torch.zeros(2, 4, 4)],
        labels=[None],
        is_first_sample=True,
        is_padded_mask=[False],
    )
    assert len(s.ev_repr) == 1
    assert DataKey.EV_REPR == "ev_repr"
    assert DataKey.IS_FIRST_SAMPLE == "is_first_sample"
