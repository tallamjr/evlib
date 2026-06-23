from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evlib.data.collate import custom_collate_random
from evlib.data.dataset_random import SequenceRandomDataset
from evlib.data.sequence import DataKey, SequenceSample
from evlib.data.sources import PreprocessedH5Source

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"


def test_len_and_padding():
    ds = SequenceRandomDataset([PreprocessedH5Source(FIX)], sequence_length=4)
    # 6 windows / 4 -> 2 sequences (second padded to 4: frames 4,5 real, 6,7 padded)
    assert len(ds) == 2
    s0 = ds[0]
    assert isinstance(s0, SequenceSample)
    assert len(s0.ev_repr) == 4 and s0.is_first_sample is True
    assert s0.is_padded_mask == [False, False, False, False]
    s1 = ds[1]
    assert s1.is_padded_mask == [False, False, True, True]
    assert (
        s1.ev_repr[2].shape == s1.ev_repr[0].shape
    )  # padded frame is a zero tensor of same shape


def test_padded_frame_is_zero():
    ds = SequenceRandomDataset([PreprocessedH5Source(FIX)], sequence_length=4)
    s1 = ds[1]
    assert torch.count_nonzero(s1.ev_repr[3]) == 0
    assert s1.labels[3] is None


def test_dataloader_two_workers_covers_all():
    ds = SequenceRandomDataset(
        [PreprocessedH5Source(FIX)], sequence_length=2
    )  # 3 sequences
    # The source holds no live h5 handle after construction (window_count reads
    # metadata then closes), so it pickles cleanly and each worker opens its own
    # handle lazily. This runs under the platform-default start method (spawn on
    # macOS), not a forced fork.
    dl = DataLoader(
        ds,
        batch_size=1,
        num_workers=2,
        collate_fn=custom_collate_random,
    )
    seen = 0
    for batch in dl:
        seen += len(batch[DataKey.IS_FIRST_SAMPLE])
    assert seen == len(ds)  # nothing dropped or duplicated
