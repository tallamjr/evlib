from pathlib import Path

from evlib.data.sources import PreprocessedH5Source
from evlib.data.dataset_stream import SequenceStreamDataset
from evlib.data.collate import custom_collate_stream
from evlib.data.sequence import DataKey

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"


def test_stream_first_sample_only_at_start():
    # one source of 6 windows, L=2 -> 3 chunks; batch_size=1 single stream
    ds = SequenceStreamDataset(
        [PreprocessedH5Source(FIX)], sequence_length=2, batch_size=1
    )
    chunks = list(iter(ds))
    firsts = [slot[0].is_first_sample for slot in chunks]
    assert firsts[0] is True
    assert all(f is False for f in firsts[1:])  # state carries after the first chunk


def test_stream_collate_shape():
    ds = SequenceStreamDataset(
        [PreprocessedH5Source(FIX)], sequence_length=2, batch_size=1
    )
    slot = next(iter(ds))
    batch = custom_collate_stream(slot)
    assert len(batch[DataKey.EV_REPR]) == 2  # T
    assert batch[DataKey.EV_REPR][0].shape[0] == 1  # B == batch_size
