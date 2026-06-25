from pathlib import Path

import pytest

pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

import numpy as np

from evlib.data.sources import PreprocessedH5Source
from evlib.data.dataset_stream import SequenceStreamDataset
from evlib.data.augment import SequenceAugmentor
from evlib.data.collate import custom_collate_stream
from evlib.data.sequence import DataKey

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"
SHORT_FIX = Path(__file__).resolve().parent / "data_fixtures" / "short_seq"


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
    # The stream path must also carry the per-worker id (0 in-process).
    assert DataKey.WORKER_ID in batch
    assert batch[DataKey.WORKER_ID] == 0


def test_stream_slot0_exhausts_first_keeps_width_and_alignment():
    # slot 0 stream is SHORTER than slot 1. With L=2: short_seq (2 windows) -> 1
    # chunk in slot 0; mini_seq (6 windows) -> 3 chunks in slot 1. After slot 0
    # exhausts it must stay positionally slot 0, padded, while slot 1 advances.
    ds = SequenceStreamDataset(
        [PreprocessedH5Source(SHORT_FIX), PreprocessedH5Source(FIX)],
        sequence_length=2,
        batch_size=2,
    )
    steps = list(iter(ds))
    # slot 1 (mini_seq) dictates the step count: 3 chunks.
    assert len(steps) == 3
    # batch width is constant at exactly batch_size on every step.
    assert all(len(step) == 2 for step in steps)

    # Step 0: both slots carry a real (non-padded) first chunk.
    assert steps[0][0].is_first_sample is True
    assert steps[0][1].is_first_sample is True
    assert all(p is False for p in steps[0][0].is_padded_mask)
    assert all(p is False for p in steps[0][1].is_padded_mask)

    # Steps 1 and 2: slot 0 stream is exhausted, so position 0 must be a fully
    # padded sample (is_first_sample False, all is_padded_mask True), while
    # position 1 keeps advancing the longer stream.
    for later in steps[1:]:
        pad_sample = later[0]
        assert pad_sample.is_first_sample is False
        assert all(p is True for p in pad_sample.is_padded_mask)
        assert all(lbl is None for lbl in pad_sample.labels)
        # position 1 still carries the longer stream's continuing chunk.
        real_sample = later[1]
        assert real_sample.is_first_sample is False
        assert any(p is False for p in real_sample.is_padded_mask)


def test_stream_rejects_random_sampler_augmentor_eagerly():
    # A sampler='random' augmentor can draw label-aware zoom-in, which cannot be
    # frozen once per source. Construction must fail EAGERLY (no iteration), not
    # lazily when for_source() first draws zoom-in mid-iteration.
    random_aug = SequenceAugmentor(sampler="random", rng=np.random.default_rng(0))
    assert random_aug.stream_safe() is False
    with pytest.raises(ValueError):
        SequenceStreamDataset(
            [PreprocessedH5Source(FIX)],
            sequence_length=2,
            batch_size=1,
            augmentor=random_aug,
        )


def test_stream_accepts_stream_sampler_augmentor():
    # A sampler='stream' augmentor disables zoom-in (zoom_in_weight=0) and so is
    # stream-safe: construction succeeds and iteration yields the chunks.
    stream_aug = SequenceAugmentor(sampler="stream", rng=np.random.default_rng(0))
    assert stream_aug.stream_safe() is True
    ds = SequenceStreamDataset(
        [PreprocessedH5Source(FIX)],
        sequence_length=2,
        batch_size=1,
        augmentor=stream_aug,
    )
    chunks = list(iter(ds))
    assert len(chunks) == 3  # mini_seq 6 windows / L=2


def test_stream_batch_size_exceeds_sources_pads_trailing_slot():
    # batch_size=2 with a single source: the trailing empty slot 1 must be
    # padded from the very first step so width stays exactly 2 throughout.
    ds = SequenceStreamDataset(
        [PreprocessedH5Source(FIX)], sequence_length=2, batch_size=2
    )
    steps = list(iter(ds))
    assert len(steps) == 3  # mini_seq 6 windows / L=2
    assert all(len(step) == 2 for step in steps)
    for step in steps:
        # slot 0 carries the real stream.
        assert any(p is False for p in step[0].is_padded_mask)
        # slot 1 is empty and always fully padded.
        assert all(p is True for p in step[1].is_padded_mask)
        assert step[1].is_first_sample is False
        assert all(lbl is None for lbl in step[1].labels)
