"""CI-tier tests for the SequenceAugmentor seam in the datasets + DataModule.

These verify the wiring (the opt-in ``augmentor`` arg is threaded through and
called with the right semantics), not the augmentation math itself, which is
covered exhaustively in ``test_data_augment.py``. The random dataset draws fresh
params PER ``__getitem__`` (RVT random semantics), while the stream dataset draws
params ONCE per source and reuses them across every chunk of that source (RVT
stream semantics).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List

import pytest

pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from evlib.data.dataset_random import SequenceRandomDataset
from evlib.data.dataset_stream import SequenceStreamDataset
from evlib.data.sequence import SequenceSample
from evlib.data.sources import PreprocessedH5Source

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"


class _TaggingAugmentor:
    """A recording stub that counts calls and tags the sample it returns.

    It returns a NEW ``SequenceSample`` so the dataset must use the augmentor's
    output (not the pre-augmentation sample) for the test to pass. ``draws``
    counts how many independent param draws happened, ``applies`` counts how many
    samples were transformed.
    """

    def __init__(self) -> None:
        self.applies = 0
        self.returned: List[SequenceSample] = []

    def __call__(self, sample: SequenceSample) -> SequenceSample:
        self.applies += 1
        out = SequenceSample(
            ev_repr=[t.clone() for t in sample.ev_repr],
            labels=list(sample.labels),
            is_first_sample=sample.is_first_sample,
            is_padded_mask=list(sample.is_padded_mask),
        )
        self.returned.append(out)
        return out


def test_random_dataset_applies_augmentor_once_per_getitem():
    aug = _TaggingAugmentor()
    ds = SequenceRandomDataset(
        [PreprocessedH5Source(FIX)], sequence_length=4, augmentor=aug
    )
    assert aug.applies == 0  # construction does not augment
    s0 = ds[0]
    assert aug.applies == 1  # one __getitem__ -> exactly one augmentor call
    # The returned sample is the augmentor's output, not the raw sample.
    assert s0 is aug.returned[-1]
    s1 = ds[1]
    assert aug.applies == 2  # a second __getitem__ -> a second call
    assert s1 is aug.returned[-1]


def test_random_dataset_without_augmentor_is_untouched():
    ds = SequenceRandomDataset([PreprocessedH5Source(FIX)], sequence_length=4)
    s0 = ds[0]
    assert isinstance(s0, SequenceSample)  # no augmentor: plain assembled sample


class _CountingSourceAugmentor:
    """Records once-per-source draws vs per-chunk applications for the stream path.

    ``for_source(sample)`` is the per-source entry point: it draws a frozen
    parameter state once and returns a per-chunk callable. Each invocation of
    that callable counts as one application but reuses the single frozen draw.
    """

    def __init__(self) -> None:
        self.draws = 0
        self.applies = 0

    def for_source(self, sample: SequenceSample):
        self.draws += 1

        def apply(chunk: SequenceSample) -> SequenceSample:
            self.applies += 1
            return chunk

        return apply


def test_stream_dataset_draws_params_once_per_source():
    aug = _CountingSourceAugmentor()
    # mini_seq has 6 windows; L=2 -> 3 chunks from the single source.
    ds = SequenceStreamDataset(
        [PreprocessedH5Source(FIX)],
        sequence_length=2,
        batch_size=1,
        augmentor=aug,
    )
    steps = list(iter(ds))
    assert len(steps) == 3
    # ONE param draw for the single source, but applied to every real chunk.
    assert aug.draws == 1
    assert aug.applies == 3


def test_stream_dataset_two_sources_draw_once_each():
    aug = _CountingSourceAugmentor()
    ds = SequenceStreamDataset(
        [PreprocessedH5Source(FIX), PreprocessedH5Source(FIX)],
        sequence_length=2,
        batch_size=1,  # both sources land in slot 0 and stream back to back
        augmentor=aug,
    )
    list(iter(ds))
    # One draw per source (two sources), applied to all 3 chunks of each.
    assert aug.draws == 2
    assert aug.applies == 6


@pytest.mark.skipif(
    importlib.util.find_spec("pytorch_lightning") is None,
    reason="pytorch_lightning not installed",
)
def test_datamodule_augments_train_only():
    import evlib.data as d

    aug = _TaggingAugmentor()
    dm = d.EventDataModule(
        train_sources=[PreprocessedH5Source(FIX)],
        val_sources=[PreprocessedH5Source(FIX)],
        sequence_length=4,
        batch_size=1,
        num_workers=0,
        sampling="random",
        augmentor=aug,
    )
    train_ds = dm.train_dataloader().dataset
    assert train_ds.augmentor is aug  # train split carries the augmentor
    val_ds = dm.val_dataloader().dataset
    assert val_ds.augmentor is None  # val split must NOT augment
    test_ds = dm.test_dataloader().dataset
    assert test_ds.augmentor is None  # test split must NOT augment


@pytest.mark.skipif(
    importlib.util.find_spec("pytorch_lightning") is None,
    reason="pytorch_lightning not installed",
)
def test_datamodule_stream_augments_train_only():
    import evlib.data as d

    aug = _CountingSourceAugmentor()
    dm = d.EventDataModule(
        train_sources=[PreprocessedH5Source(FIX)],
        val_sources=[PreprocessedH5Source(FIX)],
        sequence_length=4,
        batch_size=1,
        num_workers=0,
        sampling="stream",
        augmentor=aug,
    )
    train_ds = dm.train_dataloader().dataset
    assert train_ds.augmentor is aug
    # val/test stay random and unaugmented.
    assert dm.val_dataloader().dataset.augmentor is None
