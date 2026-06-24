import importlib.util
from pathlib import Path

import pytest

import evlib.data as d

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"


def test_public_exports():
    for name in [
        "SequenceRandomDataset",
        "SequenceStreamDataset",
        "SampleDataset",
        "PreprocessedH5Source",
        "EvlibStreamSource",
        "custom_collate_random",
        "custom_collate_stream",
        "DataKey",
        "SequenceSample",
    ]:
        assert hasattr(d, name), name


@pytest.mark.skipif(
    importlib.util.find_spec("pytorch_lightning") is None,
    reason="pytorch_lightning not installed",
)
def test_datamodule_available_when_lightning_present():
    assert hasattr(d, "EventDataModule")


@pytest.mark.skipif(
    importlib.util.find_spec("pytorch_lightning") is None,
    reason="pytorch_lightning not installed",
)
def test_datamodule_random_train_dataloader_yields_batches():
    pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    source = d.PreprocessedH5Source(FIX)
    dm = d.EventDataModule(
        train_sources=[source],
        val_sources=[d.PreprocessedH5Source(FIX)],
        sequence_length=4,
        batch_size=1,
        num_workers=0,
        sampling="random",
    )
    dl = dm.train_dataloader()
    seen = 0
    for batch in dl:
        seen += len(batch[d.DataKey.IS_FIRST_SAMPLE])
    assert seen == len(d.SequenceRandomDataset([d.PreprocessedH5Source(FIX)], 4))
