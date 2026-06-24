from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from evlib.data.dataset_sample import SampleDataset

ROOT = Path(__file__).resolve().parent / "data_fixtures" / "mini_samples"


def _build():
    labels = np.load(ROOT / "labels.npy").tolist()
    paths = [ROOT / f"sample_{i}.npy" for i in range(len(labels))]
    return SampleDataset(paths, labels)


def test_item_shape_and_label():
    ds = _build()
    x, y = ds[0]
    assert x.shape == (20, 8, 12)
    assert isinstance(y, int) and y == 0


def test_dataloader_shuffles_and_covers():
    ds = _build()
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    seen = 0
    for xb, yb in dl:
        assert xb.shape[1:] == (20, 8, 12)
        seen += xb.shape[0]
    assert seen == len(ds)
