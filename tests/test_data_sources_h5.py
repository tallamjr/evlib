from pathlib import Path

import pytest
import torch

from evlib.data.sources import PreprocessedH5Source

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"


def test_window_count_and_shape():
    src = PreprocessedH5Source(FIX)
    assert src.window_count() == 6
    ev, labels = src.read_windows(0, 6)
    assert len(ev) == 6 and len(labels) == 6
    assert ev[0].shape == (20, 8, 12) and ev[0].dtype == torch.uint8


def test_label_alignment():
    src = PreprocessedH5Source(FIX)
    _, labels = src.read_windows(0, 6)
    # boxes only on repr indices 1 (1 box) and 4 (2 boxes); rest None
    assert labels[0] is None and labels[2] is None and labels[5] is None
    assert labels[1].shape == (1, 5)
    assert labels[4].shape == (2, 5)


def test_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        PreprocessedH5Source(FIX.parent / "does_not_exist").window_count()
