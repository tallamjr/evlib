import pickle
from pathlib import Path

import pytest
import torch

pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from evlib.data.sources import PreprocessedH5Source, _nbins_from_repr_name

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"


def test_nbins_parses_both_repr_name_forms():
    # evlib-native on-disk form (no '=')
    assert _nbins_from_repr_name("stacked_histogram_dt50_nbins10") == 10
    # upstream-RVT on-disk form (with '='), used by RVT_REPR_DIR_NAME and real
    # gen4/eTram data
    assert _nbins_from_repr_name("stacked_histogram_dt=50_nbins=10") == 10


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


def test_no_persistent_handle_after_construction():
    src = PreprocessedH5Source(FIX)
    assert src._h5 is None and src._data is None
    src.window_count()
    # window_count reads metadata then closes the file: no live handle remains.
    assert src._h5 is None and src._data is None


def test_picklable_after_window_count():
    src = PreprocessedH5Source(FIX)
    assert src.window_count() == 6
    restored = pickle.loads(pickle.dumps(src))
    # The unpickled source holds no live handle and still reads correctly.
    assert restored._h5 is None and restored._data is None
    assert restored.window_count() == 6
    ev, labels = restored.read_windows(0, 6)
    assert len(ev) == 6 and len(labels) == 6


def test_data_handle_opened_only_on_read():
    src = PreprocessedH5Source(FIX)
    src.window_count()
    assert src._data is None
    src.read_windows(0, 2)
    assert src._h5 is not None and src._data is not None
