"""End-to-end test: N-Caltech101 .bin -> convert_ncaltech -> SampleDataset -> DataLoader.

Builds a tiny synthetic 2-class tree of genuine ATIS .bin files, runs them
through the real converter, and verifies the DataLoader yields the expected
tensor shapes and labels.  No .npy arrays are fabricated directly.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from evlib.data import SampleDataset, convert_ncaltech, read_atis_bin  # noqa: E402
from evlib.data import NCALTECH_HEIGHT, NCALTECH_WIDTH  # noqa: E402

NBINS = 10
CHANNELS = 2 * NBINS


# ---------------------------------------------------------------------------
# ATIS .bin encoder (same helper as tests/test_data_ncaltech.py)
# ---------------------------------------------------------------------------


def _encode_atis_event(x: int, y: int, polarity: int, t_us: int) -> bytes:
    """Encode one event as the 5 ATIS bytes (inverse of ``read_atis_bin``)."""
    b0 = x & 0xFF
    b1 = y & 0xFF
    b2 = ((polarity & 0x1) << 7) | ((t_us >> 16) & 0x7F)
    b3 = (t_us >> 8) & 0xFF
    b4 = t_us & 0xFF
    return bytes([b0, b1, b2, b3, b4])


def _write_atis_bin(path, events):
    """Write a list of ``(x, y, polarity, t_us)`` tuples to an ATIS ``.bin``."""
    raw = b"".join(_encode_atis_event(*ev) for ev in events)
    path.write_bytes(raw)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_two_class_tree(root):
    """Create a 2-class tree under *root* with 3 recordings total.

    Sorted class order: airplane (label 0) < camera (label 1).
    airplane has 2 recordings, camera has 1.
    """
    airplane = root / "airplane"
    camera = root / "camera"
    airplane.mkdir(parents=True)
    camera.mkdir(parents=True)

    _write_atis_bin(
        airplane / "image_0001.bin",
        [(10, 20, 1, 1000), (30, 40, 0, 5000)],
    )
    _write_atis_bin(
        airplane / "image_0002.bin",
        [(11, 21, 0, 2000), (31, 41, 1, 6000)],
    )
    _write_atis_bin(
        camera / "image_0001.bin",
        [(12, 22, 1, 1500), (32, 42, 0, 5500)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ncaltech_loader_covers_all_samples(tmp_path):
    """DataLoader iterates over all 3 samples exactly once."""
    root = tmp_path / "ncaltech_raw"
    _build_two_class_tree(root)
    out_dir = tmp_path / "ncaltech_out"

    sample_paths, labels = convert_ncaltech(root, out_dir, nbins=NBINS)
    assert len(sample_paths) == 3
    assert labels == [0, 0, 1]

    ds = SampleDataset(sample_paths, labels)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    seen = 0
    for x_batch, y_batch in dl:
        seen += x_batch.shape[0]

    assert seen == 3


def test_ncaltech_loader_batch_tensor_shape(tmp_path):
    """Each batch tensor has shape [B, 2*nbins, NCALTECH_HEIGHT, NCALTECH_WIDTH]."""
    root = tmp_path / "ncaltech_raw"
    _build_two_class_tree(root)
    out_dir = tmp_path / "ncaltech_out"

    sample_paths, labels = convert_ncaltech(root, out_dir, nbins=NBINS)

    ds = SampleDataset(sample_paths, labels)
    dl = DataLoader(ds, batch_size=3, shuffle=False)

    batches = list(dl)
    assert len(batches) == 1

    x_batch, y_batch = batches[0]
    assert x_batch.shape == (3, CHANNELS, NCALTECH_HEIGHT, NCALTECH_WIDTH)


def test_ncaltech_loader_labels_match_converter(tmp_path):
    """Labels returned by the DataLoader match those from convert_ncaltech."""
    root = tmp_path / "ncaltech_raw"
    _build_two_class_tree(root)
    out_dir = tmp_path / "ncaltech_out"

    sample_paths, labels = convert_ncaltech(root, out_dir, nbins=NBINS)

    ds = SampleDataset(sample_paths, labels)
    dl = DataLoader(ds, batch_size=3, shuffle=False)

    x_batch, y_batch = next(iter(dl))
    returned_labels = y_batch.tolist()
    assert returned_labels == labels


def test_ncaltech_sample_nonzero(tmp_path):
    """Each sample tensor contains at least one non-zero value."""
    root = tmp_path / "ncaltech_raw"
    _build_two_class_tree(root)
    out_dir = tmp_path / "ncaltech_out"

    sample_paths, labels = convert_ncaltech(root, out_dir, nbins=NBINS)

    ds = SampleDataset(sample_paths, labels)

    for i in range(len(ds)):
        x, y = ds[i]
        assert x.sum().item() > 0, f"sample {i} is all zeros"


def test_ncaltech_public_exports():
    """The full ncaltech public surface is importable from evlib.data."""
    from evlib.data import (  # noqa: F401
        NCALTECH_HEIGHT,
        NCALTECH_WIDTH,
        convert_ncaltech,
        read_atis_bin,
        representation_from_events,
    )

    assert NCALTECH_HEIGHT == 180
    assert NCALTECH_WIDTH == 240
