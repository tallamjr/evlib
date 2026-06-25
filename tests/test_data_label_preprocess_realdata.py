"""Byte-identical acceptance gate for RVT label preprocessing (task B3).

Runs ``preprocess_sequence`` on a real raw ``*_bbox.npy`` and asserts the four
written artifacts are array-equal (values AND dtype) to the real RVT-produced
ground truth staged on disk. A mismatch is a real transcription bug in the B1
filter chain or the B2 object-frame/grid alignment, to be fixed in
``python/evlib/data/label_preprocess.py`` (never by relaxing this gate).

Local-only ``slow`` gate. Skips cleanly when the staged data is absent (CI and
machines without the gitignored sequence). Override the sequence directory via
the ``EVLIB_GEN4_BBOX_SEQ_DIR`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from evlib.data.label_preprocess import preprocess_sequence

SEQ_NAME = "moorea_2019-06-17_test_02_000_3172500000_3232500000"
REPR_DIR_NAME = "stacked_histogram_dt=50_nbins=10"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SEQ_DIR = _PROJECT_ROOT / "data" / "gen4_label_preprocess" / SEQ_NAME


def _seq_dir() -> Path:
    override = os.environ.get("EVLIB_GEN4_BBOX_SEQ_DIR")
    return Path(override) if override else _DEFAULT_SEQ_DIR


def _assert_array_byte_identical(
    name: str, produced: np.ndarray, expected: np.ndarray
) -> None:
    """Assert two arrays match exactly in dtype, shape and every value.

    On mismatch the message names the first differing index and the two values
    so a failure points straight at the responsible filter/alignment step.
    """
    assert produced.dtype == expected.dtype, (
        f"{name}: dtype mismatch produced {produced.dtype} != expected {expected.dtype}"
    )
    assert produced.shape == expected.shape, (
        f"{name}: shape mismatch produced {produced.shape} != expected {expected.shape}"
    )
    if np.array_equal(produced, expected):
        return

    if expected.dtype.names is not None:
        # Structured array: report the first differing (row, field).
        for row in range(expected.shape[0]):
            for field in expected.dtype.names:
                if not np.array_equal(produced[field][row], expected[field][row]):
                    raise AssertionError(
                        f"{name}: first diff at row {row} field {field!r}: "
                        f"produced {produced[field][row]!r} != expected "
                        f"{expected[field][row]!r}"
                    )
        raise AssertionError(f"{name}: arrays differ but no element diff located")

    diff = np.flatnonzero(produced != expected)
    first = int(diff[0])
    raise AssertionError(
        f"{name}: first diff at index {first}: produced {produced[first]!r} "
        f"!= expected {expected[first]!r} ({diff.size} differing element(s))"
    )


@pytest.mark.slow
def test_label_preprocess_byte_identical_to_rvt(tmp_path):
    """preprocess_sequence reproduces RVT's five ground-truth arrays exactly."""
    seq_dir = _seq_dir()
    bbox_path = seq_dir / f"{SEQ_NAME}_bbox.npy"
    if not seq_dir.is_dir() or not bbox_path.is_file():
        pytest.skip(
            f"staged gen4 bbox sequence not found at {seq_dir} "
            "(set EVLIB_GEN4_BBOX_SEQ_DIR to override)"
        )

    preprocess_sequence(
        bbox_path,
        out_dir=tmp_path,
        dataset="gen4",
        split="val",
        height=720,
        width=1280,
        repr_dir_name=REPR_DIR_NAME,
    )

    produced_labels_npz = np.load(tmp_path / "labels_v2" / "labels.npz")
    expected_labels_npz = np.load(seq_dir / "labels_v2" / "labels.npz")

    # 1. labels structured array (every field incl track_id, class_confidence).
    _assert_array_byte_identical(
        "labels",
        produced_labels_npz["labels"],
        expected_labels_npz["labels"],
    )

    # 2. objframe_idx_2_label_idx (int64).
    _assert_array_byte_identical(
        "objframe_idx_2_label_idx",
        produced_labels_npz["objframe_idx_2_label_idx"],
        expected_labels_npz["objframe_idx_2_label_idx"],
    )

    # 3. labels_v2/timestamps_us.npy (599 frame timestamps, int64).
    _assert_array_byte_identical(
        "labels_v2/timestamps_us",
        np.load(tmp_path / "labels_v2" / "timestamps_us.npy"),
        np.load(seq_dir / "labels_v2" / "timestamps_us.npy"),
    )

    # 4. objframe_idx_2_repr_idx.npy (599, int64).
    _assert_array_byte_identical(
        "objframe_idx_2_repr_idx",
        np.load(
            tmp_path
            / "event_representations_v2"
            / REPR_DIR_NAME
            / "objframe_idx_2_repr_idx.npy"
        ),
        np.load(
            seq_dir
            / "event_representations_v2"
            / REPR_DIR_NAME
            / "objframe_idx_2_repr_idx.npy"
        ),
    )

    # 5. event_representations_v2/<repr>/timestamps_us.npy (1198-window grid, int64).
    _assert_array_byte_identical(
        "event_representations/timestamps_us",
        np.load(
            tmp_path / "event_representations_v2" / REPR_DIR_NAME / "timestamps_us.npy"
        ),
        np.load(
            seq_dir / "event_representations_v2" / REPR_DIR_NAME / "timestamps_us.npy"
        ),
    )
