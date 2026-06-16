import h5py
import numpy as np
import pytest
import hdf5plugin  # noqa: F401
from pathlib import Path
from tests.rvt_fixtures import raw_input_path, ref_repr_h5, ref_timestamps
from evlib.rvt.pipeline import process_sequence


@pytest.mark.slow
def test_fast_slice_first_20_windows(tmp_path):
    grid = np.load(ref_timestamps())[:20]
    out_h5 = process_sequence(
        raw_input_path(),
        tmp_path / "out20",
        dataset="gen4",
        height=720,
        width=1280,
        ev_repr_timestamps_us=grid,
        downsample_by_2=True,
        engine="streaming",
    )
    with h5py.File(out_h5, "r") as f_ours, h5py.File(ref_repr_h5(), "r") as f_ref:
        for i in range(20):
            assert np.array_equal(f_ours["data"][i], f_ref["data"][i]), (
                f"window {i} differs"
            )


@pytest.mark.slow
def test_full_sequence_matches_rvt_reference(tmp_path):
    grid = np.load(ref_timestamps())
    out_h5 = process_sequence(
        raw_input_path(),
        tmp_path / "out",
        dataset="gen4",
        height=720,
        width=1280,
        ev_repr_timestamps_us=grid,
        downsample_by_2=True,
        engine="streaming",
    )
    with h5py.File(out_h5, "r") as f_ours, h5py.File(ref_repr_h5(), "r") as f_ref:
        ours = f_ours["data"]
        ref = f_ref["data"]
        assert ours.shape == ref.shape == (1198, 20, 360, 640)
        assert ours.dtype == ref.dtype == np.uint8
        for i in range(ref.shape[0]):
            assert np.array_equal(ours[i], ref[i]), f"window {i} differs"
