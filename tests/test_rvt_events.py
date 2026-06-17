import pytest

pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

import h5py
import numpy as np
import polars as pl
import hdf5plugin  # noqa: F401
from tests.rvt_fixtures import raw_input_path, requires_reference_data
from evlib.rvt.events import convert_h5_to_parquet, correct_time_nondecreasing


def test_correct_time_is_running_max():
    t = np.array([5, 3, 7, 6, 10], dtype=np.int64)
    out = correct_time_nondecreasing(t)
    assert out.tolist() == [5, 5, 7, 7, 10]


@requires_reference_data
def test_convert_roundtrip_first_rows_match_raw(tmp_path):
    pq = tmp_path / "events.parquet"
    convert_h5_to_parquet(
        raw_input_path(), pq, chunk_rows=1_000_000, max_rows=2_000_000
    )
    df = pl.read_parquet(pq)
    assert df.columns == ["t", "x", "y", "p"]
    assert df.height == 2_000_000
    with h5py.File(raw_input_path(), "r") as f:
        raw_x = f["events"]["x"][:2_000_000]
        raw_t = f["events"]["t"][:2_000_000].astype(np.int64)
    assert df["x"].to_numpy().tolist()[:1000] == raw_x[:1000].tolist()
    tt = df["t"].to_numpy()
    assert np.all(tt[:-1] <= tt[1:])
    assert tt[0] == raw_t[0]
