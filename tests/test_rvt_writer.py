import h5py
import numpy as np
import polars as pl
import hdf5plugin  # noqa: F401
from evlib.rvt.writer import scatter_window_dense, write_event_representation_h5


def test_scatter_places_counts():
    sparse = pl.DataFrame(
        {
            "window_id": [0, 0],
            "channel": [0, 19],
            "y": [1, 2],
            "x": [3, 0],
            "count": [4, 10],
        }
    ).with_columns(pl.col("count").cast(pl.UInt32))
    dense = scatter_window_dense(
        sparse.filter(pl.col("window_id") == 0), channels=20, height=4, width=4
    )
    assert dense.dtype == np.uint8
    assert dense.shape == (20, 4, 4)
    assert dense[0, 1, 3] == 4
    assert dense[19, 2, 0] == 10
    assert dense.sum() == 14


def test_write_h5_layout(tmp_path):
    sparse = pl.DataFrame(
        {
            "window_id": [0, 1],
            "channel": [0, 5],
            "y": [0, 1],
            "x": [0, 1],
            "count": [3, 7],
        }
    ).with_columns(pl.col("count").cast(pl.UInt32))
    out = tmp_path / "ev.h5"
    write_event_representation_h5(
        out, sparse, num_windows=2, channels=20, height=4, width=4
    )
    with h5py.File(out, "r") as f:
        d = f["data"]
        assert d.shape == (2, 20, 4, 4)
        assert d.dtype == np.uint8
        assert d.chunks == (1, 20, 4, 4)
        assert list(d._filters) == ["32001"]
        assert d[0, 0, 0, 0] == 3
        assert d[1, 5, 1, 1] == 7
