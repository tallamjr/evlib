import h5py
import numpy as np
import hdf5plugin  # noqa: F401
from evlib.rvt.pipeline import process_sequence


def test_evlib_rvt_is_reachable():
    """evlib.rvt is the advertised RVT preprocessing path and must be reachable."""
    import evlib

    assert hasattr(evlib, "rvt")
    assert hasattr(evlib.rvt, "process_sequence")
    assert hasattr(evlib.rvt, "build_sparse_histogram")


def test_process_sequence_writes_full_layout(tmp_path):
    raw = tmp_path / "seq_td.h5"
    n = 100
    t = np.linspace(0, 200000, n).astype(np.uint32)
    with h5py.File(raw, "w") as f:
        g = f.create_group("events")
        g.create_dataset("t", data=t)
        g.create_dataset("x", data=(np.arange(n) % 8).astype(np.int32))
        g.create_dataset("y", data=(np.arange(n) % 6).astype(np.int32))
        g.create_dataset("p", data=(np.arange(n) % 2).astype(np.int32))
    out = tmp_path / "out"
    grid = np.array([100000, 150000, 200000], dtype=np.int64)
    process_sequence(
        raw,
        out,
        dataset="gen4",
        height=8,
        width=6,
        ev_repr_timestamps_us=grid,
        downsample_by_2=False,
        engine="in-memory",
    )
    h5 = (
        out
        / "event_representations_v2/stacked_histogram_dt50_nbins10/event_representations.h5"
    )
    assert h5.exists()
    with h5py.File(h5, "r") as f:
        assert f["data"].shape == (3, 20, 8, 6)
    assert (
        out
        / "event_representations_v2/stacked_histogram_dt50_nbins10/timestamps_us.npy"
    ).exists()


def test_cli_runs_on_synthetic(tmp_path):
    import h5py
    import numpy as np
    from evlib.rvt import pipeline

    raw = tmp_path / "seq_td.h5"
    n = 50
    with h5py.File(raw, "w") as f:
        g = f.create_group("events")
        g.create_dataset("t", data=np.linspace(0, 100000, n).astype(np.uint32))
        g.create_dataset("x", data=(np.arange(n) % 8).astype(np.int32))
        g.create_dataset("y", data=(np.arange(n) % 6).astype(np.int32))
        g.create_dataset("p", data=(np.arange(n) % 2).astype(np.int32))
    grid = tmp_path / "grid.npy"
    np.save(grid, np.array([50000, 100000], dtype=np.int64))
    rc = pipeline.main(
        [
            "--in-h5",
            str(raw),
            "--out-dir",
            str(tmp_path / "o"),
            "--dataset",
            "gen4",
            "--height",
            "8",
            "--width",
            "6",
            "--grid-npy",
            str(grid),
            "--no-downsample",
            "--engine",
            "in-memory",
        ]
    )
    assert rc == 0
    assert (
        tmp_path
        / "o"
        / "event_representations_v2"
        / "stacked_histogram_dt50_nbins10"
        / "event_representations.h5"
    ).exists()
