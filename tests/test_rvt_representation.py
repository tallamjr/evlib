import numpy as np
import polars as pl
from evlib.rvt.representation import build_sparse_histogram


def test_single_window_time_bins_and_channels():
    t = np.array([0, 10, 95, 100], dtype=np.int64)
    x = np.array([1, 1, 2, 2], dtype=np.int16)
    y = np.array([0, 0, 0, 0], dtype=np.int16)
    p = np.array([0, 0, 1, 1], dtype=np.int8)
    events = pl.DataFrame({"t": t, "x": x, "y": y, "p": p})
    sparse = build_sparse_histogram(
        events,
        ev_repr_timestamps_us=np.array([100], dtype=np.int64),
        delta_t_us=1000,
        nbins=10,
        count_cutoff=10,
        height=4,
        width=4,
        downsample_by_2=False,
        engine="in-memory",
    )
    rows = {
        (r["window_id"], r["channel"], r["y"], r["x"]): r["count"]
        for r in sparse.to_dicts()
    }
    assert rows[(0, 0, 0, 1)] == 1  # t=0,  pol0, bin0
    assert rows[(0, 1, 0, 1)] == 1  # t=10, pol0, bin1
    assert (
        rows[(0, 19, 0, 2)] == 2
    )  # t=95 -> bin9, t=100 -> clamp 9; pol1 -> channel 19


def test_count_cutoff_clips():
    t = np.zeros(25, dtype=np.int64)
    x = np.ones(25, dtype=np.int16)
    y = np.zeros(25, dtype=np.int16)
    p = np.zeros(25, dtype=np.int8)
    events = pl.DataFrame({"t": t, "x": x, "y": y, "p": p})
    sparse = build_sparse_histogram(
        events,
        ev_repr_timestamps_us=np.array([0], dtype=np.int64),
        delta_t_us=1000,
        nbins=10,
        count_cutoff=10,
        height=4,
        width=4,
        downsample_by_2=False,
        engine="in-memory",
    )
    assert sparse["count"].max() == 10


def test_gpu_engine_falls_back_on_cpu_only_host():
    import numpy as np
    import polars as pl
    from evlib.rvt.representation import build_sparse_histogram

    events = pl.DataFrame(
        {
            "t": np.array([0, 50], dtype=np.int64),
            "x": np.array([0, 1], dtype=np.int16),
            "y": np.array([0, 0], dtype=np.int16),
            "p": np.array([0, 1], dtype=np.int8),
        }
    )
    ref = build_sparse_histogram(
        events,
        np.array([50], dtype=np.int64),
        1000,
        10,
        10,
        4,
        4,
        False,
        engine="in-memory",
    )
    out = build_sparse_histogram(
        events, np.array([50], dtype=np.int64), 1000, 10, 10, 4, 4, False, engine="gpu"
    )
    # GPU must transparently fall back on a CPU-only host and produce identical output
    assert out.sort(["window_id", "channel", "y", "x"]).equals(
        ref.sort(["window_id", "channel", "y", "x"])
    )
