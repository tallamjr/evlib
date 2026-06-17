from pathlib import Path
import evlib
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVT2 = ROOT / "data/prophesee/samples/evt2/80_balls.raw"

requires_evt2 = pytest.mark.skipif(not EVT2.exists(), reason="EVT2 sample absent")


@requires_evt2
def test_roi_filter_matches_manual_polars():
    full = evlib.load_events(str(EVT2)).collect()
    sub = evlib.load_events(
        str(EVT2), min_x=100, max_x=300, min_y=50, max_y=200
    ).collect()
    ref = full.filter(
        pl.col("x").is_between(100, 300) & pl.col("y").is_between(50, 200)
    ).sort("t")
    assert sub.sort(["t", "x", "y"]).equals(ref.sort(["t", "x", "y"]))


@requires_evt2
def test_polarity_filter_matches_manual_polars():
    full = evlib.load_events(str(EVT2)).collect()
    pos = evlib.load_events(str(EVT2), polarity=1).collect()
    ref = full.filter(pl.col("polarity") == 1).sort("t")
    assert pos.sort(["t", "x", "y"]).equals(ref.sort(["t", "x", "y"]))


@requires_evt2
def test_time_window_filter_selects_subset():
    full = evlib.load_events(str(EVT2)).collect()
    t_us = full["t"].dt.total_microseconds()
    lo_s = float(t_us.min()) / 1e6
    hi_s = lo_s + (float(t_us.max()) - float(t_us.min())) / 1e6 / 2  # first half
    sub = evlib.load_events(str(EVT2), t_start=lo_s, t_end=hi_s).collect()
    assert 0 < sub.height < full.height
    assert sub["t"].dt.total_microseconds().max() <= int(hi_s * 1e6)
