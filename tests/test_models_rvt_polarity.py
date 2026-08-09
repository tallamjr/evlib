"""RVT histogram preprocessing must accept evlib's -1/1 polarity (2026-08-08 P1 finding)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evlib.models.rvt import RVT


@pytest.fixture(scope="module")
def model():
    return RVT(pretrained=False, variant="tiny")


def _events(polarity_encoding):
    rng = np.random.default_rng(42)
    n = 200
    xs = rng.integers(0, 64, n).astype(np.int64)
    ys = rng.integers(0, 48, n).astype(np.int64)
    ts = np.sort(rng.uniform(0.0, 0.05, n))
    ps = rng.integers(0, 2, n).astype(np.int64)  # 0/1
    if polarity_encoding == "-1/1":
        ps = np.where(ps == 0, -1, 1)
    return xs, ys, ts, ps


def test_negative_polarity_events_reach_negative_channels(model):
    hist, _, _ = model.preprocess_events_to_histogram(_events("-1/1"), 48, 64)
    nbins = model.temporal_bins
    assert hist[:nbins].sum() > 0, "OFF events were dropped from the negative channels"
    assert hist[nbins:].sum() > 0


def test_minus_one_one_and_zero_one_encodings_agree(model):
    hist_signed, _, _ = model.preprocess_events_to_histogram(_events("-1/1"), 48, 64)
    hist_01, _, _ = model.preprocess_events_to_histogram(_events("0/1"), 48, 64)
    assert torch.equal(hist_signed, hist_01)


def test_real_evt2_events_fill_both_polarity_halves(model):
    # Real data check: 80_balls.raw is tracked, loads via the EVT2 reader with
    # -1/1 polarity, which is exactly the encoding the bug dropped.
    from pathlib import Path

    import evlib

    raw = (
        Path(__file__).parent.parent
        / "data"
        / "prophesee"
        / "samples"
        / "evt2"
        / "80_balls.raw"
    )
    if not raw.exists():
        pytest.fail(f"required tracked fixture missing: {raw}")
    df = evlib.load_events(str(raw)).head(100_000).collect()
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    ts = df["t"].dt.total_microseconds().to_numpy().astype(np.float64) / 1e6
    ps = df["polarity"].to_numpy()
    assert set(np.unique(ps)) == {-1, 1}, "fixture should carry -1/1 polarity"
    hist, _, _ = model.preprocess_events_to_histogram((xs, ys, ts, ps), 720, 1280)
    nbins = model.temporal_bins
    assert hist[:nbins].sum() > 0
    assert hist[nbins:].sum() > 0
