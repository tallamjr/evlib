"""Conformance: evlib voxel grid must match the tonic event-volume oracle.

Validates ``evlib.representations.create_voxel_grid`` (long format) densified via
``densify_voxel_grid`` against ``tests/conformance/tonic_reference.tonic_voxel_grid``
(a faithful numpy port of tonic, see ``tests/test_tonic_reference.py``).
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import evlib
import evlib.representations as evr
from tests.conformance.tonic_reference import tonic_frame, tonic_voxel_grid

SLIDER = Path(__file__).resolve().parents[1] / "data" / "slider_depth" / "events.txt"


def _evlib_df_to_struct(df: pl.DataFrame):
    """Convert an evlib events DataFrame (sorted by t) to a tonic struct array.

    ``t`` is expressed in microseconds (int), ``p`` is left as the raw 0/1
    encoding so the oracle applies its own ``0 -> -1`` mapping, matching evlib.
    """
    df = df.sort("t")
    t_us = df["t"].dt.total_microseconds().to_numpy()
    dtype = np.dtype(
        [("x", np.int64), ("y", np.int64), ("t", np.float64), ("p", np.int64)]
    )
    events = np.zeros(len(df), dtype=dtype)
    events["x"] = df["x"].to_numpy()
    events["y"] = df["y"].to_numpy()
    events["t"] = t_us.astype(np.float64)
    events["p"] = df["polarity"].to_numpy()
    return events


def _synthetic_lazyframe(width=64, height=48, n=3000, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, width, n).astype(np.int16)
    y = rng.integers(0, height, n).astype(np.int16)
    t_us = np.sort(rng.integers(0, 1_000_000, n)).astype(np.int64)
    p = rng.integers(0, 2, n).astype(np.int8)
    df = pl.DataFrame(
        {
            "x": x,
            "y": y,
            "t": pl.Series(t_us, dtype=pl.Int64).cast(pl.Duration("us")),
            "polarity": p,
        }
    )
    return df.lazy()


def _assert_matches_tonic(events_lf, width, height, n_time_bins):
    events_df = events_lf.collect()
    long_df = evr.create_voxel_grid(events_lf, height, width, n_time_bins=n_time_bins)
    dense = evr.densify_voxel_grid(long_df, n_time_bins, height, width)

    struct = _evlib_df_to_struct(events_df)
    ref = tonic_voxel_grid(struct, (width, height, 2), n_time_bins)

    assert dense.shape == ref.shape == (n_time_bins, 1, height, width)
    atol = 1e-6 * max(1.0, np.abs(ref).max())
    assert np.allclose(dense, ref, atol=atol), (
        f"max abs diff {np.abs(dense - ref).max()} exceeds {atol}; "
        f"evlib sum {dense.sum()} vs tonic sum {ref.sum()}"
    )


def test_voxel_grid_matches_tonic_synthetic():
    width, height, n_time_bins = 64, 48, 5
    _assert_matches_tonic(
        _synthetic_lazyframe(width, height), width, height, n_time_bins
    )


@pytest.mark.skipif(not SLIDER.exists(), reason="slider_depth data not present")
def test_voxel_grid_matches_tonic_slider_depth():
    events_lf = evlib.load_events(str(SLIDER)).head(50_000)
    width, height, n_time_bins = 240, 180, 5
    _assert_matches_tonic(events_lf, width, height, n_time_bins)


def _assert_frame_matches_tonic(events_lf, width, height, n_time_bins):
    events_df = events_lf.collect()
    long_df = evr.create_event_frame(events_lf, height, width, n_time_bins=n_time_bins)
    dense = evr.densify_event_frame(long_df, n_time_bins, 2, height, width)

    struct = _evlib_df_to_struct(events_df)
    ref = tonic_frame(struct, (width, height, 2), n_time_bins)

    assert dense.shape == ref.shape == (n_time_bins, 2, height, width)
    # Counts are integers: require exact equality, not tolerance.
    assert np.array_equal(dense, ref), (
        f"event frame diverges from tonic; evlib sum {dense.sum()} "
        f"vs tonic sum {ref.sum()}, max abs diff {np.abs(dense - ref).max()}"
    )


def test_event_frame_matches_tonic_synthetic():
    width, height, n_time_bins = 64, 48, 5
    _assert_frame_matches_tonic(
        _synthetic_lazyframe(width, height), width, height, n_time_bins
    )


@pytest.mark.skipif(not SLIDER.exists(), reason="slider_depth data not present")
def test_event_frame_matches_tonic():
    events_lf = evlib.load_events(str(SLIDER)).head(50_000)
    width, height, n_time_bins = 240, 180, 5
    _assert_frame_matches_tonic(events_lf, width, height, n_time_bins)
