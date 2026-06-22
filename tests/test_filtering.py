"""
Filtering tests using real event camera data.

Covers every public function in evlib.filtering:
  filter_by_time, filter_by_roi, filter_multiple_rois, filter_by_polarity,
  filter_hot_pixels, filter_noise

Primary dataset : data/prophesee/samples/evt2/80_balls.raw  (~4.6 M events, -1/1 polarity)
Secondary dataset: data/slider_depth/events.txt              (~1 M events, 0/1 polarity)

No mock data, no pandera schemas, no HDF5 gates.
All filtering functions return a Polars LazyFrame; .collect() is called before assertions.
"""

from pathlib import Path

import polars as pl
import pytest

import evlib
import evlib.filtering as flt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BALLS_RAW = Path("data/prophesee/samples/evt2/80_balls.raw")
SLIDER_TXT = Path("data/slider_depth/events.txt")

# ---------------------------------------------------------------------------
# Session-scoped fixtures (load once, reuse across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def balls_df() -> pl.DataFrame:
    """4.6 M events from 80_balls EVT2 file, polarity in {-1, 1}."""
    assert BALLS_RAW.exists(), f"Required test file missing: {BALLS_RAW}"
    return evlib.load_events(str(BALLS_RAW)).collect()


@pytest.fixture(scope="session")
def slider_df() -> pl.DataFrame:
    """1 M events from slider_depth text file, polarity in {0, 1}."""
    assert SLIDER_TXT.exists(), f"Required test file missing: {SLIDER_TXT}"
    return evlib.load_events(str(SLIDER_TXT)).collect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_DTYPES = {
    "x": pl.Int16,
    "y": pl.Int16,
    "t": pl.Duration(time_unit="us"),
    "polarity": pl.Int8,
}


def assert_schema(df: pl.DataFrame, label: str = "") -> None:
    """Assert that the DataFrame has the canonical evlib event schema."""
    for col, expected_dtype in EXPECTED_DTYPES.items():
        assert col in df.columns, f"{label}: missing column '{col}'"
        assert df[col].dtype == expected_dtype, (
            f"{label}: column '{col}' has dtype {df[col].dtype}, expected {expected_dtype}"
        )


def t_seconds(df: pl.DataFrame) -> pl.Series:
    """Return timestamps as a float64 Series in seconds."""
    return (df["t"].dt.total_microseconds() / 1_000_000).cast(pl.Float64)


# ---------------------------------------------------------------------------
# Schema smoke-test
# ---------------------------------------------------------------------------


def test_balls_schema(balls_df):
    """Loaded 80_balls data must have the canonical event schema."""
    assert_schema(balls_df, "80_balls")
    assert len(balls_df) > 0


def test_slider_schema(slider_df):
    """Loaded slider_depth data must have the canonical event schema."""
    assert_schema(slider_df, "slider_depth")
    assert len(slider_df) > 0


# ---------------------------------------------------------------------------
# filter_by_time
# ---------------------------------------------------------------------------


def test_filter_by_time_reduces_count(balls_df):
    """Filtering to a sub-window must return fewer events than the full dataset."""
    result = flt.filter_by_time(balls_df, t_start=24.0, t_end=27.0).collect()
    assert len(result) < len(balls_df)
    assert len(result) > 0


def test_filter_by_time_bounds(balls_df):
    """Every event in the filtered result must lie within the requested window."""
    t_start, t_end = 25.0, 28.0
    result = flt.filter_by_time(balls_df, t_start=t_start, t_end=t_end).collect()
    ts = t_seconds(result)
    assert ts.min() >= t_start, "Event before t_start found"
    assert ts.max() <= t_end, "Event after t_end found"


def test_filter_by_time_start_only(balls_df):
    """t_start-only filter must keep events with t >= t_start."""
    t_start = 26.0
    result = flt.filter_by_time(balls_df, t_start=t_start).collect()
    ts = t_seconds(result)
    assert ts.min() >= t_start
    assert len(result) < len(balls_df)


def test_filter_by_time_returns_lazy(balls_df):
    """filter_by_time must return a LazyFrame before .collect()."""
    lazy = flt.filter_by_time(balls_df, t_start=25.0, t_end=28.0)
    assert isinstance(lazy, pl.LazyFrame)


# ---------------------------------------------------------------------------
# filter_by_roi
# ---------------------------------------------------------------------------


def test_filter_by_roi_reduces_count(balls_df):
    """ROI filter covering the centre of the sensor must reduce event count."""
    result = flt.filter_by_roi(
        balls_df, x_min=100, x_max=400, y_min=50, y_max=300
    ).collect()
    assert len(result) < len(balls_df)
    assert len(result) > 0


def test_filter_by_roi_spatial_bounds(balls_df):
    """All events in the ROI result must lie within the requested box."""
    x_min, x_max, y_min, y_max = 100, 400, 50, 300
    result = flt.filter_by_roi(
        balls_df, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    ).collect()
    assert result["x"].min() >= x_min
    assert result["x"].max() <= x_max
    assert result["y"].min() >= y_min
    assert result["y"].max() <= y_max


def test_filter_by_roi_schema_preserved(balls_df):
    """ROI filtering must not alter the event schema."""
    result = flt.filter_by_roi(
        balls_df, x_min=100, x_max=400, y_min=50, y_max=300
    ).collect()
    assert_schema(result, "filter_by_roi")


# ---------------------------------------------------------------------------
# filter_multiple_rois
# ---------------------------------------------------------------------------


def test_filter_multiple_rois_reduces_count(balls_df):
    """Multi-ROI filter on two small boxes must return fewer events than the full set."""
    rois = [(50, 150, 20, 100), (400, 500, 300, 400)]
    result = flt.filter_multiple_rois(balls_df, rois=rois).collect()
    assert len(result) < len(balls_df)
    assert len(result) > 0


def test_filter_multiple_rois_spatial_bounds(balls_df):
    """Every event in the multi-ROI result must fall inside at least one of the boxes."""
    roi_a = (50, 150, 20, 100)
    roi_b = (400, 500, 300, 400)
    rois = [roi_a, roi_b]
    result = flt.filter_multiple_rois(balls_df, rois=rois).collect()
    x, y = result["x"], result["y"]
    in_a = (x >= roi_a[0]) & (x <= roi_a[1]) & (y >= roi_a[2]) & (y <= roi_a[3])
    in_b = (x >= roi_b[0]) & (x <= roi_b[1]) & (y >= roi_b[2]) & (y <= roi_b[3])
    assert (in_a | in_b).all(), "Found events outside all requested ROIs"


def test_filter_multiple_rois_schema_preserved(balls_df):
    """Multi-ROI filtering must not alter the event schema."""
    rois = [(50, 150, 20, 100), (400, 500, 300, 400)]
    result = flt.filter_multiple_rois(balls_df, rois=rois).collect()
    assert_schema(result, "filter_multiple_rois")


# ---------------------------------------------------------------------------
# filter_by_polarity
# ---------------------------------------------------------------------------


def test_filter_by_polarity_positive_only(balls_df):
    """Filtering polarity=1 must return only positive events."""
    result = flt.filter_by_polarity(balls_df, polarity=1).collect()
    assert len(result) > 0
    assert result["polarity"].unique().to_list() == [1]


def test_filter_by_polarity_negative_only(balls_df):
    """Filtering polarity=-1 must return only negative events."""
    result = flt.filter_by_polarity(balls_df, polarity=-1).collect()
    assert len(result) > 0
    assert result["polarity"].unique().to_list() == [-1]


def test_filter_by_polarity_count_adds_up(balls_df):
    """Positive + negative counts must equal total event count."""
    pos = len(flt.filter_by_polarity(balls_df, polarity=1).collect())
    neg = len(flt.filter_by_polarity(balls_df, polarity=-1).collect())
    assert pos + neg == len(balls_df)


def test_filter_by_polarity_zero_one_encoding(slider_df):
    """Polarity filter works correctly on 0/1-encoded data (slider_depth)."""
    result_zero = flt.filter_by_polarity(slider_df, polarity=0).collect()
    result_one = flt.filter_by_polarity(slider_df, polarity=1).collect()
    assert result_zero["polarity"].unique().to_list() == [0]
    assert result_one["polarity"].unique().to_list() == [1]
    assert len(result_zero) + len(result_one) == len(slider_df)


def test_filter_by_polarity_returns_lazy(balls_df):
    """filter_by_polarity must return a LazyFrame."""
    lazy = flt.filter_by_polarity(balls_df, polarity=1)
    assert isinstance(lazy, pl.LazyFrame)


# ---------------------------------------------------------------------------
# filter_hot_pixels
# ---------------------------------------------------------------------------


def test_filter_hot_pixels_count_not_increased(balls_df):
    """Hot-pixel filter must not produce more events than the input."""
    result = flt.filter_hot_pixels(balls_df, threshold_percentile=99.9).collect()
    assert len(result) <= len(balls_df)


def test_filter_hot_pixels_schema_preserved(balls_df):
    """Hot-pixel filtering must not alter the event schema."""
    result = flt.filter_hot_pixels(balls_df, threshold_percentile=99.9).collect()
    assert_schema(result, "filter_hot_pixels")


def test_filter_hot_pixels_aggressive_removes_more(balls_df):
    """Aggressive threshold (95th pct) must remove at least as many events as conservative (99.9th)."""
    conservative = len(
        flt.filter_hot_pixels(balls_df, threshold_percentile=99.9).collect()
    )
    aggressive = len(
        flt.filter_hot_pixels(balls_df, threshold_percentile=95.0).collect()
    )
    assert aggressive <= conservative


def test_filter_hot_pixels_returns_lazy(balls_df):
    """filter_hot_pixels must return a LazyFrame."""
    lazy = flt.filter_hot_pixels(balls_df, threshold_percentile=99.9)
    assert isinstance(lazy, pl.LazyFrame)


# ---------------------------------------------------------------------------
# filter_noise
# ---------------------------------------------------------------------------


def test_filter_noise_count_not_increased(balls_df):
    """Noise filter must not produce more events than the input."""
    result = flt.filter_noise(
        balls_df, method="refractory", refractory_period_us=1000.0
    ).collect()
    assert len(result) <= len(balls_df)


def test_filter_noise_schema_preserved(balls_df):
    """Noise filtering must not alter the event schema."""
    result = flt.filter_noise(
        balls_df, method="refractory", refractory_period_us=1000.0
    ).collect()
    assert_schema(result, "filter_noise")


def test_filter_noise_aggressive_removes_more(balls_df):
    """A longer refractory period must remove at least as many events as a shorter one."""
    moderate = len(
        flt.filter_noise(
            balls_df, method="refractory", refractory_period_us=1000.0
        ).collect()
    )
    aggressive = len(
        flt.filter_noise(
            balls_df, method="refractory", refractory_period_us=10000.0
        ).collect()
    )
    assert aggressive <= moderate


def test_filter_noise_returns_lazy(balls_df):
    """filter_noise must return a LazyFrame."""
    lazy = flt.filter_noise(balls_df, method="refractory", refractory_period_us=1000.0)
    assert isinstance(lazy, pl.LazyFrame)


# ---------------------------------------------------------------------------
# End-to-end chain
# ---------------------------------------------------------------------------


def test_end_to_end_chain(balls_df):
    """
    Chain: filter_by_time -> filter_by_roi -> filter_by_polarity -> collect.

    Asserts:
    - final count < original count
    - all timestamps within the requested window
    - all coordinates within the requested box
    - only the requested polarity is present
    """
    t_start, t_end = 24.5, 28.5
    x_min, x_max, y_min, y_max = 100, 500, 50, 400

    result = (
        flt.filter_by_time(balls_df, t_start=t_start, t_end=t_end)
        .pipe(
            lambda lf: flt.filter_by_roi(
                lf, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )
        .pipe(lambda lf: flt.filter_by_polarity(lf, polarity=1))
        .collect()
    )

    assert len(result) > 0, "Chain produced empty result"
    assert len(result) < len(balls_df), "Chain must reduce total event count"

    ts = t_seconds(result)
    assert ts.min() >= t_start
    assert ts.max() <= t_end

    assert result["x"].min() >= x_min
    assert result["x"].max() <= x_max
    assert result["y"].min() >= y_min
    assert result["y"].max() <= y_max

    assert result["polarity"].unique().to_list() == [1]


def test_end_to_end_chain_schema(balls_df):
    """End-to-end chain must preserve the canonical event schema."""
    result = (
        flt.filter_by_time(balls_df, t_start=24.5, t_end=28.5)
        .pipe(
            lambda lf: flt.filter_by_roi(lf, x_min=100, x_max=500, y_min=50, y_max=400)
        )
        .pipe(lambda lf: flt.filter_by_polarity(lf, polarity=1))
        .collect()
    )
    assert_schema(result, "end_to_end_chain")
