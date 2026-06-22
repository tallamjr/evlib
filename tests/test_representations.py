"""Smoke tests for event representation functions driven by real tracked data."""

import polars as pl
import pytest
import evlib
import evlib.representations as evr

SLIDER_DEPTH_PATH = "data/slider_depth/events.txt"


@pytest.fixture(scope="module")
def slider_depth_events() -> pl.LazyFrame:
    """Load the tracked slider_depth events as a LazyFrame."""
    return evlib.load_events(SLIDER_DEPTH_PATH)


def test_create_stacked_histogram(slider_depth_events: pl.LazyFrame):
    """Smoke test: create_stacked_histogram returns a DataFrame with the expected schema."""
    width, height = 346, 240

    hist_df = evr.create_stacked_histogram(
        slider_depth_events, width, height, bins=5, window_duration_ms=100
    )

    assert isinstance(hist_df, pl.DataFrame)

    expected_columns = {"time_bin", "polarity", "y", "x", "count"}
    assert expected_columns.issubset(set(hist_df.columns)), (
        f"Missing columns: {expected_columns - set(hist_df.columns)}"
    )

    assert hist_df["time_bin"].dtype == pl.Int32
    assert hist_df["y"].dtype == pl.Int16
    assert hist_df["x"].dtype == pl.Int16
    assert hist_df["count"].dtype == pl.UInt32


def test_create_mixed_density_stack(slider_depth_events: pl.LazyFrame):
    """Smoke test: create_mixed_density_stack returns a DataFrame with the expected schema."""
    width, height = 346, 240

    mixed_df = evr.create_mixed_density_stack(slider_depth_events, width, height)

    assert isinstance(mixed_df, pl.DataFrame)

    expected_columns = {"x", "y", "polarity_sum", "count"}
    assert expected_columns.issubset(set(mixed_df.columns)), (
        f"Missing columns: {expected_columns - set(mixed_df.columns)}"
    )

    assert mixed_df["y"].dtype == pl.Int16
    assert mixed_df["x"].dtype == pl.Int16
    assert mixed_df["polarity_sum"].dtype == pl.Int64
    assert mixed_df["count"].dtype == pl.UInt32
