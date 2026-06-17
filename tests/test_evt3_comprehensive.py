"""EVT3 reader verification against the real Prophesee EVT3 sample.

Uses the real ``data/prophesee/samples/evt3/pedestrians.raw`` (a genuine Prophesee Gen4
recording, ~177M, gitignored and therefore absent in CI). The tests skip cleanly when the
sample is not present, mirroring the other real-data tests in this repo. There is no synthetic
event fabrication: the EVT3 reader is verified against real camera data only.
"""

from pathlib import Path

import polars as pl
import pytest

evlib = pytest.importorskip("evlib")

ROOT = Path(__file__).resolve().parents[1]
EVT3_SAMPLE = ROOT / "data/prophesee/samples/evt3/pedestrians.raw"
SENSOR_WIDTH, SENSOR_HEIGHT = 1280, 720  # Prophesee Gen4

requires_evt3_sample = pytest.mark.skipif(
    not EVT3_SAMPLE.exists(),
    reason="real EVT3 sample not present (gitignored; absent in CI)",
)


@pytest.fixture(scope="module")
def events():
    """Load the real EVT3 sample once for the module."""
    return evlib.load_events(str(EVT3_SAMPLE)).collect()


@requires_evt3_sample
def test_evt3_format_detection():
    fmt, confidence, metadata = evlib.detect_format(str(EVT3_SAMPLE))
    assert fmt == "EVT3"
    assert confidence > 0.9
    assert "detection_method" in metadata


@requires_evt3_sample
def test_evt3_loads_events(events):
    # A real Gen4 recording has millions of events.
    assert len(events) > 1_000_000


@requires_evt3_sample
def test_evt3_schema(events):
    assert events.schema["x"] == pl.Int16
    assert events.schema["y"] == pl.Int16
    assert events.schema["polarity"] == pl.Int8
    assert isinstance(events.schema["t"], pl.Duration)


@requires_evt3_sample
def test_evt3_coordinates_within_sensor(events):
    assert events["x"].min() >= 0
    assert events["x"].max() < SENSOR_WIDTH
    assert events["y"].min() >= 0
    assert events["y"].max() < SENSOR_HEIGHT


@requires_evt3_sample
def test_evt3_polarity_values(events):
    assert set(events["polarity"].unique().to_list()) <= {-1, 1}
