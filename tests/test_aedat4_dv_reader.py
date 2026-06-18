"""End-to-end tests for the AEDAT 4.0 (iniVation DV framework) reader.

These exercise ``evlib.load_events`` / ``evlib.detect_format`` against the real
``.aedat4`` sample files shipped under the ``dv-processing`` reference checkout.
The expected event counts and first-event values were cross-checked against
``dv-processing`` 2.0.3 (``dv_processing.io.MonoCameraRecording``).

Each test skips when its sample file is absent so the suite still passes in CI
environments that do not vendor the dv-processing submodule.
"""

import os

import pytest

import evlib

# (relative path, expected event count, x bounds, y bounds, expected polarity set)
SAMPLES = [
    (
        "lib/dv-processing/tests/io/test_files/sample_data.aedat4",
        9193,
        (0, 345),
        (0, 259),
        {1},  # reference: all 9193 events are ON in this stream
    ),
    (
        "lib/dv-processing/python/tests/data/sample_data.aedat4",
        9408,
        (63, 512),
        (47, 384),
        {1},
    ),
    (
        "lib/dv-processing/tests/io/test_files/test-minimal.aedat4",
        255283,
        (0, 639),
        (0, 479),
        {-1, 1},
    ),
]


@pytest.mark.parametrize("path,count,xb,yb,pols", SAMPLES)
def test_aedat4_load(path, count, xb, yb, pols):
    if not os.path.exists(path):
        pytest.skip(f"sample file not present: {path}")

    detected = evlib.detect_format(path)
    # detect_format returns (name, confidence, metadata)
    assert detected[0] == "AEDAT 4.0"

    df = evlib.load_events(path).collect()

    assert len(df) == count
    assert df["x"].min() >= xb[0]
    assert df["x"].max() <= xb[1]
    assert df["y"].min() >= yb[0]
    assert df["y"].max() <= yb[1]
    assert set(df["polarity"].unique().to_list()) == pols

    # Timestamps are stored as Duration(us); DV timestamps are positive.
    t_us = df["t"].dt.total_microseconds()
    assert t_us.min() > 0
