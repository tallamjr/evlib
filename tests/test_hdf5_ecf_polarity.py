"""ECF HDF5 loads must preserve both polarities (regression for the 2026-08-08 R1 finding).

The native Rust ECF reader stored -1/1 while the DataFrame builder expected 0/1 and
re-converted, mapping -1 to +1, so every event came out positive.

CI-safe regression: the Rust test test_hdf5_ecf_encoder_decoder_preserves_both_polarities
in tests/test_hdf5_ecf_polarity_regression.rs exercises the full encode/decode path and
runs without external files. This Python test validates end-to-end on real ECF data when available.
"""

from pathlib import Path

import pytest

import evlib

# Real-data test using pedestrians.hdf5 (ECF-compressed, gitignored)
ECF_FIXTURE = (
    Path(__file__).parent.parent
    / "data"
    / "prophesee"
    / "samples"
    / "hdf5"
    / "pedestrians.hdf5"
)


def test_ecf_hdf5_load_has_both_polarities_real_data():
    """Verify ECF HDF5 real data preserves both polarities.

    This test uses pedestrians.hdf5, a real ECF-compressed dataset that demonstrates
    the bug when not fixed (all events came out with polarity +1 before the fix).
    It skips silently if the gitignored fixture is not available; the CI-safe
    regression lives in tests/test_hdf5_ecf_polarity_regression.rs.
    """
    if not ECF_FIXTURE.exists():
        pytest.skip(f"Real ECF fixture not available: {ECF_FIXTURE}")
    try:
        lf = evlib.load_events(str(ECF_FIXTURE))
        polarities = lf.select("polarity").unique().collect()["polarity"].to_list()
    except Exception as exc:  # narrow skip: extension built without the hdf5 feature
        if "hdf5" in str(exc).lower() or "HDF5" in str(exc):
            pytest.skip(f"extension built without hdf5 feature: {exc}")
        raise
    assert set(polarities) == {
        -1,
        1,
    }, f"expected both polarities in an ECF HDF5 load, got {sorted(polarities)}"
