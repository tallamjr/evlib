"""ECF HDF5 loads must preserve both polarities (regression for the 2026-08-08 R1 finding).

The native Rust ECF reader stored -1/1 while the DataFrame builder expected 0/1 and
re-converted, mapping -1 to +1, so every event came out positive.
"""

from pathlib import Path

import pytest

import evlib

# Use pedestrians.hdf5 which is ECF-compressed and demonstrates the bug;
# val_night_011_td.h5 is standard HDF5, not ECF-compressed
ECF_FIXTURE = (
    Path(__file__).parent.parent
    / "data"
    / "prophesee"
    / "samples"
    / "hdf5"
    / "pedestrians.hdf5"
)


def test_ecf_hdf5_load_has_both_polarities():
    if not ECF_FIXTURE.exists():
        pytest.skip(f"ECF test fixture not available: {ECF_FIXTURE}")
    try:
        lf = evlib.load_events(str(ECF_FIXTURE))
        polarities = lf.select("polarity").unique().collect()["polarity"].to_list()
    except Exception as exc:  # narrow skip: extension built without the hdf5 feature
        if "hdf5" in str(exc).lower() or "HDF5" in str(exc):
            pytest.skip(f"extension built without hdf5 feature: {exc}")
        raise
    assert set(polarities) == {-1, 1}, (
        f"expected both polarities in an ECF HDF5 load, got {sorted(polarities)}"
    )
