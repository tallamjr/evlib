"""Sanity tests for evlib's build profile.

Guards against regressions where feature-gated functionality is
accidentally always-on or always-off, and verifies that the no-HDF5
build surfaces a clear error when someone tries to load a .h5 file.
"""

import sys

import pytest

import evlib
from tests.hdf5_support import HAS_HDF5


def test_has_hdf5_is_bool():
    """HAS_HDF5 must always resolve to a plain boolean."""
    assert isinstance(HAS_HDF5, bool)


def test_has_hdf5_matches_runtime_probe():
    """HAS_HDF5 must agree with a direct runtime probe of the module.

    The cfg-gated Rust HDF5 support lives on the `formats` submodule; the top-level
    evlib.save_events_to_hdf5 is an always-present pure-Python h5py wrapper and is not a
    reliable signal, so the probe targets the Rust submodule.
    """
    fmt = getattr(evlib, "formats", None)
    expected = (
        sys.platform != "win32"
        and fmt is not None
        and hasattr(fmt, "save_events_to_hdf5")
    )
    assert HAS_HDF5 is expected


def test_hdf5_absence_produces_clear_error_when_off():
    """When HDF5 is not compiled in, loading a .h5 path must raise an
    error whose message mentions HDF5 so users know what went wrong."""
    if HAS_HDF5:
        pytest.skip("HDF5 is compiled in; this test guards the no-HDF5 path")

    with pytest.raises(Exception) as excinfo:
        evlib.load_events("nonexistent_file.h5")

    message = str(excinfo.value).lower()
    assert "hdf5" in message, f"expected HDF5 mention in error, got: {excinfo.value!r}"
