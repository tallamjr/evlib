"""
HDF5 support detection helpers for the evlib test suite.

Kept in a dedicated module (rather than ``tests/conftest.py``) so that test
files can import ``HAS_HDF5`` / ``requires_hdf5`` via the package-qualified path
``from tests.hdf5_support import ...``. A bare ``from conftest import ...`` is
fragile because ``conftest`` can resolve to the wrong conftest when bare
``pytest`` collects multiple directories.
"""

import sys

import pytest

# Detect whether the evlib build includes HDF5 support.
# Mirrors the hdf5_available fixture but available at import time so
# test modules can apply `pytestmark = requires_hdf5`.
try:
    import evlib as _evlib_for_hdf5_probe

    # HDF5 read/write is a cfg-gated Rust feature (--features hdf5). The top-level
    # evlib.save_events_to_hdf5 is a pure-Python h5py wrapper that always exists, so it
    # is not a reliable signal. Probe the Rust `formats` submodule, where
    # save_events_to_hdf5 is only registered when the hdf5 feature is compiled in.
    _fmt = getattr(_evlib_for_hdf5_probe, "formats", None)
    HAS_HDF5 = (
        sys.platform != "win32"
        and _fmt is not None
        and hasattr(_fmt, "save_events_to_hdf5")
    )
    del _evlib_for_hdf5_probe, _fmt
except ImportError:
    HAS_HDF5 = False

requires_hdf5 = pytest.mark.skipif(
    not HAS_HDF5,
    reason="HDF5 feature not compiled in (build with --features hdf5)",
)
