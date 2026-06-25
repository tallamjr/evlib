"""Tests for the evlib.data package scaffold and legacy retirement."""


def test_evlib_data_imports():
    import evlib.data  # noqa: F401


def test_legacy_polars_dataset_removed():
    import importlib.util

    assert importlib.util.find_spec("evlib.pytorch") is None, (
        "legacy evlib.pytorch module should be fully removed"
    )

    import evlib

    assert not hasattr(evlib, "pytorch"), "evlib should no longer expose pytorch"
