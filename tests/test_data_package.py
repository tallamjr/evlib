"""Tests for the evlib.data package scaffold and legacy retirement."""


def test_evlib_data_imports():
    import evlib.data  # noqa: F401


def test_legacy_polars_dataset_removed():
    import evlib.pytorch as p

    assert not hasattr(p, "PolarsDataset"), "legacy PolarsDataset should be retired"
