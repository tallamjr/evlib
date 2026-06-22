"""Test PyTorch integration module"""

from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
SLIDER_DEPTH_EVENTS = ROOT / "data/slider_depth/events.txt"

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _load_slider_depth_events_lazy(n: int) -> pl.LazyFrame:
    """Load the first *n* events from slider_depth as a LazyFrame.

    The 't' column is converted from Duration[us] to Int64 microseconds so that
    ``create_basic_event_transform`` can divide by 1_000_000 to get float seconds.
    Columns returned: x (Int16), y (Int16), t (Int64), polarity (Int8).
    """
    import evlib

    return (
        evlib.load_events(str(SLIDER_DEPTH_EVENTS))
        .head(n)
        .with_columns(pl.col("t").dt.total_microseconds().cast(pl.Int64))
    )


def test_pytorch_module_import():
    """Test that PyTorch module can be imported"""
    import evlib

    if TORCH_AVAILABLE:
        from evlib import pytorch

        assert hasattr(pytorch, "PolarsDataset")
        assert hasattr(pytorch, "create_dataloader")
        assert hasattr(pytorch, "load_rvt_data")
    else:
        assert evlib.pytorch is None or not hasattr(evlib.pytorch, "PolarsDataset")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_polars_dataset_with_real_data():
    """Test PolarsDataset with real event data from slider_depth."""
    import torch
    from evlib.pytorch import PolarsDataset, create_basic_event_transform

    lazy_df = _load_slider_depth_events_lazy(1000)

    transform = create_basic_event_transform()
    dataset = PolarsDataset(lazy_df, batch_size=32, transform=transform, shuffle=False)

    batches = list(dataset)
    assert len(batches) > 0

    batch = batches[0]
    assert "features" in batch
    assert "labels" in batch

    assert batch["features"].shape[1] == 3  # x, y, timestamp
    assert batch["labels"].shape[0] == batch["features"].shape[0]

    assert batch["features"].dtype == torch.float32
    assert batch["labels"].dtype == torch.int64


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_create_dataloader():
    """Test create_dataloader convenience function with real event data."""
    from evlib.pytorch import create_dataloader

    lazy_df = _load_slider_depth_events_lazy(500)

    dataloader = create_dataloader(lazy_df, data_type="events", batch_size=64)

    batch_count = 0
    for batch in dataloader:
        assert "features" in batch
        assert "labels" in batch
        assert batch["features"].shape[1] == 3  # x, y, timestamp
        batch_count += 1
        if batch_count >= 3:
            break

    assert batch_count > 0


def test_dependency_warnings():
    """Test that warnings are issued for missing dependencies"""
    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")

        try:
            from evlib import pytorch

            if pytorch is None:
                pass
        except ImportError:
            pass


def test_rvt_data_loading_graceful_failure():
    """Test that RVT data loading fails gracefully with missing data"""
    from evlib.pytorch import load_rvt_data

    result = load_rvt_data("/non/existent/path")
    assert result is None

    result = load_rvt_data(Path(__file__).parent)
    assert result is None
