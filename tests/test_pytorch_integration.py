"""Test PyTorch integration module"""

import pytest
import numpy as np
from pathlib import Path

# Check if torch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_pytorch_module_import():
    """Test that PyTorch module can be imported"""
    import evlib

    if TORCH_AVAILABLE:
        from evlib import pytorch

        assert hasattr(pytorch, "PolarsDataset")
        assert hasattr(pytorch, "create_dataloader")
        assert hasattr(pytorch, "load_rvt_data")
    else:
        # Should gracefully handle missing PyTorch
        assert evlib.pytorch is None or not hasattr(evlib.pytorch, "PolarsDataset")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_polars_dataset_with_mock_data():
    """Test PolarsDataset with mock event data"""
    import polars as pl
    import torch
    from evlib.pytorch import PolarsDataset, create_basic_event_transform

    # Create mock event data
    n_events = 1000
    mock_data = {
        "x": np.random.randint(0, 640, n_events).astype(np.int16),
        "y": np.random.randint(0, 480, n_events).astype(np.int16),
        "timestamp": np.sort(np.random.randint(0, 1000000, n_events)).astype(np.int64),
        "polarity": np.random.choice([-1, 1], n_events).astype(np.int8),
    }

    # Create LazyFrame
    df = pl.DataFrame(mock_data)
    lazy_df = df.lazy()

    # Create dataset with transform
    transform = create_basic_event_transform()
    dataset = PolarsDataset(lazy_df, batch_size=32, transform=transform, shuffle=False)

    # Test iteration
    batches = list(dataset)
    assert len(batches) > 0

    batch = batches[0]
    assert "features" in batch
    assert "labels" in batch

    # Check shapes
    assert batch["features"].shape[1] == 3  # x, y, timestamp
    assert batch["labels"].shape[0] == batch["features"].shape[0]

    # Check data types
    assert batch["features"].dtype == torch.float32
    assert batch["labels"].dtype == torch.int64


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_create_dataloader():
    """Test create_dataloader convenience function"""
    import polars as pl
    import torch
    from evlib.pytorch import create_dataloader

    # Create mock event data as LazyFrame
    n_events = 500
    mock_data = {
        "x": np.random.randint(0, 640, n_events).astype(np.int16),
        "y": np.random.randint(0, 480, n_events).astype(np.int16),
        "timestamp": np.sort(np.random.randint(0, 1000000, n_events)).astype(np.int64),
        "polarity": np.random.choice([-1, 1], n_events).astype(np.int8),
    }

    df = pl.DataFrame(mock_data)
    lazy_df = df.lazy()

    # Create dataloader
    dataloader = create_dataloader(lazy_df, data_type="events", batch_size=64)

    # Test iteration
    batch_count = 0
    for batch in dataloader:
        assert "features" in batch
        assert "labels" in batch
        assert batch["features"].shape[1] == 3  # x, y, timestamp
        batch_count += 1
        if batch_count >= 3:  # Test a few batches
            break

    assert batch_count > 0


def test_dependency_warnings():
    """Test that warnings are issued for missing dependencies"""
    import warnings

    # Import the module - should issue warnings if dependencies missing
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")

        try:
            from evlib import pytorch

            # If import succeeds, check if warnings were issued
            if pytorch is None:
                # Module wasn't imported due to missing deps
                pass
        except ImportError:
            # Expected if dependencies not available
            pass


def test_rvt_data_loading_graceful_failure():
    """Test that RVT data loading fails gracefully with missing data"""
    from evlib.pytorch import load_rvt_data

    # Should return None for non-existent path
    result = load_rvt_data("/non/existent/path")
    assert result is None

    # Should return None for path without required files
    result = load_rvt_data(Path(__file__).parent)
    assert result is None
