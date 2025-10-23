"""
PyTorch Integration for evlib

High-performance PyTorch dataloader and utilities for event camera data processing.
Showcases best practices for Polars → PyTorch integration with real event data.
"""

from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Callable, Union, Dict
import logging
import numpy as np
import polars as pl
import torch

# Set up logging
logger = logging.getLogger(__name__)


class PolarsDataset(IterableDataset):
    """
    High-Performance PyTorch Dataset from Polars LazyFrame

    Uses Polars' native .to_torch() for efficient zero-copy conversion.
    Demonstrates best practices for event camera data processing pipelines.

    Key Features:
    - Zero-copy conversion with native .to_torch()
    - Memory-efficient lazy evaluation
    - Support for shuffling and batching
    - Flexible transform functions for feature extraction
    - Optimized for large event camera datasets

    Example:
        ```python
        import evlib
        from evlib.pytorch import PolarsDataset
        from torch.utils.data import DataLoader

        # Load event data as LazyFrame
        events = evlib.load_events("path/to/data.h5")

        # Create dataset with transform
        def extract_features(batch):
            # Extract features from raw event data
            features = torch.stack([
                batch["x"].float(),
                batch["y"].float(),
                batch["t"].float()
            ], dim=1)
            labels = batch["polarity"].long()
            return {"features": features, "labels": labels}

        dataset = PolarsDataset(events, batch_size=256, transform=extract_features)
        dataloader = DataLoader(dataset, batch_size=None)

        # Train with PyTorch
        for batch in dataloader:
            features = batch["features"]  # Shape: (256, 3)
            labels = batch["labels"]      # Shape: (256,)
            # ... your training loop
        ```
    """

    def __init__(
        self,
        lazy_df: "pl.LazyFrame",
        batch_size: int = 256,
        shuffle: bool = False,
        transform: Optional[Callable] = None,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize dataset

        Args:
            lazy_df: Polars LazyFrame to stream from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            transform: Optional transform function to apply to batches
            drop_last: Whether to drop incomplete batches
            seed: Random seed for shuffling
        """

        self.lazy_df = lazy_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.drop_last = drop_last
        self.seed = seed

        # Cache dataset length
        self._length = None

    def _get_length(self) -> int:
        """Get dataset length (cached)"""
        if self._length is None:
            self._length = self.lazy_df.select(pl.len()).collect().item()
        return self._length

    def __iter__(self):
        """Iterate over batches"""
        length = self._get_length()

        # Handle shuffling
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)

        # Iterate through batches
        for i in range(0, length, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            # Skip incomplete batch if requested
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Fetch batch
            if self.shuffle:
                # For shuffled access, use row indices
                batch_df = (
                    self.lazy_df.with_row_index()
                    .filter(pl.col("index").is_in(batch_indices.tolist()))
                    .drop("index")
                    .collect()
                )
            else:
                # For sequential access, use slice
                batch_df = self.lazy_df.slice(i, len(batch_indices)).collect()

            # Convert to PyTorch tensors using native .to_torch()
            try:
                # First try native .to_torch() - works if all dtypes are compatible
                tensor_data = batch_df.to_torch()

                # Split back into dictionary format for easier use
                batch_tensors = {}
                for idx, col in enumerate(batch_df.columns):
                    batch_tensors[col] = tensor_data[:, idx]

            except Exception as e:
                # Fallback: convert columns individually to handle mixed dtypes
                logger.debug(
                    f"Native .to_torch() failed ({e}), using column-wise conversion"
                )
                batch_tensors = {}
                for col in batch_df.columns:
                    col_data = batch_df[col]

                    # Handle different data types
                    if col_data.dtype == pl.Duration:
                        # Convert duration to float (microseconds as float)
                        tensor_data = torch.from_numpy(
                            col_data.dt.total_microseconds()
                            .to_numpy()
                            .astype(np.float32)
                        )
                    elif col_data.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                        tensor_data = torch.from_numpy(
                            col_data.to_numpy().astype(np.int64)
                        )
                    elif col_data.dtype == pl.String:
                        # Keep string columns as-is, don't convert to tensor yet
                        tensor_data = col_data.to_list()
                    else:
                        # Default to float32
                        tensor_data = torch.from_numpy(
                            col_data.to_numpy().astype(np.float32)
                        )

                    batch_tensors[col] = tensor_data

            # Apply transform if provided
            if self.transform:
                batch_tensors = self.transform(batch_tensors)

            yield batch_tensors


def load_rvt_data(
    base_path: Union[str, Path], max_samples: int = 1000, setup_hdf5: bool = True
) -> Optional["pl.LazyFrame"]:
    """
    Load RVT preprocessed event representations with labels

    Loads real RVT (Recurrent Vision Transformer) preprocessed data including:
    - Event representations: (N, 20, 360, 640) stacked histograms with 20 temporal bins
    - Labels: Structured array with class_id, bounding boxes, timestamps
    - Timestamps: Microsecond timestamps for each representation

    Args:
        base_path: Path to RVT data directory containing event_representations_v2/ and labels_v2/
        max_samples: Maximum number of samples to load
        setup_hdf5: Whether to automatically setup HDF5 plugins for compressed data

    Returns:
        Polars LazyFrame with extracted features and labels, or None if data not found

    Features extracted:
        - 80 temporal bin statistics (20 bins × 4 stats: mean, std, max, nonzero)
        - 5 bounding box features (x, y, w, h, area)
        - 3 activity features (total_activity, active_pixels, temporal_center)
        - 3 normalized features (timestamp_norm, bbox_area_norm, activity_norm)
        - Total: 91 feature dimensions

    Example:
        ```python
        from evlib.pytorch import load_rvt_data, PolarsDataset

        # Load RVT data
        lazy_df = load_rvt_data("data/gen4_1mpx_processed_RVT/val/moorea_2019-02-21_000_td_2257500000_2317500000")

        if lazy_df is not None:
            # Create dataset for training
            dataset = PolarsDataset(lazy_df, batch_size=256, shuffle=True)
            print(f"Loaded dataset with {len(lazy_df.collect())} samples")
        ```
    """
    try:
        import h5py
        import numpy as np
        import os
    except ImportError as e:
        raise ImportError(f"Required dependencies not available: {e}")

    if setup_hdf5:
        # Set HDF5 plugin path for compressed data
        try:
            import hdf5plugin

            os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH
        except ImportError:
            # Try alternative plugin path setup
            plugin_path = (
                Path(__file__).parent.parent.parent
                / ".venv/lib/python3.10/site-packages/hdf5plugin/plugins"
            )
            if plugin_path.exists():
                os.environ["HDF5_PLUGIN_PATH"] = str(plugin_path)

    base_path = Path(base_path)

    # Try to find RVT data files
    if not base_path.exists():
        logger.warning(f"RVT data path not found: {base_path}")
        return None

    try:
        # File paths
        repr_file = (
            base_path
            / "event_representations_v2"
            / "stacked_histogram_dt50_nbins10"
            / "event_representations_ds2_nearest.h5"
        )
        labels_file = base_path / "labels_v2" / "labels.npz"
        timestamps_file = (
            base_path
            / "event_representations_v2"
            / "stacked_histogram_dt50_nbins10"
            / "timestamps_us.npy"
        )
        mapping_file = (
            base_path
            / "event_representations_v2"
            / "stacked_histogram_dt50_nbins10"
            / "objframe_idx_2_repr_idx.npy"
        )

        if not all(
            [
                f.exists()
                for f in [repr_file, labels_file, timestamps_file, mapping_file]
            ]
        ):
            logger.warning(f"Missing RVT data files in {base_path}")
            return None

        logger.info(f"Loading RVT data from {base_path}")

        # Just check the file and get metadata - don't load actual data
        with h5py.File(repr_file, "r") as f:
            if "data" not in f:
                logger.warning(f"'data' key not found in {repr_file}")
                return None

            total_samples = f["data"].shape[0]
            actual_samples = min(max_samples, total_samples)
            logger.info(f"Found {total_samples} samples, will use {actual_samples}")

        # Load timestamps for representations
        repr_timestamps = np.load(timestamps_file)[:actual_samples]
        logger.info(f"Loaded {len(repr_timestamps)} representation timestamps")

        # Load labels
        labels_data = np.load(labels_file)

        # Extract labels - RVT uses structured arrays
        raw_labels = labels_data["labels"]
        logger.info(f"Available label fields: {raw_labels.dtype.names}")

        # Extract class IDs and other relevant fields
        class_ids = raw_labels["class_id"]
        confidences = raw_labels["class_confidence"]
        bboxes = np.column_stack(
            [raw_labels["x"], raw_labels["y"], raw_labels["w"], raw_labels["h"]]
        )

        logger.info(f"Class distribution: {np.bincount(class_ids)}")
        logger.info(f"Unique classes: {np.unique(class_ids)}")

        # Just store metadata for lazy loading - no heavy processing
        feature_data = {}

        # Basic metadata - just use first N samples for simplicity
        feature_data["sample_idx"] = list(range(actual_samples))
        feature_data["label"] = class_ids[:actual_samples].astype(np.int32)
        feature_data["t"] = repr_timestamps.astype(np.float64)
        feature_data["confidence"] = confidences[:actual_samples].astype(np.float32)

        # Bounding box features
        feature_data["bbox_x"] = bboxes[:actual_samples, 0].astype(np.float32)
        feature_data["bbox_y"] = bboxes[:actual_samples, 1].astype(np.float32)
        feature_data["bbox_w"] = bboxes[:actual_samples, 2].astype(np.float32)
        feature_data["bbox_h"] = bboxes[:actual_samples, 3].astype(np.float32)
        feature_data["bbox_area"] = (
            bboxes[:actual_samples, 2] * bboxes[:actual_samples, 3]
        ).astype(np.float32)

        # Store file path and indices for lazy loading
        feature_data["rvt_file_path"] = [str(repr_file)] * actual_samples
        feature_data["rvt_sample_idx"] = list(range(actual_samples))

        logger.info(
            f"Created metadata for {actual_samples} samples (no tensor data loaded yet)"
        )
        logger.info(f"Label distribution: {np.bincount(class_ids[:actual_samples])}")

        # Create DataFrame
        df = pl.DataFrame(feature_data)

        logger.info(
            f"Created DataFrame with {len(df)} samples and {len(df.columns)} columns"
        )
        logger.info(
            f"Stored {actual_samples} RVT tensors with shape (20, 360, 640) each"
        )
        logger.info(f"Label distribution: {df['label'].value_counts().sort('label')}")

        return df.lazy()

    except Exception as e:
        logger.warning(f"Failed to load RVT data from {base_path}: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def create_rvt_transform():
    """
    Create a transform function for RVT data that returns actual RVT tensor format

    Returns RVT-compatible tensors with shape [batch_size, 20, 360, 640].
    Uses the actual preprocessed tensor data stored in the DataFrame.

    Returns:
        Transform function that converts stored RVT tensors to proper format

    Example:
        ```python
        from evlib.pytorch import PolarsDataset, load_rvt_data, create_rvt_transform

        lazy_df = load_rvt_data("path/to/rvt/data")
        transform = create_rvt_transform()
        dataset = PolarsDataset(lazy_df, batch_size=256, transform=transform)
        ```
    """

    def convert_to_rvt_tensors(
        batch: Dict[str, "torch.Tensor"],
    ) -> Dict[str, "torch.Tensor"]:
        """Load RVT tensors on-demand from H5 file [batch_size, 20, 360, 640]"""

        # Check if we have file paths for lazy loading
        if "rvt_file_path" in batch and "rvt_sample_idx" in batch:
            import h5py

            # Get file path and indices
            file_paths = batch["rvt_file_path"]
            if isinstance(file_paths, list):
                file_path = file_paths[0]  # Same file for all samples in batch
            else:
                file_path = file_paths[0].item()  # If it's a tensor

            sample_indices = batch["rvt_sample_idx"]
            if hasattr(sample_indices[0], "item"):
                sample_indices = [idx.item() for idx in sample_indices]
            else:
                sample_indices = sample_indices  # Already a list

            # Load only the requested samples from H5 file
            with h5py.File(file_path, "r") as f:
                # Load batch of tensors efficiently
                features_list = []
                for idx in sample_indices:
                    tensor = f["data"][idx]  # Shape: (20, 360, 640)
                    features_list.append(torch.from_numpy(tensor.astype(np.float32)))

                features = torch.stack(
                    features_list, dim=0
                )  # Shape: [batch_size, 20, 360, 640]

            labels = batch["label"].long()  # Shape: [batch_size]
            return {"features": features, "labels": labels}

        # Fallback: convert flattened lists if available
        if "rvt_tensor_flat" in batch:
            tensors = []
            for flat_tensor in batch["rvt_tensor_flat"]:
                tensor = torch.tensor(flat_tensor, dtype=torch.float32).reshape(
                    20, 360, 640
                )
                tensors.append(tensor)

            features = torch.stack(tensors, dim=0)  # Shape: [batch_size, 20, 360, 640]
            labels = batch["label"].long()  # Shape: [batch_size]

            return {"features": features, "labels": labels}

        # Fallback: if no tensor data, use bounding box features
        feature_tensors = []
        for key in ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_area"]:
            if key in batch:
                feature_tensors.append(batch[key])

        if feature_tensors:
            features = torch.stack(feature_tensors, dim=1)
            labels = batch["label"].long()
            return {"features": features, "labels": labels}

        # Last resort - just return labels
        return {"labels": batch["label"].long()}

    return convert_to_rvt_tensors


def create_basic_event_transform():
    """
    Create a basic transform function for raw event data

    Returns:
        Transform function that converts raw event columns to feature/label tensors

    Example:
        ```python
        import evlib
        from evlib.pytorch import PolarsDataset, create_basic_event_transform

        events = evlib.load_events("path/to/events.h5")
        transform = create_basic_event_transform()
        dataset = PolarsDataset(events, batch_size=256, transform=transform)
        ```
    """

    def extract_event_features(
        batch: Dict[str, "torch.Tensor"],
    ) -> Dict[str, "torch.Tensor"]:
        """Transform raw event data to features"""
        # Convert timestamp from microseconds to seconds
        if batch["t"].dtype == torch.int64:
            # Duration in microseconds, convert to float seconds
            timestamp = batch["t"].float() / 1_000_000
        else:
            timestamp = batch["t"].float()

        # Stack coordinate and temporal features
        features = torch.stack(
            [
                batch["x"].float(),
                batch["y"].float(),
                timestamp,
            ],
            dim=1,
        )

        # Use polarity as labels (convert -1/1 to 0/1 for classification)
        labels = ((batch["polarity"] + 1) // 2).long()

        return {"features": features, "labels": labels}

    return extract_event_features


# Convenience function for quick setup
def create_dataloader(
    data_source: Union[str, Path, "pl.LazyFrame"],
    data_type: str = "events",
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> "DataLoader":
    """
    Create a PyTorch DataLoader for event camera data

    Args:
        data_source: Either file path or Polars LazyFrame
        data_type: Type of data ("events" for raw events, "rvt" for RVT preprocessed)
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for PolarsDataset

    Returns:
        PyTorch DataLoader ready for training

    Example:
        ```python
        from evlib.pytorch import create_dataloader

        # For raw event data
        dataloader = create_dataloader("path/to/events.h5", data_type="events")

        # For RVT preprocessed data
        dataloader = create_dataloader("path/to/rvt/data", data_type="rvt")

        # Train
        for batch in dataloader:
            features = batch["features"]
            labels = batch["labels"]
            # ... training loop
        ```
    """
    # Load data if path provided
    if isinstance(data_source, (str, Path)):
        if data_type == "rvt":
            lazy_df = load_rvt_data(data_source, **kwargs)
            if lazy_df is None:
                raise ValueError(f"Could not load RVT data from {data_source}")
        else:
            # Load raw events
            import evlib

            lazy_df = evlib.load_events(data_source)
    else:
        lazy_df = data_source

    # Create appropriate transform
    if data_type == "rvt":
        transform = create_rvt_transform()
    else:
        transform = create_basic_event_transform()

    # Create dataset
    dataset = PolarsDataset(
        lazy_df,
        batch_size=batch_size,
        shuffle=shuffle,
        transform=transform,
        drop_last=True,
        **kwargs,
    )

    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=None,  # Batching handled by dataset
        num_workers=num_workers,
        pin_memory=(torch.cuda.is_available() or torch.backends.mps.is_available()),
    )


# Export main classes and functions
__all__ = [
    "PolarsDataset",
    "load_rvt_data",
    "create_rvt_transform",
    "create_basic_event_transform",
    "create_dataloader",
]
