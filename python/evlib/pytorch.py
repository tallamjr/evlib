"""
PyTorch Integration for evlib

High-performance PyTorch dataloader and utilities for event camera data processing.
Showcases best practices for Polars → PyTorch integration with real event data.
"""

from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Callable, Union, Dict, Any
import logging
import numpy as np
import polars as pl
import time
import torch
import warnings

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
                batch["timestamp"].float()
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
                logger.debug(f"Native .to_torch() failed ({e}), using column-wise conversion")
                batch_tensors = {}
                for col in batch_df.columns:
                    col_data = batch_df[col]

                    # Handle different data types
                    if col_data.dtype == pl.Duration:
                        # Convert duration to float (microseconds as float)
                        tensor_data = torch.from_numpy(
                            col_data.dt.total_microseconds().to_numpy().astype(np.float32)
                        )
                    elif col_data.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                        tensor_data = torch.from_numpy(col_data.to_numpy().astype(np.int64))
                    else:
                        # Default to float32
                        tensor_data = torch.from_numpy(col_data.to_numpy().astype(np.float32))

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
            from pathlib import Path

            plugin_path = (
                Path(__file__).parent.parent.parent / ".venv/lib/python3.10/site-packages/hdf5plugin/plugins"
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
            base_path / "event_representations_v2" / "stacked_histogram_dt50_nbins10" / "timestamps_us.npy"
        )
        mapping_file = (
            base_path
            / "event_representations_v2"
            / "stacked_histogram_dt50_nbins10"
            / "objframe_idx_2_repr_idx.npy"
        )

        if not all([f.exists() for f in [repr_file, labels_file, timestamps_file, mapping_file]]):
            logger.warning(f"Missing RVT data files in {base_path}")
            return None

        logger.info(f"Loading RVT data from {base_path}")

        # Load event representations (1198, 20, 360, 640)
        with h5py.File(repr_file, "r") as f:
            if "data" not in f:
                logger.warning(f"'data' key not found in {repr_file}")
                return None

            total_samples = f["data"].shape[0]
            actual_samples = min(max_samples, total_samples)

            # Load representations
            representations = f["data"][:actual_samples]  # Shape: (N, 20, 360, 640)
            logger.info(f"Loaded representations shape: {representations.shape}")
            logger.info(f"Data range: [{representations.min()}, {representations.max()}]")

        # Load timestamps for representations
        repr_timestamps = np.load(timestamps_file)[:actual_samples]
        logger.info(f"Loaded {len(repr_timestamps)} representation timestamps")

        # Load labels and mapping
        labels_data = np.load(labels_file)
        mapping = np.load(mapping_file)

        # Extract labels - RVT uses structured arrays
        raw_labels = labels_data["labels"]
        logger.info(f"Available label fields: {raw_labels.dtype.names}")

        # Extract class IDs and other relevant fields
        class_ids = raw_labels["class_id"]
        confidences = raw_labels["class_confidence"]
        bboxes = np.column_stack([raw_labels["x"], raw_labels["y"], raw_labels["w"], raw_labels["h"]])

        logger.info(f"Class distribution: {np.bincount(class_ids)}")
        logger.info(f"Unique classes: {np.unique(class_ids)}")

        # Create training samples by matching representations to labels via mapping
        training_samples = []
        training_labels = []
        training_timestamps = []
        training_confidences = []
        training_bboxes = []

        # Use mapping to match representations with labels
        for i in range(min(actual_samples, len(mapping))):
            repr_idx = mapping[i] if i < len(mapping) else i
            if repr_idx < len(representations):
                # Find corresponding labels for this time frame
                label_start_idx = (
                    labels_data["objframe_idx_2_label_idx"][i]
                    if i < len(labels_data["objframe_idx_2_label_idx"])
                    else 0
                )

                if label_start_idx < len(class_ids):
                    training_samples.append(representations[repr_idx])
                    training_labels.append(class_ids[label_start_idx])
                    training_timestamps.append(repr_timestamps[repr_idx])
                    training_confidences.append(confidences[label_start_idx])
                    training_bboxes.append(bboxes[label_start_idx])

        training_samples = np.array(training_samples)
        training_labels = np.array(training_labels, dtype=np.int32)
        training_timestamps = np.array(training_timestamps)
        training_confidences = np.array(training_confidences)
        training_bboxes = np.array(training_bboxes)

        logger.info(f"Created {len(training_samples)} training samples")
        logger.info(f"Representation shape per sample: {training_samples[0].shape}")
        logger.info(f"Label distribution: {np.bincount(training_labels)}")

        # Extract statistical features from stacked histograms (more manageable than 4.6M features)
        n_samples = len(training_samples)
        feature_data = {}

        # Basic metadata
        feature_data["sample_idx"] = np.arange(n_samples)
        feature_data["label"] = training_labels
        feature_data["timestamp"] = training_timestamps.astype(np.float64)
        feature_data["confidence"] = training_confidences.astype(np.float32)

        # Bounding box features
        feature_data["bbox_x"] = training_bboxes[:, 0].astype(np.float32)
        feature_data["bbox_y"] = training_bboxes[:, 1].astype(np.float32)
        feature_data["bbox_w"] = training_bboxes[:, 2].astype(np.float32)
        feature_data["bbox_h"] = training_bboxes[:, 3].astype(np.float32)
        feature_data["bbox_area"] = (training_bboxes[:, 2] * training_bboxes[:, 3]).astype(np.float32)

        # Statistical features from each temporal bin (20 bins)
        for bin_idx in range(20):
            bin_data = training_samples[:, bin_idx, :, :]  # (N, 360, 640)

            # Compute statistics for each bin
            feature_data[f"bin_{bin_idx:02d}_mean"] = bin_data.mean(axis=(1, 2)).astype(np.float32)
            feature_data[f"bin_{bin_idx:02d}_std"] = bin_data.std(axis=(1, 2)).astype(np.float32)
            feature_data[f"bin_{bin_idx:02d}_max"] = bin_data.max(axis=(1, 2)).astype(np.float32)
            feature_data[f"bin_{bin_idx:02d}_nonzero"] = (bin_data > 0).sum(axis=(1, 2)).astype(np.float32)

        # Additional derived features
        feature_data["total_activity"] = training_samples.sum(axis=(1, 2, 3)).astype(np.float32)
        feature_data["active_pixels"] = (training_samples > 0).sum(axis=(1, 2, 3)).astype(np.float32)
        feature_data["temporal_center"] = np.array(
            [np.average(range(20), weights=sample.sum(axis=(1, 2)) + 1e-8) for sample in training_samples]
        ).astype(np.float32)

        # Create DataFrame
        df = pl.DataFrame(feature_data)

        # Add normalized features
        df = df.with_columns(
            [
                (pl.col("timestamp") / pl.col("timestamp").max()).alias("timestamp_norm"),
                (pl.col("bbox_area") / pl.col("bbox_area").max()).alias("bbox_area_norm"),
                (pl.col("total_activity") / pl.col("total_activity").max()).alias("activity_norm"),
            ]
        )

        logger.info(f"Created DataFrame with {len(df)} samples and {len(df.columns)} features")
        logger.info(f"Feature columns: {len([col for col in df.columns if col.startswith('bin_')])}")
        logger.info(f"Label distribution: {df['label'].value_counts().sort('label')}")

        return df.lazy()

    except Exception as e:
        logger.warning(f"Failed to load RVT data from {base_path}: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def create_rvt_transform():
    """
    Create a transform function for RVT data that extracts features and labels

    Returns:
        Transform function that converts Polars batch to PyTorch tensors

    Example:
        ```python
        from evlib.pytorch import PolarsDataset, load_rvt_data, create_rvt_transform

        lazy_df = load_rvt_data("path/to/rvt/data")
        transform = create_rvt_transform()
        dataset = PolarsDataset(lazy_df, batch_size=256, transform=transform)
        ```
    """

    def split_features_labels(batch: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        """Transform to separate RVT features and labels from Polars batch"""
        feature_tensors = []

        # Add all temporal bin features (mean, std, max, nonzero for each bin)
        for bin_idx in range(20):
            for stat in ["mean", "std", "max", "nonzero"]:
                key = f"bin_{bin_idx:02d}_{stat}"
                if key in batch:
                    feature_tensors.append(batch[key])

        # Add bounding box features
        for key in ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_area"]:
            if key in batch:
                feature_tensors.append(batch[key])

        # Add activity features
        for key in ["total_activity", "active_pixels", "temporal_center"]:
            if key in batch:
                feature_tensors.append(batch[key])

        # Add normalized features
        for key in ["timestamp_norm", "bbox_area_norm", "activity_norm"]:
            if key in batch:
                feature_tensors.append(batch[key])

        # Stack into feature matrix and extract labels
        features = torch.stack(feature_tensors, dim=1)  # Shape: (batch_size, 91)
        labels = batch["label"].long()  # Shape: (batch_size,)

        return {"features": features, "labels": labels}

    return split_features_labels


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

    def extract_event_features(batch: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        """Transform raw event data to features"""
        # Convert timestamp from microseconds to seconds
        if batch["timestamp"].dtype == torch.int64:
            # Duration in microseconds, convert to float seconds
            timestamp = batch["timestamp"].float() / 1_000_000
        else:
            timestamp = batch["timestamp"].float()

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
        lazy_df, batch_size=batch_size, shuffle=shuffle, transform=transform, drop_last=True, **kwargs
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
