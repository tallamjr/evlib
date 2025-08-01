#!/usr/bin/env python3
"""
Simplified Polars PyTorch DataLoader - Leveraging native .to_torch()

This is the cleanest solution for GitHub issue #10:
- Takes Polars LazyFrame as input
- Uses native .to_torch() for zero-copy conversion
- Minimal code, maximum efficiency
- All PyTorch features supported

Architecture:
Parquet/IPC → Polars LazyFrame → .to_torch() → PyTorch DataLoader
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import torch
from torch.utils.data import IterableDataset, DataLoader
import polars as pl

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import evlib for real data loading
try:
    import evlib

    logger.info("evlib available for real data loading")
except ImportError:
    logger.warning("evlib not available, will use synthetic data")
    evlib = None


class PolarsDataset(IterableDataset):
    """
    Minimal PyTorch Dataset from Polars LazyFrame

    Uses Polars' native .to_torch() for efficient conversion
    """

    def __init__(
        self,
        lazy_df: pl.LazyFrame,
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
            transform: Optional transform function
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
        self._indices = None

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
                for i, col in enumerate(batch_df.columns):
                    batch_tensors[col] = tensor_data[:, i]

            except Exception as e:
                # Fallback: convert columns individually to handle mixed dtypes
                logger.debug(f"Native .to_torch() failed ({e}), using column-wise conversion")
                batch_tensors = {}
                for col in batch_df.columns:
                    col_data = batch_df[col]

                    # Handle different data types
                    if col_data.dtype == pl.Duration:
                        # Convert duration to float (nanoseconds as float)
                        tensor_data = torch.from_numpy(
                            col_data.dt.total_nanoseconds().to_numpy().astype(np.float32)
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


def load_real_rvt_data(max_samples: int = 1000) -> Optional[pl.LazyFrame]:
    """Load real RVT preprocessed event representations with labels"""
    import h5py
    import numpy as np

    # Try to find RVT data files
    rvt_base_paths = [
        Path("data/gen4_1mpx_processed_RVT/val/moorea_2019-02-21_000_td_2257500000_2317500000"),
        Path("../data/gen4_1mpx_processed_RVT/val/moorea_2019-02-21_000_td_2257500000_2317500000"),
    ]

    for base_path in rvt_base_paths:
        if base_path.exists():
            try:
                # Load event representations
                repr_file = (
                    base_path
                    / "event_representations_v2"
                    / "stacked_histogram_dt50_nbins10"
                    / "event_representations_ds2_nearest.h5"
                )
                labels_file = base_path / "labels_v2" / "labels.npz"

                if not repr_file.exists() or not labels_file.exists():
                    logger.warning(f"Missing files in {base_path}")
                    continue

                logger.info(f"Loading RVT data from {base_path}")

                # Load event representations
                with h5py.File(repr_file, "r") as f:
                    logger.info(f"Available keys in HDF5: {list(f.keys())}")
                    if "data" in f:
                        representations = f["data"][:max_samples]  # Shape: (N, bins, height, width)
                        logger.info(f"Loaded representations shape: {representations.shape}")
                    else:
                        logger.warning(f"'data' key not found in {repr_file}")
                        continue

                # Load labels
                labels_data = np.load(labels_file)
                logger.info(f"Available keys in labels: {list(labels_data.keys())}")

                if "labels" in labels_data:
                    raw_labels = labels_data["labels"][:max_samples]
                    logger.info(f"Loaded raw labels shape: {raw_labels.shape}, dtype: {raw_labels.dtype}")

                    # Handle structured array labels
                    if raw_labels.dtype.names is not None:
                        # Structured array - extract class_id field
                        if "class_id" in raw_labels.dtype.names:
                            labels = raw_labels["class_id"].astype(np.int32)
                            logger.info(f"Extracted class_id labels, shape: {labels.shape}")
                            logger.info(f"Label unique values: {np.unique(labels)}")
                            logger.info(f"Label value range: [{labels.min()}, {labels.max()}]")
                        else:
                            # Use first field as fallback
                            field_name = raw_labels.dtype.names[0]
                            labels = raw_labels[field_name].astype(np.int32)
                            logger.info(f"Using field '{field_name}' as labels, shape: {labels.shape}")
                    else:
                        # Simple array
                        labels = raw_labels.astype(np.int32)
                        logger.info(f"Using simple labels, shape: {labels.shape}")

                elif "y" in labels_data:
                    labels = labels_data["y"][:max_samples].astype(np.int32)
                    logger.info(f"Loaded labels from 'y' key, shape: {labels.shape}")
                else:
                    # Create mock labels if not found
                    labels = np.random.randint(0, 5, max_samples)
                    logger.warning(f"No labels found, created mock labels shape: {labels.shape}")

                # Flatten representations for easier handling (convert to features per sample)
                n_samples = min(len(representations), len(labels))
                representations = representations[:n_samples]
                labels = labels[:n_samples]

                # Flatten each representation to a feature vector
                flattened_repr = representations.reshape(n_samples, -1)
                logger.info(f"Flattened representations to shape: {flattened_repr.shape}")

                # Create sample indices and metadata
                sample_indices = np.arange(n_samples)

                # Create Polars DataFrame
                data_dict = {
                    "sample_idx": sample_indices,
                    "label": labels.astype(np.int32),
                }

                # Add flattened features as columns (first 100 features to keep manageable)
                n_features = min(100, flattened_repr.shape[1])
                for i in range(n_features):
                    data_dict[f"feature_{i:03d}"] = flattened_repr[:, i].astype(np.float32)

                df = pl.DataFrame(data_dict)

                # Add derived features for demo
                df = df.with_columns(
                    [
                        (pl.col("sample_idx") / n_samples).alias("sample_norm"),
                        (pl.col("label").cast(pl.Float32) / pl.col("label").max()).alias("label_norm"),
                    ]
                )

                logger.info(f"Created DataFrame with {len(df)} samples and {len(df.columns)} columns")
                logger.info(f"Label distribution: {df['label'].value_counts().sort('label').to_dict()}")

                return df.lazy()

            except Exception as e:
                logger.warning(f"Failed to load RVT data from {base_path}: {e}")
                continue

    logger.warning("No RVT data files found")
    return None


def create_synthetic_data(n_events: int = 10000) -> pl.LazyFrame:
    """Create synthetic event data as fallback"""
    logger.info(f"Creating {n_events} synthetic events")

    df = pl.DataFrame(
        {
            "timestamp": np.cumsum(np.random.exponential(1e-5, n_events)),
            "x": np.random.randint(0, 640, n_events),
            "y": np.random.randint(0, 480, n_events),
            "polarity": np.random.randint(0, 2, n_events),
        }
    )

    # Add derived features
    df = df.with_columns(
        [
            (pl.col("x") / 640.0).alias("x_norm"),
            (pl.col("y") / 480.0).alias("y_norm"),
            (pl.col("polarity").cast(pl.Float32) * 2 - 1).alias("polarity_signed"),
            (pl.col("timestamp") - pl.col("timestamp").min()).alias("time_offset"),
        ]
    )

    return df.lazy()


def demo_basic_usage():
    """Demonstrate basic usage with real data"""
    logger.info("Starting basic usage demo")
    logger.info("=" * 60)

    # Try to load real data first, fall back to synthetic
    lazy_df = load_real_rvt_data(max_samples=500)
    if lazy_df is None:
        lazy_df = create_synthetic_data(10000)

    # Create dataset
    dataset = PolarsDataset(lazy_df, batch_size=128, shuffle=True)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)  # Batching handled by dataset

    # Iterate through batches
    logger.info("Processing batches...")
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i}:")
        for key, tensor in batch.items():
            logger.info(f"  {key}: {tensor.shape} ({tensor.dtype})")

        if i >= 2:  # Show first 3 batches
            break


def demo_training_pipeline():
    """Demonstrate complete training pipeline with real data"""
    logger.info("")
    logger.info("Starting training pipeline demo")
    logger.info("=" * 60)

    # Try to load real data, otherwise use synthetic
    lazy_df = load_real_rvt_data(max_samples=10000)
    if lazy_df is None:
        lazy_df = create_synthetic_data(50000)

    # RVT data already has real labels, no need to add synthetic ones
    # If we're using synthetic data, add labels
    schema = lazy_df.collect_schema()
    if "label" not in schema.names() and "x" in schema.names():
        # This is synthetic data, add labels
        lazy_df = lazy_df.with_columns(
            [(pl.col("x") % 3).alias("label")]  # Simple synthetic labels based on x coordinate
        )

    # Split features and labels using transform
    def split_features_labels(batch):
        """Transform to separate features and labels"""
        # Check if we have RVT features or synthetic features
        feature_keys = [k for k in batch.keys() if k.startswith("feature_")]

        if feature_keys:
            # RVT data with flattened representation features
            feature_tensors = [batch[k] for k in sorted(feature_keys)]
            features = torch.stack(feature_tensors, dim=1)
        elif "x_norm" in batch and "y_norm" in batch and "polarity_signed" in batch:
            # Synthetic event data
            features = torch.stack([batch["x_norm"], batch["y_norm"], batch["polarity_signed"]], dim=1)
        else:
            # Fallback: use first 3 numeric columns as features
            numeric_keys = [
                k for k in batch.keys() if k != "label" and batch[k].dtype in [torch.float32, torch.float64]
            ][:3]
            if len(numeric_keys) >= 3:
                features = torch.stack([batch[k] for k in numeric_keys], dim=1)
            else:
                raise ValueError(f"Not enough numeric features found. Available keys: {list(batch.keys())}")

        labels = batch["label"].long()
        return {"features": features, "labels": labels}

    # Create dataset with transform
    dataset = PolarsDataset(
        lazy_df, batch_size=256, shuffle=True, transform=split_features_labels, drop_last=True, seed=42
    )

    # Create DataLoader with performance features
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=(torch.cuda.is_available() or torch.backends.mps.is_available()),
        num_workers=0,  # Use 0 to avoid pickling issues with local functions
    )

    # Determine input size based on data type
    # Test with one batch to get feature dimensions
    test_batch = next(iter(dataloader))
    input_size = test_batch["features"].shape[1]
    n_classes = len(torch.unique(test_batch["labels"]))

    logger.info(f"Input size: {input_size}, Number of classes: {n_classes}")

    # Create model based on actual data dimensions
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, max(n_classes, 3)),  # At least 3 classes for demo
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    logger.info(f"Training on {device}...")
    model.train()

    batch_times = []
    for epoch in range(20):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            start_time = time.time()

            # Move to device and fix dtypes
            features = batch["features"].float().to(device)  # Ensure float32
            labels = batch["labels"].long().to(device)  # Ensure int64

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            epoch_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"  Epoch {epoch}, Batch {batch_idx}: "
                    f"Loss={loss.item():.4f}, "
                    f"Acc={100.*correct/total:.1f}%, "
                    f"Time={batch_time*1000:.1f}ms"
                )

        logger.info(
            f"Epoch {epoch} complete: "
            f"Avg Loss={epoch_loss/(batch_idx+1):.4f}, "
            f"Accuracy={100.*correct/total:.1f}%"
        )

    # Performance summary
    avg_batch_time = np.mean(batch_times) * 1000
    throughput = 256 / np.mean(batch_times)

    logger.info("")
    logger.info("Performance Summary:")
    logger.info(f"  Average batch time: {avg_batch_time:.2f}ms")
    logger.info(f"  Training throughput: {throughput:.0f} samples/sec")


def demo_advanced_features():
    """Demonstrate advanced features with real data"""
    logger.info("")
    logger.info("Starting advanced features demo")
    logger.info("=" * 60)

    # Try to load real data, otherwise use synthetic
    lazy_df = load_real_rvt_data(max_samples=2000)
    if lazy_df is None:
        lazy_df = create_synthetic_data(100000)

    # Add more derived features using Polars expressions
    # Check what type of data we have
    schema = lazy_df.collect_schema()
    if "x" in schema.names() and "y" in schema.names():
        # This is raw event data
        lazy_df = lazy_df.with_columns(
            [
                ((pl.col("x") - 320) ** 2 + (pl.col("y") - 240) ** 2).sqrt().alias("distance_from_center"),
                (pl.col("timestamp").diff().fill_null(0)).alias("inter_event_time"),
            ]
        )
    else:
        # This is RVT preprocessed data with features
        lazy_df = lazy_df.with_columns(
            [
                (pl.col("sample_idx").cast(pl.Float32) / pl.col("sample_idx").max()).alias(
                    "sample_norm_advanced"
                ),
                (pl.col("feature_000") + pl.col("feature_001")).alias("combined_feature_01"),
            ]
        )

    # Show sample of the data
    sample_df = lazy_df.head(5).collect()
    logger.info("Dataset with derived features:")
    logger.info(str(sample_df))

    # Create dataset
    dataset = PolarsDataset(lazy_df, batch_size=512, shuffle=True)

    # Test performance
    logger.info("")
    logger.info("Performance test...")
    start_time = time.time()
    n_batches = 0
    n_samples = 0

    for batch in dataset:
        n_batches += 1
        # Get batch size from any available column
        if "x" in batch:
            n_samples += len(batch["x"])
        elif "sample_idx" in batch:
            n_samples += len(batch["sample_idx"])
        else:
            # Use first available column
            first_key = next(iter(batch.keys()))
            n_samples += len(batch[first_key])

        if n_batches >= 50:  # Process 50 batches
            break

    elapsed = time.time() - start_time
    logger.info(f"  Processed {n_samples:,} samples in {n_batches} batches")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Throughput: {n_samples/elapsed:.0f} samples/sec")


def main():
    """Run all demos"""
    logger.info("Simplified Polars PyTorch DataLoader")
    logger.info("Using native .to_torch() for maximum efficiency")
    logger.info("=" * 80)

    try:
        # Demo 1: Basic usage
        demo_basic_usage()

        # Demo 2: Training pipeline
        demo_training_pipeline()

        # Demo 3: Advanced features
        demo_advanced_features()

        logger.info("")
        logger.info("=" * 80)
        logger.info("All demos completed successfully!")
        logger.info("")
        logger.info("Key advantages of this simplified approach:")
        logger.info("  - Minimal code - leverages Polars' native .to_torch()")
        logger.info("  - Zero-copy conversion when possible")
        logger.info("  - Full PyTorch compatibility")
        logger.info("  - Excellent performance")
        logger.info("  - Easy to understand and maintain")
        logger.info("  - Works with real event camera data")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
