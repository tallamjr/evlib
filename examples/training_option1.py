#!/usr/bin/env python3
"""
Training Example - Option 1: Direct PolarsDataset Usage (Recommended)

Working training pipeline using PolarsDataset directly with RVT feature extraction.
This shows a complete training example with proper losses and metrics.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from evlib.pytorch import load_rvt_data, PolarsDataset


def create_feature_transform():
    """Transform to extract features from RVT data for classification."""

    def extract_features(batch):
        """Extract statistical features from RVT data for training."""
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

        features = torch.stack(feature_tensors, dim=1)
        labels = batch["label"].long()
        return {"features": features, "labels": labels}

    return extract_features


def main():
    print("Training Example - Option 1: Direct PolarsDataset Usage")
    print("=" * 60)

    # Load RVT data
    data_path = "data/gen4_1mpx_processed_RVT/val/moorea_2019-02-21_000_td_2257500000_2317500000"
    print("Loading RVT data...")

    lazy_df = load_rvt_data(data_path, max_samples=1000)
    if lazy_df is None:
        print(f"Error: Could not load RVT data from {data_path}")
        print("Make sure the data directory exists and contains the required files.")
        return

    print("Successfully loaded RVT data!")

    # Show data info
    sample_df = lazy_df.head(5).collect()
    print(f"Dataset: {len(sample_df)} samples x {len(sample_df.columns)} features")
    print(f"Classes: {sorted(sample_df['label'].unique())}")
    print(f"Temporal bin features: {len([col for col in sample_df.columns if col.startswith('bin_')])}")

    # Create feature transform and dataset
    transform = create_feature_transform()
    dataset = PolarsDataset(lazy_df, batch_size=128, shuffle=True, transform=transform, drop_last=True)

    # Get feature dimensions from test batch
    test_batch = next(iter(dataset))
    input_size = test_batch["features"].shape[1]
    n_classes = len(torch.unique(test_batch["labels"]))

    print(f"Input features: {input_size}")
    print(f"Number of classes: {n_classes}")
    print(f"Batch shape: {test_batch['features'].shape}")

    # Create model for RVT statistical features
    model = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, max(n_classes, 3)),
    )

    # Device selection (prioritize MPS on Mac, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on device: {device}")
    print("=" * 60)

    # Training loop
    model.train()
    batch_times = []

    for epoch in range(10):
        epoch_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        for batch in dataset:
            start_time = time.time()

            # Move to device
            features = batch["features"].float().to(device)
            labels = batch["labels"].long().to(device)

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
            batch_count += 1

            # Log progress
            if batch_count % 5 == 0:
                print(
                    f"Epoch {epoch+1}, Batch {batch_count}: "
                    f"Loss={loss.item():.4f}, "
                    f"Acc={100.*correct/total:.1f}%, "
                    f"Time={batch_time*1000:.1f}ms"
                )

        # Epoch summary
        avg_loss = epoch_loss / batch_count
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1} complete: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")

    # Performance summary
    avg_batch_time = np.mean(batch_times) * 1000
    throughput = 128 / np.mean(batch_times)  # batch_size=128

    print("=" * 60)
    print("Training completed!")
    print(f"Average batch time: {avg_batch_time:.2f}ms")
    print(f"Training throughput: {throughput:.0f} samples/sec")
    print(f"Final accuracy: {accuracy:.1f}%")

    print("\nAdvantages of PolarsDataset:")
    print("- Direct iteration without DataLoader overhead")
    print("- Efficient Polars → PyTorch conversion")
    print("- Better memory management for large datasets")
    print("- Built-in shuffling and batching")


if __name__ == "__main__":
    main()
