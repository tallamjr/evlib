#!/usr/bin/env python3
"""
Demonstration of the new evlib.filtering module.

This script shows how to use the comprehensive filtering functionality
to preprocess event camera data with various filtering operations.
"""

import numpy as np
import polars as pl
import evlib.filtering as evf


def create_demo_events(num_events=10000):
    """Create synthetic event data for demonstration."""
    print("Creating synthetic event data...")

    np.random.seed(42)

    # Create realistic event data
    width, height = 640, 480
    duration_sec = 2.0

    # Most events are normal
    normal_events = int(num_events * 0.8)
    x_normal = np.random.randint(0, width, normal_events)
    y_normal = np.random.randint(0, height, normal_events)
    t_normal = np.random.uniform(0, duration_sec * 1_000_000, normal_events)

    # Some events are hot pixels (concentrated locations)
    hot_events = int(num_events * 0.1)
    hot_pixels = [(100, 100), (200, 200), (300, 300)]
    x_hot = np.random.choice([p[0] for p in hot_pixels], hot_events)
    y_hot = np.random.choice([p[1] for p in hot_pixels], hot_events)
    t_hot = np.random.uniform(0, duration_sec * 1_000_000, hot_events)

    # Some events are noise (rapid-fire at same locations)
    noise_events = num_events - normal_events - hot_events
    noise_pixels = [(150, 150), (250, 250)]
    x_noise = np.random.choice([p[0] for p in noise_pixels], noise_events)
    y_noise = np.random.choice([p[1] for p in noise_pixels], noise_events)
    # Create rapid-fire events (within 500μs of each other)
    t_noise = []
    for i in range(noise_events):
        base_time = 500_000 + i * 100
        t_noise.append(base_time + np.random.uniform(0, 500))
    t_noise = np.array(t_noise)

    # Combine all events
    x = np.concatenate([x_normal, x_hot, x_noise])
    y = np.concatenate([y_normal, y_hot, y_noise])
    timestamps_us = np.concatenate([t_normal, t_hot, t_noise])

    # Sort by timestamp
    sort_idx = np.argsort(timestamps_us)
    x = x[sort_idx]
    y = y[sort_idx]
    timestamps_us = timestamps_us[sort_idx]

    # Random polarities
    polarity = np.random.randint(0, 2, num_events)

    # Create Polars DataFrame
    events_df = pl.DataFrame(
        {
            "x": x,
            "y": y,
            "timestamp": pl.Series(timestamps_us, dtype=pl.Duration(time_unit="us")),
            "polarity": polarity,
        }
    )

    print(f"Created {num_events:,} events spanning {duration_sec:.1f}s")
    print(f"Resolution: {width}x{height}")
    print(f"Hot pixels: {len(hot_pixels)} locations")
    print(f"Noise pixels: {len(noise_pixels)} locations")

    return events_df.lazy()


def demo_individual_filters():
    """Demonstrate individual filtering functions."""
    print("\n=== Individual Filter Demonstrations ===")

    # Create demo data
    events = create_demo_events(num_events=10000)

    print(f"\nOriginal events: {len(events.collect()):,}")

    # 1. Time filtering
    print("\n1. Time Filtering:")
    time_filtered = evf.filter_by_time(events, t_start=0.5, t_end=1.5)
    print(f"   Events between 0.5s and 1.5s: {len(time_filtered.collect()):,}")

    # 2. Spatial filtering (ROI)
    print("\n2. Spatial Filtering (ROI):")
    roi_filtered = evf.filter_by_roi(events, x_min=200, x_max=400, y_min=150, y_max=350)
    print(f"   Events in ROI [200:400, 150:350]: {len(roi_filtered.collect()):,}")

    # 3. Polarity filtering
    print("\n3. Polarity Filtering:")
    positive_events = evf.filter_by_polarity(events, polarity=1)
    negative_events = evf.filter_by_polarity(events, polarity=0)
    print(f"   Positive events: {len(positive_events.collect()):,}")
    print(f"   Negative events: {len(negative_events.collect()):,}")

    # 4. Hot pixel removal
    print("\n4. Hot Pixel Removal:")
    hot_pixel_filtered = evf.filter_hot_pixels(events, threshold_percentile=95.0)
    print(f"   Events after hot pixel removal: {len(hot_pixel_filtered.collect()):,}")

    # 5. Noise filtering
    print("\n5. Noise Filtering:")
    noise_filtered = evf.filter_noise(events, method="refractory", refractory_period_us=1000)
    print(f"   Events after noise filtering: {len(noise_filtered.collect()):,}")


def demo_preprocessing_pipeline():
    """Demonstrate the complete preprocessing pipeline."""
    print("\n=== Complete Preprocessing Pipeline ===")

    # Create demo data
    events = create_demo_events(num_events=15000)

    # Apply complete preprocessing
    processed = evf.preprocess_events(
        events,
        t_start=0.2,
        t_end=1.8,
        roi=(100, 500, 100, 400),
        polarity=1,  # Keep only positive events
        remove_hot_pixels=True,
        remove_noise=True,
        hot_pixel_threshold=99.0,
        refractory_period_us=1000,
    )

    print(f"\nFinal processed events: {len(processed.collect()):,}")

    # Show sample of processed events
    sample = processed.limit(10).collect()
    print("\nSample of processed events:")
    print(sample)


def demo_chaining_filters():
    """Demonstrate chaining multiple filters."""
    print("\n=== Filter Chaining Demonstration ===")

    events = create_demo_events(num_events=8000)

    # Chain filters step by step
    print(f"Starting with {len(events.collect()):,} events")

    # Step 1: Time filter
    step1 = evf.filter_by_time(events, t_start=0.3, t_end=1.7)
    print(f"After time filter: {len(step1.collect()):,} events")

    # Step 2: ROI filter
    step2 = evf.filter_by_roi(step1, x_min=150, x_max=450, y_min=100, y_max=400)
    print(f"After ROI filter: {len(step2.collect()):,} events")

    # Step 3: Polarity filter
    step3 = evf.filter_by_polarity(step2, polarity=1)
    print(f"After polarity filter: {len(step3.collect()):,} events")

    # Step 4: Hot pixel removal
    step4 = evf.filter_hot_pixels(step3, threshold_percentile=98.0)
    print(f"After hot pixel removal: {len(step4.collect()):,} events")

    # Step 5: Noise filtering
    final = evf.filter_noise(step4, method="refractory", refractory_period_us=800)
    print(f"Final result: {len(final.collect()):,} events")

    # Show the reduction
    original_count = len(events.collect())
    final_count = len(final.collect())
    reduction = (original_count - final_count) / original_count
    print(f"\nOverall reduction: {reduction:.1%}")


def main():
    """Run all demonstrations."""
    print("evlib.filtering Module Demonstration")
    print("=" * 50)

    # Run individual demonstrations
    demo_individual_filters()
    demo_preprocessing_pipeline()
    demo_chaining_filters()

    print("\n" + "=" * 50)
    print("✓ All demonstrations completed successfully!")
    print("\nThe filtering module provides:")
    print("  • Time-based filtering")
    print("  • Spatial ROI filtering")
    print("  • Polarity-based filtering")
    print("  • Hot pixel detection and removal")
    print("  • Noise filtering with refractory periods")
    print("  • Complete preprocessing pipeline")
    print("  • Efficient Polars DataFrame operations")
    print("  • Progress reporting for large datasets")


if __name__ == "__main__":
    main()
