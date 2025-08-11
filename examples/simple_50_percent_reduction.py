#!/usr/bin/env python3
"""
Simple 50% Event Reduction Example using Polars Native Sampling

This example demonstrates the most efficient way to remove 50% of events
using Polars' native sampling functions.
"""

from pathlib import Path

import polars as pl

import evlib


def remove_50_percent_events_simple(events, seed=42):
    """
    Remove 50% of events using Polars native sampling.

    This is the RECOMMENDED approach - most efficient and idiomatic.

    Args:
        events: Input events (file path or LazyFrame)
        seed: Random seed for reproducibility

    Returns:
        LazyFrame with 50% of events removed
    """
    # Load events if file path provided
    if isinstance(events, (str, Path)):
        events_lf = evlib.load_events(events)
    else:
        events_lf = events

    # Use Polars' native sample() function - most efficient!
    df = events_lf.collect()
    sampled_df = df.sample(fraction=0.5, seed=seed)

    print(f"Original events: {len(df):,}")
    print(f"After 50% reduction: {len(sampled_df):,}")
    print(f"Reduction: {100 * (1 - len(sampled_df)/len(df)):.1f}%")

    return sampled_df.lazy()


def remove_50_percent_with_stratification(events, seed=42):
    """
    Remove 50% of events while maintaining polarity balance.

    This approach ensures that positive and negative events are
    reduced proportionally to maintain the original polarity distribution.

    Args:
        events: Input events (file path or LazyFrame)
        seed: Random seed for reproducibility

    Returns:
        LazyFrame with 50% of events removed, maintaining polarity balance
    """
    # Load events if file path provided
    if isinstance(events, (str, Path)):
        events_lf = evlib.load_events(events)
    else:
        events_lf = events

    df = events_lf.collect()

    # Check polarity values
    polarity_values = df["polarity"].unique().sort()
    print(f"Polarity values in data: {polarity_values.to_list()}")

    # Sample each polarity group separately
    sampled_groups = []
    for polarity in polarity_values:
        polarity_events = df.filter(pl.col("polarity") == polarity)
        if len(polarity_events) > 0:
            sampled_polarity = polarity_events.sample(fraction=0.5, seed=seed)
            sampled_groups.append(sampled_polarity)
            print(f"Polarity {polarity}: {len(polarity_events):,} → {len(sampled_polarity):,}")

    # Combine and sort by timestamp
    if sampled_groups:
        stratified_sampled = pl.concat(sampled_groups).sort("t")
    else:
        stratified_sampled = df.sample(fraction=0.5, seed=seed)

    print(f"Original events: {len(df):,}")
    print(f"After stratified 50% reduction: {len(stratified_sampled):,}")
    print(f"Reduction: {100 * (1 - len(stratified_sampled)/len(df)):.1f}%")

    return stratified_sampled.lazy()


def complete_preprocessing_with_50_percent_reduction(data_file, seed=42):
    """
    Complete preprocessing pipeline with 50% event reduction.

    This combines evlib filtering with Polars native sampling for
    optimal performance and clean code.

    Args:
        data_file: Path to event data file
        seed: Random seed for reproducibility

    Returns:
        LazyFrame with preprocessed and subsampled events
    """
    print("=== Complete Preprocessing + 50% Reduction ===")

    # Step 1: Standard evlib preprocessing using filter chaining
    print("\n1. Applying evlib preprocessing...")
    import evlib.filtering as evf

    events = evlib.load_events(data_file)
    filtered = evf.filter_by_time(events, t_start=0.1, t_end=0.8)
    filtered = evf.filter_hot_pixels(filtered, threshold_percentile=99.9)
    preprocessed = evf.filter_noise(filtered, method="refractory", refractory_period_us=1000)

    # Step 2: Apply 50% reduction using Polars native sampling
    print("\n2. Applying 50% reduction...")
    df = preprocessed.collect()
    reduced_df = df.sample(fraction=0.5, seed=seed)

    print(f"After preprocessing: {len(df):,} events")
    print(f"After 50% reduction: {len(reduced_df):,} events")
    print(f"Final reduction: {100 * (1 - len(reduced_df)/len(df)):.1f}%")

    return reduced_df.lazy()


def main():
    """
    Demonstrate simple 50% event reduction methods.
    """
    print("Simple 50% Event Reduction Examples")
    print("=" * 40)

    # Use available data file
    data_file = "data/slider_depth/events.txt"

    if not Path(data_file).exists():
        print(f"Data file {data_file} not found.")
        print("Please ensure you have the slider_depth dataset available.")
        return

    # Original events
    original_events = evlib.load_events(data_file)
    print(f"Dataset: {data_file}")
    print(f"Original events: {len(original_events.collect()):,}")
    print()

    # Method 1: Simple 50% reduction (RECOMMENDED)
    print("METHOD 1: Simple 50% Reduction (RECOMMENDED)")
    print("-" * 45)
    reduced_events = remove_50_percent_events_simple(original_events)
    print()

    # Method 2: Stratified 50% reduction
    print("METHOD 2: Stratified 50% Reduction")
    print("-" * 35)
    _ = remove_50_percent_with_stratification(original_events)
    print()

    # Method 3: Complete preprocessing pipeline
    print("METHOD 3: Complete Preprocessing Pipeline")
    print("-" * 40)
    final_events = complete_preprocessing_with_50_percent_reduction(data_file)
    print(final_events.collect())

    # Test with representations
    print("INTEGRATION TEST: Create Stacked Histogram")
    print("-" * 42)
    histogram = evlib.create_stacked_histogram(
        reduced_events, height=240, width=346, nbins=8, window_duration_ms=50.0
    )
    print(f"Histogram shape: {histogram.shape}")
    print("SUCCESS: Integration successful!")
    print()

    # Summary
    print("SUMMARY:")
    print("--------")
    print("SUCCESS: Use df.sample(fraction=0.5) for simple 50% reduction")
    print("SUCCESS: Use stratified sampling to maintain polarity balance")
    print("SUCCESS: Combine with evlib filtering functions for complete pipeline")
    print("SUCCESS: Polars native sampling is most efficient and idiomatic")
    print("\nCode example:")
    print("```python")
    print("import evlib")
    print("events = evlib.load_events('data.h5')")
    print("df = events.collect()")
    print("reduced = df.sample(fraction=0.5, seed=42)")
    print("histogram = evlib.create_stacked_histogram(reduced.lazy(), height=480, width=640)")
    print("```")


if __name__ == "__main__":
    main()
