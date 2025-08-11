#!/usr/bin/env python3
"""
Time-based event subsampling demonstration.

This example shows multiple approaches to reduce event density by 50%
within time windows, including uniform sampling, periodic sampling,
and adaptive sampling strategies.
"""

import polars as pl
import evlib
from pathlib import Path


def uniform_time_subsampling(events, reduction_factor=0.5, seed=42):
    """
    Uniformly subsample events to reduce density by specified factor.
    Uses Polars' native sample() function for efficiency.

    Args:
        events: Input events (file path or LazyFrame)
        reduction_factor: Fraction to keep (0.5 = keep 50%, remove 50%)
        seed: Random seed for reproducibility

    Returns:
        LazyFrame with subsampled events
    """
    # Load events if file path provided
    if isinstance(events, (str, Path)):
        events_lf = evlib.load_events(events)
    else:
        events_lf = events

    print(f"=== Uniform Time Subsampling (keep {reduction_factor:.1%}) ===")

    # Get original count
    original_count = len(events_lf.collect())
    print(f"Original events: {original_count:,}")

    # Use Polars' native sample() function - much more efficient!
    # Note: sample() works on DataFrames, so we collect, sample, then make lazy
    df = events_lf.collect()
    sampled_df = df.sample(fraction=reduction_factor, seed=seed)
    filtered_lf = sampled_df.lazy()

    final_count = len(sampled_df)
    print(f"After subsampling: {final_count:,}")
    print(f"Reduction: {100 * (1 - final_count/original_count):.1f}%")

    return filtered_lf


def periodic_time_subsampling(events, keep_ratio=0.5, window_size_ms=10.0):
    """
    Periodic subsampling: keep events in alternating time windows.

    Args:
        events: Input events (file path or LazyFrame)
        keep_ratio: Fraction of time windows to keep
        window_size_ms: Size of time windows in milliseconds

    Returns:
        LazyFrame with periodically subsampled events
    """
    # Load events if file path provided
    if isinstance(events, (str, Path)):
        events_lf = evlib.load_events(events)
    else:
        events_lf = events

    print(f"=== Periodic Time Subsampling (keep {keep_ratio:.1%} of windows) ===")

    df = events_lf.collect()
    original_count = len(df)
    print(f"Original events: {original_count:,}")

    # Convert window size to microseconds
    window_size_us = int(window_size_ms * 1000)

    # Get time range
    timestamps_us = df["t"].dt.total_microseconds()
    t_min = timestamps_us.min()
    _ = timestamps_us.max()  # t_max unused but kept for potential future use

    # Create window assignments
    df_with_windows = df.with_columns([((timestamps_us - t_min) // window_size_us).alias("window_id")])

    # Determine which windows to keep (every nth window)
    keep_every_n = int(1 / keep_ratio)

    # Filter to keep only events in selected windows
    filtered_lf = df_with_windows.filter((pl.col("window_id") % keep_every_n) == 0).drop("window_id").lazy()

    final_count = len(filtered_lf.collect())
    print(f"After periodic subsampling: {final_count:,}")
    print(f"Reduction: {100 * (1 - final_count/original_count):.1f}%")
    print(f"Window size: {window_size_ms}ms, keeping every {keep_every_n} windows")

    return filtered_lf


def adaptive_time_subsampling(events, target_reduction=0.5, window_size_ms=5.0):
    """
    Adaptive subsampling: reduce events more in high-activity regions.

    Args:
        events: Input events (file path or LazyFrame)
        target_reduction: Target fraction to keep (0.5 = 50%)
        window_size_ms: Size of analysis windows in milliseconds

    Returns:
        LazyFrame with adaptively subsampled events
    """
    # Load events if file path provided
    if isinstance(events, (str, Path)):
        events_lf = evlib.load_events(events)
    else:
        events_lf = events

    print(f"=== Adaptive Time Subsampling (target {target_reduction:.1%}) ===")

    df = events_lf.collect()
    original_count = len(df)
    print(f"Original events: {original_count:,}")

    # Convert window size to microseconds
    window_size_us = int(window_size_ms * 1000)

    # Get time range
    timestamps_us = df["t"].dt.total_microseconds()
    t_min = timestamps_us.min()
    _ = timestamps_us.max()  # t_max unused but kept for potential future use

    # Create window assignments and calculate activity per window
    df_with_windows = df.with_columns([((timestamps_us - t_min) // window_size_us).alias("window_id")])

    # Calculate events per window
    window_counts = df_with_windows.group_by("window_id").agg(pl.len().alias("events_in_window"))

    # Calculate adaptive sampling rates
    max_events = window_counts["events_in_window"].max()
    min_events = window_counts["events_in_window"].min()

    # Higher activity windows get lower sampling rates
    window_sampling_rates = window_counts.with_columns(
        [
            # Inverse relationship: high activity -> low sampling rate
            (
                target_reduction
                + (1 - target_reduction)
                * (1 - (pl.col("events_in_window") - min_events) / (max_events - min_events))
            ).alias("sampling_rate")
        ]
    )

    print(f"Activity range: {min_events} to {max_events} events per {window_size_ms}ms window")
    print(
        f"Sampling rates: {window_sampling_rates['sampling_rate'].min():.3f} to {window_sampling_rates['sampling_rate'].max():.3f}"
    )

    # Join sampling rates back to events
    df_with_rates = df_with_windows.join(window_sampling_rates, on="window_id")

    # Use Polars' map_elements for random sampling
    df_with_rand = df_with_rates.with_columns(
        [
            pl.col("sampling_rate").alias("keep_prob"),
            pl.int_range(pl.len())
            .map_elements(lambda x: hash(x + 42) % 10000 / 10000.0, return_dtype=pl.Float64)
            .alias("rand_val"),
        ]
    )

    # Filter based on adaptive sampling rate
    filtered_lf = (
        df_with_rand.filter(pl.col("rand_val") < pl.col("keep_prob"))
        .drop(["window_id", "events_in_window", "keep_prob", "rand_val"])
        .lazy()
    )

    final_count = len(filtered_lf.collect())
    print(f"After adaptive subsampling: {final_count:,}")
    print(f"Reduction: {100 * (1 - final_count/original_count):.1f}%")

    return filtered_lf


def spatial_activity_subsampling(events, target_reduction=0.5, spatial_window=20):
    """
    Spatial activity-based subsampling: reduce events more in high-activity spatial regions.

    Args:
        events: Input events (file path or LazyFrame)
        target_reduction: Target fraction to keep (0.5 = 50%)
        spatial_window: Size of spatial analysis windows in pixels

    Returns:
        LazyFrame with spatially subsampled events
    """
    # Load events if file path provided
    if isinstance(events, (str, Path)):
        events_lf = evlib.load_events(events)
    else:
        events_lf = events

    print(f"=== Spatial Activity Subsampling (target {target_reduction:.1%}) ===")

    df = events_lf.collect()
    original_count = len(df)
    print(f"Original events: {original_count:,}")

    # Create spatial bins
    df_with_bins = df.with_columns(
        [(pl.col("x") // spatial_window).alias("x_bin"), (pl.col("y") // spatial_window).alias("y_bin")]
    )

    # Calculate activity per spatial bin
    spatial_counts = df_with_bins.group_by(["x_bin", "y_bin"]).agg(pl.len().alias("events_in_bin"))

    # Calculate adaptive sampling rates
    max_events = spatial_counts["events_in_bin"].max()
    min_events = spatial_counts["events_in_bin"].min()

    # Higher activity bins get lower sampling rates
    spatial_sampling_rates = spatial_counts.with_columns(
        [
            (
                target_reduction
                + (1 - target_reduction)
                * (1 - (pl.col("events_in_bin") - min_events) / (max_events - min_events))
            ).alias("sampling_rate")
        ]
    )

    print(
        f"Spatial activity range: {min_events} to {max_events} events per {spatial_window}x{spatial_window} pixel region"
    )
    print(
        f"Sampling rates: {spatial_sampling_rates['sampling_rate'].min():.3f} to {spatial_sampling_rates['sampling_rate'].max():.3f}"
    )

    # Join sampling rates back to events
    df_with_rates = df_with_bins.join(spatial_sampling_rates, on=["x_bin", "y_bin"])

    # Use Polars' map_elements for random sampling
    df_with_rand = df_with_rates.with_columns(
        [
            pl.col("sampling_rate").alias("keep_prob"),
            pl.int_range(pl.len())
            .map_elements(lambda x: hash(x + 42) % 10000 / 10000.0, return_dtype=pl.Float64)
            .alias("rand_val"),
        ]
    )

    # Filter based on adaptive sampling rate
    filtered_lf = (
        df_with_rand.filter(pl.col("rand_val") < pl.col("keep_prob"))
        .drop(["x_bin", "y_bin", "events_in_bin", "keep_prob", "rand_val"])
        .lazy()
    )

    final_count = len(filtered_lf.collect())
    print(f"After spatial subsampling: {final_count:,}")
    print(f"Reduction: {100 * (1 - final_count/original_count):.1f}%")

    return filtered_lf


def polars_native_sampling_methods(events, reduction_factor=0.5, seed=42):
    """
    Demonstrate various Polars native sampling methods.

    Args:
        events: Input events (file path or LazyFrame)
        reduction_factor: Fraction to keep (0.5 = keep 50%, remove 50%)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with results from different sampling methods
    """
    # Load events if file path provided
    if isinstance(events, (str, Path)):
        events_lf = evlib.load_events(events)
    else:
        events_lf = events

    print(f"=== Polars Native Sampling Methods (keep {reduction_factor:.1%}) ===")

    df = events_lf.collect()
    original_count = len(df)
    print(f"Original events: {original_count:,}")

    results = {}

    # Method 1: Fraction sampling (uniform random)
    print("\n1. Fraction sampling (df.sample(fraction=0.5)):")
    sampled_fraction = df.sample(fraction=reduction_factor, seed=seed)
    results["fraction"] = sampled_fraction.lazy()
    print(f"   Result: {len(sampled_fraction):,} events ({100 * len(sampled_fraction)/original_count:.1f}%)")

    # Method 2: Fixed count sampling
    target_n = int(original_count * reduction_factor)
    print(f"\n2. Fixed count sampling (df.sample(n={target_n})):")
    sampled_n = df.sample(n=target_n, seed=seed)
    results["fixed_n"] = sampled_n.lazy()
    print(f"   Result: {len(sampled_n):,} events ({100 * len(sampled_n)/original_count:.1f}%)")

    # Method 3: Lazy sampling with filter
    print("\n3. Lazy sampling with filter (using pl.col().map_elements()):")
    # Create a random column and filter - more explicit approach
    sampled_lazy = (
        df.lazy()
        .with_columns(
            [
                pl.int_range(pl.len())
                .map_elements(lambda x: hash(x + seed) % 10000 / 10000.0, return_dtype=pl.Float64)
                .alias("rand_val")
            ]
        )
        .filter(pl.col("rand_val") < reduction_factor)
        .drop("rand_val")
    )
    sampled_lazy_df = sampled_lazy.collect()
    results["lazy_filter"] = sampled_lazy
    print(f"   Result: {len(sampled_lazy_df):,} events ({100 * len(sampled_lazy_df)/original_count:.1f}%)")

    # Method 4: Stratified sampling by polarity
    print("\n4. Stratified sampling by polarity:")
    # Sample each polarity group separately to maintain polarity balance
    pos_events = df.filter(pl.col("polarity") == 1)
    neg_events = df.filter(pl.col("polarity") == -1)

    pos_sampled = pos_events.sample(fraction=reduction_factor, seed=seed)
    neg_sampled = neg_events.sample(fraction=reduction_factor, seed=seed + 1)

    stratified_sampled = pl.concat([pos_sampled, neg_sampled]).sort("t")
    results["stratified"] = stratified_sampled.lazy()
    print(f"   Positive events: {len(pos_events):,} → {len(pos_sampled):,}")
    print(f"   Negative events: {len(neg_events):,} → {len(neg_sampled):,}")
    print(
        f"   Total result: {len(stratified_sampled):,} events ({100 * len(stratified_sampled)/original_count:.1f}%)"
    )

    return results


def combined_filtering_example():
    """
    Example combining evlib filtering with Polars native subsampling.
    """
    print("=== Combined Filtering + Polars Native Subsampling ===")

    # Load data
    data_file = "data/slider_depth/events.txt"
    if not Path(data_file).exists():
        print(f"Data file {data_file} not found. Skipping combined example.")
        return

    # Step 1: Standard evlib preprocessing using filter chaining
    print("\n1. Standard evlib preprocessing:")
    import evlib.filtering as evf

    events = evlib.load_events(data_file)
    filtered = evf.filter_by_time(events, t_start=0.1, t_end=0.8)
    filtered = evf.filter_hot_pixels(filtered, threshold_percentile=99.9)
    preprocessed = evf.filter_noise(filtered, method="refractory", refractory_period_us=1000)

    # Step 2: Apply Polars native subsampling
    print("\n2. Apply Polars native sampling:")
    sampling_results = polars_native_sampling_methods(preprocessed, reduction_factor=0.5)

    # Use fraction sampling result for further processing
    subsampled = sampling_results["fraction"]

    # Step 3: Create representation
    print("\n3. Create stacked histogram:")
    histogram = evlib.create_stacked_histogram(
        subsampled, height=240, width=346, nbins=8, window_duration_ms=25.0
    )

    print(f"Final histogram shape: {histogram.shape}")
    print("SUCCESS: Combined filtering and Polars native subsampling complete!")


def main():
    """
    Demonstrate different event subsampling strategies.
    """
    print("Event Subsampling Strategies Demonstration")
    print("=" * 50)

    # Use available data file
    data_file = "data/slider_depth/events.txt"

    if not Path(data_file).exists():
        print(f"Data file {data_file} not found.")
        print("Please ensure you have the slider_depth dataset available.")
        return

    # Load original events for comparison
    original_events = evlib.load_events(data_file)
    print(f"Dataset: {data_file}")
    print(f"Original events: {len(original_events.collect()):,}")
    print()

    # Method 1: Polars Native Sampling (RECOMMENDED)
    print("METHOD 1: Polars Native Sampling (RECOMMENDED)")
    print("-" * 50)
    _ = polars_native_sampling_methods(original_events, reduction_factor=0.5)
    print()

    # Method 2: Uniform random subsampling
    print("METHOD 2: Uniform Random Subsampling")
    print("-" * 40)
    _ = uniform_time_subsampling(original_events, reduction_factor=0.5)
    print()

    # Method 3: Periodic time window subsampling
    print("METHOD 3: Periodic Time Window Subsampling")
    print("-" * 40)
    _ = periodic_time_subsampling(original_events, keep_ratio=0.5, window_size_ms=10.0)
    print()

    # Method 4: Adaptive time-based subsampling
    print("METHOD 4: Adaptive Time-based Subsampling")
    print("-" * 40)
    _ = adaptive_time_subsampling(original_events, target_reduction=0.5, window_size_ms=5.0)
    print()

    # Method 5: Spatial activity-based subsampling
    print("METHOD 5: Spatial Activity-based Subsampling")
    print("-" * 40)
    _ = spatial_activity_subsampling(original_events, target_reduction=0.5, spatial_window=20)
    print()

    # Method 6: Combined filtering + subsampling
    combined_filtering_example()

    print("\n" + "=" * 50)
    print("Summary of Subsampling Methods:")
    print("1. Polars Native: Most efficient, built-in sampling methods")
    print("2. Uniform Random: Simple, unbiased reduction")
    print("3. Periodic Windows: Preserves temporal structure")
    print("4. Adaptive Time: Reduces more in high-activity periods")
    print("5. Spatial Activity: Reduces more in high-activity regions")
    print("6. Combined: Preprocessing + subsampling + representations")
    print("\nRECOMMENDED: Use Polars native sampling for best performance!")


if __name__ == "__main__":
    main()
