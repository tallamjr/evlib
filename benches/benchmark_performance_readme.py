#!/usr/bin/env python3
"""
Performance verification script for README metrics.

This script validates the performance claims made in the README.md file
using available test data and generates performance visualization plots.

Requirements:
    pip install matplotlib psutil
"""

import evlib
import polars as pl
import time
import os
import psutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def get_memory_usage_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_loading_speed(file_path):
    """Benchmark loading speed as claimed in README"""
    print(f"\nSTATS: Benchmarking Loading Speed: {file_path}")

    if not Path(file_path).exists():
        print(f"FAIL: File not found: {file_path}")
        return None

    start_time = time.time()
    lf = evlib.load_events(file_path)
    df = lf.collect()
    load_time = time.time() - start_time

    events_per_second = len(df) / load_time

    print(f"PASS: Loaded {len(df):,} events in {load_time:.2f}s")
    print(f"FAST: Speed: {events_per_second:,.0f} events/s")

    # Check against README claim of 600k+ events/s
    if events_per_second >= 600000:
        print("TARGET: MEETS README CLAIM: ≥600k events/s")
    else:
        print(f"WARNING: BELOW README CLAIM: {events_per_second:,.0f} < 600k events/s")

    return events_per_second, len(df), load_time


def benchmark_filter_speed(file_path):
    """Benchmark filter speed as claimed in README"""
    print(f"\nANALYSIS: Benchmarking Filter Speed: {file_path}")

    if not Path(file_path).exists():
        print(f"FAIL: File not found: {file_path}")
        return None

    lf = evlib.load_events(file_path)
    df = lf.collect()

    # Test filtering speed
    start_time = time.time()
    filtered = lf.filter((pl.col("polarity") == 1) & (pl.col("x") > 50) & (pl.col("x") < 250)).collect()
    filter_time = time.time() - start_time

    events_per_second = len(df) / filter_time

    print(f"PASS: Filtered {len(df):,} events to {len(filtered):,} in {filter_time:.4f}s")
    print(f"FAST: Filter speed: {events_per_second:,.0f} events/s")

    # Check against README claim of 400M+ events/s
    if events_per_second >= 400_000_000:
        print("TARGET: MEETS README CLAIM: ≥400M events/s")
    elif events_per_second >= 100_000_000:
        print(f"GOOD PERFORMANCE: {events_per_second:,.0f} events/s (≥100M)")
    else:
        print(f"WARNING: BELOW EXPECTED: {events_per_second:,.0f} < 100M events/s")

    return events_per_second, len(df), filter_time


def benchmark_memory_efficiency(file_path):
    """Benchmark memory efficiency as claimed in README"""
    print(f"\nMEMORY: Benchmarking Memory Efficiency: {file_path}")

    if not Path(file_path).exists():
        print(f"FAIL: File not found: {file_path}")
        return None

    # Force garbage collection to get clean baseline
    import gc

    gc.collect()

    initial_memory = get_memory_usage_mb()

    lf = evlib.load_events(file_path)
    df = lf.collect()

    peak_memory = get_memory_usage_mb()
    memory_used = peak_memory - initial_memory

    bytes_per_event = (memory_used * 1024 * 1024) / len(df)

    print(f"PASS: Loaded {len(df):,} events")
    print(f"MEMORY: Memory used: {memory_used:.1f} MB")
    print(f"STATS: Memory per event: {bytes_per_event:.1f} bytes")

    # Check against README claim of ~110 bytes/event
    if bytes_per_event <= 110:
        print("TARGET: MEETS README CLAIM: ≤110 bytes/event")
    elif bytes_per_event <= 150:
        print(f"GOOD EFFICIENCY: {bytes_per_event:.1f} bytes/event (≤150)")
    else:
        print(f"WARNING: HIGHER THAN CLAIM: {bytes_per_event:.1f} > 110 bytes/event")

    return bytes_per_event, len(df), memory_used


def test_readme_examples():
    """Test the code examples from README"""
    print("\nTesting README Code Examples")

    file_path = "data/slider_depth/events.txt"
    if not Path(file_path).exists():
        print(f"FAIL: Test file not found: {file_path}")
        return

    try:
        # Test basic loading
        lf = evlib.load_events(file_path)
        df = lf.collect()
        print(f"PASS: Basic loading: {len(df):,} events")

        # Test filtering
        filtered = lf.filter(
            (pl.col("timestamp").dt.total_microseconds() / 1_000_000 > 1.0) & (pl.col("polarity") == 1)
        ).collect()
        print(f"PASS: Filtering: {len(filtered):,} events")

        # Test analysis
        stats = (
            lf.group_by("polarity")
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("x").mean().alias("mean_x"),
                    pl.col("y").mean().alias("mean_y"),
                ]
            )
            .collect()
        )
        print(f"PASS: Analysis: {len(stats)} polarity groups")

        # Test format detection
        format_info = evlib.detect_format(file_path)
        print(f"PASS: Format detection: {format_info[0]}")

    except Exception as e:
        print(f"FAIL: Error in README examples: {e}")


def create_performance_plot(results):
    """Create and save performance visualization plot"""
    if not results:
        print("No results to plot")
        return

    # Extract data for plotting
    file_names = []
    loading_speeds = []
    filter_speeds = []
    memory_efficiency = []
    event_counts = []

    for key, value in results.items():
        if "_loading" in key:
            file_name = key.replace("_loading", "").split("/")[-1]
            file_names.append(file_name)
            loading_speeds.append(value[0] / 1_000_000)  # Convert to millions of events/s
            event_counts.append(value[1] / 1_000_000)  # Convert to millions of events
        elif "_filtering" in key:
            filter_speeds.append(value[0] / 1_000_000)  # Convert to millions of events/s
        elif "_memory" in key:
            memory_efficiency.append(value[0])  # bytes per event

    if not file_names:
        print("No data to plot")
        return

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("evlib Performance Benchmarks", fontsize=16, fontweight="bold")

    # Colors for consistency
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    # 1. Loading Speed
    bars1 = ax1.bar(range(len(file_names)), loading_speeds, color=colors[0], alpha=0.8)
    ax1.set_title("Loading Speed", fontweight="bold")
    ax1.set_ylabel("Million Events/Second")
    ax1.set_xticks(range(len(file_names)))

    # Create clear format labels
    format_labels = []
    for name in file_names:
        if ".txt" in name:
            format_labels.append("Text Format")
        elif ".h5" in name or "_td" in name:
            format_labels.append("HDF5 Format")
        elif ".raw" in name:
            format_labels.append("RAW Binary")
        else:
            format_labels.append(name)

    ax1.set_xticklabels(format_labels, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, speed in zip(bars1, loading_speeds):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{speed:.1f}M",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Filter Speed
    if filter_speeds:
        bars2 = ax2.bar(range(len(file_names)), filter_speeds, color=colors[1], alpha=0.8)
        ax2.set_title("Filter Speed", fontweight="bold")
        ax2.set_ylabel("Million Events/Second")
        ax2.set_xticks(range(len(file_names)))
        ax2.set_xticklabels(format_labels, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, speed in zip(bars2, filter_speeds):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{speed:.0f}M",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # 3. Memory Efficiency
    if memory_efficiency:
        bars3 = ax3.bar(range(len(file_names)), memory_efficiency, color=colors[2], alpha=0.8)
        ax3.set_title("Memory Efficiency", fontweight="bold")
        ax3.set_ylabel("Bytes per Event")
        ax3.set_xticks(range(len(file_names)))
        ax3.set_xticklabels(format_labels, rotation=45, ha="right")
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, memory in zip(bars3, memory_efficiency):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{memory:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # 4. Dataset Size vs Performance
    if event_counts and loading_speeds:
        ax4.scatter(event_counts, loading_speeds, s=100, c=colors[0], alpha=0.8, edgecolors="black")
        ax4.set_title("Dataset Size vs Loading Performance", fontweight="bold")
        ax4.set_xlabel("Dataset Size (Million Events)")
        ax4.set_ylabel("Loading Speed (Million Events/Second)")
        ax4.grid(True, alpha=0.3)

        # Add labels for each point
        for i, label in enumerate(format_labels):
            ax4.annotate(
                label,
                (event_counts[i], loading_speeds[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                ha="left",
            )

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_path = Path("docs/performance_benchmark.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPerformance plot saved to: {output_path}")

    # Also save to root for README
    root_path = Path("performance_benchmark.png")
    plt.savefig(root_path, dpi=300, bbox_inches="tight")
    print(f"Performance plot also saved to: {root_path}")

    plt.close()


def main():
    print("PERFORMANCE: README PERFORMANCE VERIFICATION")
    print("=" * 50)
    print("This script validates the performance claims in README.md")

    # Test files to benchmark
    test_files = [
        "data/slider_depth/events.txt",
        "data/eTram/h5/val_2/val_night_011_td.h5",
        "data/eTram/raw/val_2/val_night_011.raw",
    ]

    results = {}

    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n{'='*60}")
            print(f"TESTING: {test_file}")
            print(f"{'='*60}")

            # Benchmark loading speed
            load_result = benchmark_loading_speed(test_file)
            if load_result:
                results[f"{test_file}_loading"] = load_result

            # Benchmark filter speed
            filter_result = benchmark_filter_speed(test_file)
            if filter_result:
                results[f"{test_file}_filtering"] = filter_result

            # Benchmark memory efficiency
            memory_result = benchmark_memory_efficiency(test_file)
            if memory_result:
                results[f"{test_file}_memory"] = memory_result
        else:
            print(f"\nFAIL: Skipping {test_file} (not found)")

    # Test README examples
    test_readme_examples()

    # Summary
    print("\nTREND: PERFORMANCE SUMMARY")
    print("=" * 50)

    if results:
        loading_speeds = [r[0] for k, r in results.items() if "_loading" in k]
        filter_speeds = [r[0] for k, r in results.items() if "_filtering" in k]
        memory_efficiencies = [r[0] for k, r in results.items() if "_memory" in k]

        if loading_speeds:
            avg_loading = sum(loading_speeds) / len(loading_speeds)
            print(f"STATS: Average loading speed: {avg_loading:,.0f} events/s")
            print(
                f"TARGET: README claim: ≥600k events/s - {'PASS: MET' if avg_loading >= 600000 else 'FAIL: NOT MET'}"
            )

        if filter_speeds:
            avg_filtering = sum(filter_speeds) / len(filter_speeds)
            print(f"ANALYSIS: Average filter speed: {avg_filtering:,.0f} events/s")
            print(
                f"TARGET: README claim: ≥400M events/s - {'PASS: MET' if avg_filtering >= 400_000_000 else 'FAIL: NOT MET'}"
            )

        if memory_efficiencies:
            avg_memory = sum(memory_efficiencies) / len(memory_efficiencies)
            print(f"MEMORY: Average memory efficiency: {avg_memory:.1f} bytes/event")
            print(
                f"TARGET: README claim: ~110 bytes/event - {'PASS: MET' if avg_memory <= 110 else 'FAIL: NOT MET'}"
            )
    else:
        print("FAIL: No test files found for benchmarking")

    print("\nVerification complete!")

    # Generate performance visualization
    create_performance_plot(results)


if __name__ == "__main__":
    main()
