#!/usr/bin/env python3
"""
Performance verification script for README metrics.

This script validates the performance claims made in the README.md file
using available test data.
"""

import evlib
import polars as pl
import time
import os
import psutil
from pathlib import Path


def get_memory_usage_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_loading_speed(file_path):
    """Benchmark loading speed as claimed in README"""
    print(f"\n📊 Benchmarking Loading Speed: {file_path}")

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return None

    start_time = time.time()
    lf = evlib.load_events(file_path)
    df = lf.collect()
    load_time = time.time() - start_time

    events_per_second = len(df) / load_time

    print(f"✅ Loaded {len(df):,} events in {load_time:.2f}s")
    print(f"⚡ Speed: {events_per_second:,.0f} events/s")

    # Check against README claim of 600k+ events/s
    if events_per_second >= 600000:
        print("🎯 MEETS README CLAIM: ≥600k events/s")
    else:
        print(f"⚠️  BELOW README CLAIM: {events_per_second:,.0f} < 600k events/s")

    return events_per_second, len(df), load_time


def benchmark_filter_speed(file_path):
    """Benchmark filter speed as claimed in README"""
    print(f"\n🔍 Benchmarking Filter Speed: {file_path}")

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return None

    lf = evlib.load_events(file_path)
    df = lf.collect()

    # Test filtering speed
    start_time = time.time()
    filtered = lf.filter((pl.col("polarity") == 1) & (pl.col("x") > 50) & (pl.col("x") < 250)).collect()
    filter_time = time.time() - start_time

    events_per_second = len(df) / filter_time

    print(f"✅ Filtered {len(df):,} events to {len(filtered):,} in {filter_time:.4f}s")
    print(f"⚡ Filter speed: {events_per_second:,.0f} events/s")

    # Check against README claim of 400M+ events/s
    if events_per_second >= 400_000_000:
        print("🎯 MEETS README CLAIM: ≥400M events/s")
    elif events_per_second >= 100_000_000:
        print(f"🟡 GOOD PERFORMANCE: {events_per_second:,.0f} events/s (≥100M)")
    else:
        print(f"⚠️  BELOW EXPECTED: {events_per_second:,.0f} < 100M events/s")

    return events_per_second, len(df), filter_time


def benchmark_memory_efficiency(file_path):
    """Benchmark memory efficiency as claimed in README"""
    print(f"\n🧠 Benchmarking Memory Efficiency: {file_path}")

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
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

    print(f"✅ Loaded {len(df):,} events")
    print(f"🧠 Memory used: {memory_used:.1f} MB")
    print(f"📊 Memory per event: {bytes_per_event:.1f} bytes")

    # Check against README claim of ~110 bytes/event
    if bytes_per_event <= 110:
        print("🎯 MEETS README CLAIM: ≤110 bytes/event")
    elif bytes_per_event <= 150:
        print(f"🟡 GOOD EFFICIENCY: {bytes_per_event:.1f} bytes/event (≤150)")
    else:
        print(f"⚠️  HIGHER THAN CLAIM: {bytes_per_event:.1f} > 110 bytes/event")

    return bytes_per_event, len(df), memory_used


def test_readme_examples():
    """Test the code examples from README"""
    print("\n🧪 Testing README Code Examples")

    file_path = "data/slider_depth/events.txt"
    if not Path(file_path).exists():
        print(f"❌ Test file not found: {file_path}")
        return

    try:
        # Test basic loading
        lf = evlib.load_events(file_path)
        df = lf.collect()
        print(f"✅ Basic loading: {len(df):,} events")

        # Test filtering
        filtered = lf.filter(
            (pl.col("timestamp").dt.total_microseconds() / 1_000_000 > 1.0) & (pl.col("polarity") == 1)
        ).collect()
        print(f"✅ Filtering: {len(filtered):,} events")

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
        print(f"✅ Analysis: {len(stats)} polarity groups")

        # Test format detection
        format_info = evlib.detect_format(file_path)
        print(f"✅ Format detection: {format_info[0]}")

    except Exception as e:
        print(f"❌ Error in README examples: {e}")


def main():
    print("🚀 README PERFORMANCE VERIFICATION")
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
            print(f"\n❌ Skipping {test_file} (not found)")

    # Test README examples
    test_readme_examples()

    # Summary
    print("\n📈 PERFORMANCE SUMMARY")
    print("=" * 50)

    if results:
        loading_speeds = [r[0] for k, r in results.items() if "_loading" in k]
        filter_speeds = [r[0] for k, r in results.items() if "_filtering" in k]
        memory_efficiencies = [r[0] for k, r in results.items() if "_memory" in k]

        if loading_speeds:
            avg_loading = sum(loading_speeds) / len(loading_speeds)
            print(f"📊 Average loading speed: {avg_loading:,.0f} events/s")
            print(f"🎯 README claim: ≥600k events/s - {'✅ MET' if avg_loading >= 600000 else '❌ NOT MET'}")

        if filter_speeds:
            avg_filtering = sum(filter_speeds) / len(filter_speeds)
            print(f"🔍 Average filter speed: {avg_filtering:,.0f} events/s")
            print(
                f"🎯 README claim: ≥400M events/s - {'✅ MET' if avg_filtering >= 400_000_000 else '❌ NOT MET'}"
            )

        if memory_efficiencies:
            avg_memory = sum(memory_efficiencies) / len(memory_efficiencies)
            print(f"🧠 Average memory efficiency: {avg_memory:.1f} bytes/event")
            print(f"🎯 README claim: ~110 bytes/event - {'✅ MET' if avg_memory <= 110 else '❌ NOT MET'}")
    else:
        print("❌ No test files found for benchmarking")

    print("\n🏁 Verification complete!")


if __name__ == "__main__":
    main()
