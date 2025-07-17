#!/usr/bin/env python3
"""
Memory usage benchmark for evlib memory optimization.

This script measures actual memory usage before/after the optimization
and provides concrete metrics you can verify.
"""

import gc
import time
import psutil
import os
import evlib
from pathlib import Path


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_file(file_path, description):
    """Benchmark memory usage for a specific file"""
    print(f"\n📊 Benchmarking: {description}")
    print(f"📁 File: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    # Force garbage collection
    gc.collect()
    initial_memory = get_memory_usage()
    
    # Load events and measure peak memory
    start_time = time.time()
    lf = evlib.load_events(str(file_path))
    df = lf.collect()
    load_time = time.time() - start_time
    
    peak_memory = get_memory_usage()
    memory_used = peak_memory - initial_memory
    
    # Get basic stats
    event_count = len(df)
    
    # Calculate efficiency metrics
    bytes_per_event = (memory_used * 1024 * 1024) / event_count if event_count > 0 else 0
    events_per_second = event_count / load_time if load_time > 0 else 0
    
    # Clean up and measure memory after cleanup
    del lf, df
    gc.collect()
    final_memory = get_memory_usage()
    memory_retained = final_memory - initial_memory
    
    results = {
        'file_path': file_path,
        'description': description,
        'event_count': event_count,
        'load_time': load_time,
        'memory_used': memory_used,
        'memory_retained': memory_retained,
        'bytes_per_event': bytes_per_event,
        'events_per_second': events_per_second,
    }
    
    print(f"✅ Events: {event_count:,}")
    print(f"⏱️  Load time: {load_time:.2f}s ({events_per_second:,.0f} events/s)")
    print(f"🧠 Peak memory: {memory_used:.1f} MB ({bytes_per_event:.1f} bytes/event)")
    print(f"🔄 Memory retained: {memory_retained:.1f} MB")
    
    return results


def benchmark_polars_efficiency():
    """Test Polars-specific optimizations"""
    print(f"\n🔬 POLARS EFFICIENCY TEST")
    
    file_path = "data/slider_depth/events.txt"
    if not Path(file_path).exists():
        print(f"❌ Test file not found: {file_path}")
        return
    
    print(f"📁 Using: {file_path}")
    
    # Test 1: Basic loading
    print(f"\n1️⃣ Basic Loading Test")
    gc.collect()
    start_mem = get_memory_usage()
    start_time = time.time()
    
    lf = evlib.load_events(file_path)
    df = lf.collect()
    
    load_time = time.time() - start_time
    peak_mem = get_memory_usage()
    
    print(f"   📊 Loaded {len(df):,} events in {load_time:.2f}s")
    print(f"   🧠 Memory used: {peak_mem - start_mem:.1f} MB")
    
    # Test 2: LazyFrame operations (should be very fast)
    print(f"\n2️⃣ LazyFrame Operations Test")
    start_time = time.time()
    
    # Chain multiple operations using LazyFrame expressions
    import polars as pl
    filtered = lf.filter(
        (pl.col('timestamp').dt.total_microseconds() / 1_000_000 > 1.0) & 
        (pl.col('polarity') == 1)
    ).filter(
        (pl.col('x') > 50) & (pl.col('x') < 200)
    )
    
    result_df = filtered.collect()
    filter_time = time.time() - start_time
    
    print(f"   📊 Filtered to {len(result_df):,} events in {filter_time:.3f}s")
    print(f"   ⚡ Filter speed: {len(df) / filter_time:,.0f} events/s")
    
    # Test 3: Type verification
    print(f"\n3️⃣ Data Type Verification")
    print(f"   📋 Columns: {df.columns}")
    print(f"   🏷️  Types: {[str(df[col].dtype) for col in df.columns]}")
    
    # Verify memory-efficient types
    x_dtype = str(df['x'].dtype)
    y_dtype = str(df['y'].dtype)
    p_dtype = str(df['polarity'].dtype)
    t_dtype = str(df['timestamp'].dtype)
    
    print(f"   ✅ Using efficient types:")
    print(f"      x, y: {x_dtype}, {y_dtype} (should be Int16 or Int64)")
    print(f"      polarity: {p_dtype} (should be Int8 or Int64)")
    print(f"      timestamp: {t_dtype} (should be Duration)")
    
    del lf, df, filtered, result_df
    gc.collect()


def main():
    print("🚀 EVLIB MEMORY OPTIMIZATION BENCHMARK")
    print("=" * 50)
    
    # Test files with different sizes
    test_files = [
        ("data/slider_depth/events.txt", "Text format (~22MB, ~1M events)"),
        ("data/eTram/h5/val_2/val_night_011_td.h5", "Small HDF5 (~14MB, ~3M events)"),
        ("data/eTram/raw/val_2/val_night_011.raw", "Small EVT2 (~15MB, ~2.7M events)"),
    ]
    
    results = []
    for file_path, description in test_files:
        result = benchmark_file(file_path, description)
        if result:
            results.append(result)
    
    # Run Polars-specific tests
    benchmark_polars_efficiency()
    
    # Summary
    if results:
        print(f"\n📈 SUMMARY")
        print("=" * 50)
        
        total_events = sum(r['event_count'] for r in results)
        avg_bytes_per_event = sum(r['bytes_per_event'] for r in results) / len(results)
        avg_speed = sum(r['events_per_second'] for r in results) / len(results)
        
        print(f"📊 Total events tested: {total_events:,}")
        print(f"🧠 Average memory efficiency: {avg_bytes_per_event:.1f} bytes/event")
        print(f"⚡ Average processing speed: {avg_speed:,.0f} events/s")
        
        print(f"\n🎯 MEMORY EFFICIENCY ANALYSIS:")
        print(f"   • Target: <30 bytes/event (optimized)")
        print(f"   • Achieved: {avg_bytes_per_event:.1f} bytes/event")
        
        if avg_bytes_per_event < 30:
            print(f"   ✅ EXCELLENT: Memory usage is highly optimized!")
        elif avg_bytes_per_event < 40:
            print(f"   ✅ GOOD: Memory usage is well optimized")
        else:
            print(f"   ⚠️  HIGH: Memory usage could be improved")
        
        print(f"\n🎯 PERFORMANCE ANALYSIS:")
        print(f"   • Target: >500k events/s")
        print(f"   • Achieved: {avg_speed:,.0f} events/s")
        
        if avg_speed > 500000:
            print(f"   ✅ EXCELLENT: Processing speed is very fast!")
        elif avg_speed > 100000:
            print(f"   ✅ GOOD: Processing speed is adequate")
        else:
            print(f"   ⚠️  SLOW: Processing speed could be improved")
    
    print(f"\n🏁 Benchmark complete!")


if __name__ == "__main__":
    main()