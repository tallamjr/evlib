#!/usr/bin/env python3
"""
Benchmark for evlib comparing the Rust-backed implementation
to a simple pure Python implementation

This example demonstrates:
1. Creating large event arrays
2. Measuring performance of basic operations
3. Comparing Rust implementation with Python
4. Comparing single-core vs multi-core performance
"""
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from evlib import augmentation, core


def python_events_to_block(xs, ys, ts, ps):
    """Pure Python implementation of events_to_block"""
    n = len(xs)
    block = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        block[i, 0] = float(xs[i])
        block[i, 1] = float(ys[i])
        block[i, 2] = ts[i]
        block[i, 3] = float(ps[i])
    return block


def python_events_to_block_chunk(chunk_data):
    """Process a chunk of events for parallel execution"""
    xs_chunk, ys_chunk, ts_chunk, ps_chunk = chunk_data
    return python_events_to_block(xs_chunk, ys_chunk, ts_chunk, ps_chunk)


def python_events_to_block_parallel(xs, ys, ts, ps, num_cores):
    """Parallel Python implementation of events_to_block"""
    n = len(xs)
    chunk_size = n // num_cores
    chunks = []

    # Create chunks of data
    for i in range(num_cores):
        start = i * chunk_size
        end = start + chunk_size if i < num_cores - 1 else n
        chunks.append((xs[start:end], ys[start:end], ts[start:end], ps[start:end]))

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(python_events_to_block_chunk, chunks))

    # Combine results
    return np.vstack(results)


def python_add_random_events(xs, ys, ts, ps, to_add):
    """Pure Python implementation of add_random_events"""
    max_x = max(xs)
    max_y = max(ys)
    min_ts = min(ts)
    max_ts = max(ts)

    # Generate random events
    new_xs = np.random.randint(0, max_x + 1, size=to_add, dtype=np.int64)
    new_ys = np.random.randint(0, max_y + 1, size=to_add, dtype=np.int64)
    new_ts = np.random.uniform(min_ts, max_ts, size=to_add)
    new_ps = np.random.choice([-1, 1], size=to_add).astype(np.int64)

    # Merge with original events
    merged_xs = np.concatenate([xs, new_xs])
    merged_ys = np.concatenate([ys, new_ys])
    merged_ts = np.concatenate([ts, new_ts])
    merged_ps = np.concatenate([ps, new_ps])

    # Sort by timestamp
    indices = np.argsort(merged_ts)
    sorted_xs = merged_xs[indices]
    sorted_ys = merged_ys[indices]
    sorted_ts = merged_ts[indices]
    sorted_ps = merged_ps[indices]

    return sorted_xs, sorted_ys, sorted_ts, sorted_ps


def _generate_random_events_chunk(max_x, max_y, min_ts, max_ts, chunk_size):
    """Generate random events for a chunk"""
    new_xs = np.random.randint(0, max_x + 1, size=chunk_size, dtype=np.int64)
    new_ys = np.random.randint(0, max_y + 1, size=chunk_size, dtype=np.int64)
    new_ts = np.random.uniform(min_ts, max_ts, size=chunk_size)
    new_ps = np.random.choice([-1, 1], size=chunk_size).astype(np.int64)
    return new_xs, new_ys, new_ts, new_ps


def python_add_random_events_parallel(xs, ys, ts, ps, to_add, num_cores):
    """Parallel Python implementation of add_random_events"""
    max_x = max(xs)
    max_y = max(ys)
    min_ts = min(ts)
    max_ts = max(ts)

    # Divide the work
    chunk_size = to_add // num_cores

    # Create partial function with fixed parameters
    generate_chunk = partial(_generate_random_events_chunk, max_x, max_y, min_ts, max_ts)

    # Generate chunks in parallel
    chunks = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        chunk_sizes = [chunk_size] * (num_cores - 1) + [to_add - (num_cores - 1) * chunk_size]
        chunks = list(executor.map(generate_chunk, chunk_sizes))

    # Combine chunks
    new_xs = np.concatenate([chunk[0] for chunk in chunks])
    new_ys = np.concatenate([chunk[1] for chunk in chunks])
    new_ts = np.concatenate([chunk[2] for chunk in chunks])
    new_ps = np.concatenate([chunk[3] for chunk in chunks])

    # Merge with original events
    merged_xs = np.concatenate([xs, new_xs])
    merged_ys = np.concatenate([ys, new_ys])
    merged_ts = np.concatenate([ts, new_ts])
    merged_ps = np.concatenate([ps, new_ps])

    # Sort by timestamp
    indices = np.argsort(merged_ts)
    sorted_xs = merged_xs[indices]
    sorted_ys = merged_ys[indices]
    sorted_ts = merged_ts[indices]
    sorted_ps = merged_ps[indices]

    return sorted_xs, sorted_ys, sorted_ts, sorted_ps


def python_flip_events_x(xs, ys, ts, ps, sensor_resolution):
    """Pure Python implementation of flip_events_x"""
    flipped_xs = sensor_resolution[1] - 1 - xs
    return flipped_xs, ys, ts, ps


def python_flip_events_x_chunk(chunk_data):
    """Process a chunk for parallel flip_events_x"""
    xs_chunk, ys_chunk, ts_chunk, ps_chunk, sensor_resolution = chunk_data
    return python_flip_events_x(xs_chunk, ys_chunk, ts_chunk, ps_chunk, sensor_resolution)


def python_flip_events_x_parallel(xs, ys, ts, ps, sensor_resolution, num_cores):
    """Parallel Python implementation of flip_events_x"""
    n = len(xs)
    chunk_size = n // num_cores
    chunks = []

    # Create chunks of data
    for i in range(num_cores):
        start = i * chunk_size
        end = start + chunk_size if i < num_cores - 1 else n
        chunks.append(
            (
                xs[start:end],
                ys[start:end],
                ts[start:end],
                ps[start:end],
                sensor_resolution,
            )
        )

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(python_flip_events_x_chunk, chunks))

    # Combine results
    flipped_xs = np.concatenate([res[0] for res in results])
    flipped_ys = np.concatenate([res[1] for res in results])
    flipped_ts = np.concatenate([res[2] for res in results])
    flipped_ps = np.concatenate([res[3] for res in results])

    return flipped_xs, flipped_ys, flipped_ts, flipped_ps


def benchmark_function(func, *args, iterations=10):
    """Benchmark a function by running it multiple times"""
    times = []
    for _ in range(iterations):
        start = time.time()
        result = func(*args)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time, result


def main():
    print("evlib benchmark")

    # Number of cores for multi-core benchmarks
    num_cores = 10
    print(f"System has {multiprocessing.cpu_count()} CPU cores available")
    print(f"Using {num_cores} cores for multi-core benchmarks")

    # Create large event arrays for benchmarking
    num_events = 100000
    print(f"Benchmarking with {num_events} events")

    # Create random event data
    xs = np.random.randint(0, 640, size=num_events, dtype=np.int64)
    ys = np.random.randint(0, 480, size=num_events, dtype=np.int64)
    ts = np.random.uniform(0, 1, size=num_events)
    ps = np.random.choice([-1, 1], size=num_events).astype(np.int64)

    print("\n=== Single-Core Benchmarks ===")

    # Store benchmark results for summary table
    benchmark_results = {}

    # Benchmark events_to_block (single-core)
    print("\nBenchmarking events_to_block (single-core)...")
    py_time_block_single, py_result = benchmark_function(python_events_to_block, xs, ys, ts, ps)
    rust_time_block_single, rust_result = benchmark_function(core.events_to_block_py, xs, ys, ts, ps)

    print(f"Python implementation: {py_time_block_single:.6f} seconds")
    print(f"Rust implementation: {rust_time_block_single:.6f} seconds")
    print(f"Speedup: {py_time_block_single / rust_time_block_single:.2f}x")

    # Verify results match
    py_sum = np.sum(py_result)
    rust_sum = np.sum(rust_result)
    print(f"Results match: {np.isclose(py_sum, rust_sum)}")

    # Store results
    benchmark_results["events_to_block"] = (
        py_time_block_single,
        None,
        rust_time_block_single,
    )

    # Benchmark add_random_events (single-core)
    print("\nBenchmarking add_random_events (single-core)...")
    to_add = 10000

    py_time_add_single, (py_xs, py_ys, py_ts, py_ps) = benchmark_function(
        python_add_random_events, xs, ys, ts, ps, to_add
    )

    rust_time_add_single, (rust_xs, rust_ys, rust_ts, rust_ps) = benchmark_function(
        augmentation.add_random_events_py, xs, ys, ts, ps, to_add
    )

    print(f"Python implementation: {py_time_add_single:.6f} seconds")
    print(f"Rust implementation: {rust_time_add_single:.6f} seconds")
    print(f"Speedup: {py_time_add_single / rust_time_add_single:.2f}x")
    print(f"Output length: Python={len(py_xs)}, Rust={len(rust_xs)}")

    # Store results
    benchmark_results["add_random_events"] = (
        py_time_add_single,
        None,
        rust_time_add_single,
    )

    # Benchmark flip_events_x (single-core)
    print("\nBenchmarking flip_events_x (single-core)...")
    sensor_resolution = (480, 640)

    py_time_flip_single, py_flip_result = benchmark_function(
        python_flip_events_x, xs, ys, ts, ps, sensor_resolution
    )

    rust_time_flip_single, rust_flip_result = benchmark_function(
        augmentation.flip_events_x, xs, ys, ts, ps, sensor_resolution
    )

    print(f"Python implementation: {py_time_flip_single:.6f} seconds")
    print(f"Rust implementation: {rust_time_flip_single:.6f} seconds")
    print(f"Speedup: {py_time_flip_single / rust_time_flip_single:.2f}x")

    # Store results
    benchmark_results["flip_events_x"] = (
        py_time_flip_single,
        None,
        rust_time_flip_single,
    )

    print("\n=== Multi-Core Benchmarks ({} cores) ===".format(num_cores))

    # Benchmark events_to_block (multi-core)
    print("\nBenchmarking events_to_block (multi-core)...")
    py_time_block_multi, py_result_multi = benchmark_function(
        python_events_to_block_parallel, xs, ys, ts, ps, num_cores
    )

    print(f"Python multi-core implementation: {py_time_block_multi:.6f} seconds")
    print(f"Python speedup (multi-core vs single-core): {py_time_block_single / py_time_block_multi:.2f}x")
    print(f"Rust implementation (single-core): {rust_time_block_single:.6f} seconds")
    print(
        f"Speedup (Rust single-core vs Python multi-core): {py_time_block_multi / rust_time_block_single:.2f}x"
    )

    # Verify multi-core results match
    py_sum_multi = np.sum(py_result_multi)
    print(f"Multi-core results match single-core: {np.isclose(py_sum, py_sum_multi)}")

    # Update benchmark results
    benchmark_results["events_to_block"] = (
        py_time_block_single,
        py_time_block_multi,
        rust_time_block_single,
    )

    # Benchmark add_random_events (multi-core)
    print("\nBenchmarking add_random_events (multi-core)...")
    py_time_add_multi, (py_xs_multi, py_ys_multi, py_ts_multi, py_ps_multi) = benchmark_function(
        python_add_random_events_parallel, xs, ys, ts, ps, to_add, num_cores
    )

    print(f"Python multi-core implementation: {py_time_add_multi:.6f} seconds")
    print(f"Python speedup (multi-core vs single-core): {py_time_add_single / py_time_add_multi:.2f}x")
    print(f"Rust implementation (single-core): {rust_time_add_single:.6f} seconds")
    print(f"Speedup (Rust single-core vs Python multi-core): {py_time_add_multi / rust_time_add_single:.2f}x")
    print(f"Output length: Python multi-core={len(py_xs_multi)}")

    # Update benchmark results
    benchmark_results["add_random_events"] = (
        py_time_add_single,
        py_time_add_multi,
        rust_time_add_single,
    )

    # Benchmark flip_events_x (multi-core)
    print("\nBenchmarking flip_events_x (multi-core)...")
    py_time_flip_multi, py_flip_multi = benchmark_function(
        python_flip_events_x_parallel, xs, ys, ts, ps, sensor_resolution, num_cores
    )

    print(f"Python multi-core implementation: {py_time_flip_multi:.6f} seconds")
    print(f"Python speedup (multi-core vs single-core): {py_time_flip_single / py_time_flip_multi:.2f}x")
    print(f"Rust implementation (single-core): {rust_time_flip_single:.6f} seconds")
    print(
        f"Speedup (Rust single-core vs Python multi-core): {py_time_flip_multi / rust_time_flip_single:.2f}x"
    )

    # Update benchmark results
    benchmark_results["flip_events_x"] = (
        py_time_flip_single,
        py_time_flip_multi,
        rust_time_flip_single,
    )

    # Summary table
    print("\n=== Summary ===")
    print(
        "Operation\t| Python (1 core)\t| Python (10 cores)\t| Rust (1 core)\t| Rust vs Py (1 core)\t| Rust vs Py (10 cores)"
    )
    print("-" * 110)

    # Print summary table
    for op, (py_single, py_multi, rust_single) in benchmark_results.items():
        print(
            f"{op}\t| {py_single:.6f} s\t| {py_multi:.6f} s\t| {rust_single:.6f} s\t| {py_single / rust_single:.2f}x\t| {py_multi / rust_single:.2f}x"
        )


if __name__ == "__main__":
    main()
