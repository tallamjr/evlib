#!/usr/bin/env python3
"""
Benchmark for evlib comparing the Rust-backed implementation
to a simple pure Python implementation

This example demonstrates:
1. Creating large event arrays
2. Measuring performance of basic operations
3. Comparing Rust implementation with Python
"""
import time

import numpy as np

import evlib


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


def python_add_random_events(xs, ys, ts, ps, to_add):
    """Pure Python implementation of add_random_events"""
    n = len(xs)
    max_x = max(xs)
    max_y = max(ys)
    min_ts = min(ts)
    max_ts = max(ts)

    # Generate random events
    new_xs = np.random.randint(0, max_x + 1, size=to_add, dtype=np.int64)
    new_ys = np.random.randint(0, max_y + 1, size=to_add, dtype=np.int64)
    new_ts = np.random.uniform(min_ts, max_ts, size=to_add)
    new_ps = np.random.choice([-1, 1], size=to_add, dtype=np.int64)

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

    # Create large event arrays for benchmarking
    num_events = 100000
    print(f"Benchmarking with {num_events} events")

    # Create random event data
    xs = np.random.randint(0, 640, size=num_events, dtype=np.int64)
    ys = np.random.randint(0, 480, size=num_events, dtype=np.int64)
    ts = np.random.uniform(0, 1, size=num_events)
    ps = np.random.choice([-1, 1], size=num_events)

    # Benchmark events_to_block
    print("\nBenchmarking events_to_block...")
    py_time, py_result = benchmark_function(python_events_to_block, xs, ys, ts, ps)
    rust_time, rust_result = benchmark_function(evlib.events_to_block, xs, ys, ts, ps)

    print(f"Python implementation: {py_time:.6f} seconds")
    print(f"Rust implementation: {rust_time:.6f} seconds")
    print(f"Speedup: {py_time / rust_time:.2f}x")

    # Verify results match
    py_sum = np.sum(py_result)
    rust_sum = np.sum(rust_result)
    print(f"Results match: {np.isclose(py_sum, rust_sum)}")

    # Benchmark add_random_events
    print("\nBenchmarking add_random_events...")
    to_add = 10000

    py_time, (py_xs, py_ys, py_ts, py_ps) = benchmark_function(
        python_add_random_events, xs, ys, ts, ps, to_add
    )

    rust_time, (rust_xs, rust_ys, rust_ts, rust_ps) = benchmark_function(
        evlib.add_random_events, xs, ys, ts, ps, to_add
    )

    print(f"Python implementation: {py_time:.6f} seconds")
    print(f"Rust implementation: {rust_time:.6f} seconds")
    print(f"Speedup: {py_time / rust_time:.2f}x")
    print(f"Output length: Python={len(py_xs)}, Rust={len(rust_xs)}")

    # Benchmark flip_events_x
    print("\nBenchmarking flip_events_x...")
    sensor_resolution = (480, 640)

    py_time, _ = benchmark_function(
        python_flip_events_x, xs, ys, ts, ps, sensor_resolution
    )

    rust_time, _ = benchmark_function(
        evlib.flip_events_x, xs, ys, ts, ps, sensor_resolution
    )

    print(f"Python implementation: {py_time:.6f} seconds")
    print(f"Rust implementation: {rust_time:.6f} seconds")
    print(f"Speedup: {py_time / rust_time:.2f}x")


if __name__ == "__main__":
    main()
