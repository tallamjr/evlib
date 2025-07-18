#!/usr/bin/env python3
"""
Test memory mapping concept for large HDF5 files.
"""

import h5py
import time
import polars as pl
from pathlib import Path


def memory_mapped_load(file_path, chunk_size=100_000):
    """
    Simulate memory-mapped loading by reading in chunks.
    This demonstrates what the Rust implementation would do.
    """
    print(f"MAP: Testing memory-mapped loading with {chunk_size:,} event chunks")

    start_total = time.time()

    # Open file and determine structure
    with h5py.File(file_path, "r") as f:
        # Find the event data (adapt based on file structure)
        datasets = {}
        for key in ["x", "y", "t", "p"]:
            if key in f:
                datasets[key] = f[key]

        if not datasets:
            # Try alternative structures
            for key in f.keys():
                if hasattr(f[key], "shape") and len(f[key].shape) == 1:
                    datasets[key] = f[key]

        if not datasets:
            print("FAIL: Could not find event datasets")
            return None

        # Get total length
        total_events = len(list(datasets.values())[0])
        print(f"DATA: Total events: {total_events:,}")

        # Read in chunks and build Polars DataFrame
        chunks = []

        for i in range(0, total_events, chunk_size):
            end = min(i + chunk_size, total_events)
            chunk_events = end - i

            print(f"   Reading chunk {i//chunk_size + 1}: events {i:,}-{end:,} ({chunk_events:,} events)")

            start_chunk = time.time()

            # Read chunk data
            chunk_data = {}
            for key, dataset in datasets.items():
                chunk_data[key] = dataset[i:end]

            # Convert to Polars DataFrame (this simulates the Rust direct construction)
            chunk_df = pl.DataFrame(chunk_data)
            chunks.append(chunk_df)

            chunk_time = time.time() - start_chunk
            chunk_speed = chunk_events / chunk_time
            print(f"      TIME: Chunk time: {chunk_time:.3f}s ({chunk_speed:,.0f} events/s)")

    # Combine chunks
    print("INFO: Combining chunks...")
    start_combine = time.time()
    final_df = pl.concat(chunks)
    combine_time = time.time() - start_combine

    total_time = time.time() - start_total

    print("PASS: Memory-mapped simulation complete!")
    print(f"TIME: Total time: {total_time:.2f}s")
    print(f"TIME: Combine time: {combine_time:.3f}s")
    print(f"DATA: Final events: {len(final_df):,}")
    print(f"FAST: Overall speed: {len(final_df) / total_time:,.0f} events/s")

    return final_df


def compare_approaches(file_path):
    """Compare current vs memory-mapped approach"""
    print("INFO: COMPARISON: Current vs Memory-Mapped")
    print("=" * 50)

    # Test current approach
    print("\nSTEP1: Current evlib approach:")
    start = time.time()
    import evlib

    lf = evlib.load_events(file_path)
    df_current = lf.collect()
    current_time = time.time() - start

    print(f"   TIME: Time: {current_time:.2f}s")
    print(f"   DATA: Events: {len(df_current):,}")
    print(f"   FAST: Speed: {len(df_current) / current_time:,.0f} events/s")

    # Test memory-mapped simulation
    print("\nSTEP2: Memory-mapped simulation:")
    df_mapped = memory_mapped_load(file_path, chunk_size=50_000)

    if df_mapped is not None:
        # Verify results are similar
        print("\nINFO: Verification:")
        print(f"   Current events: {len(df_current):,}")
        print(f"   Mapped events:  {len(df_mapped):,}")
        print(f"   Match: {'PASS:' if len(df_current) == len(df_mapped) else 'FAIL:'}")


def main():
    file_path = "data/eTram/h5/val_2/val_night_011_td.h5"

    if not Path(file_path).exists():
        print(f"FAIL: File not found: {file_path}")
        return

    compare_approaches(file_path)

    print("\nTIP: MEMORY MAPPING BENEFITS:")
    print("   • Reduced memory usage (no 456MB allocation)")
    print("   • Faster startup (begin processing immediately)")
    print("   • Better scalability (handle 1GB+ files)")
    print("   • OS-level optimization (intelligent caching)")


if __name__ == "__main__":
    main()
