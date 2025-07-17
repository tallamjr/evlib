#!/usr/bin/env python3
"""
Example: RVT Stacked Histogram Replacement with evlib

This demonstrates how to replace RVT's PyTorch-based stacked histogram preprocessing
with evlib's faster Polars-based implementation using the same configuration.
"""

import time

import numpy as np

import evlib.representations as evr


def example_rvt_replacement():
    """
    Example showing RVT stacked histogram replacement with typical RVT configuration.

    RVT's typical configuration:
    - bins=10 (nbins)
    - dt=50ms (window duration)
    - count_cutoff=10 (much lower than default 255)
    - fastmode=True (uint8 output)
    """

    # Test with available data files
    test_files = [
        "data/slider_depth/events.txt",  # Small file for quick testing
        "data/eTram/h5/val_2/val_night_011_td.h5",  # Medium HDF5 file
        # "data/eTram/h5/val_2/val_night_007_td.h5",  # Large file (456MB)
    ]

    print("=== RVT Stacked Histogram Replacement Example ===")
    print()

    for events_path in test_files:
        try:
            print(f"Processing: {events_path}")

            # === RVT's typical configuration ===
            # Based on RVT's conf_preprocess/representation/stacked_hist.yaml
            rvt_config = {
                "height": 240,  # Gen1 sensor height (or 720 for Gen4)
                "width": 346,  # Gen1 sensor width (or 1280 for Gen4)
                "nbins": 10,  # RVT's standard: 10 temporal bins
                "window_duration_ms": 50.0,  # RVT's standard: 50ms windows (dt=50)
                "count_cutoff": 10,  # RVT's preference: much lower than 255
                "stride_ms": None,  # Non-overlapping windows (stride = duration)
            }

            print("  RVT Configuration:")
            print(f"    - Bins: {rvt_config['nbins']}")
            print(f"    - Window duration: {rvt_config['window_duration_ms']}ms")
            print(f"    - Count cutoff: {rvt_config['count_cutoff']}")
            print(f"    - Resolution: {rvt_config['height']}x{rvt_config['width']}")
            print()

            # === Method 1: Direct stacked histogram creation ===
            print("  Method 1: Direct stacked histogram (like RVT)")
            start_time = time.time()

            stacked_hist = evr.create_stacked_histogram(events_path, **rvt_config)

            processing_time = time.time() - start_time

            print(f"    ✓ Shape: {stacked_hist.shape}")
            print(f"    ✓ Data type: {stacked_hist.dtype}")
            print(f"    ✓ Processing time: {processing_time:.2f}s")
            print(f"    ✓ Memory usage: {stacked_hist.nbytes / 1024 / 1024:.1f} MB")
            print()

            # === Method 2: High-level API (easiest RVT replacement) ===
            print("  Method 2: High-level API (drop-in RVT replacement)")
            start_time = time.time()

            preprocessed_data = evr.preprocess_for_detection(
                events_path, representation="stacked_histogram", **rvt_config
            )

            processing_time = time.time() - start_time

            print(f"    ✓ Shape: {preprocessed_data.shape}")
            print(f"    ✓ Processing time: {processing_time:.2f}s")
            print()

            # === Validation: Check RVT compatibility ===
            print("  RVT Compatibility Check:")

            # Check output format
            num_windows, num_channels, height, width = stacked_hist.shape
            expected_channels = 2 * rvt_config["nbins"]  # 2 polarities × nbins

            assert (
                num_channels == expected_channels
            ), f"Expected {expected_channels} channels, got {num_channels}"
            assert stacked_hist.dtype == np.uint8, f"Expected uint8, got {stacked_hist.dtype}"
            assert height == rvt_config["height"], f"Height mismatch: {height} != {rvt_config['height']}"
            assert width == rvt_config["width"], f"Width mismatch: {width} != {rvt_config['width']}"

            print(
                f"    ✓ Channel layout: {num_channels} channels ({rvt_config['nbins']} bins × 2 polarities)"
            )
            print(f"    ✓ Data type: {stacked_hist.dtype} (RVT fastmode compatible)")
            print(f"    ✓ Value range: [{stacked_hist.min()}, {stacked_hist.max()}]")
            print(f"    ✓ Non-zero voxels: {np.count_nonzero(stacked_hist):,}")

            print("    ✓ All RVT compatibility checks passed!")
            print()

            # === Show actual RVT command equivalent ===
            print("  Equivalent RVT Configuration:")
            print(
                f"    stacked_histogram_dt={rvt_config['window_duration_ms']:.0f}_nbins={rvt_config['nbins']}"
            )
            print(f"    height={rvt_config['height']}, width={rvt_config['width']}")
            print(f"    count_cutoff={rvt_config['count_cutoff']}, fastmode=True")
            print()

            break  # Successfully processed one file

        except FileNotFoundError:
            print(f"  File not found: {events_path}")
            continue
        except Exception as e:
            print(f"  ✗ Error processing {events_path}: {e}")
            continue

    print("=== Performance Comparison ===")

    # Benchmark against RVT
    try:
        results = evr.benchmark_vs_rvt(events_path)
        print(f"✓ evlib Polars: {results['polars_time']:.2f}s")
        print(f"✓ Estimated RVT: {results['estimated_rvt_time']:.2f}s")
        print(f"✓ Speedup: {results['speedup']:.1f}x faster")
        print(f"✓ Output shape: {results['output_shape']}")
        print(f"✓ Memory usage: {results['memory_mb']:.1f} MB")
    except Exception as e:
        print(f"Benchmark failed: {e}")

    print()
    print("=== Migration Guide ===")
    print()
    print("To replace RVT preprocessing in your code:")
    print()
    print("# Before (RVT):")
    print("from rvt.representations import StackedHistogram")
    print("stacker = StackedHistogram(bins=10, height=240, width=346, count_cutoff=10)")
    print("hist = stacker(events)")
    print()
    print("# After (evlib):")
    print("import evlib.representations as evr")
    print("hist = evr.create_stacked_histogram(")
    print("    'events.h5',")
    print("    height=240, width=346,")
    print("    nbins=10, window_duration_ms=50, count_cutoff=10")
    print(")")
    print()
    print("# Or use high-level API:")
    print("hist = evr.preprocess_for_detection(")
    print("    'events.h5', representation='stacked_histogram',")
    print("    height=240, width=346, nbins=10, window_duration_ms=50")
    print(")")


def example_gen4_configuration():
    """
    Example for Gen4 sensors (1280x720) with RVT-style configuration.
    Often downsampled by 2 for efficiency.
    """

    print("=== Gen4 Sensor Configuration (RVT-style) ===")

    # Gen4 configuration (often downsampled by 2)
    gen4_config = {
        "height": 720 // 2,  # 360 (downsampled)
        "width": 1280 // 2,  # 640 (downsampled)
        "nbins": 10,  # RVT standard
        "window_duration_ms": 50.0,  # RVT standard
        "count_cutoff": 10,  # RVT standard
    }

    print("Gen4 sensor configuration:")
    print("  - Original resolution: 720x1280")
    print(f"  - Downsampled: {gen4_config['height']}x{gen4_config['width']}")
    print("  - RVT config: dt=50ms, nbins=10, count_cutoff=10")
    print()

    # Example with Gen4 data (if available)
    gen4_files = [
        "data/gen4_1mpx_original/val/val_night_007.h5",
        "data/eTram/h5/val_2/val_night_007_td.h5",  # This is Gen4 data
    ]

    for events_path in gen4_files:
        try:
            print(f"Processing Gen4 data: {events_path}")

            hist = evr.create_stacked_histogram(events_path, **gen4_config)

            print(f"  ✓ Generated shape: {hist.shape}")
            print("  ✓ Expected for Gen4: (num_windows, 20, 360, 640)")
            print()
            break

        except FileNotFoundError:
            print(f"  Gen4 file not found: {events_path}")
            continue
        except Exception as e:
            print(f"  ✗ Error with Gen4 data: {e}")
            continue


if __name__ == "__main__":
    # Run the main example
    example_rvt_replacement()

    print("=" * 60)

    # Run Gen4 example
    example_gen4_configuration()

    print("=" * 60)
    print("✓ RVT replacement examples completed!")
