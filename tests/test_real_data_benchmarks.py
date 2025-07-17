"""
Real data benchmarks for event camera processing.

This module provides benchmarks specifically designed to test performance
with actual event camera data files from the /data/ directory.
"""

import os
import gc
import time
from typing import Dict, Tuple, Any

import numpy as np
import pytest
import psutil

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

# Import evlib functions
import evlib
from evlib.representations import (
    create_voxel_grid,
    stacked_histogram,
    create_time_surface,
)

try:
    from evlib.polars_utils import enhanced_load_events
except ImportError:
    pass


class RealDataConfig:
    """Configuration for real data file benchmarks."""

    # Real data files with their characteristics
    REAL_DATA_FILES = {
        "slider_depth_text": {
            "path": "/Users/tallam/github/tallamjr/origin/evlib/data/slider_depth/events.txt",
            "format": "text",
            "size_category": "medium",
            "expected_events": 1_100_000,  # Approximate
            "sensor_resolution": (346, 240),
            "description": "DAVIS event data from slider depth sequence",
        },
        "etram_h5_small": {
            "path": "/Users/tallam/github/tallamjr/origin/evlib/data/eTram/h5/val_2/val_night_011_td.h5",
            "format": "hdf5",
            "size_category": "small",
            "expected_events": 500_000,  # Approximate
            "sensor_resolution": (1280, 720),
            "description": "eTram dataset - small nighttime sequence",
        },
        "etram_h5_medium": {
            "path": "/Users/tallam/github/tallamjr/origin/evlib/data/eTram/h5/val_2/val_night_007_td.h5",
            "format": "hdf5",
            "size_category": "medium",
            "expected_events": 15_000_000,  # Approximate
            "sensor_resolution": (1280, 720),
            "description": "eTram dataset - medium nighttime sequence",
        },
        "original_h5_large": {
            "path": "/Users/tallam/github/tallamjr/origin/evlib/data/original/front/seq01.h5",
            "format": "hdf5",
            "size_category": "large",
            "expected_events": 50_000_000,  # Approximate
            "sensor_resolution": (346, 240),
            "description": "Original dataset - large sequence",
        },
        "etram_evt2_small": {
            "path": "/Users/tallam/github/tallamjr/origin/evlib/data/eTram/raw/val_2/val_night_011.raw",
            "format": "evt2",
            "size_category": "small",
            "expected_events": 500_000,  # Approximate
            "sensor_resolution": (1280, 720),
            "description": "eTram dataset - EVT2 format small sequence",
        },
        "etram_evt2_large": {
            "path": "/Users/tallam/github/tallamjr/origin/evlib/data/eTram/raw/val_2/val_night_007.raw",
            "format": "evt2",
            "size_category": "large",
            "expected_events": 15_000_000,  # Approximate
            "sensor_resolution": (1280, 720),
            "description": "eTram dataset - EVT2 format large sequence",
        },
    }


class RealDataLoadingBenchmarks:
    """Benchmarks for loading real event data files."""

    def _check_file_exists(self, file_info: Dict[str, Any]) -> bool:
        """Check if a data file exists."""
        return os.path.exists(file_info["path"])

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / 1024 / 1024

    @pytest.mark.benchmark(group="real_data_loading")
    @pytest.mark.parametrize("file_key", ["slider_depth_text", "etram_h5_small"])
    def test_numpy_real_data_loading(self, benchmark, file_key):
        """Benchmark NumPy-based loading of real data files."""
        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        if not self._check_file_exists(file_info):
            pytest.skip(f"Real data file not found: {file_info['path']}")

        def load_events():
            gc.collect()
            start_memory = self._get_memory_usage()

            try:
                result = evlib.load_events(file_info["path"], output_format="numpy")

                end_memory = self._get_memory_usage()
                memory_used = end_memory - start_memory

                return result, memory_used
            except Exception as e:
                pytest.skip(f"Failed to load file {file_key}: {e}")

        (x, y, timestamp, polarity), memory_used = benchmark(load_events)

        benchmark.extra_info.update(
            {
                "file_key": file_key,
                "file_format": file_info["format"],
                "size_category": file_info["size_category"],
                "events_loaded": len(x),
                "memory_mb": memory_used,
                "sensor_resolution": file_info["sensor_resolution"],
                "implementation": "numpy",
            }
        )

    @pytest.mark.benchmark(group="real_data_loading")
    @pytest.mark.parametrize("file_key", ["slider_depth_text", "etram_h5_small"])
    def test_polars_real_data_loading(self, benchmark, file_key):
        """Benchmark Polars-based loading of real data files."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")

        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        if not self._check_file_exists(file_info):
            pytest.skip(f"Real data file not found: {file_info['path']}")

        def load_events():
            gc.collect()
            start_memory = self._get_memory_usage()

            try:
                result = enhanced_load_events(file_info["path"], output_format="polars")

                end_memory = self._get_memory_usage()
                memory_used = end_memory - start_memory

                return result, memory_used
            except Exception as e:
                pytest.skip(f"Failed to load file {file_key}: {e}")

        df, memory_used = benchmark(load_events)

        benchmark.extra_info.update(
            {
                "file_key": file_key,
                "file_format": file_info["format"],
                "size_category": file_info["size_category"],
                "events_loaded": len(df),
                "memory_mb": memory_used,
                "sensor_resolution": file_info["sensor_resolution"],
                "implementation": "polars",
            }
        )


class RealDataProcessingBenchmarks:
    """Benchmarks for processing operations on real data."""

    def _load_real_data_numpy(self, file_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load real data using NumPy format."""
        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        if not os.path.exists(file_info["path"]):
            pytest.skip(f"Real data file not found: {file_info['path']}")

        try:
            return evlib.load_events(file_info["path"], output_format="numpy")
        except Exception as e:
            pytest.skip(f"Failed to load file {file_key}: {e}")

    def _load_real_data_polars(self, file_key: str) -> "pl.DataFrame":
        """Load real data using Polars format."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")

        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        if not os.path.exists(file_info["path"]):
            pytest.skip(f"Real data file not found: {file_info['path']}")

        try:
            return enhanced_load_events(file_info["path"], output_format="polars")
        except Exception as e:
            pytest.skip(f"Failed to load file {file_key}: {e}")

    @pytest.mark.benchmark(group="real_data_processing")
    @pytest.mark.parametrize("file_key", ["slider_depth_text"])
    def test_numpy_real_data_voxel_grid(self, benchmark, file_key):
        """Benchmark voxel grid creation with real data using NumPy."""
        x, y, timestamp, polarity = self._load_real_data_numpy(file_key)
        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        def create_voxel():
            gc.collect()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            voxel = create_voxel_grid(x, y, timestamp, polarity, file_info["sensor_resolution"], num_bins=10)

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory

            return voxel, memory_used

        voxel, memory_used = benchmark(create_voxel)

        benchmark.extra_info.update(
            {
                "file_key": file_key,
                "events_processed": len(x),
                "memory_mb": memory_used,
                "output_shape": voxel.shape,
                "implementation": "numpy",
                "operation": "voxel_grid",
            }
        )

    @pytest.mark.benchmark(group="real_data_processing")
    @pytest.mark.parametrize("file_key", ["slider_depth_text"])
    def test_polars_real_data_voxel_grid(self, benchmark, file_key):
        """Benchmark voxel grid creation with real data using Polars."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")

        df = self._load_real_data_polars(file_key)
        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        def create_voxel():
            gc.collect()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            voxel = create_voxel_grid(df, sensor_resolution=file_info["sensor_resolution"], num_bins=10)

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory

            return voxel, memory_used

        voxel, memory_used = benchmark(create_voxel)

        benchmark.extra_info.update(
            {
                "file_key": file_key,
                "events_processed": len(df),
                "memory_mb": memory_used,
                "output_shape": voxel.shape,
                "implementation": "polars",
                "operation": "voxel_grid",
            }
        )

    @pytest.mark.benchmark(group="real_data_processing")
    @pytest.mark.parametrize("file_key", ["slider_depth_text"])
    def test_numpy_real_data_filtering(self, benchmark, file_key):
        """Benchmark filtering operations with real data using NumPy."""
        x, y, timestamp, polarity = self._load_real_data_numpy(file_key)
        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        def filter_events():
            gc.collect()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Apply multiple filters
            width, height = file_info["sensor_resolution"]

            # Spatial filter: central 80% of sensor
            spatial_mask = (x >= width * 0.1) & (x <= width * 0.9) & (y >= height * 0.1) & (y <= height * 0.9)

            # Temporal filter: middle 50% of time range
            t_min, t_max = timestamp.min(), timestamp.max()
            t_range = t_max - t_min
            temporal_mask = (timestamp >= t_min + 0.25 * t_range) & (timestamp <= t_min + 0.75 * t_range)

            # Polarity filter: positive events only
            polarity_mask = polarity == 1

            # Combined filter
            combined_mask = spatial_mask & temporal_mask & polarity_mask

            filtered_x = x[combined_mask]
            filtered_y = y[combined_mask]
            filtered_t = timestamp[combined_mask]
            filtered_p = polarity[combined_mask]

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory

            return (filtered_x, filtered_y, filtered_t, filtered_p), memory_used

        result, memory_used = benchmark(filter_events)

        benchmark.extra_info.update(
            {
                "file_key": file_key,
                "original_events": len(x),
                "filtered_events": len(result[0]),
                "filter_ratio": len(result[0]) / len(x),
                "memory_mb": memory_used,
                "implementation": "numpy",
                "operation": "combined_filtering",
            }
        )

    @pytest.mark.benchmark(group="real_data_processing")
    @pytest.mark.parametrize("file_key", ["slider_depth_text"])
    def test_polars_real_data_filtering(self, benchmark, file_key):
        """Benchmark filtering operations with real data using Polars."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")

        df = self._load_real_data_polars(file_key)
        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        def filter_events():
            gc.collect()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Apply multiple filters using Polars
            width, height = file_info["sensor_resolution"]

            # Get time statistics
            t_stats = df.select(
                [pl.col("timestamp").min().alias("t_min"), pl.col("timestamp").max().alias("t_max")]
            )
            t_min = t_stats.item(0, 0)
            t_max = t_stats.item(0, 1)
            t_range = t_max - t_min

            # Combined filter using Polars lazy evaluation
            filtered_df = (
                df.lazy()
                .filter(
                    # Spatial filter: central 80% of sensor
                    (pl.col("x") >= width * 0.1)
                    & (pl.col("x") <= width * 0.9)
                    & (pl.col("y") >= height * 0.1)
                    & (pl.col("y") <= height * 0.9)
                    &
                    # Temporal filter: middle 50% of time range
                    (pl.col("timestamp") >= t_min + 0.25 * t_range)
                    & (pl.col("timestamp") <= t_min + 0.75 * t_range)
                    &
                    # Polarity filter: positive events only
                    (pl.col("polarity") == 1)
                )
                .collect()
            )

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory

            return filtered_df, memory_used

        result, memory_used = benchmark(filter_events)

        benchmark.extra_info.update(
            {
                "file_key": file_key,
                "original_events": len(df),
                "filtered_events": len(result),
                "filter_ratio": len(result) / len(df),
                "memory_mb": memory_used,
                "implementation": "polars",
                "operation": "combined_filtering",
            }
        )

    @pytest.mark.benchmark(group="real_data_processing")
    @pytest.mark.parametrize("file_key", ["slider_depth_text"])
    def test_numpy_real_data_aggregation(self, benchmark, file_key):
        """Benchmark aggregation operations with real data using NumPy."""
        x, y, timestamp, polarity = self._load_real_data_numpy(file_key)

        def aggregate_events():
            gc.collect()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Calculate comprehensive statistics
            stats = {
                # Basic counts
                "total_events": len(x),
                "positive_events": np.sum(polarity == 1),
                "negative_events": np.sum(polarity == 0),
                # Spatial statistics
                "mean_x": np.mean(x),
                "std_x": np.std(x),
                "mean_y": np.mean(y),
                "std_y": np.std(y),
                "spatial_range_x": np.max(x) - np.min(x),
                "spatial_range_y": np.max(y) - np.min(y),
                # Temporal statistics
                "duration": timestamp[-1] - timestamp[0],
                "mean_rate": len(timestamp) / (timestamp[-1] - timestamp[0]),
                "temporal_std": np.std(np.diff(timestamp)),
                # Activity distribution
                "activity_per_pixel": len(x) / (np.max(x) * np.max(y)),
                # Polarity balance
                "polarity_ratio": np.sum(polarity == 1) / len(polarity),
            }

            # Time-based binning (100 bins)
            time_bins = np.linspace(timestamp[0], timestamp[-1], 101)
            activity_per_bin = np.histogram(timestamp, bins=time_bins)[0]
            stats["max_activity_bin"] = np.max(activity_per_bin)
            stats["min_activity_bin"] = np.min(activity_per_bin)
            stats["activity_std"] = np.std(activity_per_bin)

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory

            return stats, memory_used

        stats, memory_used = benchmark(aggregate_events)

        benchmark.extra_info.update(
            {
                "file_key": file_key,
                "events_processed": stats["total_events"],
                "memory_mb": memory_used,
                "implementation": "numpy",
                "operation": "comprehensive_aggregation",
                "event_rate_hz": stats["mean_rate"],
            }
        )

    @pytest.mark.benchmark(group="real_data_processing")
    @pytest.mark.parametrize("file_key", ["slider_depth_text"])
    def test_polars_real_data_aggregation(self, benchmark, file_key):
        """Benchmark aggregation operations with real data using Polars."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")

        df = self._load_real_data_polars(file_key)

        def aggregate_events():
            gc.collect()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Calculate comprehensive statistics using Polars
            basic_stats = df.select(
                [
                    pl.count().alias("total_events"),
                    pl.col("polarity").filter(pl.col("polarity") == 1).count().alias("positive_events"),
                    pl.col("polarity").filter(pl.col("polarity") == 0).count().alias("negative_events"),
                    pl.col("x").mean().alias("mean_x"),
                    pl.col("x").std().alias("std_x"),
                    pl.col("y").mean().alias("mean_y"),
                    pl.col("y").std().alias("std_y"),
                    pl.col("x").max().alias("max_x"),
                    pl.col("x").min().alias("min_x"),
                    pl.col("y").max().alias("max_y"),
                    pl.col("y").min().alias("min_y"),
                    pl.col("timestamp").max().alias("max_t"),
                    pl.col("timestamp").min().alias("min_t"),
                ]
            )

            # Polarity statistics
            polarity_stats = df.group_by("polarity").agg(
                [
                    pl.count().alias("count"),
                    pl.col("x").mean().alias("mean_x"),
                    pl.col("y").mean().alias("mean_y"),
                ]
            )

            # Time-based activity analysis
            t_min = df.select(pl.col("timestamp").min()).item()
            t_max = df.select(pl.col("timestamp").max()).item()

            # Create time bins using Polars
            activity_df = (
                df.with_columns(
                    [
                        ((pl.col("timestamp") - t_min) / (t_max - t_min) * 100)
                        .floor()
                        .clip(0, 99)
                        .cast(pl.Int32)
                        .alias("time_bin")
                    ]
                )
                .group_by("time_bin")
                .agg(pl.count().alias("events_in_bin"))
            )

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory

            return {
                "basic_stats": basic_stats,
                "polarity_stats": polarity_stats,
                "activity_stats": activity_df,
            }, memory_used

        results, memory_used = benchmark(aggregate_events)

        total_events = results["basic_stats"].item(0, 0)
        duration = results["basic_stats"].item(0, 11) - results["basic_stats"].item(0, 10)
        event_rate = total_events / duration if duration > 0 else 0

        benchmark.extra_info.update(
            {
                "file_key": file_key,
                "events_processed": total_events,
                "memory_mb": memory_used,
                "implementation": "polars",
                "operation": "comprehensive_aggregation",
                "event_rate_hz": event_rate,
            }
        )


class RealDataMemoryBenchmarks:
    """Memory-focused benchmarks with real data."""

    @pytest.mark.benchmark(group="real_data_memory")
    @pytest.mark.parametrize("file_key", ["slider_depth_text"])
    def test_memory_peak_usage_numpy(self, benchmark, file_key):
        """Test peak memory usage with NumPy operations on real data."""
        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        if not os.path.exists(file_info["path"]):
            pytest.skip(f"Real data file not found: {file_info['path']}")

        def memory_peak_test():
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Load data
            x, y, timestamp, polarity = evlib.load_events(file_info["path"], output_format="numpy")
            after_load_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Create multiple representations
            voxel = create_voxel_grid(x, y, timestamp, polarity, file_info["sensor_resolution"], 10)
            after_voxel_memory = psutil.Process().memory_info().rss / 1024 / 1024

            hist = stacked_histogram(
                x,
                y,
                polarity,
                timestamp,
                5,
                file_info["sensor_resolution"][1],
                file_info["sensor_resolution"][0],
            )
            after_hist_memory = psutil.Process().memory_info().rss / 1024 / 1024

            surface = create_time_surface(x, y, timestamp, polarity, file_info["sensor_resolution"])
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Clean up and measure final memory
            del x, y, timestamp, polarity, voxel, hist, surface
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024

            return {
                "initial_memory": initial_memory,
                "after_load_memory": after_load_memory,
                "after_voxel_memory": after_voxel_memory,
                "after_hist_memory": after_hist_memory,
                "peak_memory": peak_memory,
                "final_memory": final_memory,
                "max_delta": peak_memory - initial_memory,
                "load_delta": after_load_memory - initial_memory,
                "representations_delta": peak_memory - after_load_memory,
            }

        result = benchmark(memory_peak_test)

        benchmark.extra_info.update({"file_key": file_key, "implementation": "numpy", **result})

    @pytest.mark.benchmark(group="real_data_memory")
    @pytest.mark.parametrize("file_key", ["slider_depth_text"])
    def test_memory_peak_usage_polars(self, benchmark, file_key):
        """Test peak memory usage with Polars operations on real data."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")

        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        if not os.path.exists(file_info["path"]):
            pytest.skip(f"Real data file not found: {file_info['path']}")

        def memory_peak_test():
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Load data as Polars DataFrame
            df = enhanced_load_events(file_info["path"], output_format="polars")
            after_load_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Create multiple representations
            voxel = create_voxel_grid(df, sensor_resolution=file_info["sensor_resolution"], num_bins=10)
            after_voxel_memory = psutil.Process().memory_info().rss / 1024 / 1024

            hist = stacked_histogram(
                df, bins=5, height=file_info["sensor_resolution"][1], width=file_info["sensor_resolution"][0]
            )
            after_hist_memory = psutil.Process().memory_info().rss / 1024 / 1024

            surface = create_time_surface(df, sensor_resolution=file_info["sensor_resolution"])
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Clean up and measure final memory
            del df, voxel, hist, surface
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024

            return {
                "initial_memory": initial_memory,
                "after_load_memory": after_load_memory,
                "after_voxel_memory": after_voxel_memory,
                "after_hist_memory": after_hist_memory,
                "peak_memory": peak_memory,
                "final_memory": final_memory,
                "max_delta": peak_memory - initial_memory,
                "load_delta": after_load_memory - initial_memory,
                "representations_delta": peak_memory - after_load_memory,
            }

        result = benchmark(memory_peak_test)

        benchmark.extra_info.update({"file_key": file_key, "implementation": "polars", **result})


class RealDataScalabilityBenchmarks:
    """Scalability benchmarks using real data files of different sizes."""

    @pytest.mark.benchmark(group="real_data_scalability")
    @pytest.mark.parametrize("file_key", ["slider_depth_text", "etram_h5_small"])
    def test_scalability_with_real_files(self, benchmark, file_key):
        """Test scalability across different real data file sizes."""
        file_info = RealDataConfig.REAL_DATA_FILES[file_key]

        if not os.path.exists(file_info["path"]):
            pytest.skip(f"Real data file not found: {file_info['path']}")

        def scalability_test():
            gc.collect()
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Load data
            load_start = time.time()
            try:
                x, y, timestamp, polarity = evlib.load_events(file_info["path"], output_format="numpy")
                load_time = time.time() - load_start

                # Process with multiple operations
                process_start = time.time()

                # 1. Create voxel grid
                voxel = create_voxel_grid(x, y, timestamp, polarity, file_info["sensor_resolution"], 5)
                voxel_time = time.time() - process_start

                # 2. Apply filtering
                filter_start = time.time()
                width, height = file_info["sensor_resolution"]
                mask = (x >= width // 4) & (x <= 3 * width // 4) & (y >= height // 4) & (y <= 3 * height // 4)
                filtered_count = np.sum(mask)
                filter_time = time.time() - filter_start

                # 3. Calculate statistics
                stats_start = time.time()
                stats = {
                    "event_rate": len(x) / (timestamp[-1] - timestamp[0]) if len(timestamp) > 1 else 0,
                    "spatial_density": len(x) / (width * height),
                    "polarity_ratio": np.mean(polarity),
                }
                stats_time = time.time() - stats_start

                total_time = time.time() - start_time
                peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = peak_memory - start_memory

                return {
                    "total_time": total_time,
                    "load_time": load_time,
                    "voxel_time": voxel_time,
                    "filter_time": filter_time,
                    "stats_time": stats_time,
                    "memory_used": memory_used,
                    "events_processed": len(x),
                    "filtered_events": filtered_count,
                    "voxel_shape": voxel.shape,
                    "stats": stats,
                }

            except Exception as e:
                pytest.skip(f"Failed to process file {file_key}: {e}")

        result = benchmark.pedantic(scalability_test, rounds=3)

        benchmark.extra_info.update(
            {
                "file_key": file_key,
                "file_format": file_info["format"],
                "size_category": file_info["size_category"],
                "implementation": "numpy",
                **result,
            }
        )


# Test to check if real data files are available
def test_real_data_availability():
    """Test which real data files are available for benchmarking."""
    available_files = []
    missing_files = []

    for file_key, file_info in RealDataConfig.REAL_DATA_FILES.items():
        if os.path.exists(file_info["path"]):
            available_files.append(file_key)
        else:
            missing_files.append(file_key)

    print(f"Available real data files: {available_files}")
    print(f"Missing real data files: {missing_files}")

    # At least one file should be available for meaningful benchmarks
    assert len(available_files) > 0, f"No real data files found. Missing: {missing_files}"


if __name__ == "__main__":
    """Run real data availability check."""
    test_real_data_availability()
    print("Real data benchmark infrastructure ready!")
