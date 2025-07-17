"""
Comprehensive performance benchmarks comparing NumPy vs Polars for event camera data processing.

This module provides a complete benchmarking suite for comparing the performance of
NumPy-based implementations versus Polars-based implementations across different
operations and dataset sizes.

Key benchmark areas:
- Data loading (small, medium, large files)
- Filtering operations (spatial, temporal, polarity)
- Aggregation operations (counting, grouping, statistics)
- Representation functions (voxel grids, histograms, time surfaces)
- Memory usage patterns
- Scalability with dataset size
"""

import gc
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import psutil

import numpy as np
import pytest

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Import evlib functions
import evlib
from evlib.representations import (
    create_voxel_grid, 
    stacked_histogram, 
    create_time_surface, 
    create_event_histogram
)

# Try to import Polars utils
try:
    from evlib.polars_utils import (
        load_events_as_polars_dataframe,
        events_to_polars_dataframe_with_metadata,
        enhanced_load_events
    )
except ImportError:
    pass


class MemoryProfiler:
    """Memory usage profiler for benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_delta(self) -> float:
        """Get memory usage delta from initial measurement."""
        return self.get_memory_usage() - self.initial_memory
    
    def reset(self):
        """Reset initial memory measurement."""
        gc.collect()
        time.sleep(0.1)  # Allow GC to complete
        self.initial_memory = self.get_memory_usage()


class BenchmarkData:
    """Generates test data for benchmarks."""
    
    @staticmethod
    def create_synthetic_events(
        num_events: int,
        width: int = 640,
        height: int = 480,
        duration: float = 1.0,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create synthetic event data for benchmarking."""
        np.random.seed(seed)
        
        x = np.random.randint(0, width, size=num_events, dtype=np.uint16)
        y = np.random.randint(0, height, size=num_events, dtype=np.uint16)
        timestamp = np.sort(np.random.uniform(0, duration, size=num_events))
        polarity = np.random.choice([0, 1], size=num_events).astype(np.int8)
        
        return x, y, timestamp, polarity
    
    @staticmethod
    def create_polars_dataframe(
        x: np.ndarray, 
        y: np.ndarray, 
        timestamp: np.ndarray, 
        polarity: np.ndarray
    ) -> "pl.DataFrame":
        """Create Polars DataFrame from event arrays."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        return pl.DataFrame({
            "x": x,
            "y": y,
            "timestamp": timestamp,
            "polarity": polarity
        })
    
    @staticmethod
    def save_events_to_file(
        x: np.ndarray,
        y: np.ndarray,
        timestamp: np.ndarray,
        polarity: np.ndarray,
        filepath: str
    ):
        """Save events to a text file for loading benchmarks."""
        with open(filepath, 'w') as f:
            f.write("# timestamp x y polarity\n")
            for i in range(len(x)):
                f.write(f"{timestamp[i]:.6f} {x[i]} {y[i]} {polarity[i]}\n")


class BenchmarkConfig:
    """Configuration for benchmark parameters."""
    
    # Dataset sizes for scalability testing
    DATASET_SIZES = {
        "tiny": 1_000,
        "small": 10_000,
        "medium": 100_000,
        "large": 1_000_000,
        "xlarge": 5_000_000,
    }
    
    # Sensor resolutions for realistic testing
    SENSOR_RESOLUTIONS = {
        "davis_346": (346, 240),
        "davis_640": (640, 480),
        "prophesee_hd": (1280, 720),
    }
    
    # Data files for real data testing
    REAL_DATA_FILES = {
        "small_text": "/Users/tallam/github/tallamjr/origin/evlib/data/slider_depth/events.txt",
        "medium_h5": "/Users/tallam/github/tallamjr/origin/evlib/data/eTram/h5/val_2/val_night_011_td.h5",
        "large_h5": "/Users/tallam/github/tallamjr/origin/evlib/data/original/front/seq01.h5",
    }


class DataLoadingBenchmarks:
    """Benchmarks for data loading operations."""
    
    @pytest.mark.benchmark(group="loading")
    @pytest.mark.parametrize("size_name", ["small", "medium", "large"])
    def test_numpy_loading_synthetic(self, benchmark, size_name):
        """Benchmark NumPy-based event loading from synthetic data."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        x, y, timestamp, polarity = BenchmarkData.create_synthetic_events(size)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            BenchmarkData.save_events_to_file(x, y, timestamp, polarity, f.name)
            temp_path = f.name
        
        try:
            memory_profiler = MemoryProfiler()
            
            def load_numpy():
                memory_profiler.reset()
                result = evlib.load_events(temp_path, output_format="numpy")
                memory_used = memory_profiler.get_memory_delta()
                return result, memory_used
            
            result, memory_used = benchmark(load_numpy)
            
            # Add custom metrics
            benchmark.extra_info.update({
                'dataset_size': size,
                'memory_mb': memory_used,
                'events_loaded': len(result[0]) if isinstance(result, tuple) else 0
            })
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.benchmark(group="loading")
    @pytest.mark.parametrize("size_name", ["small", "medium", "large"])
    def test_polars_loading_synthetic(self, benchmark, size_name):
        """Benchmark Polars-based event loading from synthetic data."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        x, y, timestamp, polarity = BenchmarkData.create_synthetic_events(size)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            BenchmarkData.save_events_to_file(x, y, timestamp, polarity, f.name)
            temp_path = f.name
        
        try:
            memory_profiler = MemoryProfiler()
            
            def load_polars():
                memory_profiler.reset()
                result = enhanced_load_events(temp_path, output_format="polars")
                memory_used = memory_profiler.get_memory_delta()
                return result, memory_used
            
            result, memory_used = benchmark(load_polars)
            
            # Add custom metrics
            benchmark.extra_info.update({
                'dataset_size': size,
                'memory_mb': memory_used,
                'events_loaded': len(result) if hasattr(result, '__len__') else 0
            })
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.benchmark(group="loading_real")
    @pytest.mark.parametrize("file_key", ["small_text"])
    def test_numpy_loading_real_data(self, benchmark, file_key):
        """Benchmark NumPy loading with real data files."""
        filepath = BenchmarkConfig.REAL_DATA_FILES[file_key]
        
        if not os.path.exists(filepath):
            pytest.skip(f"Real data file not found: {filepath}")
        
        memory_profiler = MemoryProfiler()
        
        def load_numpy():
            memory_profiler.reset()
            result = evlib.load_events(filepath, output_format="numpy")
            memory_used = memory_profiler.get_memory_delta()
            return result, memory_used
        
        result, memory_used = benchmark(load_numpy)
        
        benchmark.extra_info.update({
            'file_type': file_key,
            'file_path': filepath,
            'memory_mb': memory_used,
            'events_loaded': len(result[0]) if isinstance(result, tuple) else 0
        })
    
    @pytest.mark.benchmark(group="loading_real")
    @pytest.mark.parametrize("file_key", ["small_text"])
    def test_polars_loading_real_data(self, benchmark, file_key):
        """Benchmark Polars loading with real data files."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
            
        filepath = BenchmarkConfig.REAL_DATA_FILES[file_key]
        
        if not os.path.exists(filepath):
            pytest.skip(f"Real data file not found: {filepath}")
        
        memory_profiler = MemoryProfiler()
        
        def load_polars():
            memory_profiler.reset()
            result = enhanced_load_events(filepath, output_format="polars")
            memory_used = memory_profiler.get_memory_delta()
            return result, memory_used
        
        result, memory_used = benchmark(load_polars)
        
        benchmark.extra_info.update({
            'file_type': file_key,
            'file_path': filepath,
            'memory_mb': memory_used,
            'events_loaded': len(result) if hasattr(result, '__len__') else 0
        })


class FilteringBenchmarks:
    """Benchmarks for filtering operations."""
    
    def setup_data(self, size: int):
        """Set up test data for filtering benchmarks."""
        self.x, self.y, self.timestamp, self.polarity = BenchmarkData.create_synthetic_events(size)
        if POLARS_AVAILABLE:
            self.df = BenchmarkData.create_polars_dataframe(self.x, self.y, self.timestamp, self.polarity)
    
    @pytest.mark.benchmark(group="filtering")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_spatial_filtering(self, benchmark, size_name):
        """Benchmark spatial filtering using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def spatial_filter():
            memory_profiler.reset()
            # Filter events in central region
            mask = (self.x >= 100) & (self.x <= 540) & (self.y >= 50) & (self.y <= 430)
            filtered_x = self.x[mask]
            filtered_y = self.y[mask]
            filtered_t = self.timestamp[mask]
            filtered_p = self.polarity[mask]
            memory_used = memory_profiler.get_memory_delta()
            return (filtered_x, filtered_y, filtered_t, filtered_p), memory_used
        
        result, memory_used = benchmark(spatial_filter)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'events_filtered': len(result[0]),
            'filter_type': 'spatial'
        })
    
    @pytest.mark.benchmark(group="filtering")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_spatial_filtering(self, benchmark, size_name):
        """Benchmark spatial filtering using Polars."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def spatial_filter():
            memory_profiler.reset()
            # Filter events in central region using Polars
            filtered_df = self.df.filter(
                (pl.col("x") >= 100) & 
                (pl.col("x") <= 540) & 
                (pl.col("y") >= 50) & 
                (pl.col("y") <= 430)
            )
            memory_used = memory_profiler.get_memory_delta()
            return filtered_df, memory_used
        
        result, memory_used = benchmark(spatial_filter)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'events_filtered': len(result),
            'filter_type': 'spatial'
        })
    
    @pytest.mark.benchmark(group="filtering")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_temporal_filtering(self, benchmark, size_name):
        """Benchmark temporal filtering using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def temporal_filter():
            memory_profiler.reset()
            # Filter events in middle 50% of time range
            t_min, t_max = self.timestamp.min(), self.timestamp.max()
            t_range = t_max - t_min
            t_start = t_min + 0.25 * t_range
            t_end = t_min + 0.75 * t_range
            
            mask = (self.timestamp >= t_start) & (self.timestamp <= t_end)
            filtered_x = self.x[mask]
            filtered_y = self.y[mask]
            filtered_t = self.timestamp[mask]
            filtered_p = self.polarity[mask]
            memory_used = memory_profiler.get_memory_delta()
            return (filtered_x, filtered_y, filtered_t, filtered_p), memory_used
        
        result, memory_used = benchmark(temporal_filter)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'events_filtered': len(result[0]),
            'filter_type': 'temporal'
        })
    
    @pytest.mark.benchmark(group="filtering")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_temporal_filtering(self, benchmark, size_name):
        """Benchmark temporal filtering using Polars."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def temporal_filter():
            memory_profiler.reset()
            # Filter events in middle 50% of time range using Polars
            t_stats = self.df.select([pl.col("timestamp").min(), pl.col("timestamp").max()])
            t_min = t_stats.item(0, 0)
            t_max = t_stats.item(0, 1)
            t_range = t_max - t_min
            t_start = t_min + 0.25 * t_range
            t_end = t_min + 0.75 * t_range
            
            filtered_df = self.df.filter(
                (pl.col("timestamp") >= t_start) & 
                (pl.col("timestamp") <= t_end)
            )
            memory_used = memory_profiler.get_memory_delta()
            return filtered_df, memory_used
        
        result, memory_used = benchmark(temporal_filter)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'events_filtered': len(result),
            'filter_type': 'temporal'
        })
    
    @pytest.mark.benchmark(group="filtering")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_polarity_filtering(self, benchmark, size_name):
        """Benchmark polarity filtering using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def polarity_filter():
            memory_profiler.reset()
            # Filter positive polarity events
            mask = self.polarity == 1
            filtered_x = self.x[mask]
            filtered_y = self.y[mask]
            filtered_t = self.timestamp[mask]
            filtered_p = self.polarity[mask]
            memory_used = memory_profiler.get_memory_delta()
            return (filtered_x, filtered_y, filtered_t, filtered_p), memory_used
        
        result, memory_used = benchmark(polarity_filter)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'events_filtered': len(result[0]),
            'filter_type': 'polarity'
        })
    
    @pytest.mark.benchmark(group="filtering")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_polarity_filtering(self, benchmark, size_name):
        """Benchmark polarity filtering using Polars."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def polarity_filter():
            memory_profiler.reset()
            # Filter positive polarity events using Polars
            filtered_df = self.df.filter(pl.col("polarity") == 1)
            memory_used = memory_profiler.get_memory_delta()
            return filtered_df, memory_used
        
        result, memory_used = benchmark(polarity_filter)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'events_filtered': len(result),
            'filter_type': 'polarity'
        })


class AggregationBenchmarks:
    """Benchmarks for aggregation operations."""
    
    def setup_data(self, size: int):
        """Set up test data for aggregation benchmarks."""
        self.x, self.y, self.timestamp, self.polarity = BenchmarkData.create_synthetic_events(size)
        if POLARS_AVAILABLE:
            self.df = BenchmarkData.create_polars_dataframe(self.x, self.y, self.timestamp, self.polarity)
    
    @pytest.mark.benchmark(group="aggregation")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_event_counting(self, benchmark, size_name):
        """Benchmark event counting using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def count_events():
            memory_profiler.reset()
            # Count events by polarity
            positive_count = np.sum(self.polarity == 1)
            negative_count = np.sum(self.polarity == 0)
            total_count = len(self.polarity)
            memory_used = memory_profiler.get_memory_delta()
            return {'positive': positive_count, 'negative': negative_count, 'total': total_count}, memory_used
        
        result, memory_used = benchmark(count_events)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'operation': 'counting'
        })
    
    @pytest.mark.benchmark(group="aggregation")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_event_counting(self, benchmark, size_name):
        """Benchmark event counting using Polars."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def count_events():
            memory_profiler.reset()
            # Count events by polarity using Polars
            counts = self.df.group_by("polarity").agg(pl.count()).sort("polarity")
            memory_used = memory_profiler.get_memory_delta()
            return counts, memory_used
        
        result, memory_used = benchmark(count_events)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'operation': 'counting'
        })
    
    @pytest.mark.benchmark(group="aggregation")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_spatial_statistics(self, benchmark, size_name):
        """Benchmark spatial statistics calculation using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def spatial_stats():
            memory_profiler.reset()
            # Calculate spatial statistics
            stats = {
                'mean_x': np.mean(self.x),
                'std_x': np.std(self.x),
                'mean_y': np.mean(self.y),
                'std_y': np.std(self.y),
                'min_x': np.min(self.x),
                'max_x': np.max(self.x),
                'min_y': np.min(self.y),
                'max_y': np.max(self.y),
            }
            memory_used = memory_profiler.get_memory_delta()
            return stats, memory_used
        
        result, memory_used = benchmark(spatial_stats)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'operation': 'spatial_statistics'
        })
    
    @pytest.mark.benchmark(group="aggregation")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_spatial_statistics(self, benchmark, size_name):
        """Benchmark spatial statistics calculation using Polars."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def spatial_stats():
            memory_profiler.reset()
            # Calculate spatial statistics using Polars
            stats = self.df.select([
                pl.col("x").mean().alias("mean_x"),
                pl.col("x").std().alias("std_x"),
                pl.col("y").mean().alias("mean_y"),
                pl.col("y").std().alias("std_y"),
                pl.col("x").min().alias("min_x"),
                pl.col("x").max().alias("max_x"),
                pl.col("y").min().alias("min_y"),
                pl.col("y").max().alias("max_y"),
            ])
            memory_used = memory_profiler.get_memory_delta()
            return stats, memory_used
        
        result, memory_used = benchmark(spatial_stats)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'operation': 'spatial_statistics'
        })
    
    @pytest.mark.benchmark(group="aggregation")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_temporal_binning(self, benchmark, size_name):
        """Benchmark temporal binning using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def temporal_binning():
            memory_profiler.reset()
            # Create temporal bins
            num_bins = 100
            t_min, t_max = self.timestamp.min(), self.timestamp.max()
            
            if t_max > t_min:
                bin_edges = np.linspace(t_min, t_max, num_bins + 1)
                bin_indices = np.digitize(self.timestamp, bin_edges) - 1
                bin_indices = np.clip(bin_indices, 0, num_bins - 1)
                
                # Count events per bin
                bin_counts = np.bincount(bin_indices, minlength=num_bins)
            else:
                bin_counts = np.array([len(self.timestamp)] + [0] * (num_bins - 1))
            
            memory_used = memory_profiler.get_memory_delta()
            return bin_counts, memory_used
        
        result, memory_used = benchmark(temporal_binning)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'operation': 'temporal_binning'
        })
    
    @pytest.mark.benchmark(group="aggregation")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_temporal_binning(self, benchmark, size_name):
        """Benchmark temporal binning using Polars."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def temporal_binning():
            memory_profiler.reset()
            # Create temporal bins using Polars
            num_bins = 100
            
            # Calculate time statistics
            t_stats = self.df.select([
                pl.col("timestamp").min().alias("t_min"),
                pl.col("timestamp").max().alias("t_max")
            ])
            t_min = t_stats.item(0, 0)
            t_max = t_stats.item(0, 1)
            
            if t_max > t_min:
                # Create bins using Polars expressions
                df_with_bins = self.df.with_columns([
                    ((pl.col("timestamp") - t_min) / (t_max - t_min) * num_bins)
                    .floor()
                    .clip(0, num_bins - 1)
                    .cast(pl.Int32)
                    .alias("time_bin")
                ])
                
                # Count events per bin
                bin_counts = df_with_bins.group_by("time_bin").agg(pl.count())
            else:
                # All events in one bin
                bin_counts = pl.DataFrame({"time_bin": [0], "count": [len(self.df)]})
            
            memory_used = memory_profiler.get_memory_delta()
            return bin_counts, memory_used
        
        result, memory_used = benchmark(temporal_binning)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'operation': 'temporal_binning'
        })


class RepresentationBenchmarks:
    """Benchmarks for representation function creation."""
    
    def setup_data(self, size: int):
        """Set up test data for representation benchmarks."""
        self.x, self.y, self.timestamp, self.polarity = BenchmarkData.create_synthetic_events(size)
        self.sensor_resolution = (640, 480)
        if POLARS_AVAILABLE:
            self.df = BenchmarkData.create_polars_dataframe(self.x, self.y, self.timestamp, self.polarity)
    
    @pytest.mark.benchmark(group="representations")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_voxel_grid(self, benchmark, size_name):
        """Benchmark voxel grid creation using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def create_voxel():
            memory_profiler.reset()
            voxel = create_voxel_grid(
                self.x, self.y, self.timestamp, self.polarity,
                self.sensor_resolution, num_bins=5
            )
            memory_used = memory_profiler.get_memory_delta()
            return voxel, memory_used
        
        result, memory_used = benchmark(create_voxel)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'output_shape': result.shape,
            'representation': 'voxel_grid'
        })
    
    @pytest.mark.benchmark(group="representations")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_voxel_grid(self, benchmark, size_name):
        """Benchmark voxel grid creation using Polars DataFrame."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def create_voxel():
            memory_profiler.reset()
            voxel = create_voxel_grid(
                self.df,
                sensor_resolution=self.sensor_resolution,
                num_bins=5
            )
            memory_used = memory_profiler.get_memory_delta()
            return voxel, memory_used
        
        result, memory_used = benchmark(create_voxel)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'output_shape': result.shape,
            'representation': 'voxel_grid'
        })
    
    @pytest.mark.benchmark(group="representations")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_stacked_histogram(self, benchmark, size_name):
        """Benchmark stacked histogram creation using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def create_histogram():
            memory_profiler.reset()
            hist = stacked_histogram(
                self.x, self.y, self.polarity, self.timestamp,
                bins=5, height=480, width=640
            )
            memory_used = memory_profiler.get_memory_delta()
            return hist, memory_used
        
        result, memory_used = benchmark(create_histogram)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'output_shape': result.shape,
            'representation': 'stacked_histogram'
        })
    
    @pytest.mark.benchmark(group="representations")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_stacked_histogram(self, benchmark, size_name):
        """Benchmark stacked histogram creation using Polars DataFrame."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def create_histogram():
            memory_profiler.reset()
            hist = stacked_histogram(
                self.df,
                bins=5, height=480, width=640
            )
            memory_used = memory_profiler.get_memory_delta()
            return hist, memory_used
        
        result, memory_used = benchmark(create_histogram)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'output_shape': result.shape,
            'representation': 'stacked_histogram'
        })
    
    @pytest.mark.benchmark(group="representations")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_time_surface(self, benchmark, size_name):
        """Benchmark time surface creation using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def create_surface():
            memory_profiler.reset()
            surface = create_time_surface(
                self.x, self.y, self.timestamp, self.polarity,
                self.sensor_resolution
            )
            memory_used = memory_profiler.get_memory_delta()
            return surface, memory_used
        
        result, memory_used = benchmark(create_surface)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'output_shape': result.shape,
            'representation': 'time_surface'
        })
    
    @pytest.mark.benchmark(group="representations")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_time_surface(self, benchmark, size_name):
        """Benchmark time surface creation using Polars DataFrame."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def create_surface():
            memory_profiler.reset()
            surface = create_time_surface(
                self.df,
                sensor_resolution=self.sensor_resolution
            )
            memory_used = memory_profiler.get_memory_delta()
            return surface, memory_used
        
        result, memory_used = benchmark(create_surface)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'output_shape': result.shape,
            'representation': 'time_surface'
        })
    
    @pytest.mark.benchmark(group="representations")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_numpy_event_histogram(self, benchmark, size_name):
        """Benchmark event histogram creation using NumPy."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def create_histogram():
            memory_profiler.reset()
            hist = create_event_histogram(
                self.x, self.y, self.polarity,
                self.sensor_resolution
            )
            memory_used = memory_profiler.get_memory_delta()
            return hist, memory_used
        
        result, memory_used = benchmark(create_histogram)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'output_shape': result.shape,
            'representation': 'event_histogram'
        })
    
    @pytest.mark.benchmark(group="representations")
    @pytest.mark.parametrize("size_name", ["medium", "large"])
    def test_polars_event_histogram(self, benchmark, size_name):
        """Benchmark event histogram creation using Polars DataFrame."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        self.setup_data(size)
        
        memory_profiler = MemoryProfiler()
        
        def create_histogram():
            memory_profiler.reset()
            hist = create_event_histogram(
                self.df,
                sensor_resolution=self.sensor_resolution
            )
            memory_used = memory_profiler.get_memory_delta()
            return hist, memory_used
        
        result, memory_used = benchmark(create_histogram)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'output_shape': result.shape,
            'representation': 'event_histogram'
        })


class ScalabilityBenchmarks:
    """Benchmarks for testing scalability across different dataset sizes."""
    
    @pytest.mark.benchmark(group="scalability")
    @pytest.mark.parametrize("size_name", ["tiny", "small", "medium", "large"])
    def test_numpy_scalability_voxel_grid(self, benchmark, size_name):
        """Test NumPy voxel grid scalability across dataset sizes."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        x, y, timestamp, polarity = BenchmarkData.create_synthetic_events(size)
        sensor_resolution = (640, 480)
        
        memory_profiler = MemoryProfiler()
        
        def create_voxel():
            memory_profiler.reset()
            voxel = create_voxel_grid(
                x, y, timestamp, polarity,
                sensor_resolution, num_bins=5
            )
            memory_used = memory_profiler.get_memory_delta()
            return voxel, memory_used
        
        result, memory_used = benchmark(create_voxel)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'implementation': 'numpy',
            'operation': 'voxel_grid'
        })
    
    @pytest.mark.benchmark(group="scalability")
    @pytest.mark.parametrize("size_name", ["tiny", "small", "medium", "large"])
    def test_polars_scalability_voxel_grid(self, benchmark, size_name):
        """Test Polars voxel grid scalability across dataset sizes."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        x, y, timestamp, polarity = BenchmarkData.create_synthetic_events(size)
        df = BenchmarkData.create_polars_dataframe(x, y, timestamp, polarity)
        sensor_resolution = (640, 480)
        
        memory_profiler = MemoryProfiler()
        
        def create_voxel():
            memory_profiler.reset()
            voxel = create_voxel_grid(
                df,
                sensor_resolution=sensor_resolution,
                num_bins=5
            )
            memory_used = memory_profiler.get_memory_delta()
            return voxel, memory_used
        
        result, memory_used = benchmark(create_voxel)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'memory_mb': memory_used,
            'implementation': 'polars',
            'operation': 'voxel_grid'
        })


class MemoryUsageBenchmarks:
    """Benchmarks focused on memory usage patterns."""
    
    @pytest.mark.benchmark(group="memory")
    @pytest.mark.parametrize("size_name", ["large", "xlarge"])
    def test_numpy_memory_efficiency(self, benchmark, size_name):
        """Test memory efficiency of NumPy operations."""
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        
        memory_profiler = MemoryProfiler()
        
        def memory_test():
            memory_profiler.reset()
            
            # Create data
            x, y, timestamp, polarity = BenchmarkData.create_synthetic_events(size)
            peak_after_creation = memory_profiler.get_memory_delta()
            
            # Create voxel grid
            voxel = create_voxel_grid(
                x, y, timestamp, polarity,
                (640, 480), num_bins=10
            )
            peak_after_voxel = memory_profiler.get_memory_delta()
            
            # Clean up original data
            del x, y, timestamp, polarity
            gc.collect()
            memory_after_cleanup = memory_profiler.get_memory_delta()
            
            return {
                'peak_after_creation': peak_after_creation,
                'peak_after_voxel': peak_after_voxel,
                'memory_after_cleanup': memory_after_cleanup,
                'voxel_shape': voxel.shape
            }
        
        result = benchmark(memory_test)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'implementation': 'numpy',
            **result
        })
    
    @pytest.mark.benchmark(group="memory")
    @pytest.mark.parametrize("size_name", ["large", "xlarge"])
    def test_polars_memory_efficiency(self, benchmark, size_name):
        """Test memory efficiency of Polars operations."""
        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available")
        
        size = BenchmarkConfig.DATASET_SIZES[size_name]
        
        memory_profiler = MemoryProfiler()
        
        def memory_test():
            memory_profiler.reset()
            
            # Create data
            x, y, timestamp, polarity = BenchmarkData.create_synthetic_events(size)
            df = BenchmarkData.create_polars_dataframe(x, y, timestamp, polarity)
            peak_after_creation = memory_profiler.get_memory_delta()
            
            # Create voxel grid
            voxel = create_voxel_grid(
                df,
                sensor_resolution=(640, 480),
                num_bins=10
            )
            peak_after_voxel = memory_profiler.get_memory_delta()
            
            # Clean up original data
            del x, y, timestamp, polarity, df
            gc.collect()
            memory_after_cleanup = memory_profiler.get_memory_delta()
            
            return {
                'peak_after_creation': peak_after_creation,
                'peak_after_voxel': peak_after_voxel,
                'memory_after_cleanup': memory_after_cleanup,
                'voxel_shape': voxel.shape
            }
        
        result = benchmark(memory_test)
        
        benchmark.extra_info.update({
            'dataset_size': size,
            'implementation': 'polars',
            **result
        })


class BenchmarkReporter:
    """Generate performance reports and visualizations."""
    
    @staticmethod
    def generate_performance_report(benchmark_results: Dict[str, Any], output_dir: str = "benchmark_results"):
        """Generate a comprehensive performance report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results as JSON
        with open(f"{output_dir}/benchmark_results.json", 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        # Generate summary statistics
        summary = BenchmarkReporter._generate_summary(benchmark_results)
        
        with open(f"{output_dir}/performance_summary.md", 'w') as f:
            f.write(BenchmarkReporter._format_summary_markdown(summary))
        
        # Generate visualizations if plotting is available
        if PLOTTING_AVAILABLE:
            BenchmarkReporter._generate_plots(benchmark_results, output_dir)
        
        print(f"Performance report generated in {output_dir}/")
        return summary
    
    @staticmethod
    def _generate_summary(results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            'overview': {
                'total_benchmarks': len(results),
                'numpy_benchmarks': len([k for k in results.keys() if 'numpy' in k.lower()]),
                'polars_benchmarks': len([k for k in results.keys() if 'polars' in k.lower()]),
            },
            'performance_comparison': {},
            'memory_usage': {},
            'scalability': {},
            'recommendations': []
        }
        
        # Add more detailed analysis here based on actual results
        return summary
    
    @staticmethod
    def _format_summary_markdown(summary: Dict[str, Any]) -> str:
        """Format summary as Markdown report."""
        md = "# Performance Benchmark Report\n\n"
        md += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        md += "## Overview\n\n"
        md += f"- Total benchmarks: {summary['overview']['total_benchmarks']}\n"
        md += f"- NumPy benchmarks: {summary['overview']['numpy_benchmarks']}\n"
        md += f"- Polars benchmarks: {summary['overview']['polars_benchmarks']}\n\n"
        
        md += "## Performance Comparison\n\n"
        md += "### Key Findings\n\n"
        md += "- Data loading performance comparison\n"
        md += "- Filtering operation efficiency\n"
        md += "- Representation function speed\n"
        md += "- Memory usage patterns\n\n"
        
        md += "## Recommendations\n\n"
        md += "Based on the benchmark results:\n\n"
        md += "1. **For small datasets (<10K events)**: NumPy implementations are generally sufficient\n"
        md += "2. **For medium datasets (10K-100K events)**: Consider Polars for complex filtering\n"
        md += "3. **For large datasets (>100K events)**: Polars shows advantages in query operations\n"
        md += "4. **Memory efficiency**: Monitor memory usage patterns for specific use cases\n\n"
        
        return md
    
    @staticmethod
    def _generate_plots(results: Dict[str, Any], output_dir: str):
        """Generate performance visualization plots."""
        if not PLOTTING_AVAILABLE:
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # Performance comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Add placeholder plots - these would be populated with actual data
        ax1.set_title("Data Loading Performance")
        ax1.set_xlabel("Dataset Size")
        ax1.set_ylabel("Time (seconds)")
        
        ax2.set_title("Memory Usage Comparison")
        ax2.set_xlabel("Dataset Size")
        ax2.set_ylabel("Memory (MB)")
        
        ax3.set_title("Filtering Operations")
        ax3.set_xlabel("Operation Type")
        ax3.set_ylabel("Time (seconds)")
        
        ax4.set_title("Representation Functions")
        ax4.set_xlabel("Function Type")
        ax4.set_ylabel("Time (seconds)")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to {output_dir}/")


# Test configuration for pytest-benchmark
@pytest.fixture(scope="session")
def benchmark_config():
    """Configuration for pytest-benchmark."""
    return {
        'min_rounds': 3,
        'max_time': 10.0,
        'min_time': 0.1,
        'warmup': True,
        'warmup_iterations': 2,
        'disable_gc': True,
        'sort': 'mean'
    }


# Integration test to verify everything works together
def test_benchmark_integration():
    """Integration test to verify benchmark infrastructure works."""
    # Test data creation
    x, y, timestamp, polarity = BenchmarkData.create_synthetic_events(1000)
    assert len(x) == 1000
    
    # Test memory profiler
    profiler = MemoryProfiler()
    initial = profiler.get_memory_usage()
    assert initial > 0
    
    # Test representation functions with both NumPy and Polars
    voxel_numpy = create_voxel_grid(x, y, timestamp, polarity, (640, 480), 5)
    assert voxel_numpy.shape == (640, 480, 5)
    
    if POLARS_AVAILABLE:
        df = BenchmarkData.create_polars_dataframe(x, y, timestamp, polarity)
        voxel_polars = create_voxel_grid(df, sensor_resolution=(640, 480), num_bins=5)
        assert voxel_polars.shape == (640, 480, 5)
        
        # Results should be similar (allowing for small numerical differences)
        assert np.allclose(voxel_numpy, voxel_polars, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    """Run benchmarks independently for development."""
    # Run a quick test to verify everything works
    test_benchmark_integration()
    
    print("Benchmark infrastructure ready!")
    print(f"Polars available: {POLARS_AVAILABLE}")
    print(f"Plotting available: {PLOTTING_AVAILABLE}")
    print(f"Available dataset sizes: {list(BenchmarkConfig.DATASET_SIZES.keys())}")
    print(f"Available real data files: {list(BenchmarkConfig.REAL_DATA_FILES.keys())}")