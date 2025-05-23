"""
Performance benchmark tests for E2VID reconstruction.

This module provides comprehensive benchmarks to measure the performance
characteristics of the event-to-video reconstruction pipeline.
"""

import pytest
import numpy as np
import time

try:
    import evlib

    EVLIB_AVAILABLE = True
except ImportError:
    EVLIB_AVAILABLE = False

# Skip all tests if evlib is not available
pytestmark = pytest.mark.skipif(not EVLIB_AVAILABLE, reason="evlib not available")


class TestE2VidBenchmarks:
    """Benchmark tests for E2VID reconstruction performance."""

    @pytest.fixture
    def benchmark_events(self):
        """Generate events for benchmarking."""

        def _generate_events(num_events, width=128, height=128):
            events = []
            for i in range(num_events):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                t = i * 0.0001
                polarity = 1 if i % 2 == 0 else -1
                events.append((x, y, t, polarity))

            return evlib.core.Events.from_arrays(
                [e[0] for e in events], [e[1] for e in events], [e[2] for e in events], [e[3] for e in events]
            )

        return _generate_events

    @pytest.fixture
    def e2vid_reconstructor(self):
        """Create E2VID reconstructor for benchmarking."""
        return evlib.processing.E2Vid(128, 128)

    def test_simple_reconstruction_scaling(self, benchmark_events, e2vid_reconstructor):
        """Test how simple reconstruction scales with event count."""
        event_counts = [1000, 5000, 10000, 25000, 50000]
        times = []

        for count in event_counts:
            events = benchmark_events(count)

            # Measure reconstruction time
            start_time = time.time()
            result = e2vid_reconstructor.process_events_simple(events)
            reconstruction_time = time.time() - start_time

            times.append(reconstruction_time)

            # Verify result
            assert result.shape == (128, 128)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

        # Performance should scale reasonably (not exponentially)
        # Time for 50k events should be less than 10x time for 5k events
        time_ratio = times[-1] / times[1]  # 50k vs 5k
        assert time_ratio < 10, f"Poor scaling: {time_ratio:.2f}x slower for 10x more events"

        # Print performance results
        print("\nSimple Reconstruction Scaling:")
        for count, time_taken in zip(event_counts, times):
            throughput = count / time_taken
            print(f"  {count:6d} events: {time_taken:.4f}s ({throughput:.0f} events/s)")

    def test_neural_reconstruction_scaling(self, benchmark_events, e2vid_reconstructor):
        """Test how neural reconstruction scales with event count."""
        # Initialize neural network
        e2vid_reconstructor.create_default_network()

        event_counts = [1000, 5000, 10000, 25000]  # Smaller counts for neural network
        times = []

        for count in event_counts:
            events = benchmark_events(count)

            # Measure reconstruction time
            start_time = time.time()
            result = e2vid_reconstructor.process_events(events)
            reconstruction_time = time.time() - start_time

            times.append(reconstruction_time)

            # Verify result
            assert result.shape == (128, 128)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

        # Print performance results
        print("\nNeural Reconstruction Scaling:")
        for count, time_taken in zip(event_counts, times):
            throughput = count / time_taken
            print(f"  {count:6d} events: {time_taken:.4f}s ({throughput:.0f} events/s)")

    def test_image_size_scaling(self, benchmark_events):
        """Test how reconstruction scales with image size."""
        sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
        fixed_events = 10000

        print("\nImage Size Scaling (10k events):")
        print("Size      Simple Time   Neural Time   Simple Throughput   Neural Throughput")
        print("-" * 80)

        for width, height in sizes:
            # Create reconstructor for this size
            reconstructor = evlib.processing.E2Vid(height, width)

            # Generate events for this image size
            events = []
            for i in range(fixed_events):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                t = i * 0.0001
                polarity = 1 if i % 2 == 0 else -1
                events.append((x, y, t, polarity))

            events_evlib = evlib.core.Events.from_arrays(
                [e[0] for e in events], [e[1] for e in events], [e[2] for e in events], [e[3] for e in events]
            )

            # Simple reconstruction
            start_time = time.time()
            simple_result = reconstructor.process_events_simple(events_evlib)
            simple_time = time.time() - start_time

            # Neural reconstruction
            reconstructor.create_default_network()
            start_time = time.time()
            neural_result = reconstructor.process_events(events_evlib)
            neural_time = time.time() - start_time

            # Verify results
            assert simple_result.shape == (height, width)
            assert neural_result.shape == (height, width)

            simple_throughput = fixed_events / simple_time
            neural_throughput = fixed_events / neural_time

            print(
                f"{width:3d}x{height:<3d}   {simple_time:8.4f}s   {neural_time:8.4f}s   "
                f"{simple_throughput:12.0f}       {neural_throughput:12.0f}"
            )

    def test_memory_usage_scaling(self, benchmark_events):
        """Test memory usage with different configurations."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"\nInitial memory usage: {initial_memory:.1f} MB")

        # Test different image sizes
        sizes = [(64, 64), (128, 128), (256, 256)]

        for width, height in sizes:
            reconstructor = evlib.processing.E2Vid(height, width)
            events = benchmark_events(5000, width, height)

            # Perform reconstruction
            reconstructor.process_events_simple(events)

            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            print(
                f"After {width}x{height} reconstruction: {current_memory:.1f} MB "
                f"(+{memory_increase:.1f} MB)"
            )

            # Memory increase should be reasonable
            assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f} MB"

    @pytest.mark.parametrize("num_bins", [3, 5, 7, 9])
    def test_voxel_bins_performance(self, benchmark_events, num_bins):
        """Test performance impact of different voxel bin counts."""
        config = {
            "num_bins": num_bins,
            "intensity_scale": 1.0,
            "intensity_offset": 0.0,
        }

        reconstructor = evlib.processing.E2Vid.with_config(128, 128, config)
        events = benchmark_events(10000)

        # Measure reconstruction time
        start_time = time.time()
        result = reconstructor.process_events_simple(events)
        reconstruction_time = time.time() - start_time

        # Verify result
        assert result.shape == (128, 128)

        throughput = 10000 / reconstruction_time
        print(
            f"Bins: {num_bins}, Time: {reconstruction_time:.4f}s, " f"Throughput: {throughput:.0f} events/s"
        )

        # More bins should not dramatically slow down reconstruction
        assert reconstruction_time < 1.0, f"Reconstruction too slow with {num_bins} bins"

    def test_repeated_reconstruction_performance(self, benchmark_events, e2vid_reconstructor):
        """Test performance consistency over repeated reconstructions."""
        events = benchmark_events(10000)
        times = []

        # Warm up
        for _ in range(3):
            e2vid_reconstructor.process_events_simple(events)

        # Measure repeated reconstructions
        num_iterations = 10
        for i in range(num_iterations):
            start_time = time.time()
            result = e2vid_reconstructor.process_events_simple(events)
            reconstruction_time = time.time() - start_time
            times.append(reconstruction_time)

            assert result.shape == (128, 128)

        # Performance should be consistent
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time  # Coefficient of variation

        print("\nRepeated Reconstruction Performance:")
        print(f"  Mean time: {mean_time:.4f}s")
        print(f"  Std dev: {std_time:.4f}s")
        print(f"  Coefficient of variation: {cv:.3f}")

        # Performance should be reasonably consistent (CV < 20%)
        assert cv < 0.2, f"Inconsistent performance: CV = {cv:.3f}"

    def test_concurrent_reconstruction(self, benchmark_events):
        """Test performance with multiple concurrent reconstructors."""
        import threading
        import time

        def reconstruct_worker(worker_id, events, results, times):
            """Worker function for concurrent reconstruction."""
            reconstructor = evlib.processing.E2Vid(64, 64)  # Smaller for concurrency

            start_time = time.time()
            result = reconstructor.process_events_simple(events)
            reconstruction_time = time.time() - start_time

            results[worker_id] = result
            times[worker_id] = reconstruction_time

        # Generate events for all workers
        events = benchmark_events(5000, 64, 64)

        # Test with different numbers of concurrent workers
        for num_workers in [1, 2, 4]:
            results = {}
            times = {}
            threads = []

            overall_start = time.time()

            # Create and start worker threads
            for i in range(num_workers):
                thread = threading.Thread(target=reconstruct_worker, args=(i, events, results, times))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            overall_time = time.time() - overall_start

            # Verify all results
            for i in range(num_workers):
                assert results[i].shape == (64, 64)
                assert np.all(results[i] >= 0.0)
                assert np.all(results[i] <= 1.0)

            mean_worker_time = np.mean(list(times.values()))
            total_throughput = (num_workers * 5000) / overall_time

            print(
                f"Workers: {num_workers}, Overall time: {overall_time:.3f}s, "
                f"Mean worker time: {mean_worker_time:.3f}s, "
                f"Total throughput: {total_throughput:.0f} events/s"
            )


class TestE2VidStressTests:
    """Stress tests for E2VID reconstruction."""

    def test_large_event_stream(self):
        """Test reconstruction with very large event streams."""
        if not EVLIB_AVAILABLE:
            pytest.skip("evlib not available")

        # Generate large event stream
        num_events = 100000
        print(f"\nStress test with {num_events} events...")

        events = []
        for i in range(num_events):
            x = np.random.randint(0, 256)
            y = np.random.randint(0, 256)
            t = i * 0.00001  # High temporal resolution
            polarity = 1 if i % 2 == 0 else -1
            events.append((x, y, t, polarity))

        events_evlib = evlib.core.Events.from_arrays(
            [e[0] for e in events], [e[1] for e in events], [e[2] for e in events], [e[3] for e in events]
        )

        reconstructor = evlib.processing.E2Vid(256, 256)

        start_time = time.time()
        result = reconstructor.process_events_simple(events_evlib)
        reconstruction_time = time.time() - start_time

        assert result.shape == (256, 256)
        throughput = num_events / reconstruction_time

        print(f"Large stream reconstruction: {reconstruction_time:.3f}s " f"({throughput:.0f} events/s)")

        # Should handle large streams reasonably
        assert reconstruction_time < 30.0, "Large stream reconstruction too slow"

    def test_extreme_configurations(self):
        """Test reconstruction with extreme configurations."""
        if not EVLIB_AVAILABLE:
            pytest.skip("evlib not available")

        # Test extreme configurations
        extreme_configs = [
            {"num_bins": 1, "intensity_scale": 0.1, "intensity_offset": 0.9},
            {"num_bins": 15, "intensity_scale": 5.0, "intensity_offset": 0.0},
            {"num_bins": 5, "intensity_scale": 0.0, "intensity_offset": 1.0},
        ]

        events = []
        for i in range(1000):
            events.append(
                (np.random.randint(0, 64), np.random.randint(0, 64), i * 0.001, 1 if i % 2 == 0 else -1)
            )

        events_evlib = evlib.core.Events.from_arrays(
            [e[0] for e in events], [e[1] for e in events], [e[2] for e in events], [e[3] for e in events]
        )

        for i, config in enumerate(extreme_configs):
            print(f"\nTesting extreme config {i+1}: {config}")

            reconstructor = evlib.processing.E2Vid.with_config(64, 64, config)

            result = reconstructor.process_events_simple(events_evlib)

            # Should still produce valid output
            assert result.shape == (64, 64)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))

            print(f"  Result range: [{np.min(result):.3f}, {np.max(result):.3f}]")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
