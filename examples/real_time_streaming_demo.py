#!/usr/bin/env python3
"""
Real-time Streaming Processing Demo for evlib

This example demonstrates evlib's real-time event processing capabilities
with sub-50ms latency, configurable temporal batching, and performance monitoring.

Requirements:
- evlib built with streaming features
- Optional: event data file for testing

Usage:
    python real_time_streaming_demo.py

Author: evlib contributors
"""

import time
import numpy as np
import threading
import queue

try:
    import evlib

    print("✅ evlib imported successfully")
except ImportError as e:
    print(f"❌ Failed to import evlib: {e}")
    exit(1)


class EventStreamGenerator:
    """Simulates a real-time event stream"""

    def __init__(self, event_rate=10000, duration=10.0):
        self.event_rate = event_rate  # events per second
        self.duration = duration
        self.running = False

    def generate_events(self, event_queue):
        """Generate synthetic events at specified rate"""
        start_time = time.time()
        event_count = 0

        while self.running and (time.time() - start_time) < self.duration:
            current_time = time.time() - start_time

            # Generate a batch of events
            batch_size = max(1, int(self.event_rate * 0.01))  # 10ms worth of events

            for _ in range(batch_size):
                # Create synthetic event
                event = {
                    "x": np.random.randint(0, 640),
                    "y": np.random.randint(0, 480),
                    "t": current_time + np.random.uniform(0, 0.01),
                    "p": np.random.choice([-1, 1]),
                }
                event_queue.put(event)
                event_count += 1

            # Control event rate
            time.sleep(0.01)  # 10ms intervals

        print(f"📊 Generated {event_count} events over {self.duration:.1f}s")

    def start(self, event_queue):
        """Start event generation in separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self.generate_events, args=(event_queue,))
        self.thread.start()

    def stop(self):
        """Stop event generation"""
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()


class RealTimeProcessor:
    """Real-time event processor with configurable parameters"""

    def __init__(self, config=None):
        self.config = config or {
            "batch_size": 1000,  # events per batch
            "temporal_window_ms": 20,  # temporal batching window
            "max_latency_ms": 50,  # maximum allowed latency
            "buffer_size": 10000,  # event buffer size
            "processing_threads": 2,  # parallel processing threads
        }

        self.stats = {
            "events_processed": 0,
            "batches_processed": 0,
            "total_latency_ms": 0,
            "max_latency_ms": 0,
            "dropped_events": 0,
            "start_time": None,
        }

        print("🔧 Real-time processor initialized:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

    def process_event_batch(self, events):
        """Process a batch of events"""
        batch_start = time.time()

        if not events:
            return

        # Convert to numpy arrays for processing
        xs = np.array([e["x"] for e in events])
        ys = np.array([e["y"] for e in events])
        ts = np.array([e["t"] for e in events])
        ps = np.array([e["p"] for e in events])

        # Simulate event processing (e.g., voxel grid generation)
        try:
            # This would call actual evlib processing functions
            # For demo, we simulate processing time
            processing_time = np.random.uniform(0.005, 0.015)  # 5-15ms
            time.sleep(processing_time)

            # Create sample voxel grid representation
            voxel_grid = evlib.create_voxel_grid(
                xs, ys, ts, ps, num_bins=5, resolution=(640, 480), method="count"
            )

            # Update statistics
            batch_latency = (time.time() - batch_start) * 1000  # ms
            self.stats["events_processed"] += len(events)
            self.stats["batches_processed"] += 1
            self.stats["total_latency_ms"] += batch_latency
            self.stats["max_latency_ms"] = max(self.stats["max_latency_ms"], batch_latency)

            return voxel_grid

        except Exception as e:
            print(f"❌ Processing error: {e}")
            return None

    def get_performance_stats(self):
        """Get current performance statistics"""
        if self.stats["batches_processed"] == 0:
            return None

        elapsed_time = time.time() - self.stats["start_time"]
        avg_latency = self.stats["total_latency_ms"] / self.stats["batches_processed"]
        throughput = self.stats["events_processed"] / elapsed_time

        return {
            "elapsed_time_s": elapsed_time,
            "events_processed": self.stats["events_processed"],
            "batches_processed": self.stats["batches_processed"],
            "avg_latency_ms": avg_latency,
            "max_latency_ms": self.stats["max_latency_ms"],
            "throughput_events_per_s": throughput,
            "dropped_events": self.stats["dropped_events"],
            "meets_latency_target": self.stats["max_latency_ms"] <= self.config["max_latency_ms"],
        }


def demonstrate_basic_streaming():
    """Demonstrate basic real-time streaming"""
    print("🚀 Basic Real-time Streaming Demo")
    print("=" * 40)

    # Create processor and event generator
    processor = RealTimeProcessor()
    generator = EventStreamGenerator(event_rate=5000, duration=5.0)

    # Start streaming
    event_queue = queue.Queue(maxsize=processor.config["buffer_size"])

    print("📡 Starting event stream...")
    generator.start(event_queue)
    processor.stats["start_time"] = time.time()

    # Process events in real-time
    current_batch = []
    last_process_time = time.time()

    try:
        while generator.running or not event_queue.empty():
            try:
                # Get event with timeout
                event = event_queue.get(timeout=0.1)
                current_batch.append(event)

                # Check if we should process current batch
                should_process = False

                # Batch size trigger
                if len(current_batch) >= processor.config["batch_size"]:
                    should_process = True

                # Temporal window trigger
                time_since_last = (time.time() - last_process_time) * 1000
                if time_since_last >= processor.config["temporal_window_ms"]:
                    should_process = True

                if should_process and current_batch:
                    result = processor.process_event_batch(current_batch)
                    if result is not None:
                        print(
                            f"✅ Processed batch: {len(current_batch)} events → " f"voxel grid {result.shape}"
                        )

                    current_batch = []
                    last_process_time = time.time()

            except queue.Empty:
                # Process remaining events if temporal window expired
                if current_batch:
                    time_since_last = (time.time() - last_process_time) * 1000
                    if time_since_last >= processor.config["temporal_window_ms"]:
                        processor.process_event_batch(current_batch)
                        current_batch = []
                        last_process_time = time.time()
                continue

    except KeyboardInterrupt:
        print("\n⏹️ Streaming interrupted by user")

    # Stop generator and process final batch
    generator.stop()
    if current_batch:
        processor.process_event_batch(current_batch)

    # Print performance statistics
    stats = processor.get_performance_stats()
    if stats:
        print("\n📊 Performance Statistics:")
        print(f"  Events processed: {stats['events_processed']:,}")
        print(f"  Batches processed: {stats['batches_processed']}")
        print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"  Maximum latency: {stats['max_latency_ms']:.2f}ms")
        print(f"  Throughput: {stats['throughput_events_per_s']:.0f} events/second")
        print(f"  Latency target met: {'✅ Yes' if stats['meets_latency_target'] else '❌ No'}")


def demonstrate_adaptive_batching():
    """Demonstrate adaptive batching based on event rate"""
    print("\n🎛️ Adaptive Batching Demo")
    print("=" * 40)

    # Test different event rates
    test_rates = [1000, 5000, 20000, 50000]  # events per second

    for rate in test_rates:
        print(f"\n📈 Testing event rate: {rate:,} events/second")

        # Adapt batch size based on event rate
        if rate < 5000:
            batch_size = 500
            temporal_window = 50  # ms
        elif rate < 20000:
            batch_size = 1000
            temporal_window = 20
        else:
            batch_size = 2000
            temporal_window = 10

        config = {
            "batch_size": batch_size,
            "temporal_window_ms": temporal_window,
            "max_latency_ms": 50,
            "buffer_size": 20000,
        }

        print(f"  Adapted config: batch_size={batch_size}, window={temporal_window}ms")

        # Run short test
        processor = RealTimeProcessor(config)
        generator = EventStreamGenerator(event_rate=rate, duration=2.0)

        event_queue = queue.Queue(maxsize=config["buffer_size"])
        generator.start(event_queue)
        processor.stats["start_time"] = time.time()

        # Quick processing loop
        processed_batches = 0
        start_time = time.time()

        while (time.time() - start_time) < 2.0 and (generator.running or not event_queue.empty()):
            batch = []
            batch_start = time.time()

            # Collect events for one temporal window
            while (time.time() - batch_start) < (temporal_window / 1000) and len(batch) < batch_size:
                try:
                    event = event_queue.get(timeout=0.001)
                    batch.append(event)
                except queue.Empty:
                    break

            if batch:
                processor.process_event_batch(batch)
                processed_batches += 1

        generator.stop()

        # Show results
        stats = processor.get_performance_stats()
        if stats:
            print(
                f"  Results: {stats['avg_latency_ms']:.1f}ms avg latency, "
                f"{stats['throughput_events_per_s']:.0f} events/s throughput"
            )


def demonstrate_performance_monitoring():
    """Demonstrate real-time performance monitoring"""
    print("\n📊 Performance Monitoring Demo")
    print("=" * 40)

    processor = RealTimeProcessor(
        {"batch_size": 1000, "temporal_window_ms": 25, "max_latency_ms": 50, "buffer_size": 15000}
    )

    generator = EventStreamGenerator(event_rate=10000, duration=8.0)
    event_queue = queue.Queue(maxsize=15000)

    # Start monitoring
    print("📡 Starting performance monitoring...")
    generator.start(event_queue)
    processor.stats["start_time"] = time.time()

    # Process with periodic monitoring
    monitor_interval = 1.0  # seconds
    last_monitor = time.time()
    current_batch = []

    while generator.running or not event_queue.empty():
        try:
            event = event_queue.get(timeout=0.1)
            current_batch.append(event)

            # Process batch when ready
            if len(current_batch) >= processor.config["batch_size"]:
                processor.process_event_batch(current_batch)
                current_batch = []

            # Periodic monitoring
            if time.time() - last_monitor >= monitor_interval:
                stats = processor.get_performance_stats()
                if stats:
                    print(
                        f"⏱️ {stats['elapsed_time_s']:.1f}s: "
                        f"{stats['events_processed']:,} events, "
                        f"{stats['avg_latency_ms']:.1f}ms avg latency, "
                        f"{stats['throughput_events_per_s']:.0f} events/s"
                    )

                last_monitor = time.time()

        except queue.Empty:
            continue

    generator.stop()

    # Final statistics
    final_stats = processor.get_performance_stats()
    if final_stats:
        print("\n🏁 Final Performance Report:")
        print(f"  Total events: {final_stats['events_processed']:,}")
        print(f"  Processing time: {final_stats['elapsed_time_s']:.2f}s")
        print(f"  Average latency: {final_stats['avg_latency_ms']:.2f}ms")
        print(f"  Peak latency: {final_stats['max_latency_ms']:.2f}ms")
        print(f"  Average throughput: {final_stats['throughput_events_per_s']:.0f} events/s")
        print(f"  Latency SLA: {'✅ Met' if final_stats['meets_latency_target'] else '❌ Missed'}")


def main():
    """Main demo function"""
    print("⚡ Real-time Streaming Processing Demo")
    print("=" * 50)
    print("This demo showcases evlib's real-time event processing capabilities")
    print("with sub-50ms latency and configurable temporal batching.\n")

    # Run demonstrations
    demonstrate_basic_streaming()
    demonstrate_adaptive_batching()
    demonstrate_performance_monitoring()

    print("\n🎉 Demo completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    print("  ⚡ Sub-50ms processing latency")
    print("  🎛️ Configurable temporal batching")
    print("  📊 Real-time performance monitoring")
    print("  🔧 Adaptive configuration based on event rate")
    print("  🚀 High-throughput event processing")

    print("\n📚 Next Steps:")
    print("  - Test with real event camera data")
    print("  - Integrate with reconstruction models")
    print("  - Experiment with different batch configurations")
    print("  - Monitor performance in production")


if __name__ == "__main__":
    main()
