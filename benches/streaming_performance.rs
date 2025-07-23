use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use evlib::ev_core::Event;
use evlib::ev_formats::streaming::{PolarsEventStreamer, StreamingConfig};
use evlib::ev_formats::EventFormat;
use std::hint::black_box as hint_black_box;

/// Generate synthetic events with realistic patterns for streaming benchmarks
fn generate_events_with_pattern(count: usize, pattern: &str) -> Vec<Event> {
    let mut events = Vec::with_capacity(count);
    let mut rng = 42u64; // Simple LCG for reproducibility

    match pattern {
        "uniform" => {
            for i in 0..count {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let x = (rng % 640) as u16;

                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let y = (rng % 480) as u16;

                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let polarity = (rng % 2) == 0;

                events.push(Event {
                    t: i as f64 * 0.00001, // 10μs intervals
                    x,
                    y,
                    polarity,
                });
            }
        }
        "clustered" => {
            for i in 0..count {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let cluster_x = (rng % 4) as u16 * 160; // 4 clusters horizontally
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let cluster_y = (rng % 3) as u16 * 160; // 3 clusters vertically

                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let x = cluster_x + (rng % 160) as u16;
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let y = cluster_y + (rng % 160) as u16;

                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let polarity = (rng % 2) == 0;

                events.push(Event {
                    t: i as f64 * 0.00001,
                    x,
                    y,
                    polarity,
                });
            }
        }
        "sparse" => {
            for i in 0..count {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let x = (rng % 1280) as u16;

                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let y = (rng % 720) as u16;

                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let polarity = (rng % 2) == 0;

                events.push(Event {
                    t: i as f64 * 0.0001, // 100μs intervals (sparser)
                    x,
                    y,
                    polarity,
                });
            }
        }
        _ => panic!("Unknown pattern: {pattern}"),
    }

    events
}

/// Benchmark streaming with different chunk sizes
fn benchmark_streaming_chunk_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_chunk_sizes");

    let event_count = 5_000_000;
    let events = generate_events_with_pattern(event_count, "uniform");

    // Test different chunk sizes
    let chunk_sizes = vec![50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000];

    for chunk_size in chunk_sizes {
        group.throughput(Throughput::Elements(event_count as u64));

        group.bench_with_input(
            BenchmarkId::new("chunk_size", chunk_size),
            &(&events, chunk_size),
            |b, (events, chunk_size)| {
                b.iter(|| {
                    let streamer = PolarsEventStreamer::new(*chunk_size, EventFormat::HDF5);

                    #[cfg(feature = "polars")]
                    {
                        let result = streamer.stream_to_polars(events.iter().cloned());
                        match result {
                            Ok(df) => hint_black_box(df.height()),
                            Err(_) => hint_black_box(0),
                        }
                    }

                    #[cfg(not(feature = "polars"))]
                    {
                        hint_black_box(events.len());
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark adaptive chunk sizing performance
fn benchmark_adaptive_chunk_sizing(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_chunk_sizing");

    let event_counts = vec![1_000_000, 5_000_000, 10_000_000, 20_000_000];
    let memory_limits = vec![128, 256, 512, 1024]; // MB

    for event_count in event_counts {
        for memory_limit in &memory_limits {
            group.throughput(Throughput::Elements(event_count as u64));

            let events = generate_events_with_pattern(event_count, "uniform");

            group.bench_with_input(
                BenchmarkId::new(
                    "adaptive_sizing",
                    format!("{}M_events_{}MB", event_count / 1_000_000, memory_limit),
                ),
                &(&events, *memory_limit),
                |b, (events, memory_limit)| {
                    b.iter(|| {
                        let chunk_size = PolarsEventStreamer::calculate_optimal_chunk_size(
                            events.len(),
                            *memory_limit,
                        );
                        let streamer = PolarsEventStreamer::new(chunk_size, EventFormat::HDF5);

                        #[cfg(feature = "polars")]
                        {
                            let result = streamer.stream_to_polars(events.iter().cloned());
                            match result {
                                Ok(df) => hint_black_box(df.height()),
                                Err(_) => hint_black_box(0),
                            }
                        }

                        #[cfg(not(feature = "polars"))]
                        {
                            hint_black_box(events.len());
                        }
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark concatenation overhead for streaming
fn benchmark_concatenation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("concatenation_overhead");

    #[cfg(feature = "polars")]
    {
        use polars::prelude::*;

        let event_count = 5_000_000;
        let events = generate_events_with_pattern(event_count, "uniform");

        // Test different numbers of chunks
        let chunk_counts = vec![1, 5, 10, 20, 50];

        for chunk_count in chunk_counts {
            group.throughput(Throughput::Elements(event_count as u64));

            let chunk_size = event_count / chunk_count;

            group.bench_with_input(
                BenchmarkId::new("concat_chunks", chunk_count),
                &(&events, chunk_size),
                |b, (events, chunk_size)| {
                    b.iter(|| {
                        let streamer = PolarsEventStreamer::new(*chunk_size, EventFormat::HDF5);

                        // Create multiple chunks
                        let mut dataframes = Vec::new();
                        for chunk in events.chunks(*chunk_size) {
                            let chunk_df = streamer.build_chunk(chunk).unwrap();
                            if !chunk_df.is_empty() {
                                dataframes.push(chunk_df);
                            }
                        }

                        // Benchmark concatenation
                        let final_df = if dataframes.len() == 1 {
                            dataframes.into_iter().next().unwrap()
                        } else {
                            let lazy_frames: Vec<LazyFrame> =
                                dataframes.into_iter().map(|df| df.lazy()).collect();
                            concat(&lazy_frames, UnionArgs::default())
                                .unwrap()
                                .collect()
                                .unwrap()
                        };

                        hint_black_box(final_df.height());
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark streaming vs direct performance crossover
fn benchmark_streaming_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_crossover");

    // Test around the crossover point
    let event_counts = vec![1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000];

    for count in event_counts {
        group.throughput(Throughput::Elements(count as u64));

        let events = generate_events_with_pattern(count, "uniform");

        // Benchmark direct processing (single chunk)
        group.bench_with_input(
            BenchmarkId::new("direct_processing", count),
            &events,
            |b, events| {
                b.iter(|| {
                    let streamer = PolarsEventStreamer::new(events.len(), EventFormat::HDF5);

                    #[cfg(feature = "polars")]
                    {
                        let result = streamer.stream_to_polars(events.iter().cloned());
                        match result {
                            Ok(df) => hint_black_box(df.height()),
                            Err(_) => hint_black_box(0),
                        }
                    }

                    #[cfg(not(feature = "polars"))]
                    {
                        hint_black_box(events.len());
                    }
                })
            },
        );

        // Benchmark streaming processing
        group.bench_with_input(
            BenchmarkId::new("streaming_processing", count),
            &events,
            |b, events| {
                b.iter(|| {
                    let chunk_size =
                        PolarsEventStreamer::calculate_optimal_chunk_size(events.len(), 256);
                    let streamer = PolarsEventStreamer::new(chunk_size, EventFormat::HDF5);

                    #[cfg(feature = "polars")]
                    {
                        let result = streamer.stream_to_polars(events.iter().cloned());
                        match result {
                            Ok(df) => hint_black_box(df.height()),
                            Err(_) => hint_black_box(0),
                        }
                    }

                    #[cfg(not(feature = "polars"))]
                    {
                        hint_black_box(events.len());
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark streaming with different event patterns
fn benchmark_event_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_patterns");

    let event_count = 3_000_000;
    let patterns = vec!["uniform", "clustered", "sparse"];

    for pattern in patterns {
        group.throughput(Throughput::Elements(event_count as u64));

        let events = generate_events_with_pattern(event_count, pattern);

        group.bench_with_input(
            BenchmarkId::new("pattern_streaming", pattern),
            &events,
            |b, events| {
                b.iter(|| {
                    let chunk_size =
                        PolarsEventStreamer::calculate_optimal_chunk_size(events.len(), 256);
                    let streamer = PolarsEventStreamer::new(chunk_size, EventFormat::HDF5);

                    #[cfg(feature = "polars")]
                    {
                        let result = streamer.stream_to_polars(events.iter().cloned());
                        match result {
                            Ok(df) => hint_black_box(df.height()),
                            Err(_) => hint_black_box(0),
                        }
                    }

                    #[cfg(not(feature = "polars"))]
                    {
                        hint_black_box(events.len());
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark streaming configuration impact
fn benchmark_streaming_configuration(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_configuration");

    let event_count = 5_000_000;
    let events = generate_events_with_pattern(event_count, "uniform");

    // Test different streaming configurations
    let configs = vec![
        ("default", StreamingConfig::default()),
        (
            "high_memory",
            StreamingConfig {
                chunk_size: 2_000_000,
                _memory_limit_mb: 1024,
                progress_interval: 5_000_000,
            },
        ),
        (
            "low_memory",
            StreamingConfig {
                chunk_size: 250_000,
                _memory_limit_mb: 128,
                progress_interval: 1_000_000,
            },
        ),
        (
            "optimized",
            StreamingConfig {
                chunk_size: 1_000_000,
                _memory_limit_mb: 512,
                progress_interval: 10_000_000,
            },
        ),
    ];

    for (config_name, config) in configs {
        group.throughput(Throughput::Elements(event_count as u64));

        group.bench_with_input(
            BenchmarkId::new("config", config_name),
            &(&events, config),
            |b, (events, config)| {
                b.iter(|| {
                    let streamer =
                        PolarsEventStreamer::with_config(config.clone(), EventFormat::HDF5);

                    #[cfg(feature = "polars")]
                    {
                        let result = streamer.stream_to_polars(events.iter().cloned());
                        match result {
                            Ok(df) => hint_black_box(df.height()),
                            Err(_) => hint_black_box(0),
                        }
                    }

                    #[cfg(not(feature = "polars"))]
                    {
                        hint_black_box(events.len());
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_streaming_chunk_sizes,
    benchmark_adaptive_chunk_sizing,
    benchmark_concatenation_overhead,
    benchmark_streaming_crossover,
    benchmark_event_patterns,
    benchmark_streaming_configuration
);
criterion_main!(benches);
