// Standalone benchmark for evlib performance measurements
// This benchmark doesn't require the main evlib library and focuses on core algorithms

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::hint::black_box;

/// Simple Event structure mirroring the main library
#[derive(Clone, Debug)]
struct Event {
    t: f64,
    x: u16,
    y: u16,
    polarity: bool,
}

/// Generate synthetic events for consistent benchmarking
fn generate_events(count: usize, width: u16, height: u16) -> Vec<Event> {
    let mut events = Vec::with_capacity(count);
    let mut rng = 42u64; // Simple LCG for reproducibility
    
    for i in 0..count {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = (rng % width as u64) as u16;
        
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = (rng % height as u64) as u16;
        
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let polarity = (rng % 2) == 0;
        
        events.push(Event {
            t: i as f64 * 0.00001, // 10μs intervals
            x,
            y,
            polarity,
        });
    }
    
    events
}

/// Benchmark memory efficiency: direct vs streaming approaches
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    let event_counts = vec![1_000_000, 2_500_000, 5_000_000, 10_000_000];
    
    for count in event_counts {
        group.throughput(Throughput::Elements(count as u64));
        
        let events = generate_events(count, 640, 480);
        
        // Direct processing (load everything into memory)
        group.bench_with_input(
            BenchmarkId::new("direct_processing", count),
            &events,
            |b, events| {
                b.iter(|| {
                    let processed: Vec<_> = events.iter()
                        .map(|e| (e.x as u32, e.y as u32, (e.t * 1_000_000.0) as i64, if e.polarity { 1i8 } else { 0i8 }))
                        .collect();
                    black_box(processed.len())
                })
            },
        );
        
        // Chunked processing (streaming approach)
        group.bench_with_input(
            BenchmarkId::new("chunked_processing", count),
            &events,
            |b, events| {
                b.iter(|| {
                    let chunk_size = calculate_optimal_chunk_size(events.len(), 256);
                    let mut processed_count = 0;
                    
                    for chunk in events.chunks(chunk_size) {
                        let _processed: Vec<_> = chunk.iter()
                            .map(|e| (e.x as u32, e.y as u32, (e.t * 1_000_000.0) as i64, if e.polarity { 1i8 } else { 0i8 }))
                            .collect();
                        processed_count += chunk.len();
                    }
                    
                    black_box(processed_count)
                })
            },
        );
    }
    
    group.finish();
}

/// Calculate optimal chunk size based on memory constraints
fn calculate_optimal_chunk_size(total_events: usize, available_memory_mb: usize) -> usize {
    const BYTES_PER_EVENT: usize = 15;
    
    let target_memory_bytes = (available_memory_mb * 1024 * 1024) / 4;
    let memory_based_chunk_size = target_memory_bytes / BYTES_PER_EVENT;
    
    let chunk_size = memory_based_chunk_size.clamp(100_000, 10_000_000);
    
    if total_events > 100_000_000 {
        chunk_size.min(1_000_000)
    } else {
        chunk_size
    }
}

/// Benchmark streaming performance with different chunk sizes
fn benchmark_streaming_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_performance");
    
    let event_count = 5_000_000;
    let events = generate_events(event_count, 640, 480);
    
    let chunk_sizes = vec![100_000, 500_000, 1_000_000, 2_000_000];
    
    for chunk_size in chunk_sizes {
        group.throughput(Throughput::Elements(event_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("chunk_size", chunk_size),
            &(&events, chunk_size),
            |b, (events, chunk_size)| {
                b.iter(|| {
                    let mut processed_count = 0;
                    
                    for chunk in events.chunks(*chunk_size) {
                        // Simulate processing with polarity conversion
                        let _processed: Vec<_> = chunk.iter()
                            .map(|e| {
                                let polarity = if e.polarity { 1i8 } else { 0i8 };
                                let timestamp = (e.t * 1_000_000.0) as i64;
                                (e.x, e.y, timestamp, polarity)
                            })
                            .collect();
                        processed_count += chunk.len();
                    }
                    
                    black_box(processed_count)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark format-specific polarity encoding
fn benchmark_format_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_comparison");
    
    let event_count = 1_000_000;
    let events = generate_events(event_count, 640, 480);
    
    group.throughput(Throughput::Elements(event_count as u64));
    
    // HDF5 format (0/1 encoding)
    group.bench_with_input(
        BenchmarkId::new("hdf5_polarity", "0_1_encoding"),
        &events,
        |b, events| {
            b.iter(|| {
                let processed: Vec<_> = events.iter()
                    .map(|e| if e.polarity { 1i8 } else { 0i8 })
                    .collect();
                black_box(processed.len())
            })
        },
    );
    
    // EVT2 format (-1/1 encoding)
    group.bench_with_input(
        BenchmarkId::new("evt2_polarity", "neg1_1_encoding"),
        &events,
        |b, events| {
            b.iter(|| {
                let processed: Vec<_> = events.iter()
                    .map(|e| if e.polarity { 1i8 } else { -1i8 })
                    .collect();
                black_box(processed.len())
            })
        },
    );
    
    // Timestamp conversion performance
    group.bench_with_input(
        BenchmarkId::new("timestamp_conversion", "seconds_to_microseconds"),
        &events,
        |b, events| {
            b.iter(|| {
                let processed: Vec<_> = events.iter()
                    .map(|e| {
                        if e.t > 1_000_000.0 {
                            e.t as i64 // Already in microseconds
                        } else {
                            (e.t * 1_000_000.0) as i64 // Convert seconds to microseconds
                        }
                    })
                    .collect();
                black_box(processed.len())
            })
        },
    );
    
    group.finish();
}

/// Benchmark memory usage estimation
fn benchmark_memory_usage_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_estimation");
    
    let event_counts = vec![1_000_000, 5_000_000, 10_000_000, 50_000_000];
    
    for count in event_counts {
        group.throughput(Throughput::Elements(count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("estimate_memory", count),
            &count,
            |b, count| {
                b.iter(|| {
                    const BYTES_PER_EVENT: usize = 30;
                    let estimated_bytes = count * BYTES_PER_EVENT;
                    let estimated_mb = estimated_bytes / (1024 * 1024);
                    black_box(estimated_mb)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark should_use_streaming decision
fn benchmark_streaming_decision(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_decision");
    
    let event_counts = vec![1_000_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000];
    
    for count in event_counts {
        group.throughput(Throughput::Elements(count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("should_use_streaming", count),
            &count,
            |b, count| {
                b.iter(|| {
                    let default_threshold = 5_000_000;
                    let decision = *count > default_threshold;
                    black_box(decision)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark data type efficiency
fn benchmark_data_type_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_type_efficiency");
    
    let event_count = 2_000_000;
    let events = generate_events(event_count, 1024, 768);
    
    group.throughput(Throughput::Elements(event_count as u64));
    
    // Optimal types (i16 for coordinates, i8 for polarity)
    group.bench_with_input(
        BenchmarkId::new("optimal_types", "i16_i8"),
        &events,
        |b, events| {
            b.iter(|| {
                let processed: Vec<_> = events.iter()
                    .map(|e| (e.x as i16, e.y as i16, if e.polarity { 1i8 } else { 0i8 }))
                    .collect();
                black_box(processed.len())
            })
        },
    );
    
    // Suboptimal types (i32 for coordinates, i32 for polarity)
    group.bench_with_input(
        BenchmarkId::new("suboptimal_types", "i32_i32"),
        &events,
        |b, events| {
            b.iter(|| {
                let processed: Vec<_> = events.iter()
                    .map(|e| (e.x as i32, e.y as i32, if e.polarity { 1i32 } else { 0i32 }))
                    .collect();
                black_box(processed.len())
            })
        },
    );
    
    group.finish();
}

/// Benchmark adaptive chunk sizing
fn benchmark_adaptive_chunk_sizing(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_chunk_sizing");
    
    let event_counts = vec![1_000_000, 5_000_000, 10_000_000, 20_000_000];
    let memory_limits = vec![128, 256, 512, 1024]; // MB
    
    for event_count in event_counts {
        for memory_limit in &memory_limits {
            group.throughput(Throughput::Elements(event_count as u64));
            
            let events = generate_events(event_count, 640, 480);
            
            group.bench_with_input(
                BenchmarkId::new("adaptive_sizing", format!("{}M_events_{}MB", event_count / 1_000_000, memory_limit)),
                &(&events, *memory_limit),
                |b, (events, memory_limit)| {
                    b.iter(|| {
                        let chunk_size = calculate_optimal_chunk_size(events.len(), *memory_limit);
                        let mut processed_count = 0;
                        
                        for chunk in events.chunks(chunk_size) {
                            let _processed: Vec<_> = chunk.iter()
                                .map(|e| (e.x, e.y, (e.t * 1_000_000.0) as i64, if e.polarity { 1i8 } else { 0i8 }))
                                .collect();
                            processed_count += chunk.len();
                        }
                        
                        black_box(processed_count)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark crossover point between direct and streaming
fn benchmark_crossover_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover_point");
    
    let event_counts = vec![1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000];
    
    for count in event_counts {
        group.throughput(Throughput::Elements(count as u64));
        
        let events = generate_events(count, 640, 480);
        
        // Direct processing
        group.bench_with_input(
            BenchmarkId::new("direct", count),
            &events,
            |b, events| {
                b.iter(|| {
                    let processed: Vec<_> = events.iter()
                        .map(|e| (e.x, e.y, (e.t * 1_000_000.0) as i64, if e.polarity { 1i8 } else { 0i8 }))
                        .collect();
                    black_box(processed.len())
                })
            },
        );
        
        // Streaming processing
        group.bench_with_input(
            BenchmarkId::new("streaming", count),
            &events,
            |b, events| {
                b.iter(|| {
                    let chunk_size = calculate_optimal_chunk_size(events.len(), 256);
                    let mut processed_count = 0;
                    
                    for chunk in events.chunks(chunk_size) {
                        let _processed: Vec<_> = chunk.iter()
                            .map(|e| (e.x, e.y, (e.t * 1_000_000.0) as i64, if e.polarity { 1i8 } else { 0i8 }))
                            .collect();
                        processed_count += chunk.len();
                    }
                    
                    black_box(processed_count)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_memory_efficiency,
    benchmark_streaming_performance,
    benchmark_format_comparison,
    benchmark_memory_usage_estimation,
    benchmark_streaming_decision,
    benchmark_data_type_efficiency,
    benchmark_adaptive_chunk_sizing,
    benchmark_crossover_point
);
criterion_main!(benches);