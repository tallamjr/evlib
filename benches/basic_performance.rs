use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::hint::black_box;

/// Simple Event structure for benchmarking
#[derive(Clone, Debug)]
struct Event {
    t: f64,
    x: u16,
    y: u16,
    polarity: bool,
}

/// Generate synthetic events for benchmarking
fn generate_events(count: usize) -> Vec<Event> {
    let mut events = Vec::with_capacity(count);
    let mut rng = 42u64; // Simple LCG for reproducibility
    
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
    
    events
}

/// Benchmark event generation
fn benchmark_event_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_generation");
    
    let counts = vec![100_000, 500_000, 1_000_000, 5_000_000];
    
    for count in counts {
        group.throughput(Throughput::Elements(count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("generate_events", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let events = generate_events(count);
                    black_box(events.len())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark event filtering
fn benchmark_event_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("event_filtering");
    
    let event_count = 1_000_000;
    let events = generate_events(event_count);
    
    group.throughput(Throughput::Elements(event_count as u64));
    
    // Time filter
    group.bench_with_input(
        BenchmarkId::new("time_filter", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let filtered: Vec<_> = events.iter()
                    .filter(|e| e.t >= 0.1 && e.t <= 0.5)
                    .collect();
                black_box(filtered.len())
            })
        },
    );
    
    // Spatial filter
    group.bench_with_input(
        BenchmarkId::new("spatial_filter", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let filtered: Vec<_> = events.iter()
                    .filter(|e| e.x >= 100 && e.x <= 500 && e.y >= 100 && e.y <= 400)
                    .collect();
                black_box(filtered.len())
            })
        },
    );
    
    // Polarity filter
    group.bench_with_input(
        BenchmarkId::new("polarity_filter", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let filtered: Vec<_> = events.iter()
                    .filter(|e| e.polarity)
                    .collect();
                black_box(filtered.len())
            })
        },
    );
    
    // Combined filter
    group.bench_with_input(
        BenchmarkId::new("combined_filter", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let filtered: Vec<_> = events.iter()
                    .filter(|e| e.t >= 0.1 && e.t <= 0.5 
                            && e.x >= 100 && e.x <= 500 
                            && e.y >= 100 && e.y <= 400 
                            && e.polarity)
                    .collect();
                black_box(filtered.len())
            })
        },
    );
    
    group.finish();
}

/// Benchmark memory usage patterns
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    let event_count = 1_000_000;
    let events = generate_events(event_count);
    
    group.throughput(Throughput::Elements(event_count as u64));
    
    // Vector allocation and copying
    group.bench_with_input(
        BenchmarkId::new("vector_clone", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let cloned = events.clone();
                black_box(cloned.len())
            })
        },
    );
    
    // Memory estimation
    group.bench_with_input(
        BenchmarkId::new("memory_estimation", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let estimated_bytes = events.len() * std::mem::size_of::<Event>();
                black_box(estimated_bytes)
            })
        },
    );
    
    group.finish();
}

/// Benchmark chunked processing
fn benchmark_chunked_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunked_processing");
    
    let event_count = 5_000_000;
    let events = generate_events(event_count);
    
    let chunk_sizes = vec![10_000, 50_000, 100_000, 500_000, 1_000_000];
    
    for chunk_size in chunk_sizes {
        group.throughput(Throughput::Elements(event_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("chunked_process", chunk_size),
            &(&events, chunk_size),
            |b, (events, chunk_size)| {
                b.iter(|| {
                    let mut processed_count = 0;
                    for chunk in events.chunks(*chunk_size) {
                        // Simulate processing
                        let chunk_sum: u64 = chunk.iter()
                            .map(|e| e.x as u64 + e.y as u64)
                            .sum();
                        processed_count += chunk.len();
                        black_box(chunk_sum);
                    }
                    black_box(processed_count)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark format conversion performance
fn benchmark_format_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_conversion");
    
    let event_count = 1_000_000;
    let events = generate_events(event_count);
    
    group.throughput(Throughput::Elements(event_count as u64));
    
    // Convert to different polarity encodings
    group.bench_with_input(
        BenchmarkId::new("polarity_0_1", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let converted: Vec<_> = events.iter()
                    .map(|e| if e.polarity { 1i8 } else { 0i8 })
                    .collect();
                black_box(converted.len())
            })
        },
    );
    
    group.bench_with_input(
        BenchmarkId::new("polarity_neg1_1", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let converted: Vec<_> = events.iter()
                    .map(|e| if e.polarity { 1i8 } else { -1i8 })
                    .collect();
                black_box(converted.len())
            })
        },
    );
    
    // Convert timestamp to microseconds
    group.bench_with_input(
        BenchmarkId::new("timestamp_to_microseconds", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let converted: Vec<_> = events.iter()
                    .map(|e| (e.t * 1_000_000.0) as i64)
                    .collect();
                black_box(converted.len())
            })
        },
    );
    
    group.finish();
}

/// Benchmark sorting performance
fn benchmark_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting");
    
    let event_count = 1_000_000;
    
    // Generate events with random timestamps
    let mut rng = 42u64;
    let mut events = Vec::with_capacity(event_count);
    
    for i in 0..event_count {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let random_time = (rng % 10000) as f64 / 10000.0;
        
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = (rng % 640) as u16;
        
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = (rng % 480) as u16;
        
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let polarity = (rng % 2) == 0;
        
        events.push(Event {
            t: random_time,
            x,
            y,
            polarity,
        });
    }
    
    group.throughput(Throughput::Elements(event_count as u64));
    
    group.bench_with_input(
        BenchmarkId::new("sort_by_timestamp", event_count),
        &events,
        |b, events| {
            b.iter(|| {
                let mut events_copy = events.clone();
                events_copy.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
                black_box(events_copy.len())
            })
        },
    );
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_event_generation,
    benchmark_event_filtering,
    benchmark_memory_usage,
    benchmark_chunked_processing,
    benchmark_format_conversion,
    benchmark_sorting
);
criterion_main!(benches);