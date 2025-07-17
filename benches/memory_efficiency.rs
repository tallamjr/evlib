use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use evlib::ev_core::Event;
use evlib::ev_formats::{load_events_with_config, LoadConfig};
use evlib::ev_formats::streaming::{PolarsEventStreamer, should_use_streaming, estimate_memory_usage};
use evlib::ev_formats::EventFormat;
use std::hint::black_box as hint_black_box;
use tempfile::NamedTempFile;
use std::io::Write;

/// Generate synthetic events for consistent benchmarking
fn generate_synthetic_events(count: usize, width: u16, height: u16) -> Vec<Event> {
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

/// Write events to temporary text file for benchmarking
fn write_events_to_temp_file(events: &[Event]) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let mut temp_file = NamedTempFile::new()?;
    
    for event in events {
        writeln!(temp_file, "{:.6} {} {} {}", 
                event.t, event.x, event.y, 
                if event.polarity { 1 } else { 0 })?;
    }
    
    temp_file.flush()?;
    Ok(temp_file)
}

/// Benchmark direct loading vs streaming for different event counts
fn benchmark_direct_vs_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency_direct_vs_streaming");
    
    // Test different event counts
    let event_counts = vec![1_000_000, 2_500_000, 5_000_000, 10_000_000];
    
    for count in event_counts {
        group.throughput(Throughput::Elements(count as u64));
        
        // Generate synthetic events
        let events = generate_synthetic_events(count, 640, 480);
        
        // Create temporary file for loading benchmarks
        let temp_file = write_events_to_temp_file(&events).unwrap();
        let file_path = temp_file.path().to_str().unwrap();
        
        // Benchmark direct loading
        group.bench_with_input(
            BenchmarkId::new("direct_loading", count),
            &file_path,
            |b, path| {
                b.iter(|| {
                    let config = LoadConfig::new();
                    let loaded_events = load_events_with_config(path, &config).unwrap();
                    hint_black_box(loaded_events.len());
                })
            },
        );
        
        // Benchmark streaming loading
        group.bench_with_input(
            BenchmarkId::new("streaming_loading", count),
            &events,
            |b, events| {
                b.iter(|| {
                    let chunk_size = PolarsEventStreamer::calculate_optimal_chunk_size(events.len(), 256);
                    let streamer = PolarsEventStreamer::new(chunk_size, EventFormat::Text);
                    
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

/// Benchmark memory usage patterns for different formats
fn benchmark_memory_usage_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_patterns");
    
    let event_count = 1_000_000;
    let events = generate_synthetic_events(event_count, 640, 480);
    
    // Test different formats
    let formats = vec![
        ("HDF5", EventFormat::HDF5),
        ("EVT2", EventFormat::EVT2),
        ("Text", EventFormat::Text),
    ];
    
    for (format_name, format) in formats {
        group.throughput(Throughput::Elements(event_count as u64));
        
        // Benchmark memory estimation
        group.bench_with_input(
            BenchmarkId::new("memory_estimation", format_name),
            &event_count,
            |b, count| {
                b.iter(|| {
                    let usage = estimate_memory_usage(*count);
                    hint_black_box(usage);
                })
            },
        );
        
        // Benchmark polarity conversion overhead
        group.bench_with_input(
            BenchmarkId::new("polarity_conversion", format_name),
            &(&events, format),
            |b, (events, format)| {
                b.iter(|| {
                    let streamer = PolarsEventStreamer::new(100_000, *format);
                    let mut converted_count = 0;
                    for event in events.iter() {
                        let _converted = streamer.convert_polarity(event.polarity);
                        converted_count += 1;
                    }
                    hint_black_box(converted_count);
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark different chunk sizes for streaming
fn benchmark_chunk_size_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_size_optimization");
    
    let event_count = 5_000_000;
    let events = generate_synthetic_events(event_count, 640, 480);
    
    // Test different chunk sizes
    let chunk_sizes = vec![100_000, 500_000, 1_000_000, 2_000_000];
    
    for chunk_size in chunk_sizes {
        group.throughput(Throughput::Elements(event_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("streaming_chunk_size", chunk_size),
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

/// Benchmark streaming decision threshold
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
                    let decision = should_use_streaming(*count, None);
                    hint_black_box(decision);
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency with different data types
fn benchmark_data_type_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_type_efficiency");
    
    let event_count = 2_000_000;
    let events = generate_synthetic_events(event_count, 1024, 768);
    
    #[cfg(feature = "polars")]
    {
        use polars::prelude::*;
        
        // Benchmark different data type strategies
        group.bench_with_input(
            BenchmarkId::new("optimal_types", "i16_i8_i64"),
            &events,
            |b, events| {
                b.iter(|| {
                    let mut x_builder = PrimitiveChunkedBuilder::<Int16Type>::new("x", events.len());
                    let mut y_builder = PrimitiveChunkedBuilder::<Int16Type>::new("y", events.len());
                    let mut timestamp_builder = PrimitiveChunkedBuilder::<Int64Type>::new("timestamp", events.len());
                    let mut polarity_builder = PrimitiveChunkedBuilder::<Int8Type>::new("polarity", events.len());
                    
                    for event in events {
                        x_builder.append_value(event.x as i16);
                        y_builder.append_value(event.y as i16);
                        timestamp_builder.append_value((event.t * 1_000_000.0) as i64);
                        polarity_builder.append_value(if event.polarity { 1i8 } else { 0i8 });
                    }
                    
                    let x_series = x_builder.finish().into_series();
                    let y_series = y_builder.finish().into_series();
                    let timestamp_series = timestamp_builder.finish().into_series();
                    let polarity_series = polarity_builder.finish().into_series();
                    
                    let df = DataFrame::new(vec![x_series, y_series, timestamp_series, polarity_series]).unwrap();
                    hint_black_box(df.height());
                })
            },
        );
        
        // Benchmark less optimal types for comparison
        group.bench_with_input(
            BenchmarkId::new("suboptimal_types", "i32_i32_i64"),
            &events,
            |b, events| {
                b.iter(|| {
                    let mut x_builder = PrimitiveChunkedBuilder::<Int32Type>::new("x", events.len());
                    let mut y_builder = PrimitiveChunkedBuilder::<Int32Type>::new("y", events.len());
                    let mut timestamp_builder = PrimitiveChunkedBuilder::<Int64Type>::new("timestamp", events.len());
                    let mut polarity_builder = PrimitiveChunkedBuilder::<Int32Type>::new("polarity", events.len());
                    
                    for event in events {
                        x_builder.append_value(event.x as i32);
                        y_builder.append_value(event.y as i32);
                        timestamp_builder.append_value((event.t * 1_000_000.0) as i64);
                        polarity_builder.append_value(if event.polarity { 1i32 } else { 0i32 });
                    }
                    
                    let x_series = x_builder.finish().into_series();
                    let y_series = y_builder.finish().into_series();
                    let timestamp_series = timestamp_builder.finish().into_series();
                    let polarity_series = polarity_builder.finish().into_series();
                    
                    let df = DataFrame::new(vec![x_series, y_series, timestamp_series, polarity_series]).unwrap();
                    hint_black_box(df.height());
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_direct_vs_streaming,
    benchmark_memory_usage_patterns,
    benchmark_chunk_size_optimization,
    benchmark_streaming_decision,
    benchmark_data_type_efficiency
);
criterion_main!(benches);