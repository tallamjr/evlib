use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use evlib::ev_formats::Event;
use evlib::ev_formats::{load_events_with_config, LoadConfig};
use std::hint::black_box as hint_black_box;
use std::io::Write;
use tempfile::NamedTempFile;

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
        let polarity = if rng.is_multiple_of(2) { 1i8 } else { -1i8 };

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
fn write_events_to_temp_file(
    events: &[Event],
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let mut temp_file = NamedTempFile::new()?;

    for event in events {
        writeln!(
            temp_file,
            "{:.6} {} {} {}",
            event.t,
            event.x,
            event.y,
            if event.polarity > 0 { 1 } else { 0 }
        )?;
    }

    temp_file.flush()?;
    Ok(temp_file)
}

/// Benchmark direct loading for different event counts
fn benchmark_direct_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency_direct_loading");

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
                    hint_black_box(loaded_events.height());
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
                    let mut x_builder =
                        PrimitiveChunkedBuilder::<Int16Type>::new("x".into(), events.len());
                    let mut y_builder =
                        PrimitiveChunkedBuilder::<Int16Type>::new("y".into(), events.len());
                    let mut timestamp_builder =
                        PrimitiveChunkedBuilder::<Int64Type>::new("timestamp".into(), events.len());
                    let mut polarity_builder =
                        PrimitiveChunkedBuilder::<Int8Type>::new("polarity".into(), events.len());

                    for event in events {
                        x_builder.append_value(event.x as i16);
                        y_builder.append_value(event.y as i16);
                        timestamp_builder.append_value((event.t * 1_000_000.0) as i64);
                        polarity_builder.append_value(if event.polarity > 0 { 1i8 } else { 0i8 });
                    }

                    let x_series = x_builder.finish().into_series();
                    let y_series = y_builder.finish().into_series();
                    let timestamp_series = timestamp_builder.finish().into_series();
                    let polarity_series = polarity_builder.finish().into_series();

                    let df = DataFrame::new(vec![
                        x_series.into(),
                        y_series.into(),
                        timestamp_series.into(),
                        polarity_series.into(),
                    ])
                    .unwrap();
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
                    let mut x_builder =
                        PrimitiveChunkedBuilder::<Int32Type>::new("x".into(), events.len());
                    let mut y_builder =
                        PrimitiveChunkedBuilder::<Int32Type>::new("y".into(), events.len());
                    let mut timestamp_builder =
                        PrimitiveChunkedBuilder::<Int64Type>::new("timestamp".into(), events.len());
                    let mut polarity_builder =
                        PrimitiveChunkedBuilder::<Int32Type>::new("polarity".into(), events.len());

                    for event in events {
                        x_builder.append_value(event.x as i32);
                        y_builder.append_value(event.y as i32);
                        timestamp_builder.append_value((event.t * 1_000_000.0) as i64);
                        polarity_builder.append_value(if event.polarity > 0 { 1i32 } else { 0i32 });
                    }

                    let x_series = x_builder.finish().into_series();
                    let y_series = y_builder.finish().into_series();
                    let timestamp_series = timestamp_builder.finish().into_series();
                    let polarity_series = polarity_builder.finish().into_series();

                    let df = DataFrame::new(vec![
                        x_series.into(),
                        y_series.into(),
                        timestamp_series.into(),
                        polarity_series.into(),
                    ])
                    .unwrap();
                    hint_black_box(df.height());
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_direct_loading,
    benchmark_data_type_efficiency
);
criterion_main!(benches);
