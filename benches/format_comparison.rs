use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use evlib::ev_core::Event;
use evlib::ev_formats::{format_detector, load_events_with_config, LoadConfig};
use evlib::ev_formats::streaming::PolarsEventStreamer;
use evlib::ev_formats::{EventFormat, FormatDetector};
use std::hint::black_box as hint_black_box;
use tempfile::NamedTempFile;
use std::io::Write;

/// Generate synthetic events for format comparison
fn generate_test_events(count: usize) -> Vec<Event> {
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

/// Write events to text file with specified format
fn write_text_file(events: &[Event], polarity_encoding: &str) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let mut temp_file = NamedTempFile::new()?;
    
    for event in events {
        let polarity_value = match polarity_encoding {
            "0_1" => if event.polarity { 1 } else { 0 },
            "neg1_1" => if event.polarity { 1 } else { -1 },
            _ => if event.polarity { 1 } else { 0 },
        };
        
        writeln!(temp_file, "{:.6} {} {} {}", 
                event.t, event.x, event.y, polarity_value)?;
    }
    
    temp_file.flush()?;
    Ok(temp_file)
}

/// Write events to HDF5 file (mock implementation using text for benchmarking)
fn write_hdf5_file(events: &[Event]) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    // For benchmarking purposes, we'll use a text file with HDF5 extension
    // The actual format detection is based on content, not extension
    let mut temp_file = NamedTempFile::with_suffix(".h5")?;
    
    for event in events {
        writeln!(temp_file, "{:.6} {} {} {}", 
                event.t, event.x, event.y, 
                if event.polarity { 1 } else { 0 })?;
    }
    
    temp_file.flush()?;
    Ok(temp_file)
}

/// Benchmark format detection speed
fn benchmark_format_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_detection");
    
    let event_count = 100_000;
    let events = generate_test_events(event_count);
    
    // Create test files for different formats
    let text_file = write_text_file(&events, "0_1").unwrap();
    let hdf5_file = write_hdf5_file(&events).unwrap();
    
    let test_files = vec![
        ("text", text_file.path().to_str().unwrap()),
        ("hdf5", hdf5_file.path().to_str().unwrap()),
    ];
    
    for (format_name, file_path) in test_files {
        group.bench_with_input(
            BenchmarkId::new("detect_format", format_name),
            &file_path,
            |b, path| {
                b.iter(|| {
                    let result = format_detector::detect_event_format(path);
                    match result {
                        Ok(detection) => hint_black_box(detection.confidence),
                        Err(_) => hint_black_box(0.0),
                    }
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark loading performance across different formats
fn benchmark_format_loading_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_loading_performance");
    
    let event_counts = vec![100_000, 500_000, 1_000_000];
    
    for count in event_counts {
        group.throughput(Throughput::Elements(count as u64));
        
        let events = generate_test_events(count);
        
        // Create test files
        let text_file = write_text_file(&events, "0_1").unwrap();
        let hdf5_file = write_hdf5_file(&events).unwrap();
        
        let test_files = vec![
            ("text", text_file.path().to_str().unwrap()),
            ("hdf5", hdf5_file.path().to_str().unwrap()),
        ];
        
        for (format_name, file_path) in test_files {
            group.bench_with_input(
                BenchmarkId::new("load_format", format!("{}_{}", format_name, count)),
                &file_path,
                |b, path| {
                    b.iter(|| {
                        let config = LoadConfig::new();
                        let result = load_events_with_config(path, &config);
                        match result {
                            Ok(events) => hint_black_box(events.len()),
                            Err(_) => hint_black_box(0),
                        }
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark polarity encoding performance
fn benchmark_polarity_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("polarity_encoding");
    
    let event_count = 1_000_000;
    
    // Test different polarity encodings
    let encodings = vec![
        ("0_1", "0_1"),
        ("neg1_1", "neg1_1"),
    ];
    
    for (encoding_name, encoding) in encodings {
        group.throughput(Throughput::Elements(event_count as u64));
        
        let events = generate_test_events(event_count);
        let text_file = write_text_file(&events, encoding).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("polarity_encoding", encoding_name),
            &text_file.path().to_str().unwrap(),
            |b, path| {
                b.iter(|| {
                    let config = LoadConfig::new();
                    let result = load_events_with_config(path, &config);
                    match result {
                        Ok(events) => {
                            let polarity_sum: i32 = events.iter()
                                .map(|e| if e.polarity { 1 } else { 0 })
                                .sum();
                            hint_black_box(polarity_sum)
                        },
                        Err(_) => hint_black_box(0),
                    }
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark format-specific streaming performance
fn benchmark_format_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_streaming");
    
    let event_count = 2_000_000;
    let events = generate_test_events(event_count);
    
    // Test different formats
    let formats = vec![
        ("HDF5", EventFormat::HDF5),
        ("EVT2", EventFormat::EVT2),
        ("EVT21", EventFormat::EVT21),
        ("Text", EventFormat::Text),
    ];
    
    for (format_name, format) in formats {
        group.throughput(Throughput::Elements(event_count as u64));
        
        group.bench_with_input(
            BenchmarkId::new("format_streaming", format_name),
            &(&events, format),
            |b, (events, format)| {
                b.iter(|| {
                    let chunk_size = PolarsEventStreamer::calculate_optimal_chunk_size(events.len(), 256);
                    let streamer = PolarsEventStreamer::new(chunk_size, *format);
                    
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

/// Benchmark format-specific polarity conversion
fn benchmark_format_polarity_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_polarity_conversion");
    
    let event_count = 1_000_000;
    let events = generate_test_events(event_count);
    
    // Test different formats
    let formats = vec![
        ("HDF5", EventFormat::HDF5),
        ("EVT2", EventFormat::EVT2),
        ("EVT21", EventFormat::EVT21),
        ("Text", EventFormat::Text),
    ];
    
    for (format_name, format) in formats {
        group.throughput(Throughput::Elements(event_count as u64));
        
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

/// Benchmark format description retrieval
fn benchmark_format_description(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_description");
    
    let formats = vec![
        EventFormat::HDF5,
        EventFormat::EVT2,
        EventFormat::EVT21,
        EventFormat::Text,
        EventFormat::AEDAT1,
        EventFormat::AEDAT2,
        EventFormat::AEDAT3,
        EventFormat::AEDAT4,
        EventFormat::AER,
        EventFormat::EVT3,
        EventFormat::Binary,
        EventFormat::Unknown,
    ];
    
    for format in formats {
        group.bench_with_input(
            BenchmarkId::new("get_description", format.to_string()),
            &format,
            |b, format| {
                b.iter(|| {
                    let description = FormatDetector::get_format_description(format);
                    hint_black_box(description.len());
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark format-specific metadata extraction
fn benchmark_format_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_metadata");
    
    let event_count = 100_000;
    let events = generate_test_events(event_count);
    
    // Create test files
    let text_file = write_text_file(&events, "0_1").unwrap();
    let hdf5_file = write_hdf5_file(&events).unwrap();
    
    let test_files = vec![
        ("text", text_file.path().to_str().unwrap()),
        ("hdf5", hdf5_file.path().to_str().unwrap()),
    ];
    
    for (format_name, file_path) in test_files {
        group.bench_with_input(
            BenchmarkId::new("extract_metadata", format_name),
            &file_path,
            |b, path| {
                b.iter(|| {
                    let result = format_detector::detect_event_format(path);
                    match result {
                        Ok(detection) => hint_black_box(detection.metadata.properties.len()),
                        Err(_) => hint_black_box(0),
                    }
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_format_detection,
    benchmark_format_loading_performance,
    benchmark_polarity_encoding,
    benchmark_format_streaming,
    benchmark_format_polarity_conversion,
    benchmark_format_description,
    benchmark_format_metadata
);
criterion_main!(benches);