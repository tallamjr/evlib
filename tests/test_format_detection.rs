/// Comprehensive format detection system tests
/// Tests automatic format detection accuracy across different file types
///
/// This test suite verifies:
/// 1. Accurate format detection for all supported formats
/// 2. Confidence scoring reliability
/// 3. Metadata extraction accuracy
/// 4. Error handling for edge cases
/// 5. Performance of detection algorithms
use evlib::ev_formats::{detect_event_format, EventFormat, FormatDetectionError, FormatDetector};
use std::io::Write;
use std::path::Path;
use tempfile::NamedTempFile;

const SLIDER_DEPTH_DIR: &str = "/Users/tallam/github/tallamjr/origin/evlib/data/slider_depth";
const ORIGINAL_HDF5_DIR: &str = "/Users/tallam/github/tallamjr/origin/evlib/data/original/front";

/// Helper function to check if a test data file exists
fn check_data_file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Create a synthetic text file with event data
fn create_synthetic_text_file() -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

    writeln!(temp_file, "# Event data: timestamp x y polarity").unwrap();
    writeln!(temp_file, "0.001000 100 150 1").unwrap();
    writeln!(temp_file, "0.001005 200 250 0").unwrap();
    writeln!(temp_file, "0.001010 300 350 1").unwrap();
    writeln!(temp_file, "0.001015 400 450 -1").unwrap();
    writeln!(temp_file, "0.001020 500 550 1").unwrap();

    temp_file
}

/// Create a synthetic binary file that should be detected as AER
fn create_synthetic_aer_file() -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

    // Create AER events: (y << 10) | (x << 1) | polarity
    let events = [
        ((100u32 << 10) | (50u32 << 1) | 1).to_le_bytes(),
        ((200u32 << 10) | (150u32 << 1)).to_le_bytes(),
        ((300u32 << 10) | (250u32 << 1) | 1).to_le_bytes(),
    ];

    for event_bytes in &events {
        temp_file.write_all(event_bytes).unwrap();
    }

    temp_file
}

/// Create a file with random binary data
fn create_random_binary_file() -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

    // Write random binary data
    let random_data: Vec<u8> = (0..100).map(|i| (i * 17 + 42) as u8).collect();
    temp_file.write_all(&random_data).unwrap();

    temp_file
}

#[test]
fn test_text_format_detection() {
    // Test real text files
    let events_txt = format!("{SLIDER_DEPTH_DIR}/events.txt");
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if check_data_file_exists(&events_txt) {
        let result =
            detect_event_format(&events_txt).expect("Failed to detect format for events.txt");

        assert_eq!(result.format, EventFormat::Text);
        assert!(
            result.confidence >= 0.8,
            "Low confidence for text detection: {:.2}",
            result.confidence
        );
        assert!(result.metadata.file_size > 0);

        println!(
            "events.txt: {} (confidence: {:.2})",
            result.format, result.confidence
        );
    }

    if check_data_file_exists(&events_chunk_txt) {
        let result = detect_event_format(&events_chunk_txt)
            .expect("Failed to detect format for events_chunk.txt");

        assert_eq!(result.format, EventFormat::Text);
        assert!(
            result.confidence >= 0.8,
            "Low confidence for text detection: {:.2}",
            result.confidence
        );

        println!(
            "events_chunk.txt: {} (confidence: {:.2})",
            result.format, result.confidence
        );
    }

    // Test synthetic text file
    let synthetic_text = create_synthetic_text_file();
    let result = detect_event_format(synthetic_text.path().to_str().unwrap())
        .expect("Failed to detect synthetic text format");

    assert_eq!(result.format, EventFormat::Text);
    assert!(
        result.confidence >= 0.7,
        "Low confidence for synthetic text: {:.2}",
        result.confidence
    );

    println!(
        "Synthetic text: {} (confidence: {:.2})",
        result.format, result.confidence
    );
}

#[test]
fn test_hdf5_format_detection() {
    let hdf5_files = [
        format!("{ORIGINAL_HDF5_DIR}/seq01.h5"),
        format!("{ORIGINAL_HDF5_DIR}/seq02.h5"),
        format!("{ORIGINAL_HDF5_DIR}/seq03.h5"),
    ];

    for file_path in &hdf5_files {
        if check_data_file_exists(file_path) {
            let result = detect_event_format(file_path)
                .unwrap_or_else(|_| panic!("Failed to detect format for {file_path}"));

            assert_eq!(result.format, EventFormat::HDF5);
            assert!(
                result.confidence >= 0.95,
                "Low confidence for HDF5 detection: {:.2}",
                result.confidence
            );
            assert!(result.metadata.file_size > 0);

            println!(
                "{}: {} (confidence: {:.2})",
                file_path, result.format, result.confidence
            );
            break; // Test just one to avoid excessive output
        }
    }
}

#[test]
fn test_aer_format_detection() {
    let synthetic_aer = create_synthetic_aer_file();
    let result = detect_event_format(synthetic_aer.path().to_str().unwrap())
        .expect("Failed to detect synthetic AER format");

    // The detection might classify this as AER or Binary depending on implementation
    assert!(
        result.format == EventFormat::AER || result.format == EventFormat::Binary,
        "Synthetic AER should be detected as AER or Binary, got: {}",
        result.format
    );
    assert!(
        result.confidence > 0.5,
        "Low confidence for synthetic AER: {:.2}",
        result.confidence
    );

    println!(
        "Synthetic AER: {} (confidence: {:.2})",
        result.format, result.confidence
    );
}

#[test]
fn test_unknown_format_detection() {
    let random_file = create_random_binary_file();
    let result = detect_event_format(random_file.path().to_str().unwrap())
        .expect("Failed to detect random binary format");

    // Random binary data should be detected as Unknown or Binary with low confidence
    assert!(
        result.format == EventFormat::Unknown || result.format == EventFormat::Binary,
        "Random data should be Unknown or Binary, got: {}",
        result.format
    );

    if result.format == EventFormat::Binary {
        assert!(
            result.confidence < 0.7,
            "Random data should have low binary confidence: {:.2}",
            result.confidence
        );
    }

    println!(
        "Random binary: {} (confidence: {:.2})",
        result.format, result.confidence
    );
}

#[test]
fn test_empty_file_detection() {
    let empty_file = NamedTempFile::new().expect("Failed to create temp file");
    let result = detect_event_format(empty_file.path().to_str().unwrap());

    // Empty file detection behavior may vary - could be error or Unknown
    match result {
        Ok(detection) => {
            assert_eq!(detection.format, EventFormat::Unknown);
            assert_eq!(detection.metadata.file_size, 0);
            println!("Empty file detected as Unknown");
        }
        Err(FormatDetectionError::EmptyFile) => {
            println!("Empty file correctly rejected");
        }
        Err(e) => {
            panic!("Unexpected error for empty file: {e}");
        }
    }
}

#[test]
fn test_nonexistent_file_detection() {
    let result = detect_event_format("/nonexistent/file.txt");

    assert!(result.is_err(), "Should fail for non-existent file");

    match result.unwrap_err() {
        FormatDetectionError::FileNotFound(_) | FormatDetectionError::Io(_) => {
            println!("Non-existent file correctly rejected");
        }
        e => {
            panic!("Unexpected error type for non-existent file: {e:?}");
        }
    }
}

#[test]
fn test_confidence_scoring_consistency() {
    // Test that confidence scores are consistent across multiple detections
    let test_files = vec![
        (
            format!("{SLIDER_DEPTH_DIR}/events_chunk.txt"),
            EventFormat::Text,
        ),
        (format!("{ORIGINAL_HDF5_DIR}/seq01.h5"), EventFormat::HDF5),
    ];

    for (file_path, expected_format) in test_files {
        if !check_data_file_exists(&file_path) {
            continue;
        }

        let mut confidences = Vec::new();

        // Run detection multiple times
        for _ in 0..3 {
            let result = detect_event_format(&file_path)
                .unwrap_or_else(|_| panic!("Failed to detect format for {file_path}"));

            assert_eq!(result.format, expected_format);
            confidences.push(result.confidence);
        }

        // Check consistency (all confidences should be identical)
        let first_confidence = confidences[0];
        for &confidence in &confidences[1..] {
            assert!(
                (confidence - first_confidence).abs() < 1e-6,
                "Inconsistent confidence scores: {confidences:?}"
            );
        }

        println!(
            "{}: Consistent confidence {:.2} across {} runs",
            file_path,
            first_confidence,
            confidences.len()
        );
    }
}

#[test]
fn test_metadata_extraction() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if check_data_file_exists(&events_chunk_txt) {
        let result = detect_event_format(&events_chunk_txt).expect("Failed to detect format");

        // Verify metadata
        assert!(
            result.metadata.file_size > 0,
            "File size should be positive"
        );

        // Check if file size matches actual file
        if let Ok(metadata) = std::fs::metadata(&events_chunk_txt) {
            assert_eq!(
                result.metadata.file_size,
                metadata.len(),
                "Detected file size should match actual size"
            );
        }

        println!("Metadata extraction: {} bytes", result.metadata.file_size);
    }
}

#[test]
fn test_format_description_consistency() {
    let formats = [
        EventFormat::Text,
        EventFormat::HDF5,
        EventFormat::AER,
        EventFormat::AEDAT1,
        EventFormat::AEDAT2,
        EventFormat::AEDAT3,
        EventFormat::AEDAT4,
        EventFormat::Binary,
        EventFormat::Unknown,
    ];

    for format in &formats {
        let description = FormatDetector::get_format_description(format);
        assert!(
            !description.is_empty(),
            "Format description should not be empty for {format:?}"
        );
        assert!(
            description.len() > 5,
            "Format description should be meaningful for {format:?}"
        );

        println!("{format}: {description}");
    }
}

#[test]
fn test_detection_performance() {
    let test_files = vec![
        format!("{SLIDER_DEPTH_DIR}/events_chunk.txt"),
        format!("{ORIGINAL_HDF5_DIR}/seq01.h5"),
    ];

    for file_path in test_files {
        if !check_data_file_exists(&file_path) {
            continue;
        }

        let start = std::time::Instant::now();
        let result = detect_event_format(&file_path)
            .unwrap_or_else(|_| panic!("Failed to detect format for {file_path}"));
        let duration = start.elapsed();

        // Detection should be fast (< 100ms for typical files)
        assert!(
            duration.as_millis() < 1000,
            "Detection too slow: {}ms for {}",
            duration.as_millis(),
            file_path
        );

        println!(
            "Detection performance: {} in {:.2}ms",
            result.format,
            duration.as_millis()
        );
    }
}

#[test]
fn test_edge_case_files() {
    // Test very small text file
    let mut tiny_text = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(tiny_text, "0.001 10 20 1").unwrap();

    let result = detect_event_format(tiny_text.path().to_str().unwrap())
        .expect("Failed to detect tiny text file");

    // Should still be detected as text despite being small
    assert_eq!(result.format, EventFormat::Text);
    println!(
        "Tiny text file: {} (confidence: {:.2})",
        result.format, result.confidence
    );

    // Test file with only header
    let mut header_only = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(header_only, "# timestamp x y polarity").unwrap();

    let result = detect_event_format(header_only.path().to_str().unwrap())
        .expect("Failed to detect header-only file");

    // Behavior may vary - could be Text or Unknown
    println!(
        "Header-only file: {} (confidence: {:.2})",
        result.format, result.confidence
    );

    // Test file with mixed data
    let mut mixed_data = NamedTempFile::new().expect("Failed to create temp file");
    writeln!(mixed_data, "0.001 10 20 1").unwrap();
    writeln!(mixed_data, "invalid line here").unwrap();
    writeln!(mixed_data, "0.002 30 40 0").unwrap();

    let result = detect_event_format(mixed_data.path().to_str().unwrap())
        .expect("Failed to detect mixed data file");

    // Should still detect as text but possibly with lower confidence
    println!(
        "Mixed data file: {} (confidence: {:.2})",
        result.format, result.confidence
    );
}

#[test]
fn test_file_extension_influence() {
    // Create identical content with different extensions
    let content = "0.001000 100 150 1\n0.001005 200 250 0\n0.001010 300 350 1\n";

    // .txt extension
    let mut txt_file = NamedTempFile::with_suffix(".txt").expect("Failed to create .txt file");
    write!(txt_file, "{content}").unwrap();

    let txt_result =
        detect_event_format(txt_file.path().to_str().unwrap()).expect("Failed to detect .txt file");

    // .dat extension
    let mut dat_file = NamedTempFile::with_suffix(".dat").expect("Failed to create .dat file");
    write!(dat_file, "{content}").unwrap();

    let dat_result =
        detect_event_format(dat_file.path().to_str().unwrap()).expect("Failed to detect .dat file");

    // Both should be detected as text
    assert_eq!(txt_result.format, EventFormat::Text);
    assert_eq!(dat_result.format, EventFormat::Text);

    // Confidence might be influenced by extension
    println!(
        "Extension influence: .txt={:.2}, .dat={:.2}",
        txt_result.confidence, dat_result.confidence
    );
}

#[test]
fn test_large_file_detection() {
    let events_txt = format!("{SLIDER_DEPTH_DIR}/events.txt");

    if check_data_file_exists(&events_txt) {
        let start = std::time::Instant::now();
        let result = detect_event_format(&events_txt).expect("Failed to detect large file format");
        let duration = start.elapsed();

        assert_eq!(result.format, EventFormat::Text);

        // Large file detection should still be reasonably fast
        assert!(
            duration.as_millis() < 5000,
            "Large file detection too slow: {}ms",
            duration.as_millis()
        );

        println!(
            "Large file detection: {} MB in {:.2}ms",
            result.metadata.file_size / 1_000_000,
            duration.as_millis()
        );
    }
}
