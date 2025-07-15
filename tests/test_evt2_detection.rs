use crate::ev_formats::format_detector::{detect_event_format, EventFormat};

#[test]
fn test_evt2_detection() {
    let evt2_file =
        "/Users/tallam/github/tallamjr/origin/evlib/data/eTram/raw/val_2/val_night_007.raw";

    // Test format detection
    match detect_event_format(evt2_file) {
        Ok(result) => {
            println!("Detected format: {:?}", result.format);
            println!("Confidence: {:.2}", result.confidence);
            println!("Metadata: {:?}", result.metadata);

            // Check if it's detected as EVT2
            assert_eq!(result.format, EventFormat::EVT2);
            assert!(result.confidence > 0.8);

            // Check if resolution is extracted
            if let Some(resolution) = result.metadata.sensor_resolution {
                assert_eq!(resolution, (1280, 720));
            }
        }
        Err(e) => {
            panic!("Format detection failed: {}", e);
        }
    }
}
