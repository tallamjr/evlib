//! Regression test for HDF5 ECF polarity conversion bug.
//!
//! Verifies that ECF-compressed HDF5 files preserve both ON and OFF events.
//! This test uses the ECF encoder to create synthetic encoded data with both
//! polarities, then reads it back using the native decoder to verify both
//! polarity values are preserved.
//!
//! Only runs when hdf5 feature is enabled.

#![cfg(all(test, feature = "hdf5"))]

use evlib::ev_formats::prophesee_ecf_codec::{
    PropheseeECFDecoder, PropheseeECFEncoder, PropheseeEvent,
};

#[test]
fn test_hdf5_ecf_encoder_decoder_preserves_both_polarities() {
    // Create synthetic events with both ON (positive) and OFF (negative) polarities
    let mut events = Vec::new();

    // Add 200 events with alternating polarities to ensure good mixing
    for i in 0..200 {
        let polarity = if i % 2 == 0 { 1i16 } else { -1i16 };
        events.push(PropheseeEvent {
            x: (i as u16 % 1280),
            y: (i as u16 % 720),
            p: polarity,
            t: (i as i64) * 1000, // microseconds
        });
    }

    // Verify we have both polarities in the input
    assert!(
        events.iter().any(|e| e.p > 0),
        "test setup: missing positive polarity"
    );
    assert!(
        events.iter().any(|e| e.p < 0),
        "test setup: missing negative polarity"
    );

    // Encode using PropheseeECFEncoder
    let encoder = PropheseeECFEncoder::new();
    let encoded_data = encoder
        .encode(&events)
        .expect("ECF encoding should succeed");

    assert!(!encoded_data.is_empty(), "encoded data should not be empty");

    // Decode using PropheseeECFDecoder
    let decoder = PropheseeECFDecoder::new();
    let decoded_events = decoder
        .decode(&encoded_data)
        .expect("ECF decoding should succeed");

    // Verify event count
    assert_eq!(
        decoded_events.len(),
        events.len(),
        "decoded event count should match input"
    );

    // Key assertion: collect unique polarities from decoded events
    // After the ECF encoder/decoder round-trip, polarities should be preserved
    let mut polarities: Vec<i16> = decoded_events
        .iter()
        .map(|e| e.p)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    polarities.sort();

    // The codec should preserve both polarity values
    assert_eq!(
        polarities,
        vec![-1, 1],
        "decoded events must contain both polarity values (-1 for OFF, 1 for ON)"
    );

    // Verify that negative polarities (OFF events) are actually present,
    // not all converted to positive (the original bug would cause all to be +1)
    let negative_count = decoded_events.iter().filter(|e| e.p < 0).count();
    let positive_count = decoded_events.iter().filter(|e| e.p > 0).count();

    assert!(
        negative_count > 0,
        "there should be negative polarity events (OFF events)"
    );
    assert!(
        positive_count > 0,
        "there should be positive polarity events (ON events)"
    );
    let diff = if negative_count > positive_count {
        negative_count - positive_count
    } else {
        positive_count - negative_count
    };
    assert!(
        diff <= 1,
        "negative and positive counts should be approximately equal (diff: {})",
        diff
    );
}
