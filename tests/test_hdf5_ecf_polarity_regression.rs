//! Regression tests for the HDF5 ECF polarity double-conversion bug (2026-08-08 R1 finding).
//!
//! The native ECF reader (`read_prophesee_hdf5_native` in `src/ev_formats/hdf5_reader.rs`)
//! used to store polarity as -1/1 in the returned `Event` structs. Every other reader stores
//! 0/1, and `build_polars_dataframe`'s HDF5 branch converts 0 to -1 and everything else to +1.
//! Feeding it -1/1 mapped -1 to +1, so every event from an ECF-compressed HDF5 file came out
//! positive. The fix (hdf5_reader.rs, in the `PropheseeEvent` to `Event` conversion loop) stores
//! 0/1 like every other reader.
//!
//! Only runs when the `hdf5` feature is enabled.

#![cfg(all(test, feature = "hdf5"))]

use evlib::ev_formats::hdf5_reader::read_prophesee_hdf5_native;
use evlib::ev_formats::prophesee_ecf_codec::{
    PropheseeECFDecoder, PropheseeECFEncoder, PropheseeEvent,
};
use hdf5_metno_sys::{h5d, h5f, h5g, h5p, h5s, h5t, h5z};
use std::ffi::CString;
use std::os::raw::{c_uint, c_void};

/// Sanity check on the codec itself: an ECF encode/decode round trip preserves both
/// polarities. This is NOT a regression test for the reader bug above (that bug lived
/// downstream of the decoder, in the `PropheseeEvent` -> `Event` conversion); it only
/// guards the codec's own correctness.
#[test]
fn test_ecf_codec_roundtrip_preserves_both_polarities() {
    let mut events = Vec::new();
    for i in 0..200 {
        let polarity = if i % 2 == 0 { 1i16 } else { -1i16 };
        events.push(PropheseeEvent {
            x: i as u16 % 1280,
            y: i as u16 % 720,
            p: polarity,
            t: (i as i64) * 1000,
        });
    }

    let encoder = PropheseeECFEncoder::new();
    let encoded_data = encoder
        .encode(&events)
        .expect("ECF encoding should succeed");
    assert!(!encoded_data.is_empty(), "encoded data should not be empty");

    let decoder = PropheseeECFDecoder::new();
    let decoded_events = decoder
        .decode(&encoded_data)
        .expect("ECF decoding should succeed");

    assert_eq!(decoded_events.len(), events.len());
    assert!(decoded_events.iter().any(|e| e.p > 0), "missing ON events");
    assert!(decoded_events.iter().any(|e| e.p < 0), "missing OFF events");
}

/// End-to-end regression test for the reader bug: builds a synthetic HDF5 file whose
/// "CD/events" dataset is a single ECF-compressed chunk (filter id 36559, registered as
/// H5Z_FLAG_OPTIONAL so no filter plugin implementation is required, written directly with
/// H5Dwrite_chunk so no encoder plugin is needed either), then calls
/// `read_prophesee_hdf5_native` end to end and asserts the returned `Event`s contain both
/// polarity 0 and polarity 1 (the post-fix 0/1 storage convention).
#[test]
fn test_read_prophesee_hdf5_native_preserves_both_polarities() {
    const NUM_EVENTS: usize = 2000;
    const ECF_FILTER_ID: c_uint = 36559;

    // Build synthetic PropheseeEvents with a realistic mix of p = 1 (ON) and p = -1 (OFF).
    // The timestamp origin must be non-zero: `is_valid_ecf_header`'s payload-location
    // heuristic in `extract_ecf_payload_from_chunk` requires the decoded timestamp origin
    // to be > 0 to accept a candidate header, exactly like real Prophesee recordings (which
    // never start at t=0). A zero origin makes it reject the true header and fall through
    // to a spurious match elsewhere in the chunk.
    const TIME_ORIGIN_US: i64 = 1_000_000;
    let mut source_events = Vec::with_capacity(NUM_EVENTS);
    for i in 0..NUM_EVENTS {
        let polarity: i16 = if i % 3 == 0 { -1 } else { 1 };
        source_events.push(PropheseeEvent {
            x: (i % 1280) as u16,
            y: (i % 720) as u16,
            p: polarity,
            t: TIME_ORIGIN_US + (i as i64) * 1000, // microseconds, strictly increasing
        });
    }
    assert!(source_events.iter().any(|e| e.p > 0));
    assert!(source_events.iter().any(|e| e.p < 0));

    let encoder = PropheseeECFEncoder::new();
    let payload = encoder
        .encode(&source_events)
        .expect("ECF encoding should succeed");

    let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let file_path = tmp_dir.path().join("synthetic_ecf.h5");
    let path_str = file_path.to_str().unwrap();

    write_synthetic_ecf_hdf5(path_str, NUM_EVENTS, ECF_FILTER_ID, &payload);

    let events = read_prophesee_hdf5_native(path_str)
        .expect("read_prophesee_hdf5_native should decode the synthetic ECF chunk");

    assert_eq!(
        events.len(),
        NUM_EVENTS,
        "expected all synthetic events to be decoded"
    );
    assert!(
        events.iter().any(|e| e.polarity == 1),
        "expected at least one polarity == 1 (ON) event, got polarities: {:?}",
        events.iter().map(|e| e.polarity).collect::<Vec<_>>()
    );
    assert!(
        events.iter().any(|e| e.polarity == 0),
        "expected at least one polarity == 0 (OFF) event; the pre-fix reader stored -1/1 \
         which the DataFrame builder then remapped to all +1, destroying OFF events. \
         Got polarities: {:?}",
        events.iter().map(|e| e.polarity).collect::<Vec<_>>()
    );
}

/// Write a minimal Prophesee-style HDF5 file: group "CD", chunked 1-D compound dataset
/// "events" (x: u16, y: u16, p: i16, t: i64, matching `PropheseeEvent`'s repr(C) layout),
/// a single chunk covering the whole dataset, with the ECF filter (id 36559) registered as
/// optional so HDF5 does not require an actual filter plugin to be installed. The
/// pre-encoded ECF payload is written directly into that chunk with H5Dwrite_chunk, bypassing
/// the (unimplemented) filter pipeline entirely, mirroring how `read_compressed_chunk` reads
/// raw chunk bytes with H5Dread_chunk in `src/ev_formats/hdf5_reader.rs`.
fn write_synthetic_ecf_hdf5(path: &str, num_events: usize, filter_id: c_uint, payload: &[u8]) {
    unsafe {
        let path_c = CString::new(path).unwrap();
        let file_id = h5f::H5Fcreate(
            path_c.as_ptr(),
            h5f::H5F_ACC_TRUNC,
            h5p::H5P_DEFAULT,
            h5p::H5P_DEFAULT,
        );
        assert!(file_id >= 0, "H5Fcreate failed");

        let cd_name = CString::new("CD").unwrap();
        let group_id = h5g::H5Gcreate2(
            file_id,
            cd_name.as_ptr(),
            h5p::H5P_DEFAULT,
            h5p::H5P_DEFAULT,
            h5p::H5P_DEFAULT,
        );
        assert!(group_id >= 0, "H5Gcreate2 failed");

        // Compound type matching PropheseeEvent's repr(C) layout: x(u16)@0, y(u16)@2,
        // p(i16)@4, t(i64)@8 (6 bytes padding for i64 alignment), size 16. This is only the
        // nominal on-disk element type: the actual bytes we write are the raw ECF payload,
        // never interpreted through this type (read_prophesee_hdf5_native reads chunks with
        // H5Dread_chunk, which bypasses per-element typed access entirely).
        let type_id = h5t::H5Tcreate(h5t::H5T_class_t::H5T_COMPOUND, 16);
        assert!(type_id >= 0, "H5Tcreate failed");

        let x_name = CString::new("x").unwrap();
        let y_name = CString::new("y").unwrap();
        let p_name = CString::new("p").unwrap();
        let t_name = CString::new("t").unwrap();
        assert!(
            h5t::H5Tinsert(
                type_id,
                x_name.as_ptr(),
                0,
                *hdf5_metno::globals::H5T_STD_U16LE
            ) >= 0
        );
        assert!(
            h5t::H5Tinsert(
                type_id,
                y_name.as_ptr(),
                2,
                *hdf5_metno::globals::H5T_STD_U16LE
            ) >= 0
        );
        assert!(
            h5t::H5Tinsert(
                type_id,
                p_name.as_ptr(),
                4,
                *hdf5_metno::globals::H5T_STD_I16LE
            ) >= 0
        );
        assert!(
            h5t::H5Tinsert(
                type_id,
                t_name.as_ptr(),
                8,
                *hdf5_metno::globals::H5T_STD_I64LE
            ) >= 0
        );

        let dims: [u64; 1] = [num_events as u64];
        let space_id = h5s::H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
        assert!(space_id >= 0, "H5Screate_simple failed");

        let dcpl_id = h5p::H5Pcreate(*hdf5_metno::globals::H5P_DATASET_CREATE);
        assert!(dcpl_id >= 0, "H5Pcreate failed");
        assert!(
            h5p::H5Pset_chunk(dcpl_id, 1, dims.as_ptr()) >= 0,
            "H5Pset_chunk failed"
        );
        // Register the ECF filter id as optional: HDF5 records it in the dataset's filter
        // pipeline metadata (so get_dataset_filters() in hdf5_reader.rs sees it) without
        // requiring an actual registered filter implementation, since we never invoke the
        // normal (filtered) read/write path below.
        assert!(
            h5p::H5Pset_filter(
                dcpl_id,
                filter_id as i32,
                h5z::H5Z_FLAG_OPTIONAL,
                0,
                std::ptr::null(),
            ) >= 0,
            "H5Pset_filter failed"
        );

        let events_name = CString::new("events").unwrap();
        let dset_id = h5d::H5Dcreate2(
            group_id,
            events_name.as_ptr(),
            type_id,
            space_id,
            h5p::H5P_DEFAULT,
            dcpl_id,
            h5p::H5P_DEFAULT,
        );
        assert!(dset_id >= 0, "H5Dcreate2 failed");

        // Write the pre-encoded ECF payload directly into chunk 0, bypassing the filter
        // pipeline (there is no real ECF encoder filter plugin registered on this system).
        let offset: [u64; 1] = [0];
        let write_status = h5d::H5Dwrite_chunk(
            dset_id,
            h5p::H5P_DEFAULT,
            0, // filter_mask: 0 means "filters were applied" (we pre-applied ECF ourselves)
            offset.as_ptr(),
            payload.len(),
            payload.as_ptr() as *const c_void,
        );
        assert!(write_status >= 0, "H5Dwrite_chunk failed");

        h5d::H5Dclose(dset_id);
        h5p::H5Pclose(dcpl_id);
        h5s::H5Sclose(space_id);
        h5t::H5Tclose(type_id);
        h5g::H5Gclose(group_id);
        h5f::H5Fclose(file_id);
    }
}
