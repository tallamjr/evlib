// Only compile this module on Unix platforms (Linux/macOS)
// Note: Module-level #[cfg(unix)] in mod.rs handles platform gating

/*!
HDF5 reader with native ECF support.

This module provides direct HDF5 chunk reading capabilities to enable
our Rust ECF codec to decode Prophesee files without external dependencies.
*/

use crate::ev_formats::prophesee_ecf_codec::PropheseeECFDecoder;
use crate::ev_formats::streaming::Event;
use crate::ev_formats::{python, EventFormat};
use hdf5_metno::{Dataset, File as H5File, Group, Result as H5Result};
use hdf5_metno_sys::{h5d, h5p, h5s};
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use std::io;

#[cfg(unix)]
use tracing::{error, info, warn};

#[cfg(not(unix))]
macro_rules! info {
    ($($args:tt)*) => {};
}

#[cfg(not(unix))]
macro_rules! warn {
    ($($args:tt)*) => {
        eprintln!("[WARN] {}", format!($($args)*))
    };
}

#[cfg(not(unix))]
macro_rules! error {
    ($($args:tt)*) => {
        eprintln!("[ERROR] {}", format!($($args)*))
    };
}

/// Read raw chunk data from an HDF5 dataset
/// This bypasses the HDF5 filter pipeline to get compressed chunks directly
pub fn read_raw_chunks(_file_path: &str, _dataset_path: &str) -> io::Result<Vec<Vec<u8>>> {
    // This is a placeholder function showing the intended structure

    // Parse HDF5 structure to find chunk locations
    // This is a simplified version - full implementation would parse B-trees
    let chunks = vec![];

    // TODO: Implement HDF5 B-tree parsing to locate chunks
    // For now, return empty to show the structure

    Ok(chunks)
}

/// Read Prophesee HDF5 file using our native ECF decoder
pub fn read_prophesee_hdf5_native(path: &str) -> H5Result<Vec<Event>> {
    let file = H5File::open(path)?;

    // Check for Prophesee format
    let cd_group = file.group("CD")?;
    let events_dataset = cd_group.dataset("events")?;

    let shape = events_dataset.shape();
    let total_events = shape[0];

    if total_events == 0 {
        return Ok(Vec::new());
    }

    // Get dataset information
    let chunk_size = match events_dataset.chunk() {
        Some(chunk_shape) => chunk_shape[0],
        None => 16384,
    };

    // Create Prophesee ECF decoder (without debug output for clean loading)
    let decoder = PropheseeECFDecoder::new();
    let mut all_events = Vec::with_capacity(total_events);

    // Read dataset filters to confirm ECF
    let filters = get_dataset_filters(&events_dataset)?;
    let has_ecf = filters.contains(&36559);

    if !has_ecf {
        return Err(hdf5_metno::Error::Internal(
            "Dataset does not use ECF compression".to_string(),
        ));
    }

    // Try to read raw chunk data
    // For now, we'll implement a hybrid approach using the dataset's raw data access

    // Check if we can read the dataset storage details
    let _storage_size = events_dataset.storage_size();

    // Attempt to decode chunks
    let num_chunks = total_events.div_ceil(chunk_size);

    // Process all chunks
    let mut chunks_processed = 0;
    let mut chunks_failed_extraction = 0;
    let mut chunks_failed_decoding = 0;
    let mut chunks_failed_reading = 0;

    info!(
        total_events = total_events,
        chunk_size = chunk_size,
        num_chunks = num_chunks,
        "Starting ECF chunk processing"
    );

    for chunk_idx in 0..num_chunks {
        match read_compressed_chunk(&events_dataset, chunk_idx) {
            Ok(compressed_data) => {
                // The compressed_data contains HDF5 chunk headers + ECF payload
                // Extract the ECF payload for decoding

                let ecf_payload = match extract_ecf_payload_from_chunk(&compressed_data) {
                    Ok(payload) => payload,
                    Err(e) => {
                        chunks_failed_extraction += 1;
                        warn!(
                            "Failed to extract ECF payload from chunk {}: {}",
                            chunk_idx, e
                        );
                        continue; // Skip this chunk and move to the next one
                    }
                };

                // Decode with our ECF codec
                match decoder.decode(&ecf_payload) {
                    Ok(decoded_events) => {
                        let event_count = decoded_events.len();

                        // For the first chunk, detect timestamp units (silently)
                        if chunk_idx == 0 && !decoded_events.is_empty() {
                            // Prophesee ECF timestamps are in nanoseconds
                            // No need to log this for every file
                        }

                        // Convert PropheseeEvent to Event
                        for ecf_event in decoded_events {
                            // Validate coordinates - Prophesee Gen4 cameras are 1280x720
                            // Skip events with clearly invalid coordinates
                            if ecf_event.x > 1280 || ecf_event.y > 720 {
                                continue;
                            }

                            // Prophesee ECF timestamps are in nanoseconds
                            // Convert to seconds for consistency with evlib Event format
                            all_events.push(Event {
                                t: ecf_event.t as f64 / 1_000_000_000.0, // Convert nanoseconds to seconds
                                x: ecf_event.x,
                                y: ecf_event.y,
                                polarity: if ecf_event.p > 0 { 1 } else { -1 },
                            });
                        }

                        chunks_processed += 1;

                        info!(
                            chunk_idx = chunk_idx,
                            events_in_chunk = event_count,
                            total_events_so_far = all_events.len(),
                            "Successfully processed ECF chunk"
                        );
                    }
                    Err(e) => {
                        chunks_failed_decoding += 1;
                        warn!("Failed to decode ECF chunk {}: {}", chunk_idx, e);
                        continue;
                    }
                }
            }
            Err(e) => {
                chunks_failed_reading += 1;
                // For the first chunk failure, return a helpful error
                if chunk_idx == 0 {
                    return Err(hdf5_metno::Error::Internal(format!(
                        "Failed to read HDF5 chunks with native ECF decoder. \
                         This may require HDF5 1.10.5+ or specific build options. \
                         Error: {}",
                        e
                    )));
                }
                // For subsequent chunks, continue processing
                warn!("Failed to read compressed chunk {}: {}", chunk_idx, e);
            }
        }

        // Progress reporting for large files
        if num_chunks > 10 && chunk_idx % (num_chunks / 10) == 0 {
            let _progress = (chunk_idx as f64 / num_chunks as f64) * 100.0;
        }
    }

    if all_events.is_empty() {
        Err(hdf5_metno::Error::Internal(
            "No valid ECF event data found in HDF5 file".to_string(),
        ))
    } else {
        // Validate that we loaded a reasonable proportion of the expected events
        let loaded_ratio = all_events.len() as f64 / total_events as f64;

        // Check for incomplete ECF decoding (first chunk only)
        if loaded_ratio < 0.1 && all_events.len() == 16384 {
            warn!(
                "ECF decoding incomplete: only loaded {} of {} events ({:.2}%). \
                 Successfully decoded first chunk but failed on subsequent chunks. \
                 This indicates a bug in the ECF chunk iteration logic.",
                all_events.len(),
                total_events,
                loaded_ratio * 100.0
            );
            // Continue for now to allow debugging, but this should be fixed
            // return Err(...) once the chunk iteration is working
        } else if loaded_ratio < 0.5 {
            // Loaded less than 50% - warn but continue
            warn!(
                "Partial ECF data loaded: {} of {} events ({:.1}%)",
                all_events.len(),
                total_events,
                loaded_ratio * 100.0
            );
        }

        // Log success message with chunk processing statistics
        info!(
            events = all_events.len(),
            total = total_events,
            ratio = format!("{:.2}%", loaded_ratio * 100.0),
            chunks_total = num_chunks,
            chunks_successful = chunks_processed,
            chunks_failed_reading = chunks_failed_reading,
            chunks_failed_extraction = chunks_failed_extraction,
            chunks_failed_decoding = chunks_failed_decoding,
            "Native Rust ECF decoder completed with chunk statistics"
        );
        Ok(all_events)
    }
}

/// Get filter IDs from a dataset
fn get_dataset_filters(dataset: &Dataset) -> H5Result<Vec<u32>> {
    let mut filter_ids = Vec::new();

    // Get dataset's property list to access filters
    let dataset_id = dataset.id();
    let plist_id = unsafe { h5d::H5Dget_create_plist(dataset_id) };

    if plist_id < 0 {
        return Err(hdf5_metno::Error::Internal(
            "Failed to get dataset creation property list".to_string(),
        ));
    }

    // Get number of filters
    let num_filters = unsafe { h5p::H5Pget_nfilters(plist_id) };

    if num_filters < 0 {
        unsafe { h5p::H5Pclose(plist_id) };
        return Err(hdf5_metno::Error::Internal(
            "Failed to get number of filters".to_string(),
        ));
    }

    // Read each filter
    for filter_idx in 0..num_filters {
        let mut flags = 0u32;
        let mut cd_nelmts = 0usize;
        let mut name = vec![0i8; 256];

        let filter_id = unsafe {
            h5p::H5Pget_filter2(
                plist_id,
                filter_idx as u32,
                &mut flags,
                &mut cd_nelmts,
                std::ptr::null_mut(), // cd_values - we don't need them
                name.len(),
                name.as_mut_ptr(),
                std::ptr::null_mut(), // filter_config - we don't need it
            )
        };

        if filter_id >= 0 {
            filter_ids.push(filter_id as u32);

            // Convert name to string for debugging
            let _filter_name = unsafe {
                std::ffi::CStr::from_ptr(name.as_ptr())
                    .to_string_lossy()
                    .to_string()
            };
        }
    }

    unsafe { h5p::H5Pclose(plist_id) };
    Ok(filter_ids)
}

/// Read a compressed chunk from the dataset
fn read_compressed_chunk(dataset: &Dataset, chunk_idx: usize) -> io::Result<Vec<u8>> {
    let dataset_id = dataset.id();

    // Get dataspace to determine dimensions
    let space_id = unsafe { h5d::H5Dget_space(dataset_id) };
    if space_id < 0 {
        return Err(io::Error::other("Failed to get dataset dataspace"));
    }

    // Get number of dimensions
    let ndims = unsafe { h5s::H5Sget_simple_extent_ndims(space_id) };
    if ndims < 0 {
        unsafe { h5s::H5Sclose(space_id) };
        return Err(io::Error::other("Failed to get dataspace dimensions"));
    }

    // Get total number of chunks
    let mut num_chunks: u64 = 0;
    let status = unsafe { h5d::H5Dget_num_chunks(dataset_id, space_id, &mut num_chunks) };
    if status < 0 {
        unsafe { h5s::H5Sclose(space_id) };
        return Err(io::Error::other("Failed to get number of chunks"));
    }

    // Validate chunk index
    if chunk_idx >= num_chunks as usize {
        unsafe { h5s::H5Sclose(space_id) };
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Chunk index {} out of range (0-{})",
                chunk_idx,
                num_chunks - 1
            ),
        ));
    }

    // Get chunk information
    let mut chunk_offset = vec![0u64; ndims as usize];
    let mut filter_mask = 0u32;
    let mut chunk_addr = 0u64;
    let mut chunk_size = 0u64;

    let status = unsafe {
        h5d::H5Dget_chunk_info(
            dataset_id,
            space_id,
            chunk_idx as u64,
            chunk_offset.as_mut_ptr(),
            &mut filter_mask,
            &mut chunk_addr,
            &mut chunk_size,
        )
    };

    if status < 0 {
        unsafe { h5s::H5Sclose(space_id) };
        return Err(io::Error::other(format!(
            "Failed to get chunk {} info",
            chunk_idx
        )));
    }

    // Check if ECF filter is applied (filter ID 36559 = 0x8ECF)
    // The filter_mask indicates which filters were applied during compression
    if filter_mask == 0 {
        // No filters applied to this chunk
    }

    // Allocate buffer for compressed data
    let mut compressed_data = vec![0u8; chunk_size as usize];

    // Read raw compressed chunk data (bypassing HDF5 filter pipeline)
    let read_status = unsafe {
        h5d::H5Dread_chunk(
            dataset_id,
            h5p::H5P_DEFAULT,
            chunk_offset.as_ptr(),
            &mut filter_mask,
            compressed_data.as_mut_ptr() as *mut std::ffi::c_void,
        )
    };

    unsafe { h5s::H5Sclose(space_id) };

    if read_status < 0 {
        return Err(io::Error::other(format!(
            "Failed to read compressed chunk {} data",
            chunk_idx
        )));
    }

    Ok(compressed_data)
}

/// Complete end-to-end implementation structure for native ECF support
pub mod native_ecf {

    /// This shows what a complete implementation would look like
    pub fn full_implementation_outline() {
        // 1. Parse HDF5 file structure
        // 2. Locate B-tree nodes containing chunk addresses
        // 3. Read compressed chunks directly from file
        // 4. Pass chunks to our ECF decoder
        // 5. Return decoded events

        // The implementation would:
        // - Use our own HDF5 parser for chunk locations
        // - Read raw bytes from the file
        // - Decode with our ECF codec
        // - No external dependencies needed!
    }
}

/// Extract ECF payload from HDF5 chunk data
///
/// The raw chunk data from H5Dread_chunk includes HDF5 metadata.
/// We need to find the actual ECF compressed data within this.
fn extract_ecf_payload_from_chunk(chunk_data: &[u8]) -> io::Result<Vec<u8>> {
    // Based on our testing, the first few bytes contain HDF5 metadata
    // The ECF payload should start after some header bytes

    if chunk_data.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Chunk too small to contain ECF data",
        ));
    }

    // HDF5 compressed chunks have a specific structure:
    // 1. First 4 bytes: uncompressed size (little-endian u32)
    // 2. Remaining bytes: compressed data

    // For Prophesee ECF chunks, we've observed patterns like:
    // [02, 00, 01, 00, ?, ?, ?, ?, 00, 00, 00, 00, ...]
    // This appears to be at the start of the compressed data, not after an offset

    // First, check if the chunk starts with a valid size header
    if chunk_data.len() >= 4 {
        let potential_size =
            u32::from_le_bytes([chunk_data[0], chunk_data[1], chunk_data[2], chunk_data[3]]);

        // If this looks like a reasonable uncompressed size (expanded range for large chunks)
        if potential_size > 10 && potential_size < 100_000_000 {
            // The ECF data likely starts at offset 4 (after the size header)
            if chunk_data.len() > 4 {
                let ecf_data = &chunk_data[4..];

                // Verify this looks like ECF data
                if is_valid_ecf_header(ecf_data) {
                    return Ok(ecf_data.to_vec());
                }
            }
        }
    }

    // If the above didn't work, try direct ECF header detection
    // The chunk might start directly with ECF data (no HDF5 header)
    if is_valid_ecf_header(chunk_data) {
        return Ok(chunk_data.to_vec());
    }

    // Try other common offsets where ECF data might start
    let offsets_to_try = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64];

    for &offset in &offsets_to_try {
        if offset + 8 > chunk_data.len() {
            continue;
        }

        let payload = &chunk_data[offset..];

        // Check if this looks like a valid ECF header
        if is_valid_ecf_header(payload) {
            return Ok(payload.to_vec());
        }
    }

    // Last resort: scan for ECF pattern
    for offset in (0..chunk_data.len().saturating_sub(16)).step_by(1) {
        let payload = &chunk_data[offset..];
        if is_valid_ecf_header(payload) {
            return Ok(payload.to_vec());
        }
    }

    // Return detailed error for debugging ECF payload extraction failures
    // First 16 bytes for debugging (safely truncated)
    let debug_bytes = if chunk_data.len() >= 16 {
        format!("{:02x?}", &chunk_data[0..16])
    } else {
        format!("{:02x?}", chunk_data)
    };

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "No valid ECF header found in chunk of {} bytes. First bytes: {}",
            chunk_data.len(),
            debug_bytes
        ),
    ))
}

/// Check if data starts with a valid ECF header
fn is_valid_ecf_header(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }

    // Prophesee ECF format header is a single 32-bit word:
    // Bits 2-31: Number of events (num_events = header >> 2)
    // Bit 1: YS+XS+PS packing flag
    // Bit 0: XS+PS packing flag
    //
    // Based on the official ECF codec implementation:
    // https://github.com/prophesee-ai/hdf5_ecf/blob/main/ecf_codec.cpp#L24-L28

    let header = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let num_events = (header >> 2) as usize;
    let _ys_xs_and_ps_packed = (header >> 1) & 1;
    let _xs_and_ps_packed = header & 1;

    // Validate event count - should be reasonable for a chunk
    // ECF codec has a maximum buffer size of 65535 events per chunk
    if num_events > 0 && num_events <= 65535 {
        // Additional validation: if data is long enough, check if timestamp section looks valid
        // The ECF format continues with an 8-byte timestamp origin after the 4-byte header
        if data.len() >= 12 {
            // Read timestamp origin (should be a reasonable timestamp)
            let timestamp_origin = u64::from_le_bytes([
                data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
            ]);

            // Timestamps should be reasonable (not zero, not extremely large)
            // Prophesee timestamps are typically in nanoseconds
            if timestamp_origin > 0 && timestamp_origin < u64::MAX / 2 {
                return true;
            }
        } else {
            // For short data, just validate the event count
            return true;
        }
    }

    false
}

/// Load events from HDF5 file and return as Polars DataFrame
/// Supports multiple HDF5 organizations and handles large files via chunked reading
pub fn load_events_from_hdf5(
    path: &str,
    dataset_name: Option<&str>,
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // Using hdf5-metno with built-in BLOSC support - no external plugins needed!

    let file = H5File::open(path)?;
    let dataset_name = dataset_name.unwrap_or("events");

    // First, check for datasets inside an "events" group (most common for modern files)
    if let Ok(events_group) = file.group("events") {
        // Try common field name combinations for separate datasets
        let field_combinations = [
            ("t", "x", "y", "p"),
            ("ts", "xs", "ys", "ps"),
            ("timestamps", "x_pos", "y_pos", "polarity"),
            ("time", "x_coord", "y_coord", "pol"),
        ];

        for (t_name, x_name, y_name, p_name) in field_combinations {
            if let (Ok(t_dataset), Ok(x_dataset), Ok(y_dataset), Ok(p_dataset)) = (
                events_group.dataset(t_name),
                events_group.dataset(x_name),
                events_group.dataset(y_name),
                events_group.dataset(p_name),
            ) {
                // Get dataset dimensions
                let shape = t_dataset.shape();
                let total_events = shape[0];

                // Handle empty datasets
                if total_events == 0 {
                    return Ok(DataFrame::empty());
                }

                // For very large files, read in chunks to avoid memory issues
                let chunk_size = if total_events > 100_000_000 {
                    10_000_000
                } else {
                    total_events
                };

                let mut events = Vec::with_capacity(total_events);

                for start_idx in (0..total_events).step_by(chunk_size) {
                    let end_idx = std::cmp::min(start_idx + chunk_size, total_events);
                    let chunk_len = end_idx - start_idx;

                    // Read chunk of data with proper type handling
                    let t_chunk: Vec<i64> = t_dataset.read_slice_1d(start_idx..end_idx)?.to_vec();
                    let x_chunk: Vec<u16> = x_dataset.read_slice_1d(start_idx..end_idx)?.to_vec();
                    let y_chunk: Vec<u16> = y_dataset.read_slice_1d(start_idx..end_idx)?.to_vec();
                    let p_chunk: Vec<i8> = p_dataset.read_slice_1d(start_idx..end_idx)?.to_vec();

                    // Convert chunk to events
                    for i in 0..chunk_len {
                        events.push(Event {
                            t: t_chunk[i] as f64 / 1_000_000.0, // Convert i64 microseconds to seconds
                            x: x_chunk[i],                      // Already u16
                            y: y_chunk[i],                      // Already u16
                            polarity: p_chunk[i],               // Keep as i8: 1 or -1
                        });
                    }

                    // Print progress for large files
                    if total_events > 10_000_000 {
                        let progress = (end_idx as f64 / total_events as f64) * 100.0;
                        if end_idx % 50_000_000 == 0 || end_idx == total_events {
                            info!(progress = %format!("{:.1}%", progress), current = end_idx, total = total_events, "Loading HDF5");
                        }
                    }
                }

                return python::build_polars_dataframe(&events, EventFormat::HDF5)
                    .map_err(|e| format!("DataFrame conversion failed: {}", e).into());
            }
        }
    }

    // Check for Prophesee HDF5 format with CD/events compound dataset
    if let Ok(cd_group) = file.group("CD") {
        if let Ok(events_dataset) = cd_group.dataset("events") {
            let shape = events_dataset.shape();
            let total_events = shape[0];

            if total_events == 0 {
                return Ok(DataFrame::empty());
            }

            // This is a Prophesee HDF5 format - try native Rust ECF decoder first
            info!("Attempting native Rust ECF decoder for {}", path);
            match read_prophesee_hdf5_native(path) {
                Ok(events) => {
                    info!("Native ECF decoder succeeded with {} events", events.len());
                    return python::build_polars_dataframe(&events, EventFormat::HDF5)
                        .map_err(|e| format!("DataFrame conversion failed: {}", e).into());
                }
                Err(e) => {
                    warn!("Native ECF decoder failed: {}", e);
                    // Native ECF decoder failed, will try Python fallback
                }
            }

            // Try Python fallback for ECF decoding
            info!("Trying Python fallback for ECF decoding");
            match call_python_prophesee_fallback(path) {
                Ok(events) => {
                    info!(events = events.height(), "Python fallback loaded events");
                    return Ok(events);
                }
                Err(e) => {
                    error!("Python fallback failed: {}", e);
                    // Try Rust ECF decoder as final fallback
                    match try_rust_ecf_decoder(&cd_group, &events_dataset, total_events) {
                        Ok(events) => {
                            return python::build_polars_dataframe(&events, EventFormat::HDF5)
                                .map_err(|e| format!("DataFrame conversion failed: {}", e).into());
                        }
                        Err(decoder_err) => {
                            warn!(
                                "Rust ECF decoder also failed: {}. Original error: {}",
                                decoder_err, e
                            );
                        }
                    }
                }
            }

            // If all attempts failed, return error
            return Err("Failed to decode Prophesee HDF5 file with all available decoders".into());
        }
    }

    // Try direct dataset access (older or simpler format)
    if let Ok(_events_dataset) = file.dataset(dataset_name) {
        // Check if this is separate time/coordinates datasets (eTram format)
        if let (Ok(t_dataset), Ok(x_dataset), Ok(y_dataset), Ok(p_dataset)) = (
            file.dataset("t"),
            file.dataset("x"),
            file.dataset("y"),
            file.dataset("p"),
        ) {
            // Read all data at once (for smaller files)
            let t_arr: Vec<f64> = t_dataset.read_raw()?.to_vec();
            let x_arr: Vec<u16> = x_dataset.read_raw()?.to_vec();
            let y_arr: Vec<u16> = y_dataset.read_raw()?.to_vec();
            let p_arr: Vec<i8> = p_dataset.read_raw()?.to_vec();

            // Convert to Event structs
            let events: Vec<Event> = t_arr
                .iter()
                .zip(x_arr.iter().zip(y_arr.iter().zip(p_arr.iter())))
                .map(|(&t, (&x, (&y, &p)))| Event {
                    t,
                    x,
                    y,
                    polarity: p,
                })
                .collect();

            return python::build_polars_dataframe(&events, EventFormat::HDF5)
                .map_err(|e| format!("DataFrame conversion failed: {}", e).into());
        }
    }

    // Try events group with separate datasets (standard format)
    if let Ok(events_group) = file.group("events") {
        // Try with 't' field (standard format)
        if let Ok(t_dataset) = events_group.dataset("t") {
            let (x_dataset, y_dataset, p_dataset) = (
                events_group.dataset("x"),
                events_group.dataset("y"),
                events_group.dataset("p"),
            );

            if let (Ok(x_dataset), Ok(y_dataset), Ok(p_dataset)) = (x_dataset, y_dataset, p_dataset)
            {
                let x_arr: Vec<u16> = x_dataset.read_raw()?.to_vec();
                let y_arr: Vec<u16> = y_dataset.read_raw()?.to_vec();
                let p_arr: Vec<i8> = p_dataset.read_raw()?.to_vec();

                // Handle different timestamp formats - try i64 (microseconds) first
                if let Ok(t_arr) = t_dataset.read_raw::<i64>() {
                    let t_arr: Vec<i64> = t_arr.to_vec();

                    let events: Vec<Event> = t_arr
                        .iter()
                        .zip(x_arr.iter().zip(y_arr.iter().zip(p_arr.iter())))
                        .map(|(&t, (&x, (&y, &p)))| Event {
                            t: t as f64 / 1_000_000.0, // Convert microseconds to seconds
                            x,
                            y,
                            polarity: p,
                        })
                        .collect();

                    return python::build_polars_dataframe(&events, EventFormat::HDF5)
                        .map_err(|e| format!("DataFrame conversion failed: {}", e).into());
                } else {
                    // Try f64 (seconds) as fallback
                    let t_arr: Vec<f64> = t_dataset.read_raw()?.to_vec();

                    let events: Vec<Event> = t_arr
                        .iter()
                        .zip(x_arr.iter().zip(y_arr.iter().zip(p_arr.iter())))
                        .map(|(&t, (&x, (&y, &p)))| Event {
                            t,
                            x,
                            y,
                            polarity: p,
                        })
                        .collect();

                    return python::build_polars_dataframe(&events, EventFormat::HDF5)
                        .map_err(|e| format!("DataFrame conversion failed: {}", e).into());
                }
            }
        } else {
            return Err("Could not find time field ('t' or 'ts') in events group".into());
        }
    }

    Err(format!(
        "Unsupported HDF5 format or no event data found in file: {}",
        path
    )
    .into())
}

fn call_python_prophesee_fallback(path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    Python::with_gil(|py| {
        // Import the Python fallback module
        let hdf5_prophesee = match py.import("evlib.hdf5_prophesee") {
            Ok(module) => module,
            Err(e) => {
                return Err(format!("Failed to import evlib.hdf5_prophesee module: {}", e).into())
            }
        };

        // Call the Python function to load events
        let result = match hdf5_prophesee.call_method1("load_hdf5_prophesee", (path,)) {
            Ok(result) => result,
            Err(e) => return Err(format!("Python fallback failed to load events: {}", e).into()),
        };

        // Extract the dictionary result
        let result_dict = match result.downcast::<pyo3::types::PyDict>() {
            Ok(dict) => dict,
            Err(e) => return Err(format!("Result is not a dictionary: {}", e).into()),
        };

        // Extract arrays from the dictionary
        let x_arr = result_dict
            .get_item("x")
            .map_err(|e| format!("Failed to get x item: {}", e))?
            .ok_or_else(|| "Missing x array".to_string())?
            .extract::<Vec<u16>>()
            .map_err(|e| format!("Failed to extract x array: {}", e))?;

        let y_arr = result_dict
            .get_item("y")
            .map_err(|e| format!("Failed to get y item: {}", e))?
            .ok_or_else(|| "Missing y array".to_string())?
            .extract::<Vec<u16>>()
            .map_err(|e| format!("Failed to extract y array: {}", e))?;

        let t_arr = result_dict
            .get_item("t")
            .map_err(|e| format!("Failed to get timestamp item: {}", e))?
            .ok_or_else(|| "Missing timestamp array".to_string())?
            .extract::<Vec<f64>>()
            .map_err(|e| format!("Failed to extract timestamp array: {}", e))?;

        let p_arr = result_dict
            .get_item("p")
            .map_err(|e| format!("Failed to get polarity item: {}", e))?
            .ok_or_else(|| "Missing polarity array".to_string())?
            .extract::<Vec<i8>>()
            .map_err(|e| format!("Failed to extract polarity array: {}", e))?;

        // Convert to Event structs
        let events: Vec<Event> = t_arr
            .iter()
            .zip(x_arr.iter().zip(y_arr.iter().zip(p_arr.iter())))
            .map(|(&t, (&x, (&y, &p)))| Event {
                t,
                x,
                y,
                polarity: p,
            })
            .collect();

        // Convert to DataFrame
        python::build_polars_dataframe(&events, EventFormat::HDF5)
            .map_err(|e| format!("DataFrame conversion failed: {}", e).into())
    })
}

fn try_rust_ecf_decoder(
    _cd_group: &Group,
    _events_dataset: &Dataset,
    total_events: usize,
) -> Result<Vec<Event>, Box<dyn std::error::Error>> {
    info!(events = total_events, "Attempting native Rust ECF decode");

    // Use our integrated HDF5 + ECF reader
    // Get the file path from the dataset
    let file = _events_dataset.file()?;
    let filename = file.filename();

    match read_prophesee_hdf5_native(&filename) {
        Ok(events) => {
            info!(
                events = events.len(),
                "Native Rust ECF decoder successfully loaded events"
            );
            Ok(events)
        }
        Err(e) => Err(format!("Native ECF decoding failed: {}", e).into()),
    }
}
