// Only compile this module on Unix platforms (Linux/macOS)
// Note: Module-level #[cfg(unix)] in mod.rs handles platform gating

/*!
HDF5 reader with native ECF support.

This module provides direct HDF5 chunk reading capabilities to enable
our Rust ECF codec to decode Prophesee files without external dependencies.
*/

use crate::ev_formats::prophesee_ecf_codec::PropheseeECFDecoder;
use crate::ev_formats::{python, Event, EventFormat, TimestampUnit};
use hdf5_metno::{Dataset, File as H5File, Result as H5Result};
use hdf5_metno_sys::{h5d, h5p, h5s};
use polars::prelude::DataFrame;
use std::io;

use tracing::info;

/// Sensor geometry from the file level "geometry" attribute ("WIDTHxHEIGHT"),
/// written by Prophesee's HDF5 tooling. None when absent or unparseable, in
/// which case no coordinate validation is applied.
fn read_sensor_geometry(file: &H5File) -> Option<(u16, u16)> {
    use hdf5_metno::types::{VarLenAscii, VarLenUnicode};
    let attr = file.attr("geometry").ok()?;
    let text = attr
        .read_scalar::<VarLenAscii>()
        .map(|s| s.to_string())
        .or_else(|_| attr.read_scalar::<VarLenUnicode>().map(|s| s.to_string()))
        .ok()?;
    let (w, h) = text.split_once('x')?;
    Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
}

/// Read Prophesee HDF5 file using our native ECF decoder
pub fn read_prophesee_hdf5_native(path: &str) -> H5Result<Vec<Event>> {
    let file = H5File::open(path)?;
    let sensor_geometry = read_sensor_geometry(&file);

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

    // Process all chunks. Any chunk that fails to read, extract or decode
    // aborts the whole load (2026-08-08 review, R8): a partially decoded
    // file is silent corruption, not a usable result.
    let mut chunks_processed = 0;

    info!(
        total_events = total_events,
        chunk_size = chunk_size,
        num_chunks = num_chunks,
        "Starting ECF chunk processing"
    );

    for chunk_idx in 0..num_chunks {
        let compressed_data = read_compressed_chunk(&events_dataset, chunk_idx).map_err(|e| {
            hdf5_metno::Error::Internal(format!(
                "Failed to read HDF5 chunk {chunk_idx} with native ECF decoder. \
                 This may require HDF5 1.10.5+ or specific build options. Error: {e}"
            ))
        })?;

        // The compressed_data contains HDF5 chunk headers + ECF payload;
        // extract the ECF payload for decoding.
        let ecf_payload = extract_ecf_payload_from_chunk(&compressed_data).map_err(|e| {
            hdf5_metno::Error::Internal(format!(
                "Failed to extract ECF payload from chunk {chunk_idx}: {e}"
            ))
        })?;

        let decoded_events = decoder.decode(&ecf_payload).map_err(|e| {
            hdf5_metno::Error::Internal(format!("Failed to decode ECF chunk {chunk_idx}: {e}"))
        })?;
        let event_count = decoded_events.len();

        // Convert PropheseeEvent to Event
        for ecf_event in decoded_events {
            // Validate against the sensor geometry declared by
            // the file, if any. An out of range coordinate from
            // a chunk that "decoded" means the ECF payload was
            // misparsed, so it is corruption, not noise.
            if let Some((width, height)) = sensor_geometry {
                if ecf_event.x >= width || ecf_event.y >= height {
                    return Err(hdf5_metno::Error::Internal(format!(
                        "Decoded event ({}, {}) outside declared sensor geometry {width}x{height} in chunk {chunk_idx}",
                        ecf_event.x, ecf_event.y
                    )));
                }
            }

            // Prophesee ECF timestamps are integer microseconds; stored
            // losslessly in f64 and declared Microseconds at the
            // build_polars_dataframe call site (2026-08-08 review, R4).
            all_events.push(Event {
                t: ecf_event.t as f64,
                x: ecf_event.x,
                y: ecf_event.y,
                // Store 0/1 like every other reader: build_polars_dataframe's
                // HDF5 branch converts 0/1 to -1/1 (mod.rs); storing -1/1 here
                // made that conversion map -1 to +1, destroying OFF events.
                polarity: if ecf_event.p > 0 { 1 } else { 0 },
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

    // A decoded count short of the dataset's declared length is silent
    // truncation (the geometry check above rejects out-of-range coordinates
    // with a hard error rather than dropping them) and must error, not warn.
    if all_events.len() != total_events {
        return Err(hdf5_metno::Error::Internal(format!(
            "ECF decode incomplete: decoded {} of {} events",
            all_events.len(),
            total_events
        )));
    }

    info!(
        events = all_events.len(),
        total = total_events,
        chunks_total = num_chunks,
        chunks_successful = chunks_processed,
        "Native Rust ECF decoder completed"
    );
    Ok(all_events)
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

                    // Convert chunk to events; t_chunk holds integer microseconds,
                    // declared Microseconds at the build_polars_dataframe call below.
                    for i in 0..chunk_len {
                        events.push(Event {
                            t: t_chunk[i] as f64, // Integer microseconds
                            x: x_chunk[i],        // Already u16
                            y: y_chunk[i],        // Already u16
                            polarity: p_chunk[i], // Keep as i8: 1 or -1
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

                return python::build_polars_dataframe(
                    &events,
                    EventFormat::HDF5,
                    TimestampUnit::Microseconds,
                )
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

            // This is a Prophesee HDF5 format: decode with the native Rust ECF
            // decoder. There is no fallback: the Python fallback imported a
            // module that has never existed under python/evlib, and the old
            // Rust fallback re-ran this same decoder that had just failed
            // (2026-08-08 review, R8). A decode failure now propagates with
            // its specific error instead of a generic "all decoders failed".
            info!("Decoding Prophesee ECF HDF5 file {}", path);
            let events = read_prophesee_hdf5_native(path)
                .map_err(|e| format!("Native ECF decode failed for {path}: {e}"))?;
            return python::build_polars_dataframe(
                &events,
                EventFormat::HDF5,
                TimestampUnit::Microseconds,
            )
            .map_err(|e| format!("DataFrame conversion failed: {e}").into());
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
            let x_arr: Vec<u16> = x_dataset.read_raw()?.to_vec();
            let y_arr: Vec<u16> = y_dataset.read_raw()?.to_vec();
            let p_arr: Vec<i8> = p_dataset.read_raw()?.to_vec();

            // read_raw::<f64>() would silently convert an integer-typed t
            // dataset to f64 through HDF5's implicit type conversion, so
            // an integer dataset must be routed through the microsecond
            // path explicitly rather than mislabelled as seconds.
            let t_descriptor = t_dataset.dtype()?.to_descriptor()?;
            let (events, unit) = match t_descriptor {
                hdf5_metno::types::TypeDescriptor::Integer(_)
                | hdf5_metno::types::TypeDescriptor::Unsigned(_) => {
                    let t_arr: Vec<i64> = t_dataset.read_raw()?.to_vec();
                    let events: Vec<Event> = t_arr
                        .iter()
                        .zip(x_arr.iter().zip(y_arr.iter().zip(p_arr.iter())))
                        .map(|(&t, (&x, (&y, &p)))| Event {
                            t: t as f64, // Integer microseconds
                            x,
                            y,
                            polarity: p,
                        })
                        .collect();
                    (events, TimestampUnit::Microseconds)
                }
                _ => {
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
                    (events, TimestampUnit::Seconds)
                }
            };

            return python::build_polars_dataframe(&events, EventFormat::HDF5, unit)
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
                            t: t as f64, // Integer microseconds
                            x,
                            y,
                            polarity: p,
                        })
                        .collect();

                    return python::build_polars_dataframe(
                        &events,
                        EventFormat::HDF5,
                        TimestampUnit::Microseconds,
                    )
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

                    return python::build_polars_dataframe(
                        &events,
                        EventFormat::HDF5,
                        TimestampUnit::Seconds,
                    )
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
