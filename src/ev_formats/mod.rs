// Data formats module
// Handles reading and writing events from various file formats

use crate::ev_core::{Event, Events};
use hdf5::File as H5File;
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufRead, BufReader, Result as IoResult};

/// Load events from an HDF5 file
///
/// Expects a dataset "events" or similar with fields t, x, y, polarity.
///
/// # Arguments
/// * `path` - Path to the HDF5 file
/// * `dataset_name` - Name of the dataset containing events (default: "events")
pub fn load_events_from_hdf5(path: &str, dataset_name: Option<&str>) -> hdf5::Result<Events> {
    let file = H5File::open(path)?;
    let dataset_name = dataset_name.unwrap_or("events");

    // Check if we have a compound dataset
    if let Ok(dataset) = file.dataset(dataset_name) {
        // Try reading as an array of tuples (t,x,y,p)
        let data: Vec<(f64, u16, u16, i8)> = dataset.read_raw()?.to_vec();

        // Convert into our Event struct
        let events: Events = data
            .into_iter()
            .map(|(t, x, y, p)| Event {
                t,
                x,
                y,
                polarity: p,
            })
            .collect();

        return Ok(events);
    }

    // Fallback: check for separate datasets
    let field_names = [
        ("t", "x", "y", "p"),
        ("timestamps", "x_pos", "y_pos", "polarity"),
        ("ts", "xs", "ys", "ps"),
    ];

    for (t_name, x_name, y_name, p_name) in field_names {
        if let (Ok(t_dataset), Ok(x_dataset), Ok(y_dataset), Ok(p_dataset)) = (
            file.dataset(t_name),
            file.dataset(x_name),
            file.dataset(y_name),
            file.dataset(p_name),
        ) {
            let t_arr: Vec<f64> = t_dataset.read_raw()?.to_vec();
            let x_arr: Vec<u16> = x_dataset.read_raw()?.to_vec();
            let y_arr: Vec<u16> = y_dataset.read_raw()?.to_vec();
            let p_arr: Vec<i8> = p_dataset.read_raw()?.to_vec();

            let n = t_arr.len();
            let mut events = Vec::with_capacity(n);

            for i in 0..n {
                events.push(Event {
                    t: t_arr[i],
                    x: x_arr[i],
                    y: y_arr[i],
                    polarity: p_arr[i],
                });
            }

            return Ok(events);
        }
    }

    // If we get here, we couldn't find the data in any expected format
    Err(hdf5::Error::Internal(format!(
        "Could not find event data in HDF5 file {}",
        path
    )))
}

/// Load events from a plain text file (one event per line)
///
/// Format is expected as: "t x y p" (floating timestamp, int x, int y, int polarity)
/// Each line contains space-separated values for one event.
///
/// # Arguments
/// * `path` - Path to the text file
pub fn load_events_from_text(path: &str) -> IoResult<Events> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut events = Events::new();

    // Estimate capacity if possible
    if let Ok(metadata) = std::fs::metadata(path) {
        let file_size = metadata.len() as usize;
        // Assuming average of 20 bytes per line
        events.reserve(file_size / 20);
    } else {
        events.reserve(1000000); // Default pre-allocation
    }

    for line_res in reader.lines() {
        let line = line_res?;
        if line.is_empty() || line.starts_with('#') {
            continue; // Skip empty lines and comments
        }

        // Parse the four values
        let mut parts = line.split_whitespace();

        if let (Some(t_str), Some(x_str), Some(y_str), Some(p_str)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        {
            // Parse values
            if let (Ok(t), Ok(x), Ok(y), Ok(p)) = (
                t_str.parse::<f64>(),
                x_str.parse::<u16>(),
                y_str.parse::<u16>(),
                p_str.parse::<i8>(),
            ) {
                events.push(Event {
                    t,
                    x,
                    y,
                    polarity: p,
                });
            }
        }
    }

    Ok(events)
}

/// Memory-map a binary event file and return events
///
/// This is useful for large files that would not fit in memory.
/// The binary format should be a sequence of Event structures.
///
/// # Arguments
/// * `path` - Path to the binary file
pub fn mmap_events(path: &str) -> IoResult<Events> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Calculate the number of events in the file
    let event_size = std::mem::size_of::<Event>();
    let n = mmap.len() / event_size;

    // Get a pointer to the mapped memory and interpret as Event array
    let ptr = mmap.as_ptr() as *const Event;
    let events_slice = unsafe { std::slice::from_raw_parts(ptr, n) };

    // Copy events to a Vec to own the data
    let events = events_slice.to_vec();

    Ok(events)
}

/// Struct for iterating through a text file of events line by line
/// without loading everything into memory at once
pub struct EventFileIterator {
    reader: BufReader<File>,
}

impl EventFileIterator {
    /// Create a new iterator from a text file path
    pub fn new(path: &str) -> IoResult<Self> {
        let file = File::open(path)?;
        Ok(EventFileIterator {
            reader: BufReader::new(file),
        })
    }
}

impl Iterator for EventFileIterator {
    type Item = IoResult<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();

        // Read the next line
        match self.reader.read_line(&mut line) {
            Ok(0) => None, // EOF
            Ok(_) => {
                // Skip empty lines and comments
                if line.trim().is_empty() || line.starts_with('#') {
                    return self.next();
                }

                // Parse the line
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 4 {
                    return self.next(); // Not enough fields
                }

                // Parse values
                let t = match parts[0].parse::<f64>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                    }
                };

                let x = match parts[1].parse::<u16>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                    }
                };

                let y = match parts[2].parse::<u16>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                    }
                };

                let p = match parts[3].parse::<i8>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                    }
                };

                // Create and return event
                Some(Ok(Event {
                    t,
                    x,
                    y,
                    polarity: p,
                }))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

// Window-based event iterator that returns chunks of events based on time windows
pub struct TimeWindowIter<'a> {
    events: &'a Events,
    window_duration: f64,
    current_idx: usize,
    start_time: f64,
    end_time: f64,
}

impl<'a> TimeWindowIter<'a> {
    /// Create a new iterator that returns time-windowed chunks of events
    ///
    /// # Arguments
    /// * `events` - Event array to iterate over
    /// * `window_duration` - Duration of each time window in seconds
    pub fn new(events: &'a Events, window_duration: f64) -> Self {
        let start_time = if !events.is_empty() { events[0].t } else { 0.0 };

        let end_time = start_time + window_duration;

        TimeWindowIter {
            events,
            window_duration,
            current_idx: 0,
            start_time,
            end_time,
        }
    }
}

impl Iterator for TimeWindowIter<'_> {
    type Item = Vec<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.events.len() {
            return None;
        }

        let mut window = Vec::new();
        let mut idx = self.current_idx;

        // Collect events within current time window
        while idx < self.events.len() && self.events[idx].t < self.end_time {
            window.push(self.events[idx]);
            idx += 1;
        }

        // Update state for next iteration
        self.current_idx = idx;
        self.start_time = self.end_time;
        self.end_time += self.window_duration;

        // Only return Some if we found events in this window
        if window.is_empty() {
            self.next()
        } else {
            Some(window)
        }
    }
}

/// Python bindings for the formats module
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use numpy::{PyArray1, PyReadonlyArray1};
    use pyo3::prelude::*;
    use std::io::Write;
    use std::path::Path;

    /// Load events from a file (text, HDF5, or binary)
    ///
    /// Automatically detects the format based on file extension
    #[pyfunction]
    #[pyo3(name = "load_events")]
    pub fn load_events_py(
        py: Python<'_>,
        path: &str,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
        // Determine file format
        let path_obj = Path::new(path);
        let events = if let Some(ext) = path_obj.extension() {
            match ext.to_str().unwrap_or("").to_lowercase().as_str() {
                "h5" | "hdf5" => load_events_from_hdf5(path, None).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("HDF5 error: {}", e))
                })?,
                "bin" | "dat" => mmap_events(path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Binary file error: {}",
                        e
                    ))
                })?,
                _ => load_events_from_text(path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Text file error: {}", e))
                })?,
            }
        } else {
            // Default to text file
            load_events_from_text(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Text file error: {}", e))
            })?
        };

        // Separate event fields into arrays
        let n = events.len();

        let mut timestamps = Vec::with_capacity(n);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut polarities = Vec::with_capacity(n);

        for ev in events {
            timestamps.push(ev.t);
            xs.push(ev.x as i64);
            ys.push(ev.y as i64);
            polarities.push(ev.polarity as i64);
        }

        // Convert to numpy arrays
        let ts_array = PyArray1::from_vec(py, timestamps);
        let xs_array = PyArray1::from_vec(py, xs);
        let ys_array = PyArray1::from_vec(py, ys);
        let ps_array = PyArray1::from_vec(py, polarities);

        Ok((
            xs_array.to_object(py),
            ys_array.to_object(py),
            ts_array.to_object(py),
            ps_array.to_object(py),
        ))
    }

    /// Save events to an HDF5 file
    #[pyfunction]
    #[pyo3(name = "save_events_to_hdf5")]
    pub fn save_events_to_hdf5_py(
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        path: &str,
    ) -> PyResult<()> {
        // Validate array lengths
        let n = ts.len();
        if xs.len() != n || ys.len() != n || ps.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Arrays must have the same length",
            ));
        }

        // Create HDF5 file
        let file = H5File::create(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to create HDF5 file: {}",
                e
            ))
        })?;

        // Create a group to store the data
        let group = file.create_group("events").map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create group: {}", e))
        })?;

        // Convert arrays to Rust vectors
        let xs_vec: Vec<u16> = xs.as_array().iter().map(|&x| x as u16).collect();
        let ys_vec: Vec<u16> = ys.as_array().iter().map(|&y| y as u16).collect();
        let ts_vec: Vec<f64> = ts.as_slice().unwrap().to_vec();
        let ps_vec: Vec<i8> = ps.as_array().iter().map(|&p| p as i8).collect();

        // Create datasets for each component
        let xs_shape = [n];
        let xs_dataset = group
            .new_dataset::<u16>()
            .shape(xs_shape)
            .create("xs")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create xs dataset: {}",
                    e
                ))
            })?;
        xs_dataset.write(&xs_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write xs data: {}", e))
        })?;

        let ys_dataset = group
            .new_dataset::<u16>()
            .shape(xs_shape)
            .create("ys")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create ys dataset: {}",
                    e
                ))
            })?;
        ys_dataset.write(&ys_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write ys data: {}", e))
        })?;

        let ts_dataset = group
            .new_dataset::<f64>()
            .shape(xs_shape)
            .create("ts")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create ts dataset: {}",
                    e
                ))
            })?;
        ts_dataset.write(&ts_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write ts data: {}", e))
        })?;

        let ps_dataset = group
            .new_dataset::<i8>()
            .shape(xs_shape)
            .create("ps")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create ps dataset: {}",
                    e
                ))
            })?;
        ps_dataset.write(&ps_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write ps data: {}", e))
        })?;

        Ok(())
    }

    /// Save events to a text file, one event per line: "t x y p"
    #[pyfunction]
    #[pyo3(name = "save_events_to_text")]
    pub fn save_events_to_text_py(
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        path: &str,
    ) -> PyResult<()> {
        // Validate array lengths
        let n = ts.len();
        if xs.len() != n || ys.len() != n || ps.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Arrays must have the same length",
            ));
        }

        // Create output file
        let mut file = std::fs::File::create(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create file: {}", e))
        })?;

        // Write header
        file.write_all(b"# timestamp x y polarity\n").map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write header: {}", e))
        })?;

        // Write events
        for i in 0..n {
            let line = format!(
                "{:.12} {} {} {}\n",
                ts.get(i).unwrap(),
                xs.get(i).unwrap(),
                ys.get(i).unwrap(),
                ps.get(i).unwrap()
            );
            file.write_all(line.as_bytes()).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write line: {}", e))
            })?;
        }

        Ok(())
    }

    // Python wrapper for EventFileIterator
    #[pyclass]
    pub struct PyEventFileIterator {
        path: String,
        reader: Option<EventFileIterator>,
    }

    #[pymethods]
    impl PyEventFileIterator {
        #[new]
        fn new(path: String) -> Self {
            PyEventFileIterator { path, reader: None }
        }

        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&mut self, _py: Python<'_>) -> PyResult<Option<(f64, i64, i64, i64)>> {
            // Initialize reader if needed
            if self.reader.is_none() {
                self.reader = Some(EventFileIterator::new(&self.path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to open file: {}",
                        e
                    ))
                })?);
            }

            // Read next event
            if let Some(ref mut reader) = self.reader {
                match reader.next() {
                    Some(Ok(event)) => Ok(Some((
                        event.t,
                        event.x as i64,
                        event.y as i64,
                        event.polarity as i64,
                    ))),
                    Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error reading event: {}",
                        e
                    ))),
                    None => Ok(None),
                }
            } else {
                // This shouldn't happen, but just in case
                Ok(None)
            }
        }
    }

    // Python wrapper for TimeWindowIter
    #[pyclass]
    pub struct PyTimeWindowIter {
        events_xs: Vec<i64>,
        events_ys: Vec<i64>,
        events_ts: Vec<f64>,
        events_ps: Vec<i64>,
        window_duration: f64,
        current_idx: usize,
        start_time: f64,
        end_time: f64,
    }

    #[pymethods]
    impl PyTimeWindowIter {
        #[new]
        fn new(
            xs: PyReadonlyArray1<i64>,
            ys: PyReadonlyArray1<i64>,
            ts: PyReadonlyArray1<f64>,
            ps: PyReadonlyArray1<i64>,
            window_duration: f64,
        ) -> PyResult<Self> {
            // Convert to Rust vectors
            let xs_vec = xs.as_slice().unwrap().to_vec();
            let ys_vec = ys.as_slice().unwrap().to_vec();
            let ts_vec = ts.as_slice().unwrap().to_vec();
            let ps_vec = ps.as_slice().unwrap().to_vec();

            // Validate
            let n = ts_vec.len();
            if xs_vec.len() != n || ys_vec.len() != n || ps_vec.len() != n {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Arrays must have the same length",
                ));
            }

            let start_time = if !ts_vec.is_empty() { ts_vec[0] } else { 0.0 };
            let end_time = start_time + window_duration;

            Ok(PyTimeWindowIter {
                events_xs: xs_vec,
                events_ys: ys_vec,
                events_ts: ts_vec,
                events_ps: ps_vec,
                window_duration,
                current_idx: 0,
                start_time,
                end_time,
            })
        }

        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(
            &mut self,
            py: Python<'_>,
        ) -> PyResult<Option<(PyObject, PyObject, PyObject, PyObject)>> {
            if self.current_idx >= self.events_ts.len() {
                return Ok(None);
            }

            let mut xs_window = Vec::new();
            let mut ys_window = Vec::new();
            let mut ts_window = Vec::new();
            let mut ps_window = Vec::new();
            let mut idx = self.current_idx;

            // Collect events within current time window
            while idx < self.events_ts.len() && self.events_ts[idx] < self.end_time {
                xs_window.push(self.events_xs[idx]);
                ys_window.push(self.events_ys[idx]);
                ts_window.push(self.events_ts[idx]);
                ps_window.push(self.events_ps[idx]);
                idx += 1;
            }

            // Update state for next iteration
            self.current_idx = idx;
            self.start_time = self.end_time;
            self.end_time += self.window_duration;

            // If no events in this window, move to the next one
            if xs_window.is_empty() {
                return self.__next__(py);
            }

            // Convert to numpy arrays
            let xs_array = PyArray1::from_vec(py, xs_window);
            let ys_array = PyArray1::from_vec(py, ys_window);
            let ts_array = PyArray1::from_vec(py, ts_window);
            let ps_array = PyArray1::from_vec(py, ps_window);

            Ok(Some((
                xs_array.to_object(py),
                ys_array.to_object(py),
                ts_array.to_object(py),
                ps_array.to_object(py),
            )))
        }
    }
}
