// Data formats module
// Handles reading and writing events from various file formats

use crate::ev_core::{Event, Events};
use hdf5::File as H5File;
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufRead, BufReader, Result as IoResult};
use std::path::Path;

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
                        return Some(Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Failed to parse timestamp: {}", e),
                        )));
                    }
                };

                let x = match parts[1].parse::<u16>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Failed to parse x-coordinate: {}", e),
                        )));
                    }
                };

                let y = match parts[2].parse::<u16>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Failed to parse y-coordinate: {}", e),
                        )));
                    }
                };

                let polarity = match parts[3].parse::<i8>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Failed to parse polarity: {}", e),
                        )));
                    }
                };

                Some(Ok(Event { t, x, y, polarity }))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Detect file format based on extension and load events accordingly
pub fn load_events(path: &str) -> Result<Events, Box<dyn std::error::Error>> {
    let path = Path::new(path);

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("h5") | Some("hdf5") => Ok(load_events_from_hdf5(path.to_str().unwrap(), None)?),
        Some("txt") | Some("csv") => Ok(load_events_from_text(path.to_str().unwrap())?),
        Some("bin") | Some("dat") => Ok(mmap_events(path.to_str().unwrap())?),
        _ => {
            // Try to guess based on file content
            let file = File::open(path)?;
            let mut buffer = [0; 8];

            // Read first few bytes to detect binary vs text
            if let Ok(n) = std::io::Read::read(&mut file.try_clone()?, &mut buffer) {
                if n >= 4 {
                    // Check if it looks like an HDF5 signature (89 48 44 46 0d 0a 1a 0a)
                    if buffer[0] == 0x89
                        && buffer[1] == 0x48
                        && buffer[2] == 0x44
                        && buffer[3] == 0x46
                    {
                        return Ok(load_events_from_hdf5(path.to_str().unwrap(), None)?);
                    }

                    // Check if it looks like text (ASCII range)
                    let is_text = buffer[..n].iter().all(|&b| {
                        b.is_ascii() && !b.is_ascii_control()
                            || b == b'\n'
                            || b == b'\t'
                            || b == b'\r'
                    });

                    if is_text {
                        return Ok(load_events_from_text(path.to_str().unwrap())?);
                    } else {
                        return Ok(mmap_events(path.to_str().unwrap())?);
                    }
                }
            }

            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Could not determine format of event file {}",
                    path.display()
                ),
            )))
        }
    }
}

/// Struct for chunking a stream of events into time windows
pub struct TimeWindowIter<'a> {
    events: &'a [Event],
    dt: f64, // time window length
    curr_index: usize,
}

impl<'a> TimeWindowIter<'a> {
    /// Create a new time window iterator
    pub fn new(events: &'a [Event], dt: f64) -> Self {
        TimeWindowIter {
            events,
            dt,
            curr_index: 0,
        }
    }
}

impl<'a> Iterator for TimeWindowIter<'a> {
    type Item = &'a [Event];

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_index >= self.events.len() {
            return None;
        }

        let start_time = self.events[self.curr_index].t;
        let end_time = start_time + self.dt;

        // Find the range of events in this time window
        let start_index = self.curr_index;

        while self.curr_index < self.events.len() && self.events[self.curr_index].t < end_time {
            self.curr_index += 1;
        }

        if start_index == self.curr_index {
            // No events in this window
            self.curr_index += 1; // Avoid infinite loop
            return self.next();
        }

        // Return slice of events in this window
        Some(&self.events[start_index..self.curr_index])
    }
}

/// Save events to an HDF5 file
pub fn save_events_to_hdf5(
    events: &Events,
    path: &str,
    dataset_name: Option<&str>,
) -> hdf5::Result<()> {
    let file = H5File::create(path)?;
    let _dataset_name = dataset_name.unwrap_or("events");

    // Prepare arrays
    let n = events.len();
    let mut ts = Vec::with_capacity(n);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut ps = Vec::with_capacity(n);

    for ev in events {
        ts.push(ev.t);
        xs.push(ev.x);
        ys.push(ev.y);
        ps.push(ev.polarity);
    }

    // Create datasets
    let t_dataset = file.new_dataset::<f64>().shape([n]).create("t")?;
    let x_dataset = file.new_dataset::<u16>().shape([n]).create("x")?;
    let y_dataset = file.new_dataset::<u16>().shape([n]).create("y")?;
    let p_dataset = file.new_dataset::<i8>().shape([n]).create("p")?;

    // Write data
    t_dataset.write(&ts)?;
    x_dataset.write(&xs)?;
    y_dataset.write(&ys)?;
    p_dataset.write(&ps)?;

    Ok(())
}

/// Save events to a text file
pub fn save_events_to_text(events: &Events, path: &str) -> IoResult<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    // Write header
    writeln!(file, "# t x y p")?;

    // Write events
    for ev in events {
        writeln!(file, "{} {} {} {}", ev.t, ev.x, ev.y, ev.polarity)?;
    }

    Ok(())
}

/// Python bindings for the data formats module
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::ev_core::from_numpy_arrays;
    use numpy::IntoPyArray;
    use pyo3::exceptions::PyIOError;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;

    /// Load events from a file
    #[pyfunction]
    pub fn load_events_py<'py>(py: Python<'py>, path: &str) -> PyResult<PyObject> {
        let events = load_events(path).map_err(|e| PyErr::new::<PyIOError, _>(format!("{}", e)))?;

        // Convert to numpy arrays
        let n = events.len();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut ts = Vec::with_capacity(n);
        let mut ps = Vec::with_capacity(n);

        for ev in &events {
            xs.push(ev.x as i64);
            ys.push(ev.y as i64);
            ts.push(ev.t);
            ps.push(ev.polarity as i64);
        }

        // Create numpy arrays
        let xs_array = numpy::ndarray::Array::from_vec(xs);
        let ys_array = numpy::ndarray::Array::from_vec(ys);
        let ts_array = numpy::ndarray::Array::from_vec(ts);
        let ps_array = numpy::ndarray::Array::from_vec(ps);

        // Convert to Python objects
        let xs_py = xs_array.into_pyarray(py).to_object(py);
        let ys_py = ys_array.into_pyarray(py).to_object(py);
        let ts_py = ts_array.into_pyarray(py).to_object(py);
        let ps_py = ps_array.into_pyarray(py).to_object(py);

        // Create result tuple
        let result = PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);

        Ok(result.into())
    }

    /// Save events to an HDF5 file
    #[pyfunction]
    #[pyo3(signature = (xs, ys, ts, ps, path, dataset_name=None))]
    pub fn save_events_to_hdf5_py<'py>(
        _py: Python<'py>,
        xs: numpy::PyReadonlyArray1<i64>,
        ys: numpy::PyReadonlyArray1<i64>,
        ts: numpy::PyReadonlyArray1<f64>,
        ps: numpy::PyReadonlyArray1<i64>,
        path: &str,
        dataset_name: Option<&str>,
    ) -> PyResult<()> {
        // Convert numpy arrays to our internal Events type
        let events = from_numpy_arrays(xs, ys, ts, ps);

        // Save to HDF5
        save_events_to_hdf5(&events, path, dataset_name)
            .map_err(|e| PyErr::new::<PyIOError, _>(format!("Failed to save HDF5 file: {}", e)))
    }

    /// Save events to a text file
    #[pyfunction]
    pub fn save_events_to_text_py<'py>(
        _py: Python<'py>,
        xs: numpy::PyReadonlyArray1<i64>,
        ys: numpy::PyReadonlyArray1<i64>,
        ts: numpy::PyReadonlyArray1<f64>,
        ps: numpy::PyReadonlyArray1<i64>,
        path: &str,
    ) -> PyResult<()> {
        // Convert numpy arrays to our internal Events type
        let events = from_numpy_arrays(xs, ys, ts, ps);

        // Save to text file
        save_events_to_text(&events, path)
            .map_err(|e| PyErr::new::<PyIOError, _>(format!("Failed to save text file: {}", e)))
    }

    /// Python class for EventFileIterator
    #[pyclass(name = "EventFileIterator")]
    pub struct PyEventFileIterator {
        inner: EventFileIterator,
    }

    #[pymethods]
    impl PyEventFileIterator {
        #[new]
        fn new(path: &str) -> PyResult<Self> {
            let inner = EventFileIterator::new(path).map_err(|e| {
                PyErr::new::<PyIOError, _>(format!("Failed to open event file: {}", e))
            })?;
            Ok(PyEventFileIterator { inner })
        }

        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(i64, i64, f64, i64)> {
            match slf.inner.next() {
                Some(Ok(event)) => Some((
                    event.x as i64,
                    event.y as i64,
                    event.t,
                    event.polarity as i64,
                )),
                Some(Err(_)) => None, // Skip errors
                None => None,
            }
        }
    }

    /// Python class for TimeWindowIter
    #[pyclass(name = "TimeWindowIter")]
    pub struct PyTimeWindowIter {
        events: Vec<Event>,
        dt: f64,
        inner: Option<TimeWindowIter<'static>>,
    }

    #[pymethods]
    impl PyTimeWindowIter {
        #[new]
        fn new(
            xs: numpy::PyReadonlyArray1<i64>,
            ys: numpy::PyReadonlyArray1<i64>,
            ts: numpy::PyReadonlyArray1<f64>,
            ps: numpy::PyReadonlyArray1<i64>,
            dt: f64,
        ) -> Self {
            // Convert numpy arrays to our internal Events type
            let events = from_numpy_arrays(xs, ys, ts, ps);

            // We'll recreate the iterator in __iter__ to avoid lifetime issues
            PyTimeWindowIter {
                events,
                dt,
                inner: None,
            }
        }

        fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
            // Recreate the iterator each time __iter__ is called
            let events_slice =
                unsafe { std::slice::from_raw_parts(slf.events.as_ptr(), slf.events.len()) };
            slf.inner = Some(TimeWindowIter::new(events_slice, slf.dt)); // Use the dt specified in new()
            slf
        }

        fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Vec<(i64, i64, f64, i64)>> {
            if let Some(iter) = &mut slf.inner {
                if let Some(event_slice) = iter.next() {
                    let mut result = Vec::with_capacity(event_slice.len());
                    for event in event_slice {
                        result.push((
                            event.x as i64,
                            event.y as i64,
                            event.t,
                            event.polarity as i64,
                        ));
                    }
                    Some(result)
                } else {
                    None
                }
            } else {
                None
            }
        }
    }
}
