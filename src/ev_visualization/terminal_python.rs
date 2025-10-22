// Only compile this module when terminal feature is enabled
#![cfg(feature = "terminal")]

/*!
Python bindings for terminal visualization.

This module provides PyO3 bindings for the terminal event visualizer,
allowing Python code to use the terminal UI features.
*/

use super::terminal::{TerminalEventVisualizer, TerminalVisualizationConfig};
use crate::from_numpy_arrays;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Terminal visualization bindings (optional)
#[pyclass]
#[derive(Clone)]
pub struct PyTerminalVisualizationConfig {
    pub inner: TerminalVisualizationConfig,
}

#[pymethods]
impl PyTerminalVisualizationConfig {
    #[new]
    #[pyo3(signature = (
        event_decay_ms = None,
        max_events = None,
        target_fps = None,
        show_stats = None,
        canvas_scale = None
    ))]
    pub fn new(
        event_decay_ms: Option<f32>,
        max_events: Option<usize>,
        target_fps: Option<f32>,
        show_stats: Option<bool>,
        canvas_scale: Option<f32>,
    ) -> Self {
        let mut config = TerminalVisualizationConfig::default();

        if let Some(decay) = event_decay_ms {
            config.event_decay_ms = decay;
        }
        if let Some(max) = max_events {
            config.max_events = max;
        }
        if let Some(fps) = target_fps {
            config.target_fps = fps;
        }
        if let Some(stats) = show_stats {
            config.show_stats = stats;
        }
        if let Some(scale) = canvas_scale {
            config.canvas_scale = scale;
        }

        Self { inner: config }
    }
}

#[pyclass]
pub struct PyTerminalEventVisualizer {
    visualizer: Option<TerminalEventVisualizer>,
}

#[pymethods]
impl PyTerminalEventVisualizer {
    #[new]
    pub fn new(config: &PyTerminalVisualizationConfig) -> PyResult<Self> {
        let visualizer = TerminalEventVisualizer::new(config.inner.clone()).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create terminal visualizer: {}",
                e
            ))
        })?;

        Ok(Self {
            visualizer: Some(visualizer),
        })
    }

    /// Add events to the visualizer
    pub fn add_events(
        &mut self,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
    ) -> PyResult<()> {
        if let Some(ref mut viz) = self.visualizer {
            let events = from_numpy_arrays(xs, ys, ts, ps);
            viz.add_events(events);
        }
        Ok(())
    }

    /// Handle input and return whether to continue
    pub fn handle_input(&mut self) -> PyResult<bool> {
        if let Some(ref mut viz) = self.visualizer {
            viz.handle_input().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Input error: {}", e))
            })?;
            Ok(!viz.should_quit())
        } else {
            Ok(false)
        }
    }

    /// Render a frame
    pub fn render_frame(&mut self) -> PyResult<()> {
        if let Some(ref mut viz) = self.visualizer {
            viz.render_frame().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Render error: {}", e))
            })?;
        }
        Ok(())
    }

    /// Check if should quit
    pub fn should_quit(&self) -> bool {
        self.visualizer.as_ref().map_or(true, |v| v.should_quit())
    }

    /// Check if paused
    pub fn is_paused(&self) -> bool {
        self.visualizer.as_ref().map_or(false, |v| v.is_paused())
    }

    /// Get statistics as tuple (frames_rendered, events_processed, fps)
    pub fn get_stats(&self) -> (u64, u64, f32) {
        if let Some(ref viz) = self.visualizer {
            let stats = viz.get_stats();
            (
                stats.frames_rendered,
                stats.events_processed,
                stats.current_fps,
            )
        } else {
            (0, 0, 0.0)
        }
    }
}

/// Create a terminal event stream viewer
#[pyfunction]
pub fn create_terminal_event_viewer(
    config: Option<&PyTerminalVisualizationConfig>,
) -> PyResult<PyTerminalEventVisualizer> {
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();
    PyTerminalEventVisualizer::new(&PyTerminalVisualizationConfig { inner: config })
}
