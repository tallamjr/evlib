//! Event camera simulator: per-pixel log-intensity threshold crossing (ESIM model).

pub mod config;

pub use config::{threshold_maps, SimError, SimulatorConfig, THRESHOLD_FLOOR};
