//! Event camera simulator: per-pixel log-intensity threshold crossing (ESIM model).

pub mod config;
pub mod pixel;

pub use config::{threshold_maps, SimError, SimulatorConfig, THRESHOLD_FLOOR};
pub use pixel::{step_pixel, PixelParams, PixelState, NO_EVENT};
