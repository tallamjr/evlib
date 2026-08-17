//! Event camera simulator: per-pixel log-intensity threshold crossing (ESIM model).

pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod pixel;
pub mod simulator;

#[cfg(feature = "python")]
pub mod python;

pub use config::{threshold_maps, SimError, SimulatorConfig, THRESHOLD_FLOOR};
#[cfg(feature = "cuda")]
pub use cuda::{cuda_available, EventSimulatorCuda};
pub use pixel::{step_pixel, PixelParams, PixelState, NO_EVENT};
pub use simulator::{EventBatch, EventSimulator};
