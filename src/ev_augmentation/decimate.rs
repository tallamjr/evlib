//! Decimate augmentation (placeholder)
//!
//! This functionality is available in ev_filtering::downsampling with
//! SpatialDecimation strategy. Use that implementation instead.

use crate::ev_augmentation::{AugmentationError, AugmentationResult, Validatable};
// Removed: use crate::Events; - legacy type no longer exists
use crate::ev_filtering::downsampling::DownsamplingFilter;

/// Decimate augmentation (redirects to existing downsampling)
#[derive(Debug, Clone)]
pub struct DecimateAugmentation {
    /// Decimation factor (keep every nth event per pixel)
    pub n: usize,
}

impl DecimateAugmentation {
    /// Create a new decimate augmentation
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        format!("n={}", self.n)
    }
}

impl Validatable for DecimateAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.n == 0 {
            return Err(AugmentationError::InvalidConfig(
                "Decimation factor must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/* Commented out - legacy SingleAugmentation trait no longer exists
impl SingleAugmentation for DecimateAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        // Use existing spatial decimation functionality
        let filter = DownsamplingFilter::spatial_decimation(self.n);
        crate::ev_filtering::downsampling::apply_downsampling_filter(events, &filter)
            .map_err(|e| AugmentationError::ProcessingError(e.to_string()))
    }

    fn description(&self) -> String {
        format!("Decimate: {}", self.description())
    }
}
*/

/* Commented out - legacy Events type no longer exists
/// Convenience function for decimation
pub fn decimate_events(events: &Events, n: usize) -> AugmentationResult<Events> {
    let aug = DecimateAugmentation::new(n);
    aug.apply(events)
}
*/

/// Apply decimate using Polars operations
#[cfg(feature = "polars")]
use polars::prelude::*;

#[cfg(feature = "polars")]
pub fn apply_decimate_polars(
    df: LazyFrame,
    config: &DecimateAugmentation,
) -> PolarsResult<LazyFrame> {
    let filter = DownsamplingFilter::spatial_decimation(config.n);
    crate::ev_filtering::downsampling::apply_downsampling_filter_polars(df, &filter)
}
