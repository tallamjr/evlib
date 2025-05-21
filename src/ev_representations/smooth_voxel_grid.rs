// Smooth voxel grid representation for events
// This implementation is based on interpolated voxel grids for event-based vision

use crate::ev_core::{Events, DEVICE};
use candle_core::{Result as CandleResult, Tensor};

/// A structure to convert events into a smooth voxel grid representation.
/// This implementation uses temporal interpolation to create a smoother representation
/// of the event stream, preserving more temporal information.
pub struct SmoothEventsToVoxelGrid {
    /// Number of bins in the voxel grid
    pub num_bins: usize,
    /// Width of the sensor/output grid
    pub width: usize,
    /// Height of the sensor/output grid
    pub height: usize,
    /// Method to use for voxel grid creation
    /// - "trilinear" (default): Trilinear interpolation between bins and neighboring pixels
    /// - "bilinear": Bilinear interpolation between bins, discrete pixels
    /// - "temporal": Temporal interpolation only between bins
    pub interpolation: String,
}

impl SmoothEventsToVoxelGrid {
    /// Create a new smooth voxel grid converter with the specified parameters
    pub fn new(
        num_bins: usize,
        width: usize,
        height: usize,
        interpolation: Option<String>,
    ) -> Self {
        let interpolation = interpolation.unwrap_or_else(|| "trilinear".to_string());

        Self {
            num_bins,
            width,
            height,
            interpolation,
        }
    }

    /// Convert events to a smooth voxel grid representation
    pub fn convert(&self, events: &Events) -> CandleResult<Tensor> {
        if events.is_empty() {
            // Return empty voxel grid
            let voxel_data = vec![0.0f32; self.num_bins * self.height * self.width];
            return Tensor::from_vec(
                voxel_data,
                (self.num_bins, self.height, self.width),
                &DEVICE,
            );
        }

        // Initialize empty voxel grid
        let mut voxel_grid = vec![0.0f32; self.num_bins * self.height * self.width];

        // Get the timestamp range of events
        let t_min = events.first().map(|e| e.t).unwrap_or(0.0);
        let t_max = events.last().map(|e| e.t).unwrap_or(1.0);

        // Avoid division by zero if all events happen at the same time
        let dt = if (t_max - t_min).abs() < 1e-6 {
            1.0
        } else {
            t_max - t_min
        };

        // Process each event based on the interpolation method
        match self.interpolation.as_str() {
            "bilinear" => self.bilinear_interpolation(events, &mut voxel_grid, t_min, dt),
            "temporal" => self.temporal_interpolation(events, &mut voxel_grid, t_min, dt),
            _ => self.trilinear_interpolation(events, &mut voxel_grid, t_min, dt), // Default
        }

        // Create tensor from voxel grid
        Tensor::from_vec(
            voxel_grid,
            (self.num_bins, self.height, self.width),
            &DEVICE,
        )
    }

    /// Process events using trilinear interpolation (time + spatial)
    fn trilinear_interpolation(
        &self,
        events: &Events,
        voxel_grid: &mut [f32],
        t_min: f64,
        dt: f64,
    ) {
        for event in events {
            // Skip events that are outside the frame
            if event.x >= self.width as u16 || event.y >= self.height as u16 {
                continue;
            }

            // Calculate normalized timestamp
            let t_norm = (event.t - t_min) / dt;

            // Calculate the temporal position with interpolation weights
            let bin_pos = t_norm * (self.num_bins - 1) as f64;
            let bin_idx_lower = bin_pos.floor() as usize;
            let bin_idx_upper = (bin_idx_lower + 1).min(self.num_bins - 1);

            // Temporal interpolation weight
            let bin_weight_upper = bin_pos - bin_idx_lower as f64;
            let bin_weight_lower = 1.0 - bin_weight_upper;

            // Calculate the spatial positions with interpolation weights
            let x = event.x as f64;
            let y = event.y as f64;

            // Get integer positions and weights for spatial interpolation
            let x_lower = x.floor() as usize;
            let y_lower = y.floor() as usize;
            let x_upper = (x_lower + 1).min(self.width - 1);
            let y_upper = (y_lower + 1).min(self.height - 1);

            // Spatial interpolation weights
            let x_weight_upper = x - x_lower as f64;
            let x_weight_lower = 1.0 - x_weight_upper;
            let y_weight_upper = y - y_lower as f64;
            let y_weight_lower = 1.0 - y_weight_upper;

            // Polarity value
            let p_value = event.polarity as f32;

            // Update voxel grid with trilinear interpolation
            // For each of the 8 neighboring grid points, use interpolation weights

            // Lower time bin
            if bin_weight_lower > 0.0 {
                // Lower time, lower y, lower x
                let idx_ll_ll =
                    bin_idx_lower * self.height * self.width + y_lower * self.width + x_lower;
                voxel_grid[idx_ll_ll] +=
                    (bin_weight_lower * y_weight_lower * x_weight_lower) as f32 * p_value;

                // Lower time, lower y, upper x
                if x_upper != x_lower {
                    let idx_ll_lu =
                        bin_idx_lower * self.height * self.width + y_lower * self.width + x_upper;
                    voxel_grid[idx_ll_lu] +=
                        (bin_weight_lower * y_weight_lower * x_weight_upper) as f32 * p_value;
                }

                // Lower time, upper y, lower x
                if y_upper != y_lower {
                    let idx_ll_ul =
                        bin_idx_lower * self.height * self.width + y_upper * self.width + x_lower;
                    voxel_grid[idx_ll_ul] +=
                        (bin_weight_lower * y_weight_upper * x_weight_lower) as f32 * p_value;
                }

                // Lower time, upper y, upper x
                if x_upper != x_lower && y_upper != y_lower {
                    let idx_ll_uu =
                        bin_idx_lower * self.height * self.width + y_upper * self.width + x_upper;
                    voxel_grid[idx_ll_uu] +=
                        (bin_weight_lower * y_weight_upper * x_weight_upper) as f32 * p_value;
                }
            }

            // Upper time bin
            if bin_idx_upper != bin_idx_lower && bin_weight_upper > 0.0 {
                // Upper time, lower y, lower x
                let idx_ul_ll =
                    bin_idx_upper * self.height * self.width + y_lower * self.width + x_lower;
                voxel_grid[idx_ul_ll] +=
                    (bin_weight_upper * y_weight_lower * x_weight_lower) as f32 * p_value;

                // Upper time, lower y, upper x
                if x_upper != x_lower {
                    let idx_ul_lu =
                        bin_idx_upper * self.height * self.width + y_lower * self.width + x_upper;
                    voxel_grid[idx_ul_lu] +=
                        (bin_weight_upper * y_weight_lower * x_weight_upper) as f32 * p_value;
                }

                // Upper time, upper y, lower x
                if y_upper != y_lower {
                    let idx_ul_ul =
                        bin_idx_upper * self.height * self.width + y_upper * self.width + x_lower;
                    voxel_grid[idx_ul_ul] +=
                        (bin_weight_upper * y_weight_upper * x_weight_lower) as f32 * p_value;
                }

                // Upper time, upper y, upper x
                if x_upper != x_lower && y_upper != y_lower {
                    let idx_ul_uu =
                        bin_idx_upper * self.height * self.width + y_upper * self.width + x_upper;
                    voxel_grid[idx_ul_uu] +=
                        (bin_weight_upper * y_weight_upper * x_weight_upper) as f32 * p_value;
                }
            }
        }
    }

    /// Process events using bilinear interpolation (time only)
    fn bilinear_interpolation(&self, events: &Events, voxel_grid: &mut [f32], t_min: f64, dt: f64) {
        for event in events {
            // Skip events that are outside the frame
            if event.x >= self.width as u16 || event.y >= self.height as u16 {
                continue;
            }

            // Calculate normalized timestamp
            let t_norm = (event.t - t_min) / dt;

            // Calculate the temporal position with interpolation weights
            let bin_pos = t_norm * (self.num_bins - 1) as f64;
            let bin_idx_lower = bin_pos.floor() as usize;
            let bin_idx_upper = (bin_idx_lower + 1).min(self.num_bins - 1);

            // Temporal interpolation weight
            let bin_weight_upper = bin_pos - bin_idx_lower as f64;
            let bin_weight_lower = 1.0 - bin_weight_upper;

            // Calculate index in the flattened voxel grid (no spatial interpolation)
            let x = event.x as usize;
            let y = event.y as usize;

            // Polarity value
            let p_value = event.polarity as f32;

            // Lower time bin
            if bin_weight_lower > 0.0 {
                let idx_lower = bin_idx_lower * self.height * self.width + y * self.width + x;
                voxel_grid[idx_lower] += bin_weight_lower as f32 * p_value;
            }

            // Upper time bin
            if bin_idx_upper != bin_idx_lower && bin_weight_upper > 0.0 {
                let idx_upper = bin_idx_upper * self.height * self.width + y * self.width + x;
                voxel_grid[idx_upper] += bin_weight_upper as f32 * p_value;
            }
        }
    }

    /// Process events using only temporal interpolation
    fn temporal_interpolation(&self, events: &Events, voxel_grid: &mut [f32], t_min: f64, dt: f64) {
        for event in events {
            // Skip events that are outside the frame
            if event.x >= self.width as u16 || event.y >= self.height as u16 {
                continue;
            }

            // Calculate normalized timestamp
            let t_norm = (event.t - t_min) / dt;

            // Calculate the temporal position with interpolation weights
            let bin_pos = t_norm * (self.num_bins - 1) as f64;
            let bin_idx_lower = bin_pos.floor() as usize;
            let bin_idx_upper = (bin_idx_lower + 1).min(self.num_bins - 1);

            // Temporal interpolation weight
            let bin_weight_upper = bin_pos - bin_idx_lower as f64;
            let bin_weight_lower = 1.0 - bin_weight_upper;

            // Calculate index in the flattened voxel grid
            let x = event.x as usize;
            let y = event.y as usize;

            // Polarity value
            let p_value = event.polarity as f32;

            // Lower time bin
            let idx_lower = bin_idx_lower * self.height * self.width + y * self.width + x;
            voxel_grid[idx_lower] += bin_weight_lower as f32 * p_value;

            // Upper time bin (if not at the last bin)
            if bin_idx_upper != bin_idx_lower {
                let idx_upper = bin_idx_upper * self.height * self.width + y * self.width + x;
                voxel_grid[idx_upper] += bin_weight_upper as f32 * p_value;
            }
        }
    }
}

/// Convert events to a smooth voxel grid representation using the specified parameters
///
/// # Arguments
/// * `events` - The events to convert
/// * `num_bins` - Number of time bins in the voxel grid
/// * `width` - Width of the output grid
/// * `height` - Height of the output grid
/// * `interpolation` - Interpolation method: "trilinear", "bilinear", or "temporal"
///
/// # Returns
/// * A tensor of shape (num_bins, height, width) containing the voxel grid
pub fn events_to_smooth_voxel_grid(
    events: &Events,
    num_bins: usize,
    width: usize,
    height: usize,
    interpolation: Option<String>,
) -> CandleResult<Tensor> {
    let converter = SmoothEventsToVoxelGrid::new(num_bins, width, height, interpolation);
    converter.convert(events)
}

/// Convert events to a smooth voxel grid using trilinear interpolation by default
pub fn events_to_trilinear_voxel_grid(
    events: &Events,
    num_bins: usize,
    width: usize,
    height: usize,
) -> CandleResult<Tensor> {
    let interpolation = Some("trilinear".to_string());
    events_to_smooth_voxel_grid(events, num_bins, width, height, interpolation)
}
