// Voxel grid representation for events
// Based on the method described in "Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion"

use crate::ev_core::{Events, DEVICE};
use candle_core::{Result as CandleResult, Tensor};

/// A structure that converts events to voxel grid representation
pub struct EventsToVoxelGrid {
    pub num_bins: usize,
    pub width: usize,
    pub height: usize,
}

impl EventsToVoxelGrid {
    pub fn new(num_bins: usize, width: usize, height: usize) -> Self {
        Self {
            num_bins,
            width,
            height,
        }
    }

    /// Convert events to a voxel grid representation
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

        // Initialize an empty voxel grid
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

        // Process each event
        for event in events {
            // Skip events that are outside the frame
            if event.x >= self.width as u16 || event.y >= self.height as u16 {
                continue;
            }

            // Calculate normalized timestamp
            let t_norm = (event.t - t_min) / dt;

            // Map the timestamp to a bin index
            let bin_idx = ((t_norm * self.num_bins as f64).floor() as usize).min(self.num_bins - 1);

            // Calculate the index in the flattened voxel grid
            let x = event.x as usize;
            let y = event.y as usize;
            let idx = bin_idx * self.height * self.width + y * self.width + x;

            // Increment the bin value based on polarity
            voxel_grid[idx] += event.polarity as f32;
        }

        // Create tensor from voxel grid
        Tensor::from_vec(
            voxel_grid,
            (self.num_bins, self.height, self.width),
            &DEVICE,
        )
    }
}
