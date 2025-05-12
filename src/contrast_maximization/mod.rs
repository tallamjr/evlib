// Contrast maximization module
// Tools for event alignment and motion estimation via contrast maximization

use crate::events_core::{Event, Events, DEVICE};
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use std::ops::Range;

/// Trait for a warp model that can warp events based on parameters.
pub trait WarpModel {
    type Params: Clone + std::fmt::Debug; // parameter type (e.g., a struct or tuple of floats)

    /// Warp the events according to the model parameters, returning a new list of events.
    fn warp_events(&self, events: &Events, params: &Self::Params) -> Events;
}

/// A constant optical flow warp model
pub struct FlowWarp {
    pub t_ref: f64, // reference time to warp events to
}

#[derive(Clone, Debug)]
pub struct FlowParams {
    pub vx: f64,
    pub vy: f64,
}

impl Default for FlowParams {
    fn default() -> Self {
        FlowParams { vx: 0.0, vy: 0.0 }
    }
}

impl WarpModel for FlowWarp {
    type Params = FlowParams;

    fn warp_events(&self, events: &Events, params: &FlowParams) -> Events {
        let mut warped = Vec::with_capacity(events.len());

        for e in events {
            let dt = e.t - self.t_ref;
            // Shift coordinates by velocity * time difference
            let new_x = (e.x as f64 + params.vx * dt).round();
            let new_y = (e.y as f64 + params.vy * dt).round();

            // Ensure coordinates are within bounds
            if new_x >= 0.0 && new_y >= 0.0 && new_x <= u16::MAX as f64 && new_y <= u16::MAX as f64
            {
                warped.push(Event {
                    t: self.t_ref, // All events are now at reference time
                    x: new_x as u16,
                    y: new_y as u16,
                    polarity: e.polarity,
                });
            }
        }

        warped
    }
}

/// Trait for an objective function to evaluate the quality of warped events.
pub trait ObjectiveFunction {
    /// Evaluate the objective (higher is better) given events and resolution.
    fn evaluate(&self, events: &Events, resolution: (u16, u16)) -> f64;

    /// Optionally, provide an analytical gradient w.rt warp parameters if available.
    fn gradient(&self, _events: &Events, _resolution: (u16, u16)) -> Option<(f64, f64)> {
        None // default: no analytic gradient
    }
}

/// Variance objective – measures the variance of the image of warped events.
pub struct VarianceObjective;

impl ObjectiveFunction for VarianceObjective {
    fn evaluate(&self, events: &Events, resolution: (u16, u16)) -> f64 {
        // Create an image (2D array) from events and compute intensity variance
        let (width, height) = (resolution.0 as usize, resolution.1 as usize);
        let mut image = vec![0u32; width * height];

        for e in events {
            if e.x < resolution.0 && e.y < resolution.1 {
                let idx = e.y as usize * width + e.x as usize;
                // Here we simply accumulate count of events per pixel as the "intensity"
                image[idx] += 1;
            }
        }

        // Compute variance of non-zero pixels
        let count: u32 = image.iter().sum();
        if count == 0 {
            return 0.0;
        }

        let mean = count as f64 / (width * height) as f64;
        let var = image
            .iter()
            .map(|&v| {
                let diff = v as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (width * height) as f64;

        var
    }
}

/// Count objective - calculates the number of "active" pixels (pixels with at least one event).
pub struct ActivePixelCountObjective;

impl ObjectiveFunction for ActivePixelCountObjective {
    fn evaluate(&self, events: &Events, resolution: (u16, u16)) -> f64 {
        // Create an image (2D array) from events and count non-zero pixels
        let (width, height) = (resolution.0 as usize, resolution.1 as usize);
        let mut image = vec![false; width * height];

        for e in events {
            if e.x < resolution.0 && e.y < resolution.1 {
                let idx = e.y as usize * width + e.x as usize;
                image[idx] = true;
            }
        }

        // Return the negative count (since we want to maximize contrast, which means minimizing active pixels)
        -1.0 * image.iter().filter(|&&active| active).count() as f64
    }
}

/// Perform a brute-force grid search over parameter space to maximize the objective.
pub fn grid_search_optimization<M, O>(
    warp_model: &M,
    objective: &O,
    param_grid: &[Range<f64>], // e.g. [vx_range, vy_range]
    step: f64,
    events: &Events,
    resolution: (u16, u16),
) -> (M::Params, f64)
where
    M: WarpModel,
    O: ObjectiveFunction,
    M::Params: Default + Clone + std::fmt::Debug,
{
    let mut best_score = f64::MIN;
    let mut best_params = M::Params::default();

    // Here we assume two parameters for FlowParams (vx, vy)
    if param_grid.len() != 2 {
        return (best_params, best_score);
    }

    let vx_range = &param_grid[0];
    let vy_range = &param_grid[1];

    let mut vx = vx_range.start;
    while vx <= vx_range.end {
        let mut vy = vy_range.start;
        while vy <= vy_range.end {
            // For FlowParams, we need to cast
            let params =
                if let Some(flow_params) = best_params.as_ref().downcast_ref::<FlowParams>() {
                    FlowParams { vx, vy }
                } else {
                    // Default generic approach - this won't work for all param types
                    // In a real implementation, this would need a better design
                    return (best_params, best_score);
                };

            // Warp events
            let warped_events = warp_model.warp_events(events, &params);

            // Evaluate objective
            let score = objective.evaluate(&warped_events, resolution);

            if score > best_score {
                best_score = score;
                best_params = params.clone();
            }

            vy += step;
        }
        vx += step;
    }

    (best_params, best_score)
}

/// Perform gradient-based optimization using Candle for autograd.
pub fn optimize<M, O>(
    warp_model: &M,
    objective: &O,
    initial_params: M::Params,
    events: &Events,
    resolution: (u16, u16),
    learning_rate: f64,
    max_iters: usize,
) -> CandleResult<M::Params>
where
    M: WarpModel,
    O: ObjectiveFunction,
    M::Params: Clone + std::fmt::Debug,
{
    // Extract the flow parameters (assuming FlowParams for this implementation)
    let flow_params = if let Some(params) = initial_params.as_ref().downcast_ref::<FlowParams>() {
        params.clone()
    } else {
        // For non-FlowParams types, we would need a different implementation
        return Ok(initial_params);
    };

    let mut params = flow_params.clone();
    let device = &*DEVICE;

    // For each iteration
    for _iter in 0..max_iters {
        // Represent current params as Candle tensors with gradient
        let vx_t = Tensor::new(params.vx as f32, &*device)?.requires_grad(true);
        let vy_t = Tensor::new(params.vy as f32, &*device)?.requires_grad(true);

        // Create tensors for event coords
        let mut xs = Vec::with_capacity(events.len());
        let mut ys = Vec::with_capacity(events.len());
        let mut ts = Vec::with_capacity(events.len());

        for ev in events {
            xs.push(ev.x as f32);
            ys.push(ev.y as f32);
            ts.push(ev.t as f32);
        }

        let xs_t = Tensor::from_vec(xs, (xs.len(),), &*device)?;
        let ys_t = Tensor::from_vec(ys, (ys.len(),), &*device)?;
        let ts_t = Tensor::from_vec(ts, (ts.len(),), &*device)?;

        // Reference time for warping
        let t_ref = events.last().map(|e| e.t).unwrap_or(0.0) as f32;

        // Compute time differences
        let dt = &ts_t - t_ref;

        // Compute warped coordinates
        let new_xs = &xs_t + &vx_t * &dt?;
        let new_ys = &ys_t + &vy_t * &dt?;

        // Create the image for calculating variance
        let (width, height) = (resolution.0 as usize, resolution.1 as usize);

        // Get warped coordinates as vectors
        let xs_warped = new_xs.to_vec1()?;
        let ys_warped = new_ys.to_vec1()?;

        // Accumulate histogram
        let mut histogram = vec![0f32; width * height];

        for (x, y) in xs_warped.iter().zip(ys_warped.iter()) {
            let xi = x.round() as i32;
            let yi = y.round() as i32;

            if xi >= 0 && yi >= 0 && xi < width as i32 && yi < height as i32 {
                histogram[(yi as usize * width + xi as usize)] += 1.0;
            }
        }

        // Create tensor from histogram
        let histogram_t = Tensor::from_vec(histogram, (height, width), device)?;

        // Compute variance
        let mean = histogram_t.mean_all()?;
        let variance = (histogram_t - &mean)?.sqr()?.mean_all()?;

        // Compute gradient
        variance.backward()?;

        // Extract gradients
        let grad_vx = vx_t.grad()?.to_scalar::<f32>()?;
        let grad_vy = vy_t.grad()?.to_scalar::<f32>()?;

        // Update parameters
        params.vx += learning_rate * grad_vx as f64;
        params.vy += learning_rate * grad_vy as f64;
    }

    // Return the optimized parameters
    if let Some(flow_ref) = initial_params.as_ref().downcast_ref::<FlowParams>() {
        Ok(params.clone().into())
    } else {
        Ok(initial_params)
    }
}

/// Module for applying contrast maximization to specific motion models
pub mod models {
    use super::*;

    /// Estimates optical flow from events using contrast maximization
    pub fn estimate_optical_flow(
        events: &Events,
        resolution: (u16, u16),
        vx_range: Range<f64>,
        vy_range: Range<f64>,
        step: f64,
    ) -> (f64, f64, f64) {
        // Create flow warp model and objective function
        let t_ref = events.last().map(|e| e.t).unwrap_or(0.0);
        let warp_model = FlowWarp { t_ref };
        let objective = VarianceObjective;

        // Do coarse grid search
        let (params, score) = grid_search_optimization(
            &warp_model,
            &objective,
            &[vx_range, vy_range],
            step,
            events,
            resolution,
        );

        // Return estimated flow and score
        (params.vx, params.vy, score)
    }
}
