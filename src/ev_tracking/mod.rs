// Event tracking module
// Tools for tracking points and objects through event streams

use crate::ev_core::Events;
use candle_core::{Result as CandleResult, Tensor};
use std::collections::HashMap;

/// Represents a 2D point for tracking
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

impl Point2D {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point2D) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// A query point with frame index and coordinates for ETAP tracking
#[derive(Debug, Clone)]
pub struct QueryPoint {
    pub frame_idx: i32,
    pub point: Point2D,
}

impl QueryPoint {
    pub fn new(frame_idx: i32, x: f32, y: f32) -> Self {
        Self {
            frame_idx,
            point: Point2D::new(x, y),
        }
    }

    /// Convert to ETAP format: [frame_idx, x, y]
    pub fn to_etap_format(&self) -> [f32; 3] {
        [self.frame_idx as f32, self.point.x, self.point.y]
    }
}

/// Track result containing predicted coordinates and visibility
#[derive(Debug, Clone)]
pub struct TrackResult {
    pub coords: Vec<Point2D>,
    pub visibility: Vec<f32>,
    pub frame_indices: Vec<i32>,
}

impl Default for TrackResult {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackResult {
    pub fn new() -> Self {
        Self {
            coords: Vec::new(),
            visibility: Vec::new(),
            frame_indices: Vec::new(),
        }
    }

    pub fn add_frame(&mut self, frame_idx: i32, point: Point2D, vis: f32) {
        self.frame_indices.push(frame_idx);
        self.coords.push(point);
        self.visibility.push(vis);
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Get visible points (visibility > threshold)
    pub fn visible_points(&self, threshold: f32) -> Vec<Point2D> {
        self.coords
            .iter()
            .zip(self.visibility.iter())
            .filter(|(_, &vis)| vis > threshold)
            .map(|(point, _)| *point)
            .collect()
    }
}

/// A simple point tracker for event streams
/// This provides basic tracking functionality and can be extended with ETAP
pub struct EventPointTracker {
    tracks: HashMap<u32, TrackResult>,
    next_track_id: u32,
    frame_idx: i32,
}

impl Default for EventPointTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl EventPointTracker {
    pub fn new() -> Self {
        Self {
            tracks: HashMap::new(),
            next_track_id: 0,
            frame_idx: 0,
        }
    }

    /// Initialize tracking with query points
    pub fn initialize_tracks(&mut self, query_points: &[QueryPoint]) -> Vec<u32> {
        let mut track_ids = Vec::new();

        for query in query_points {
            let track_id = self.next_track_id;
            self.next_track_id += 1;

            let mut track_result = TrackResult::new();
            track_result.add_frame(query.frame_idx, query.point, 1.0);

            self.tracks.insert(track_id, track_result);
            track_ids.push(track_id);
        }

        track_ids
    }

    /// Update tracks with new frame
    pub fn update_frame(&mut self) {
        self.frame_idx += 1;
    }

    /// Get track result for a specific track ID
    pub fn get_track(&self, track_id: u32) -> Option<&TrackResult> {
        self.tracks.get(&track_id)
    }

    /// Get all current track IDs
    pub fn track_ids(&self) -> Vec<u32> {
        self.tracks.keys().copied().collect()
    }

    /// Get current frame index
    pub fn current_frame(&self) -> i32 {
        self.frame_idx
    }
}

/// Keypoint extraction methods for object tracking
pub mod keypoints {
    use super::Point2D;

    /// Keypoint extraction method
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum KeypointMethod {
        /// Extract points from object contours
        Contour,
        /// Extract points using grid sampling
        Grid,
        /// Extract skeleton-based keypoints
        Skeleton,
        /// Extract corner features
        Corners,
    }

    /// Configuration for keypoint extraction
    #[derive(Debug, Clone)]
    pub struct KeypointConfig {
        pub method: KeypointMethod,
        pub num_points: usize,
        pub min_distance: f32,
        pub quality_threshold: f32,
    }

    impl Default for KeypointConfig {
        fn default() -> Self {
            Self {
                method: KeypointMethod::Contour,
                num_points: 10,
                min_distance: 5.0,
                quality_threshold: 0.01,
            }
        }
    }

    /// Extract keypoints from a binary mask
    pub fn extract_keypoints_from_mask(
        mask: &[bool],
        width: usize,
        height: usize,
        config: &KeypointConfig,
    ) -> Vec<Point2D> {
        match config.method {
            KeypointMethod::Grid => extract_grid_keypoints(mask, width, height, config),
            KeypointMethod::Contour => extract_contour_keypoints(mask, width, height, config),
            KeypointMethod::Skeleton => extract_skeleton_keypoints(mask, width, height, config),
            KeypointMethod::Corners => extract_corner_keypoints(mask, width, height, config),
        }
    }

    fn extract_grid_keypoints(
        mask: &[bool],
        width: usize,
        height: usize,
        config: &KeypointConfig,
    ) -> Vec<Point2D> {
        let mut keypoints = Vec::new();

        // Find bounding box of the mask
        let mut min_x = width;
        let mut max_x = 0;
        let mut min_y = height;
        let mut max_y = 0;

        for y in 0..height {
            for x in 0..width {
                if mask[y * width + x] {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        if min_x >= max_x || min_y >= max_y {
            return keypoints;
        }

        // Calculate grid dimensions
        let grid_size = (config.num_points as f32).sqrt().ceil() as usize;
        let x_step = (max_x - min_x) as f32 / (grid_size - 1).max(1) as f32;
        let y_step = (max_y - min_y) as f32 / (grid_size - 1).max(1) as f32;

        // Sample points from grid
        for i in 0..grid_size {
            for j in 0..grid_size {
                if keypoints.len() >= config.num_points {
                    break;
                }

                let x = min_x as f32 + i as f32 * x_step;
                let y = min_y as f32 + j as f32 * y_step;

                let x_int = x.round() as usize;
                let y_int = y.round() as usize;

                // Check if point is within mask
                if x_int < width && y_int < height && mask[y_int * width + x_int] {
                    let point = Point2D::new(x, y);

                    // Check minimum distance constraint
                    let too_close = keypoints
                        .iter()
                        .any(|kp: &Point2D| kp.distance_to(&point) < config.min_distance);

                    if !too_close {
                        keypoints.push(point);
                    }
                }
            }
            if keypoints.len() >= config.num_points {
                break;
            }
        }

        // Add centroid if we need more points
        if keypoints.len() < config.num_points {
            let centroid = calculate_centroid(mask, width, height);
            if let Some(center) = centroid {
                let too_close = keypoints
                    .iter()
                    .any(|kp| kp.distance_to(&center) < config.min_distance);
                if !too_close {
                    keypoints.push(center);
                }
            }
        }

        keypoints
    }

    fn extract_contour_keypoints(
        mask: &[bool],
        width: usize,
        height: usize,
        config: &KeypointConfig,
    ) -> Vec<Point2D> {
        let mut keypoints = Vec::new();

        // Find contour points (edge pixels)
        let mut contour_points = Vec::new();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                if mask[idx] {
                    // Check if this is an edge pixel
                    let neighbors = [
                        mask[idx - width], // top
                        mask[idx + width], // bottom
                        mask[idx - 1],     // left
                        mask[idx + 1],     // right
                    ];

                    // If any neighbor is false, this is an edge pixel
                    if neighbors.iter().any(|&n| !n) {
                        contour_points.push(Point2D::new(x as f32, y as f32));
                    }
                }
            }
        }

        if contour_points.is_empty() {
            return keypoints;
        }

        // Calculate centroid
        let centroid = calculate_centroid(mask, width, height);

        if let Some(center) = centroid {
            // Sort contour points by distance from center
            contour_points.sort_by(|a, b| {
                let dist_a = a.distance_to(&center);
                let dist_b = b.distance_to(&center);
                dist_b
                    .partial_cmp(&dist_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Select spaced-out points
            for point in contour_points {
                if keypoints.len() >= config.num_points - 1 {
                    break;
                }

                let too_close = keypoints
                    .iter()
                    .any(|kp| kp.distance_to(&point) < config.min_distance);

                if !too_close {
                    keypoints.push(point);
                }
            }

            // Add center point
            keypoints.push(center);
        }

        keypoints
    }

    fn extract_skeleton_keypoints(
        mask: &[bool],
        width: usize,
        height: usize,
        config: &KeypointConfig,
    ) -> Vec<Point2D> {
        // Implement morphological skeleton extraction using Zhang-Suen thinning algorithm
        let mut skeleton = mask.to_vec();
        let mut changed = true;

        // Apply Zhang-Suen thinning algorithm
        while changed {
            changed = false;
            let mut to_remove = Vec::new();

            // Sub-iteration 1
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    if skeleton[y * width + x] && should_remove_pixel(&skeleton, x, y, width, true)
                    {
                        to_remove.push(y * width + x);
                        changed = true;
                    }
                }
            }

            for &idx in &to_remove {
                skeleton[idx] = false;
            }

            to_remove.clear();

            // Sub-iteration 2
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    if skeleton[y * width + x] && should_remove_pixel(&skeleton, x, y, width, false)
                    {
                        to_remove.push(y * width + x);
                        changed = true;
                    }
                }
            }

            for &idx in &to_remove {
                skeleton[idx] = false;
            }
        }

        // Extract keypoints from skeleton
        let mut keypoints = Vec::new();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                if skeleton[y * width + x] {
                    // Check if this is a junction or endpoint
                    let neighbor_count = count_skeleton_neighbors(&skeleton, x, y, width);
                    if neighbor_count == 1 || neighbor_count >= 3 {
                        let point = Point2D::new(x as f32, y as f32);

                        // Check minimum distance constraint
                        let too_close = keypoints
                            .iter()
                            .any(|kp: &Point2D| kp.distance_to(&point) < config.min_distance);

                        if !too_close && keypoints.len() < config.num_points {
                            keypoints.push(point);
                        }
                    }
                }
            }
        }

        keypoints
    }

    fn extract_corner_keypoints(
        mask: &[bool],
        width: usize,
        height: usize,
        config: &KeypointConfig,
    ) -> Vec<Point2D> {
        // Implement Harris corner detection on binary mask
        let mut keypoints = Vec::new();

        // Convert boolean mask to float for gradient computation
        let mut image: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

        // Apply Gaussian smoothing
        image = gaussian_blur(&image, width, height, 1.0);

        // Compute image gradients
        let (grad_x, grad_y) = compute_gradients(&image, width, height);

        // Compute Harris response
        let k = 0.04; // Harris corner detection parameter
        let window_size = 3;

        for y in window_size..height - window_size {
            for x in window_size..width - window_size {
                let mut ixx = 0.0;
                let mut iyy = 0.0;
                let mut ixy = 0.0;

                // Compute structure tensor in local window
                for dy in -(window_size as i32)..(window_size as i32 + 1) {
                    for dx in -(window_size as i32)..(window_size as i32 + 1) {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        let idx = ny * width + nx;

                        let gx = grad_x[idx];
                        let gy = grad_y[idx];

                        ixx += gx * gx;
                        iyy += gy * gy;
                        ixy += gx * gy;
                    }
                }

                // Compute Harris response
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let response = det - k * trace * trace;

                // Check if this is a corner
                if response > config.quality_threshold {
                    let point = Point2D::new(x as f32, y as f32);

                    // Check minimum distance constraint
                    let too_close = keypoints
                        .iter()
                        .any(|kp: &Point2D| kp.distance_to(&point) < config.min_distance);

                    if !too_close && keypoints.len() < config.num_points {
                        keypoints.push(point);
                    }
                }
            }
        }

        // Sort by Harris response (would need to store responses, simplified here)
        keypoints.truncate(config.num_points);
        keypoints
    }

    fn calculate_centroid(mask: &[bool], width: usize, height: usize) -> Option<Point2D> {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0;

        for y in 0..height {
            for x in 0..width {
                if mask[y * width + x] {
                    sum_x += x as f32;
                    sum_y += y as f32;
                    count += 1;
                }
            }
        }

        if count > 0 {
            Some(Point2D::new(sum_x / count as f32, sum_y / count as f32))
        } else {
            None
        }
    }

    // Helper functions for skeleton extraction (Zhang-Suen thinning)
    fn should_remove_pixel(
        mask: &[bool],
        x: usize,
        y: usize,
        width: usize,
        first_subiteration: bool,
    ) -> bool {
        let idx = y * width + x;
        if !mask[idx] {
            return false;
        }

        // Get 8-connected neighbors in clockwise order starting from top
        let neighbors = [
            mask[(y - 1) * width + x],     // P2 (top)
            mask[(y - 1) * width + x + 1], // P3 (top-right)
            mask[y * width + x + 1],       // P4 (right)
            mask[(y + 1) * width + x + 1], // P5 (bottom-right)
            mask[(y + 1) * width + x],     // P6 (bottom)
            mask[(y + 1) * width + x - 1], // P7 (bottom-left)
            mask[y * width + x - 1],       // P8 (left)
            mask[(y - 1) * width + x - 1], // P9 (top-left)
        ];

        // Count the number of black-to-white transitions in the sequence P2,P3,P4,P5,P6,P7,P8,P9,P2
        let mut transitions = 0;
        for i in 0..8 {
            if !neighbors[i] && neighbors[(i + 1) % 8] {
                transitions += 1;
            }
        }

        // Count the number of black pixel neighbors
        let black_neighbors = neighbors.iter().filter(|&&n| n).count();

        // Zhang-Suen conditions
        let condition1 = (2..=6).contains(&black_neighbors);
        let condition2 = transitions == 1;

        if first_subiteration {
            let condition3 = !neighbors[0] || !neighbors[2] || !neighbors[4]; // P2 * P4 * P6 = 0
            let condition4 = !neighbors[2] || !neighbors[4] || !neighbors[6]; // P4 * P6 * P8 = 0
            condition1 && condition2 && condition3 && condition4
        } else {
            let condition3 = !neighbors[0] || !neighbors[2] || !neighbors[6]; // P2 * P4 * P8 = 0
            let condition4 = !neighbors[0] || !neighbors[4] || !neighbors[6]; // P2 * P6 * P8 = 0
            condition1 && condition2 && condition3 && condition4
        }
    }

    fn count_skeleton_neighbors(mask: &[bool], x: usize, y: usize, width: usize) -> usize {
        let neighbors = [
            mask[(y - 1) * width + x],     // top
            mask[(y - 1) * width + x + 1], // top-right
            mask[y * width + x + 1],       // right
            mask[(y + 1) * width + x + 1], // bottom-right
            mask[(y + 1) * width + x],     // bottom
            mask[(y + 1) * width + x - 1], // bottom-left
            mask[y * width + x - 1],       // left
            mask[(y - 1) * width + x - 1], // top-left
        ];

        neighbors.iter().filter(|&&n| n).count()
    }

    // Helper functions for corner detection
    fn gaussian_blur(image: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
        let kernel_size = (6.0 * sigma).ceil() as usize | 1; // Ensure odd size
        let half_size = kernel_size / 2;
        let mut kernel = vec![0.0; kernel_size];

        // Generate 1D Gaussian kernel
        let mut sum = 0.0;
        for (i, kernel_val) in kernel.iter_mut().enumerate().take(kernel_size) {
            let x = i as f32 - half_size as f32;
            *kernel_val = (-0.5 * (x / sigma).powi(2)).exp();
            sum += *kernel_val;
        }

        // Normalize kernel
        for k in &mut kernel {
            *k /= sum;
        }

        // Apply separable filter (horizontal then vertical)
        let mut temp = vec![0.0; width * height];
        let mut result = vec![0.0; width * height];

        // Horizontal pass
        for y in 0..height {
            for x in 0..width {
                let mut value = 0.0;
                for (i, &kernel_val) in kernel.iter().enumerate().take(kernel_size) {
                    let sx = x as i32 + i as i32 - half_size as i32;
                    let sx = sx.max(0).min(width as i32 - 1) as usize;
                    value += image[y * width + sx] * kernel_val;
                }
                temp[y * width + x] = value;
            }
        }

        // Vertical pass
        for y in 0..height {
            for x in 0..width {
                let mut value = 0.0;
                for (i, &kernel_val) in kernel.iter().enumerate().take(kernel_size) {
                    let sy = y as i32 + i as i32 - half_size as i32;
                    let sy = sy.max(0).min(height as i32 - 1) as usize;
                    value += temp[sy * width + x] * kernel_val;
                }
                result[y * width + x] = value;
            }
        }

        result
    }

    fn compute_gradients(image: &[f32], width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
        let mut grad_x = vec![0.0; width * height];
        let mut grad_y = vec![0.0; width * height];

        // Sobel operators
        let sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        let pixel_value = image[ny * width + nx];
                        let kernel_idx = ((dy + 1) * 3 + (dx + 1)) as usize;

                        gx += pixel_value * sobel_x[kernel_idx];
                        gy += pixel_value * sobel_y[kernel_idx];
                    }
                }

                grad_x[y * width + x] = gx;
                grad_y[y * width + x] = gy;
            }
        }

        (grad_x, grad_y)
    }
}

/// ETAP integration utilities
pub mod etap {
    use super::*;
    use crate::ev_representations::events_to_voxel_grid;

    /// Configuration for ETAP tracking
    #[derive(Debug, Clone)]
    pub struct ETAPConfig {
        pub window_length: usize,
        pub stride: usize,
        pub num_bins: u32,
        pub voxel_method: String,
        pub model_resolution: (u16, u16),
    }

    impl Default for ETAPConfig {
        fn default() -> Self {
            Self {
                window_length: 8,
                stride: 4,
                num_bins: 5,
                voxel_method: "count".to_string(),
                model_resolution: (512, 512),
            }
        }
    }

    /// Prepare event data for ETAP processing
    pub fn prepare_event_representation(
        events: &Events,
        resolution: (u16, u16),
        config: &ETAPConfig,
    ) -> CandleResult<Tensor> {
        // Create voxel grid representation
        let voxel_tensor =
            events_to_voxel_grid(events, resolution, config.num_bins, &config.voxel_method)?;

        // Reshape to ETAP format: [B, T, C, H, W]
        // For single batch and single time step
        let shape = voxel_tensor.shape();
        let dims = shape.dims();

        // Reshape from [C, H, W] to [1, 1, C, H, W] for batch and time dimensions
        voxel_tensor.reshape((1, 1, dims[0], dims[1], dims[2]))
    }

    /// Convert query points to ETAP format
    pub fn format_query_points(query_points: &[QueryPoint]) -> Vec<Vec<f32>> {
        query_points
            .iter()
            .map(|qp| qp.to_etap_format().to_vec())
            .collect()
    }

    /// Demonstration tracking function
    ///
    /// NOTE: This is a placeholder implementation for demonstration purposes.
    /// Real ETAP tracking would require:
    /// 1. Loading pre-trained ETAP model weights
    /// 2. Processing event representations through the neural network
    /// 3. Extracting point trajectories from model predictions
    ///
    /// This function provides basic motion estimation based on event intensity gradients
    /// for educational and testing purposes only.
    pub fn track_points_demo(
        event_representation: &Tensor,
        query_points: &[QueryPoint],
        num_frames: usize,
    ) -> CandleResult<HashMap<usize, TrackResult>> {
        let mut results = HashMap::new();

        // Extract event representation data for basic motion estimation
        let tensor_data = event_representation.flatten_all()?.to_vec1::<f32>()?;
        let shape = event_representation.shape();
        let dims = shape.dims();

        // Expect format [B, T, C, H, W] or similar
        if dims.len() < 3 {
            return Err(candle_core::Error::Msg(
                "Invalid tensor shape for tracking".to_string(),
            ));
        }

        let height = dims[dims.len() - 2];
        let width = dims[dims.len() - 1];

        for (i, query) in query_points.iter().enumerate() {
            let mut track_result = TrackResult::new();
            let mut current_point = query.point;

            for frame_idx in 0..num_frames {
                // Basic motion estimation using local event intensity
                let (motion_x, motion_y) =
                    estimate_local_motion(&tensor_data, current_point, width, height, frame_idx);

                // Update point position with estimated motion
                current_point.x += motion_x;
                current_point.y += motion_y;

                // Clamp to image boundaries
                current_point.x = current_point.x.max(0.0).min(width as f32 - 1.0);
                current_point.y = current_point.y.max(0.0).min(height as f32 - 1.0);

                // Estimate visibility based on local event activity
                let visibility = estimate_visibility(&tensor_data, current_point, width, height);

                track_result.add_frame(frame_idx as i32, current_point, visibility);
            }

            results.insert(i, track_result);
        }

        Ok(results)
    }

    /// Estimate local motion based on event gradients
    fn estimate_local_motion(
        tensor_data: &[f32],
        point: Point2D,
        width: usize,
        height: usize,
        _frame_idx: usize,
    ) -> (f32, f32) {
        let x = point.x.round() as usize;
        let y = point.y.round() as usize;

        if x == 0 || x >= width - 1 || y == 0 || y >= height - 1 {
            return (0.0, 0.0);
        }

        // Simple gradient-based motion estimation
        let window_size = 3i32;
        let mut grad_x = 0.0;
        let mut grad_y = 0.0;
        let mut count = 0;

        for dy in -window_size..=window_size {
            for dx in -window_size..=window_size {
                let nx = (x as i32 + dx).max(0).min(width as i32 - 1) as usize;
                let ny = (y as i32 + dy).max(0).min(height as i32 - 1) as usize;

                if nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1 {
                    let idx = ny * width + nx;
                    if idx < tensor_data.len() {
                        let curr_val = tensor_data[idx];

                        // Compute local gradients
                        if nx > 0 && (ny * width + nx - 1) < tensor_data.len() {
                            grad_x += curr_val - tensor_data[ny * width + nx - 1];
                        }
                        if ny > 0 && ((ny - 1) * width + nx) < tensor_data.len() {
                            grad_y += curr_val - tensor_data[(ny - 1) * width + nx];
                        }
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            grad_x /= count as f32;
            grad_y /= count as f32;

            // Scale motion based on gradient magnitude
            let motion_scale = 0.1;
            (grad_x * motion_scale, grad_y * motion_scale)
        } else {
            (0.0, 0.0)
        }
    }

    /// Estimate point visibility based on local event activity
    fn estimate_visibility(
        tensor_data: &[f32],
        point: Point2D,
        width: usize,
        height: usize,
    ) -> f32 {
        let x = point.x.round() as usize;
        let y = point.y.round() as usize;

        if x >= width || y >= height {
            return 0.0;
        }

        let window_size = 2i32;
        let mut activity = 0.0;
        let mut count = 0;

        for dy in -window_size..=window_size {
            for dx in -window_size..=window_size {
                let nx = (x as i32 + dx).max(0).min(width as i32 - 1) as usize;
                let ny = (y as i32 + dy).max(0).min(height as i32 - 1) as usize;
                let idx = ny * width + nx;

                if idx < tensor_data.len() {
                    activity += tensor_data[idx].abs();
                    count += 1;
                }
            }
        }

        if count > 0 {
            let avg_activity = activity / count as f32;
            // Normalize visibility to [0, 1] range
            (avg_activity * 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Python bindings for the tracking module
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::ev_core::from_numpy_arrays;
    use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;
    use std::collections::HashMap;

    /// Python wrapper for Point2D
    #[pyclass]
    #[derive(Clone)]
    pub struct PyPoint2D {
        #[pyo3(get, set)]
        pub x: f32,
        #[pyo3(get, set)]
        pub y: f32,
    }

    #[pymethods]
    impl PyPoint2D {
        #[new]
        pub fn new(x: f32, y: f32) -> Self {
            Self { x, y }
        }

        pub fn distance_to(&self, other: &PyPoint2D) -> f32 {
            Point2D::new(self.x, self.y).distance_to(&Point2D::new(other.x, other.y))
        }

        pub fn __repr__(&self) -> String {
            format!("Point2D(x={}, y={})", self.x, self.y)
        }
    }

    impl From<Point2D> for PyPoint2D {
        fn from(point: Point2D) -> Self {
            Self {
                x: point.x,
                y: point.y,
            }
        }
    }

    impl From<PyPoint2D> for Point2D {
        fn from(point: PyPoint2D) -> Self {
            Self {
                x: point.x,
                y: point.y,
            }
        }
    }

    /// Python wrapper for QueryPoint
    #[pyclass]
    #[derive(Clone)]
    pub struct PyQueryPoint {
        #[pyo3(get, set)]
        pub frame_idx: i32,
        #[pyo3(get, set)]
        pub point: PyPoint2D,
    }

    #[pymethods]
    impl PyQueryPoint {
        #[new]
        pub fn new(frame_idx: i32, x: f32, y: f32) -> Self {
            Self {
                frame_idx,
                point: PyPoint2D::new(x, y),
            }
        }

        pub fn to_etap_format(&self) -> Vec<f32> {
            vec![self.frame_idx as f32, self.point.x, self.point.y]
        }

        pub fn __repr__(&self) -> String {
            format!(
                "QueryPoint(frame_idx={}, x={}, y={})",
                self.frame_idx, self.point.x, self.point.y
            )
        }
    }

    /// Python wrapper for TrackResult
    #[pyclass]
    #[derive(Clone)]
    pub struct PyTrackResult {
        inner: TrackResult,
    }

    impl Default for PyTrackResult {
        fn default() -> Self {
            Self::new()
        }
    }

    #[pymethods]
    impl PyTrackResult {
        #[new]
        pub fn new() -> Self {
            Self {
                inner: TrackResult::new(),
            }
        }

        #[getter]
        pub fn coords(&self) -> Vec<PyPoint2D> {
            self.inner.coords.iter().map(|&p| p.into()).collect()
        }

        #[getter]
        pub fn visibility(&self) -> Vec<f32> {
            self.inner.visibility.clone()
        }

        #[getter]
        pub fn frame_indices(&self) -> Vec<i32> {
            self.inner.frame_indices.clone()
        }

        pub fn __len__(&self) -> usize {
            self.inner.len()
        }

        pub fn visible_points(&self, threshold: f32) -> Vec<PyPoint2D> {
            self.inner
                .visible_points(threshold)
                .into_iter()
                .map(|p| p.into())
                .collect()
        }

        pub fn add_frame(&mut self, frame_idx: i32, point: PyPoint2D, visibility: f32) {
            self.inner.add_frame(frame_idx, point.into(), visibility);
        }

        pub fn __repr__(&self) -> String {
            format!("TrackResult(length={})", self.inner.len())
        }
    }

    impl From<TrackResult> for PyTrackResult {
        fn from(result: TrackResult) -> Self {
            Self { inner: result }
        }
    }

    /// Extract keypoints from a binary mask
    #[pyfunction]
    #[pyo3(name = "extract_keypoints_from_mask")]
    pub fn extract_keypoints_from_mask_py(
        mask: PyReadonlyArray2<bool>,
        method: &str,
        num_points: usize,
        min_distance: f32,
    ) -> PyResult<Vec<PyPoint2D>> {
        let mask_array = mask.as_array();
        let (height, width) = mask_array.dim();
        let mask_flat: Vec<bool> = mask_array.iter().copied().collect();

        let keypoint_method = match method {
            "grid" => keypoints::KeypointMethod::Grid,
            "contour" => keypoints::KeypointMethod::Contour,
            "skeleton" => keypoints::KeypointMethod::Skeleton,
            "corners" => keypoints::KeypointMethod::Corners,
            _ => keypoints::KeypointMethod::Contour,
        };

        let config = keypoints::KeypointConfig {
            method: keypoint_method,
            num_points,
            min_distance,
            quality_threshold: 0.01,
        };

        let keypoints = keypoints::extract_keypoints_from_mask(&mask_flat, width, height, &config);

        Ok(keypoints.into_iter().map(|p| p.into()).collect())
    }

    /// Prepare event representation for ETAP tracking
    #[pyfunction]
    #[pyo3(name = "prepare_event_representation")]
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_event_representation_py(
        py: Python<'_>,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        resolution: (i64, i64),
        window_length: usize,
        num_bins: u32,
        voxel_method: &str,
    ) -> PyResult<PyObject> {
        // Convert to events
        let events = from_numpy_arrays(xs, ys, ts, ps);

        // Create ETAP config
        let config = etap::ETAPConfig {
            window_length,
            stride: 4,
            num_bins,
            voxel_method: voxel_method.to_string(),
            model_resolution: (512, 512),
        };

        // Prepare representation
        let representation = etap::prepare_event_representation(
            &events,
            (resolution.0 as u16, resolution.1 as u16),
            &config,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Convert to numpy array
        let shape = representation.shape();
        let dims = shape.dims();

        // Flatten the tensor to get the raw data
        let data: Vec<f32> = representation
            .flatten_all()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?
            .to_vec1()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Create numpy array with proper shape
        let numpy_array = numpy::ndarray::Array::from_shape_vec(
            (dims[0], dims[1], dims[2], dims[3], dims[4]),
            data,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        Ok(numpy_array.into_pyarray(py).to_object(py))
    }

    /// Demonstration ETAP tracking function for Python
    #[pyfunction]
    #[pyo3(name = "track_points_demo")]
    pub fn track_points_demo_py(
        _py: Python<'_>,
        _event_representation: PyObject,
        query_points: Vec<PyQueryPoint>,
        num_frames: usize,
    ) -> PyResult<HashMap<usize, PyTrackResult>> {
        // Convert query points
        let query_points_rust: Vec<QueryPoint> = query_points
            .into_iter()
            .map(|qp| QueryPoint::new(qp.frame_idx, qp.point.x, qp.point.y))
            .collect();

        // Create dummy tensor (in real implementation, would use the provided array)
        // For demonstration implementation, we create a simple tensor
        let tensor = candle_core::Tensor::zeros(
            (1, 1, 5, 256, 256),
            candle_core::DType::F32,
            &crate::ev_core::DEVICE,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Track points using demonstration algorithm
        let results = etap::track_points_demo(&tensor, &query_points_rust, num_frames)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Convert to Python types
        let py_results: HashMap<usize, PyTrackResult> =
            results.into_iter().map(|(k, v)| (k, v.into())).collect();

        Ok(py_results)
    }
}
