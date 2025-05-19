// Visualization module
// Tools for converting events into visualizations and images

use crate::ev_core::Events;
use candle_core::{Result as CandleResult, Tensor};
use image::{Rgb, RgbImage};

/// Render events to an RGB image, coloring by polarity
///
/// Positive polarity = red, Negative polarity = blue
///
/// # Arguments
/// * `events` - Events to visualize
/// * `resolution` - Image dimensions (width, height)
/// * `color_mode` - Method for coloring events: "polarity", "time", "polarity_time"
pub fn draw_events_to_image(events: &Events, resolution: (u16, u16), color_mode: &str) -> RgbImage {
    let (width, height) = (resolution.0 as u32, resolution.1 as u32);

    // Create a black background image
    let mut img = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));

    if events.is_empty() {
        return img;
    }

    // Get time range for normalization
    let t_min = events
        .iter()
        .map(|e| e.t)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    let t_max = events
        .iter()
        .map(|e| e.t)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(1.0);
    let t_range = t_max - t_min;

    match color_mode {
        "time" => {
            // Color by time: recent events are brighter
            for e in events {
                if e.x as u32 >= width || e.y as u32 >= height {
                    continue; // Skip events outside image bounds
                }

                // Normalize time to [0, 255]
                let t_norm = if t_range > 0.0 {
                    (((e.t - t_min) / t_range) * 255.0) as u8
                } else {
                    255
                };

                let color = Rgb([t_norm, t_norm, t_norm]);
                img.put_pixel(e.x as u32, e.y as u32, color);
            }
        }
        "polarity_time" => {
            // Color by polarity and time: positive=red, negative=blue, brightness by time
            for e in events {
                if e.x as u32 >= width || e.y as u32 >= height {
                    continue; // Skip events outside image bounds
                }

                // Normalize time to [50, 255] to avoid too dark pixels
                let t_norm = if t_range > 0.0 {
                    50 + (((e.t - t_min) / t_range) * 205.0) as u8
                } else {
                    255
                };

                let color = if e.polarity > 0 {
                    Rgb([t_norm, 0, 0]) // Red for positive, brightness by time
                } else {
                    Rgb([0, 0, t_norm]) // Blue for negative, brightness by time
                };

                img.put_pixel(e.x as u32, e.y as u32, color);
            }
        }
        _ => {
            // "polarity" (default)
            // Color by polarity: positive=red, negative=blue
            for e in events {
                if e.x as u32 >= width || e.y as u32 >= height {
                    continue; // Skip events outside image bounds
                }

                let color = if e.polarity > 0 {
                    Rgb([255, 0, 0]) // Red for positive
                } else {
                    Rgb([0, 0, 255]) // Blue for negative
                };

                img.put_pixel(e.x as u32, e.y as u32, color);
            }
        }
    }

    img
}

/// Overlay events on an existing image (e.g., grayscale frame)
///
/// # Arguments
/// * `base_frame` - Base image to overlay events on
/// * `events` - Events to visualize
/// * `alpha` - Opacity of event overlay (0.0-1.0)
/// * `color_positive` - RGB color for positive events (default: red)
/// * `color_negative` - RGB color for negative events (default: blue)
pub fn overlay_events_on_frame(
    base_frame: &RgbImage,
    events: &Events,
    alpha: f32,
    color_positive: Option<[u8; 3]>,
    color_negative: Option<[u8; 3]>,
) -> RgbImage {
    let (width, height) = (base_frame.width(), base_frame.height());

    // Create a copy of the base frame
    let mut output = base_frame.clone();

    // Default colors
    let pos_color = color_positive.unwrap_or([255, 0, 0]); // Red
    let neg_color = color_negative.unwrap_or([0, 0, 255]); // Blue

    // Alpha blending lambda
    let blend = |base: &[u8; 3], overlay: &[u8; 3], alpha: f32| -> [u8; 3] {
        [
            ((1.0 - alpha) * base[0] as f32 + alpha * overlay[0] as f32) as u8,
            ((1.0 - alpha) * base[1] as f32 + alpha * overlay[1] as f32) as u8,
            ((1.0 - alpha) * base[2] as f32 + alpha * overlay[2] as f32) as u8,
        ]
    };

    // Draw events
    for e in events {
        if e.x as u32 >= width || e.y as u32 >= height {
            continue; // Skip events outside image bounds
        }

        let pixel = output.get_pixel_mut(e.x as u32, e.y as u32);
        let base_color = [pixel[0], pixel[1], pixel[2]];

        // Blend the event color with the existing pixel
        let new_color = if e.polarity > 0 {
            blend(&base_color, &pos_color, alpha)
        } else {
            blend(&base_color, &neg_color, alpha)
        };

        *pixel = Rgb(new_color);
    }

    output
}

/// Convert a tensor to an RGB image
///
/// # Arguments
/// * `tensor` - 2D or 3D tensor to convert to image
/// * `colormap` - Optional colormap to use: "gray", "jet", "viridis", "plasma"
pub fn tensor_to_image(tensor: &Tensor, colormap: Option<&str>) -> CandleResult<RgbImage> {
    let shape = tensor.shape();

    // Get tensor data as flattened vector
    let data = tensor.to_vec1()?;

    // If tensor is already normalized to [0,1], use as is
    // Otherwise, normalize it
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let normalized = if min_val >= 0.0 && max_val <= 1.0 && min_val != max_val {
        data
    } else if min_val != max_val {
        // Normalize to [0,1]
        data.iter()
            .map(|&x| (x - min_val) / (max_val - min_val))
            .collect()
    } else {
        // Constant tensor
        vec![0.0; data.len()]
    };

    let dims = shape.dims();
    if dims.len() == 2 || (dims.len() == 3 && dims[0] == 1) {
        // 2D tensor or 3D with single channel
        let height = dims[dims.len() - 2] as u32;
        let width = dims[dims.len() - 1] as u32;

        // Apply colormap
        let colormap_fn = match colormap.unwrap_or("gray") {
            "gray" => |v: f32| {
                let intensity = (v * 255.0) as u8;
                Rgb([intensity, intensity, intensity])
            },
            "jet" => |v: f32| {
                // Simplified jet colormap
                let r = (1.5 - f32::abs(2.0 * v - 1.0)).clamp(0.0, 1.0) * 255.0;
                let g = (1.5 - f32::abs(2.0 * v - 0.5)).clamp(0.0, 1.0) * 255.0;
                let b = (1.5 - f32::abs(2.0 * v - 0.0)).clamp(0.0, 1.0) * 255.0;
                Rgb([r as u8, g as u8, b as u8])
            },
            // Add other colormaps here
            _ => |v: f32| {
                let intensity = (v * 255.0) as u8;
                Rgb([intensity, intensity, intensity])
            },
        };

        // Create image
        let mut img = RgbImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                if idx < normalized.len() {
                    img.put_pixel(x, y, colormap_fn(normalized[idx]));
                }
            }
        }

        Ok(img)
    } else if dims.len() == 3 && dims[0] == 3 {
        // RGB tensor
        let height = dims[1] as u32;
        let width = dims[2] as u32;
        let mut img = RgbImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let r_idx = (y * width + x) as usize;
                let g_idx = (height * width + y * width + x) as usize;
                let b_idx = (2 * height * width + y * width + x) as usize;

                if r_idx < normalized.len() && g_idx < normalized.len() && b_idx < normalized.len()
                {
                    let r = (normalized[r_idx] * 255.0) as u8;
                    let g = (normalized[g_idx] * 255.0) as u8;
                    let b = (normalized[b_idx] * 255.0) as u8;
                    img.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
        }

        Ok(img)
    } else {
        // Handle other tensor shapes
        Err(candle_core::Error::Msg(format!(
            "Unsupported tensor shape for visualization: {:?}",
            shape
        )))
    }
}

/// Generate a visualization of the temporal distribution of events
///
/// Creates a histogram showing the number of events per time bin
///
/// # Arguments
/// * `events` - Events to visualize
/// * `num_bins` - Number of time bins
pub fn visualize_temporal_histogram(events: &Events, num_bins: usize) -> CandleResult<RgbImage> {
    if events.is_empty() {
        return Ok(RgbImage::new(num_bins as u32, 100));
    }

    // Determine time range
    let t_min = events
        .iter()
        .map(|e| e.t)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let t_max = events
        .iter()
        .map(|e| e.t)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let t_range = t_max - t_min;

    // Create histogram
    let mut histogram_pos = vec![0u32; num_bins];
    let mut histogram_neg = vec![0u32; num_bins];

    for e in events {
        let bin = if t_range > 0.0 {
            (((e.t - t_min) / t_range) * (num_bins as f64 - 1.0)) as usize
        } else {
            0
        };

        if bin < num_bins {
            if e.polarity > 0 {
                histogram_pos[bin] += 1;
            } else {
                histogram_neg[bin] += 1;
            }
        }
    }

    // Find maximum count for scaling
    let max_count = histogram_pos
        .iter()
        .chain(histogram_neg.iter())
        .fold(0, |acc, &count| acc.max(count));

    // Create image (100 pixels tall)
    let height = 100u32;
    let width = num_bins as u32;
    let mut img = RgbImage::from_pixel(width, height, Rgb([255, 255, 255]));

    // Draw histogram
    for (bin, (pos_count, neg_count)) in histogram_pos.iter().zip(histogram_neg.iter()).enumerate()
    {
        let bin_x = bin as u32;

        // Scale counts to fit in image height
        let pos_height = if max_count > 0 {
            ((*pos_count as f64 / max_count as f64) * (height as f64 / 2.0)) as u32
        } else {
            0
        };

        let neg_height = if max_count > 0 {
            ((*neg_count as f64 / max_count as f64) * (height as f64 / 2.0)) as u32
        } else {
            0
        };

        // Draw positive events (above middle, in red)
        for y in 0..pos_height {
            if height / 2 - y > 0 {
                img.put_pixel(bin_x, height / 2 - y - 1, Rgb([255, 0, 0]));
            }
        }

        // Draw negative events (below middle, in blue)
        for y in 0..neg_height {
            if height / 2 + y < height {
                img.put_pixel(bin_x, height / 2 + y, Rgb([0, 0, 255]));
            }
        }

        // Draw middle line
        img.put_pixel(bin_x, height / 2, Rgb([0, 0, 0]));
    }

    Ok(img)
}

/// Visualize the flow field estimated from events
///
/// # Arguments
/// * `resolution` - Image dimensions (width, height)
/// * `flow` - Array of flow vectors (x, y) for each grid cell
/// * `grid_size` - Size of grid cells (e.g., 16 means 16x16 pixel cells)
pub fn visualize_flow_field(
    resolution: (u16, u16),
    flow: &[(f32, f32)], // vx, vy for each grid cell
    grid_size: u32,
) -> RgbImage {
    let (width, height) = (resolution.0 as u32, resolution.1 as u32);
    let grid_cols = width.div_ceil(grid_size);
    let grid_rows = height.div_ceil(grid_size);

    // Create a white background image
    let mut img = RgbImage::from_pixel(width, height, Rgb([255, 255, 255]));

    // Calculate number of expected flow vectors
    let expected_flow_count = (grid_cols * grid_rows) as usize;

    // Skip if flow array doesn't match expected size
    if flow.len() != expected_flow_count {
        return img;
    }

    // Find maximum flow magnitude for normalization
    let max_magnitude = flow
        .iter()
        .map(|(vx, vy)| (vx.powi(2) + vy.powi(2)).sqrt())
        .fold(0.0f32, |acc, mag| acc.max(mag));

    // Draw flow vectors
    for row in 0..grid_rows {
        for col in 0..grid_cols {
            let grid_idx = (row * grid_cols + col) as usize;
            if grid_idx >= flow.len() {
                continue;
            }

            let (vx, vy) = flow[grid_idx];

            // Skip zero flow
            if vx.abs() < 1e-5 && vy.abs() < 1e-5 {
                continue;
            }

            // Calculate center of grid cell
            let center_x = col * grid_size + grid_size / 2;
            let center_y = row * grid_size + grid_size / 2;

            // Skip if center is outside image
            if center_x >= width || center_y >= height {
                continue;
            }

            // Normalize flow to fit in grid cell
            let magnitude = (vx.powi(2) + vy.powi(2)).sqrt();
            let scale = if max_magnitude > 0.0 {
                (grid_size as f32 / 2.5) * (magnitude / max_magnitude)
            } else {
                0.0
            };

            let end_x =
                (center_x as f32 + vx / magnitude * scale).clamp(0.0, width as f32 - 1.0) as u32;
            let end_y =
                (center_y as f32 + vy / magnitude * scale).clamp(0.0, height as f32 - 1.0) as u32;

            // Draw flow vector
            draw_line(&mut img, center_x, center_y, end_x, end_y, Rgb([0, 0, 255]));

            // Draw arrowhead
            draw_arrowhead(&mut img, center_x, center_y, end_x, end_y, Rgb([0, 0, 255]));
        }
    }

    img
}

/// Helper function to draw a line on an image
fn draw_line(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, color: Rgb<u8>) {
    // Bresenham's line algorithm
    let dx = if x0 > x1 { x0 - x1 } else { x1 - x0 };
    let dy = if y0 > y1 { y0 - y1 } else { y1 - y0 };

    // Convert to i32 for signed operations
    let dx_i32 = dx as i32;
    let dy_i32 = dy as i32;

    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    let mut err = if dx > dy { dx_i32 } else { -dy_i32 } / 2;
    let mut err2;

    let mut x = x0 as i32;
    let mut y = y0 as i32;

    let width = img.width() as i32;
    let height = img.height() as i32;

    loop {
        if x >= 0 && x < width && y >= 0 && y < height {
            img.put_pixel(x as u32, y as u32, color);
        }

        if x == x1 as i32 && y == y1 as i32 {
            break;
        }

        err2 = err;

        if err2 > -dx_i32 {
            err -= dy_i32;
            x += sx;
        }

        if err2 < dy_i32 {
            err += dx_i32;
            y += sy;
        }
    }
}

/// Helper function to draw an arrowhead
fn draw_arrowhead(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, color: Rgb<u8>) {
    let angle = (y1 as f32 - y0 as f32).atan2(x1 as f32 - x0 as f32);
    let length = 5.0; // Length of arrow head

    let angle1 = angle + std::f32::consts::PI * 3.0 / 4.0;
    let angle2 = angle - std::f32::consts::PI * 3.0 / 4.0;

    let x2 = (x1 as f32 + angle1.cos() * length).round() as u32;
    let y2 = (y1 as f32 + angle1.sin() * length).round() as u32;

    let x3 = (x1 as f32 + angle2.cos() * length).round() as u32;
    let y3 = (y1 as f32 + angle2.sin() * length).round() as u32;

    draw_line(img, x1, y1, x2, y2, color);
    draw_line(img, x1, y1, x3, y3, color);
}

/// Python bindings for the visualization module
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::ev_core::from_numpy_arrays;
    use numpy::{IntoPyArray, PyReadonlyArray1};
    use pyo3::prelude::*;

    /// Convert events to an RGB image for visualization
    #[pyfunction]
    #[pyo3(name = "draw_events_to_image")]
    pub fn draw_events_to_image_py(
        py: Python<'_>,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        resolution: Option<(i64, i64)>,
        color_mode: Option<&str>,
    ) -> PyResult<PyObject> {
        // Convert to our internal types
        let events = from_numpy_arrays(xs, ys, ts, ps);

        // Determine resolution
        let res = match resolution {
            Some((w, h)) => (w as u16, h as u16),
            None => {
                let max_x = events.iter().map(|e| e.x).max().unwrap_or(0) + 1;
                let max_y = events.iter().map(|e| e.y).max().unwrap_or(0) + 1;
                (max_x, max_y)
            }
        };

        // Draw events to image
        let img = draw_events_to_image(&events, res, color_mode.unwrap_or("polarity"));

        // Convert to numpy array
        let (width, height) = (img.width() as usize, img.height() as usize);
        let mut array = numpy::ndarray::Array3::<u8>::zeros((height, width, 3));

        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x as u32, y as u32);
                array[[y, x, 0]] = pixel[0];
                array[[y, x, 1]] = pixel[1];
                array[[y, x, 2]] = pixel[2];
            }
        }

        Ok(array.into_pyarray(py).to_object(py))
    }
}
