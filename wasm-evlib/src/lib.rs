use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::console;

// Core Event structure (copied from evlib to avoid dependencies)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Event {
    pub t: f64,       // timestamp in seconds
    pub x: u16,       // x coordinate
    pub y: u16,       // y coordinate
    pub polarity: i8, // +1 or -1
}

// Simplified ESIM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsimConfig {
    pub width: u32,
    pub height: u32,
    pub contrast_threshold_pos: f64,
    pub contrast_threshold_neg: f64,
    pub refractory_period_ms: f64,
}

impl Default for EsimConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            contrast_threshold_pos: 0.15,
            contrast_threshold_neg: 0.15,
            refractory_period_ms: 0.1,
        }
    }
}

// Per-pixel state for ESIM
#[derive(Debug, Clone)]
struct PixelState {
    last_intensity: f32,
    last_event_time: f64,
    intensity_buffer: f64,
}

impl PixelState {
    fn new() -> Self {
        Self {
            last_intensity: 0.5,
            last_event_time: 0.0,
            intensity_buffer: 0.0,
        }
    }
}

// WASM-compatible ESIM implementation
#[wasm_bindgen]
pub struct WasmEsim {
    config: EsimConfig,
    pixel_states: Vec<PixelState>,
    last_timestamp: f64,
    initialized: bool,
}

#[wasm_bindgen]
impl WasmEsim {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        console::log_1(&"Initializing WASM ESIM...".into());

        let config = EsimConfig {
            width,
            height,
            ..Default::default()
        };

        let total_pixels = (width * height) as usize;
        let pixel_states = vec![PixelState::new(); total_pixels];

        console::log_1(&format!("ESIM initialized: {}x{} pixels", width, height).into());

        Self {
            config,
            pixel_states,
            last_timestamp: 0.0,
            initialized: false,
        }
    }

    // Set configuration parameters
    pub fn set_thresholds(&mut self, pos: f64, neg: f64) {
        self.config.contrast_threshold_pos = pos;
        self.config.contrast_threshold_neg = neg;
        console::log_1(&format!("Thresholds set: +{:.2}, -{:.2}", pos, neg).into());
    }

    // Process a frame and generate events
    pub fn process_frame(
        &mut self,
        image_data: &[u8],
        timestamp_ms: f64,
    ) -> Result<Vec<u8>, JsValue> {
        let timestamp_s = timestamp_ms / 1000.0;
        let mut events = Vec::new();

        // Validate image data size (RGBA format)
        let expected_size = (self.config.width * self.config.height * 4) as usize;
        if image_data.len() != expected_size {
            return Err(JsValue::from_str(&format!(
                "Invalid image data size: expected {}, got {}",
                expected_size,
                image_data.len()
            )));
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Process each pixel
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let pixel_idx = idx * 4;

                // Convert RGB to grayscale intensity (0-1)
                let r = image_data[pixel_idx] as f32 / 255.0;
                let g = image_data[pixel_idx + 1] as f32 / 255.0;
                let b = image_data[pixel_idx + 2] as f32 / 255.0;
                let intensity = 0.299 * r + 0.587 * g + 0.114 * b;

                let pixel_state = &mut self.pixel_states[idx];

                // Skip first frame to establish baseline
                if !self.initialized {
                    pixel_state.last_intensity = intensity;
                    continue;
                }

                // Calculate log intensity change (ESIM algorithm)
                let log_intensity = intensity.max(0.001).ln();
                let log_last = pixel_state.last_intensity.max(0.001).ln();
                let log_change = log_intensity - log_last;

                // Accumulate intensity change
                pixel_state.intensity_buffer += log_change as f64;

                // Check refractory period
                let time_since_last = timestamp_s - pixel_state.last_event_time;
                if time_since_last < self.config.refractory_period_ms / 1000.0 {
                    pixel_state.last_intensity = intensity;
                    continue;
                }

                // Generate positive events
                while pixel_state.intensity_buffer >= self.config.contrast_threshold_pos {
                    events.push(Event {
                        t: timestamp_s,
                        x: x as u16,
                        y: y as u16,
                        polarity: 1,
                    });
                    pixel_state.intensity_buffer -= self.config.contrast_threshold_pos;
                    pixel_state.last_event_time = timestamp_s;
                }

                // Generate negative events
                while pixel_state.intensity_buffer <= -self.config.contrast_threshold_neg {
                    events.push(Event {
                        t: timestamp_s,
                        x: x as u16,
                        y: y as u16,
                        polarity: -1,
                    });
                    pixel_state.intensity_buffer += self.config.contrast_threshold_neg;
                    pixel_state.last_event_time = timestamp_s;
                }

                // Update last intensity
                pixel_state.last_intensity = intensity;
            }
        }

        self.initialized = true;
        self.last_timestamp = timestamp_s;

        // Convert events to binary format for efficient transfer
        Ok(events_to_binary(&events))
    }

    // Reset the simulator state
    pub fn reset(&mut self) {
        for state in &mut self.pixel_states {
            *state = PixelState::new();
        }
        self.last_timestamp = 0.0;
        self.initialized = false;
        console::log_1(&"ESIM state reset".into());
    }

    // Get current configuration as JSON
    pub fn get_config(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self.config)?)
    }
}

// Convert events to binary format (same as WebSocket format)
fn events_to_binary(events: &[Event]) -> Vec<u8> {
    // Header: message_type (1) + timestamp (8) + event_count (4) = 13 bytes
    // Events: x (2) + y (2) + timestamp (8) + polarity (1) = 13 bytes per event
    let mut buffer = Vec::with_capacity(13 + events.len() * 13);

    // Header
    buffer.push(1u8); // Message type: events
    if let Some(first_event) = events.first() {
        let timestamp_us = (first_event.t * 1_000_000.0) as u64;
        buffer.extend_from_slice(&timestamp_us.to_le_bytes());
    } else {
        buffer.extend_from_slice(&0u64.to_le_bytes());
    }
    buffer.extend_from_slice(&(events.len() as u32).to_le_bytes());

    // Events
    for event in events {
        buffer.extend_from_slice(&event.x.to_le_bytes());
        buffer.extend_from_slice(&event.y.to_le_bytes());
        let timestamp_us = (event.t * 1_000_000.0) as u64;
        buffer.extend_from_slice(&timestamp_us.to_le_bytes());
        buffer.push(if event.polarity > 0 { 1u8 } else { 0u8 });
    }

    buffer
}

// Initialize logging
#[wasm_bindgen(start)]
pub fn main() {
    console::log_1(&"WASM ESIM module loaded successfully!".into());
}
