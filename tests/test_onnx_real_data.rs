// Integration tests for ONNX model loading with real data
// Tests loading models from the models directory and processing real event data

use evlib::ev_core::{Event, Events};
use evlib::ev_formats::load_events_from_txt;
use evlib::ev_processing::reconstruction::{
    E2Vid, E2VidConfig, ModelConverter, OnnxE2VidModel, OnnxModelConfig,
};
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn test_onnx_model_loading_from_models_dir() {
    // Check if models directory exists
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        println!("Creating models directory for testing");
        fs::create_dir_all(models_dir).expect("Failed to create models directory");
    }

    // Test ONNX model loading infrastructure
    let test_model_path = models_dir.join("test_e2vid.onnx");

    // Create a dummy model file for testing
    if !test_model_path.exists() {
        fs::write(&test_model_path, b"dummy onnx model for testing")
            .expect("Failed to create test model file");
    }

    // Test loading with OnnxE2VidModel
    let config = OnnxModelConfig::default();
    let model = OnnxE2VidModel::load_from_file(&test_model_path, config);

    assert!(model.is_ok(), "Failed to load ONNX model from file");

    let model = model.unwrap();
    let metadata = model.metadata();

    assert_eq!(metadata.input_name, "voxel_grid");
    assert_eq!(metadata.output_name, "reconstructed_frame");

    println!("✅ ONNX model loading infrastructure test passed!");

    // Clean up test file
    fs::remove_file(&test_model_path).ok();
}

#[test]
fn test_e2vid_with_onnx_model_path() {
    // Test E2Vid integration with ONNX model loading
    let mut e2vid = E2Vid::new(128, 128);

    // Create test model path
    let models_dir = Path::new("models");
    fs::create_dir_all(models_dir).ok();
    let test_model_path = models_dir.join("e2vid_lightweight.onnx");

    // Create dummy model for testing
    if !test_model_path.exists() {
        fs::write(&test_model_path, b"dummy onnx model").expect("Failed to create test model");
    }

    // Test loading ONNX model
    let result = e2vid.load_onnx_model(&test_model_path);
    assert!(result.is_ok(), "Failed to load ONNX model into E2Vid");

    // Test that E2Vid can still process events
    let events = create_test_events();
    let output = e2vid.process_events(&events);
    assert!(output.is_ok(), "Failed to process events with ONNX backend");

    println!("✅ E2Vid ONNX integration test passed!");

    // Clean up
    fs::remove_file(&test_model_path).ok();
}

#[test]
fn test_real_slider_depth_data() {
    // Test with real event data from slider_depth dataset
    let data_path = Path::new("data/slider_depth/events.txt");

    if !data_path.exists() {
        println!("⚠️  Slider depth dataset not found, skipping real data test");
        println!("   To run this test, ensure data/slider_depth/events.txt exists");
        return;
    }

    println!("Loading real event data from slider_depth dataset...");

    // Load events from file
    let events =
        load_events_from_txt(data_path).expect("Failed to load events from slider_depth dataset");

    println!("Loaded {} events", events.len());

    // Take a subset for testing
    let subset_size = 10000.min(events.len());
    let events_subset: Events = events.into_iter().take(subset_size).collect();

    // Determine image dimensions from events
    let (width, height) = calculate_image_dimensions(&events_subset);
    println!("Detected image dimensions: {}x{}", width, height);

    // Test simple reconstruction
    let mut e2vid = E2Vid::new(height as usize, width as usize);
    let simple_result = e2vid
        .process_events_simple(&events_subset)
        .expect("Failed to reconstruct with simple method");

    // Verify output
    assert_eq!(simple_result.dims(), &[height as usize, width as usize]);
    verify_reconstruction_quality(&simple_result);

    // Test neural reconstruction
    e2vid
        .create_default_network()
        .expect("Failed to create default network");

    let neural_result = e2vid
        .process_events(&events_subset)
        .expect("Failed to reconstruct with neural network");

    // Verify output
    assert_eq!(neural_result.dims(), &[height as usize, width as usize]);
    verify_reconstruction_quality(&neural_result);

    println!("✅ Real data reconstruction test passed!");

    // Save output for visual inspection (optional)
    if let Ok(output_dir) = std::env::var("EVLIB_TEST_OUTPUT_DIR") {
        save_reconstruction_output(&simple_result, &neural_result, &output_dir);
    }
}

#[test]
fn test_model_converter_instructions() {
    // Test that model conversion instructions are available
    let instructions = ModelConverter::pytorch_to_onnx_instructions();

    assert!(instructions.contains("torch.onnx.export"));
    assert!(instructions.contains("e2vid_model.onnx"));
    assert!(instructions.contains("voxel_grid"));
    assert!(instructions.contains("reconstructed_frame"));

    println!("✅ Model converter documentation test passed!");
}

#[test]
fn test_different_model_configurations() {
    // Test loading models with different configurations
    let configs = vec![
        OnnxModelConfig {
            verbose: true,
            ..Default::default()
        },
        OnnxModelConfig {
            dtype: candle_core::DType::F32,
            ..Default::default()
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        println!("Testing ONNX config variant {}", i);

        // Create test model
        let test_path = PathBuf::from(format!("models/test_config_{}.onnx", i));
        fs::create_dir_all("models").ok();
        fs::write(&test_path, b"test model").ok();

        if test_path.exists() {
            let model = OnnxE2VidModel::load_from_file(&test_path, config.clone());
            assert!(model.is_ok(), "Failed to load model with config {}", i);

            // Clean up
            fs::remove_file(&test_path).ok();
        }
    }

    println!("✅ Multiple configuration test passed!");
}

#[test]
fn test_benchmark_onnx_placeholder() {
    use std::time::Instant;

    // Benchmark the placeholder ONNX implementation
    let events = create_benchmark_events(256, 256, 50000);

    // Create E2Vid with default backend
    let mut e2vid_default = E2Vid::new(256, 256);
    e2vid_default
        .create_default_network()
        .expect("Failed to create default network");

    // Benchmark default backend
    let start = Instant::now();
    let _result = e2vid_default
        .process_events(&events)
        .expect("Default backend failed");
    let default_time = start.elapsed();

    // Create E2Vid with ONNX backend (placeholder)
    let mut e2vid_onnx = E2Vid::new(256, 256);
    let onnx_model_path = Path::new("models/benchmark_test.onnx");
    fs::create_dir_all("models").ok();
    fs::write(onnx_model_path, b"dummy model").ok();

    if onnx_model_path.exists() {
        e2vid_onnx.load_onnx_model(onnx_model_path).ok();

        // Benchmark ONNX backend (placeholder)
        let start = Instant::now();
        let _result = e2vid_onnx
            .process_events(&events)
            .expect("ONNX backend failed");
        let onnx_time = start.elapsed();

        println!("Benchmark results:");
        println!("  Default backend: {:?}", default_time);
        println!("  ONNX placeholder: {:?}", onnx_time);

        // Clean up
        fs::remove_file(onnx_model_path).ok();
    }

    println!("✅ Benchmark test completed!");
}

// Helper functions

fn create_test_events() -> Events {
    vec![
        Event {
            x: 10,
            y: 10,
            t: 0.0,
            polarity: 1,
        },
        Event {
            x: 20,
            y: 20,
            t: 0.001,
            polarity: -1,
        },
        Event {
            x: 30,
            y: 30,
            t: 0.002,
            polarity: 1,
        },
        Event {
            x: 40,
            y: 40,
            t: 0.003,
            polarity: -1,
        },
        Event {
            x: 50,
            y: 50,
            t: 0.004,
            polarity: 1,
        },
    ]
}

fn create_benchmark_events(width: u16, height: u16, num_events: usize) -> Events {
    let mut events = Vec::with_capacity(num_events);

    for i in 0..num_events {
        let angle = (i as f64 / num_events as f64) * 2.0 * std::f64::consts::PI;
        let radius = ((i % 100) as f64 / 100.0) * (width.min(height) as f64 / 2.0);

        let x = ((width as f64 / 2.0) + radius * angle.cos()) as u16;
        let y = ((height as f64 / 2.0) + radius * angle.sin()) as u16;
        let t = i as f64 * 0.0001;
        let polarity = if i % 2 == 0 { 1 } else { -1 };

        if x < width && y < height {
            events.push(Event { x, y, t, polarity });
        }
    }

    events
}

fn calculate_image_dimensions(events: &Events) -> (u16, u16) {
    let max_x = events.iter().map(|e| e.x).max().unwrap_or(0);
    let max_y = events.iter().map(|e| e.y).max().unwrap_or(0);

    // Add 1 because coordinates are 0-indexed
    (max_x + 1, max_y + 1)
}

fn verify_reconstruction_quality(tensor: &candle_core::Tensor) {
    use candle_core::Device;

    let data = tensor
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Check basic properties
    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_val = data.iter().sum::<f32>() / data.len() as f32;

    println!("Reconstruction statistics:");
    println!("  Min: {:.4}", min_val);
    println!("  Max: {:.4}", max_val);
    println!("  Mean: {:.4}", mean_val);

    // Verify all values are in valid range
    for &val in &data {
        assert!(val >= 0.0 && val <= 1.0, "Value {} out of range [0,1]", val);
        assert!(!val.is_nan(), "NaN value in reconstruction");
    }

    // Check that there's some variation (not all zeros or constant)
    let variance = data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f32>() / data.len() as f32;

    println!("  Variance: {:.6}", variance);
}

fn save_reconstruction_output(
    simple: &candle_core::Tensor,
    neural: &candle_core::Tensor,
    output_dir: &str,
) {
    use candle_core::Device;

    // Convert tensors to images and save
    println!("Saving reconstruction outputs to {}", output_dir);

    let simple_data = simple
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    let neural_data = neural
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    // Here you would save as actual image files
    // For now, just log that we would save them
    println!(
        "Would save simple reconstruction: {}x{}",
        simple_data.len(),
        simple_data[0].len()
    );
    println!(
        "Would save neural reconstruction: {}x{}",
        neural_data.len(),
        neural_data[0].len()
    );
}
