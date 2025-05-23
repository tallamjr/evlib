// Integration tests for event-to-video reconstruction pipeline
// Tests the complete workflow from events to reconstructed images

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use evlib::ev_core::{Event, Events};
use evlib::ev_processing::reconstruction::{
    e2vid::E2VidConfig, E2Vid, E2VidNet, ModelLoaderConfig,
};

#[test]
fn test_complete_reconstruction_pipeline() {
    // Create synthetic events
    let events = create_synthetic_events(64, 64, 1000);

    // Test simple reconstruction
    let mut e2vid = E2Vid::new(64, 64);
    let simple_result = e2vid
        .process_events_simple(&events)
        .expect("Simple reconstruction failed");

    // Verify simple result
    assert_eq!(simple_result.dims(), &[64, 64]);
    verify_intensity_range(&simple_result);

    // Test neural network reconstruction
    e2vid
        .create_default_network()
        .expect("Failed to create default network");

    let neural_result = e2vid
        .process_events(&events)
        .expect("Neural reconstruction failed");

    // Verify neural result
    assert_eq!(neural_result.dims(), &[64, 64]);
    verify_intensity_range(&neural_result);

    println!("Complete reconstruction pipeline test passed!");
}

#[test]
fn test_reconstruction_with_different_configs() {
    let events = create_synthetic_events(128, 128, 2000);

    let configs = vec![
        E2VidConfig {
            num_bins: 3,
            intensity_scale: 0.5,
            intensity_offset: 0.2,
            ..Default::default()
        },
        E2VidConfig {
            num_bins: 7,
            intensity_scale: 2.0,
            intensity_offset: 0.0,
            ..Default::default()
        },
        E2VidConfig {
            num_bins: 5,
            intensity_scale: 1.0,
            intensity_offset: 0.5,
            ..Default::default()
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        println!(
            "Testing config {}: bins={}, scale={}, offset={}",
            i, config.num_bins, config.intensity_scale, config.intensity_offset
        );

        let mut e2vid = E2Vid::with_config(128, 128, config.clone());

        let result = e2vid
            .process_events_simple(&events)
            .expect(&format!("Reconstruction failed for config {}", i));

        assert_eq!(result.dims(), &[128, 128]);
        verify_intensity_range(&result);

        // Verify that intensity offset is applied
        let result_data = result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        if config.intensity_offset > 0.0 {
            let mean_intensity = result_data.iter().sum::<f32>() / result_data.len() as f32;
            assert!(
                mean_intensity >= config.intensity_offset * 0.5,
                "Intensity offset not properly applied"
            );
        }
    }
}

#[test]
fn test_neural_network_consistency() {
    let events = create_synthetic_events(64, 64, 500);

    // Create two identical reconstructors
    let mut e2vid1 = E2Vid::new(64, 64);
    let mut e2vid2 = E2Vid::new(64, 64);

    // Initialize with the same network architecture
    e2vid1
        .create_default_network()
        .expect("Failed to create network 1");
    e2vid2
        .create_default_network()
        .expect("Failed to create network 2");

    // Process the same events
    let result1 = e2vid1
        .process_events(&events)
        .expect("First reconstruction failed");
    let result2 = e2vid2
        .process_events(&events)
        .expect("Second reconstruction failed");

    // Results should have the same shape
    assert_eq!(result1.dims(), result2.dims());

    // Both should be valid intensity images
    verify_intensity_range(&result1);
    verify_intensity_range(&result2);

    println!("Neural network consistency test passed!");
}

#[test]
fn test_empty_and_edge_cases() {
    // Test with empty events
    let empty_events = Events::new();
    let mut e2vid = E2Vid::new(32, 32);

    let empty_result = e2vid
        .process_events_simple(&empty_events)
        .expect("Empty events reconstruction failed");

    assert_eq!(empty_result.dims(), &[32, 32]);
    verify_intensity_range(&empty_result);

    // Test with single event
    let single_event = create_single_event(16, 16, 0.0, 1);
    let single_result = e2vid
        .process_events_simple(&single_event)
        .expect("Single event reconstruction failed");

    assert_eq!(single_result.dims(), &[32, 32]);
    verify_intensity_range(&single_result);

    // Test with events outside image bounds (should be handled gracefully)
    let out_of_bounds_events = create_out_of_bounds_events(32, 32);
    let oob_result = e2vid
        .process_events_simple(&out_of_bounds_events)
        .expect("Out of bounds events reconstruction failed");

    assert_eq!(oob_result.dims(), &[32, 32]);
    verify_intensity_range(&oob_result);

    println!("Edge cases test passed!");
}

#[test]
fn test_temporal_processing() {
    // Create events with different temporal patterns
    let fast_events = create_temporal_events(64, 64, 1000, 0.0001); // Fast events
    let slow_events = create_temporal_events(64, 64, 1000, 0.001); // Slow events

    let mut e2vid = E2Vid::new(64, 64);

    let fast_result = e2vid
        .process_events_simple(&fast_events)
        .expect("Fast events reconstruction failed");

    let slow_result = e2vid
        .process_events_simple(&slow_events)
        .expect("Slow events reconstruction failed");

    // Both should produce valid results
    assert_eq!(fast_result.dims(), &[64, 64]);
    assert_eq!(slow_result.dims(), &[64, 64]);
    verify_intensity_range(&fast_result);
    verify_intensity_range(&slow_result);

    // Results should be different due to different temporal binning
    let fast_data = fast_result.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let slow_data = slow_result.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let mse = fast_data
        .iter()
        .zip(slow_data.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / fast_data.len() as f32;

    assert!(
        mse > 1e-6,
        "Fast and slow events should produce different results"
    );

    println!("Temporal processing test passed!");
}

#[test]
fn test_model_loading_integration() {
    // Test the model loading infrastructure
    let _config = ModelLoaderConfig::default();

    // Test network creation without loading
    let device = Device::Cpu;
    let dtype = DType::F32;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

    let network = E2VidNet::new(&vs).expect("Failed to create E2VidNet");

    // Test forward pass with dummy input
    let dummy_input = Tensor::randn(0.0f32, 1.0f32, (1, 5, 64, 64), &device)
        .expect("Failed to create dummy input")
        .to_dtype(dtype)
        .expect("Failed to convert dtype");

    let output = network.forward(&dummy_input).expect("Forward pass failed");

    assert_eq!(output.dims(), &[1, 1, 64, 64]);

    // Verify output is in valid range (sigmoid output)
    let output_data = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &output_data {
        assert!(
            val >= 0.0 && val <= 1.0,
            "Output value {} out of range [0,1]",
            val
        );
    }

    println!("Model loading integration test passed!");
}

#[test]
fn test_performance_baseline() {
    use std::time::Instant;

    let events = create_synthetic_events(128, 128, 10000);
    let mut e2vid = E2Vid::new(128, 128);

    // Benchmark simple reconstruction
    let start = Instant::now();
    let _simple_result = e2vid
        .process_events_simple(&events)
        .expect("Simple reconstruction failed");
    let simple_time = start.elapsed();

    // Benchmark neural reconstruction
    e2vid
        .create_default_network()
        .expect("Failed to create network");

    let start = Instant::now();
    let _neural_result = e2vid
        .process_events(&events)
        .expect("Neural reconstruction failed");
    let neural_time = start.elapsed();

    println!("Performance baseline:");
    println!("  Simple reconstruction: {:?}", simple_time);
    println!("  Neural reconstruction: {:?}", neural_time);

    // Basic performance checks (should complete within reasonable time)
    assert!(simple_time.as_secs() < 5, "Simple reconstruction too slow");
    assert!(neural_time.as_secs() < 10, "Neural reconstruction too slow");

    println!("Performance baseline test passed!");
}

// Helper functions

fn create_synthetic_events(width: u16, height: u16, num_events: usize) -> Events {
    let mut events_data = Vec::new();

    for i in 0..num_events {
        let x = (i % width as usize) as u16;
        let y = ((i / width as usize) % height as usize) as u16;
        let t = i as f64 * 0.001;
        let polarity = if i % 2 == 0 { 1 } else { -1 };

        events_data.push(Event { x, y, t, polarity });
    }

    events_data
}

fn create_single_event(x: u16, y: u16, t: f64, polarity: i8) -> Events {
    vec![Event { x, y, t, polarity }]
}

fn create_out_of_bounds_events(width: u16, height: u16) -> Events {
    vec![
        Event {
            x: width + 10,
            y: height + 10,
            t: 0.0,
            polarity: 1,
        }, // Far out of bounds
        Event {
            x: width,
            y: height,
            t: 0.001,
            polarity: -1,
        }, // Just out of bounds
        Event {
            x: width / 2,
            y: height / 2,
            t: 0.002,
            polarity: 1,
        }, // Valid event
    ]
}

fn create_temporal_events(width: u16, height: u16, num_events: usize, dt: f64) -> Events {
    let mut events_data = Vec::new();

    for i in 0..num_events {
        let x = (width / 2) + ((i % 20) as u16) - 10;
        let y = (height / 2) + ((i / 20) as u16) - 10;
        let t = i as f64 * dt;
        let polarity = if i % 3 == 0 { 1 } else { -1 };

        if x < width && y < height {
            events_data.push(Event { x, y, t, polarity });
        }
    }

    events_data
}

fn verify_intensity_range(tensor: &Tensor) {
    let data = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    for &val in &data {
        assert!(val >= 0.0, "Intensity value {} below 0", val);
        assert!(val <= 1.0, "Intensity value {} above 1", val);
        assert!(!val.is_nan(), "NaN intensity value");
        assert!(!val.is_infinite(), "Infinite intensity value");
    }
}
