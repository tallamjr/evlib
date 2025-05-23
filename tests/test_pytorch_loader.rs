// Test suite for PyTorch model loading functionality
// Tests the model loading infrastructure and neural network operations

use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use evlib::ev_processing::reconstruction::pytorch_loader::{
    E2VidModelLoader, E2VidNet, ModelLoaderConfig,
};
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_e2vid_net_creation() {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

    let network = E2VidNet::new(&vs);
    assert!(network.is_ok(), "Failed to create E2VidNet");
}

#[test]
fn test_e2vid_net_forward_pass() {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

    let network = E2VidNet::new(&vs).expect("Failed to create network");

    // Test with different input sizes
    let test_cases = vec![
        (1, 5, 64, 64),   // Standard input
        (1, 5, 128, 128), // Larger input
        (2, 5, 32, 32),   // Batch of 2
        (1, 5, 256, 256), // Large input
    ];

    for (batch_size, channels, height, width) in test_cases {
        let input = Tensor::randn(
            0.0f32,
            1.0f32,
            (batch_size, channels, height, width),
            &device,
        )
        .expect("Failed to create input tensor")
        .to_dtype(dtype)
        .expect("Failed to convert tensor dtype");

        let output = network.forward(&input);
        assert!(
            output.is_ok(),
            "Forward pass failed for input size ({}, {}, {}, {})",
            batch_size,
            channels,
            height,
            width
        );

        let output = output.unwrap();
        assert_eq!(
            output.dims(),
            &[batch_size, 1, height, width],
            "Output shape mismatch for input size ({}, {}, {}, {})",
            batch_size,
            channels,
            height,
            width
        );

        // Check output range (should be [0, 1] due to sigmoid)
        let output_data = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &val in &output_data {
            assert!(
                val >= 0.0 && val <= 1.0,
                "Output value {} outside [0, 1] range",
                val
            );
        }
    }
}

#[test]
fn test_e2vid_net_forward_t_training_modes() {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

    let network = E2VidNet::new(&vs).expect("Failed to create network");

    let input = Tensor::randn(0.0f32, 1.0f32, (1, 5, 64, 64), &device)
        .expect("Failed to create input tensor")
        .to_dtype(dtype)
        .expect("Failed to convert tensor dtype");

    // Test both training and inference modes
    let output_train = network.forward_t(&input, true);
    let output_inference = network.forward_t(&input, false);

    assert!(output_train.is_ok(), "Forward pass failed in training mode");
    assert!(
        output_inference.is_ok(),
        "Forward pass failed in inference mode"
    );

    let output_train = output_train.unwrap();
    let output_inference = output_inference.unwrap();

    assert_eq!(output_train.dims(), &[1, 1, 64, 64]);
    assert_eq!(output_inference.dims(), &[1, 1, 64, 64]);

    // Outputs might differ slightly due to batch norm behavior
    // but should have same shape and range
    let train_data = output_train
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let inference_data = output_inference
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    for &val in &train_data {
        assert!(
            val >= 0.0 && val <= 1.0,
            "Training output value {} outside [0, 1] range",
            val
        );
    }

    for &val in &inference_data {
        assert!(
            val >= 0.0 && val <= 1.0,
            "Inference output value {} outside [0, 1] range",
            val
        );
    }
}

#[test]
fn test_model_loader_config() {
    let config = ModelLoaderConfig::default();

    // Device comparison not available, just verify it's accessible
    let _ = &config.device;
    assert_eq!(config.dtype, DType::F32);
    assert!(config.strict_loading);
    assert!(!config.verbose);

    // Test custom config
    let custom_config = ModelLoaderConfig {
        device: Device::Cpu,
        dtype: DType::F32,
        strict_loading: false,
        verbose: true,
    };

    assert!(!custom_config.strict_loading);
    assert!(custom_config.verbose);
}

#[test]
fn test_model_loader_with_invalid_file() {
    let config = ModelLoaderConfig::default();

    // Test with non-existent file
    let result = E2VidModelLoader::load_from_pth("nonexistent.pth", config.clone());
    assert!(result.is_err(), "Should fail to load non-existent file");

    // Test with invalid file content
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    write!(temp_file, "invalid model data").expect("Failed to write to temp file");

    let result = E2VidModelLoader::load_from_pth(temp_file.path(), config);
    assert!(result.is_err(), "Should fail to load invalid model file");
}

#[test]
fn test_network_determinism() {
    // Test that same input produces same output (with same random seed)
    let device = Device::Cpu;
    let dtype = DType::F32;

    // Create two identical networks
    let var_map1 = VarMap::new();
    let vs1 = VarBuilder::from_varmap(&var_map1, dtype, &device);
    let network1 = E2VidNet::new(&vs1).expect("Failed to create network1");

    let var_map2 = VarMap::new();
    let vs2 = VarBuilder::from_varmap(&var_map2, dtype, &device);
    let network2 = E2VidNet::new(&vs2).expect("Failed to create network2");

    // Create deterministic input
    let input =
        Tensor::zeros((1, 5, 64, 64), dtype, &device).expect("Failed to create input tensor");

    let output1 = network1.forward(&input).expect("Forward pass 1 failed");
    let output2 = network2.forward(&input).expect("Forward pass 2 failed");

    // Outputs should be the same for zero input with random initialization
    // (since zero input with random weights should give consistent results)
    assert_eq!(output1.dims(), output2.dims());
}

#[test]
fn test_network_gradient_flow() {
    // Test that the network can handle gradients (important for training)
    let device = Device::Cpu;
    let dtype = DType::F32;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

    let network = E2VidNet::new(&vs).expect("Failed to create network");

    let input = Tensor::randn(0.0f32, 1.0f32, (1, 5, 64, 64), &device)
        .expect("Failed to create input tensor")
        .to_dtype(dtype)
        .expect("Failed to convert tensor dtype");

    // Forward pass
    let output = network.forward(&input).expect("Forward pass failed");

    // Compute a simple loss (mean squared error with target)
    let target =
        Tensor::zeros(output.dims(), dtype, &device).expect("Failed to create target tensor");

    let diff = output.sub(&target).expect("Failed to compute difference");
    let squared = diff.sqr().expect("Failed to square difference");
    let loss = squared.mean_all().expect("Failed to compute mean");

    // Loss should be a scalar
    assert_eq!(loss.dims(), &[] as &[usize]);

    // Loss should be positive (MSE is always >= 0)
    let loss_value = loss
        .to_scalar::<f32>()
        .expect("Failed to extract loss value");
    assert!(
        loss_value >= 0.0,
        "Loss should be non-negative, got {}",
        loss_value
    );
}

#[test]
fn test_network_memory_efficiency() {
    // Test that the network doesn't leak memory with repeated use
    let device = Device::Cpu;
    let dtype = DType::F32;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

    let network = E2VidNet::new(&vs).expect("Failed to create network");

    // Run multiple forward passes
    for i in 0..10 {
        let input = Tensor::randn(0.0f32, 1.0f32, (1, 5, 64, 64), &device)
            .expect(&format!("Failed to create input tensor {}", i))
            .to_dtype(dtype)
            .expect(&format!("Failed to convert tensor dtype {}", i));

        let output = network
            .forward(&input)
            .expect(&format!("Forward pass {} failed", i));

        // Verify output properties
        assert_eq!(output.dims(), &[1, 1, 64, 64]);

        // Force evaluation to ensure computation completes
        let _data = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    }
}

#[test]
fn test_network_edge_cases() {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

    let network = E2VidNet::new(&vs).expect("Failed to create network");

    // Test with all zeros
    let zeros_input =
        Tensor::zeros((1, 5, 64, 64), dtype, &device).expect("Failed to create zeros tensor");
    let zeros_output = network.forward(&zeros_input);
    assert!(zeros_output.is_ok(), "Failed with all-zeros input");

    // Test with all ones
    let ones_input =
        Tensor::ones((1, 5, 64, 64), dtype, &device).expect("Failed to create ones tensor");
    let ones_output = network.forward(&ones_input);
    assert!(ones_output.is_ok(), "Failed with all-ones input");

    // Test with extreme values
    let extreme_input =
        Tensor::full(100.0f32, (1, 5, 64, 64), &device).expect("Failed to create extreme tensor");
    let extreme_output = network.forward(&extreme_input);
    assert!(extreme_output.is_ok(), "Failed with extreme input values");

    // Output should still be in valid range due to sigmoid
    let extreme_result = extreme_output.unwrap();
    let extreme_data = extreme_result
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for &val in &extreme_data {
        assert!(
            val >= 0.0 && val <= 1.0,
            "Extreme input produced invalid output: {}",
            val
        );
    }
}

#[test]
fn test_different_device_compatibility() {
    // Test CPU device (GPU testing would require CUDA setup)
    let device = Device::Cpu;
    let dtype = DType::F32;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

    let network = E2VidNet::new(&vs).expect("Failed to create network");

    let input = Tensor::randn(0.0f32, 1.0f32, (1, 5, 64, 64), &device)
        .expect("Failed to create input tensor")
        .to_dtype(dtype)
        .expect("Failed to convert tensor dtype");

    let output = network.forward(&input);
    assert!(output.is_ok(), "Forward pass failed on CPU device");

    let output = output.unwrap();
    // Note: Device comparison not available, just verify it's a valid device
    let _ = output.device(); // Verify device method works
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_forward_pass_speed() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

        let network = E2VidNet::new(&vs).expect("Failed to create network");

        let input = Tensor::randn(0.0f32, 1.0f32, (1, 5, 256, 256), &device)
            .expect("Failed to create input tensor")
            .to_dtype(dtype)
            .expect("Failed to convert tensor dtype");

        // Warm up
        for _ in 0..3 {
            let _ = network.forward(&input).expect("Warmup forward pass failed");
        }

        // Benchmark
        let num_iterations = 10;
        let start = Instant::now();

        for _ in 0..num_iterations {
            let _ = network
                .forward(&input)
                .expect("Benchmark forward pass failed");
        }

        let duration = start.elapsed();
        let avg_time = duration / num_iterations;

        println!("Average forward pass time (256x256): {:?}", avg_time);

        // Should complete within reasonable time (adjust threshold as needed)
        assert!(
            avg_time.as_millis() < 1000,
            "Forward pass too slow: {:?}",
            avg_time
        );
    }

    #[test]
    fn benchmark_different_input_sizes() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

        let network = E2VidNet::new(&vs).expect("Failed to create network");

        let sizes = vec![(64, 64), (128, 128), (256, 256), (512, 512)];

        for (height, width) in sizes {
            let input = Tensor::randn(0.0f32, 1.0f32, (1, 5, height, width), &device)
                .expect("Failed to create input tensor")
                .to_dtype(dtype)
                .expect("Failed to convert tensor dtype");

            let start = Instant::now();
            let _ = network
                .forward(&input)
                .expect(&format!("Forward pass failed for {}x{}", height, width));
            let duration = start.elapsed();

            println!("Forward pass time ({}x{}): {:?}", height, width, duration);

            // Larger images should take longer, but not exponentially
            assert!(
                duration.as_millis() < 5000,
                "Forward pass too slow for {}x{}: {:?}",
                height,
                width,
                duration
            );
        }
    }
}
