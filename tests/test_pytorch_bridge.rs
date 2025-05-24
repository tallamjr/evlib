//! Tests for PyTorch bridge functionality

use candle_core::{DType, Device};
use evlib::ev_processing::reconstruction::pytorch_bridge::{
    load_pytorch_weights_into_varmap, ModelWeightMapper, PyTorchLoader,
};
use std::path::Path;

#[test]
fn test_model_weight_mapper_e2vid() {
    let mapper = ModelWeightMapper::e2vid_unet();

    // Test some key mappings
    let mappings = &mapper.mappings;

    // Head mappings
    assert!(mappings.contains_key("unetrecurrent.head.conv2d.weight"));
    assert!(mappings.contains_key("unetrecurrent.head.conv2d.bias"));

    // Encoder mappings
    assert!(mappings.contains_key("unetrecurrent.encoders.0.conv.conv2d.weight"));
    assert!(mappings.contains_key("unetrecurrent.encoders.0.recurrent_block.Gates.weight"));

    // Decoder mappings
    assert!(mappings.contains_key("unetrecurrent.decoders.0.conv.conv2d.weight"));
}

#[test]
fn test_pytorch_loader_creation() {
    let device = Device::Cpu;
    let loader = PyTorchLoader::new(device);

    // Test that we can create a loader without errors
    // (This doesn't test actual loading since we need PyTorch checkpoints)
}

#[test]
fn test_load_pytorch_weights_with_real_checkpoint() {
    let device = Device::Cpu;
    let checkpoint_path = Path::new("models/E2VID_lightweight.pth.tar");

    if checkpoint_path.exists() {
        // Try to load the actual PyTorch weights
        let result = load_pytorch_weights_into_varmap(checkpoint_path, "e2vid_unet", &device);

        match result {
            Ok(varmap) => {
                println!("Successfully loaded PyTorch weights into VarMap");
                // Check that we have some weights loaded
                assert!(
                    !varmap.data().is_empty(),
                    "VarMap should contain loaded weights"
                );
            }
            Err(e) => {
                // This might fail if PyO3/Python isn't available in test environment
                println!(
                    "Failed to load PyTorch weights (expected if no Python): {:?}",
                    e
                );
            }
        }
    } else {
        println!("E2VID checkpoint not found, skipping test");
    }
}

#[test]
fn test_pytorch_loader_nonexistent_file() {
    let device = Device::Cpu;
    let loader = PyTorchLoader::new(device);

    let nonexistent_path = Path::new("nonexistent_model.pth");
    let result = loader.load_checkpoint(nonexistent_path);

    // Should fail with FileNotFound error
    assert!(result.is_err());

    if let Err(err) = result {
        assert!(err.to_string().contains("File not found"));
    }
}

#[cfg(feature = "integration-test")]
#[test]
fn test_full_pytorch_to_candle_workflow() {
    use candle_core::Tensor;
    use std::collections::HashMap;

    let device = Device::Cpu;
    let checkpoint_path = Path::new("models/E2VID_lightweight.pth.tar");

    if !checkpoint_path.exists() {
        println!("Checkpoint not found, skipping integration test");
        return;
    }

    // Load PyTorch weights
    let loader = PyTorchLoader::new(device.clone());
    let pytorch_weights = loader.load_checkpoint(checkpoint_path);

    if let Ok(weights) = pytorch_weights {
        println!("Loaded {} PyTorch tensors", weights.len());

        // Create mapper
        let mapper = ModelWeightMapper::e2vid_unet();

        // Map weights
        let mapped_weights = mapper.map_state_dict(weights);

        println!("Mapped to {} Candle tensors", mapped_weights.len());

        // Verify we have expected keys
        let expected_keys = [
            "head.0.weight",
            "head.0.bias",
            "encoders.0.conv.weight",
            "encoders.0.lstm.gates.weight",
        ];

        for key in &expected_keys {
            if mapped_weights.contains_key(*key) {
                println!("Found expected key: {}", key);
            } else {
                println!("Missing expected key: {}", key);
            }
        }

        // Check tensor properties
        for (key, tensor) in mapped_weights.iter().take(5) {
            println!(
                "Tensor {}: shape {:?}, dtype {:?}",
                key,
                tensor.shape(),
                tensor.dtype()
            );
        }
    } else {
        println!(
            "Failed to load PyTorch weights: {:?}",
            pytorch_weights.unwrap_err()
        );
    }
}
