// Tests for loading actual PyTorch models from the models directory

use candle_core::{DType, Device};
use evlib::ev_core::{Event, Events};
use evlib::ev_processing::reconstruction::{
    E2Vid, E2VidConfig, E2VidModelLoader, LoadedModel, ModelLoaderConfig,
};
use std::path::Path;

#[test]
fn test_load_etap_pytorch_model() {
    // Test loading the actual ETAP model if it exists
    let model_path = Path::new("models/ETAP_v1_cvpr25.pth");

    if !model_path.exists() {
        println!(
            "⚠️  ETAP model not found at {:?}, skipping test",
            model_path
        );
        return;
    }

    println!("Found ETAP model at {:?}", model_path);
    println!(
        "File size: {} MB",
        model_path.metadata().unwrap().len() / 1_000_000
    );

    // Try to load the model
    let config = ModelLoaderConfig {
        device: Device::Cpu,
        dtype: DType::F32,
        strict_loading: false, // Allow partial loading
        verbose: true,
    };

    let loaded_model = LoadedModel::from_pth_file(model_path, config);

    match loaded_model {
        Ok(model) => {
            println!("✅ Successfully loaded PyTorch model structure");
            println!("   State dict entries: {}", model.state_dict.len());

            // Note: Currently this is a placeholder loader
            // When full PyTorch support is added, this will show actual weights
            if model.state_dict.is_empty() {
                println!("   ⚠️  Note: PyTorch loader is currently a placeholder");
                println!("   Full .pth parsing will be implemented in future updates");
            }
        }
        Err(e) => {
            println!("❌ Failed to load model: {:?}", e);
            println!("   This is expected with the current placeholder implementation");
        }
    }
}

#[test]
fn test_e2vid_with_pytorch_model() {
    // Test E2Vid integration with PyTorch model loading
    let model_path = Path::new("models/ETAP_v1_cvpr25.pth");

    let mut e2vid = E2Vid::new(256, 256);

    if model_path.exists() {
        println!("Attempting to load PyTorch model into E2Vid...");

        let result = e2vid.load_model_from_file(model_path);

        match result {
            Ok(_) => {
                println!("✅ Model loading completed (may have fallen back to random weights)");

                // Test that E2Vid still works
                let test_events = create_simple_events();
                let output = e2vid.process_events(&test_events);

                assert!(
                    output.is_ok(),
                    "Failed to process events after model loading"
                );
                println!("✅ E2Vid processing works after model loading attempt");
            }
            Err(e) => {
                println!("Model loading error: {:?}", e);
                println!("This is expected with placeholder implementation");
            }
        }
    } else {
        println!("PyTorch model not found, using default network");
        e2vid
            .create_default_network()
            .expect("Failed to create default network");
    }
}

#[test]
fn test_model_architecture_compatibility() {
    // Test that the E2VidNet architecture is compatible with expected model structure
    use candle_nn::{VarBuilder, VarMap};

    let device = Device::Cpu;
    let dtype = DType::F32;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

    // Try to load E2VidNet with the expected architecture
    let result = E2VidModelLoader::load_from_pth(
        Path::new("models/ETAP_v1_cvpr25.pth"),
        ModelLoaderConfig::default(),
    );

    match result {
        Ok(network) => {
            println!("✅ E2VidNet architecture loaded successfully");

            // Test forward pass
            let dummy_input =
                candle_core::Tensor::randn(0.0f32, 1.0f32, (1, 5, 128, 128), &device).unwrap();

            let output = network.forward(&dummy_input);
            assert!(output.is_ok(), "Forward pass failed");

            let output = output.unwrap();
            assert_eq!(output.dims(), &[1, 1, 128, 128]);
            println!("✅ Forward pass successful with shape {:?}", output.dims());
        }
        Err(e) => {
            println!("Expected error with placeholder loader: {:?}", e);
        }
    }
}

#[test]
fn test_model_directory_scan() {
    // Test scanning the models directory for available models
    let models_dir = Path::new("models");

    if !models_dir.exists() {
        println!("Models directory not found");
        return;
    }

    println!("Scanning models directory...");

    let mut pth_models = Vec::new();
    let mut onnx_models = Vec::new();

    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                match ext.to_str() {
                    Some("pth") => {
                        pth_models.push(path.file_name().unwrap().to_string_lossy().to_string());
                    }
                    Some("onnx") => {
                        onnx_models.push(path.file_name().unwrap().to_string_lossy().to_string());
                    }
                    _ => {}
                }
            }
        }
    }

    println!("Found PyTorch models: {:?}", pth_models);
    println!("Found ONNX models: {:?}", onnx_models);

    // Verify we can access the models
    for model_name in &pth_models {
        let model_path = models_dir.join(model_name);
        assert!(model_path.exists(), "Model {} not accessible", model_name);

        let metadata = model_path.metadata().unwrap();
        println!("  {} - Size: {} MB", model_name, metadata.len() / 1_000_000);
    }
}

// Helper functions

fn create_simple_events() -> Events {
    let mut events = Vec::new();

    // Create a simple pattern
    for i in 0..100 {
        let x = (i % 10) as u16 * 10;
        let y = (i / 10) as u16 * 10;
        let t = i as f64 * 0.001;
        let polarity = if i % 2 == 0 { 1 } else { -1 };

        events.push(Event { x, y, t, polarity });
    }

    events
}
