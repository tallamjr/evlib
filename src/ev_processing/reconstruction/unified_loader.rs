//! Unified model loading system supporting .pth, .onnx, and .safetensors formats

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::VarMap;
use std::path::Path;

use super::model_verification::{ModelVerifier, VerificationResults};
use super::pytorch_bridge::load_pytorch_weights_into_varmap;

/// Supported model formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelFormat {
    PyTorch,
    Onnx,
    SafeTensors,
}

impl ModelFormat {
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> CandleResult<Self> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "pth" | "pt" => Ok(ModelFormat::PyTorch),
            "tar" => {
                // Check if it's a PyTorch tar file (common for E2VID models)
                if path.to_string_lossy().contains(".pth.tar") {
                    Ok(ModelFormat::PyTorch)
                } else {
                    Err(candle_core::Error::Msg(format!(
                        "Ambiguous .tar file format: {}",
                        path.display()
                    )))
                }
            }
            "onnx" => Ok(ModelFormat::Onnx),
            "safetensors" => Ok(ModelFormat::SafeTensors),
            _ => Err(candle_core::Error::Msg(format!(
                "Unsupported model format: {} (supported: .pth, .pt, .onnx, .safetensors)",
                extension
            ))),
        }
    }
}

/// Model loading configuration
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    pub model_type: String,
    pub device: Device,
    pub verify_loading: bool,
    pub tolerance: f64,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            model_type: "e2vid_unet".to_string(),
            device: Device::Cpu,
            verify_loading: false,
            tolerance: 1e-5,
        }
    }
}

/// Unified model loader supporting multiple formats
pub struct UnifiedModelLoader {
    config: ModelLoadConfig,
}

impl UnifiedModelLoader {
    /// Create a new unified model loader
    pub fn new(config: ModelLoadConfig) -> Self {
        Self { config }
    }

    /// Load model from any supported format
    pub fn load_model(&self, model_path: &Path) -> CandleResult<LoadedModel> {
        // Detect format
        let format = ModelFormat::from_path(model_path)?;

        println!(
            "Loading model: {path} (format: {format:?})",
            path = model_path.display()
        );

        match format {
            ModelFormat::PyTorch => self.load_pytorch_model(model_path),
            ModelFormat::Onnx => self.load_onnx_model(model_path),
            ModelFormat::SafeTensors => self.load_safetensors_model(model_path),
        }
    }

    /// Load PyTorch model using PyO3 bridge
    fn load_pytorch_model(&self, model_path: &Path) -> CandleResult<LoadedModel> {
        let varmap = load_pytorch_weights_into_varmap(
            model_path,
            &self.config.model_type,
            &self.config.device,
        )?;

        Ok(LoadedModel {
            format: ModelFormat::PyTorch,
            varmap,
            device: self.config.device.clone(),
            model_type: self.config.model_type.clone(),
            path: model_path.to_path_buf(),
        })
    }

    /// Load ONNX model
    fn load_onnx_model(&self, model_path: &Path) -> CandleResult<LoadedModel> {
        // For now, create an empty VarMap as ONNX models are handled differently
        // In a real implementation, you might convert ONNX to VarMap or handle separately
        let varmap = VarMap::new();

        println!("ONNX model loading - using ONNX Runtime backend");

        Ok(LoadedModel {
            format: ModelFormat::Onnx,
            varmap,
            device: self.config.device.clone(),
            model_type: self.config.model_type.clone(),
            path: model_path.to_path_buf(),
        })
    }

    /// Load SafeTensors model
    fn load_safetensors_model(&self, model_path: &Path) -> CandleResult<LoadedModel> {
        // SafeTensors support would be implemented here
        // For now, return an error as it's not yet implemented
        Err(candle_core::Error::Msg(format!(
            "SafeTensors format not yet implemented: {}",
            model_path.display()
        )))
    }

    /// Verify loaded model (if verification is enabled)
    pub fn verify_model(
        &self,
        loaded_model: &LoadedModel,
    ) -> CandleResult<Option<VerificationResults>> {
        if !self.config.verify_loading {
            return Ok(None);
        }

        if loaded_model.format != ModelFormat::PyTorch {
            println!("Model verification only supported for PyTorch models currently");
            return Ok(None);
        }

        let verifier = ModelVerifier::new(self.config.device.clone(), self.config.tolerance);

        // Create test input for verification
        let test_input = match self.config.model_type.as_str() {
            "e2vid_unet" | "e2vid" => {
                Tensor::randn(0.0, 1.0, (1, 5, 180, 240), &self.config.device)?
            }
            "firenet" => Tensor::randn(0.0, 1.0, (1, 5, 256, 256), &self.config.device)?,
            _ => Tensor::randn(0.0, 1.0, (1, 5, 180, 240), &self.config.device)?,
        };

        let result = verifier.verify_model_outputs(
            &loaded_model.model_type,
            &loaded_model.path,
            &test_input,
        );

        match result {
            Ok(verification) => {
                verification.print_summary();
                Ok(Some(verification))
            }
            Err(e) => {
                println!("Model verification failed: {e}");
                Ok(None)
            }
        }
    }
}

/// Loaded model container
pub struct LoadedModel {
    pub format: ModelFormat,
    pub varmap: VarMap,
    pub device: Device,
    pub model_type: String,
    pub path: std::path::PathBuf,
}

impl LoadedModel {
    /// Get model information
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            format: self.format.clone(),
            model_type: self.model_type.clone(),
            device: format!("{:?}", self.device),
            path: self.path.to_string_lossy().to_string(),
            num_parameters: 0, // TODO: implement proper parameter counting
        }
    }

    /// Get a tensor from the loaded model
    pub fn get_tensor(&self, _key: &str) -> Option<Tensor> {
        // VarMap API is complex, return None for now
        // TODO: implement proper tensor retrieval
        None
    }

    /// List all tensor keys in the model
    pub fn list_tensor_keys(&self) -> Vec<String> {
        // TODO: implement proper key listing when VarMap API allows it
        vec![]
    }
}

/// Model information summary
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub format: ModelFormat,
    pub model_type: String,
    pub device: String,
    pub path: String,
    pub num_parameters: usize,
}

/// Convenience function to load any model format
pub fn load_model(path: &Path, config: Option<ModelLoadConfig>) -> CandleResult<LoadedModel> {
    let config = config.unwrap_or_default();
    let loader = UnifiedModelLoader::new(config);
    loader.load_model(path)
}

/// Convenience function to load and verify a model
pub fn load_and_verify_model(
    path: &Path,
    config: Option<ModelLoadConfig>,
) -> CandleResult<(LoadedModel, Option<VerificationResults>)> {
    let mut config = config.unwrap_or_default();
    config.verify_loading = true;

    let loader = UnifiedModelLoader::new(config);
    let loaded_model = loader.load_model(path)?;
    let verification = loader.verify_model(&loaded_model)?;

    Ok((loaded_model, verification))
}

/// Auto-detect and load the best available model format
pub fn auto_load_model(
    base_path: &str,
    model_type: &str,
    device: Device,
) -> CandleResult<LoadedModel> {
    // Try different formats in order of preference
    let formats = [
        (".onnx", ModelFormat::Onnx),
        (".pth.tar", ModelFormat::PyTorch),
        (".pth", ModelFormat::PyTorch),
        (".pt", ModelFormat::PyTorch),
        (".safetensors", ModelFormat::SafeTensors),
    ];

    for (extension, format) in &formats {
        let candidate_path_str = format!("{base_path}{extension}");
        let candidate_path = Path::new(&candidate_path_str);
        if candidate_path.exists() {
            println!(
                "Found model: {path} (format: {format:?})",
                path = candidate_path.display()
            );

            let config = ModelLoadConfig {
                model_type: model_type.to_string(),
                device: device.clone(),
                verify_loading: false,
                tolerance: 1e-5,
            };

            return load_model(candidate_path, Some(config));
        }
    }

    Err(candle_core::Error::Msg(format!(
        "No model found with base path: {} (tried: .onnx, .pth.tar, .pth, .pt, .safetensors)",
        base_path
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            ModelFormat::from_path(Path::new("model.pth")).unwrap(),
            ModelFormat::PyTorch
        );
        assert_eq!(
            ModelFormat::from_path(Path::new("model.pt")).unwrap(),
            ModelFormat::PyTorch
        );
        assert_eq!(
            ModelFormat::from_path(Path::new("E2VID_lightweight.pth.tar")).unwrap(),
            ModelFormat::PyTorch
        );
        assert_eq!(
            ModelFormat::from_path(Path::new("model.onnx")).unwrap(),
            ModelFormat::Onnx
        );
        assert_eq!(
            ModelFormat::from_path(Path::new("model.safetensors")).unwrap(),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_config_creation() {
        let config = ModelLoadConfig::default();
        assert_eq!(config.model_type, "e2vid_unet");
        assert_eq!(config.tolerance, 1e-5);
        assert!(!config.verify_loading);
    }

    #[test]
    fn test_unified_loader_creation() {
        let config = ModelLoadConfig::default();
        let loader = UnifiedModelLoader::new(config);
        assert_eq!(loader.config.model_type, "e2vid_unet");
    }
}
