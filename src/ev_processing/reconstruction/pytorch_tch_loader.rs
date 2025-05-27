//! PyTorch model loading using tch-rs (LibTorch bindings)
//! This provides native PyTorch .pth file loading without Python dependencies

#[cfg(feature = "pytorch")]
use tch::{nn, Device as TchDevice, Kind, Tensor as TchTensor};

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Error types for tch-rs PyTorch operations
#[derive(Debug)]
pub enum TchLoaderError {
    TchError(String),
    ConversionError(String),
    DeviceError(String),
    ModelNotFound(String),
}

impl std::fmt::Display for TchLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TchLoaderError::TchError(msg) => write!(f, "PyTorch error: {}", msg),
            TchLoaderError::ConversionError(msg) => write!(f, "Tensor conversion error: {}", msg),
            TchLoaderError::DeviceError(msg) => write!(f, "Device error: {}", msg),
            TchLoaderError::ModelNotFound(msg) => write!(f, "Model not found: {}", msg),
        }
    }
}

impl std::error::Error for TchLoaderError {}

#[cfg(feature = "pytorch")]
impl From<tch::TchError> for TchLoaderError {
    fn from(error: tch::TchError) -> Self {
        TchLoaderError::TchError(error.to_string())
    }
}

/// PyTorch model loader using tch-rs
#[cfg(feature = "pytorch")]
pub struct TchModelLoader {
    device: TchDevice,
}

#[cfg(feature = "pytorch")]
impl TchModelLoader {
    /// Create a new tch-rs model loader
    pub fn new(use_gpu: bool) -> Result<Self, TchLoaderError> {
        let device = if use_gpu && tch::Cuda::is_available() {
            TchDevice::Cuda(0)
        } else {
            TchDevice::Cpu
        };

        Ok(Self { device })
    }

    /// Load a PyTorch model directly using tch-rs
    pub fn load_model<P: AsRef<Path>>(&self, model_path: P) -> Result<TchModel, TchLoaderError> {
        if !model_path.as_ref().exists() {
            return Err(TchLoaderError::ModelNotFound(format!(
                "Model file not found: {:?}",
                model_path.as_ref()
            )));
        }

        // Load the model using tch-rs
        let vs = nn::VarStore::new(self.device);

        // Load state dict
        vs.load(model_path.as_ref())?;

        println!("Successfully loaded PyTorch model using tch-rs");

        Ok(TchModel {
            vs,
            device: self.device,
        })
    }

    /// Convert tch tensor to Candle tensor
    pub fn tch_to_candle_tensor(
        &self,
        tch_tensor: &TchTensor,
        target_device: &Device,
    ) -> Result<Tensor, TchLoaderError> {
        // Get tensor data as Vec<f32>
        let data: Vec<f32> = tch_tensor
            .to_kind(Kind::Float)
            .to_device(TchDevice::Cpu)
            .try_into()
            .map_err(|e| {
                TchLoaderError::ConversionError(format!("Failed to extract tensor data: {}", e))
            })?;

        // Get tensor shape
        let shape: Vec<usize> = tch_tensor.size().into_iter().map(|s| s as usize).collect();

        // Create Candle tensor
        Tensor::from_vec(data, shape, target_device).map_err(|e| {
            TchLoaderError::ConversionError(format!("Failed to create Candle tensor: {}", e))
        })
    }

    /// Get state dict as HashMap for compatibility with existing code
    pub fn extract_state_dict(
        &self,
        model: &TchModel,
        target_device: &Device,
    ) -> Result<HashMap<String, Tensor>, TchLoaderError> {
        let mut state_dict = HashMap::new();

        // Iterate through named variables in the VarStore
        for (name, tensor) in model.vs.variables() {
            let candle_tensor = self.tch_to_candle_tensor(&tensor, target_device)?;
            state_dict.insert(name, candle_tensor);
        }

        Ok(state_dict)
    }
}

/// Wrapper for a loaded tch-rs model
#[cfg(feature = "pytorch")]
pub struct TchModel {
    pub vs: nn::VarStore,
    pub device: TchDevice,
}

#[cfg(feature = "pytorch")]
impl TchModel {
    /// Perform inference on the model
    pub fn forward(&self, input: &TchTensor) -> Result<TchTensor, TchLoaderError> {
        // This would need to be implemented per-model architecture
        // For now, return the input (identity function)
        Ok(input.copy())
    }

    /// Get a specific parameter by name
    pub fn get_parameter(&self, name: &str) -> Option<TchTensor> {
        self.vs
            .variables()
            .into_iter()
            .find(|(param_name, _)| param_name == name)
            .map(|(_, tensor)| tensor)
    }
}

/// Fallback implementation when pytorch feature is disabled
#[cfg(not(feature = "pytorch"))]
pub struct TchModelLoader;

#[cfg(not(feature = "pytorch"))]
impl TchModelLoader {
    pub fn new(_use_gpu: bool) -> Result<Self, TchLoaderError> {
        Err(TchLoaderError::TchError(
            "tch-rs support not enabled. Compile with --features pytorch".to_string(),
        ))
    }

    pub fn load_model<P: AsRef<Path>>(&self, _model_path: P) -> Result<TchModel, TchLoaderError> {
        Err(TchLoaderError::TchError(
            "tch-rs support not enabled. Compile with --features pytorch".to_string(),
        ))
    }
}

#[cfg(not(feature = "pytorch"))]
pub struct TchModel;

/// Utility functions for common model operations
pub struct ModelConverter;

impl ModelConverter {
    /// Convert E2VID PyTorch model to Candle-compatible format
    #[cfg(feature = "pytorch")]
    pub fn convert_e2vid_model<P: AsRef<Path>>(
        model_path: P,
        target_device: &Device,
    ) -> Result<HashMap<String, Tensor>, TchLoaderError> {
        let loader = TchModelLoader::new(matches!(target_device, Device::Cuda(_)))?;
        let model = loader.load_model(model_path)?;
        loader.extract_state_dict(&model, target_device)
    }

    #[cfg(not(feature = "pytorch"))]
    pub fn convert_e2vid_model<P: AsRef<Path>>(
        _model_path: P,
        _target_device: &Device,
    ) -> Result<HashMap<String, Tensor>, TchLoaderError> {
        Err(TchLoaderError::TchError(
            "tch-rs support not enabled. Compile with --features pytorch".to_string(),
        ))
    }
}

#[cfg(all(test, feature = "pytorch"))]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_tch_loader_creation() {
        let loader = TchModelLoader::new(false);
        assert!(loader.is_ok(), "Failed to create tch-rs loader");
    }

    #[test]
    fn test_missing_model_error() {
        let loader = TchModelLoader::new(false).unwrap();
        let temp_dir = TempDir::new().unwrap();
        let non_existent = temp_dir.path().join("nonexistent.pth");

        let result = loader.load_model(&non_existent);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TchLoaderError::ModelNotFound(_)
        ));
    }
}
