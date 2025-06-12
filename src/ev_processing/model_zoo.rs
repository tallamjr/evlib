use candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Information about a pre-trained model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name (e.g., "e2vid_unet", "firenet", "spade_e2vid")
    pub name: String,
    /// Model variant (e.g., "base", "plus", "lite")
    pub variant: String,
    /// URL to download the model
    pub url: String,
    /// SHA256 checksum for verification
    pub checksum: String,
    /// Model architecture type
    pub architecture: ModelArchitecture,
    /// Model format (e.g., "onnx", "pytorch")
    pub format: String,
    /// Model size in bytes
    pub size: u64,
    /// Model description
    pub description: String,
}

/// Supported model architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelArchitecture {
    E2VidUNet,
    FireNet,
    E2VidPlus,
    FireNetPlus,
    SpadeE2Vid,
    HybridSpadeE2Vid,
    SpadeE2VidLite,
    SslE2Vid,
    ETNet,
    HyperE2Vid,
}

/// Configuration for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Base number of channels
    pub base_channels: usize,
    /// Number of layers/blocks
    pub num_layers: usize,
    /// Additional architecture-specific parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            in_channels: 5,  // Default for voxel grid with 5 time bins
            out_channels: 1, // Grayscale output
            base_channels: 64,
            num_layers: 4,
            extra_params: HashMap::new(),
        }
    }
}

/// Model Zoo for managing pre-trained models
pub struct ModelZoo {
    /// Available models
    models: HashMap<String, ModelInfo>,
    /// Cache directory for downloaded models
    cache_dir: PathBuf,
    /// Device for loading models
    device: Device,
}

impl ModelZoo {
    /// Create a new model zoo
    pub fn new(cache_dir: Option<PathBuf>) -> CandleResult<Self> {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".evlib").join("models")
        });

        // Create cache directory if it doesn't exist
        fs::create_dir_all(&cache_dir).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create cache directory: {}", e))
        })?;

        let device = Device::cuda_if_available(0)?;

        let mut zoo = Self {
            models: HashMap::new(),
            cache_dir,
            device,
        };

        // Initialize with available models
        zoo.initialize_models();

        Ok(zoo)
    }

    /// Initialize the model registry with available models
    fn initialize_models(&mut self) {
        // E2VID UNet - Official pretrained model from RPG
        self.models.insert(
            "e2vid_unet".to_string(),
            ModelInfo {
                name: "e2vid_unet".to_string(),
                variant: "lightweight".to_string(),
                url: "https://download.ifi.uzh.ch/rpg/web/data/E2VID/models/E2VID_lightweight.pth.tar"
                    .to_string(),
                checksum: "sha256:4cfeb2c850bf48fc9fa907e969cb8a04e3c51314da2d65bdb81145ac96574128".to_string(),
                architecture: ModelArchitecture::E2VidUNet,
                format: "pytorch".to_string(),
                size: 42_878_232, // ~41MB actual size
                description: "E2VID lightweight UNet from RPG (official pre-trained model)"
                    .to_string(),
            },
        );

        // NOTE: FireNet removed from model zoo until downloadable model weights are available
        // The repository at https://github.com/cedric-scheerlinck/rpg_e2vid/tree/cedric/firenet
        // does not provide direct download URLs for pre-trained weights

        // E2VID+ - Not yet available, removing from model zoo
        // TODO: Add E2VID+ when a real implementation is available

        // NOTE: SPADE-E2VID removed from model zoo until downloadable model weights are available
        // The repository at https://github.com/RodrigoGantier/SPADE_E2VID
        // does not provide direct download URLs for pre-trained weights

        // NOTE: SSL-E2VID removed from model zoo until downloadable model weights are available
        // The repository at https://github.com/tudelft/ssl_e2vid
        // does not provide direct download URLs for pre-trained weights

        // FireNet+ - Not yet available, removing from model zoo
        // TODO: Add FireNet+ when a real implementation is available

        // NOTE: ET-Net removed from model zoo until downloadable model weights are available
        // The repository at https://github.com/WarranWeng/ET-Net
        // does not provide direct download URLs for pre-trained weights
        // Placeholder checksum "et_net_checkpoint_pending_release" indicates unavailable weights

        // HyperE2VID - Not yet available, removing from model zoo
        // TODO: Add HyperE2VID when a real implementation is available
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// Get model info by name
    pub fn get_model_info(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }

    /// Verify that a model has a valid download URL
    pub fn verify_model_availability(&self, name: &str) -> Result<bool, String> {
        let info = self
            .models
            .get(name)
            .ok_or_else(|| format!("Model '{}' not found in model zoo", name))?;

        // Check if URL is a direct download link (not a repository URL)
        if info.url.contains("github.com") && !info.url.contains("/releases/download/") {
            return Err(format!(
                "Model '{}' has repository URL instead of direct download link: {}",
                name, info.url
            ));
        }

        // Check for placeholder checksums
        if info.checksum.contains("from_repository") || info.checksum.contains("pending_release") {
            return Err(format!(
                "Model '{}' has placeholder checksum: {}",
                name, info.checksum
            ));
        }

        Ok(true)
    }

    /// List only models with verified download URLs
    pub fn list_verified_models(&self) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|info| self.verify_model_availability(&info.name).is_ok())
            .collect()
    }

    /// Get the path where a model would be cached
    pub fn get_model_path(&self, name: &str) -> Option<PathBuf> {
        self.models.get(name).map(|info| {
            let extension = match info.format.as_str() {
                "onnx" => "onnx",
                "pytorch" => "pth",
                _ => "bin",
            };
            self.cache_dir.join(format!("{}.{}", info.name, extension))
        })
    }

    /// Check if a model is already downloaded
    pub fn is_downloaded(&self, name: &str) -> bool {
        if let Some(path) = self.get_model_path(name) {
            path.exists()
        } else {
            false
        }
    }

    /// Download a model if not already cached
    pub async fn download_model(&self, name: &str) -> CandleResult<PathBuf> {
        // Verify model availability before attempting download
        if let Err(e) = self.verify_model_availability(name) {
            return Err(candle_core::Error::Msg(format!(
                "Cannot download model '{}': {}",
                name, e
            )));
        }

        let info = self
            .models
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Model '{}' not found", name)))?;

        let extension = match info.format.as_str() {
            "onnx" => "onnx",
            "pytorch" => "pth",
            _ => "bin",
        };
        let model_path = self.cache_dir.join(format!("{}.{}", info.name, extension));

        if model_path.exists() {
            println!("Model '{}' already downloaded", name);
            return Ok(model_path);
        }

        println!("Downloading model '{}' from {}", name, info.url);

        // Download the model
        let response = reqwest::get(&info.url)
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to download model: {}", e)))?;

        let bytes = response
            .bytes()
            .await
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read response: {}", e)))?;

        // Save to file
        fs::write(&model_path, bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to save model: {}", e)))?;

        println!("Model '{}' downloaded successfully", name);
        Ok(model_path)
    }

    /// Download a model synchronously
    pub fn download_model_sync(&self, name: &str) -> CandleResult<PathBuf> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create runtime: {}", e)))?;
        runtime.block_on(self.download_model(name))
    }

    /// Create a VarBuilder for loading model weights
    fn create_var_builder(&self, model_path: &Path) -> CandleResult<VarBuilder> {
        let extension = model_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension {
            "pth" | "pt" => {
                // Try to load PyTorch weights using the bridge
                self.load_pytorch_weights(model_path)
            }
            "onnx" => {
                // For ONNX files, we still initialize with random weights
                // as ONNX loading is handled separately
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
                eprintln!(
                    "Note: ONNX model at {:?} - use events_to_video_advanced() for ONNX inference",
                    model_path
                );
                Ok(vb)
            }
            _ => Err(candle_core::Error::Msg(format!(
                "Unsupported model format for file: {:?}. Supported formats: .pth, .pt, .onnx",
                model_path
            ))),
        }
    }

    /// Load PyTorch weights using the PyTorch bridge
    fn load_pytorch_weights(&self, model_path: &Path) -> CandleResult<VarBuilder> {
        // Determine model type from filename
        let filename = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        let model_type = if filename.contains("e2vid") {
            "e2vid_unet"
        } else if filename.contains("firenet") {
            "firenet"
        } else if filename.contains("et_net") {
            "et_net"
        } else {
            return Err(candle_core::Error::Msg(format!(
                "Cannot determine model type from filename: {:?}",
                model_path
            )));
        };

        // Try to load PyTorch weights
        match super::reconstruction::load_pytorch_weights_into_varmap(
            model_path,
            model_type,
            &self.device,
        ) {
            Ok(varmap) => {
                eprintln!("Successfully loaded PyTorch weights from {:?}", model_path);
                Ok(VarBuilder::from_varmap(&varmap, DType::F32, &self.device))
            }
            Err(e) => {
                Err(candle_core::Error::Msg(format!(
                    "Failed to load PyTorch weights from {:?}: {}. Ensure the model file is valid and accessible.",
                    model_path, e
                )))
            }
        }
    }

    /// Load a model by name
    pub fn load_model(
        &self,
        name: &str,
        config: Option<ModelConfig>,
    ) -> CandleResult<Box<dyn EventToVideoModel>> {
        let info = self
            .models
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Model '{}' not found", name)))?;

        let config = config.unwrap_or_default();

        // Ensure model is downloaded
        let model_path = self.download_model_sync(name)?;

        // Create VarBuilder for loading weights
        let vb = self.create_var_builder(&model_path)?;

        // Load the appropriate model based on architecture
        match info.architecture {
            ModelArchitecture::E2VidUNet => {
                let model = super::reconstruction::E2VidUNet::new(
                    vb,
                    config.in_channels,
                    config.base_channels,
                )?;
                Ok(Box::new(ModelWrapper::E2VidUNet(model)))
            }
            ModelArchitecture::FireNet => {
                let model = super::reconstruction::FireNet::new(vb, config.in_channels)?;
                Ok(Box::new(ModelWrapper::FireNet(model)))
            }
            ModelArchitecture::E2VidPlus => {
                let model = super::reconstruction::E2VidPlus::new(
                    vb,
                    config.out_channels,
                    config.base_channels,
                    config.in_channels,
                )?;
                Ok(Box::new(ModelWrapper::E2VidPlus(model)))
            }
            ModelArchitecture::SpadeE2Vid => {
                let use_skip = config
                    .extra_params
                    .get("use_skip_connections")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let model = super::reconstruction::SpadeE2Vid::new(
                    vb,
                    config.out_channels,
                    config.base_channels,
                    use_skip,
                )?;
                Ok(Box::new(ModelWrapper::SpadeE2Vid(model)))
            }
            ModelArchitecture::ETNet => {
                let et_config = super::reconstruction::ETNetConfig {
                    in_channels: config.in_channels,
                    hidden_dim: config
                        .extra_params
                        .get("hidden_dim")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(768) as usize,
                    num_layers: config.num_layers,
                    num_heads: config
                        .extra_params
                        .get("num_heads")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(12) as usize,
                    ff_dim: config
                        .extra_params
                        .get("ff_dim")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(3072) as usize,
                    dropout: config
                        .extra_params
                        .get("dropout")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.1) as f32,
                    patch_size: config
                        .extra_params
                        .get("patch_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(16) as usize,
                    out_channels: config.out_channels,
                };
                let model = super::reconstruction::ETNet::new(vb, et_config)?;
                Ok(Box::new(ModelWrapper::ETNet(model)))
            }
            ModelArchitecture::HyperE2Vid => {
                let hyper_config = super::reconstruction::HyperE2VidConfig {
                    in_channels: config.in_channels,
                    base_channels: config.base_channels,
                    out_channels: config.out_channels,
                    num_blocks: config.num_layers,
                    hyper_hidden_dim: config
                        .extra_params
                        .get("hyper_hidden_dim")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(256) as usize,
                    num_kernels: config
                        .extra_params
                        .get("num_kernels")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(3) as usize,
                    use_skip_connections: config
                        .extra_params
                        .get("use_skip_connections")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true),
                };
                let model = super::reconstruction::HyperE2Vid::new(vb, hyper_config)?;
                Ok(Box::new(ModelWrapper::HyperE2Vid(model)))
            }
            _ => Err(candle_core::Error::Msg(format!(
                "Architecture {:?} not yet implemented",
                info.architecture
            ))),
        }
    }
}

/// Trait for event-to-video reconstruction models
pub trait EventToVideoModel: Send + Sync {
    /// Forward pass through the model
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor>;

    /// Get model name
    fn name(&self) -> &str;
}

/// Wrapper enum for different model types
enum ModelWrapper {
    E2VidUNet(super::reconstruction::E2VidUNet),
    FireNet(super::reconstruction::FireNet),
    E2VidPlus(super::reconstruction::E2VidPlus),
    SpadeE2Vid(super::reconstruction::SpadeE2Vid),
    ETNet(super::reconstruction::ETNet),
    HyperE2Vid(super::reconstruction::HyperE2Vid),
}

impl EventToVideoModel for ModelWrapper {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            ModelWrapper::E2VidUNet(model) => model.forward(x),
            ModelWrapper::FireNet(model) => model.forward(x),
            ModelWrapper::E2VidPlus(model) => model.forward(x),
            ModelWrapper::SpadeE2Vid(model) => model.forward(x),
            ModelWrapper::ETNet(model) => model.forward(x),
            ModelWrapper::HyperE2Vid(model) => model.forward(x),
        }
    }

    fn name(&self) -> &str {
        match self {
            ModelWrapper::E2VidUNet(_) => "E2VidUNet",
            ModelWrapper::FireNet(_) => "FireNet",
            ModelWrapper::E2VidPlus(_) => "E2VidPlus",
            ModelWrapper::SpadeE2Vid(_) => "SpadeE2Vid",
            ModelWrapper::ETNet(_) => "ETNet",
            ModelWrapper::HyperE2Vid(_) => "HyperE2Vid",
        }
    }
}

/// Python bindings for model zoo
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyfunction]
    pub fn list_available_models() -> PyResult<Vec<String>> {
        let zoo = ModelZoo::new(None).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create model zoo: {}", e))
        })?;

        Ok(zoo
            .list_models()
            .iter()
            .map(|info| info.name.clone())
            .collect())
    }

    #[pyfunction]
    pub fn download_model(name: &str) -> PyResult<String> {
        let zoo = ModelZoo::new(None).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create model zoo: {}", e))
        })?;

        let path = zoo.download_model_sync(name).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to download model: {}", e))
        })?;

        Ok(path.to_string_lossy().to_string())
    }

    #[pyfunction]
    pub fn get_model_info_py(py: Python, name: &str) -> PyResult<PyObject> {
        let zoo = ModelZoo::new(None).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create model zoo: {}", e))
        })?;

        let info = zoo.get_model_info(name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Model '{}' not found", name))
        })?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("name", &info.name)?;
        dict.set_item("variant", &info.variant)?;
        dict.set_item("url", &info.url)?;
        dict.set_item("checksum", &info.checksum)?;
        dict.set_item("architecture", format!("{:?}", info.architecture))?;
        dict.set_item("format", &info.format)?;
        dict.set_item("size", info.size)?;
        dict.set_item("description", &info.description)?;
        Ok(dict.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_zoo_creation() {
        let zoo = ModelZoo::new(None).unwrap();
        assert!(!zoo.models.is_empty());
    }

    #[test]
    fn test_list_models() {
        let zoo = ModelZoo::new(None).unwrap();
        let models = zoo.list_models();
        assert!(models.len() >= 1); // We have at least 1 verified model (e2vid_unet)
    }

    #[test]
    fn test_list_verified_models() {
        let zoo = ModelZoo::new(None).unwrap();
        let verified_models = zoo.list_verified_models();
        // Only models with real download URLs should be included
        assert!(!verified_models.is_empty());
        // e2vid_unet should be available as it has a real download URL
        assert!(verified_models.iter().any(|m| m.name == "e2vid_unet"));
    }

    #[test]
    fn test_verify_model_availability() {
        let zoo = ModelZoo::new(None).unwrap();
        // e2vid_unet should be verified
        assert!(zoo.verify_model_availability("e2vid_unet").is_ok());
        // Non-existent model should fail
        assert!(zoo.verify_model_availability("non_existent_model").is_err());
    }

    #[test]
    fn test_get_model_info() {
        let zoo = ModelZoo::new(None).unwrap();
        let info = zoo.get_model_info("e2vid_unet");
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "e2vid_unet");
    }
}
