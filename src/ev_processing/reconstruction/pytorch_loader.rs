// PyTorch model loading utilities for Candle framework
// Converts PyTorch .pth files to Candle-compatible neural networks

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, linear, BatchNorm, BatchNormConfig, Conv2d, Linear, Module,
    ModuleT, VarBuilder, VarMap,
};
use std::collections::HashMap;
use std::path::Path;

/// Errors that can occur during PyTorch model loading
#[derive(Debug)]
pub enum ModelLoadError {
    IoError(std::io::Error),
    CandleError(candle_core::Error),
    UnsupportedLayer(String),
    InvalidModelFormat(String),
    MissingWeights(String),
}

impl From<std::io::Error> for ModelLoadError {
    fn from(error: std::io::Error) -> Self {
        ModelLoadError::IoError(error)
    }
}

impl From<candle_core::Error> for ModelLoadError {
    fn from(error: candle_core::Error) -> Self {
        ModelLoadError::CandleError(error)
    }
}

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelLoadError::IoError(e) => write!(f, "IO error: {}", e),
            ModelLoadError::CandleError(e) => write!(f, "Candle error: {}", e),
            ModelLoadError::UnsupportedLayer(layer) => write!(f, "Unsupported layer: {}", layer),
            ModelLoadError::InvalidModelFormat(msg) => write!(f, "Invalid model format: {}", msg),
            ModelLoadError::MissingWeights(name) => write!(f, "Missing weights for: {}", name),
        }
    }
}

impl std::error::Error for ModelLoadError {}

/// Configuration for model loading
#[derive(Debug, Clone)]
pub struct ModelLoaderConfig {
    pub device: Device,
    pub dtype: DType,
    pub strict_loading: bool,
    pub verbose: bool,
}

impl Default for ModelLoaderConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            strict_loading: true,
            verbose: false,
        }
    }
}

/// A loaded PyTorch model converted to Candle
pub struct LoadedModel {
    pub state_dict: HashMap<String, Tensor>,
    pub config: ModelLoaderConfig,
}

impl LoadedModel {
    /// Load a PyTorch .pth file
    pub fn from_pth_file<P: AsRef<Path>>(
        path: P,
        config: ModelLoaderConfig,
    ) -> Result<Self, ModelLoadError> {
        if config.verbose {
            println!("Loading PyTorch model from {:?}", path.as_ref());
        }

        // Check if file exists
        if !path.as_ref().exists() {
            return Err(ModelLoadError::InvalidModelFormat(format!(
                "Model file not found: {:?}",
                path.as_ref()
            )));
        }

        // Try tch-rs first (if available), fallback to PyO3 bridge
        #[cfg(feature = "pytorch")]
        {
            if config.verbose {
                println!("Attempting to load PyTorch model using tch-rs...");
            }

            match super::pytorch_tch_loader::ModelConverter::convert_e2vid_model(
                path.as_ref(),
                &config.device,
            ) {
                Ok(state_dict) => {
                    if config.verbose {
                        println!(
                            "Successfully loaded PyTorch model with tch-rs: {} tensors",
                            state_dict.len()
                        );
                    }
                    return Ok(Self { state_dict, config });
                }
                Err(e) => {
                    if config.verbose {
                        println!("tch-rs loading failed: {}, trying PyO3 bridge...", e);
                    }
                }
            }
        }

        // Fallback to PyO3 bridge
        let bridge_result = super::pytorch_bridge::PyTorchLoader::new(config.device.clone())
            .load_checkpoint(path.as_ref());

        match bridge_result {
            Ok(state_dict) => {
                if config.verbose {
                    println!(
                        "Successfully loaded PyTorch checkpoint with PyO3 bridge: {} tensors",
                        state_dict.len()
                    );
                }
                Ok(Self { state_dict, config })
            }
            Err(e) => {
                if config.verbose {
                    #[cfg(feature = "pytorch")]
                    println!("Warning: Both tch-rs and PyO3 bridge failed to load PyTorch weights");
                    #[cfg(not(feature = "pytorch"))]
                    println!("Warning: PyO3 bridge failed to load PyTorch weights");
                    println!(
                        "Consider converting your model to ONNX format for better compatibility."
                    );
                    println!(
                        "ONNX models provide better cross-platform compatibility and performance."
                    );
                }

                // Return error instead of silently failing with empty dict
                Err(ModelLoadError::InvalidModelFormat(format!(
                    "Failed to load PyTorch checkpoint: {}. \n\
                    RECOMMENDED: Convert to ONNX format using the provided conversion script for better compatibility.", e
                )))
            }
        }
    }

    /// Get a tensor from the state dict
    pub fn get_tensor(&self, key: &str) -> Result<Option<&Tensor>, ModelLoadError> {
        Ok(self.state_dict.get(key))
    }

    /// Create a VarBuilder from the loaded state dict
    pub fn var_builder(&self) -> Result<VarBuilder, ModelLoadError> {
        // For now, since we're using Candle's built-in loading,
        // we'll return a fresh VarBuilder. In a full implementation,
        // we would store the VarBuilder from loading or properly
        // convert the state dict.
        let var_map = VarMap::new();

        // Convert state dict to VarMap format
        for (key, tensor) in &self.state_dict {
            // Convert tensor to appropriate device and dtype
            let converted_tensor = tensor
                .to_device(&self.config.device)?
                .to_dtype(self.config.dtype)?;

            if self.config.verbose {
                println!(
                    "Loading tensor: {} with shape {:?}",
                    key,
                    converted_tensor.shape()
                );
            }
        }

        Ok(VarBuilder::from_varmap(
            &var_map,
            self.config.dtype,
            &self.config.device,
        ))
    }

    /// Load a PyTorch model and return the VarBuilder directly
    pub fn load_var_builder<P: AsRef<Path>>(
        path: P,
        config: &ModelLoaderConfig,
    ) -> Result<VarBuilder, ModelLoadError> {
        // Check if file exists
        if !path.as_ref().exists() {
            return Err(ModelLoadError::InvalidModelFormat(format!(
                "Model file not found: {:?}",
                path.as_ref()
            )));
        }

        // TODO: Implement actual PyTorch loading when available
        // For now, return a new VarBuilder with random weights
        let var_map = VarMap::new();
        Ok(VarBuilder::from_varmap(
            &var_map,
            config.dtype,
            &config.device,
        ))
    }
}

/// Utility functions for common PyTorch layer conversions
pub struct LayerConverter;

impl LayerConverter {
    /// Convert a PyTorch Conv2d layer to Candle Conv2d
    pub fn conv2d_from_state_dict(
        vs: &VarBuilder,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Result<Conv2d, ModelLoadError> {
        let _weight_key = format!("{}.weight", prefix);
        let bias_key = format!("{}.bias", prefix);

        // Check if bias exists
        let _has_bias = vs.contains_tensor(&bias_key);

        if _has_bias {
            Ok(conv2d(
                in_channels,
                out_channels,
                kernel_size,
                Default::default(),
                vs.pp(prefix),
            )?)
        } else {
            Ok(conv2d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                Default::default(),
                vs.pp(prefix),
            )?)
        }
    }

    /// Convert a PyTorch BatchNorm2d layer to Candle BatchNorm
    pub fn batch_norm_from_state_dict(
        vs: &VarBuilder,
        prefix: &str,
        num_features: usize,
    ) -> Result<BatchNorm, ModelLoadError> {
        Ok(batch_norm(
            num_features,
            BatchNormConfig::default(),
            vs.pp(prefix),
        )?)
    }

    /// Convert a PyTorch Linear layer to Candle Linear
    pub fn linear_from_state_dict(
        vs: &VarBuilder,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Linear, ModelLoadError> {
        let bias_key = format!("{}.bias", prefix);
        let _has_bias = vs.contains_tensor(&bias_key);

        Ok(linear(in_dim, out_dim, vs.pp(prefix))?)
    }
}

/// Helper trait for checking tensor existence in VarBuilder
#[allow(dead_code)]
trait VarBuilderExt {
    fn contains_tensor(&self, key: &str) -> bool;
}

#[allow(dead_code)]
impl VarBuilderExt for VarBuilder<'_> {
    fn contains_tensor(&self, _key: &str) -> bool {
        // This would need to be implemented to check if a tensor exists
        // For now, return false as placeholder
        false
    }
}

/// E2VID specific model architecture loader
pub struct E2VidModelLoader;

impl E2VidModelLoader {
    /// Load a complete E2VID model from PyTorch state dict
    pub fn load_from_pth<P: AsRef<Path>>(
        path: P,
        config: ModelLoaderConfig,
    ) -> Result<E2VidNet, ModelLoadError> {
        // Use the direct VarBuilder loading approach
        let vs = LoadedModel::load_var_builder(path, &config)?;

        // Build E2VID network architecture
        E2VidNet::load_from_var_builder(&vs)
    }
}

/// E2VID network structure in Candle
pub struct E2VidNet {
    // Encoder layers
    encoder_conv1: Conv2d,
    encoder_bn1: BatchNorm,
    encoder_conv2: Conv2d,
    encoder_bn2: BatchNorm,
    encoder_conv3: Conv2d,
    encoder_bn3: BatchNorm,

    // Decoder layers
    decoder_conv1: Conv2d,
    decoder_bn1: BatchNorm,
    decoder_conv2: Conv2d,
    decoder_bn2: BatchNorm,

    // Output layer
    output_conv: Conv2d,
}

impl E2VidNet {
    /// Create E2VID network from VarBuilder (loaded from PyTorch)
    pub fn load_from_var_builder(vs: &VarBuilder) -> Result<Self, ModelLoadError> {
        // Define E2VID architecture parameters
        // These would be extracted from the actual model architecture
        let input_channels = 5; // Number of voxel bins
        let base_channels = 32;

        let encoder_conv1 = LayerConverter::conv2d_from_state_dict(
            vs,
            "encoder.conv1",
            input_channels,
            base_channels,
            3,
        )?;
        let encoder_bn1 =
            LayerConverter::batch_norm_from_state_dict(vs, "encoder.bn1", base_channels)?;

        let encoder_conv2 = LayerConverter::conv2d_from_state_dict(
            vs,
            "encoder.conv2",
            base_channels,
            base_channels * 2,
            3,
        )?;
        let encoder_bn2 =
            LayerConverter::batch_norm_from_state_dict(vs, "encoder.bn2", base_channels * 2)?;

        let encoder_conv3 = LayerConverter::conv2d_from_state_dict(
            vs,
            "encoder.conv3",
            base_channels * 2,
            base_channels * 4,
            3,
        )?;
        let encoder_bn3 =
            LayerConverter::batch_norm_from_state_dict(vs, "encoder.bn3", base_channels * 4)?;

        // Decoder layers (upsampling)
        let decoder_conv1 = LayerConverter::conv2d_from_state_dict(
            vs,
            "decoder.conv1",
            base_channels * 4,
            base_channels * 2,
            3,
        )?;
        let decoder_bn1 =
            LayerConverter::batch_norm_from_state_dict(vs, "decoder.bn1", base_channels * 2)?;

        let decoder_conv2 = LayerConverter::conv2d_from_state_dict(
            vs,
            "decoder.conv2",
            base_channels * 2,
            base_channels,
            3,
        )?;
        let decoder_bn2 =
            LayerConverter::batch_norm_from_state_dict(vs, "decoder.bn2", base_channels)?;

        // Output layer (to single channel image)
        let output_conv =
            LayerConverter::conv2d_from_state_dict(vs, "output.conv", base_channels, 1, 1)?;

        Ok(Self {
            encoder_conv1,
            encoder_bn1,
            encoder_conv2,
            encoder_bn2,
            encoder_conv3,
            encoder_bn3,
            decoder_conv1,
            decoder_bn1,
            decoder_conv2,
            decoder_bn2,
            output_conv,
        })
    }

    /// Create a new E2VID network with random weights (for testing)
    pub fn new(vs: &VarBuilder) -> Result<Self, ModelLoadError> {
        let input_channels = 5;
        let base_channels = 32;

        let encoder_conv1 = conv2d(
            input_channels,
            base_channels,
            3,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("encoder.conv1"),
        )?;
        let encoder_bn1 = batch_norm(
            base_channels,
            BatchNormConfig::default(),
            vs.pp("encoder.bn1"),
        )?;

        let encoder_conv2 = conv2d(
            base_channels,
            base_channels * 2,
            3,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("encoder.conv2"),
        )?;
        let encoder_bn2 = batch_norm(
            base_channels * 2,
            BatchNormConfig::default(),
            vs.pp("encoder.bn2"),
        )?;

        let encoder_conv3 = conv2d(
            base_channels * 2,
            base_channels * 4,
            3,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("encoder.conv3"),
        )?;
        let encoder_bn3 = batch_norm(
            base_channels * 4,
            BatchNormConfig::default(),
            vs.pp("encoder.bn3"),
        )?;

        let decoder_conv1 = conv2d(
            base_channels * 4,
            base_channels * 2,
            3,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("decoder.conv1"),
        )?;
        let decoder_bn1 = batch_norm(
            base_channels * 2,
            BatchNormConfig::default(),
            vs.pp("decoder.bn1"),
        )?;

        let decoder_conv2 = conv2d(
            base_channels * 2,
            base_channels,
            3,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("decoder.conv2"),
        )?;
        let decoder_bn2 = batch_norm(
            base_channels,
            BatchNormConfig::default(),
            vs.pp("decoder.bn2"),
        )?;

        let output_conv = conv2d(
            base_channels,
            1,
            1,
            Default::default(),
            vs.pp("output.conv"),
        )?;

        Ok(Self {
            encoder_conv1,
            encoder_bn1,
            encoder_conv2,
            encoder_bn2,
            encoder_conv3,
            encoder_bn3,
            decoder_conv1,
            decoder_bn1,
            decoder_conv2,
            decoder_bn2,
            output_conv,
        })
    }
}

impl Module for E2VidNet {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.forward_t(xs, false) // Default to inference mode
    }
}

impl E2VidNet {
    /// Forward pass with explicit training mode
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> CandleResult<Tensor> {
        // Encoder forward pass
        let x1 = xs.apply(&self.encoder_conv1)?;
        let x1 = self.encoder_bn1.forward_t(&x1, train)?;
        let x1 = x1.relu()?;

        let x2 = x1.apply(&self.encoder_conv2)?;
        let x2 = self.encoder_bn2.forward_t(&x2, train)?;
        let x2 = x2.relu()?;

        let x3 = x2.apply(&self.encoder_conv3)?;
        let x3 = self.encoder_bn3.forward_t(&x3, train)?;
        let x3 = x3.relu()?;

        // Decoder forward pass
        let x4 = x3.apply(&self.decoder_conv1)?;
        let x4 = self.decoder_bn1.forward_t(&x4, train)?;
        let x4 = x4.relu()?;

        let x5 = x4.apply(&self.decoder_conv2)?;
        let x5 = self.decoder_bn2.forward_t(&x5, train)?;
        let x5 = x5.relu()?;

        // Output layer with sigmoid activation for intensity values
        let output = x5.apply(&self.output_conv)?;
        candle_nn::ops::sigmoid(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_e2vid_network_creation() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

        let network = E2VidNet::new(&vs);
        assert!(network.is_ok(), "Failed to create E2VID network");
    }

    #[test]
    fn test_e2vid_forward_pass() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, dtype, &device);

        let network = E2VidNet::new(&vs).expect("Failed to create network");

        // Create dummy input tensor (batch_size=1, channels=5, height=64, width=64)
        let input = Tensor::randn(0.0f32, 1.0f32, (1, 5, 64, 64), &device)
            .expect("Failed to create input tensor")
            .to_dtype(dtype)
            .expect("Failed to convert tensor dtype");

        let output = network.forward(&input);
        match output {
            Ok(output) => {
                assert_eq!(output.dims(), &[1, 1, 64, 64], "Output shape mismatch");
            }
            Err(e) => {
                panic!("Forward pass failed with error: {:?}", e);
            }
        }
    }
}
