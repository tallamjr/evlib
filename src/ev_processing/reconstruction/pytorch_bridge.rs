// PyTorch Bridge - Load PyTorch .pth files into Candle models
// Uses PyO3 to interface with Python's torch library

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::VarMap;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Error types for PyTorch bridge operations
#[derive(Debug)]
pub enum PyTorchBridgeError {
    PythonError(String),
    TensorConversionError(String),
    ModelMappingError(String),
    FileNotFound(String),
}

impl std::fmt::Display for PyTorchBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PyTorchBridgeError::PythonError(msg) => write!(f, "Python error: {}", msg),
            PyTorchBridgeError::TensorConversionError(msg) => {
                write!(f, "Tensor conversion error: {}", msg)
            }
            PyTorchBridgeError::ModelMappingError(msg) => write!(f, "Model mapping error: {}", msg),
            PyTorchBridgeError::FileNotFound(msg) => write!(f, "File not found: {}", msg),
        }
    }
}

impl std::error::Error for PyTorchBridgeError {}

impl From<PyErr> for PyTorchBridgeError {
    fn from(err: PyErr) -> Self {
        Python::with_gil(|py| PyTorchBridgeError::PythonError(err.value(py).to_string()))
    }
}

/// PyTorch model loader that interfaces with Python
pub struct PyTorchLoader {
    device: Device,
}

impl PyTorchLoader {
    /// Create a new PyTorch loader
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Load a PyTorch checkpoint file and extract state dict
    pub fn load_checkpoint(
        &self,
        path: &Path,
    ) -> Result<HashMap<String, Tensor>, PyTorchBridgeError> {
        if !path.exists() {
            return Err(PyTorchBridgeError::FileNotFound(
                path.to_string_lossy().to_string(),
            ));
        }

        Python::with_gil(|py| {
            // Import torch
            let torch = py
                .import("torch")
                .map_err(|e| PyTorchBridgeError::PythonError(e.to_string()))?;

            // Load the checkpoint
            let checkpoint = torch
                .call_method1("load", (path.to_string_lossy().to_string(), "cpu"))
                .map_err(|e| PyTorchBridgeError::PythonError(e.to_string()))?;

            // Extract state dict - handle nested structure
            let state_dict = if checkpoint.contains("state_dict")? {
                let nested_state_dict = checkpoint
                    .get_item("state_dict")
                    .map_err(|e| PyTorchBridgeError::PythonError(e.to_string()))?;

                println!("Found nested state_dict, extracting model weights...");
                nested_state_dict
            } else {
                // Assume checkpoint is already a state dict
                println!("Treating checkpoint as direct state dict...");
                checkpoint
            };

            // Convert state dict to HashMap of Candle tensors
            self.convert_state_dict(py, state_dict)
        })
    }

    /// Convert PyTorch state dict to Candle tensors
    fn convert_state_dict(
        &self,
        py: Python,
        state_dict: &PyAny,
    ) -> Result<HashMap<String, Tensor>, PyTorchBridgeError> {
        let mut candle_state_dict = HashMap::new();

        // Get all keys from state dict
        let keys = state_dict
            .call_method0("keys")
            .map_err(|e| PyTorchBridgeError::PythonError(e.to_string()))?;

        println!(
            "Processing state dict with {} keys",
            keys.len().unwrap_or(0)
        );

        for key in keys.iter()? {
            let key = key?;
            let key_str = key
                .extract::<String>()
                .map_err(|e| PyTorchBridgeError::PythonError(e.to_string()))?;

            // Get the value
            let value = state_dict
                .get_item(key)
                .map_err(|e| PyTorchBridgeError::PythonError(e.to_string()))?;

            // Check if it's a tensor by looking for tensor-specific attributes
            if value.hasattr("detach").unwrap_or(false) && value.hasattr("shape").unwrap_or(false) {
                // Convert to Candle tensor
                match self.pytorch_to_candle(py, value) {
                    Ok(candle_tensor) => {
                        println!("Converted tensor: {}", key_str);
                        candle_state_dict.insert(key_str, candle_tensor);
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to convert tensor for key '{}': {}",
                            key_str, e
                        );
                        // Skip this key rather than failing entirely
                    }
                }
            } else {
                // Skip non-tensor values (e.g., strings, metadata)
                eprintln!("Skipping non-tensor key: {}", key_str);
            }
        }

        Ok(candle_state_dict)
    }

    /// Convert a single PyTorch tensor to Candle tensor
    fn pytorch_to_candle(&self, _py: Python, tensor: &PyAny) -> Result<Tensor, PyTorchBridgeError> {
        // Convert tensor to numpy array
        let numpy_array = tensor
            .call_method0("detach")
            .and_then(|t| t.call_method0("cpu"))
            .and_then(|t| t.call_method0("numpy"))
            .map_err(|e| PyTorchBridgeError::TensorConversionError(e.to_string()))?;

        // Get shape
        let shape_tuple = numpy_array
            .getattr("shape")
            .map_err(|e| PyTorchBridgeError::TensorConversionError(e.to_string()))?;

        let shape: Vec<usize> = shape_tuple
            .extract()
            .map_err(|e| PyTorchBridgeError::TensorConversionError(e.to_string()))?;

        // Get dtype
        let dtype_str = numpy_array
            .getattr("dtype")
            .and_then(|d| d.getattr("name"))
            .and_then(|n| n.extract::<String>())
            .map_err(|e| PyTorchBridgeError::TensorConversionError(e.to_string()))?;

        let candle_dtype = match dtype_str.as_str() {
            "float32" => DType::F32,
            "float64" => DType::F64,
            "int32" => DType::I64,
            "int64" => DType::I64,
            "uint8" => DType::U8,
            _ => {
                return Err(PyTorchBridgeError::TensorConversionError(format!(
                    "Unsupported dtype: {}",
                    dtype_str
                )))
            }
        };

        // Get data as bytes and convert
        let data_bytes = numpy_array
            .call_method0("tobytes")
            .map_err(|e| PyTorchBridgeError::TensorConversionError(e.to_string()))?;

        let bytes: &[u8] = data_bytes
            .extract()
            .map_err(|e| PyTorchBridgeError::TensorConversionError(e.to_string()))?;

        // Create Candle tensor from bytes
        match candle_dtype {
            DType::F32 => {
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Tensor::from_vec(data, shape, &self.device)
                    .map_err(|e| PyTorchBridgeError::TensorConversionError(e.to_string()))
            }
            DType::F64 => {
                let data: Vec<f64> = bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect();
                Tensor::from_vec(data, shape, &self.device)
                    .map_err(|e| PyTorchBridgeError::TensorConversionError(e.to_string()))
            }
            _ => Err(PyTorchBridgeError::TensorConversionError(
                "Dtype conversion not implemented".to_string(),
            )),
        }
    }
}

/// Model-specific weight mapping configurations
pub struct ModelWeightMapper {
    mappings: HashMap<String, String>,
}

impl ModelWeightMapper {
    /// Create mapper for E2VID UNet (handles both simple and recurrent versions)
    pub fn e2vid_unet() -> Self {
        let mut mappings = HashMap::new();

        // Head mappings
        mappings.insert(
            "unetrecurrent.head.conv2d.weight".to_string(),
            "head.0.weight".to_string(),
        );
        mappings.insert(
            "unetrecurrent.head.conv2d.bias".to_string(),
            "head.0.bias".to_string(),
        );

        // Encoder mappings - the actual E2VID uses ConvLSTM blocks
        for i in 0..3 {
            // Conv layers in encoders
            mappings.insert(
                format!("unetrecurrent.encoders.{}.conv.conv2d.weight", i),
                format!("encoders.{}.conv.weight", i),
            );
            mappings.insert(
                format!("unetrecurrent.encoders.{}.conv.norm_layer.weight", i),
                format!("encoders.{}.bn.weight", i),
            );
            mappings.insert(
                format!("unetrecurrent.encoders.{}.conv.norm_layer.bias", i),
                format!("encoders.{}.bn.bias", i),
            );

            // ConvLSTM gates
            mappings.insert(
                format!("unetrecurrent.encoders.{}.recurrent_block.Gates.weight", i),
                format!("encoders.{}.lstm.gates.weight", i),
            );
            mappings.insert(
                format!("unetrecurrent.encoders.{}.recurrent_block.Gates.bias", i),
                format!("encoders.{}.lstm.gates.bias", i),
            );
        }

        // Residual blocks in the middle
        for i in 0..2 {
            mappings.insert(
                format!("unetrecurrent.resblocks.{}.conv1.weight", i),
                format!("resblocks.{}.conv1.weight", i),
            );
            mappings.insert(
                format!("unetrecurrent.resblocks.{}.conv2.weight", i),
                format!("resblocks.{}.conv2.weight", i),
            );
        }

        // Decoder mappings
        for i in 0..3 {
            mappings.insert(
                format!("unetrecurrent.decoders.{}.conv.conv2d.weight", i),
                format!("decoders.{}.conv.weight", i),
            );
            mappings.insert(
                format!("unetrecurrent.decoders.{}.recurrent_block.Gates.weight", i),
                format!("decoders.{}.lstm.gates.weight", i),
            );
        }

        // Tail/output layer
        mappings.insert(
            "unetrecurrent.tail.conv2d.weight".to_string(),
            "tail.0.weight".to_string(),
        );
        mappings.insert(
            "unetrecurrent.tail.conv2d.bias".to_string(),
            "tail.0.bias".to_string(),
        );

        Self { mappings }
    }

    /// Create mapper for FireNet
    pub fn firenet() -> Self {
        let mut mappings = HashMap::new();

        // Head mappings
        mappings.insert("head.conv.weight".to_string(), "head.0.weight".to_string());
        mappings.insert("head.conv.bias".to_string(), "head.0.bias".to_string());

        // Add more mappings

        Self { mappings }
    }

    /// Create mapper for ET-Net
    pub fn et_net() -> Self {
        let mut mappings = HashMap::new();

        // Patch embedding
        mappings.insert(
            "patch_embed.proj.weight".to_string(),
            "patch_embed.proj.weight".to_string(),
        );
        mappings.insert(
            "patch_embed.proj.bias".to_string(),
            "patch_embed.proj.bias".to_string(),
        );

        // Transformer layers
        for i in 0..12 {
            let prefix = format!("blocks.{}", i);
            let candle_prefix = format!("layer_{}", i);

            // Self-attention
            mappings.insert(
                format!("{}.attn.qkv.weight", prefix),
                format!("{}.self_attn.q_proj.weight", candle_prefix),
            );

            // Add more transformer mappings
        }

        Self { mappings }
    }

    /// Map PyTorch state dict keys to Candle model keys
    pub fn map_state_dict(
        &self,
        pytorch_state_dict: HashMap<String, Tensor>,
    ) -> HashMap<String, Tensor> {
        let mut candle_state_dict = HashMap::new();

        for (pytorch_key, tensor) in pytorch_state_dict {
            if let Some(candle_key) = self.mappings.get(&pytorch_key) {
                candle_state_dict.insert(candle_key.clone(), tensor);
            } else {
                // Keep unmapped keys as-is (might be useful for debugging)
                eprintln!("Warning: No mapping found for key: {}", pytorch_key);
                candle_state_dict.insert(pytorch_key, tensor);
            }
        }

        candle_state_dict
    }
}

/// Load PyTorch weights into a Candle VarMap
pub fn load_pytorch_weights_into_varmap(
    checkpoint_path: &Path,
    model_type: &str,
    device: &Device,
) -> CandleResult<VarMap> {
    let loader = PyTorchLoader::new(device.clone());

    // Load checkpoint
    let pytorch_state_dict = loader
        .load_checkpoint(checkpoint_path)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    // Get appropriate mapper
    let mapper = match model_type {
        "e2vid_unet" => ModelWeightMapper::e2vid_unet(),
        "firenet" => ModelWeightMapper::firenet(),
        "et_net" => ModelWeightMapper::et_net(),
        _ => {
            return Err(candle_core::Error::Msg(format!(
                "Unknown model type: {}",
                model_type
            )))
        }
    };

    // Map state dict
    let candle_state_dict = mapper.map_state_dict(pytorch_state_dict);

    // Create VarMap and populate it
    let mut varmap = VarMap::new();
    for (key, tensor) in candle_state_dict {
        varmap.set_one(&key, tensor)?;
    }

    Ok(varmap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_weight_mapper_creation() {
        let mapper = ModelWeightMapper::e2vid_unet();
        assert!(!mapper.mappings.is_empty());

        let mapper = ModelWeightMapper::firenet();
        assert!(!mapper.mappings.is_empty());

        let mapper = ModelWeightMapper::et_net();
        assert!(!mapper.mappings.is_empty());
    }
}
