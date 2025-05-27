// Enhanced ONNX model loading for E2VID reconstruction
// Uses ONNX Runtime (ort) for efficient model inference with full feature support

use candle_core::{DType, Device, Tensor};
use ort::{
    inputs, session::builder::GraphOptimizationLevel, session::Session, value::Tensor as OrtTensor,
};
use std::collections::HashMap;
use std::path::Path;

/// Errors that can occur during ONNX model operations
#[derive(Debug)]
pub enum OnnxLoadError {
    IoError(std::io::Error),
    OrtError(String),
    CandleError(candle_core::Error),
    InvalidModelFormat(String),
    InferenceFailed(String),
}

impl From<std::io::Error> for OnnxLoadError {
    fn from(error: std::io::Error) -> Self {
        OnnxLoadError::IoError(error)
    }
}

impl From<candle_core::Error> for OnnxLoadError {
    fn from(error: candle_core::Error) -> Self {
        OnnxLoadError::CandleError(error)
    }
}

impl std::fmt::Display for OnnxLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxLoadError::IoError(e) => write!(f, "IO error: {}", e),
            OnnxLoadError::OrtError(e) => write!(f, "ONNX Runtime error: {}", e),
            OnnxLoadError::CandleError(e) => write!(f, "Candle error: {}", e),
            OnnxLoadError::InvalidModelFormat(msg) => write!(f, "Invalid model format: {}", msg),
            OnnxLoadError::InferenceFailed(msg) => write!(f, "Inference failed: {}", msg),
        }
    }
}

impl std::error::Error for OnnxLoadError {}

/// Configuration for ONNX model loading
#[derive(Debug)]
pub struct OnnxModelConfig {
    pub device: Device,
    pub dtype: DType,
    pub verbose: bool,
    pub use_gpu: bool,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
}

impl Default for OnnxModelConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
            verbose: false,
            use_gpu: false,
            intra_threads: Some(4),
            inter_threads: Some(1),
        }
    }
}

impl OnnxModelConfig {
    /// Create config optimized for CPU inference
    pub fn cpu_optimized() -> Self {
        Self {
            device: Device::Cpu,
            use_gpu: false,
            intra_threads: Some(num_cpus::get()),
            inter_threads: Some(1),
            ..Default::default()
        }
    }

    /// Create config optimized for GPU inference
    pub fn gpu_optimized() -> Self {
        Self {
            device: Device::Cpu, // Candle device for tensor operations
            use_gpu: true,       // ONNX Runtime GPU execution
            intra_threads: Some(1),
            inter_threads: Some(1),
            ..Default::default()
        }
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// ONNX model wrapper for E2VID inference
/// Uses ONNX Runtime for efficient model inference
pub struct OnnxE2VidModel {
    config: OnnxModelConfig,
    model_path: std::path::PathBuf,
    session: Session,
    input_specs: Vec<InputSpec>,
    output_specs: Vec<OutputSpec>,
}

/// Input specification for ONNX model
#[derive(Debug, Clone)]
pub struct InputSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}

/// Output specification for ONNX model
#[derive(Debug, Clone)]
pub struct OutputSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}

impl OnnxE2VidModel {
    /// Load an ONNX model from file
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        config: OnnxModelConfig,
    ) -> Result<Self, OnnxLoadError> {
        if config.verbose {
            println!("Loading ONNX model from {:?}", path.as_ref());
        }

        // Check if file exists
        if !path.as_ref().exists() {
            return Err(OnnxLoadError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Model file not found: {:?}", path.as_ref()),
            )));
        }

        // Create session builder
        let mut session_builder = Session::builder().map_err(|e| {
            OnnxLoadError::OrtError(format!("Failed to create session builder: {}", e))
        })?;

        // Configure optimization
        session_builder = session_builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                OnnxLoadError::OrtError(format!("Failed to set optimization level: {}", e))
            })?;

        // Configure threading
        if let Some(intra_threads) = config.intra_threads {
            session_builder = session_builder
                .with_intra_threads(intra_threads)
                .map_err(|e| {
                    OnnxLoadError::OrtError(format!("Failed to set intra threads: {}", e))
                })?;
        }

        if let Some(inter_threads) = config.inter_threads {
            session_builder = session_builder
                .with_inter_threads(inter_threads)
                .map_err(|e| {
                    OnnxLoadError::OrtError(format!("Failed to set inter threads: {}", e))
                })?;
        }

        // Configure execution providers
        if config.use_gpu {
            if config.verbose {
                println!("GPU acceleration requested, attempting to enable CUDA...");
            }
            // Note: Specific execution provider configuration may vary by ort version
            // For now, rely on ort's automatic provider selection
            if config.verbose {
                println!("Using ort's automatic execution provider selection");
            }
        }

        // Load the model
        let session = session_builder
            .commit_from_file(path.as_ref())
            .map_err(|e| OnnxLoadError::OrtError(format!("Failed to load model: {}", e)))?;

        // Extract model metadata
        let input_specs = session
            .inputs
            .iter()
            .map(|input| InputSpec {
                name: input.name.clone(),
                shape: input
                    .input_type
                    .tensor_dimensions()
                    .unwrap_or(&vec![])
                    .clone(),
                dtype: format!("{:?}", input.input_type.tensor_type()),
            })
            .collect();

        let output_specs = session
            .outputs
            .iter()
            .map(|output| OutputSpec {
                name: output.name.clone(),
                shape: output
                    .output_type
                    .tensor_dimensions()
                    .unwrap_or(&vec![])
                    .clone(),
                dtype: format!("{:?}", output.output_type.tensor_type()),
            })
            .collect();

        if config.verbose {
            println!("Model loaded successfully:");
            println!("  Inputs: {:?}", input_specs);
            println!("  Outputs: {:?}", output_specs);
        }

        Ok(Self {
            config,
            model_path: path.as_ref().to_path_buf(),
            session,
            input_specs,
            output_specs,
        })
    }

    /// Perform inference on a batch of voxel grids
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, OnnxLoadError> {
        if self.config.verbose {
            println!(
                "Running ONNX inference with input shape: {:?}",
                input.dims()
            );
        }

        // Convert Candle tensor to ONNX format
        // First, get tensor as f32 vec
        let input_shape = input.dims();
        let input_data: Vec<f32> = input
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;

        // Create ONNX input tensor
        let input_tensor = OrtTensor::from_array((input_shape, input_data.into_boxed_slice()))
            .map_err(|e| {
                OnnxLoadError::OrtError(format!("Failed to create input tensor: {}", e))
            })?;

        // Get input/output names
        let input_name = &self.input_specs[0].name;
        let output_name = &self.output_specs[0].name;

        // Run inference
        let outputs =
            self.session
                .run(inputs![input_name => input_tensor].map_err(|e| {
                    OnnxLoadError::OrtError(format!("Failed to create inputs: {}", e))
                })?)
                .map_err(|e| OnnxLoadError::InferenceFailed(format!("Inference failed: {}", e)))?;

        // Extract output tensor
        let (output_shape_i64, output_data_slice) = outputs[output_name.as_str()]
            .try_extract_raw_tensor::<f32>()
            .map_err(|e| OnnxLoadError::OrtError(format!("Failed to extract output: {}", e)))?;

        // Convert back to Candle tensor
        let output_shape: Vec<usize> = output_shape_i64.iter().map(|&x| x as usize).collect();
        let output_data: Vec<f32> = output_data_slice.to_vec();

        let candle_output = Tensor::from_vec(output_data, output_shape, input.device())?
            .to_dtype(self.config.dtype)?;

        Ok(candle_output)
    }

    /// Get model metadata
    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            input_specs: self.input_specs.clone(),
            output_specs: self.output_specs.clone(),
            model_path: self.model_path.clone(),
        }
    }

    /// Get input specifications
    pub fn input_specs(&self) -> &[InputSpec] {
        &self.input_specs
    }

    /// Get output specifications
    pub fn output_specs(&self) -> &[OutputSpec] {
        &self.output_specs
    }

    /// Run inference with multiple inputs
    pub fn forward_multiple_inputs(
        &self,
        inputs: HashMap<String, &Tensor>,
    ) -> Result<HashMap<String, Tensor>, OnnxLoadError> {
        if self.config.verbose {
            println!("Running ONNX inference with {} inputs", inputs.len());
        }

        // Convert all inputs to ONNX format
        let mut onnx_inputs = HashMap::new();
        for (name, tensor) in inputs {
            let input_shape = tensor.dims();
            let input_data: Vec<f32> = tensor
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1()?;

            let input_tensor = OrtTensor::from_array((input_shape, input_data.into_boxed_slice()))
                .map_err(|e| {
                    OnnxLoadError::OrtError(format!(
                        "Failed to create input tensor {}: {}",
                        name, e
                    ))
                })?;

            onnx_inputs.insert(name.clone(), input_tensor);
        }

        // Run inference
        let outputs = self
            .session
            .run(onnx_inputs)
            .map_err(|e| OnnxLoadError::InferenceFailed(format!("Inference failed: {}", e)))?;

        // Convert all outputs back to Candle tensors
        let mut result_outputs = HashMap::new();
        for (name, output_tensor) in outputs {
            let (output_shape_i64, output_data_slice) =
                output_tensor.try_extract_raw_tensor::<f32>().map_err(|e| {
                    OnnxLoadError::OrtError(format!("Failed to extract output {}: {}", name, e))
                })?;

            let output_shape: Vec<usize> = output_shape_i64.iter().map(|&x| x as usize).collect();
            let output_data: Vec<f32> = output_data_slice.to_vec();

            let candle_output = Tensor::from_vec(output_data, output_shape, &self.config.device)?
                .to_dtype(self.config.dtype)?;

            result_outputs.insert(name.to_string(), candle_output);
        }

        Ok(result_outputs)
    }

    /// Validate input tensor against model specifications
    pub fn validate_input(
        &self,
        input: &Tensor,
        input_name: Option<&str>,
    ) -> Result<(), OnnxLoadError> {
        let input_name = input_name.unwrap_or(&self.input_specs[0].name);

        let spec = self
            .input_specs
            .iter()
            .find(|s| s.name == input_name)
            .ok_or_else(|| {
                OnnxLoadError::InvalidModelFormat(format!(
                    "Input '{}' not found in model",
                    input_name
                ))
            })?;

        let input_dims = input.dims();

        // Check dimensions (allowing for dynamic batch size)
        for (i, (&expected, &actual)) in spec.shape.iter().zip(input_dims.iter()).enumerate() {
            if expected != -1 && expected as usize != actual {
                return Err(OnnxLoadError::InvalidModelFormat(format!(
                    "Input dimension mismatch at index {}: expected {}, got {}",
                    i, expected, actual
                )));
            }
        }

        Ok(())
    }

    /// Get optimal batch size for inference
    pub fn get_optimal_batch_size(&self) -> usize {
        // Simple heuristic: use 1 for GPU, 4 for CPU
        if self.config.use_gpu {
            1
        } else {
            4
        }
    }

    /// Process input in batches for efficient inference
    pub fn forward_batched(
        &self,
        input: &Tensor,
        batch_size: Option<usize>,
    ) -> Result<Tensor, OnnxLoadError> {
        let batch_size = batch_size.unwrap_or_else(|| self.get_optimal_batch_size());
        let input_dims = input.dims();

        if input_dims[0] <= batch_size {
            // Small enough to process in one go
            return self.forward(input);
        }

        // Process in batches
        let mut results = Vec::new();
        let total_batches = input_dims[0].div_ceil(batch_size);

        for batch_idx in 0..total_batches {
            let start = batch_idx * batch_size;
            let end = std::cmp::min(start + batch_size, input_dims[0]);

            // Extract batch
            let batch = input.narrow(0, start, end - start)?;

            // Process batch
            let batch_result = self.forward(&batch)?;
            results.push(batch_result);

            if self.config.verbose && batch_idx % 10 == 0 {
                println!("Processed batch {}/{}", batch_idx + 1, total_batches);
            }
        }

        // Concatenate results
        let final_result = Tensor::cat(&results, 0)?;
        Ok(final_result)
    }
}

/// Model metadata information
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub input_specs: Vec<InputSpec>,
    pub output_specs: Vec<OutputSpec>,
    pub model_path: std::path::PathBuf,
}

/// Utility for converting PyTorch models to ONNX format
pub struct ModelConverter;

impl ModelConverter {
    /// Instructions for converting PyTorch E2VID model to ONNX
    pub fn pytorch_to_onnx_instructions() -> &'static str {
        r#"
To convert a PyTorch E2VID model to ONNX format:

1. In Python with PyTorch and the E2VID model:

```python
import torch
import torch.onnx

# Load your PyTorch E2VID model
model = load_e2vid_model()  # Your model loading code
model.eval()

# Create dummy input (batch_size=1, channels=5, height=256, width=256)
dummy_input = torch.randn(1, 5, 256, 256)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "e2vid_model.onnx",
    export_params=True,
    opset_version=17,  # Use latest stable opset
    do_constant_folding=True,
    input_names=['voxel_grid'],
    output_names=['reconstructed_frame'],
    dynamic_axes={
        'voxel_grid': {0: 'batch_size', 2: 'height', 3: 'width'},
        'reconstructed_frame': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)
```

2. Verify the exported model:

```python
import onnx
import onnxruntime as ort
import numpy as np

# Check the model
onnx_model = onnx.load("e2vid_model.onnx")
onnx.checker.check_model(onnx_model)

# Test inference with ONNX Runtime
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("e2vid_model.onnx", providers=providers)

# Print model info
print("Model inputs:")
for input in session.get_inputs():
    print(f"  {input.name}: {input.shape} ({input.type})")
print("Model outputs:")
for output in session.get_outputs():
    print(f"  {output.name}: {output.shape} ({output.type})")

# Test inference
test_input = np.random.randn(1, 5, 256, 256).astype(np.float32)
result = session.run(None, {"voxel_grid": test_input})
print("Output shape:", result[0].shape)
```

3. Use the ONNX model with evlib:

```rust
use evlib::processing::OnnxModelConfig;

// CPU-optimized configuration
let config = OnnxModelConfig::cpu_optimized().with_verbose(true);

// Or GPU-optimized configuration
let config = OnnxModelConfig::gpu_optimized().with_verbose(true);

// Load and use the model
let model = OnnxE2VidModel::load_from_file("e2vid_model.onnx", config)?;
let output = model.forward(&voxel_grid_tensor)?;
```

4. Optimization tips:

- Use opset version 17 for best compatibility
- Enable dynamic axes for flexible input sizes
- Test with different batch sizes for optimal performance
- Consider quantization for faster inference:

```python
# Optional: Quantize the model for faster inference
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "e2vid_model.onnx",
    "e2vid_model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```
"#
    }

    /// Load and validate an ONNX model
    pub fn load_and_validate<P: AsRef<Path>>(
        model_path: P,
        config: OnnxModelConfig,
    ) -> Result<OnnxE2VidModel, OnnxLoadError> {
        // Load the model
        let model = OnnxE2VidModel::load_from_file(model_path, config)?;

        // Basic validation
        if model.input_specs.is_empty() {
            return Err(OnnxLoadError::InvalidModelFormat(
                "Model has no inputs".to_string(),
            ));
        }

        if model.output_specs.is_empty() {
            return Err(OnnxLoadError::InvalidModelFormat(
                "Model has no outputs".to_string(),
            ));
        }

        // Check for common E2VID input structure
        let first_input = &model.input_specs[0];
        if first_input.shape.len() != 4 {
            eprintln!(
                "Warning: Expected 4D input (NCHW), got {}D",
                first_input.shape.len()
            );
        }

        Ok(model)
    }

    /// Benchmark model inference performance
    pub fn benchmark_model(
        model: &OnnxE2VidModel,
        input_shape: &[usize],
        num_iterations: usize,
    ) -> Result<(f64, f64), OnnxLoadError> {
        use std::time::Instant;

        // Create dummy input
        let dummy_data = vec![0.5f32; input_shape.iter().product()];
        let dummy_input = Tensor::from_vec(dummy_data, input_shape, &Device::Cpu)?;

        let mut durations = Vec::new();

        // Warmup
        for _ in 0..3 {
            let _ = model.forward(&dummy_input)?;
        }

        // Benchmark
        for _ in 0..num_iterations {
            let start = Instant::now();
            let _ = model.forward(&dummy_input)?;
            durations.push(start.elapsed().as_secs_f64());
        }

        let mean_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let fps = 1.0 / mean_duration;

        Ok((mean_duration, fps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_onnx_config_default() {
        let config = OnnxModelConfig::default();
        assert!(matches!(config.device, Device::Cpu));
        assert_eq!(config.dtype, DType::F32);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_onnx_config_optimizations() {
        let cpu_config = OnnxModelConfig::cpu_optimized();
        assert!(!cpu_config.use_gpu);
        assert!(cpu_config.intra_threads.unwrap() > 1);

        let gpu_config = OnnxModelConfig::gpu_optimized();
        assert!(gpu_config.use_gpu);
        assert_eq!(gpu_config.intra_threads, Some(1));

        let verbose_config = OnnxModelConfig::default().with_verbose(true);
        assert!(verbose_config.verbose);
    }

    #[test]
    fn test_model_converter_instructions() {
        let instructions = ModelConverter::pytorch_to_onnx_instructions();
        assert!(instructions.contains("torch.onnx.export"));
        assert!(instructions.contains("e2vid_model.onnx"));
        assert!(instructions.contains("opset_version=17"));
        assert!(instructions.contains("dynamic_axes"));
    }

    #[test]
    fn test_input_output_specs() {
        let input_spec = InputSpec {
            name: "input".to_string(),
            shape: vec![1, 5, 256, 256],
            dtype: "float32".to_string(),
        };

        let output_spec = OutputSpec {
            name: "output".to_string(),
            shape: vec![1, 1, 256, 256],
            dtype: "float32".to_string(),
        };

        assert_eq!(input_spec.name, "input");
        assert_eq!(input_spec.shape.len(), 4);
        assert_eq!(output_spec.name, "output");
    }

    #[test]
    fn test_model_metadata() {
        let input_spec = InputSpec {
            name: "voxel_grid".to_string(),
            shape: vec![-1, 5, -1, -1],
            dtype: "float32".to_string(),
        };

        let output_spec = OutputSpec {
            name: "reconstructed_frame".to_string(),
            shape: vec![-1, 1, -1, -1],
            dtype: "float32".to_string(),
        };

        let metadata = ModelMetadata {
            input_specs: vec![input_spec],
            output_specs: vec![output_spec],
            model_path: std::path::PathBuf::from("test.onnx"),
        };

        assert_eq!(metadata.input_specs.len(), 1);
        assert_eq!(metadata.output_specs.len(), 1);
        assert_eq!(metadata.input_specs[0].name, "voxel_grid");
    }

    // Note: We can't test actual ONNX model loading without a valid ONNX file
    // Integration tests should be added for real model loading scenarios
}
