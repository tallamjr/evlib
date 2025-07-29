// GPU utilities for event-to-video reconstruction
// Provides device selection and optimization for CUDA and Metal

use candle_core::{Device, Result};
use std::sync::Once;
use tracing::info;

static INIT: Once = Once::new();
static mut GPU_AVAILABLE: bool = false;
static mut GPU_TYPE: GpuType = GpuType::None;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuType {
    None,
    Cuda,
    Metal,
}

/// Initialize GPU detection (called once)
fn init_gpu_detection() {
    INIT.call_once(|| {
        unsafe {
            // Check for CUDA
            #[cfg(feature = "cuda")]
            {
                if let Ok(_) = Device::new_cuda(0) {
                    GPU_AVAILABLE = true;
                    GPU_TYPE = GpuType::Cuda;
                    return;
                }
            }

            // Check for Metal (macOS)
            #[cfg(feature = "metal")]
            {
                if let Ok(_) = Device::new_metal(0) {
                    GPU_AVAILABLE = true;
                    GPU_TYPE = GpuType::Metal;
                    return;
                }
            }

            // Default to CPU
            GPU_AVAILABLE = false;
            GPU_TYPE = GpuType::None;
        }
    });
}

/// Check if GPU is available
pub fn is_gpu_available() -> bool {
    init_gpu_detection();
    unsafe { GPU_AVAILABLE }
}

/// Get the type of GPU available
pub fn get_gpu_type() -> GpuType {
    init_gpu_detection();
    unsafe { GPU_TYPE }
}

/// Get the best available device for computation
pub fn get_best_device() -> Result<Device> {
    init_gpu_detection();

    unsafe {
        match GPU_TYPE {
            #[cfg(feature = "cuda")]
            GpuType::Cuda => Device::new_cuda(0),
            #[cfg(feature = "metal")]
            GpuType::Metal => Device::new_metal(0),
            _ => Ok(Device::Cpu),
        }
    }
}

/// Get device with preference
pub fn get_device_with_preference(prefer_gpu: bool) -> Result<Device> {
    if prefer_gpu {
        get_best_device()
    } else {
        Ok(Device::Cpu)
    }
}

/// Device-specific optimization hints
pub struct GpuOptimizationHints {
    pub batch_size: usize,
    pub use_mixed_precision: bool,
    pub use_tensor_cores: bool,
    pub preferred_memory_format: MemoryFormat,
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryFormat {
    Contiguous,
    ChannelsLast, // NHWC format, often faster on modern GPUs
}

impl GpuOptimizationHints {
    /// Get optimization hints for the current device
    pub fn for_device(device: &Device) -> Self {
        match device {
            Device::Cuda(_) => Self {
                batch_size: 8, // Larger batches for CUDA
                use_mixed_precision: true,
                use_tensor_cores: true,
                preferred_memory_format: MemoryFormat::ChannelsLast,
            },
            Device::Metal(_) => Self {
                batch_size: 4, // Moderate batches for Metal
                use_mixed_precision: true,
                use_tensor_cores: false,
                preferred_memory_format: MemoryFormat::Contiguous,
            },
            Device::Cpu => Self {
                batch_size: 1, // Small batches for CPU
                use_mixed_precision: false,
                use_tensor_cores: false,
                preferred_memory_format: MemoryFormat::Contiguous,
            },
        }
    }
}

/// Memory pool for efficient GPU memory management
pub struct GpuMemoryPool {
    device: Device,
    allocated_buffers: Vec<candle_core::Tensor>,
    max_buffers: usize,
}

impl GpuMemoryPool {
    pub fn new(device: Device, max_buffers: usize) -> Self {
        Self {
            device,
            allocated_buffers: Vec::new(),
            max_buffers,
        }
    }

    /// Get a buffer of specified shape, reusing if possible
    pub fn get_buffer(
        &mut self,
        shape: &[usize],
        dtype: candle_core::DType,
    ) -> Result<candle_core::Tensor> {
        // Check if we have a suitable buffer
        for (i, buffer) in self.allocated_buffers.iter().enumerate() {
            if buffer.dims() == shape && buffer.dtype() == dtype {
                // Reuse this buffer
                return Ok(self.allocated_buffers.remove(i));
            }
        }

        // Create new buffer
        candle_core::Tensor::zeros(shape, dtype, &self.device)
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer: candle_core::Tensor) {
        if self.allocated_buffers.len() < self.max_buffers {
            self.allocated_buffers.push(buffer);
        }
    }
}

/// Benchmark utilities for comparing CPU vs GPU performance
pub mod benchmark {
    use super::*;
    use std::time::Instant;

    pub struct BenchmarkResult {
        pub device_name: String,
        pub elapsed_ms: f64,
        pub throughput_fps: f64,
    }

    /// Benchmark a computation on different devices
    pub fn benchmark_devices<F>(
        computation: F,
        _input_shape: &[usize],
        n_iterations: usize,
    ) -> Vec<BenchmarkResult>
    where
        F: Fn(&Device) -> Result<()>,
    {
        let mut results = Vec::new();

        // Benchmark CPU
        let cpu_device = Device::Cpu;
        if let Ok(elapsed) = benchmark_on_device(&computation, &cpu_device, n_iterations) {
            results.push(BenchmarkResult {
                device_name: "CPU".to_string(),
                elapsed_ms: elapsed,
                throughput_fps: (n_iterations as f64) / (elapsed / 1000.0),
            });
        }

        // Benchmark GPU if available
        if let Ok(gpu_device) = get_best_device() {
            if !matches!(gpu_device, Device::Cpu) {
                let gpu_name = match get_gpu_type() {
                    GpuType::Cuda => "CUDA",
                    GpuType::Metal => "Metal",
                    _ => "GPU",
                };

                if let Ok(elapsed) = benchmark_on_device(&computation, &gpu_device, n_iterations) {
                    results.push(BenchmarkResult {
                        device_name: gpu_name.to_string(),
                        elapsed_ms: elapsed,
                        throughput_fps: (n_iterations as f64) / (elapsed / 1000.0),
                    });
                }
            }
        }

        results
    }

    fn benchmark_on_device<F>(computation: F, device: &Device, n_iterations: usize) -> Result<f64>
    where
        F: Fn(&Device) -> Result<()>,
    {
        // Warmup
        for _ in 0..3 {
            computation(device)?;
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..n_iterations {
            computation(device)?;
        }
        let elapsed = start.elapsed();

        Ok(elapsed.as_secs_f64() * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_detection() {
        let gpu_available = is_gpu_available();
        let gpu_type = get_gpu_type();

        info!(gpu_available = gpu_available, gpu_type = ?gpu_type, "GPU detection results");

        // Test should pass regardless of GPU availability
        assert!(true);
    }

    #[test]
    fn test_device_selection() {
        let device = get_best_device().unwrap();

        match device {
            Device::Cpu => info!("Using CPU"),
            Device::Cuda(_) => info!("Using CUDA GPU"),
            Device::Metal(_) => info!("Using Metal GPU"),
        }
    }

    #[test]
    fn test_optimization_hints() {
        let cpu_device = Device::Cpu;
        let cpu_hints = GpuOptimizationHints::for_device(&cpu_device);

        assert_eq!(cpu_hints.batch_size, 1);
        assert!(!cpu_hints.use_mixed_precision);
    }
}
