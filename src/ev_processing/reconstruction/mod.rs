// Event-based reconstruction module
// Tools for reconstructing frames from event data

pub mod convlstm;
pub mod e2vid;
pub mod e2vid_arch;
pub mod e2vid_plus;
pub mod gpu_utils;
pub mod metrics;
pub mod onnx_loader_simple;
pub mod python;
pub mod python_temporal;
pub mod pytorch_loader;
pub mod spade;
pub mod spade_e2vid;
pub mod ssl_e2vid;
pub mod ssl_losses;

// Re-export main items for easier access
pub use convlstm::{BiConvLSTM, ConvLSTM, ConvLSTMCell, MergeMode};
pub use e2vid::E2Vid;
pub use e2vid_arch::{E2VidUNet, FireNet};
pub use e2vid_plus::{E2VidPlus, FireNetPlus};
pub use gpu_utils::{get_best_device, get_gpu_type, is_gpu_available, GpuOptimizationHints};
pub use metrics::{ms_ssim, mse, psnr, ssim, ReconstructionMetrics};
pub use onnx_loader_simple::{ModelConverter, OnnxE2VidModel, OnnxModelConfig};
pub use pytorch_loader::{E2VidModelLoader, E2VidNet, LoadedModel, ModelLoaderConfig};
pub use spade::{SpadeGenerator, SpadeNorm, SpadeResBlock};
pub use spade_e2vid::{HybridSpadeE2Vid, SpadeE2Vid, SpadeE2VidLite};
pub use ssl_e2vid::{EventAugmentation, SslE2Vid, SslE2VidMomentum, SslTrainer};
pub use ssl_losses::{
    ContrastiveLoss, EventReconstructionLoss, PhotometricLoss, SSLLoss, SSLLossComponents,
    SSLLossConfig, TemporalConsistencyLoss,
};
