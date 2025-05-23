// SPADE (Spatially-Adaptive Normalization) implementation
// Based on "Semantic Image Synthesis with Spatially-Adaptive Normalization"
// Adapted for event-based reconstruction in SPADE-E2VID

use candle_core::{Module, Result, Tensor};
use candle_nn::{batch_norm, conv2d, BatchNorm, Conv2d, Conv2dConfig, ModuleT, VarBuilder};

/// SPADE normalization layer
/// Uses spatial information to modulate normalization parameters
pub struct SpadeNorm {
    /// Number of features to normalize
    _norm_channels: usize,
    /// Batch normalization layer
    batch_norm: BatchNorm,
    /// Convolutional layers for computing scale and bias
    shared_conv: Conv2d,
    scale_conv: Conv2d,
    bias_conv: Conv2d,
    /// Hidden dimension for modulation network
    _hidden_dim: usize,
}

impl SpadeNorm {
    /// Create a new SPADE normalization layer
    ///
    /// # Arguments
    /// * `vb` - Variable builder
    /// * `norm_channels` - Number of channels to normalize
    /// * `label_channels` - Number of channels in the conditioning input
    /// * `hidden_dim` - Hidden dimension for the modulation network
    /// * `kernel_size` - Kernel size for convolutions
    pub fn new(
        vb: VarBuilder,
        norm_channels: usize,
        label_channels: usize,
        hidden_dim: Option<usize>,
        kernel_size: usize,
    ) -> Result<Self> {
        let hidden_dim = hidden_dim.unwrap_or(128.min(norm_channels));
        let padding = kernel_size / 2;
        let conv_config = Conv2dConfig {
            padding,
            ..Default::default()
        };

        // Batch normalization without affine parameters (SPADE will provide them)
        let batch_norm = batch_norm(norm_channels, 1e-5, vb.pp("bn"))?;

        // Shared convolution for processing the conditioning input
        let shared_conv = conv2d(
            label_channels,
            hidden_dim,
            kernel_size,
            conv_config,
            vb.pp("shared"),
        )?;

        // Separate convolutions for scale (gamma) and bias (beta)
        let scale_conv = conv2d(
            hidden_dim,
            norm_channels,
            kernel_size,
            conv_config,
            vb.pp("scale"),
        )?;

        let bias_conv = conv2d(
            hidden_dim,
            norm_channels,
            kernel_size,
            conv_config,
            vb.pp("bias"),
        )?;

        Ok(Self {
            _norm_channels: norm_channels,
            batch_norm,
            shared_conv,
            scale_conv,
            bias_conv,
            _hidden_dim: hidden_dim,
        })
    }

    /// Forward pass through SPADE normalization
    ///
    /// # Arguments
    /// * `x` - Input features to normalize (B, C, H, W)
    /// * `segmap` - Conditioning input (B, C_label, H, W)
    pub fn forward(&self, x: &Tensor, segmap: &Tensor) -> Result<Tensor> {
        // Normalize input
        let normalized = self.batch_norm.forward_t(x, false)?;

        // Resize segmap to match x dimensions if needed
        let x_h = x.dims()[2];
        let x_w = x.dims()[3];
        let seg_h = segmap.dims()[2];
        let seg_w = segmap.dims()[3];

        let segmap = if seg_h != x_h || seg_w != x_w {
            segmap.upsample_nearest2d(x_h, x_w)?
        } else {
            segmap.clone()
        };

        // Process conditioning input
        let shared = self.shared_conv.forward(&segmap)?.relu()?;

        // Compute spatially-adaptive scale and bias
        let gamma = self.scale_conv.forward(&shared)?;
        let beta = self.bias_conv.forward(&shared)?;

        // Apply modulation: out = gamma * normalized + beta
        (normalized * gamma)? + beta
    }
}

/// SPADE ResBlock with spatially-adaptive normalization
pub struct SpadeResBlock {
    /// Learnable skip connection
    learned_shortcut: bool,
    /// Main path convolutions
    conv1: Conv2d,
    conv2: Conv2d,
    /// Skip connection convolution (if learnable)
    conv_skip: Option<Conv2d>,
    /// SPADE normalization layers
    spade1: SpadeNorm,
    spade2: SpadeNorm,
    spade_skip: Option<SpadeNorm>,
}

impl SpadeResBlock {
    /// Create a new SPADE ResBlock
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        label_channels: usize,
        hidden_dim: Option<usize>,
    ) -> Result<Self> {
        let learned_shortcut = in_channels != out_channels;
        let middle_channels = out_channels.min(in_channels);

        // Main path
        let conv1 = conv2d(
            in_channels,
            middle_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;

        let conv2 = conv2d(
            middle_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        // SPADE layers
        let spade1 = SpadeNorm::new(vb.pp("spade1"), in_channels, label_channels, hidden_dim, 3)?;

        let spade2 = SpadeNorm::new(
            vb.pp("spade2"),
            middle_channels,
            label_channels,
            hidden_dim,
            3,
        )?;

        // Skip connection
        let (conv_skip, spade_skip) = if learned_shortcut {
            let conv = conv2d(
                in_channels,
                out_channels,
                1,
                Conv2dConfig::default(),
                vb.pp("conv_skip"),
            )?;
            let spade = SpadeNorm::new(
                vb.pp("spade_skip"),
                in_channels,
                label_channels,
                hidden_dim,
                3,
            )?;
            (Some(conv), Some(spade))
        } else {
            (None, None)
        };

        Ok(Self {
            learned_shortcut,
            conv1,
            conv2,
            conv_skip,
            spade1,
            spade2,
            spade_skip,
        })
    }

    /// Forward pass through SPADE ResBlock
    pub fn forward(&self, x: &Tensor, segmap: &Tensor) -> Result<Tensor> {
        // Skip connection
        let x_skip = if self.learned_shortcut {
            let normalized = self.spade_skip.as_ref().unwrap().forward(x, segmap)?;
            self.conv_skip.as_ref().unwrap().forward(&normalized)?
        } else {
            x.clone()
        };

        // Main path
        let out = self.spade1.forward(x, segmap)?;
        let out = out.relu()?;
        let out = self.conv1.forward(&out)?;

        let out = self.spade2.forward(&out, segmap)?;
        let out = out.relu()?;
        let out = self.conv2.forward(&out)?;

        // Residual connection
        out + x_skip
    }
}

/// SPADE Generator for event-to-video reconstruction
pub struct SpadeGenerator {
    /// Initial projection
    fc: Conv2d,
    /// Upsampling blocks with SPADE normalization
    head_0: SpadeResBlock,
    up_0: SpadeResBlock,
    up_1: SpadeResBlock,
    up_2: SpadeResBlock,
    up_3: SpadeResBlock,
    /// Final output layer
    conv_img: Conv2d,
}

impl SpadeGenerator {
    /// Create a new SPADE generator
    ///
    /// # Arguments
    /// * `vb` - Variable builder
    /// * `label_channels` - Number of channels in conditioning input
    /// * `base_channels` - Base number of channels
    /// * `z_dim` - Dimension of noise input (optional)
    pub fn new(
        vb: VarBuilder,
        label_channels: usize,
        base_channels: usize,
        z_dim: Option<usize>,
    ) -> Result<Self> {
        let nf = base_channels;
        let input_channels = label_channels + z_dim.unwrap_or(0);

        // Initial projection
        let fc = conv2d(
            input_channels,
            16 * nf,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("fc"),
        )?;

        // Progressive upsampling with SPADE
        let head_0 = SpadeResBlock::new(vb.pp("head_0"), 16 * nf, 16 * nf, label_channels, None)?;
        let up_0 = SpadeResBlock::new(vb.pp("up_0"), 16 * nf, 8 * nf, label_channels, None)?;
        let up_1 = SpadeResBlock::new(vb.pp("up_1"), 8 * nf, 4 * nf, label_channels, None)?;
        let up_2 = SpadeResBlock::new(vb.pp("up_2"), 4 * nf, 2 * nf, label_channels, None)?;
        let up_3 = SpadeResBlock::new(vb.pp("up_3"), 2 * nf, nf, label_channels, None)?;

        // Output layer
        let conv_img = conv2d(
            nf,
            1, // Grayscale output for events
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_img"),
        )?;

        Ok(Self {
            fc,
            head_0,
            up_0,
            up_1,
            up_2,
            up_3,
            conv_img,
        })
    }

    /// Forward pass through generator
    ///
    /// # Arguments
    /// * `segmap` - Conditioning input (e.g., voxel grid)
    /// * `z` - Optional noise input
    pub fn forward(&self, segmap: &Tensor, z: Option<&Tensor>) -> Result<Tensor> {
        // Prepare input
        let input = if let Some(z) = z {
            // Broadcast z to spatial dimensions
            let batch_size = segmap.dims()[0];
            let h = segmap.dims()[2];
            let w = segmap.dims()[3];
            let z_expanded =
                z.unsqueeze(2)?
                    .unsqueeze(3)?
                    .expand(&[batch_size, z.dims()[1], h, w])?;
            Tensor::cat(&[segmap, &z_expanded], 1)?
        } else {
            segmap.clone()
        };

        // Initial features at lowest resolution
        let h = segmap.dims()[2];
        let w = segmap.dims()[3];
        let x = self
            .fc
            .forward(&input.upsample_nearest2d(h / 16, w / 16)?)?;

        // Progressive upsampling with SPADE modulation
        let x = self.head_0.forward(&x, segmap)?;

        let x = x.upsample_nearest2d(h / 8, w / 8)?;
        let x = self.up_0.forward(&x, segmap)?;

        let x = x.upsample_nearest2d(h / 4, w / 4)?;
        let x = self.up_1.forward(&x, segmap)?;

        let x = x.upsample_nearest2d(h / 2, w / 2)?;
        let x = self.up_2.forward(&x, segmap)?;

        let x = x.upsample_nearest2d(h, w)?;
        let x = self.up_3.forward(&x, segmap)?;

        // Final output
        let out = self.conv_img.forward(&x.relu()?)?;
        candle_nn::ops::sigmoid(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_spade_norm() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create SPADE normalization layer
        let spade = SpadeNorm::new(vb, 64, 5, Some(32), 3).unwrap();

        // Test forward pass
        let x = Tensor::randn(0.0f32, 1.0, &[2, 64, 32, 32], &device).unwrap();
        let segmap = Tensor::randn(0.0f32, 1.0, &[2, 5, 32, 32], &device).unwrap();

        let output = spade.forward(&x, &segmap).unwrap();
        assert_eq!(output.dims(), &[2, 64, 32, 32]);
    }

    #[test]
    fn test_spade_resblock() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create SPADE ResBlock
        let block = SpadeResBlock::new(vb, 64, 32, 5, None).unwrap();

        // Test forward pass
        let x = Tensor::randn(0.0f32, 1.0, &[1, 64, 16, 16], &device).unwrap();
        let segmap = Tensor::randn(0.0f32, 1.0, &[1, 5, 16, 16], &device).unwrap();

        let output = block.forward(&x, &segmap).unwrap();
        assert_eq!(output.dims(), &[1, 32, 16, 16]);
    }

    #[test]
    fn test_spade_generator() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create SPADE generator
        let generator = SpadeGenerator::new(vb, 5, 16, None).unwrap();

        // Test forward pass
        let segmap = Tensor::randn(0.0f32, 1.0, &[1, 5, 64, 64], &device).unwrap();
        let output = generator.forward(&segmap, None).unwrap();

        assert_eq!(output.dims(), &[1, 1, 64, 64]);
    }
}
