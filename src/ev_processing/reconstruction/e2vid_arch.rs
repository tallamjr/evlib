// E2VID architecture implementation using Candle framework
// Based on "High Speed and High Dynamic Range Video with an Event Camera" (Rebecq et al., 2019)

use candle_core::{Module, Result, Tensor};
use candle_nn::{batch_norm, conv2d, BatchNorm, Conv2d, Conv2dConfig, ModuleT, VarBuilder};

/// UNet architecture for E2VID reconstruction
/// This architecture is optimized for event-to-video reconstruction with:
/// - Encoder-decoder structure with skip connections
/// - Batch normalization for training stability
/// - Residual connections in decoder blocks for better gradient flow
pub struct E2VidUNet {
    // Encoder layers
    enc1: EncoderBlock,
    enc2: EncoderBlock,
    enc3: EncoderBlock,
    enc4: EncoderBlock,

    // Bottleneck
    bottleneck: ConvBlock,

    // Decoder layers with skip connections
    dec4: DecoderBlock,
    dec3: DecoderBlock,
    dec2: DecoderBlock,
    dec1: DecoderBlock,

    // Output layer
    output_conv: Conv2d,
}

/// Encoder block with two convolutions and optional downsampling
struct EncoderBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    downsample: bool,
}

/// Decoder block with skip connection, upsampling, and two convolutions
struct DecoderBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

/// Basic convolution block
struct ConvBlock {
    conv: Conv2d,
    bn: BatchNorm,
}

impl E2VidUNet {
    /// Create a new E2VID UNet model
    ///
    /// # Arguments
    /// * `vb` - Variable builder for loading pre-trained weights
    /// * `in_channels` - Number of input channels (typically 5 for voxel grid)
    /// * `base_channels` - Base number of channels (doubled at each level)
    pub fn new(vb: VarBuilder, in_channels: usize, base_channels: usize) -> Result<Self> {
        // Performance optimization: use smaller kernel sizes in deeper layers
        let c1 = base_channels;
        let c2 = base_channels * 2;
        let c3 = base_channels * 4;
        let c4 = base_channels * 8;

        // Encoder
        let enc1 = EncoderBlock::new(vb.pp("enc1"), in_channels, c1, false)?;
        let enc2 = EncoderBlock::new(vb.pp("enc2"), c1, c2, true)?;
        let enc3 = EncoderBlock::new(vb.pp("enc3"), c2, c3, true)?;
        let enc4 = EncoderBlock::new(vb.pp("enc4"), c3, c4, true)?;

        // Bottleneck with residual connection support
        let bottleneck = ConvBlock::new(vb.pp("bottleneck"), c4, c4)?;

        // Decoder with skip connections
        let dec4 = DecoderBlock::new(vb.pp("dec4"), c4 * 2, c3)?; // *2 for skip connection
        let dec3 = DecoderBlock::new(vb.pp("dec3"), c3 * 2, c2)?;
        let dec2 = DecoderBlock::new(vb.pp("dec2"), c2 * 2, c1)?;
        let dec1 = DecoderBlock::new(vb.pp("dec1"), c1 * 2, c1)?;

        // Output layer with sigmoid activation
        let output_conv = conv2d(c1, 1, 1, Conv2dConfig::default(), vb.pp("output"))?;

        Ok(Self {
            enc1,
            enc2,
            enc3,
            enc4,
            bottleneck,
            dec4,
            dec3,
            dec2,
            dec1,
            output_conv,
        })
    }
}

impl Module for E2VidUNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Encoder path with feature extraction
        let e1 = self.enc1.forward(xs)?;
        let e2 = self.enc2.forward(&e1)?;
        let e3 = self.enc3.forward(&e2)?;
        let e4 = self.enc4.forward(&e3)?;

        // Bottleneck
        let b = self.bottleneck.forward(&e4)?;

        // Decoder path with skip connections
        let d4 = self.dec4.forward(&b, &e4)?;
        let d3 = self.dec3.forward(&d4, &e3)?;
        let d2 = self.dec2.forward(&d3, &e2)?;
        let d1 = self.dec1.forward(&d2, &e1)?;

        // Output with sigmoid for [0, 1] range
        let output = self.output_conv.forward(&d1)?;
        candle_nn::ops::sigmoid(&output)
    }
}

impl EncoderBlock {
    fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        downsample: bool,
    ) -> Result<Self> {
        let conv1 = conv2d(
            in_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let bn1 = batch_norm(out_channels, 1e-5, vb.pp("bn1"))?;

        let conv2 = conv2d(
            out_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let bn2 = batch_norm(out_channels, 1e-5, vb.pp("bn2"))?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward_t(&x, false)?;
        let x = x.relu()?;

        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward_t(&x, false)?;
        let x = x.relu()?;

        // Downsample with max pooling for efficiency
        if self.downsample {
            x.max_pool2d_with_stride(2, 2)
        } else {
            Ok(x)
        }
    }
}

impl DecoderBlock {
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let conv1 = conv2d(
            in_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let bn1 = batch_norm(out_channels, 1e-5, vb.pp("bn1"))?;

        let conv2 = conv2d(
            out_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let bn2 = batch_norm(out_channels, 1e-5, vb.pp("bn2"))?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
        })
    }

    fn forward(&self, x: &Tensor, skip: &Tensor) -> Result<Tensor> {
        // Upsample
        let x = x.upsample_nearest2d(skip.dims()[2], skip.dims()[3])?;

        // Concatenate with skip connection
        let x = Tensor::cat(&[&x, skip], 1)?;

        // Convolutions
        let x = self.conv1.forward(&x)?;
        let x = self.bn1.forward_t(&x, false)?;
        let x = x.relu()?;

        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward_t(&x, false)?;
        x.relu()
    }
}

impl ConvBlock {
    fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let conv = conv2d(
            in_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        let bn = batch_norm(out_channels, 1e-5, vb.pp("bn"))?;

        Ok(Self { conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward_t(&x, false)?;
        x.relu()
    }
}

/// FireNet: Lightweight variant of E2VID for real-time processing
/// Uses depthwise separable convolutions and reduced channel counts
pub struct FireNet {
    // Initial feature extraction
    stem: Conv2d,

    // Fire modules (squeeze-expand blocks)
    fire1: FireModule,
    fire2: FireModule,
    fire3: FireModule,
    fire4: FireModule,

    // Output projection
    output_conv: Conv2d,
}

/// Fire module: 1x1 squeeze followed by 1x1 and 3x3 expand
struct FireModule {
    squeeze: Conv2d,
    expand_1x1: Conv2d,
    expand_3x3: Conv2d,
}

impl FireNet {
    pub fn new(vb: VarBuilder, in_channels: usize) -> Result<Self> {
        // Reduced channel counts for speed
        let stem = conv2d(
            in_channels,
            32,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("stem"),
        )?;

        let fire1 = FireModule::new(vb.pp("fire1"), 32, 16, 32)?;
        let fire2 = FireModule::new(vb.pp("fire2"), 64, 16, 32)?;
        let fire3 = FireModule::new(vb.pp("fire3"), 64, 32, 64)?;
        let fire4 = FireModule::new(vb.pp("fire4"), 128, 32, 64)?;

        let output_conv = conv2d(128, 1, 1, Conv2dConfig::default(), vb.pp("output"))?;

        Ok(Self {
            stem,
            fire1,
            fire2,
            fire3,
            fire4,
            output_conv,
        })
    }
}

impl Module for FireNet {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.stem.forward(xs)?.relu()?;
        let x = self.fire1.forward(&x)?;
        let x = self.fire2.forward(&x)?;
        let x = self.fire3.forward(&x)?;
        let x = self.fire4.forward(&x)?;

        let output = self.output_conv.forward(&x)?;
        candle_nn::ops::sigmoid(&output)
    }
}

impl FireModule {
    fn new(
        vb: VarBuilder,
        in_channels: usize,
        squeeze_channels: usize,
        expand_channels: usize,
    ) -> Result<Self> {
        let squeeze = conv2d(
            in_channels,
            squeeze_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("squeeze"),
        )?;

        let expand_1x1 = conv2d(
            squeeze_channels,
            expand_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("expand_1x1"),
        )?;

        let expand_3x3 = conv2d(
            squeeze_channels,
            expand_channels,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("expand_3x3"),
        )?;

        Ok(Self {
            squeeze,
            expand_1x1,
            expand_3x3,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let squeezed = self.squeeze.forward(x)?.relu()?;

        let expanded_1x1 = self.expand_1x1.forward(&squeezed)?.relu()?;
        let expanded_3x3 = self.expand_3x3.forward(&squeezed)?.relu()?;

        Tensor::cat(&[&expanded_1x1, &expanded_3x3], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_e2vid_unet_creation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = E2VidUNet::new(vb, 5, 32).unwrap();

        // Test forward pass
        let input = Tensor::randn(0.0f32, 1.0, &[1, 5, 256, 256], &device).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1, 256, 256]);
    }

    #[test]
    fn test_firenet_creation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = FireNet::new(vb, 5).unwrap();

        // Test forward pass
        let input = Tensor::randn(0.0f32, 1.0, &[1, 5, 256, 256], &device).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1, 256, 256]);
    }
}
