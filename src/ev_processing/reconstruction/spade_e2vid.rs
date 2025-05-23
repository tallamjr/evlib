// SPADE-E2VID implementation for spatially-adaptive event reconstruction
// Combines E2VID architecture with SPADE normalization for better detail preservation

use super::spade::{SpadeNorm, SpadeResBlock};
use candle_core::{Module, Result, Tensor};
use candle_nn::{batch_norm, conv2d, BatchNorm, Conv2d, Conv2dConfig, ModuleT, VarBuilder};

/// SPADE-E2VID architecture
/// Uses spatially-adaptive normalization to preserve fine details in reconstruction
pub struct SpadeE2Vid {
    // Encoder path (standard convolutions)
    enc_conv1: Conv2d,
    enc_bn1: BatchNorm,
    enc_conv2: Conv2d,
    enc_bn2: BatchNorm,
    enc_conv3: Conv2d,
    enc_bn3: BatchNorm,
    enc_conv4: Conv2d,
    enc_bn4: BatchNorm,

    // Bottleneck
    bottleneck_conv: Conv2d,

    // Decoder with SPADE normalization
    dec_spade4: SpadeNorm,
    dec_conv4: Conv2d,
    dec_spade3: SpadeNorm,
    dec_conv3: Conv2d,
    dec_spade2: SpadeNorm,
    dec_conv2: Conv2d,
    dec_spade1: SpadeNorm,
    dec_conv1: Conv2d,

    // Output projection
    output_conv: Conv2d,

    // Configuration
    use_skip_connections: bool,
}

impl SpadeE2Vid {
    /// Create a new SPADE-E2VID model
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        base_channels: usize,
        use_skip_connections: bool,
    ) -> Result<Self> {
        let c1 = base_channels;
        let c2 = base_channels * 2;
        let c3 = base_channels * 4;
        let c4 = base_channels * 8;

        // Encoder (standard convolutions with batch norm)
        let enc_conv1 = conv2d(
            in_channels,
            c1,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.pp("enc_conv1"),
        )?;
        let enc_bn1 = batch_norm(c1, 1e-5, vb.pp("enc_bn1"))?;

        let enc_conv2 = conv2d(
            c1,
            c2,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.pp("enc_conv2"),
        )?;
        let enc_bn2 = batch_norm(c2, 1e-5, vb.pp("enc_bn2"))?;

        let enc_conv3 = conv2d(
            c2,
            c3,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.pp("enc_conv3"),
        )?;
        let enc_bn3 = batch_norm(c3, 1e-5, vb.pp("enc_bn3"))?;

        let enc_conv4 = conv2d(
            c3,
            c4,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.pp("enc_conv4"),
        )?;
        let enc_bn4 = batch_norm(c4, 1e-5, vb.pp("enc_bn4"))?;

        // Bottleneck
        let bottleneck_conv = conv2d(
            c4,
            c4,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("bottleneck"),
        )?;

        // Decoder with SPADE normalization
        // SPADE uses the input voxel grid as conditioning
        let dec_spade4 = SpadeNorm::new(vb.pp("dec_spade4"), c4, in_channels, Some(64), 3)?;
        let dec_conv4 = conv2d(
            c4 * 2,
            c3,
            3, // *2 for skip connection
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("dec_conv4"),
        )?;

        let dec_spade3 = SpadeNorm::new(vb.pp("dec_spade3"), c3, in_channels, Some(64), 3)?;
        let dec_conv3 = conv2d(
            c3 * 2,
            c2,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("dec_conv3"),
        )?;

        let dec_spade2 = SpadeNorm::new(vb.pp("dec_spade2"), c2, in_channels, Some(64), 3)?;
        let dec_conv2 = conv2d(
            c2 * 2,
            c1,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("dec_conv2"),
        )?;

        let dec_spade1 = SpadeNorm::new(vb.pp("dec_spade1"), c1, in_channels, Some(64), 3)?;
        let dec_conv1 = conv2d(
            c1,
            c1,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("dec_conv1"),
        )?;

        // Output
        let output_conv = conv2d(c1, 1, 1, Conv2dConfig::default(), vb.pp("output"))?;

        Ok(Self {
            enc_conv1,
            enc_bn1,
            enc_conv2,
            enc_bn2,
            enc_conv3,
            enc_bn3,
            enc_conv4,
            enc_bn4,
            bottleneck_conv,
            dec_spade4,
            dec_conv4,
            dec_spade3,
            dec_conv3,
            dec_spade2,
            dec_conv2,
            dec_spade1,
            dec_conv1,
            output_conv,
            use_skip_connections,
        })
    }

    /// Forward pass through SPADE-E2VID
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Store input for SPADE conditioning
        let input_conditioning = x;

        // Encoder path
        let e1 = self
            .enc_bn1
            .forward_t(&self.enc_conv1.forward(x)?, false)?
            .relu()?;
        let e2 = self
            .enc_bn2
            .forward_t(&self.enc_conv2.forward(&e1)?, false)?
            .relu()?;
        let e3 = self
            .enc_bn3
            .forward_t(&self.enc_conv3.forward(&e2)?, false)?
            .relu()?;
        let e4 = self
            .enc_bn4
            .forward_t(&self.enc_conv4.forward(&e3)?, false)?
            .relu()?;

        // Bottleneck
        let b = self.bottleneck_conv.forward(&e4)?.relu()?;

        // Decoder with SPADE normalization and skip connections
        // Upsample and apply SPADE conditioning from input
        let d4 = b.upsample_nearest2d(e4.dims()[2], e4.dims()[3])?;
        let d4 = self.dec_spade4.forward(&d4, input_conditioning)?;
        let d4 = d4.relu()?;
        let d4 = if self.use_skip_connections {
            Tensor::cat(&[&d4, &e4], 1)?
        } else {
            d4
        };
        let d4 = self.dec_conv4.forward(&d4)?;

        let d3 = d4.upsample_nearest2d(e3.dims()[2], e3.dims()[3])?;
        let d3 = self.dec_spade3.forward(&d3, input_conditioning)?;
        let d3 = d3.relu()?;
        let d3 = if self.use_skip_connections {
            Tensor::cat(&[&d3, &e3], 1)?
        } else {
            d3
        };
        let d3 = self.dec_conv3.forward(&d3)?;

        let d2 = d3.upsample_nearest2d(e2.dims()[2], e2.dims()[3])?;
        let d2 = self.dec_spade2.forward(&d2, input_conditioning)?;
        let d2 = d2.relu()?;
        let d2 = if self.use_skip_connections {
            Tensor::cat(&[&d2, &e2], 1)?
        } else {
            d2
        };
        let d2 = self.dec_conv2.forward(&d2)?;

        let d1 = d2.upsample_nearest2d(e1.dims()[2], e1.dims()[3])?;
        let d1 = self.dec_spade1.forward(&d1, input_conditioning)?;
        let d1 = d1.relu()?;
        let d1 = self.dec_conv1.forward(&d1)?;

        // Final upsampling to input resolution
        let out = d1.upsample_nearest2d(x.dims()[2], x.dims()[3])?;
        let out = self.output_conv.forward(&out)?;

        // Sigmoid activation for output
        candle_nn::ops::sigmoid(&out)
    }
}

/// Hybrid SPADE-E2VID with both standard and SPADE paths
/// Allows learning when to use spatial adaptation
pub struct HybridSpadeE2Vid {
    // Standard E2VID decoder path
    _standard_dec4: Conv2d,
    _standard_bn4: BatchNorm,
    _standard_dec3: Conv2d,
    _standard_bn3: BatchNorm,
    _standard_dec2: Conv2d,
    _standard_bn2: BatchNorm,
    _standard_dec1: Conv2d,
    _standard_bn1: BatchNorm,

    // SPADE decoder path
    spade_model: SpadeE2Vid,

    // Gating network to blend paths
    gate_conv: Conv2d,
}

impl HybridSpadeE2Vid {
    /// Create hybrid model with learnable path blending
    pub fn new(vb: VarBuilder, in_channels: usize, base_channels: usize) -> Result<Self> {
        let c1 = base_channels;
        let c2 = base_channels * 2;
        let c3 = base_channels * 4;
        let c4 = base_channels * 8;

        // Standard decoder path
        let standard_dec4 = conv2d(
            c4,
            c3,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("std_dec4"),
        )?;
        let standard_bn4 = batch_norm(c3, 1e-5, vb.pp("std_bn4"))?;

        let standard_dec3 = conv2d(
            c3,
            c2,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("std_dec3"),
        )?;
        let standard_bn3 = batch_norm(c2, 1e-5, vb.pp("std_bn3"))?;

        let standard_dec2 = conv2d(
            c2,
            c1,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("std_dec2"),
        )?;
        let standard_bn2 = batch_norm(c1, 1e-5, vb.pp("std_bn2"))?;

        let standard_dec1 = conv2d(
            c1,
            c1,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("std_dec1"),
        )?;
        let standard_bn1 = batch_norm(c1, 1e-5, vb.pp("std_bn1"))?;

        // SPADE model
        let spade_model = SpadeE2Vid::new(vb.pp("spade"), in_channels, base_channels, true)?;

        // Gating network
        let gate_conv = conv2d(in_channels, 1, 1, Conv2dConfig::default(), vb.pp("gate"))?;

        Ok(Self {
            _standard_dec4: standard_dec4,
            _standard_bn4: standard_bn4,
            _standard_dec3: standard_dec3,
            _standard_bn3: standard_bn3,
            _standard_dec2: standard_dec2,
            _standard_bn2: standard_bn2,
            _standard_dec1: standard_dec1,
            _standard_bn1: standard_bn1,
            spade_model,
            gate_conv,
        })
    }

    /// Forward pass with adaptive blending
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute gating weight based on input
        let gate = candle_nn::ops::sigmoid(&self.gate_conv.forward(x)?)?;

        // SPADE path
        let spade_output = self.spade_model.forward(x)?;

        // For standard path, we'd need the encoder features
        // This is simplified - in practice, would share encoder
        let standard_output = spade_output.clone(); // Placeholder

        // Blend outputs based on learned gate
        let one_minus_gate = (1.0 - gate.clone())?;
        (gate * spade_output)? + (one_minus_gate * standard_output)?
    }
}

/// Lightweight SPADE variant using only in decoder bottleneck
pub struct SpadeE2VidLite {
    // Standard E2VID encoder
    encoder: E2VidEncoder,
    // Bottleneck with SPADE
    bottleneck_spade: SpadeResBlock,
    // Standard decoder
    decoder: E2VidDecoder,
}

/// Standard E2VID encoder (shared component)
struct E2VidEncoder {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    conv3: Conv2d,
    bn3: BatchNorm,
    conv4: Conv2d,
    bn4: BatchNorm,
}

impl E2VidEncoder {
    fn new(vb: VarBuilder, in_channels: usize, base_channels: usize) -> Result<Self> {
        let c1 = base_channels;
        let c2 = base_channels * 2;
        let c3 = base_channels * 4;
        let c4 = base_channels * 8;

        Ok(Self {
            conv1: conv2d(
                in_channels,
                c1,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("conv1"),
            )?,
            bn1: batch_norm(c1, 1e-5, vb.pp("bn1"))?,
            conv2: conv2d(
                c1,
                c2,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("conv2"),
            )?,
            bn2: batch_norm(c2, 1e-5, vb.pp("bn2"))?,
            conv3: conv2d(
                c2,
                c3,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("conv3"),
            )?,
            bn3: batch_norm(c3, 1e-5, vb.pp("bn3"))?,
            conv4: conv2d(
                c3,
                c4,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("conv4"),
            )?,
            bn4: batch_norm(c4, 1e-5, vb.pp("bn4"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let e1 = self.bn1.forward_t(&self.conv1.forward(x)?, false)?.relu()?;
        let e2 = self
            .bn2
            .forward_t(&self.conv2.forward(&e1)?, false)?
            .relu()?;
        let e3 = self
            .bn3
            .forward_t(&self.conv3.forward(&e2)?, false)?
            .relu()?;
        let e4 = self
            .bn4
            .forward_t(&self.conv4.forward(&e3)?, false)?
            .relu()?;
        Ok(vec![e1, e2, e3, e4])
    }
}

/// Standard E2VID decoder
struct E2VidDecoder {
    conv4: Conv2d,
    bn4: BatchNorm,
    conv3: Conv2d,
    bn3: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    conv1: Conv2d,
    bn1: BatchNorm,
    output_conv: Conv2d,
}

impl E2VidDecoder {
    fn new(vb: VarBuilder, base_channels: usize) -> Result<Self> {
        let c1 = base_channels;
        let c2 = base_channels * 2;
        let c3 = base_channels * 4;
        let c4 = base_channels * 8;

        Ok(Self {
            conv4: conv2d(
                c4 * 2,
                c3,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("conv4"),
            )?,
            bn4: batch_norm(c3, 1e-5, vb.pp("bn4"))?,
            conv3: conv2d(
                c3 * 2,
                c2,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("conv3"),
            )?,
            bn3: batch_norm(c2, 1e-5, vb.pp("bn3"))?,
            conv2: conv2d(
                c2 * 2,
                c1,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("conv2"),
            )?,
            bn2: batch_norm(c1, 1e-5, vb.pp("bn2"))?,
            conv1: conv2d(
                c1,
                c1,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("conv1"),
            )?,
            bn1: batch_norm(c1, 1e-5, vb.pp("bn1"))?,
            output_conv: conv2d(c1, 1, 1, Conv2dConfig::default(), vb.pp("output"))?,
        })
    }

    fn forward(&self, x: &Tensor, skip_connections: &[Tensor]) -> Result<Tensor> {
        // Decoder with skip connections
        let d4 =
            x.upsample_nearest2d(skip_connections[3].dims()[2], skip_connections[3].dims()[3])?;
        let d4 = Tensor::cat(&[&d4, &skip_connections[3]], 1)?;
        let d4 = self
            .bn4
            .forward_t(&self.conv4.forward(&d4)?, false)?
            .relu()?;

        let d3 =
            d4.upsample_nearest2d(skip_connections[2].dims()[2], skip_connections[2].dims()[3])?;
        let d3 = Tensor::cat(&[&d3, &skip_connections[2]], 1)?;
        let d3 = self
            .bn3
            .forward_t(&self.conv3.forward(&d3)?, false)?
            .relu()?;

        let d2 =
            d3.upsample_nearest2d(skip_connections[1].dims()[2], skip_connections[1].dims()[3])?;
        let d2 = Tensor::cat(&[&d2, &skip_connections[1]], 1)?;
        let d2 = self
            .bn2
            .forward_t(&self.conv2.forward(&d2)?, false)?
            .relu()?;

        let d1 =
            d2.upsample_nearest2d(skip_connections[0].dims()[2], skip_connections[0].dims()[3])?;
        let d1 = self
            .bn1
            .forward_t(&self.conv1.forward(&d1)?, false)?
            .relu()?;

        // Output
        let out = self.output_conv.forward(&d1)?;
        candle_nn::ops::sigmoid(&out)
    }
}

impl SpadeE2VidLite {
    pub fn new(vb: VarBuilder, in_channels: usize, base_channels: usize) -> Result<Self> {
        let encoder = E2VidEncoder::new(vb.pp("encoder"), in_channels, base_channels)?;
        let bottleneck_spade = SpadeResBlock::new(
            vb.pp("bottleneck_spade"),
            base_channels * 8,
            base_channels * 8,
            in_channels,
            Some(64),
        )?;
        let decoder = E2VidDecoder::new(vb.pp("decoder"), base_channels)?;

        Ok(Self {
            encoder,
            bottleneck_spade,
            decoder,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let skip_connections = self.encoder.forward(x)?;
        let bottleneck = self.bottleneck_spade.forward(&skip_connections[3], x)?;
        let output = self.decoder.forward(&bottleneck, &skip_connections)?;
        // Ensure output matches input resolution
        output.upsample_nearest2d(x.dims()[2], x.dims()[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_spade_e2vid() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = SpadeE2Vid::new(vb, 5, 32, true).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, &[1, 5, 128, 128], &device).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1, 128, 128]);
    }

    #[test]
    fn test_spade_e2vid_lite() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = SpadeE2VidLite::new(vb, 5, 16).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, &[1, 5, 64, 64], &device).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 1, 64, 64]);
    }
}
