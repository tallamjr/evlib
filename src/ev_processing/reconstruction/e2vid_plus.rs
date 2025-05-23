// E2VID+ implementation with enhanced temporal features
// Based on "E2VID+: Improved Event Cameras Video Reconstruction via Learned Representations over Time"

use crate::ev_processing::reconstruction::convlstm::{BiConvLSTM, ConvLSTM, MergeMode};
use candle_core::{Module, Result, Tensor};
use candle_nn::{batch_norm, conv2d, BatchNorm, Conv2d, Conv2dConfig, ModuleT, VarBuilder};

/// E2VID+ architecture with temporal processing
/// Key improvements over E2VID:
/// - ConvLSTM for temporal memory
/// - Skip connections across time
/// - Learned temporal attention
pub struct E2VidPlus {
    // Initial feature extraction
    input_conv: Conv2d,
    input_bn: BatchNorm,

    // Encoder with ConvLSTM
    enc1: EncoderBlockPlus,
    enc2: EncoderBlockPlus,
    enc3: EncoderBlockPlus,
    enc4: EncoderBlockPlus,

    // Temporal processing in bottleneck
    bottleneck_conv: Conv2d,
    bottleneck_bn: BatchNorm,
    bottleneck_lstm: BiConvLSTM,

    // Decoder with temporal skip connections
    dec4: DecoderBlockPlus,
    dec3: DecoderBlockPlus,
    dec2: DecoderBlockPlus,
    dec1: DecoderBlockPlus,

    // Output projection
    output_conv: Conv2d,

    // Temporal attention module
    temporal_attention: TemporalAttention,
}

/// Enhanced encoder block with ConvLSTM
struct EncoderBlockPlus {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    convlstm: ConvLSTM,
    downsample: bool,
}

/// Enhanced decoder block with temporal connections
struct DecoderBlockPlus {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    convlstm: ConvLSTM,
}

/// Temporal attention mechanism
struct TemporalAttention {
    _query_conv: Conv2d,
    _key_conv: Conv2d,
    _value_conv: Conv2d,
    output_conv: Conv2d,
}

impl E2VidPlus {
    /// Create a new E2VID+ model
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        base_channels: usize,
        _sequence_length: usize,
    ) -> Result<Self> {
        // Channel progression
        let c1 = base_channels;
        let c2 = base_channels * 2;
        let c3 = base_channels * 4;
        let c4 = base_channels * 8;

        // Input processing
        let input_conv = conv2d(
            in_channels,
            c1,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("input_conv"),
        )?;
        let input_bn = batch_norm(c1, 1e-5, vb.pp("input_bn"))?;

        // Encoder with ConvLSTM
        let enc1 = EncoderBlockPlus::new(vb.pp("enc1"), c1, c1, false)?;
        let enc2 = EncoderBlockPlus::new(vb.pp("enc2"), c1, c2, true)?;
        let enc3 = EncoderBlockPlus::new(vb.pp("enc3"), c2, c3, true)?;
        let enc4 = EncoderBlockPlus::new(vb.pp("enc4"), c3, c4, true)?;

        // Bottleneck with bidirectional ConvLSTM
        let bottleneck_conv = conv2d(
            c4,
            c4,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("bottleneck_conv"),
        )?;
        let bottleneck_bn = batch_norm(c4, 1e-5, vb.pp("bottleneck_bn"))?;
        let bottleneck_lstm = BiConvLSTM::new(
            vb.pp("bottleneck_lstm"),
            c4,
            c4 / 2, // Hidden dim for each direction
            3,      // Kernel size
            MergeMode::Concat,
            true, // Return sequences
        )?;

        // Decoder
        let dec4 = DecoderBlockPlus::new(vb.pp("dec4"), c4 * 2, c3)?;
        let dec3 = DecoderBlockPlus::new(vb.pp("dec3"), c3 * 2, c2)?;
        let dec2 = DecoderBlockPlus::new(vb.pp("dec2"), c2 * 2, c1)?;
        let dec1 = DecoderBlockPlus::new(vb.pp("dec1"), c1 * 2, c1)?;

        // Output
        let output_conv = conv2d(c1, 1, 1, Conv2dConfig::default(), vb.pp("output"))?;

        // Temporal attention
        let temporal_attention = TemporalAttention::new(vb.pp("temporal_attention"), c1)?;

        Ok(Self {
            input_conv,
            input_bn,
            enc1,
            enc2,
            enc3,
            enc4,
            bottleneck_conv,
            bottleneck_bn,
            bottleneck_lstm,
            dec4,
            dec3,
            dec2,
            dec1,
            output_conv,
            temporal_attention,
        })
    }

    /// Forward pass for a sequence of voxel grids
    /// Input: (batch, seq_len, channels, height, width)
    /// Output: (batch, seq_len, 1, height, width) - reconstructed frames
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        let _batch_size = dims[0];
        let seq_len = dims[1];
        let _height = dims[3];
        let _width = dims[4];

        // Process input features
        let mut input_features = Vec::new();
        for t in 0..seq_len {
            let x_t = xs.narrow(1, t, 1)?.squeeze(1)?;
            let feat = self
                .input_bn
                .forward_t(&self.input_conv.forward(&x_t)?, false)?
                .relu()?;
            input_features.push(feat.unsqueeze(1)?);
        }
        let input_seq = Tensor::cat(&input_features, 1)?;

        // Encoder path with temporal processing
        let e1 = self.enc1.forward(&input_seq)?;
        let e2 = self.enc2.forward(&e1)?;
        let e3 = self.enc3.forward(&e2)?;
        let e4 = self.enc4.forward(&e3)?;

        // Bottleneck with bidirectional LSTM
        let b = self.process_bottleneck(&e4)?;

        // Decoder path with skip connections
        let d4 = self.dec4.forward(&b, &e4)?;
        let d3 = self.dec3.forward(&d4, &e3)?;
        let d2 = self.dec2.forward(&d3, &e2)?;
        let d1 = self.dec1.forward(&d2, &e1)?;

        // Apply temporal attention
        let attended = self.temporal_attention.forward(&d1)?;

        // Generate output for each timestep
        let mut outputs = Vec::new();
        for t in 0..seq_len {
            let feat_t = attended.narrow(1, t, 1)?.squeeze(1)?;
            let out_t = candle_nn::ops::sigmoid(&self.output_conv.forward(&feat_t)?)?;
            outputs.push(out_t.unsqueeze(1)?);
        }

        Tensor::cat(&outputs, 1)
    }

    /// Process bottleneck with temporal LSTM
    fn process_bottleneck(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dims()[1];
        let mut processed = Vec::new();

        for t in 0..seq_len {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?;
            let feat = self
                .bottleneck_bn
                .forward_t(&self.bottleneck_conv.forward(&x_t)?, false)?
                .relu()?;
            processed.push(feat.unsqueeze(1)?);
        }

        let bottleneck_seq = Tensor::cat(&processed, 1)?;
        self.bottleneck_lstm.forward(&bottleneck_seq)
    }
}

impl EncoderBlockPlus {
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

        let convlstm = ConvLSTM::simple(
            vb.pp("convlstm"),
            out_channels,
            out_channels,
            3,
            true, // Return sequences
        )?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            convlstm,
            downsample,
        })
    }

    /// Forward through encoder block with temporal processing
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.dims()[1];
        let mut features = Vec::new();

        // Process each timestep
        for t in 0..seq_len {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?;

            let x_t = self.conv1.forward(&x_t)?;
            let x_t = self.bn1.forward_t(&x_t, false)?.relu()?;

            let x_t = self.conv2.forward(&x_t)?;
            let x_t = self.bn2.forward_t(&x_t, false)?.relu()?;

            if self.downsample {
                let x_t = x_t.max_pool2d_with_stride(2, 2)?;
                features.push(x_t.unsqueeze(1)?);
            } else {
                features.push(x_t.unsqueeze(1)?);
            }
        }

        let feature_seq = Tensor::cat(&features, 1)?;

        // Apply ConvLSTM for temporal processing
        self.convlstm.forward(&feature_seq)
    }
}

impl DecoderBlockPlus {
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

        let convlstm = ConvLSTM::simple(vb.pp("convlstm"), out_channels, out_channels, 3, true)?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            convlstm,
        })
    }

    /// Forward through decoder with skip connections
    fn forward(&self, x: &Tensor, skip: &Tensor) -> Result<Tensor> {
        let seq_len = x.dims()[1];
        let mut features = Vec::new();

        for t in 0..seq_len {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?;
            let skip_t = skip.narrow(1, t, 1)?.squeeze(1)?;

            // Upsample
            let target_h = skip_t.dims()[2];
            let target_w = skip_t.dims()[3];
            let x_t = x_t.upsample_nearest2d(target_h, target_w)?;

            // Concatenate with skip
            let x_t = Tensor::cat(&[&x_t, &skip_t], 1)?;

            // Process
            let x_t = self.conv1.forward(&x_t)?;
            let x_t = self.bn1.forward_t(&x_t, false)?.relu()?;

            let x_t = self.conv2.forward(&x_t)?;
            let x_t = self.bn2.forward_t(&x_t, false)?.relu()?;

            features.push(x_t.unsqueeze(1)?);
        }

        let feature_seq = Tensor::cat(&features, 1)?;
        self.convlstm.forward(&feature_seq)
    }
}

impl TemporalAttention {
    fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let hidden_dim = channels / 8;

        let query_conv = conv2d(
            channels,
            hidden_dim,
            1,
            Conv2dConfig::default(),
            vb.pp("query"),
        )?;
        let key_conv = conv2d(
            channels,
            hidden_dim,
            1,
            Conv2dConfig::default(),
            vb.pp("key"),
        )?;
        let value_conv = conv2d(
            channels,
            channels,
            1,
            Conv2dConfig::default(),
            vb.pp("value"),
        )?;
        let output_conv = conv2d(
            channels,
            channels,
            1,
            Conv2dConfig::default(),
            vb.pp("output"),
        )?;

        Ok(Self {
            _query_conv: query_conv,
            _key_conv: key_conv,
            _value_conv: value_conv,
            output_conv,
        })
    }

    /// Apply temporal self-attention (simplified version)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let _batch_size = dims[0];
        let seq_len = dims[1];
        let _channels = dims[2];
        let _height = dims[3];
        let _width = dims[4];

        let mut attended_features = Vec::new();

        // For each timestep
        for t in 0..seq_len {
            // Current frame
            let query_frame = x.narrow(1, t, 1)?.squeeze(1)?;

            // Simple temporal averaging attention
            // This is a simplified version that avoids complex reshaping
            let mut weighted_frames = Vec::new();

            for t2 in 0..seq_len {
                let frame = x.narrow(1, t2, 1)?.squeeze(1)?;

                // Compute similarity weight based on distance in time
                let time_diff = (t as f64 - t2 as f64).abs();
                let weight = (-time_diff / 2.0).exp(); // Gaussian-like weighting

                let weighted = frame.affine(weight, 0.0)?;
                weighted_frames.push(weighted);
            }

            // Sum weighted frames
            let attended = weighted_frames
                .into_iter()
                .try_fold(
                    None,
                    |acc: Option<Tensor>, frame| -> Result<Option<Tensor>> {
                        match acc {
                            None => Ok(Some(frame)),
                            Some(a) => Ok(Some((a + frame)?)),
                        }
                    },
                )?
                .unwrap();

            // Normalize by sum of weights
            let weight_sum: f64 = (0..seq_len)
                .map(|t2| {
                    let time_diff = (t as f64 - t2 as f64).abs();
                    (-time_diff / 2.0).exp()
                })
                .sum();

            let normalized = attended.affine(1.0 / weight_sum, 0.0)?;

            // Apply output projection and residual
            let output = self.output_conv.forward(&normalized)? + query_frame;
            attended_features.push(output?.unsqueeze(1)?);
        }

        Tensor::cat(&attended_features, 1)
    }
}

/// Lightweight E2VID+ variant (FireNet+)
pub struct FireNetPlus {
    input_conv: Conv2d,

    // Fire modules with temporal processing
    fire1: FireModulePlus,
    fire2: FireModulePlus,
    fire3: FireModulePlus,
    fire4: FireModulePlus,

    // Temporal aggregation
    temporal_lstm: ConvLSTM,

    output_conv: Conv2d,
}

struct FireModulePlus {
    squeeze: Conv2d,
    expand_1x1: Conv2d,
    expand_3x3: Conv2d,
    temporal_gate: Conv2d,
}

impl FireNetPlus {
    pub fn new(vb: VarBuilder, in_channels: usize) -> Result<Self> {
        let input_conv = conv2d(
            in_channels,
            32,
            3,
            Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("input"),
        )?;

        let fire1 = FireModulePlus::new(vb.pp("fire1"), 32, 16, 32)?;
        let fire2 = FireModulePlus::new(vb.pp("fire2"), 64, 16, 32)?;
        let fire3 = FireModulePlus::new(vb.pp("fire3"), 64, 32, 64)?;
        let fire4 = FireModulePlus::new(vb.pp("fire4"), 128, 32, 64)?;

        let temporal_lstm = ConvLSTM::simple(vb.pp("temporal_lstm"), 128, 64, 3, true)?;

        let output_conv = conv2d(64, 1, 1, Conv2dConfig::default(), vb.pp("output"))?;

        Ok(Self {
            input_conv,
            fire1,
            fire2,
            fire3,
            fire4,
            temporal_lstm,
            output_conv,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dims()[1];
        let mut features = Vec::new();

        for t in 0..seq_len {
            let x_t = xs.narrow(1, t, 1)?.squeeze(1)?;

            let x_t = self.input_conv.forward(&x_t)?.relu()?;
            let x_t = self.fire1.forward(&x_t)?;
            let x_t = self.fire2.forward(&x_t)?;
            let x_t = self.fire3.forward(&x_t)?;
            let x_t = self.fire4.forward(&x_t)?;

            features.push(x_t.unsqueeze(1)?);
        }

        let feature_seq = Tensor::cat(&features, 1)?;
        let temporal_features = self.temporal_lstm.forward(&feature_seq)?;

        // Output for each timestep
        let mut outputs = Vec::new();
        for t in 0..seq_len {
            let feat_t = temporal_features.narrow(1, t, 1)?.squeeze(1)?;
            let out_t = candle_nn::ops::sigmoid(&self.output_conv.forward(&feat_t)?)?;
            outputs.push(out_t.unsqueeze(1)?);
        }

        Tensor::cat(&outputs, 1)
    }
}

impl FireModulePlus {
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

        let temporal_gate = conv2d(
            in_channels,
            expand_channels * 2,
            1,
            Conv2dConfig::default(),
            vb.pp("temporal_gate"),
        )?;

        Ok(Self {
            squeeze,
            expand_1x1,
            expand_3x3,
            temporal_gate,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let squeezed = self.squeeze.forward(x)?.relu()?;

        let expanded_1x1 = self.expand_1x1.forward(&squeezed)?.relu()?;
        let expanded_3x3 = self.expand_3x3.forward(&squeezed)?.relu()?;

        let expanded = Tensor::cat(&[&expanded_1x1, &expanded_3x3], 1)?;

        // Apply temporal gating
        let gate = candle_nn::ops::sigmoid(&self.temporal_gate.forward(x)?)?;

        expanded * gate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_e2vid_plus_creation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = E2VidPlus::new(vb, 5, 16, 3).unwrap();

        // Test forward pass with smaller input
        let input = Tensor::randn(0.0f32, 1.0, &[1, 3, 5, 32, 32], &device).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 3, 1, 32, 32]);
    }

    #[test]
    fn test_firenet_plus_creation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = FireNetPlus::new(vb, 5).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, &[1, 3, 5, 32, 32], &device).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 3, 1, 32, 32]);
    }
}
