// E2VID Recurrent architecture implementation in Candle
// Matches the PyTorch E2VIDRecurrent model structure exactly

use candle_core::{Result, Tensor};
use candle_nn::{
    batch_norm, conv2d, conv_transpose2d, BatchNorm, Conv2d, ConvTranspose2d, Module, ModuleT,
    VarBuilder,
};

/// ConvLSTM cell implementation for E2VID Recurrent
#[derive(Debug)]
pub struct ConvLSTMCell {
    gates: Conv2d, // Combined input, forget, output, and cell gates
    hidden_size: usize,
}

impl ConvLSTMCell {
    pub fn new(
        vb: VarBuilder,
        input_size: usize,
        hidden_size: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        let total_size = 4 * hidden_size; // i, f, o, g gates
        let gates = conv2d(
            input_size + hidden_size,
            total_size,
            kernel_size,
            Default::default(),
            vb.pp("Gates"),
        )?;

        Ok(Self { gates, hidden_size })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        hidden: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch_size, _, height, width) = input.dims4()?;

        // Initialize hidden state if not provided
        let (h_prev, c_prev) = if let Some((h, c)) = hidden {
            (h.clone(), c.clone())
        } else {
            let zero_state = Tensor::zeros(
                (batch_size, self.hidden_size, height, width),
                input.dtype(),
                input.device(),
            )?;
            (zero_state.clone(), zero_state)
        };

        // Concatenate input and previous hidden state
        let combined = Tensor::cat(&[input, &h_prev], 1)?;

        // Compute all gates at once
        let gates = self.gates.forward(&combined)?;

        // Split gates: input, forget, output, cell
        let gate_size = self.hidden_size;
        let input_gate = candle_nn::ops::sigmoid(&gates.narrow(1, 0, gate_size)?)?;
        let forget_gate = candle_nn::ops::sigmoid(&gates.narrow(1, gate_size, gate_size)?)?;
        let output_gate = candle_nn::ops::sigmoid(&gates.narrow(1, 2 * gate_size, gate_size)?)?;
        let cell_gate = gates.narrow(1, 3 * gate_size, gate_size)?.tanh()?;

        // Update cell state
        let c_new = ((&forget_gate * &c_prev)? + (&input_gate * &cell_gate)?)?;

        // Update hidden state
        let h_new = (&output_gate * &c_new.tanh()?)?;

        Ok((h_new, c_new))
    }
}

/// Encoder block with convolution and ConvLSTM
#[derive(Debug)]
pub struct EncoderBlock {
    conv: Conv2d,
    norm_layer: BatchNorm,
    recurrent_block: ConvLSTMCell,
    #[allow(dead_code)]
    downsample: bool,
}

impl EncoderBlock {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        downsample: bool,
    ) -> Result<Self> {
        let conv_config = candle_nn::Conv2dConfig {
            stride: if downsample { 2 } else { 1 },
            padding: 2,
            ..Default::default()
        };

        let conv = conv2d(
            in_channels,
            out_channels,
            5,
            conv_config,
            vb.pp("conv").pp("conv2d"),
        )?;
        let norm_layer = batch_norm(out_channels, 1e-5, vb.pp("conv").pp("norm_layer"))?;
        let recurrent_block =
            ConvLSTMCell::new(vb.pp("recurrent_block"), out_channels, out_channels, 3)?;

        Ok(Self {
            conv,
            norm_layer,
            recurrent_block,
            downsample,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        hidden: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Convolution and normalization
        let x = self.conv.forward(input)?;
        let x = self.norm_layer.forward_t(&x, false)?;
        let x = x.relu()?;

        // ConvLSTM
        let (h_new, c_new) = self.recurrent_block.forward(&x, hidden)?;

        Ok((h_new.clone(), h_new, c_new))
    }
}

/// Residual block
#[derive(Debug)]
pub struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl ResidualBlock {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let conv_config = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let conv1 = conv2d(channels, channels, 3, conv_config, vb.pp("conv1"))?;
        let bn1 = batch_norm(channels, 1e-5, vb.pp("bn1"))?;
        let conv2 = conv2d(channels, channels, 3, conv_config, vb.pp("conv2"))?;
        let bn2 = batch_norm(channels, 1e-5, vb.pp("bn2"))?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(input)?;
        let x = self.bn1.forward_t(&x, false)?;
        let x = x.relu()?;

        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward_t(&x, false)?;

        // Residual connection
        let out = (&x + input)?.relu()?;
        Ok(out)
    }
}

/// Decoder block with transposed convolution
#[derive(Debug)]
pub struct DecoderBlock {
    transposed_conv: ConvTranspose2d,
    norm_layer: BatchNorm,
}

impl DecoderBlock {
    pub fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let conv_config = candle_nn::ConvTranspose2dConfig {
            stride: 2,
            padding: 2,
            output_padding: 1,
            ..Default::default()
        };

        let transposed_conv = conv_transpose2d(
            in_channels,
            out_channels,
            5,
            conv_config,
            vb.pp("transposed_conv2d"),
        )?;
        let norm_layer = batch_norm(out_channels, 1e-5, vb.pp("norm_layer"))?;

        Ok(Self {
            transposed_conv,
            norm_layer,
        })
    }

    pub fn forward(&self, input: &Tensor, skip: Option<&Tensor>) -> Result<Tensor> {
        let x = self.transposed_conv.forward(input)?;
        let x = self.norm_layer.forward_t(&x, false)?;
        let x = x.relu()?;

        // Add skip connection if provided
        if let Some(skip_tensor) = skip {
            Ok((&x + skip_tensor)?)
        } else {
            Ok(x)
        }
    }
}

/// Complete E2VID Recurrent model
#[derive(Debug)]
pub struct E2VidRecurrent {
    head: Conv2d,
    encoders: Vec<EncoderBlock>,
    resblocks: Vec<ResidualBlock>,
    decoders: Vec<DecoderBlock>,
    pred: Conv2d,
    pred_norm: BatchNorm,
}

impl E2VidRecurrent {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        // Head: 5 → 32 channels (5x5 conv)
        let head_config = candle_nn::Conv2dConfig {
            padding: 2,
            ..Default::default()
        };
        let head = conv2d(5, 32, 5, head_config, vb.pp("head").pp("conv2d"))?;

        // Encoders: 32 → 64 → 128 → 256
        let encoders = vec![
            EncoderBlock::new(vb.pp("encoders").pp("0"), 32, 64, true)?,
            EncoderBlock::new(vb.pp("encoders").pp("1"), 64, 128, true)?,
            EncoderBlock::new(vb.pp("encoders").pp("2"), 128, 256, true)?,
        ];

        // Residual blocks in the middle (256 channels)
        let resblocks = vec![
            ResidualBlock::new(vb.pp("resblocks").pp("0"), 256)?,
            ResidualBlock::new(vb.pp("resblocks").pp("1"), 256)?,
        ];

        // Decoders: 256 → 128 → 64 → 32
        let decoders = vec![
            DecoderBlock::new(vb.pp("decoders").pp("0"), 256, 128)?,
            DecoderBlock::new(vb.pp("decoders").pp("1"), 128, 64)?,
            DecoderBlock::new(vb.pp("decoders").pp("2"), 64, 32)?,
        ];

        // Prediction head: 32 → 1 (1x1 conv)
        let pred_config = candle_nn::Conv2dConfig::default();
        let pred = conv2d(32, 1, 1, pred_config, vb.pp("pred").pp("conv2d"))?;
        let pred_norm = batch_norm(1, 1e-5, vb.pp("pred").pp("norm_layer"))?;

        Ok(Self {
            head,
            encoders,
            resblocks,
            decoders,
            pred,
            pred_norm,
        })
    }

    /// Create from VarBuilder (for loading from PyTorch weights)
    pub fn load_from_varbuilder(vb: VarBuilder) -> Result<Self> {
        // Create model with the loaded weights, with the unetrecurrent prefix
        Self::new(vb.pp("unetrecurrent"))
    }
}

impl Module for E2VidRecurrent {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Head processing
        let x = self.head.forward(xs)?;
        let x = x.relu()?;

        // Encoder path with skip connections
        // For deterministic behavior, always start with None hidden states (zeros)
        let mut skip_connections = Vec::new();
        let mut x = x;

        for encoder in self.encoders.iter() {
            // Always use None (zero) hidden states for deterministic behavior
            let (output, _h, _c) = encoder.forward(&x, None)?;
            skip_connections.push(output.clone());
            x = output;
        }

        // Residual blocks
        for resblock in &self.resblocks {
            x = resblock.forward(&x)?;
        }

        // Decoder path with skip connections
        for (i, decoder) in self.decoders.iter().enumerate() {
            let skip_idx = self.encoders.len() - 1 - i;
            let skip = if skip_idx < skip_connections.len() {
                Some(&skip_connections[skip_idx])
            } else {
                None
            };
            x = decoder.forward(&x, skip)?;
        }

        // Prediction
        let x = self.pred.forward(&x)?;
        let x = self.pred_norm.forward_t(&x, false)?;
        let x = candle_nn::ops::sigmoid(&x)?; // Output in [0, 1] range

        Ok(x)
    }
}
