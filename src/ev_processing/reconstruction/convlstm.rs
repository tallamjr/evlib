// Convolutional LSTM implementation for temporal processing in event-based reconstruction
// Based on "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, Conv2d, Conv2dConfig, VarBuilder};

/// ConvLSTM cell for processing temporal sequences with spatial structure
/// Maintains spatial dimensions while processing temporal information
#[derive(Debug)]
pub struct ConvLSTMCell {
    // Input dimension
    _input_dim: usize,
    // Hidden dimension
    hidden_dim: usize,
    // Kernel size for convolutions
    _kernel_size: usize,
    // Padding
    _padding: usize,
    // Gates
    conv_xi: Conv2d, // Input gate, input component
    conv_hi: Conv2d, // Input gate, hidden component
    conv_xf: Conv2d, // Forget gate, input component
    conv_hf: Conv2d, // Forget gate, hidden component
    conv_xo: Conv2d, // Output gate, input component
    conv_ho: Conv2d, // Output gate, hidden component
    conv_xc: Conv2d, // Cell gate, input component
    conv_hc: Conv2d, // Cell gate, hidden component
}

impl ConvLSTMCell {
    /// Create a new ConvLSTM cell
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        hidden_dim: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        let padding = kernel_size / 2;
        let conv_config = Conv2dConfig {
            padding,
            ..Default::default()
        };

        // Input gate
        let conv_xi = conv2d(
            input_dim,
            hidden_dim,
            kernel_size,
            conv_config,
            vb.pp("conv_xi"),
        )?;
        let conv_hi = conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            conv_config,
            vb.pp("conv_hi"),
        )?;

        // Forget gate
        let conv_xf = conv2d(
            input_dim,
            hidden_dim,
            kernel_size,
            conv_config,
            vb.pp("conv_xf"),
        )?;
        let conv_hf = conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            conv_config,
            vb.pp("conv_hf"),
        )?;

        // Output gate
        let conv_xo = conv2d(
            input_dim,
            hidden_dim,
            kernel_size,
            conv_config,
            vb.pp("conv_xo"),
        )?;
        let conv_ho = conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            conv_config,
            vb.pp("conv_ho"),
        )?;

        // Cell gate
        let conv_xc = conv2d(
            input_dim,
            hidden_dim,
            kernel_size,
            conv_config,
            vb.pp("conv_xc"),
        )?;
        let conv_hc = conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            conv_config,
            vb.pp("conv_hc"),
        )?;

        Ok(Self {
            _input_dim: input_dim,
            hidden_dim,
            _kernel_size: kernel_size,
            _padding: padding,
            conv_xi,
            conv_hi,
            conv_xf,
            conv_hf,
            conv_xo,
            conv_ho,
            conv_xc,
            conv_hc,
        })
    }

    /// Forward pass through the ConvLSTM cell
    /// Returns (new_hidden, new_cell)
    pub fn forward(
        &self,
        input: &Tensor,
        hidden: &Tensor,
        cell: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Input gate
        let i = candle_nn::ops::sigmoid(
            &(self.conv_xi.forward(input)? + self.conv_hi.forward(hidden)?)?,
        )?;

        // Forget gate
        let f = candle_nn::ops::sigmoid(
            &(self.conv_xf.forward(input)? + self.conv_hf.forward(hidden)?)?,
        )?;

        // Output gate
        let o = candle_nn::ops::sigmoid(
            &(self.conv_xo.forward(input)? + self.conv_ho.forward(hidden)?)?,
        )?;

        // Cell candidate
        let c_candidate = (self.conv_xc.forward(input)? + self.conv_hc.forward(hidden)?)?.tanh()?;

        // New cell state
        let new_cell = ((f * cell)? + (i * c_candidate)?)?;

        // New hidden state
        let new_hidden = (o * new_cell.tanh()?)?;

        Ok((new_hidden, new_cell))
    }

    /// Initialize hidden and cell states with zeros
    pub fn init_states(
        &self,
        batch_size: usize,
        height: usize,
        width: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let shape = [batch_size, self.hidden_dim, height, width];
        let hidden = Tensor::zeros(&shape, DType::F32, device)?;
        let cell = Tensor::zeros(&shape, DType::F32, device)?;
        Ok((hidden, cell))
    }
}

/// ConvLSTM layer that processes sequences
pub struct ConvLSTM {
    cells: Vec<ConvLSTMCell>,
    return_sequences: bool,
}

impl ConvLSTM {
    /// Create a new ConvLSTM layer
    ///
    /// # Arguments
    /// * `vb` - Variable builder for parameters
    /// * `input_dim` - Number of input channels
    /// * `hidden_dims` - Hidden dimensions for each layer
    /// * `kernel_sizes` - Kernel sizes for each layer
    /// * `num_layers` - Number of stacked ConvLSTM layers
    /// * `return_sequences` - Whether to return all timesteps or just the last
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        hidden_dims: Vec<usize>,
        kernel_sizes: Vec<usize>,
        return_sequences: bool,
    ) -> Result<Self> {
        assert_eq!(
            hidden_dims.len(),
            kernel_sizes.len(),
            "hidden_dims and kernel_sizes must have the same length"
        );

        let num_layers = hidden_dims.len();
        let mut cells = Vec::with_capacity(num_layers);
        let mut current_input_dim = input_dim;

        for i in 0..num_layers {
            let cell = ConvLSTMCell::new(
                vb.pp(format!("cell_{}", i)),
                current_input_dim,
                hidden_dims[i],
                kernel_sizes[i],
            )?;
            cells.push(cell);
            current_input_dim = hidden_dims[i];
        }

        Ok(Self {
            cells,
            return_sequences,
        })
    }

    /// Forward pass through the ConvLSTM
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch, seq_len, channels, height, width)
    ///
    /// # Returns
    /// * If return_sequences=true: (batch, seq_len, hidden_dim, height, width)
    /// * If return_sequences=false: (batch, hidden_dim, height, width)
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.dims();
        assert_eq!(
            dims.len(),
            5,
            "Input must be 5D: (batch, seq_len, channels, height, width)"
        );

        let batch_size = dims[0];
        let seq_len = dims[1];
        let height = dims[3];
        let width = dims[4];
        let device = input.device();

        let mut layer_outputs = Vec::new();
        let mut current_input = input.clone();

        // Process through each layer
        for (layer_idx, cell) in self.cells.iter().enumerate() {
            let mut outputs = Vec::new();

            // Initialize states for this layer
            let (mut hidden, mut cell_state) =
                cell.init_states(batch_size, height, width, device)?;

            // Process each timestep
            for t in 0..seq_len {
                // Extract input at timestep t: (batch, channels, height, width)
                let input_t = current_input.narrow(1, t, 1)?.squeeze(1)?;

                // Forward through cell
                let (new_hidden, new_cell) = cell.forward(&input_t, &hidden, &cell_state)?;

                if self.return_sequences || t == seq_len - 1 {
                    outputs.push(new_hidden.clone());
                }

                hidden = new_hidden;
                cell_state = new_cell;
            }

            // Stack outputs along time dimension
            let layer_output = if self.return_sequences {
                // Stack all outputs: (batch, seq_len, hidden_dim, height, width)
                Tensor::stack(&outputs, 1)?
            } else {
                // Only return last output: (batch, hidden_dim, height, width)
                outputs.into_iter().next_back().unwrap()
            };

            layer_outputs.push(layer_output.clone());

            // Prepare input for next layer
            if layer_idx < self.cells.len() - 1 && self.return_sequences {
                current_input = layer_output;
            }
        }

        // Return output from last layer
        Ok(layer_outputs.into_iter().next_back().unwrap())
    }

    /// Create a simple single-layer ConvLSTM
    pub fn simple(
        vb: VarBuilder,
        input_dim: usize,
        hidden_dim: usize,
        kernel_size: usize,
        return_sequences: bool,
    ) -> Result<Self> {
        Self::new(
            vb,
            input_dim,
            vec![hidden_dim],
            vec![kernel_size],
            return_sequences,
        )
    }
}

/// Bidirectional ConvLSTM for processing sequences in both directions
pub struct BiConvLSTM {
    forward_lstm: ConvLSTM,
    backward_lstm: ConvLSTM,
    merge_mode: MergeMode,
}

#[derive(Debug, Clone, Copy)]
pub enum MergeMode {
    Concat,
    Sum,
    Average,
    Multiply,
}

impl BiConvLSTM {
    /// Create a new bidirectional ConvLSTM
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        hidden_dim: usize,
        kernel_size: usize,
        merge_mode: MergeMode,
        return_sequences: bool,
    ) -> Result<Self> {
        let forward_lstm = ConvLSTM::simple(
            vb.pp("forward"),
            input_dim,
            hidden_dim,
            kernel_size,
            return_sequences,
        )?;

        let backward_lstm = ConvLSTM::simple(
            vb.pp("backward"),
            input_dim,
            hidden_dim,
            kernel_size,
            return_sequences,
        )?;

        Ok(Self {
            forward_lstm,
            backward_lstm,
            merge_mode,
        })
    }

    /// Forward pass through bidirectional ConvLSTM
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Forward pass
        let forward_out = self.forward_lstm.forward(input)?;

        // Backward pass - reverse the sequence
        let seq_len = input.dims()[1];
        let mut reversed_indices = Vec::with_capacity(seq_len);
        for i in (0..seq_len).rev() {
            reversed_indices.push(input.narrow(1, i, 1)?);
        }
        let reversed_input = Tensor::cat(&reversed_indices, 1)?;

        let backward_out = self.backward_lstm.forward(&reversed_input)?;

        // Merge outputs based on mode
        match self.merge_mode {
            MergeMode::Concat => {
                // Concatenate along channel dimension
                let concat_dim = if self.forward_lstm.return_sequences {
                    2
                } else {
                    1
                };
                Tensor::cat(&[&forward_out, &backward_out], concat_dim)
            }
            MergeMode::Sum => Ok((forward_out + backward_out)?),
            MergeMode::Average => Ok(((forward_out + backward_out)? / 2.0)?),
            MergeMode::Multiply => Ok((forward_out * backward_out)?),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_convlstm_cell_creation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let cell = ConvLSTMCell::new(vb, 3, 16, 3).unwrap();
        assert_eq!(cell.hidden_dim, 16);
        assert_eq!(cell._kernel_size, 3);
    }

    #[test]
    fn test_convlstm_forward() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let lstm = ConvLSTM::simple(vb, 3, 16, 3, true).unwrap();

        // Input: (batch=2, seq_len=10, channels=3, height=32, width=32)
        let input = Tensor::randn(0.0f32, 1.0, &[2, 10, 3, 32, 32], &device).unwrap();
        let output = lstm.forward(&input).unwrap();

        // Output should be (2, 10, 16, 32, 32) for return_sequences=true
        assert_eq!(output.dims(), &[2, 10, 16, 32, 32]);
    }

    #[test]
    fn test_convlstm_no_sequences() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let lstm = ConvLSTM::simple(vb, 3, 16, 3, false).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, &[2, 10, 3, 32, 32], &device).unwrap();
        let output = lstm.forward(&input).unwrap();

        // Output should be (2, 16, 32, 32) for return_sequences=false
        assert_eq!(output.dims(), &[2, 16, 32, 32]);
    }
}
