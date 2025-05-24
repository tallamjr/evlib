// Self-supervised loss functions for SSL-E2VID
// Enables training without ground truth frames

use candle_core::{Result, Tensor};
use candle_nn::loss;

/// Photometric consistency loss
/// Enforces that events should correspond to brightness changes
pub struct PhotometricLoss {
    /// Weight for positive events
    pos_weight: f32,
    /// Weight for negative events
    neg_weight: f32,
    /// Small epsilon to avoid division by zero
    _eps: f32,
}

impl PhotometricLoss {
    pub fn new(pos_weight: f32, neg_weight: f32) -> Self {
        Self {
            pos_weight,
            neg_weight,
            _eps: 1e-8,
        }
    }

    /// Compute photometric consistency loss
    ///
    /// # Arguments
    /// * `frames` - Reconstructed frames (B, T, 1, H, W)
    /// * `events` - Event voxel grid (B, T, C, H, W) where C is polarity bins
    pub fn forward(&self, frames: &Tensor, events: &Tensor) -> Result<Tensor> {
        let _batch_size = frames.dims()[0];
        let seq_len = frames.dims()[1];

        if seq_len < 2 {
            // Need at least 2 frames for temporal difference
            return Tensor::new(&[0.0f32], frames.device());
        }

        let mut losses = Vec::new();

        for t in 1..seq_len {
            // Get consecutive frames
            let frame_prev = frames.narrow(1, t - 1, 1)?.squeeze(1)?;
            let frame_curr = frames.narrow(1, t, 1)?.squeeze(1)?;

            // Compute brightness change
            let brightness_diff = (&frame_curr - &frame_prev)?;

            // Get events between frames
            let events_t = events.narrow(1, t - 1, 1)?.squeeze(1)?;

            // Split positive and negative events
            let pos_events = events_t.narrow(1, 0, events_t.dims()[1] / 2)?;
            let neg_events = events_t.narrow(1, events_t.dims()[1] / 2, events_t.dims()[1] / 2)?;

            // Sum events across polarity dimension
            let pos_sum = pos_events.sum(1)?;
            let neg_sum = neg_events.sum(1)?;

            // Positive events should correspond to brightness increase
            let pos_loss = (&pos_sum * brightness_diff.relu()?)?.mean_all()?;

            // Negative events should correspond to brightness decrease
            let neg_loss = neg_sum.mul(&brightness_diff.neg()?.relu()?)?.mean_all()?;

            // Combine losses
            let pos_weight_tensor = Tensor::new(&[self.pos_weight], brightness_diff.device())?;
            let neg_weight_tensor = Tensor::new(&[self.neg_weight], brightness_diff.device())?;
            let loss_t = pos_weight_tensor
                .mul(&pos_loss)?
                .add(&neg_weight_tensor.mul(&neg_loss)?)?;
            losses.push(loss_t);
        }

        // Average over time
        let total_loss = Tensor::stack(&losses, 0)?.mean(0)?;
        Ok(total_loss)
    }
}

/// Temporal consistency loss
/// Enforces smooth transitions between reconstructed frames
pub struct TemporalConsistencyLoss {
    /// Weight for L1 penalty
    l1_weight: f32,
    /// Weight for gradient penalty
    grad_weight: f32,
}

impl TemporalConsistencyLoss {
    pub fn new(l1_weight: f32, grad_weight: f32) -> Self {
        Self {
            l1_weight,
            grad_weight,
        }
    }

    /// Compute temporal consistency loss
    pub fn forward(&self, frames: &Tensor) -> Result<Tensor> {
        let seq_len = frames.dims()[1];

        if seq_len < 2 {
            return Tensor::new(&[0.0f32], frames.device());
        }

        let mut losses = Vec::new();

        for t in 1..seq_len {
            let frame_prev = frames.narrow(1, t - 1, 1)?.squeeze(1)?;
            let frame_curr = frames.narrow(1, t, 1)?.squeeze(1)?;

            // L1 difference between consecutive frames
            let l1_diff = (&frame_curr - &frame_prev)?.abs()?.mean_all()?;

            // Gradient consistency
            let grad_x_prev = compute_gradient_x(&frame_prev)?;
            let grad_y_prev = compute_gradient_y(&frame_prev)?;
            let grad_x_curr = compute_gradient_x(&frame_curr)?;
            let grad_y_curr = compute_gradient_y(&frame_curr)?;

            let grad_diff_x = (&grad_x_curr - &grad_x_prev)?.abs()?.mean_all()?;
            let grad_diff_y = (&grad_y_curr - &grad_y_prev)?.abs()?.mean_all()?;

            let l1_weight_tensor = Tensor::new(&[self.l1_weight], frame_curr.device())?;
            let grad_weight_tensor = Tensor::new(&[self.grad_weight], frame_curr.device())?;
            let grad_sum = grad_diff_x.add(&grad_diff_y)?;
            let loss_t = l1_weight_tensor
                .mul(&l1_diff)?
                .add(&grad_weight_tensor.mul(&grad_sum)?)?;
            losses.push(loss_t);
        }

        Tensor::stack(&losses, 0)?.mean(0)
    }
}

/// Event reconstruction loss
/// Ensures reconstructed frames can explain observed events
pub struct EventReconstructionLoss {
    /// Threshold for event generation
    threshold: f32,
    /// Temperature for soft thresholding
    temperature: f32,
}

impl EventReconstructionLoss {
    pub fn new(threshold: f32, temperature: f32) -> Self {
        Self {
            threshold,
            temperature,
        }
    }

    /// Forward pass
    pub fn forward(&self, frames: &Tensor, events: &Tensor) -> Result<Tensor> {
        let seq_len = frames.dims()[1];
        let mut losses = Vec::new();

        for t in 1..seq_len {
            let frame_prev = frames.narrow(1, t - 1, 1)?.squeeze(1)?;
            let frame_curr = frames.narrow(1, t, 1)?.squeeze(1)?;

            // Log intensity change
            let eps = Tensor::new(&[1e-5f32], frame_prev.device())?;
            let log_prev = frame_prev.add(&eps)?.log()?;
            let eps = Tensor::new(&[1e-5f32], frame_curr.device())?;
            let log_curr = frame_curr.add(&eps)?.log()?;
            let log_diff = (&log_curr - &log_prev)?;

            // Soft thresholding to generate events
            let pos_events_pred = soft_threshold(&log_diff, self.threshold, self.temperature)?;
            let neg_events_pred =
                soft_threshold(&log_diff.neg()?, self.threshold, self.temperature)?;

            // Get actual events
            let events_t = events.narrow(1, t - 1, 1)?.squeeze(1)?;
            let pos_events_actual = events_t.narrow(1, 0, 1)?.sum(1)?;
            let neg_events_actual = events_t.narrow(1, 1, 1)?.sum(1)?;

            // MSE between predicted and actual events
            let pos_loss = loss::mse(&pos_events_pred, &pos_events_actual)?;
            let neg_loss = loss::mse(&neg_events_pred, &neg_events_actual)?;

            losses.push(pos_loss.add(&neg_loss)?);
        }

        Tensor::stack(&losses, 0)?.mean(0)
    }
}

/// Contrastive loss for learning event representations
/// Encourages similar event patterns to have similar reconstructions
pub struct ContrastiveLoss {
    /// Temperature parameter for InfoNCE
    temperature: f32,
    /// Margin for triplet loss variant
    margin: f32,
}

impl ContrastiveLoss {
    pub fn new(temperature: f32, margin: f32) -> Self {
        Self {
            temperature,
            margin,
        }
    }

    /// Compute InfoNCE contrastive loss
    pub fn info_nce(
        &self,
        anchor: &Tensor,
        positive: &Tensor,
        negatives: &[Tensor],
    ) -> Result<Tensor> {
        // Normalize features
        let anchor_norm = normalize_l2(anchor)?;
        let positive_norm = normalize_l2(positive)?;

        // Positive similarity
        let pos_sim = (&anchor_norm * &positive_norm)?.sum_keepdim(1)?;
        let temp_tensor = Tensor::new(&[self.temperature], pos_sim.device())?;
        let pos_score = pos_sim.div(&temp_tensor)?;

        // Negative similarities
        let mut neg_scores = Vec::new();
        for negative in negatives {
            let negative_norm = normalize_l2(negative)?;
            let neg_sim = (&anchor_norm * &negative_norm)?.sum_keepdim(1)?;
            let temp_tensor = Tensor::new(&[self.temperature], neg_sim.device())?;
            let neg_score = neg_sim.div(&temp_tensor)?;
            neg_scores.push(neg_score);
        }

        // Concatenate all scores
        let all_scores = Tensor::cat(&[vec![pos_score], neg_scores].concat(), 0)?;

        // InfoNCE loss
        let log_softmax = candle_nn::ops::log_softmax(&all_scores, 0)?;
        let loss = log_softmax.narrow(0, 0, 1)?.mean_all()?.neg()?;

        Ok(loss)
    }

    /// Compute triplet loss
    pub fn triplet(&self, anchor: &Tensor, positive: &Tensor, negative: &Tensor) -> Result<Tensor> {
        let anchor_norm = normalize_l2(anchor)?;
        let positive_norm = normalize_l2(positive)?;
        let negative_norm = normalize_l2(negative)?;

        // Distances
        let pos_dist = (&anchor_norm - &positive_norm)?.sqr()?.sum_keepdim(1)?;
        let neg_dist = (&anchor_norm - &negative_norm)?.sqr()?.sum_keepdim(1)?;

        // Triplet loss with margin
        let margin_tensor = Tensor::new(&[self.margin], pos_dist.device())?;
        let loss = pos_dist
            .sub(&neg_dist)?
            .add(&margin_tensor)?
            .relu()?
            .mean_all()?;
        Ok(loss)
    }
}

/// Combined SSL loss for training
pub struct SSLLoss {
    photometric_loss: PhotometricLoss,
    temporal_loss: TemporalConsistencyLoss,
    event_recon_loss: EventReconstructionLoss,
    contrastive_loss: ContrastiveLoss,

    // Loss weights
    photo_weight: f32,
    temporal_weight: f32,
    event_weight: f32,
    contrast_weight: f32,
}

impl SSLLoss {
    pub fn new(config: SSLLossConfig) -> Self {
        Self {
            photometric_loss: PhotometricLoss::new(config.pos_weight, config.neg_weight),
            temporal_loss: TemporalConsistencyLoss::new(config.l1_weight, config.grad_weight),
            event_recon_loss: EventReconstructionLoss::new(
                config.event_threshold,
                config.temperature,
            ),
            contrastive_loss: ContrastiveLoss::new(config.contrast_temp, config.margin),
            photo_weight: config.photo_weight,
            temporal_weight: config.temporal_weight,
            event_weight: config.event_weight,
            contrast_weight: config.contrast_weight,
        }
    }

    /// Compute combined loss
    pub fn forward(
        &self,
        frames: &Tensor,
        events: &Tensor,
        features: Option<(&Tensor, &Tensor, &[Tensor])>,
    ) -> Result<(Tensor, SSLLossComponents)> {
        // Individual losses
        let photo_loss = self.photometric_loss.forward(frames, events)?;
        let temporal_loss = self.temporal_loss.forward(frames)?;
        let event_loss = self.event_recon_loss.forward(frames, events)?;

        let contrast_loss = if let Some((anchor, positive, negatives)) = features {
            self.contrastive_loss
                .info_nce(anchor, positive, negatives)?
        } else {
            Tensor::new(&[0.0f32], frames.device())?
        };

        // Combined loss
        let device = frames.device();
        let photo_weight_tensor = Tensor::new(&[self.photo_weight], device)?;
        let temporal_weight_tensor = Tensor::new(&[self.temporal_weight], device)?;
        let event_weight_tensor = Tensor::new(&[self.event_weight], device)?;
        let contrast_weight_tensor = Tensor::new(&[self.contrast_weight], device)?;

        let total_loss = photo_weight_tensor
            .mul(&photo_loss)?
            .add(&temporal_weight_tensor.mul(&temporal_loss)?)?
            .add(&event_weight_tensor.mul(&event_loss)?)?
            .add(&contrast_weight_tensor.mul(&contrast_loss)?)?;

        let components = SSLLossComponents {
            photometric: photo_loss,
            temporal: temporal_loss,
            event_reconstruction: event_loss,
            contrastive: contrast_loss,
            total: total_loss.clone(),
        };

        Ok((total_loss, components))
    }
}

/// Configuration for SSL loss
#[derive(Clone)]
pub struct SSLLossConfig {
    // Photometric loss
    pub pos_weight: f32,
    pub neg_weight: f32,

    // Temporal consistency
    pub l1_weight: f32,
    pub grad_weight: f32,

    // Event reconstruction
    pub event_threshold: f32,
    pub temperature: f32,

    // Contrastive
    pub contrast_temp: f32,
    pub margin: f32,

    // Overall weights
    pub photo_weight: f32,
    pub temporal_weight: f32,
    pub event_weight: f32,
    pub contrast_weight: f32,
}

impl Default for SSLLossConfig {
    fn default() -> Self {
        Self {
            pos_weight: 1.0,
            neg_weight: 1.0,
            l1_weight: 1.0,
            grad_weight: 0.5,
            event_threshold: 0.2,
            temperature: 0.1,
            contrast_temp: 0.07,
            margin: 0.2,
            photo_weight: 1.0,
            temporal_weight: 0.5,
            event_weight: 1.0,
            contrast_weight: 0.1,
        }
    }
}

/// Components of the SSL loss for logging
pub struct SSLLossComponents {
    pub photometric: Tensor,
    pub temporal: Tensor,
    pub event_reconstruction: Tensor,
    pub contrastive: Tensor,
    pub total: Tensor,
}

// Helper functions

/// Compute horizontal gradient
fn compute_gradient_x(x: &Tensor) -> Result<Tensor> {
    let left = x.narrow(3, 0, x.dims()[3] - 1)?;
    let right = x.narrow(3, 1, x.dims()[3] - 1)?;
    right - left
}

/// Compute vertical gradient
fn compute_gradient_y(x: &Tensor) -> Result<Tensor> {
    let top = x.narrow(2, 0, x.dims()[2] - 1)?;
    let bottom = x.narrow(2, 1, x.dims()[2] - 1)?;
    bottom - top
}

/// Soft thresholding function
fn soft_threshold(x: &Tensor, threshold: f32, temperature: f32) -> Result<Tensor> {
    let threshold_tensor = Tensor::new(&[threshold], x.device())?;
    let temperature_tensor = Tensor::new(&[temperature], x.device())?;
    let scaled = x.sub(&threshold_tensor)?.div(&temperature_tensor)?;
    candle_nn::ops::sigmoid(&scaled)
}

/// L2 normalize tensor
fn normalize_l2(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    let eps = Tensor::new(&[1e-8f32], x.device())?;
    x.div(&norm.add(&eps)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_photometric_loss() {
        let device = Device::Cpu;
        let loss = PhotometricLoss::new(1.0, 1.0);

        // Create test data
        let frames = Tensor::randn(0.5f32, 0.1, &[2, 3, 1, 32, 32], &device).unwrap();
        let events = Tensor::randn(0.0f32, 0.5, &[2, 3, 4, 32, 32], &device).unwrap();

        let result = loss.forward(&frames, &events).unwrap();
        assert_eq!(result.dims().len(), 0); // Scalar loss
    }

    #[test]
    fn test_temporal_consistency_loss() {
        let device = Device::Cpu;
        let loss = TemporalConsistencyLoss::new(1.0, 0.5);

        let frames = Tensor::randn(0.5f32, 0.1, &[2, 4, 1, 16, 16], &device).unwrap();
        let result = loss.forward(&frames).unwrap();

        assert_eq!(result.dims().len(), 0); // Scalar loss
    }

    #[test]
    fn test_ssl_combined_loss() {
        let device = Device::Cpu;
        let config = SSLLossConfig::default();
        let ssl_loss = SSLLoss::new(config);

        let frames = Tensor::randn(0.5f32, 0.1, &[1, 3, 1, 32, 32], &device).unwrap();
        let events = Tensor::randn(0.0f32, 0.5, &[1, 3, 2, 32, 32], &device).unwrap();

        let (total, components) = ssl_loss.forward(&frames, &events, None).unwrap();

        assert_eq!(total.dims().len(), 0);
        assert_eq!(components.photometric.dims().len(), 0);
        assert_eq!(components.temporal.dims().len(), 0);
    }
}
