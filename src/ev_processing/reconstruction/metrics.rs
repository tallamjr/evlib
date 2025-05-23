// Event-to-video reconstruction quality metrics
// Based on EVREAL benchmarking framework

use candle_core::{Result, Tensor};

/// Mean Squared Error (MSE) between two images
pub fn mse(pred: &Tensor, target: &Tensor) -> Result<f32> {
    let diff = (pred - target)?;
    let squared = diff.sqr()?;
    let sum = squared.sum_all()?;
    let n_elements = pred.elem_count();

    Ok(sum.to_scalar::<f32>()? / n_elements as f32)
}

/// Peak Signal-to-Noise Ratio (PSNR) in dB
pub fn psnr(pred: &Tensor, target: &Tensor, max_val: f32) -> Result<f32> {
    let mse_val = mse(pred, target)?;

    if mse_val < 1e-10 {
        Ok(100.0) // Perfect match
    } else {
        Ok(20.0 * (max_val / mse_val.sqrt()).log10())
    }
}

/// Structural Similarity Index (SSIM)
/// Simplified implementation - full version would use sliding windows
pub fn ssim(pred: &Tensor, target: &Tensor) -> Result<f32> {
    // Constants from the SSIM paper
    let c1 = 0.01_f32.powi(2);
    let c2 = 0.03_f32.powi(2);

    // Compute means
    let mu_x = pred.mean_all()?;
    let mu_y = target.mean_all()?;
    let mu_x_val = mu_x.to_scalar::<f32>()?;
    let mu_y_val = mu_y.to_scalar::<f32>()?;

    // Compute variances and covariance
    let pred_centered = (pred - mu_x)?;
    let target_centered = (target - mu_y)?;

    let var_x = pred_centered.sqr()?.mean_all()?;
    let var_y = target_centered.sqr()?.mean_all()?;
    let cov_xy = (pred_centered * target_centered)?.mean_all()?;

    let var_x_val = var_x.to_scalar::<f32>()?;
    let var_y_val = var_y.to_scalar::<f32>()?;
    let cov_xy_val = cov_xy.to_scalar::<f32>()?;

    // SSIM formula
    let numerator = (2.0 * mu_x_val * mu_y_val + c1) * (2.0 * cov_xy_val + c2);
    let denominator = (mu_x_val.powi(2) + mu_y_val.powi(2) + c1) * (var_x_val + var_y_val + c2);

    Ok(numerator / denominator)
}

/// Multi-Scale SSIM (MS-SSIM)
/// Uses multiple scales to evaluate structural similarity
pub fn ms_ssim(pred: &Tensor, target: &Tensor, scales: usize) -> Result<f32> {
    let mut ms_ssim_val = 1.0;
    let mut current_pred = pred.clone();
    let mut current_target = target.clone();

    // Weights for different scales (from MS-SSIM paper)
    let weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

    for (scale, weight) in weights.iter().enumerate().take(scales.min(5)) {
        let ssim_val = ssim(&current_pred, &current_target)?;
        ms_ssim_val *= ssim_val.powf(*weight);

        if scale < scales - 1 {
            // Downsample by factor of 2
            current_pred = downsample_2x(&current_pred)?;
            current_target = downsample_2x(&current_target)?;
        }
    }

    Ok(ms_ssim_val)
}

/// Downsample tensor by factor of 2 using average pooling
fn downsample_2x(tensor: &Tensor) -> Result<Tensor> {
    // Simple 2x2 average pooling
    tensor.avg_pool2d_with_stride(2, 2)
}

/// Temporal consistency metric for video sequences
/// Measures smoothness between consecutive frames
pub fn temporal_consistency(frames: &[Tensor]) -> Result<f32> {
    if frames.len() < 2 {
        return Ok(1.0); // Perfect consistency for single frame
    }

    let mut total_diff = 0.0;

    for i in 1..frames.len() {
        let diff = mse(&frames[i], &frames[i - 1])?;
        total_diff += diff;
    }

    Ok(total_diff / (frames.len() - 1) as f32)
}

/// Event density weighted MSE
/// Gives more weight to regions with higher event density
pub fn event_weighted_mse(pred: &Tensor, target: &Tensor, event_density: &Tensor) -> Result<f32> {
    // Normalize event density to [0, 1]
    let density_min = event_density.min_all()?;
    let density_max = event_density.max_all()?;
    let density_range = (&density_max - &density_min)?;

    let normalized_density = if density_range.to_scalar::<f32>()? > 1e-6 {
        ((event_density - &density_min)? / &density_range)?
    } else {
        Tensor::ones_like(event_density)?
    };

    // Weighted MSE
    let diff = (pred - target)?;
    let weighted_squared = (diff.sqr()? * &normalized_density)?;
    let sum = weighted_squared.sum_all()?;
    let weight_sum = normalized_density.sum_all()?;

    Ok(sum.to_scalar::<f32>()? / weight_sum.to_scalar::<f32>()?)
}

/// Perceptual loss placeholder (would require pre-trained VGG model)
/// For now, returns a simple gradient-based metric
pub fn perceptual_loss(pred: &Tensor, target: &Tensor) -> Result<f32> {
    // Compute image gradients as a simple perceptual feature
    let pred_grad = image_gradient(pred)?;
    let target_grad = image_gradient(target)?;

    mse(&pred_grad, &target_grad)
}

/// Compute image gradients (Sobel-like)
fn image_gradient(image: &Tensor) -> Result<Tensor> {
    let dims = image.dims();
    let height = dims[dims.len() - 2];
    let width = dims[dims.len() - 1];

    // Compute differences in x and y directions
    let dx = if width > 1 {
        let left = image.narrow(dims.len() - 1, 0, width - 1)?;
        let right = image.narrow(dims.len() - 1, 1, width - 1)?;
        (right - left)?
    } else {
        Tensor::zeros_like(image)?
    };

    let dy = if height > 1 {
        let top = image.narrow(dims.len() - 2, 0, height - 1)?;
        let bottom = image.narrow(dims.len() - 2, 1, height - 1)?;
        (bottom - top)?
    } else {
        Tensor::zeros_like(image)?
    };

    // Magnitude of gradient
    let dx_sq = dx.sqr()?;
    let dy_sq = dy.sqr()?;

    // Ensure compatible shapes by taking minimum dimensions
    let min_height = height.min(dx_sq.dims()[dims.len() - 2]);
    let min_width = width.min(dx_sq.dims()[dims.len() - 1]);

    let dx_cropped =
        dx_sq
            .narrow(dims.len() - 2, 0, min_height)?
            .narrow(dims.len() - 1, 0, min_width)?;
    let dy_cropped =
        dy_sq
            .narrow(dims.len() - 2, 0, min_height)?
            .narrow(dims.len() - 1, 0, min_width)?;

    (dx_cropped + dy_cropped)?.sqrt()
}

/// Metrics collection for evaluation
#[derive(Debug, Clone)]
pub struct ReconstructionMetrics {
    pub mse: f32,
    pub psnr: f32,
    pub ssim: f32,
    pub ms_ssim: Option<f32>,
    pub temporal_consistency: Option<f32>,
    pub perceptual_loss: Option<f32>,
}

impl ReconstructionMetrics {
    /// Compute all metrics between prediction and target
    pub fn compute(pred: &Tensor, target: &Tensor) -> Result<Self> {
        let mse_val = mse(pred, target)?;
        let psnr_val = psnr(pred, target, 1.0)?; // Assuming normalized [0, 1] range
        let ssim_val = ssim(pred, target)?;

        Ok(Self {
            mse: mse_val,
            psnr: psnr_val,
            ssim: ssim_val,
            ms_ssim: None,
            temporal_consistency: None,
            perceptual_loss: None,
        })
    }

    /// Compute full metrics including optional ones
    pub fn compute_full(
        pred: &Tensor,
        target: &Tensor,
        compute_ms_ssim: bool,
        compute_perceptual: bool,
    ) -> Result<Self> {
        let mut metrics = Self::compute(pred, target)?;

        if compute_ms_ssim {
            metrics.ms_ssim = Some(ms_ssim(pred, target, 5)?);
        }

        if compute_perceptual {
            metrics.perceptual_loss = Some(perceptual_loss(pred, target)?);
        }

        Ok(metrics)
    }

    /// Format metrics as a string
    pub fn format(&self) -> String {
        let mut s = format!(
            "MSE: {:.4}, PSNR: {:.2} dB, SSIM: {:.4}",
            self.mse, self.psnr, self.ssim
        );

        if let Some(ms_ssim) = self.ms_ssim {
            s.push_str(&format!(", MS-SSIM: {:.4}", ms_ssim));
        }

        if let Some(tc) = self.temporal_consistency {
            s.push_str(&format!(", Temporal: {:.4}", tc));
        }

        if let Some(pl) = self.perceptual_loss {
            s.push_str(&format!(", Perceptual: {:.4}", pl));
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_mse_identical() {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0f32, 1.0, &[1, 1, 32, 32], &device).unwrap();
        let mse_val = mse(&tensor, &tensor).unwrap();
        assert!(mse_val < 1e-6);
    }

    #[test]
    fn test_psnr_range() {
        let device = Device::Cpu;
        let pred = Tensor::zeros(&[1, 1, 32, 32], candle_core::DType::F32, &device).unwrap();
        let target = Tensor::ones(&[1, 1, 32, 32], candle_core::DType::F32, &device).unwrap();

        let psnr_val = psnr(&pred, &target, 1.0).unwrap();
        assert!(psnr_val > 0.0 && psnr_val < 50.0);
    }

    #[test]
    fn test_ssim_range() {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.5f32, 0.1, &[1, 1, 32, 32], &device).unwrap();
        let ssim_val = ssim(&tensor, &tensor).unwrap();
        assert!(ssim_val >= 0.99 && ssim_val <= 1.0);
    }
}
