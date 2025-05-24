// Test SSL-E2VID implementation
#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use evlib::ev_processing::reconstruction::{
        ContrastiveLoss, EventAugmentation, EventReconstructionLoss, PhotometricLoss, SSLLoss,
        SSLLossConfig, SslE2Vid, SslE2VidMomentum, SslTrainer, TemporalConsistencyLoss,
    };

    #[test]
    fn test_ssl_e2vid_forward() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = SSLLossConfig::default();
        let model = SslE2Vid::new(vb, 5, 32, 128, config).unwrap();

        // Test single forward pass
        let events = Tensor::randn(0.0f32, 1.0, &[2, 5, 64, 64], &device).unwrap();
        let output = model.forward(&events).unwrap();

        assert_eq!(output.dims(), &[2, 1, 64, 64]);
        assert_eq!(output.dtype(), DType::F32);
    }

    #[test]
    fn test_ssl_e2vid_with_features() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = SSLLossConfig::default();
        let model = SslE2Vid::new(vb, 5, 32, 128, config).unwrap();

        let events = Tensor::randn(0.0f32, 1.0, &[2, 5, 64, 64], &device).unwrap();
        let (frames, features) = model.forward_with_features(&events).unwrap();

        assert_eq!(frames.dims(), &[2, 1, 64, 64]);
        assert_eq!(features.dims(), &[2, 128]);
    }

    #[test]
    fn test_ssl_loss_computation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = SSLLossConfig::default();
        let model = SslE2Vid::new(vb, 5, 32, 128, config).unwrap();

        // Create event sequence
        let events = Tensor::randn(0.0f32, 1.0, &[1, 5, 32, 32], &device).unwrap();
        let augmented = Tensor::randn(0.0f32, 1.0, &[1, 5, 32, 32], &device).unwrap();
        let negative1 = Tensor::randn(0.0f32, 1.0, &[1, 5, 32, 32], &device).unwrap();
        let negative2 = Tensor::randn(0.0f32, 1.0, &[1, 5, 32, 32], &device).unwrap();

        let (loss, components) = model
            .compute_loss(&events, Some(&augmented), Some(&vec![negative1, negative2]))
            .unwrap();

        // Verify loss components
        assert_eq!(loss.dims().len(), 0); // Scalar loss
        assert_eq!(components.photometric.dims().len(), 0);
        assert_eq!(components.temporal.dims().len(), 0);
        assert_eq!(components.event_reconstruction.dims().len(), 0);
        assert_eq!(components.contrastive.dims().len(), 0);
        assert_eq!(components.total.dims().len(), 0);
    }

    #[test]
    fn test_photometric_loss() {
        let device = Device::Cpu;
        let loss_fn = PhotometricLoss::new(1.0, 1.0);

        // Create frames sequence
        let frames = Tensor::randn(0.5f32, 0.1, &[2, 3, 1, 32, 32], &device).unwrap();
        // Create events with polarity bins
        let events = Tensor::randn(0.0f32, 0.5, &[2, 3, 4, 32, 32], &device).unwrap();

        let loss = loss_fn.forward(&frames, &events).unwrap();
        assert_eq!(loss.dims().len(), 0); // Scalar loss
    }

    #[test]
    fn test_temporal_consistency_loss() {
        let device = Device::Cpu;
        let loss_fn = TemporalConsistencyLoss::new(1.0, 0.5);

        let frames = Tensor::randn(0.5f32, 0.1, &[2, 4, 1, 16, 16], &device).unwrap();
        let loss = loss_fn.forward(&frames).unwrap();

        assert_eq!(loss.dims().len(), 0); // Scalar loss
    }

    #[test]
    fn test_event_reconstruction_loss() {
        let device = Device::Cpu;
        let loss_fn = EventReconstructionLoss::new(0.2, 0.1);

        let frames = Tensor::randn(0.5f32, 0.1, &[1, 3, 1, 32, 32], &device).unwrap();
        let events = Tensor::randn(0.0f32, 0.5, &[1, 3, 2, 32, 32], &device).unwrap();

        let loss = loss_fn.forward(&frames, &events).unwrap();
        assert_eq!(loss.dims().len(), 0); // Scalar loss
    }

    #[test]
    fn test_contrastive_loss_info_nce() {
        let device = Device::Cpu;
        let loss_fn = ContrastiveLoss::new(0.07, 0.2);

        let anchor = Tensor::randn(0.0f32, 1.0, &[4, 128], &device).unwrap();
        let positive = Tensor::randn(0.0f32, 1.0, &[4, 128], &device).unwrap();
        let neg1 = Tensor::randn(0.0f32, 1.0, &[4, 128], &device).unwrap();
        let neg2 = Tensor::randn(0.0f32, 1.0, &[4, 128], &device).unwrap();

        let loss = loss_fn.info_nce(&anchor, &positive, &[neg1, neg2]).unwrap();
        assert_eq!(loss.dims().len(), 0); // Scalar loss
    }

    #[test]
    fn test_contrastive_loss_triplet() {
        let device = Device::Cpu;
        let loss_fn = ContrastiveLoss::new(0.07, 0.2);

        let anchor = Tensor::randn(0.0f32, 1.0, &[4, 128], &device).unwrap();
        let positive = Tensor::randn(0.0f32, 1.0, &[4, 128], &device).unwrap();
        let negative = Tensor::randn(0.0f32, 1.0, &[4, 128], &device).unwrap();

        let loss = loss_fn.triplet(&anchor, &positive, &negative).unwrap();
        assert_eq!(loss.dims().len(), 0); // Scalar loss
    }

    #[test]
    fn test_event_augmentation() {
        let device = Device::Cpu;
        let augmentor = EventAugmentation::new(0.1, 5, 0.05, 0.1);

        let events = Tensor::randn(0.0f32, 1.0, &[2, 5, 32, 32], &device).unwrap();
        let augmented = augmentor.augment(&events).unwrap();

        assert_eq!(augmented.dims(), events.dims());

        // Test individual augmentation methods
        let noisy = augmentor.add_noise(&events).unwrap();
        assert_eq!(noisy.dims(), events.dims());

        let dropped = augmentor.drop_events(&events).unwrap();
        assert_eq!(dropped.dims(), events.dims());
    }

    #[test]
    fn test_ssl_trainer() {
        let device = Device::Cpu;
        let config = SSLLossConfig::default();

        let mut trainer = SslTrainer::new(5, 32, 128, config, 0.001, &device).unwrap();

        // Create batch
        let events = Tensor::randn(0.0f32, 1.0, &[2, 5, 32, 32], &device).unwrap();
        let augmented = Tensor::randn(0.0f32, 1.0, &[2, 5, 32, 32], &device).unwrap();
        let negative = Tensor::randn(0.0f32, 1.0, &[2, 5, 32, 32], &device).unwrap();

        // Training step
        let components = trainer
            .train_step(&events, Some(&augmented), Some(&vec![negative]))
            .unwrap();

        // Check loss components
        assert_eq!(components.total.dims().len(), 0);

        // Test inference
        let output = trainer.model().forward(&events).unwrap();
        assert_eq!(output.dims(), &[2, 1, 32, 32]);
    }

    #[test]
    fn test_ssl_e2vid_momentum() {
        let device = Device::Cpu;
        let varmap_online = VarMap::new();
        let varmap_target = VarMap::new();
        let vb_online = VarBuilder::from_varmap(&varmap_online, DType::F32, &device);
        let vb_target = VarBuilder::from_varmap(&varmap_target, DType::F32, &device);

        let config = SSLLossConfig::default();
        let mut model =
            SslE2VidMomentum::new(vb_online, vb_target, 5, 32, 128, config, 0.999).unwrap();

        let events = Tensor::randn(0.0f32, 1.0, &[2, 5, 64, 64], &device).unwrap();
        let output = model.forward(&events).unwrap();

        assert_eq!(output.dims(), &[2, 1, 64, 64]);

        // Test target update
        model.update_target().unwrap();
    }

    #[test]
    fn test_ssl_loss_config() {
        let config = SSLLossConfig::default();

        assert_eq!(config.pos_weight, 1.0);
        assert_eq!(config.neg_weight, 1.0);
        assert_eq!(config.l1_weight, 1.0);
        assert_eq!(config.grad_weight, 0.5);
        assert_eq!(config.event_threshold, 0.2);
        assert_eq!(config.temperature, 0.1);
        assert_eq!(config.contrast_temp, 0.07);
        assert_eq!(config.margin, 0.2);
        assert_eq!(config.photo_weight, 1.0);
        assert_eq!(config.temporal_weight, 0.5);
        assert_eq!(config.event_weight, 1.0);
        assert_eq!(config.contrast_weight, 0.1);
    }

    #[test]
    fn test_ssl_loss_combined() {
        let device = Device::Cpu;
        let config = SSLLossConfig {
            pos_weight: 1.5,
            neg_weight: 1.5,
            l1_weight: 0.8,
            grad_weight: 0.3,
            event_threshold: 0.15,
            temperature: 0.05,
            contrast_temp: 0.1,
            margin: 0.3,
            photo_weight: 2.0,
            temporal_weight: 0.8,
            event_weight: 1.5,
            contrast_weight: 0.2,
        };

        let ssl_loss = SSLLoss::new(config);

        // Create test data
        let frames = Tensor::randn(0.5f32, 0.1, &[1, 3, 1, 32, 32], &device).unwrap();
        let events = Tensor::randn(0.0f32, 0.5, &[1, 3, 2, 32, 32], &device).unwrap();
        let anchor = Tensor::randn(0.0f32, 1.0, &[1, 128], &device).unwrap();
        let positive = Tensor::randn(0.0f32, 1.0, &[1, 128], &device).unwrap();
        let negative = Tensor::randn(0.0f32, 1.0, &[1, 128], &device).unwrap();

        let (total_loss, components) = ssl_loss
            .forward(
                &frames,
                &events,
                Some((&anchor, &positive, &vec![negative])),
            )
            .unwrap();

        assert_eq!(total_loss.dims().len(), 0);
        assert_eq!(components.photometric.dims().len(), 0);
        assert_eq!(components.temporal.dims().len(), 0);
        assert_eq!(components.event_reconstruction.dims().len(), 0);
        assert_eq!(components.contrastive.dims().len(), 0);
        assert_eq!(components.total.dims().len(), 0);
    }
}
