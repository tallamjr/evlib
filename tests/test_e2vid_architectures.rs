#[cfg(test)]
mod test_e2vid_architectures {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use evlib::ev_core::{Event, Events};
    use evlib::ev_processing::reconstruction::{E2Vid, E2VidConfig, E2VidMode, E2VidUNet, FireNet};

    fn create_test_events(n_events: usize, width: u16, height: u16) -> Events {
        use rand::prelude::*;
        let mut rng = thread_rng();

        let mut events = Vec::with_capacity(n_events);
        for i in 0..n_events {
            events.push(Event {
                x: rng.gen_range(0..width),
                y: rng.gen_range(0..height),
                t: i as f64 / n_events as f64,
                polarity: if rng.gen::<bool>() { 1 } else { -1 },
            });
        }
        events
    }

    #[test]
    fn test_unet_architecture_creation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Test UNet creation
        let unet = E2VidUNet::new(vb, 5, 32);
        assert!(unet.is_ok());
    }

    #[test]
    fn test_firenet_architecture_creation() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Test FireNet creation
        let firenet = FireNet::new(vb, 5);
        assert!(firenet.is_ok());
    }

    #[test]
    fn test_e2vid_with_unet() {
        let height = 256;
        let width = 256;
        let config = E2VidConfig::default();

        let mut e2vid = E2Vid::with_config(height, width, config);

        // Create UNet model
        let result = e2vid.create_network(E2VidMode::UNet);
        assert!(result.is_ok());

        // Process events
        let events = create_test_events(1000, width as u16, height as u16);
        let output = e2vid.process_events(&events);
        assert!(output.is_ok());

        let frame = output.unwrap();
        assert_eq!(frame.dims(), &[height, width]);
    }

    #[test]
    fn test_e2vid_with_firenet() {
        let height = 256;
        let width = 256;
        let config = E2VidConfig::default();

        let mut e2vid = E2Vid::with_config(height, width, config);

        // Create FireNet model
        let result = e2vid.create_network(E2VidMode::FireNet);
        assert!(result.is_ok());

        // Process events
        let events = create_test_events(1000, width as u16, height as u16);
        let output = e2vid.process_events(&events);
        assert!(output.is_ok());

        let frame = output.unwrap();
        assert_eq!(frame.dims(), &[height, width]);
    }

    #[test]
    fn test_architecture_output_range() {
        let height = 128;
        let width = 128;
        let config = E2VidConfig::default();

        for mode in [E2VidMode::UNet, E2VidMode::FireNet] {
            let mut e2vid = E2Vid::with_config(height, width, config.clone());
            e2vid.create_network(mode.clone()).unwrap();

            let events = create_test_events(500, width as u16, height as u16);
            let output = e2vid.process_events(&events).unwrap();

            // Check output is in [0, 1] range
            let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for val in output_vec {
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Output value {} out of range for {:?}",
                    val,
                    mode
                );
            }
        }
    }

    #[test]
    fn test_varying_input_sizes() {
        let config = E2VidConfig::default();
        let test_sizes = vec![(128, 128), (256, 256), (256, 512)];

        for (height, width) in test_sizes {
            let mut e2vid = E2Vid::with_config(height, width, config.clone());
            e2vid.create_network(E2VidMode::UNet).unwrap();

            let events = create_test_events(100, width as u16, height as u16);
            let output = e2vid.process_events(&events);

            assert!(output.is_ok());
            assert_eq!(output.unwrap().dims(), &[height, width]);
        }
    }

    #[test]
    fn test_empty_events() {
        let height = 256;
        let width = 256;
        let config = E2VidConfig::default();

        let mut e2vid = E2Vid::with_config(height, width, config);
        e2vid.create_network(E2VidMode::FireNet).unwrap();

        let empty_events = Vec::new();
        let output = e2vid.process_events(&empty_events);

        assert!(output.is_ok());
        let frame = output.unwrap();
        assert_eq!(frame.dims(), &[height, width]);
    }

    #[test]
    fn test_model_inference_consistency() {
        let height = 128;
        let width = 128;
        let config = E2VidConfig::default();

        let mut e2vid = E2Vid::with_config(height, width, config);
        e2vid.create_network(E2VidMode::UNet).unwrap();

        let events = create_test_events(500, width as u16, height as u16);

        // Run inference twice with same input
        let output1 = e2vid.process_events(&events).unwrap();
        let output2 = e2vid.process_events(&events).unwrap();

        // Results should be identical
        let diff = (output1 - output2).unwrap().abs().unwrap();
        let max_diff = diff
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            max_diff < 1e-6,
            "Inference results not consistent: max diff = {}",
            max_diff
        );
    }
}
