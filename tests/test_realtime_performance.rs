//! Tests for real-time event streaming performance and functionality
//!
//! These tests verify the RealtimeEventStream implementation works correctly
//! and meets performance requirements for live webcam processing.
//!
//! **IMPORTANT: ALL TESTS IN THIS FILE ARE CURRENTLY DISABLED**
//!
//! These tests depend on the ev_simulation module and candle_core dependency
//! which have been removed from the codebase. The tests are commented out
//! to prevent compilation errors whilst preserving the test code for
//! potential future use if these dependencies are re-added.

#[cfg(test)]
mod tests {
    // Disabled: ev_simulation module and candle_core dependency removed
    // #[cfg(feature = "gstreamer")]
    // use candle_core::Device;
    // #[cfg(feature = "gstreamer")]
    // use evlib::ev_simulation::esim::EsimConfig;
    // #[cfg(feature = "gstreamer")]
    // use evlib::ev_simulation::realtime_stream::{
    //     EventStreamManager, RealtimeEventStream, RealtimeStreamConfig, StreamingStats,
    // };
    // Disabled: time imports not needed since tests are commented out
    // #[cfg(feature = "gstreamer")]
    // use std::time::{Duration, Instant};

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_realtime_config_validation() {
    //     let config = RealtimeStreamConfig::default();
    //
    //     // Verify default values are sensible
    //     assert!(config.target_fps > 0.0 && config.target_fps <= 60.0);
    //     assert!(config.max_buffer_size > 0);
    //     assert!(config.processing_timeout_ms > 0);
    //     assert_eq!(config.device_id, 0);
    //
    //     // ESIM config should be optimised for real-time
    //     assert!(!config.esim_config.use_bilinear_interpolation); // Should be disabled for speed
    //     assert!(!config.esim_config.adaptive_thresholding); // Should be disabled for speed
    //     assert!(config.esim_config.base_config.contrast_threshold_pos > 0.0);
    //     assert!(config.esim_config.base_config.contrast_threshold_neg > 0.0);
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_realtime_stream_creation() {
    //     let config = RealtimeStreamConfig::default();
    //     let device = Device::Cpu;
    //
    //     let result = RealtimeEventStream::new(config, device);
    //     assert!(
    //         result.is_ok(),
    //         "Failed to create RealtimeEventStream: {:?}",
    //         result.err()
    //     );
    //
    //     let stream = result.unwrap();
    //     assert!(!stream.is_streaming());
    //
    //     let stats = stream.get_stats();
    //     assert_eq!(stats.frames_processed, 0);
    //     assert_eq!(stats.events_generated, 0);
    //     assert_eq!(stats.current_fps, 0.0);
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_streaming_stats_default() {
    //     let stats = StreamingStats::default();
    //
    //     assert_eq!(stats.frames_processed, 0);
    //     assert_eq!(stats.events_generated, 0);
    //     assert_eq!(stats.current_fps, 0.0);
    //     assert_eq!(stats.avg_events_per_frame, 0.0);
    //     assert_eq!(stats.buffer_size, 0);
    //     assert_eq!(stats.dropped_frames, 0);
    //     assert_eq!(stats.avg_latency_ms, 0.0);
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_event_stream_manager_creation() {
    //     let config = RealtimeStreamConfig::default();
    //     let device = Device::Cpu;
    //
    //     let result = EventStreamManager::new(config, device);
    //     assert!(
    //         result.is_ok(),
    //         "Failed to create EventStreamManager: {:?}",
    //         result.err()
    //     );
    //
    //     let manager = result.unwrap();
    //     let stats = manager.get_stats();
    //     assert_eq!(stats.frames_processed, 0);
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_get_empty_events() {
    //     let config = RealtimeStreamConfig::default();
    //     let device = Device::Cpu;
    //     let stream = RealtimeEventStream::new(config, device).unwrap();
    //
    //     // Should return empty vec when no events are available
    //     let events = stream.get_events(Some(10));
    //     assert!(events.is_empty());
    //
    //     let events = stream.get_events(None);
    //     assert!(events.is_empty());
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_config_parameter_updates() {
    //     let mut config = RealtimeStreamConfig::default();
    //     let device = Device::Cpu;
    //     let mut stream = RealtimeEventStream::new(config.clone(), device).unwrap();
    //
    //     // Test updating contrast threshold
    //     let new_threshold = 0.25;
    //     let result = stream.update_esim_params(Some(new_threshold), None);
    //     assert!(result.is_ok());
    //
    //     // Test updating FPS
    //     let new_fps = 15.0;
    //     let result = stream.update_esim_params(None, Some(new_fps));
    //     assert!(result.is_ok());
    //
    //     // Test updating both
    //     let result = stream.update_esim_params(Some(0.3), Some(25.0));
    //     assert!(result.is_ok());
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_performance_optimised_config() {
    //     let config = RealtimeStreamConfig::default();
    //
    //     // Verify configuration is optimised for real-time performance
    //     let esim_config = &config.esim_config;
    //
    //     // These features should be disabled for performance
    //     assert!(
    //         !esim_config.use_bilinear_interpolation,
    //         "Bilinear interpolation should be disabled for real-time performance"
    //     );
    //     assert!(
    //         !esim_config.adaptive_thresholding,
    //         "Adaptive thresholding should be disabled for real-time performance"
    //     );
    //
    //     // Resolution should be reasonable for real-time processing
    //     let (width, height) = esim_config.base_config.resolution;
    //     assert!(
    //         width <= 1920 && height <= 1080,
    //         "Resolution {}x{} may be too high for real-time processing",
    //         width,
    //         height
    //     );
    //
    //     // Contrast thresholds should be reasonable
    //     assert!(
    //         esim_config.base_config.contrast_threshold_pos >= 0.1,
    //         "Contrast threshold too low for stable real-time processing"
    //     );
    //     assert!(
    //         esim_config.base_config.contrast_threshold_pos <= 0.5,
    //         "Contrast threshold too high, may miss events"
    //     );
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_buffer_overflow_protection() {
    //     let mut config = RealtimeStreamConfig::default();
    //     config.max_buffer_size = 5; // Very small buffer for testing
    //
    //     let device = Device::Cpu;
    //     let stream = RealtimeEventStream::new(config, device).unwrap();
    //
    //     // Create some mock events (we can't actually generate events without frames)
    //     let empty_events = stream.get_events(Some(10));
    //     assert!(empty_events.is_empty());
    //
    //     // Verify buffer size is tracked correctly
    //     let stats = stream.get_stats();
    //     assert_eq!(stats.buffer_size, 0);
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_fps_adjustment_logic() {
    //     let mut config = RealtimeStreamConfig::default();
    //     config.auto_adjust_fps = true;
    //     config.target_fps = 30.0;
    //
    //     let device = Device::Cpu;
    //     let stream = RealtimeEventStream::new(config, device).unwrap();
    //
    //     // Verify auto-adjustment is enabled
    //     assert!(stream.get_config().auto_adjust_fps);
    //     assert_eq!(stream.get_config().target_fps, 30.0);
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_stream_lifecycle() {
    //     let config = RealtimeStreamConfig::default();
    //     let device = Device::Cpu;
    //     let mut stream = RealtimeEventStream::new(config, device).unwrap();
    //
    //     // Initially not streaming
    //     assert!(!stream.is_streaming());
    //
    //     // Reset should work even when not streaming
    //     stream.reset();
    //     let stats = stream.get_stats();
    //     assert_eq!(stats.frames_processed, 0);
    //
    //     // Note: We can't test actual streaming without GStreamer setup
    //     // but we can test the state management
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_esim_config_inheritance() {
    //     let mut esim_config = EsimConfig::default();
    //     esim_config.base_config.contrast_threshold_pos = 0.35;
    //     esim_config.base_config.resolution = (320, 240);
    //
    //     let mut realtime_config = RealtimeStreamConfig::default();
    //     realtime_config.esim_config = esim_config;
    //
    //     let device = Device::Cpu;
    //     let stream = RealtimeEventStream::new(realtime_config, device).unwrap();
    //
    //     // Verify ESIM config was properly inherited
    //     let config = stream.get_config();
    //     assert_eq!(config.esim_config.base_config.contrast_threshold_pos, 0.35);
    //     assert_eq!(config.esim_config.base_config.resolution, (320, 240));
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_thread_safety_basics() {
    //     use std::sync::Arc;
    //     use std::thread;
    //
    //     let config = RealtimeStreamConfig::default();
    //     let device = Device::Cpu;
    //     let manager = Arc::new(EventStreamManager::new(config, device).unwrap());
    //
    //     // Test that we can access stats from different threads
    //     let manager_clone = Arc::clone(&manager);
    //     let handle = thread::spawn(move || {
    //         let stats = manager_clone.get_stats();
    //         assert_eq!(stats.frames_processed, 0);
    //     });
    //
    //     handle.join().unwrap();
    //
    //     // Original manager should still be accessible
    //     let stats = manager.get_stats();
    //     assert_eq!(stats.frames_processed, 0);
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_error_handling() {
    //     let config = RealtimeStreamConfig::default();
    //     let device = Device::Cpu;
    //     let mut stream = RealtimeEventStream::new(config, device).unwrap();
    //
    //     // Should return error when trying to process frame without streaming
    //     let result = stream.process_next_frame();
    //     assert!(result.is_err());
    //
    //     // Should return error message about not streaming
    //     if let Err(e) = result {
    //         let error_msg = format!("{:?}", e);
    //         assert!(error_msg.contains("Not currently streaming"));
    //     }
    // }

    #[cfg(not(feature = "gstreamer"))]
    #[test]
    fn test_gstreamer_not_available() {
        // This test verifies that the code compiles even without GStreamer
        // In a real scenario without GStreamer, the types wouldn't be available
        println!("GStreamer feature not enabled - real-time streaming not available");
        assert!(true); // Test passes - we just want to ensure compilation works
    }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_performance_benchmarking_setup() {
    //     // This test sets up basic performance measurement infrastructure
    //     let config = RealtimeStreamConfig::default();
    //     let device = Device::Cpu;
    //     let stream = RealtimeEventStream::new(config, device).unwrap();
    //
    //     let start_time = Instant::now();
    //
    //     // Simulate some basic operations
    //     let _stats = stream.get_stats();
    //     let _events = stream.get_events(Some(100));
    //     let _is_streaming = stream.is_streaming();
    //
    //     let elapsed = start_time.elapsed();
    //
    //     // Basic operations should be very fast (< 1ms)
    //     assert!(
    //         elapsed < Duration::from_millis(1),
    //         "Basic operations took too long: {:?}",
    //         elapsed
    //     );
    //
    //     println!("Basic operations completed in: {:?}", elapsed);
    // }

    // DISABLED: Dependencies ev_simulation and candle_core removed
    // #[cfg(feature = "gstreamer")]
    // #[test]
    // fn test_memory_usage_awareness() {
    //     let mut config = RealtimeStreamConfig::default();
    //
    //     // Test with different buffer sizes
    //     let buffer_sizes = vec![100, 1000, 10000];
    //
    //     for buffer_size in buffer_sizes {
    //         config.max_buffer_size = buffer_size;
    //         let device = Device::Cpu;
    //         let stream = RealtimeEventStream::new(config.clone(), device).unwrap();
    //
    //         // Verify configuration was applied
    //         assert_eq!(stream.get_config().max_buffer_size, buffer_size);
    //
    //         // Initial buffer should be empty
    //         let stats = stream.get_stats();
    //         assert_eq!(stats.buffer_size, 0);
    //     }
    // }
}
