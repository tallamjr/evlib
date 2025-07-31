use evlib::ev_core::{Event, Events};
use evlib::ev_representations::{
    to_averaged_timesurface_enhanced, to_timesurface_enhanced, DecayType, TimeSurfaceConfig,
};
#[cfg(feature = "polars")]
use evlib::ev_representations::{
    to_averaged_timesurface_enhanced_polars, to_timesurface_enhanced_polars,
};
use ndarray::s;

// ============================================================================
// Tests for Basic Time Surface (to_timesurface_enhanced)
// ============================================================================

#[test]
fn test_timesurface_empty_events() {
    let events: Events = vec![];
    let sensor_size = (10, 10, 2);
    let dt = 1000.0; // 1ms
    let tau = 5000.0; // 5ms

    let result = to_timesurface_enhanced(&events, sensor_size, dt, tau, None, false);
    assert!(result.is_ok());

    let time_surfaces = result.unwrap();
    assert_eq!(time_surfaces.shape(), &[1, 2, 10, 10]);

    // All values should be zero
    for &value in time_surfaces.iter() {
        assert_eq!(value, 0.0);
    }
}

#[test]
fn test_timesurface_invalid_polarity_channels() {
    let events: Events = vec![Event {
        x: 5,
        y: 5,
        t: 1000.0,
        polarity: true,
    }];
    let sensor_size = (10, 10, 3); // Invalid: should be 1 or 2
    let dt = 1000.0;
    let tau = 5000.0;

    let result = to_timesurface_enhanced(&events, sensor_size, dt, tau, None, false);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Expected 1 or 2 polarity channels"));
}

#[test]
fn test_timesurface_invalid_parameters() {
    let events: Events = vec![Event {
        x: 5,
        y: 5,
        t: 1000.0,
        polarity: true,
    }];
    let sensor_size = (10, 10, 2);

    // Test negative dt
    let result = to_timesurface_enhanced(&events, sensor_size, -1000.0, 5000.0, None, false);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("dt and tau must be positive"));

    // Test negative tau
    let result = to_timesurface_enhanced(&events, sensor_size, 1000.0, -5000.0, None, false);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("dt and tau must be positive"));
}

#[test]
fn test_timesurface_single_event_single_slice() {
    let events: Events = vec![Event {
        x: 5,
        y: 7,
        t: 1000.0,
        polarity: true,
    }];
    let sensor_size = (10, 10, 2);
    let dt = 2000.0; // 2ms - large enough to include the event
    let tau = 5000.0; // 5ms

    let result = to_timesurface_enhanced(&events, sensor_size, dt, tau, None, false);
    assert!(result.is_ok());

    let time_surfaces = result.unwrap();
    assert_eq!(time_surfaces.shape(), &[1, 2, 10, 10]);

    // Check that the event appears in the positive polarity channel (channel 1)
    let surface_value = time_surfaces[[0, 1, 7, 5]]; // Note: y=7, x=5
    assert!(surface_value > 0.0); // Should have some decay value
    assert!(surface_value <= 1.0); // Should not exceed 1.0

    // Negative channel should be zero at this location
    assert_eq!(time_surfaces[[0, 0, 7, 5]], 0.0);
}

#[test]
fn test_timesurface_exponential_decay() {
    // Create events with known time differences to test decay
    let events: Events = vec![
        Event {
            x: 0,
            y: 0,
            t: 0.0,
            polarity: true,
        },
        Event {
            x: 0,
            y: 0,
            t: 5000.0,
            polarity: true,
        }, // 5ms later
    ];
    let sensor_size = (10, 10, 2);
    let dt = 10000.0; // 10ms - covers both events
    let tau = 5000.0; // 5ms decay constant

    let result = to_timesurface_enhanced(&events, sensor_size, dt, tau, None, false);
    assert!(result.is_ok());

    let time_surfaces = result.unwrap();

    // The surface value should be exp(-0/5000) = 1.0 at the end of the slice
    // since the last event at t=5000 is at the pixel
    let surface_value = time_surfaces[[0, 1, 0, 0]];
    let expected_decay = (-5000.0 / tau).exp(); // exp(-1) ≈ 0.368

    // Should be close to the expected exponential decay
    assert!((surface_value - expected_decay as f32).abs() < 1e-3);
}

#[test]
fn test_timesurface_multiple_time_slices() {
    let events: Events = vec![
        Event {
            x: 0,
            y: 0,
            t: 500.0,
            polarity: true,
        }, // First slice
        Event {
            x: 1,
            y: 0,
            t: 1500.0,
            polarity: false,
        }, // Second slice
        Event {
            x: 2,
            y: 0,
            t: 2500.0,
            polarity: true,
        }, // Third slice
    ];
    let sensor_size = (10, 10, 2);
    let dt = 1000.0; // 1ms slices
    let tau = 2000.0; // 2ms decay

    let result = to_timesurface_enhanced(&events, sensor_size, dt, tau, None, false);
    assert!(result.is_ok());

    let time_surfaces = result.unwrap();
    assert_eq!(time_surfaces.shape()[0], 3); // Should have 3 time slices

    // Check that events appear in correct time slices and channels
    assert!(time_surfaces[[0, 1, 0, 0]] > 0.0); // First event, positive channel
    assert!(time_surfaces[[1, 0, 0, 1]] > 0.0); // Second event, negative channel
    assert!(time_surfaces[[2, 1, 0, 2]] > 0.0); // Third event, positive channel
}

#[test]
fn test_timesurface_overlap() {
    let events: Events = vec![
        Event {
            x: 0,
            y: 0,
            t: 0.0,
            polarity: true,
        },
        Event {
            x: 1,
            y: 0,
            t: 800.0,
            polarity: true,
        },
        Event {
            x: 2,
            y: 0,
            t: 1600.0,
            polarity: true,
        },
    ];
    let sensor_size = (10, 10, 2);
    let dt = 1000.0; // 1ms slices
    let tau = 2000.0;
    let overlap = Some(500); // 0.5ms overlap

    let result = to_timesurface_enhanced(&events, sensor_size, dt, tau, overlap, false);
    assert!(result.is_ok());

    let time_surfaces = result.unwrap();

    // With overlap, should have more slices covering the events
    assert!(time_surfaces.shape()[0] >= 2);

    // Events should appear in overlapping time slices
    let first_slice_total: f32 = time_surfaces.slice(s![0, 1, .., ..]).iter().sum();
    assert!(first_slice_total > 0.0);
}

#[test]
fn test_timesurface_single_polarity_sensor() {
    let events: Events = vec![
        Event {
            x: 0,
            y: 0,
            t: 1000.0,
            polarity: true,
        },
        Event {
            x: 1,
            y: 0,
            t: 2000.0,
            polarity: true,
        },
    ];
    let sensor_size = (10, 10, 1); // Single polarity sensor
    let dt = 5000.0;
    let tau = 2000.0;

    let result = to_timesurface_enhanced(&events, sensor_size, dt, tau, None, false);
    assert!(result.is_ok());

    let time_surfaces = result.unwrap();
    assert_eq!(time_surfaces.shape(), &[1, 1, 10, 10]); // Only one polarity channel

    // Both events should appear in the single channel
    assert!(time_surfaces[[0, 0, 0, 0]] > 0.0);
    assert!(time_surfaces[[0, 0, 0, 1]] > 0.0);
}

#[test]
fn test_timesurface_out_of_bounds_clipping() {
    let events: Events = vec![
        Event {
            x: 15,
            y: 15,
            t: 1000.0,
            polarity: true,
        }, // Out of bounds
        Event {
            x: 5,
            y: 5,
            t: 1000.0,
            polarity: true,
        }, // In bounds
    ];
    let sensor_size = (10, 10, 2);
    let dt = 2000.0;
    let tau = 5000.0;

    let result = to_timesurface_enhanced(&events, sensor_size, dt, tau, None, false);
    assert!(result.is_ok());

    let time_surfaces = result.unwrap();

    // Only the in-bounds event should contribute
    assert!(time_surfaces[[0, 1, 5, 5]] > 0.0);

    // Out of bounds location should remain zero
    let total_nonzero: usize = time_surfaces.iter().filter(|&&x| x > 0.0).count();
    assert_eq!(total_nonzero, 1); // Only one pixel should be non-zero
}

// ============================================================================
// Tests for Averaged Time Surface (HATS) - to_averaged_timesurface_enhanced
// ============================================================================

#[test]
fn test_averaged_timesurface_empty_events() {
    let events: Events = vec![];
    let sensor_size = (20, 20, 2);
    let config = TimeSurfaceConfig::new(10, 5, 10000.0, 5000.0);

    let result = to_averaged_timesurface_enhanced(&events, sensor_size, config);
    assert!(result.is_ok());

    let histograms = result.unwrap();
    // Should have 2x2 = 4 cells for a 20x20 sensor with cell_size=10
    assert_eq!(histograms.shape(), &[4, 2, 5, 5]);

    // All values should be zero
    for &value in histograms.iter() {
        assert_eq!(value, 0.0);
    }
}

#[test]
fn test_averaged_timesurface_invalid_config() {
    let events: Events = vec![Event {
        x: 5,
        y: 5,
        t: 1000.0,
        polarity: true,
    }];
    let sensor_size = (20, 20, 2);

    // Test surface_size > cell_size
    let invalid_config = TimeSurfaceConfig::new(5, 10, 10000.0, 5000.0);
    let result = to_averaged_timesurface_enhanced(&events, sensor_size, invalid_config);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("surface_size must be <= cell_size"));

    // Test even surface_size
    let invalid_config = TimeSurfaceConfig::new(10, 4, 10000.0, 5000.0);
    let result = to_averaged_timesurface_enhanced(&events, sensor_size, invalid_config);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("surface_size must be odd"));
}

#[test]
fn test_averaged_timesurface_single_event() {
    let events: Events = vec![Event {
        x: 5,
        y: 5,
        t: 1000.0,
        polarity: true,
    }];
    let sensor_size = (20, 20, 2);
    let config = TimeSurfaceConfig::new(10, 3, 10000.0, 5000.0);

    let result = to_averaged_timesurface_enhanced(&events, sensor_size, config);
    assert!(result.is_ok());

    let histograms = result.unwrap();
    assert_eq!(histograms.shape(), &[4, 2, 3, 3]);

    // Event at (5,5) should be in cell 0 (top-left cell)
    // and should appear in the positive polarity channel
    let cell_0_pos_channel = histograms.slice(s![0, 1, .., ..]);

    // The center of the surface (1,1) should have value 1.0 from the current event
    assert!((cell_0_pos_channel[(1, 1)] - 1.0).abs() < 1e-6);

    // Other cells should be empty
    for cell_id in 1..4 {
        let cell_values: f32 = histograms.slice(s![cell_id, .., .., ..]).iter().sum();
        assert_eq!(cell_values, 0.0);
    }
}

#[test]
fn test_averaged_timesurface_exponential_decay() {
    let events: Events = vec![
        Event {
            x: 5,
            y: 5,
            t: 0.0,
            polarity: true,
        }, // Earlier event
        Event {
            x: 5,
            y: 5,
            t: 5000.0,
            polarity: true,
        }, // Later event, 5ms after
    ];
    let sensor_size = (20, 20, 2);
    let config = TimeSurfaceConfig::new(10, 3, 10000.0, 5000.0) // tau = 5ms
        .with_decay(DecayType::Exponential);

    let result = to_averaged_timesurface_enhanced(&events, sensor_size, config);
    assert!(result.is_ok());

    let histograms = result.unwrap();

    // Both events are in the same cell and same polarity
    let cell_0_pos_channel = histograms.slice(s![0, 1, .., ..]);

    // The center pixel should have accumulated value from both events
    // The first event contributes exp(-5000/5000) = exp(-1) ≈ 0.368 when processing the second event
    // The second event contributes 1.0
    // Total averaged over 2 events: (0.368 + 1.0 + 1.0) / 2 = 1.184
    let center_value = cell_0_pos_channel[(1, 1)];
    assert!(center_value > 1.0); // Should be greater than 1 due to accumulation
}

#[test]
fn test_averaged_timesurface_linear_decay() {
    let events: Events = vec![
        Event {
            x: 5,
            y: 5,
            t: 0.0,
            polarity: true,
        },
        Event {
            x: 5,
            y: 5,
            t: 3000.0,
            polarity: true,
        }, // 3ms later
    ];
    let sensor_size = (20, 20, 2);
    let config = TimeSurfaceConfig::new(10, 3, 10000.0, 5000.0) // tau = 5ms
        .with_decay(DecayType::Linear);

    let result = to_averaged_timesurface_enhanced(&events, sensor_size, config);
    assert!(result.is_ok());

    let histograms = result.unwrap();

    let cell_0_pos_channel = histograms.slice(s![0, 1, .., ..]);

    // Linear decay: -(3000)/(3*5000) + 1 = -3000/15000 + 1 = 0.8
    // Total accumulated: (0.8 + 1.0 + 1.0) / 2 = 1.4
    let center_value = cell_0_pos_channel[(1, 1)];
    assert!(center_value > 1.0);
}

#[test]
fn test_averaged_timesurface_spatial_neighborhood() {
    let events: Events = vec![
        Event {
            x: 5,
            y: 5,
            t: 0.0,
            polarity: true,
        }, // Center event
        Event {
            x: 6,
            y: 5,
            t: 1000.0,
            polarity: true,
        }, // Neighbor event (within surface_size=3)
    ];
    let sensor_size = (20, 20, 2);
    let config = TimeSurfaceConfig::new(10, 3, 10000.0, 5000.0);

    let result = to_averaged_timesurface_enhanced(&events, sensor_size, config);
    assert!(result.is_ok());

    let histograms = result.unwrap();

    let cell_0_pos_channel = histograms.slice(s![0, 1, .., ..]);

    // When processing the second event at (6,5), the first event at (5,5) should
    // contribute to the time surface at relative position (5-6, 5-5) = (-1, 0)
    // which maps to surface coordinates (1-1, 1+0) = (0, 1)
    assert!(cell_0_pos_channel[(1, 0)] > 0.0); // Should have decay contribution
    assert!(cell_0_pos_channel[(1, 1)] > 0.0); // Should have the current event
}

#[test]
fn test_averaged_timesurface_multiple_cells() {
    let events: Events = vec![
        Event {
            x: 5,
            y: 5,
            t: 1000.0,
            polarity: true,
        }, // Cell 0
        Event {
            x: 15,
            y: 5,
            t: 1000.0,
            polarity: false,
        }, // Cell 1
        Event {
            x: 5,
            y: 15,
            t: 1000.0,
            polarity: true,
        }, // Cell 2
        Event {
            x: 15,
            y: 15,
            t: 1000.0,
            polarity: false,
        }, // Cell 3
    ];
    let sensor_size = (20, 20, 2);
    let config = TimeSurfaceConfig::new(10, 3, 10000.0, 5000.0);

    let result = to_averaged_timesurface_enhanced(&events, sensor_size, config);
    assert!(result.is_ok());

    let histograms = result.unwrap();

    // Each cell should have exactly one event
    for cell_id in 0..4 {
        let cell_total: f32 = histograms.slice(s![cell_id, .., .., ..]).iter().sum();
        assert!((cell_total - 1.0).abs() < 1e-6); // Each cell should sum to 1.0
    }

    // Check polarity distribution
    assert!(histograms[(0, 1, 1, 1)] > 0.0); // Cell 0, positive
    assert!(histograms[(1, 0, 1, 1)] > 0.0); // Cell 1, negative
    assert!(histograms[(2, 1, 1, 1)] > 0.0); // Cell 2, positive
    assert!(histograms[(3, 0, 1, 1)] > 0.0); // Cell 3, negative
}

#[test]
fn test_averaged_timesurface_time_window_filtering() {
    let events: Events = vec![
        Event {
            x: 5,
            y: 5,
            t: 0.0,
            polarity: true,
        }, // Outside time window
        Event {
            x: 5,
            y: 5,
            t: 20000.0,
            polarity: true,
        }, // Current event
    ];
    let sensor_size = (20, 20, 2);
    let config = TimeSurfaceConfig::new(10, 3, 5000.0, 5000.0); // time_window = 5ms

    let result = to_averaged_timesurface_enhanced(&events, sensor_size, config);
    assert!(result.is_ok());

    let histograms = result.unwrap();

    let cell_0_pos_channel = histograms.slice(s![0, 1, .., ..]);

    // Only the current event should contribute since the first event is outside the time window
    // Total should be (1.0 + 1.0) / 2 = 1.0 (each event gets one surface, averaged)
    let center_value = cell_0_pos_channel[(1, 1)];
    assert!((center_value - 1.0).abs() < 1e-6);
}

// ============================================================================
// Polars Tests
// ============================================================================

#[cfg(feature = "polars")]
#[test]
fn test_timesurface_polars_empty_events() {
    let events: Events = vec![];
    let sensor_size = (10, 10, 2);
    let dt = 1000.0;
    let tau = 5000.0;

    let result = to_timesurface_enhanced_polars(&events, sensor_size, dt, tau, None, false);
    assert!(result.is_ok());

    let lazy_frame = result.unwrap();
    let df = lazy_frame.collect().unwrap();

    // Empty events should produce empty DataFrame
    assert_eq!(df.height(), 0);

    // But it should have the correct schema
    assert!(df.column("time_slice").is_ok());
    assert!(df.column("polarity_channel").is_ok());
    assert!(df.column("y").is_ok());
    assert!(df.column("x").is_ok());
    assert!(df.column("surface_value").is_ok());
}

#[cfg(feature = "polars")]
#[test]
fn test_timesurface_polars_single_event() {
    let events: Events = vec![Event {
        x: 5,
        y: 7,
        t: 1000.0,
        polarity: true,
    }];
    let sensor_size = (10, 10, 2);
    let dt = 2000.0;
    let tau = 5000.0;

    let result = to_timesurface_enhanced_polars(&events, sensor_size, dt, tau, None, false);
    assert!(result.is_ok());

    let lazy_frame = result.unwrap();
    let df = lazy_frame.collect().unwrap();

    // Should have at least one row
    assert!(df.height() > 0);

    // Check that polarity channel is correct (1 for positive)
    let polarity_channels: Vec<i32> = df
        .column("polarity_channel")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();

    for &channel in &polarity_channels {
        assert_eq!(channel, 1); // Positive events go to channel 1
    }

    // Check that coordinates are correct
    let x_coords: Vec<i32> = df
        .column("x")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let y_coords: Vec<i32> = df
        .column("y")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();

    for &x in &x_coords {
        assert_eq!(x, 5);
    }
    for &y in &y_coords {
        assert_eq!(y, 7);
    }

    // Surface values should be positive and <= 1.0
    let values: Vec<f32> = df
        .column("surface_value")
        .unwrap()
        .f32()
        .unwrap()
        .into_no_null_iter()
        .collect();

    for &value in &values {
        assert!(value > 0.0);
        assert!(value <= 1.0);
    }
}

#[cfg(feature = "polars")]
#[test]
fn test_averaged_timesurface_polars_empty_events() {
    let events: Events = vec![];
    let sensor_size = (20, 20, 2);
    let config = TimeSurfaceConfig::new(10, 5, 10000.0, 5000.0);

    let result = to_averaged_timesurface_enhanced_polars(&events, sensor_size, config);
    assert!(result.is_ok());

    let lazy_frame = result.unwrap();
    let df = lazy_frame.collect().unwrap();

    // Empty events should produce empty DataFrame
    assert_eq!(df.height(), 0);

    // But it should have the correct schema
    assert!(df.column("cell_id").is_ok());
    assert!(df.column("polarity_channel").is_ok());
    assert!(df.column("surface_y").is_ok());
    assert!(df.column("surface_x").is_ok());
    assert!(df.column("averaged_value").is_ok());
}

#[cfg(feature = "polars")]
#[test]
fn test_averaged_timesurface_polars_single_event() {
    let events: Events = vec![Event {
        x: 5,
        y: 5,
        t: 1000.0,
        polarity: true,
    }];
    let sensor_size = (20, 20, 2);
    let config = TimeSurfaceConfig::new(10, 3, 10000.0, 5000.0);

    let result = to_averaged_timesurface_enhanced_polars(&events, sensor_size, config);
    assert!(result.is_ok());

    let lazy_frame = result.unwrap();
    let df = lazy_frame.collect().unwrap();

    // Should have rows for surface coordinates
    assert!(df.height() > 0);

    // Check schema
    assert!(df.column("cell_id").is_ok());
    assert!(df.column("polarity_channel").is_ok());
    assert!(df.column("surface_y").is_ok());
    assert!(df.column("surface_x").is_ok());
    assert!(df.column("averaged_value").is_ok());

    // Check that we have the expected number of surface coordinates per cell
    // For a 3x3 surface, we should have 9 coordinates per cell per polarity
    let expected_rows = 4 * 2 * 9; // 4 cells * 2 polarities * 9 surface positions
    assert_eq!(df.height(), expected_rows);
}
