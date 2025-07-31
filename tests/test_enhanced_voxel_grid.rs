use evlib::ev_core::{Event, Events};
use evlib::ev_representations::to_voxel_grid_enhanced;
#[cfg(feature = "polars")]
use evlib::ev_representations::to_voxel_grid_enhanced_polars;
use ndarray::{s, Array4};

#[test]
fn test_enhanced_voxel_grid_empty_events() {
    let events: Events = vec![];
    let sensor_size = (10, 10, 2);
    let n_time_bins = 5;

    let result = to_voxel_grid_enhanced(&events, sensor_size, n_time_bins);
    assert!(result.is_ok());

    let voxel_grid = result.unwrap();
    assert_eq!(voxel_grid.shape(), &[5, 2, 10, 10]);

    // All values should be zero
    for &value in voxel_grid.iter() {
        assert_eq!(value, 0.0);
    }
}

#[test]
fn test_enhanced_voxel_grid_invalid_polarity_channels() {
    let events: Events = vec![Event {
        x: 5,
        y: 5,
        t: 1.0,
        polarity: true,
    }];
    let sensor_size = (10, 10, 3); // Invalid: should be 2
    let n_time_bins = 5;

    let result = to_voxel_grid_enhanced(&events, sensor_size, n_time_bins);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Expected 2 polarity channels"));
}

#[test]
fn test_enhanced_voxel_grid_single_event() {
    let events: Events = vec![Event {
        x: 5,
        y: 7,
        t: 1.5,
        polarity: true,
    }];
    let sensor_size = (10, 10, 2);
    let n_time_bins = 10;

    let result = to_voxel_grid_enhanced(&events, sensor_size, n_time_bins);
    assert!(result.is_ok());

    let voxel_grid = result.unwrap();
    assert_eq!(voxel_grid.shape(), &[10, 2, 10, 10]);

    // Check that the event is only in the positive polarity channel (channel 1)
    let positive_channel = voxel_grid.slice(s![.., 1, .., ..]);
    let negative_channel = voxel_grid.slice(s![.., 0, .., ..]);

    // Negative channel should be empty
    for &value in negative_channel.iter() {
        assert_eq!(value, 0.0);
    }

    // Positive channel should have the event distributed across time bins
    let total_positive: f32 = positive_channel.iter().sum();
    assert!((total_positive - 1.0).abs() < 1e-6); // Should sum to 1.0 (polarity value)
}

#[test]
fn test_enhanced_voxel_grid_bilinear_interpolation() {
    // Create events that should be distributed between time bins
    let events: Events = vec![
        Event {
            x: 0,
            y: 0,
            t: 0.25,
            polarity: true,
        }, // 25% through time range
        Event {
            x: 1,
            y: 0,
            t: 0.75,
            polarity: false,
        }, // 75% through time range
    ];
    let sensor_size = (10, 10, 2);
    let n_time_bins = 4;

    let result = to_voxel_grid_enhanced(&events, sensor_size, n_time_bins);
    assert!(result.is_ok());

    let voxel_grid = result.unwrap();

    // For t=0.25 with 4 bins: normalised t = 1.0, so t_int=1, t_frac=0.0
    // Should go entirely to bin 1 with value 1.0
    assert!((voxel_grid[[1, 1, 0, 0]] - 1.0).abs() < 1e-6);

    // For t=0.75 with 4 bins: normalised t = 3.0, so t_int=3, t_frac=0.0
    // Should go entirely to bin 3 with value -1.0
    assert!((voxel_grid[[3, 0, 0, 1]] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_enhanced_voxel_grid_fractional_interpolation() {
    // Create events with timestamps that require fractional interpolation
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
            t: 1.0,
            polarity: true,
        },
    ];
    let sensor_size = (10, 10, 2);
    let n_time_bins = 3;

    let result = to_voxel_grid_enhanced(&events, sensor_size, n_time_bins);
    assert!(result.is_ok());

    let voxel_grid = result.unwrap();

    // First event at t=0.0: normalised t = 0.0, should go to bin 0
    // Second event at t=1.0: normalised t = 3.0, but clamped to bin 2
    // Each event contributes polarity value 1.0

    let total_in_positive_channel: f32 = voxel_grid.slice(s![.., 1, 0, 0]).iter().sum();
    assert!((total_in_positive_channel - 2.0).abs() < 1e-6);
}

#[test]
fn test_enhanced_voxel_grid_mixed_polarities() {
    let events: Events = vec![
        Event {
            x: 5,
            y: 5,
            t: 0.0,
            polarity: true,
        }, // +1
        Event {
            x: 5,
            y: 5,
            t: 0.5,
            polarity: false,
        }, // -1
        Event {
            x: 5,
            y: 5,
            t: 1.0,
            polarity: true,
        }, // +1
    ];
    let sensor_size = (10, 10, 2);
    let n_time_bins = 2;

    let result = to_voxel_grid_enhanced(&events, sensor_size, n_time_bins);
    assert!(result.is_ok());

    let voxel_grid = result.unwrap();

    // Check that polarities are handled correctly
    let pos_channel_total: f32 = voxel_grid.slice(s![.., 1, 5, 5]).iter().sum();
    let neg_channel_total: f32 = voxel_grid.slice(s![.., 0, 5, 5]).iter().sum();

    assert!((pos_channel_total - 2.0).abs() < 1e-6); // Two positive events
    assert!((neg_channel_total - (-1.0)).abs() < 1e-6); // One negative event
}

#[test]
fn test_enhanced_voxel_grid_out_of_bounds_clipping() {
    let events: Events = vec![
        Event {
            x: 15,
            y: 15,
            t: 0.5,
            polarity: true,
        }, // Out of bounds
        Event {
            x: 5,
            y: 5,
            t: 0.5,
            polarity: true,
        }, // In bounds
    ];
    let sensor_size = (10, 10, 2);
    let n_time_bins = 2;

    let result = to_voxel_grid_enhanced(&events, sensor_size, n_time_bins);
    assert!(result.is_ok());

    let voxel_grid = result.unwrap();

    // Only the in-bounds event should contribute
    let total_value: f32 = voxel_grid.iter().sum();
    assert!((total_value - 1.0).abs() < 1e-6);

    // Check that the value is at the correct location
    let pos_channel_value: f32 = voxel_grid.slice(s![.., 1, 5, 5]).iter().sum();
    assert!((pos_channel_value - 1.0).abs() < 1e-6);
}

#[test]
fn test_enhanced_voxel_grid_same_timestamp() {
    // Test edge case where all events have the same timestamp
    let events: Events = vec![
        Event {
            x: 0,
            y: 0,
            t: 1.0,
            polarity: true,
        },
        Event {
            x: 1,
            y: 1,
            t: 1.0,
            polarity: false,
        },
        Event {
            x: 2,
            y: 2,
            t: 1.0,
            polarity: true,
        },
    ];
    let sensor_size = (10, 10, 2);
    let n_time_bins = 5;

    let result = to_voxel_grid_enhanced(&events, sensor_size, n_time_bins);
    assert!(result.is_ok());

    let voxel_grid = result.unwrap();

    // All events should go to the first time bin
    let first_bin_total: f32 = voxel_grid.slice(s![0, .., .., ..]).iter().sum();
    let other_bins_total: f32 = voxel_grid.slice(s![1.., .., .., ..]).iter().sum();

    assert!((first_bin_total - 1.0).abs() < 1e-6); // 2 positive - 1 negative = 1
    assert!((other_bins_total - 0.0).abs() < 1e-6); // Other bins should be empty
}

#[cfg(feature = "polars")]
#[test]
fn test_enhanced_voxel_grid_polars_empty_events() {
    let events: Events = vec![];
    let sensor_size = (10, 10, 2);
    let n_time_bins = 5;

    let result = to_voxel_grid_enhanced_polars(&events, sensor_size, n_time_bins);
    assert!(result.is_ok());

    let lazy_frame = result.unwrap();
    let df = lazy_frame.collect().unwrap();

    // Empty events should produce empty DataFrame
    assert_eq!(df.height(), 0);

    // But it should have the correct schema
    assert!(df.column("time_bin").is_ok());
    assert!(df.column("polarity_channel").is_ok());
    assert!(df.column("y").is_ok());
    assert!(df.column("x").is_ok());
    assert!(df.column("value").is_ok());
}

#[cfg(feature = "polars")]
#[test]
fn test_enhanced_voxel_grid_polars_single_event() {
    let events: Events = vec![Event {
        x: 5,
        y: 7,
        t: 1.5,
        polarity: true,
    }];
    let sensor_size = (10, 10, 2);
    let n_time_bins = 10;

    let result = to_voxel_grid_enhanced_polars(&events, sensor_size, n_time_bins);
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

    // Total value should sum to 1.0 (polarity value)
    let values: Vec<f32> = df
        .column("value")
        .unwrap()
        .f32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let total_value: f32 = values.iter().sum();
    assert!((total_value - 1.0).abs() < 1e-6);
}

#[cfg(feature = "polars")]
#[test]
fn test_enhanced_voxel_grid_polars_consistency_with_array() {
    // Test that Polars and Array implementations produce consistent results
    let events: Events = vec![
        Event {
            x: 0,
            y: 0,
            t: 0.0,
            polarity: true,
        },
        Event {
            x: 1,
            y: 1,
            t: 0.5,
            polarity: false,
        },
        Event {
            x: 2,
            y: 2,
            t: 1.0,
            polarity: true,
        },
    ];
    let sensor_size = (5, 5, 2);
    let n_time_bins = 3;

    // Get results from both implementations
    let array_result = to_voxel_grid_enhanced(&events, sensor_size, n_time_bins).unwrap();
    let polars_result = to_voxel_grid_enhanced_polars(&events, sensor_size, n_time_bins).unwrap();
    let polars_df = polars_result.collect().unwrap();

    // Convert Polars result to dense representation for comparison
    let mut polars_dense = Array4::<f32>::zeros((
        n_time_bins,
        2,
        sensor_size.1 as usize,
        sensor_size.0 as usize,
    ));

    // Fill dense array from Polars DataFrame
    let time_bins: Vec<i32> = polars_df
        .column("time_bin")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let polarity_channels: Vec<i32> = polars_df
        .column("polarity_channel")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let y_coords: Vec<i32> = polars_df
        .column("y")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let x_coords: Vec<i32> = polars_df
        .column("x")
        .unwrap()
        .i32()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let values: Vec<f32> = polars_df
        .column("value")
        .unwrap()
        .f32()
        .unwrap()
        .into_no_null_iter()
        .collect();

    for i in 0..polars_df.height() {
        let t = time_bins[i] as usize;
        let p = polarity_channels[i] as usize;
        let y = y_coords[i] as usize;
        let x = x_coords[i] as usize;
        let value = values[i];

        polars_dense[[t, p, y, x]] += value;
    }

    // Compare the two representations
    // Allow for small floating point differences
    for t in 0..n_time_bins {
        for p in 0..2 {
            for y in 0..sensor_size.1 as usize {
                for x in 0..sensor_size.0 as usize {
                    let array_val = array_result[[t, p, y, x]];
                    let polars_val = polars_dense[[t, p, y, x]];
                    assert!(
                        (array_val - polars_val).abs() < 1e-6,
                        "Mismatch at [{}, {}, {}, {}]: array={}, polars={}",
                        t,
                        p,
                        y,
                        x,
                        array_val,
                        polars_val
                    );
                }
            }
        }
    }
}
