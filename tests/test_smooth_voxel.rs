//! Tests for smooth voxel grid representation

#[cfg(test)]
mod tests {
    use evlib::ev_core::{Event, Events};
    use evlib::ev_representations::smooth_voxel_grid::{
        events_to_smooth_voxel_grid, events_to_trilinear_voxel_grid, SmoothEventsToVoxelGrid,
    };

    /// Create some test events for use in tests
    fn create_test_events() -> Events {
        let mut events = Events::with_capacity(5);
        events.push(Event {
            x: 10,
            y: 15,
            t: 0.1,
            polarity: 1,
        });
        events.push(Event {
            x: 20,
            y: 25,
            t: 0.2,
            polarity: -1,
        });
        events.push(Event {
            x: 30,
            y: 35,
            t: 0.3,
            polarity: 1,
        });
        events.push(Event {
            x: 40,
            y: 45,
            t: 0.4,
            polarity: -1,
        });
        events.push(Event {
            x: 50,
            y: 55,
            t: 0.5,
            polarity: 1,
        });
        events
    }

    #[test]
    fn test_smooth_voxel_grid_constructor() {
        let converter = SmoothEventsToVoxelGrid::new(5, 100, 100, Some("trilinear".to_string()));
        assert_eq!(converter.num_bins, 5);
        assert_eq!(converter.width, 100);
        assert_eq!(converter.height, 100);
        assert_eq!(converter.interpolation, "trilinear");

        // Test default interpolation
        let converter = SmoothEventsToVoxelGrid::new(5, 100, 100, None);
        assert_eq!(converter.interpolation, "trilinear");
    }

    #[test]
    fn test_empty_events() {
        let events = Events::new();
        let result = events_to_smooth_voxel_grid(&events, 5, 100, 100, None).unwrap();
        let shape = result.shape();

        assert_eq!(shape.dims().len(), 3);
        assert_eq!(shape.dims()[0], 5);
        assert_eq!(shape.dims()[1], 100);
        assert_eq!(shape.dims()[2], 100);

        // Verify all values are zero
        let data = result.to_vec1().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_trilinear_interpolation() {
        let events = create_test_events();
        let result =
            events_to_smooth_voxel_grid(&events, 5, 100, 100, Some("trilinear".to_string()))
                .unwrap();
        let shape = result.shape();

        assert_eq!(shape.dims().len(), 3);
        assert_eq!(shape.dims()[0], 5);
        assert_eq!(shape.dims()[1], 100);
        assert_eq!(shape.dims()[2], 100);

        // Verify the grid has non-zero values
        let data = result.to_vec1().unwrap();
        assert!(data.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_bilinear_interpolation() {
        let events = create_test_events();
        let result =
            events_to_smooth_voxel_grid(&events, 5, 100, 100, Some("bilinear".to_string()))
                .unwrap();
        let shape = result.shape();

        assert_eq!(shape.dims().len(), 3);
        assert_eq!(shape.dims()[0], 5);
        assert_eq!(shape.dims()[1], 100);
        assert_eq!(shape.dims()[2], 100);

        // Verify the grid has non-zero values
        let data = result.to_vec1().unwrap();
        assert!(data.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_temporal_interpolation() {
        let events = create_test_events();
        let result =
            events_to_smooth_voxel_grid(&events, 5, 100, 100, Some("temporal".to_string()))
                .unwrap();
        let shape = result.shape();

        assert_eq!(shape.dims().len(), 3);
        assert_eq!(shape.dims()[0], 5);
        assert_eq!(shape.dims()[1], 100);
        assert_eq!(shape.dims()[2], 100);

        // Verify the grid has non-zero values
        let data = result.to_vec1().unwrap();
        assert!(data.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_convenience_function() {
        let events = create_test_events();
        let result = events_to_trilinear_voxel_grid(&events, 5, 100, 100).unwrap();
        let shape = result.shape();

        assert_eq!(shape.dims().len(), 3);
        assert_eq!(shape.dims()[0], 5);
        assert_eq!(shape.dims()[1], 100);
        assert_eq!(shape.dims()[2], 100);

        // Verify the grid has non-zero values
        let data = result.to_vec1().unwrap();
        assert!(data.iter().any(|&x| x != 0.0));
    }
}
