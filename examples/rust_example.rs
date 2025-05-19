use evlib::{
    ev_augmentation as augmentation,
    ev_core::{infer_resolution, Event, Events},
    // ev_processing is disabled pending API updates
    // ev_processing::{
    //     FlowParams, FlowWarp, ObjectiveFunction, VarianceObjective, WarpModel,
    //     grid_search_optimization,
    // },
    ev_formats as formats,
    ev_representations as representations,
    ev_visualization as visualization,
};
use image::{ImageFormat, RgbImage};
use std::error::Error;
use std::path::Path;

/// Example demonstrating core functionality of the evlib crate.
/// This shows how to load events, apply augmentations, create
/// representations, estimate motion, and visualize results.
fn main() -> Result<(), Box<dyn Error>> {
    println!("evlib Rust Example");

    // Create synthetic events for demonstration
    let events = create_synthetic_events(500);
    println!("Created {} synthetic events", events.len());

    // Infer resolution from events
    let resolution = infer_resolution(&events);
    println!("Inferred resolution: {} x {}", resolution.0, resolution.1);

    // Apply augmentations
    let mut augmented_events = events.clone();
    println!("Applying augmentations...");

    // Add random noise events
    let t_min = events.first().map(|e| e.t).unwrap_or(0.0);
    let t_max = events.last().map(|e| e.t).unwrap_or(1.0);
    augmentation::add_random_events(
        &mut augmented_events,
        100,
        resolution.0,
        resolution.1,
        (t_min, t_max),
    );
    println!(
        "Added 100 random events. New count: {}",
        augmented_events.len()
    );

    // Add correlated events (blurring)
    augmentation::add_correlated_events(&mut augmented_events, 50, 2.0, 0.001);
    println!(
        "Added 50 correlated events. New count: {}",
        augmented_events.len()
    );

    // Rotate events
    let cx = resolution.0 as f32 / 2.0;
    let cy = resolution.1 as f32 / 2.0;
    augmentation::rotate_events(&mut augmented_events, 45.0, cx, cy, true, Some(resolution));
    println!("Rotated events by 45 degrees");

    // Create representations
    println!("\nCreating representations...");

    // Create voxel grid
    let voxel_grid =
        representations::events_to_voxel_grid(&augmented_events, resolution, 5, "count")?;
    println!("Created voxel grid with shape: {:?}", voxel_grid.shape());

    // Create timestamp image
    let timestamp_img =
        representations::events_to_timestamp_image(&augmented_events, resolution, true, false)?;
    println!(
        "Created timestamp image with shape: {:?}",
        timestamp_img.shape()
    );

    // Create event frame
    let event_frame =
        representations::events_to_frame(&augmented_events, resolution, "polarity", true)?;
    println!("Created event frame with shape: {:?}", event_frame.shape());

    // Contrast maximization is disabled pending API updates
    println!("\nNote: Contrast maximization is disabled pending API updates.");

    // Visualization
    println!("\nCreating visualizations...");

    // Draw events to image
    let event_img = visualization::draw_events_to_image(&events, resolution, "polarity");
    println!("Created event visualization");

    // Draw augmented events to image
    let aug_event_img =
        visualization::draw_events_to_image(&augmented_events, resolution, "polarity_time");
    println!("Created augmented event visualization");

    // Save the images
    let output_dir = Path::new("output");
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)?;
    }

    event_img.save(output_dir.join("events.png"))?;
    aug_event_img.save(output_dir.join("augmented_events.png"))?;

    println!("\nSaved visualizations to the 'output' directory");
    println!("Example completed successfully!");

    Ok(())
}

/// Create synthetic spiral events for demonstration
fn create_synthetic_events(num_events: usize) -> Events {
    let mut events = Vec::with_capacity(num_events);
    let width = 128;
    let height = 128;
    let center_x = width / 2;
    let center_y = height / 2;

    for i in 0..num_events {
        // Create a spiral pattern
        let t = (i as f64) / (num_events as f64);
        let angle = t * 4.0 * std::f64::consts::PI;
        let radius = t * 40.0;

        let x = center_x as f64 + radius * angle.cos();
        let y = center_y as f64 + radius * angle.sin();

        // Alternate polarity
        let polarity = if i % 2 == 0 { 1 } else { -1 };

        events.push(Event {
            t,
            x: x.round() as u16,
            y: y.round() as u16,
            polarity,
        });
    }

    events
}
