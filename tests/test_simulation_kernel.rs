//! Kernel behaviour on the tracked slider_depth DAVIS frames (240x180, 87 frames).

use evlib::ev_simulation::{EventSimulator, SimulatorConfig};
use std::path::Path;

fn load_slider_depth() -> (Vec<u8>, Vec<i64>, u32, u32) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/slider_depth");
    let list = std::fs::read_to_string(root.join("images.txt")).expect("images.txt");
    let mut frames = Vec::new();
    let mut t = Vec::new();
    let (mut w, mut h) = (0u32, 0u32);
    for line in list.lines() {
        let mut parts = line.split_whitespace();
        let secs: f64 = parts.next().unwrap().parse().unwrap();
        let rel = parts.next().unwrap();
        let img = image::open(root.join(rel)).expect("png").into_luma8();
        (w, h) = img.dimensions();
        frames.extend_from_slice(img.as_raw());
        t.push((secs * 1e9).round() as i64);
    }
    (frames, t, w, h)
}

#[test]
fn slider_depth_produces_ordered_balanced_events() {
    let (frames, t, w, h) = load_slider_depth();
    let cfg = SimulatorConfig {
        width: w,
        height: h,
        ..Default::default()
    };
    let mut sim = EventSimulator::new(cfg).unwrap();
    let out = sim.run_u8(&frames, &t, true).unwrap();
    assert!(out.len() > 100_000, "got {}", out.len());
    assert!(out.t_ns.windows(2).all(|p| p[0] <= p[1]));
    assert!(out.t_ns[0] >= t[0] && *out.t_ns.last().unwrap() <= *t.last().unwrap());
    let pos = out.p.iter().filter(|&&p| p == 1).count() as f64 / out.len() as f64;
    assert!(pos > 0.3 && pos < 0.7, "positive fraction {pos}");
    assert!(out.x.iter().all(|&x| (x as u32) < w) && out.y.iter().all(|&y| (y as u32) < h));
}
