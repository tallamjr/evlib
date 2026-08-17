//! Turn a real frame sequence into synthetic events, and (when ground truth
//! is available) render the real sensor-recorded events alongside for
//! comparison. Pure Rust: calls `evlib::ev_simulation::EventSimulator`
//! directly, no Python.
//!
//! evlib deliberately does not decode video itself (that stays in lumin's
//! stack), so this reads an already-decoded frame sequence: a directory of
//! grayscale PGM (P5) files plus a `timestamps.txt` of "seconds filename"
//! lines, one per frame. Default input is
//! `examples/output/uzhfpv_indoor_forward_3_frames/`, real DAVIS camera
//! frames from the UZH-FPV dataset (research use only, not committed).
//!
//! When `<out-dir>/<name>_real_events.bin.zst` sits next to the default
//! input, it is decompressed with the `zstd` CLI and rendered too: real
//! events recorded by the same DAVIS sensor on the same footage, packed as
//! 9-byte little-endian records (x: u16, y: u16, t_us: i32, polarity: i8),
//! sorted by time. Both the frame directory and this sidecar are produced
//! once, out of band, from the UZH-FPV rosbag (a one-off data-prep step
//! using evflow's already-tested bag reader; not part of this repository).
//!
//! Run:
//!     cargo run --release --example video_to_events
//!     cargo run --release --example video_to_events -- --frames-dir my_frames --out-dir out
//!
//! Acceptance: prints the event counts and writes a synthetic-events video
//! (and a real-events video, when the sidecar is present) whose red
//! (positive) / blue (negative) traces track the source's moving edges.

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use evlib::ev_simulation::{EventBatch, EventSimulator, SimulatorConfig};

struct Args {
    frames_dir: PathBuf,
    out_dir: PathBuf,
    positive_threshold: f32,
    negative_threshold: f32,
    event_fps: f64,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            frames_dir: PathBuf::from("examples/output/uzhfpv_indoor_forward_3_frames"),
            out_dir: PathBuf::from("examples/output"),
            positive_threshold: 0.2,
            negative_threshold: 0.2,
            event_fps: 30.0,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = || it.next().expect("missing value after {flag}");
            match flag.as_str() {
                "--frames-dir" => args.frames_dir = PathBuf::from(value()),
                "--out-dir" => args.out_dir = PathBuf::from(value()),
                "--positive-threshold" => {
                    args.positive_threshold = value().parse().expect("bad float")
                }
                "--negative-threshold" => {
                    args.negative_threshold = value().parse().expect("bad float")
                }
                "--event-fps" => args.event_fps = value().parse().expect("bad float"),
                other => panic!("unknown flag {other}"),
            }
        }
        args
    }
}

/// One decoded grayscale frame: raw row-major pixels plus its timestamp.
struct Frame {
    t_ns: i64,
    pixels: Vec<u8>,
}

/// Read one binary PGM (P5): "P5\n{width} {height}\n{maxval}\n" then raw bytes.
fn read_pgm(path: &Path) -> (u32, u32, Vec<u8>) {
    let mut file = fs::File::open(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut header = [0u8; 2];
    file.read_exact(&mut header).expect("PGM magic");
    assert_eq!(&header, b"P5", "{path:?} is not a binary PGM");

    // Skip whitespace/comments, then read three whitespace-separated ASCII
    // integers (width, height, maxval), then exactly one whitespace byte.
    let mut reader = BufReader::new(file);
    let read_token = |reader: &mut BufReader<fs::File>| -> String {
        let mut byte = [0u8; 1];
        let mut token = String::new();
        loop {
            reader.read_exact(&mut byte).expect("PGM header");
            let c = byte[0] as char;
            if c.is_ascii_whitespace() {
                if token.is_empty() {
                    continue;
                }
                break;
            }
            token.push(c);
        }
        token
    };
    let width: u32 = read_token(&mut reader).parse().expect("PGM width");
    let height: u32 = read_token(&mut reader).parse().expect("PGM height");
    let _maxval: String = read_token(&mut reader);

    let mut pixels = vec![0u8; (width * height) as usize];
    reader.read_exact(&mut pixels).expect("PGM pixel data");
    (width, height, pixels)
}

fn read_frames(dir: &Path) -> (u32, u32, Vec<Frame>) {
    let timestamps_path = dir.join("timestamps.txt");
    let file =
        fs::File::open(&timestamps_path).unwrap_or_else(|e| panic!("{timestamps_path:?}: {e}"));
    let mut frames = Vec::new();
    let (mut width, mut height) = (0u32, 0u32);
    for line in BufReader::new(file).lines() {
        let line = line.expect("read timestamps.txt");
        let mut parts = line.split_whitespace();
        let t_s: f64 = parts.next().expect("timestamp").parse().expect("t_s");
        let name = parts.next().expect("filename");
        let (w, h, pixels) = read_pgm(&dir.join(name));
        width = w;
        height = h;
        frames.push(Frame {
            t_ns: (t_s * 1e9).round() as i64,
            pixels,
        });
    }
    (width, height, frames)
}

/// Decompress a zstd-compressed real-events sidecar and unpack its 9-byte
/// records (x: u16 LE, y: u16 LE, t_us: i32 LE, polarity: i8) into a batch.
fn read_real_events(path: &Path) -> EventBatch {
    let output = Command::new("zstd")
        .args(["-dc"])
        .arg(path)
        .stderr(Stdio::inherit())
        .output()
        .expect("zstd must be on PATH to decompress the real-events sidecar");
    assert!(
        output.status.success(),
        "zstd decompress failed for {path:?}"
    );

    const RECORD_BYTES: usize = 9;
    let raw = output.stdout;
    assert_eq!(
        raw.len() % RECORD_BYTES,
        0,
        "{path:?} is not a whole number of 9-byte records"
    );
    let n = raw.len() / RECORD_BYTES;
    let mut batch = EventBatch::with_capacity(n);
    for record in raw.chunks_exact(RECORD_BYTES) {
        let x = u16::from_le_bytes([record[0], record[1]]);
        let y = u16::from_le_bytes([record[2], record[3]]);
        let t_us = i32::from_le_bytes([record[4], record[5], record[6], record[7]]);
        let p = record[8] as i8;
        batch.x.push(x);
        batch.y.push(y);
        batch.t_ns.push(t_us as i64 * 1000);
        batch.p.push(p);
    }
    batch
}

/// Bin events into fps-spaced windows and write red/blue-on-black PNGs.
fn render_event_frames(
    batch: &EventBatch,
    width: u32,
    height: u32,
    fps: f64,
    out_dir: &Path,
) -> usize {
    let window_ns = (1e9 / fps).round() as i64;
    let n_windows = if batch.t_ns.is_empty() {
        0
    } else {
        (batch.t_ns.iter().max().copied().unwrap_or(0) / window_ns) as usize + 1
    };

    let mut buffers = vec![vec![0u8; (width * height * 3) as usize]; n_windows];
    for i in 0..batch.t_ns.len() {
        let w = (batch.t_ns[i] / window_ns) as usize;
        let px = (batch.y[i] as usize * width as usize + batch.x[i] as usize) * 3;
        let buf = &mut buffers[w];
        if batch.p[i] == 1 {
            buf[px] = 255; // red: positive
        } else {
            buf[px + 2] = 255; // blue: negative
        }
    }

    for (i, buf) in buffers.iter().enumerate() {
        let img = image::RgbImage::from_raw(width, height, buf.clone())
            .expect("frame buffer matches width*height*3");
        img.save(out_dir.join(format!("frame_{i:08}.png")))
            .expect("write PNG");
    }
    n_windows
}

fn encode_video(png_dir: &Path, fps: f64, out_path: &Path) {
    let status = Command::new("ffmpeg")
        .args(["-y", "-framerate", &fps.to_string(), "-i"])
        .arg(png_dir.join("frame_%08d.png"))
        .args(["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"])
        .arg(out_path)
        .status()
        .expect("ffmpeg must be on PATH");
    assert!(status.success(), "ffmpeg encode failed");
}

fn render_and_encode(batch: &EventBatch, width: u32, height: u32, fps: f64, out_path: &Path) {
    let tmp_dir = tempfile::tempdir().expect("create temp render dir");
    render_event_frames(batch, width, height, fps, tmp_dir.path());
    encode_video(tmp_dir.path(), fps, out_path);
}

fn report(label: &str, batch: &EventBatch, duration_s: f64) {
    let n = batch.len();
    let n_pos = batch.p.iter().filter(|&&p| p == 1).count();
    println!(
        "{label}: {n} events ({n_pos} positive, {} negative, {:.0} ev/s)",
        n - n_pos,
        n as f64 / duration_s
    );
}

fn main() {
    let args = Args::parse();
    if !args.frames_dir.exists() {
        eprintln!("frames directory not found: {:?}", args.frames_dir);
        eprintln!("point --frames-dir at a directory of PGM frames + timestamps.txt");
        std::process::exit(1);
    }
    fs::create_dir_all(&args.out_dir).expect("create out-dir");

    let (width, height, frames) = read_frames(&args.frames_dir);
    let duration_s = frames.last().unwrap().t_ns as f64 / 1e9;
    println!(
        "frames: {} ({width}x{height}, {duration_s:.1} s)",
        frames.len()
    );

    let cfg = SimulatorConfig {
        width,
        height,
        c_pos: args.positive_threshold,
        c_neg: args.negative_threshold,
        ..SimulatorConfig::default()
    };
    let mut simulator = EventSimulator::new(cfg).expect("valid simulator config");

    let pixel_data: Vec<u8> = frames
        .iter()
        .flat_map(|f| f.pixels.iter().copied())
        .collect();
    let timestamps: Vec<i64> = frames.iter().map(|f| f.t_ns).collect();
    let synthetic = simulator
        .run_u8(&pixel_data, &timestamps, true)
        .expect("simulate");
    report("synthetic", &synthetic, duration_s);

    let stem = args
        .frames_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("frames")
        .trim_end_matches("_frames");
    let synthetic_path = args.out_dir.join(format!("{stem}_synthetic_events.mp4"));
    render_and_encode(&synthetic, width, height, args.event_fps, &synthetic_path);
    println!("synthetic event video: {}", synthetic_path.display());

    let real_events_path = args.out_dir.join(format!("{stem}_real_events.bin.zst"));
    if real_events_path.exists() {
        let real = read_real_events(&real_events_path);
        report("real (sensor-recorded)", &real, duration_s);

        let real_video_path = args.out_dir.join(format!("{stem}_real_events.mp4"));
        render_and_encode(&real, width, height, args.event_fps, &real_video_path);
        println!("real event video: {}", real_video_path.display());
    } else {
        println!(
            "no real-events sidecar at {}; skipping the ground-truth comparison",
            real_events_path.display()
        );
    }
}
