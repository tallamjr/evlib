//! CUDA event simulator backend.
//!
//! Loads the nvcc-built `libevsim.so` at runtime through `libloading`, so the
//! Rust crate has no link-time CUDA dependency. The library path comes from
//! `EVLIB_CUDA_SIM_LIB` (full path) or `libevsim.so` on the loader search path.
//! Threshold maps and the u8 log LUT are computed on the host, so this backend
//! shares them bit-for-bit with `EventSimulator`; the LUT is uploaded once and
//! applied on the device, so u8 frames cross the bus as one byte per pixel.
//!
//! Each call copies the frames into the library's pinned staging buffer
//! (parallel memcpy), runs `evsim_run2` (which sorts on the device when asked),
//! and copies the results out of the library's pinned result buffers into
//! fresh `Vec`s (parallel memcpy). Device and pinned buffers live in the handle
//! and grow geometrically, so there is no capacity guess and no retry. A failing `evsim_destroy` in `Drop` is
//! reported with `eprintln!` and not surfaced as an error.

use rayon::prelude::*;
use std::os::raw::{c_char, c_int, c_longlong, c_ushort, c_void};
use std::sync::OnceLock;

use super::config::{threshold_maps, SimError, SimulatorConfig};
use super::simulator::{check_finite, EventBatch};

type CreateFn = unsafe extern "C" fn(
    c_int,      // width
    c_int,      // height
    *const f32, // c_pos (host, H*W)
    *const f32, // c_neg (host, H*W)
    c_longlong, // refractory_ns
    *mut *mut c_void,
) -> c_int;
type ResetFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type SetLutFn = unsafe extern "C" fn(*mut c_void, *const f32) -> c_int;
type StageFn = unsafe extern "C" fn(*mut c_void, c_longlong, *mut *mut c_void) -> c_int;
type Run2Fn = unsafe extern "C" fn(
    *mut c_void,
    c_int,                  // kind: 0 f32 log, 1 u8
    *const c_void,          // frames (host; ignored with STAGED)
    *const c_longlong,      // t_ns (host, T)
    c_int,                  // n_frames
    c_int,                  // flags
    *mut *const c_ushort,   // out_x (pinned, owned by the handle)
    *mut *const c_ushort,   // out_y
    *mut *const c_longlong, // out_t
    *mut *const c_char,     // out_p
    *mut c_longlong,        // n_events
) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut c_void) -> c_int;

/// `evsim_run2` flags: sort by time on the device; frames are already staged.
const FLAG_SORT: c_int = 1;
const FLAG_STAGED: c_int = 2;

/// Resolved entry points of the shared library; `_lib` keeps it mapped.
struct Api {
    _lib: libloading::Library,
    create: CreateFn,
    reset: ResetFn,
    set_lut: SetLutFn,
    stage: StageFn,
    run2: Run2Fn,
    destroy: DestroyFn,
}

fn lib_path() -> String {
    std::env::var("EVLIB_CUDA_SIM_LIB").unwrap_or_else(|_| "libevsim.so".to_string())
}

// Loaded once per process: dlopen plus CUDA context creation is expensive.
// The load result (library and resolved symbols, or the error) is cached.
static API: OnceLock<Result<Api, String>> = OnceLock::new();

fn load_api() -> Result<Api, String> {
    let path = lib_path();
    let lib = unsafe { libloading::Library::new(&path) }
        .map_err(|e| format!("failed to load CUDA library {path}: {e}"))?;
    let (create, reset, set_lut, stage, run2, destroy) = unsafe {
        (
            *lib.get::<CreateFn>(b"evsim_create")
                .map_err(|e| format!("missing symbol evsim_create: {e}"))?,
            *lib.get::<ResetFn>(b"evsim_reset")
                .map_err(|e| format!("missing symbol evsim_reset: {e}"))?,
            *lib.get::<SetLutFn>(b"evsim_set_lut")
                .map_err(|e| format!("missing symbol evsim_set_lut: {e}"))?,
            *lib.get::<StageFn>(b"evsim_stage")
                .map_err(|e| format!("missing symbol evsim_stage: {e}"))?,
            *lib.get::<Run2Fn>(b"evsim_run2")
                .map_err(|e| format!("missing symbol evsim_run2: {e}"))?,
            *lib.get::<DestroyFn>(b"evsim_destroy")
                .map_err(|e| format!("missing symbol evsim_destroy: {e}"))?,
        )
    };
    Ok(Api {
        _lib: lib,
        create,
        reset,
        set_lut,
        stage,
        run2,
        destroy,
    })
}

fn api() -> Result<&'static Api, SimError> {
    API.get_or_init(load_api)
        .as_ref()
        .map_err(|e| SimError::Backend(e.clone()))
}

/// True when the library loads and a 1x1 simulator can be created on a device.
/// The probe runs once per process and the result is cached.
pub fn cuda_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let cfg = SimulatorConfig {
            width: 1,
            height: 1,
            ..Default::default()
        };
        EventSimulatorCuda::new(cfg).is_ok()
    })
}

/// Bytes per rayon task in the pinned-memory copies; below one chunk the copy is serial.
const COPY_CHUNK: usize = 4 << 20;

/// Parallel `memcpy` of `bytes` from `src` to `dst`.
///
/// # Safety
/// `src` and `dst` are valid for `bytes` bytes and do not overlap.
unsafe fn copy_bytes_parallel(src: *const u8, dst: *mut u8, bytes: usize) {
    if bytes <= COPY_CHUNK {
        std::ptr::copy_nonoverlapping(src, dst, bytes);
        return;
    }
    let (src_addr, dst_addr) = (src as usize, dst as usize);
    let n_chunks = bytes.div_ceil(COPY_CHUNK);
    (0..n_chunks).into_par_iter().for_each(|c| {
        let start = c * COPY_CHUNK;
        let len = COPY_CHUNK.min(bytes - start);
        // SAFETY: chunks are disjoint sub-ranges of the caller's valid ranges.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (src_addr + start) as *const u8,
                (dst_addr + start) as *mut u8,
                len,
            )
        };
    });
}

/// Copy `n` elements out of a pinned result buffer into a fresh `Vec`.
///
/// # Safety
/// `src` is valid for `n` elements of `T`.
unsafe fn vec_from_pinned<T: Copy>(src: *const T, n: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(n);
    if n > 0 {
        // SAFETY: the Vec has capacity n; the source holds n elements.
        unsafe {
            copy_bytes_parallel(
                src as *const u8,
                v.as_mut_ptr() as *mut u8,
                n * std::mem::size_of::<T>(),
            );
            v.set_len(n);
        }
    }
    v
}

/// Input stack for one call: float32 log intensity or uint8 intensity.
#[derive(Clone, Copy)]
enum Frames<'a> {
    Log(&'a [f32]),
    U8(&'a [u8]),
}

impl Frames<'_> {
    fn len(&self) -> usize {
        match self {
            Frames::Log(f) => f.len(),
            Frames::U8(f) => f.len(),
        }
    }
    /// (kind for evsim_run2, byte length, byte pointer).
    fn raw(&self) -> (c_int, usize, *const u8) {
        match self {
            Frames::Log(f) => (0, std::mem::size_of_val(*f), f.as_ptr() as *const u8),
            Frames::U8(f) => (1, f.len(), f.as_ptr()),
        }
    }
}

pub struct EventSimulatorCuda {
    cfg: SimulatorConfig,
    api: &'static Api,
    handle: *mut c_void,
    c_pos: Vec<f32>,
    c_neg: Vec<f32>,
    lut: [f32; 256],
    prev_t: i64,
    initialised: bool,
}

// The raw handle is only touched through `&mut self` and `Drop`; `&self`
// methods read host fields only, so Send and Sync are sound.
unsafe impl Send for EventSimulatorCuda {}
unsafe impl Sync for EventSimulatorCuda {}

impl Drop for EventSimulatorCuda {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let rc = unsafe { (self.api.destroy)(self.handle) };
            if rc != 0 {
                eprintln!("evsim_destroy returned CUDA error code {rc}");
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

impl EventSimulatorCuda {
    pub fn new(cfg: SimulatorConfig) -> Result<Self, SimError> {
        cfg.validate()?;
        let api = api()?;
        let (c_pos, c_neg) = threshold_maps(&cfg);
        let mut lut = [0f32; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = ((i as f32) / 255.0 + cfg.log_eps).ln();
        }
        let mut handle: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            (api.create)(
                cfg.width as c_int,
                cfg.height as c_int,
                c_pos.as_ptr(),
                c_neg.as_ptr(),
                cfg.refractory_ns as c_longlong,
                &mut handle,
            )
        };
        if rc != 0 || handle.is_null() {
            return Err(SimError::Backend(format!(
                "evsim_create returned CUDA error code {rc}"
            )));
        }
        let rc = unsafe { (api.set_lut)(handle, lut.as_ptr()) };
        if rc != 0 {
            let rc_destroy = unsafe { (api.destroy)(handle) };
            return Err(SimError::Backend(format!(
                "evsim_set_lut returned CUDA error code {rc} (destroy {rc_destroy})"
            )));
        }
        Ok(Self {
            cfg,
            api,
            handle,
            c_pos,
            c_neg,
            lut,
            prev_t: 0,
            initialised: false,
        })
    }
    pub fn config(&self) -> &SimulatorConfig {
        &self.cfg
    }
    pub fn thresholds(&self) -> (&[f32], &[f32]) {
        (&self.c_pos, &self.c_neg)
    }
    pub fn is_initialised(&self) -> bool {
        self.initialised
    }
    pub fn log_lut(&self) -> &[f32; 256] {
        &self.lut
    }
    pub fn reset(&mut self) -> Result<(), SimError> {
        let rc = unsafe { (self.api.reset)(self.handle) };
        if rc != 0 {
            return Err(SimError::Backend(format!(
                "evsim_reset returned CUDA error code {rc}"
            )));
        }
        self.initialised = false;
        Ok(())
    }

    fn check_batch(&self, frames_len: usize, t_ns: &[i64]) -> Result<(), SimError> {
        if t_ns.is_empty() {
            return Err(SimError::EmptyBatch);
        }
        let n = self.cfg.pixels();
        if frames_len != n * t_ns.len() {
            return Err(SimError::ShapeMismatch {
                expected: n * t_ns.len(),
                got: frames_len,
            });
        }
        if t_ns.len() > c_int::MAX as usize {
            return Err(SimError::Backend("too many frames in one batch".into()));
        }
        if self.initialised && t_ns[0] <= self.prev_t {
            return Err(SimError::NonMonotonicTime { index: 0 });
        }
        for k in 1..t_ns.len() {
            if t_ns[k] <= t_ns[k - 1] {
                return Err(SimError::NonMonotonicTime { index: k });
            }
        }
        Ok(())
    }

    /// Stage the frames into pinned memory, run the device pipeline, copy the
    /// results out. `sort` orders the batch by time on the device (stable for
    /// equal timestamps), so no host sort follows.
    fn run_frames(
        &mut self,
        frames: Frames<'_>,
        t_ns: &[i64],
        sort: bool,
    ) -> Result<EventBatch, SimError> {
        self.check_batch(frames.len(), t_ns)?;
        if let Frames::Log(f) = frames {
            check_finite(f)?;
        }
        let (kind, bytes, src) = frames.raw();
        let mut pinned: *mut c_void = std::ptr::null_mut();
        // SAFETY: the handle is live; `pinned` receives a buffer of at least `bytes` bytes.
        let rc = unsafe { (self.api.stage)(self.handle, bytes as c_longlong, &mut pinned) };
        if rc != 0 || pinned.is_null() {
            return Err(SimError::Backend(format!(
                "evsim_stage returned CUDA error code {rc}"
            )));
        }
        // SAFETY: `src` holds `bytes` bytes (check_batch); the staging buffer holds at
        // least `bytes` (evsim_stage) and is distinct from the caller's memory.
        unsafe { copy_bytes_parallel(src, pinned as *mut u8, bytes) };
        let mut px: *const c_ushort = std::ptr::null();
        let mut py: *const c_ushort = std::ptr::null();
        let mut pt: *const c_longlong = std::ptr::null();
        let mut pp: *const c_char = std::ptr::null();
        let mut n_events: c_longlong = 0;
        // SAFETY: t_ns has T elements; the frames were staged above; the out
        // pointers receive handle-owned pinned buffers valid until the next call.
        let rc = unsafe {
            (self.api.run2)(
                self.handle,
                kind,
                std::ptr::null(),
                t_ns.as_ptr(),
                t_ns.len() as c_int,
                FLAG_STAGED | if sort { FLAG_SORT } else { 0 },
                &mut px,
                &mut py,
                &mut pt,
                &mut pp,
                &mut n_events,
            )
        };
        if rc != 0 {
            return Err(SimError::Backend(format!(
                "evsim_run2 returned CUDA error code {rc}"
            )));
        }
        let n = n_events as usize;
        if n > 0 && (px.is_null() || py.is_null() || pt.is_null() || pp.is_null()) {
            return Err(SimError::Backend(
                "evsim_run2 returned null result pointers".into(),
            ));
        }
        // SAFETY: on rc 0 each result buffer holds exactly n elements.
        let out = unsafe {
            EventBatch {
                x: vec_from_pinned(px, n),
                y: vec_from_pinned(py, n),
                t_ns: vec_from_pinned(pt, n),
                p: vec_from_pinned(pp as *const i8, n),
            }
        };
        self.prev_t = *t_ns.last().expect("non-empty batch");
        self.initialised = true;
        Ok(out)
    }

    pub fn run_log(
        &mut self,
        frames: &[f32],
        t_ns: &[i64],
        sort: bool,
    ) -> Result<EventBatch, SimError> {
        self.run_frames(Frames::Log(frames), t_ns, sort)
    }

    /// uint8 frames are uploaded as bytes; the log LUT is applied on the device.
    pub fn run_u8(
        &mut self,
        frames: &[u8],
        t_ns: &[i64],
        sort: bool,
    ) -> Result<EventBatch, SimError> {
        self.run_frames(Frames::U8(frames), t_ns, sort)
    }

    /// One frame; the first frame initialises the state and yields no events.
    pub fn step_log(&mut self, log_frame: &[f32], t_ns: i64) -> Result<EventBatch, SimError> {
        self.run_log(log_frame, &[t_ns], false)
    }

    pub fn step_u8(&mut self, frame: &[u8], t_ns: i64) -> Result<EventBatch, SimError> {
        self.run_u8(frame, &[t_ns], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_simulation::simulator::EventSimulator;

    fn lib_present() -> bool {
        match std::env::var("EVLIB_CUDA_SIM_LIB") {
            Ok(p) if std::path::Path::new(&p).is_file() => true,
            Ok(p) => {
                eprintln!("skipping CUDA simulator test: EVLIB_CUDA_SIM_LIB={p} is not a file");
                false
            }
            Err(_) => {
                eprintln!("skipping CUDA simulator test: EVLIB_CUDA_SIM_LIB is unset");
                false
            }
        }
    }

    fn random_sequence(w: usize, h: usize, frames: usize, seed: u64) -> (Vec<f32>, Vec<i64>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let n = w * h;
        let mut data = vec![0f32; n * frames];
        for v in data.iter_mut().take(n) {
            *v = rng.gen_range(-3.0..0.0);
        }
        for k in 1..frames {
            for i in 0..n {
                data[k * n + i] = data[(k - 1) * n + i] + rng.gen_range(-0.5..0.5);
            }
        }
        // Irregular spacing so crossing times exercise the f64 fraction and floor.
        let mut t = Vec::with_capacity(frames);
        let mut acc = 0i64;
        for _ in 0..frames {
            t.push(acc);
            acc += rng.gen_range(700_000..1_300_000);
        }
        (data, t)
    }

    fn key(b: &EventBatch) -> Vec<(i64, u16, u16, i8)> {
        let mut v: Vec<(i64, u16, u16, i8)> = (0..b.len())
            .map(|i| (b.t_ns[i], b.y[i], b.x[i], b.p[i]))
            .collect();
        v.sort_unstable();
        v
    }

    fn cfg() -> SimulatorConfig {
        SimulatorConfig {
            width: 128,
            height: 96,
            c_pos: 0.15,
            c_neg: 0.1,
            threshold_sigma: 0.2,
            refractory_ns: 250_000,
            seed: 11,
            ..Default::default()
        }
    }

    #[test]
    fn cuda_matches_cpu_on_random_log_sequence() {
        if !lib_present() {
            return;
        }
        let c = cfg();
        let (frames, t) = random_sequence(128, 96, 32, 5);
        let mut cpu = EventSimulator::new(c).unwrap();
        let want = cpu.run_log(&frames, &t, true).unwrap();
        let mut gpu = EventSimulatorCuda::new(c).unwrap();
        let got = gpu.run_log(&frames, &t, true).unwrap();
        assert!(want.len() > 10_000, "cpu produced {}", want.len());
        assert_eq!(got.len(), want.len());
        assert_eq!(key(&got), key(&want));
        assert!(got.t_ns.windows(2).all(|w| w[0] <= w[1]));
    }

    /// Buffers grow across calls (a 2-frame batch first, then a large one) and
    /// state persists on the device between batches.
    #[test]
    fn buffer_growth_and_state_persistence_match_cpu() {
        if !lib_present() {
            return;
        }
        let c = cfg();
        let (frames, t) = random_sequence(128, 96, 32, 6);
        let n = c.pixels();
        let mut cpu = EventSimulator::new(c).unwrap();
        let mut gpu = EventSimulatorCuda::new(c).unwrap();
        // Three batches: tiny, large (forces growth of every buffer), medium.
        let (f0, rest) = frames.split_at(2 * n);
        let (t0, trest) = t.split_at(2);
        let (fa, fb) = rest.split_at(18 * n);
        let (ta, tb) = trest.split_at(18);
        let want_0 = cpu.run_log(f0, t0, true).unwrap();
        let got_0 = gpu.run_log(f0, t0, true).unwrap();
        assert_eq!(key(&got_0), key(&want_0));
        let want_a = cpu.run_log(fa, ta, true).unwrap();
        let got_a = gpu.run_log(fa, ta, true).unwrap();
        assert!(want_a.len() > 10 * want_0.len().max(1));
        assert_eq!(key(&got_a), key(&want_a));
        let want_b = cpu.run_log(fb, tb, true).unwrap();
        let got_b = gpu.run_log(fb, tb, true).unwrap();
        assert!(!want_b.is_empty());
        assert_eq!(key(&got_b), key(&want_b));
        // Non-monotonic join is rejected without touching the state.
        assert!(matches!(
            gpu.run_log(fb, tb, true),
            Err(SimError::NonMonotonicTime { index: 0 })
        ));
    }

    /// Mirrors the CPU test: NaN or -inf is rejected before the device runs.
    #[test]
    fn non_finite_log_input_is_rejected_and_state_is_untouched() {
        if !lib_present() {
            return;
        }
        let c = SimulatorConfig {
            width: 1,
            height: 1,
            ..cfg()
        };
        let mut gpu = EventSimulatorCuda::new(c).unwrap();
        let mut stack = vec![0.0f32, f32::NAN, 5.0];
        assert!(matches!(
            gpu.run_log(&stack, &[0, 1_000, 2_000], true),
            Err(SimError::InvalidInput { index: 1 })
        ));
        assert!(!gpu.is_initialised());
        stack[1] = f32::NEG_INFINITY;
        assert!(matches!(
            gpu.run_log(&stack, &[0, 1_000, 2_000], true),
            Err(SimError::InvalidInput { index: 1 })
        ));
        assert!(!gpu.is_initialised());
        assert!(gpu.step_log(&[0.5], 10).unwrap().is_empty());
        assert!(matches!(
            gpu.step_log(&[f32::NAN], 20),
            Err(SimError::InvalidInput { index: 0 })
        ));
        assert!(matches!(
            gpu.step_log(&[f32::NEG_INFINITY], 20),
            Err(SimError::InvalidInput { index: 0 })
        ));
        // Timestamp 20 is still accepted, so the failed calls moved no state.
        let mut cpu = EventSimulator::new(c).unwrap();
        let mut want = EventBatch::default();
        cpu.step_log(&[0.5], 10, &mut want).unwrap();
        cpu.step_log(&[1.5], 20, &mut want).unwrap();
        let got = gpu.step_log(&[1.5], 20).unwrap();
        assert_eq!(key(&got), key(&want));
        assert!(!got.is_empty());
    }

    /// u8 frames through the device LUT, sorted, in two batches, against the CPU.
    #[test]
    fn u8_batches_match_cpu_sorted() {
        if !lib_present() {
            return;
        }
        let c = cfg();
        let n = c.pixels();
        let frames = 24;
        let raw: Vec<u8> = (0..n * frames)
            .map(|i| {
                let k = i / n;
                let px = i % n;
                (((px * 7 + k * k * 13) % 251) as u8).wrapping_add((k * 5) as u8)
            })
            .collect();
        let t: Vec<i64> = (0..frames as i64)
            .map(|k| k * 900_000 + k * k * 1_000)
            .collect();
        let mut cpu = EventSimulator::new(c).unwrap();
        let mut gpu = EventSimulatorCuda::new(c).unwrap();
        let (fa, fb) = raw.split_at(15 * n);
        let (ta, tb) = t.split_at(15);
        let want_a = cpu.run_u8(fa, ta, true).unwrap();
        let got_a = gpu.run_u8(fa, ta, true).unwrap();
        assert!(want_a.len() > 10_000, "cpu produced {}", want_a.len());
        assert_eq!(got_a.len(), want_a.len());
        assert_eq!(key(&got_a), key(&want_a));
        assert!(got_a.t_ns.windows(2).all(|w| w[0] <= w[1]));
        let want_b = cpu.run_u8(fb, tb, false).unwrap();
        let got_b = gpu.run_u8(fb, tb, false).unwrap();
        assert!(!want_b.is_empty());
        assert_eq!(key(&got_b), key(&want_b));
    }

    /// Negative and sign-crossing timestamps sort correctly on the device.
    #[test]
    fn negative_timestamps_sort_on_device() {
        if !lib_present() {
            return;
        }
        let c = SimulatorConfig {
            width: 64,
            height: 32,
            ..cfg()
        };
        let (frames, t) = random_sequence(64, 32, 16, 9);
        let t: Vec<i64> = t.iter().map(|&v| v - 7_000_000).collect();
        assert!(t[0] < 0 && *t.last().unwrap() > 0);
        let mut cpu = EventSimulator::new(c).unwrap();
        let mut gpu = EventSimulatorCuda::new(c).unwrap();
        let want = cpu.run_log(&frames, &t, true).unwrap();
        let got = gpu.run_log(&frames, &t, true).unwrap();
        assert!(!want.is_empty());
        assert_eq!(key(&got), key(&want));
        assert!(got.t_ns.windows(2).all(|w| w[0] <= w[1]));
        assert!(got.t_ns[0] < 0);
    }

    #[test]
    fn step_reset_and_u8_match_cpu() {
        if !lib_present() {
            return;
        }
        let c = SimulatorConfig {
            width: 32,
            height: 16,
            ..cfg()
        };
        let n = c.pixels();
        let raw: Vec<u8> = (0..n * 5)
            .map(|i| ((i * 37 + i / n * 11) % 256) as u8)
            .collect();
        let t = [0i64, 1_000, 3_000, 3_500, 9_000];
        let mut cpu = EventSimulator::new(c).unwrap();
        let mut gpu = EventSimulatorCuda::new(c).unwrap();
        let want = cpu.run_u8(&raw, &t, true).unwrap();
        let got = gpu.run_u8(&raw, &t, true).unwrap();
        assert!(!want.is_empty());
        assert_eq!(key(&got), key(&want));
        // Stepping one frame at a time from a reset state gives the same events.
        gpu.reset().unwrap();
        assert!(!gpu.is_initialised());
        let mut stepped = EventBatch::default();
        for k in 0..t.len() {
            let b = gpu.step_u8(&raw[k * n..(k + 1) * n], t[k]).unwrap();
            if k == 0 {
                assert!(b.is_empty());
            }
            stepped.extend(&b);
        }
        assert_eq!(key(&stepped), key(&want));
    }
}
