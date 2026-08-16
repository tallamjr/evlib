//! CUDA event simulator backend.
//!
//! Loads the nvcc-built `libevsim.so` at runtime through `libloading`, so the
//! Rust crate has no link-time CUDA dependency. The library path comes from
//! `EVLIB_CUDA_SIM_LIB` (full path) or `libevsim.so` on the loader search path.
//! Threshold maps and the u8 log LUT are computed on the host, so this backend
//! shares them bit-for-bit with `EventSimulator`; the LUT is uploaded once and
//! applied on the device, so u8 frames cross the bus as one byte per pixel.
//! A failing `evsim_destroy` in `Drop` is reported with `eprintln!` and not
//! surfaced as an error.

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
type RunFn = unsafe extern "C" fn(
    *mut c_void,
    *const f32,        // frames (host, T*H*W)
    *const c_longlong, // t_ns (host, T)
    c_int,             // n_frames
    *mut c_ushort,     // out_x
    *mut c_ushort,     // out_y
    *mut c_longlong,   // out_t
    *mut c_char,       // out_p
    c_longlong,        // capacity
    *mut c_longlong,   // n_events
) -> c_int;
type RunU8Fn = unsafe extern "C" fn(
    *mut c_void,
    *const u8,         // frames (host, T*H*W)
    *const c_longlong, // t_ns (host, T)
    c_int,             // n_frames
    *mut c_ushort,     // out_x
    *mut c_ushort,     // out_y
    *mut c_longlong,   // out_t
    *mut c_char,       // out_p
    c_longlong,        // capacity
    *mut c_longlong,   // n_events
) -> c_int;
type SetLutFn = unsafe extern "C" fn(*mut c_void, *const f32) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut c_void) -> c_int;

/// Resolved entry points of the shared library; `_lib` keeps it mapped.
struct Api {
    _lib: libloading::Library,
    create: CreateFn,
    reset: ResetFn,
    run: RunFn,
    run_u8: RunU8Fn,
    set_lut: SetLutFn,
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
    let (create, reset, run, run_u8, set_lut, destroy) = unsafe {
        (
            *lib.get::<CreateFn>(b"evsim_create")
                .map_err(|e| format!("missing symbol evsim_create: {e}"))?,
            *lib.get::<ResetFn>(b"evsim_reset")
                .map_err(|e| format!("missing symbol evsim_reset: {e}"))?,
            *lib.get::<RunFn>(b"evsim_run")
                .map_err(|e| format!("missing symbol evsim_run: {e}"))?,
            *lib.get::<RunU8Fn>(b"evsim_run_u8")
                .map_err(|e| format!("missing symbol evsim_run_u8: {e}"))?,
            *lib.get::<SetLutFn>(b"evsim_set_lut")
                .map_err(|e| format!("missing symbol evsim_set_lut: {e}"))?,
            *lib.get::<DestroyFn>(b"evsim_destroy")
                .map_err(|e| format!("missing symbol evsim_destroy: {e}"))?,
        )
    };
    Ok(Api {
        _lib: lib,
        create,
        reset,
        run,
        run_u8,
        set_lut,
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

/// Smallest event capacity tried for a batch; the hint doubles the last count.
const MIN_CAPACITY: usize = 1 << 16;

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
    capacity_hint: usize,
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
            capacity_hint: MIN_CAPACITY,
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

    /// One `evsim_run` call at `capacity`; `Ok(None)` means the batch needs a
    /// larger buffer and `required` events; the device state is unchanged.
    fn run_once(
        &mut self,
        frames: Frames<'_>,
        t_ns: &[i64],
        capacity: usize,
    ) -> Result<Result<EventBatch, usize>, SimError> {
        let mut x: Vec<u16> = Vec::with_capacity(capacity);
        let mut y: Vec<u16> = Vec::with_capacity(capacity);
        let mut t: Vec<i64> = Vec::with_capacity(capacity);
        let mut p: Vec<i8> = Vec::with_capacity(capacity);
        let mut n_events: c_longlong = 0;
        // SAFETY: frames has n * T elements (check_batch), t_ns has T, and the
        // four output buffers hold `capacity` elements each.
        let rc = unsafe {
            match frames {
                Frames::Log(f) => (self.api.run)(
                    self.handle,
                    f.as_ptr(),
                    t_ns.as_ptr(),
                    t_ns.len() as c_int,
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    t.as_mut_ptr(),
                    p.as_mut_ptr() as *mut c_char,
                    capacity as c_longlong,
                    &mut n_events,
                ),
                Frames::U8(f) => (self.api.run_u8)(
                    self.handle,
                    f.as_ptr(),
                    t_ns.as_ptr(),
                    t_ns.len() as c_int,
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    t.as_mut_ptr(),
                    p.as_mut_ptr() as *mut c_char,
                    capacity as c_longlong,
                    &mut n_events,
                ),
            }
        };
        if rc < 0 {
            return Err(SimError::Backend(format!(
                "evsim_run returned CUDA error code {rc}"
            )));
        }
        if rc == 1 {
            return Ok(Err(n_events as usize));
        }
        let n = n_events as usize;
        if n > capacity {
            return Err(SimError::Backend(format!(
                "evsim_run reported {n} events above capacity {capacity} with rc 0"
            )));
        }
        // SAFETY: on rc 0 the C side wrote exactly n <= capacity elements into
        // each buffer (pass 2 fills offset + k for every counted event), so the
        // first n elements are initialised.
        unsafe {
            x.set_len(n);
            y.set_len(n);
            t.set_len(n);
            p.set_len(n);
        }
        Ok(Ok(EventBatch { x, y, t_ns: t, p }))
    }

    /// Run a batch with an explicit starting capacity, retrying once with the
    /// required size if the device reports overflow.
    fn run_with_capacity(
        &mut self,
        frames: Frames<'_>,
        t_ns: &[i64],
        sort: bool,
        capacity: usize,
    ) -> Result<EventBatch, SimError> {
        self.check_batch(frames.len(), t_ns)?;
        if let Frames::Log(f) = frames {
            check_finite(f)?;
        }
        let mut out = match self.run_once(frames, t_ns, capacity)? {
            Ok(b) => b,
            Err(required) => match self.run_once(frames, t_ns, required)? {
                Ok(b) => b,
                Err(again) => {
                    return Err(SimError::Backend(format!(
                        "evsim_run still reports overflow ({again} > {required}) after retry"
                    )))
                }
            },
        };
        self.prev_t = *t_ns.last().expect("non-empty batch");
        self.initialised = true;
        self.capacity_hint = MIN_CAPACITY.max(2 * out.len());
        if sort {
            out.sort_by_time();
        }
        Ok(out)
    }

    /// float32 log frames with an explicit starting capacity (tests the retry path).
    pub fn run_log_with_capacity(
        &mut self,
        frames: &[f32],
        t_ns: &[i64],
        sort: bool,
        capacity: usize,
    ) -> Result<EventBatch, SimError> {
        self.run_with_capacity(Frames::Log(frames), t_ns, sort, capacity)
    }

    /// uint8 frames with an explicit starting capacity (tests the retry path).
    pub fn run_u8_with_capacity(
        &mut self,
        frames: &[u8],
        t_ns: &[i64],
        sort: bool,
        capacity: usize,
    ) -> Result<EventBatch, SimError> {
        self.run_with_capacity(Frames::U8(frames), t_ns, sort, capacity)
    }

    pub fn run_log(
        &mut self,
        frames: &[f32],
        t_ns: &[i64],
        sort: bool,
    ) -> Result<EventBatch, SimError> {
        let capacity = self.capacity_hint;
        self.run_with_capacity(Frames::Log(frames), t_ns, sort, capacity)
    }

    /// uint8 frames are uploaded as bytes; the log LUT is applied on the device.
    pub fn run_u8(
        &mut self,
        frames: &[u8],
        t_ns: &[i64],
        sort: bool,
    ) -> Result<EventBatch, SimError> {
        let capacity = self.capacity_hint;
        self.run_with_capacity(Frames::U8(frames), t_ns, sort, capacity)
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

    #[test]
    fn capacity_retry_and_state_persistence_match_cpu() {
        if !lib_present() {
            return;
        }
        let c = cfg();
        let (frames, t) = random_sequence(128, 96, 32, 6);
        let n = c.pixels();
        let mut cpu = EventSimulator::new(c).unwrap();
        let mut gpu = EventSimulatorCuda::new(c).unwrap();
        // Split the sequence into two batches so the second starts from device state.
        let (fa, fb) = frames.split_at(20 * n);
        let (ta, tb) = t.split_at(20);
        let want_a = cpu.run_log(fa, ta, true).unwrap();
        let got_a = gpu.run_log_with_capacity(fa, ta, true, 16).unwrap();
        assert_eq!(key(&got_a), key(&want_a));
        let want_b = cpu.run_log(fb, tb, true).unwrap();
        let got_b = gpu.run_log_with_capacity(fb, tb, true, 16).unwrap();
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

    /// u8 frames through the device LUT, forced retry, sorted, against the CPU.
    #[test]
    fn u8_capacity_retry_matches_cpu_sorted() {
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
        let got_a = gpu.run_u8_with_capacity(fa, ta, true, 8).unwrap();
        assert!(want_a.len() > 10_000, "cpu produced {}", want_a.len());
        assert_eq!(got_a.len(), want_a.len());
        assert_eq!(key(&got_a), key(&want_a));
        assert!(got_a.t_ns.windows(2).all(|w| w[0] <= w[1]));
        let want_b = cpu.run_u8(fb, tb, false).unwrap();
        let got_b = gpu.run_u8(fb, tb, false).unwrap();
        assert!(!want_b.is_empty());
        assert_eq!(key(&got_b), key(&want_b));
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
