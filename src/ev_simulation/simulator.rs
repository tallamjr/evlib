//! Rows-parallel event simulator over log-intensity or u8 frames.

use rayon::prelude::*;

use super::config::{threshold_maps, SimError, SimulatorConfig};
use super::pixel::{step_pixel, PixelParams, PixelState, NO_EVENT};

/// Struct-of-arrays event batch. Timestamps are nanoseconds; polarity is -1 or 1.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct EventBatch {
    pub x: Vec<u16>,
    pub y: Vec<u16>,
    pub t_ns: Vec<i64>,
    pub p: Vec<i8>,
}

impl EventBatch {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            t_ns: Vec::with_capacity(n),
            p: Vec::with_capacity(n),
        }
    }
    pub fn len(&self) -> usize {
        self.t_ns.len()
    }
    pub fn is_empty(&self) -> bool {
        self.t_ns.is_empty()
    }
    pub fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
        self.t_ns.clear();
        self.p.clear();
    }
    pub fn extend(&mut self, other: &EventBatch) {
        self.x.extend_from_slice(&other.x);
        self.y.extend_from_slice(&other.y);
        self.t_ns.extend_from_slice(&other.t_ns);
        self.p.extend_from_slice(&other.p);
    }
    #[inline]
    fn push(&mut self, x: u16, y: u16, t: i64, p: i8) {
        self.x.push(x);
        self.y.push(y);
        self.t_ns.push(t);
        self.p.push(p);
    }
    /// Sort all four columns by timestamp (unstable; equal timestamps keep no particular order).
    /// Parallel sort and gather: the CUDA backend returns events grouped by pixel,
    /// which is far from time order, so a serial sort dominated its runtime.
    pub fn sort_by_time(&mut self) {
        let mut idx: Vec<u32> = (0..self.len() as u32).collect();
        idx.par_sort_unstable_by_key(|&i| self.t_ns[i as usize]);
        self.x = idx.par_iter().map(|&i| self.x[i as usize]).collect();
        self.y = idx.par_iter().map(|&i| self.y[i as usize]).collect();
        self.p = idx.par_iter().map(|&i| self.p[i as usize]).collect();
        self.t_ns = idx.par_iter().map(|&i| self.t_ns[i as usize]).collect();
    }
}

pub struct EventSimulator {
    cfg: SimulatorConfig,
    c_pos: Vec<f32>,
    c_neg: Vec<f32>,
    state: Vec<PixelState>,
    /// Log intensity of the previous frame; empty until initialised.
    prev: Vec<f32>,
    prev_t: i64,
    initialised: bool,
    lut: [f32; 256],
}

impl EventSimulator {
    pub fn new(cfg: SimulatorConfig) -> Result<Self, SimError> {
        cfg.validate()?;
        let (c_pos, c_neg) = threshold_maps(&cfg);
        let mut lut = [0f32; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = ((i as f32) / 255.0 + cfg.log_eps).ln();
        }
        Ok(Self {
            cfg,
            c_pos,
            c_neg,
            state: Vec::new(),
            prev: Vec::new(),
            prev_t: 0,
            initialised: false,
            lut,
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
    pub fn reset(&mut self) {
        self.state.clear();
        self.prev.clear();
        self.initialised = false;
    }

    fn check_frame(&self, len: usize) -> Result<(), SimError> {
        let n = self.cfg.pixels();
        if len != n {
            return Err(SimError::ShapeMismatch {
                expected: n,
                got: len,
            });
        }
        Ok(())
    }

    /// Monotonic check for `run_*`: covers both the in-batch timestamps and,
    /// if the simulator already holds a previous frame, the join against it.
    fn check_run_monotonic(&self, t_ns: &[i64]) -> Result<(), SimError> {
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

    fn initialise(&mut self, log_frame: &[f32], t_ns: i64) {
        self.state = log_frame
            .iter()
            .map(|&l| PixelState {
                l_ref: l,
                t_last: NO_EVENT,
            })
            .collect();
        self.prev = log_frame.to_vec();
        self.prev_t = t_ns;
        self.initialised = true;
    }

    /// Advance from the stored previous frame to `log_frame`, appending events per row.
    fn advance(&mut self, log_frame: &[f32], t_ns: i64, out: &mut EventBatch) -> usize {
        let w = self.cfg.width as usize;
        let refractory_ns = self.cfg.refractory_ns;
        let t0 = self.prev_t;
        let (c_pos, c_neg) = (&self.c_pos, &self.c_neg);
        let prev = &self.prev;
        let rows: Vec<EventBatch> = self
            .state
            .par_chunks_mut(w)
            .enumerate()
            .map(|(row, states)| {
                let mut local = EventBatch::default();
                let base = row * w;
                for (col, st) in states.iter_mut().enumerate() {
                    let i = base + col;
                    let params = PixelParams {
                        c_pos: c_pos[i],
                        c_neg: c_neg[i],
                        refractory_ns,
                    };
                    step_pixel(
                        st,
                        &params,
                        prev[i],
                        t0,
                        log_frame[i],
                        t_ns,
                        &mut |te, pol| {
                            local.push(col as u16, row as u16, te, pol);
                        },
                    );
                }
                local
            })
            .collect();
        let count: usize = rows.iter().map(EventBatch::len).sum();
        out.x.reserve(count);
        out.y.reserve(count);
        out.t_ns.reserve(count);
        out.p.reserve(count);
        for r in &rows {
            out.extend(r);
        }
        self.prev.copy_from_slice(log_frame);
        self.prev_t = t_ns;
        count
    }

    pub fn step_log(
        &mut self,
        log_frame: &[f32],
        t_ns: i64,
        out: &mut EventBatch,
    ) -> Result<usize, SimError> {
        self.check_frame(log_frame.len())?;
        if !self.initialised {
            self.initialise(log_frame, t_ns);
            return Ok(0);
        }
        if t_ns <= self.prev_t {
            return Err(SimError::NonMonotonicTime { index: 0 });
        }
        Ok(self.advance(log_frame, t_ns, out))
    }

    pub fn step_u8(
        &mut self,
        frame: &[u8],
        t_ns: i64,
        out: &mut EventBatch,
    ) -> Result<usize, SimError> {
        self.check_frame(frame.len())?;
        let log: Vec<f32> = frame.iter().map(|&v| self.lut[v as usize]).collect();
        self.step_log(&log, t_ns, out)
    }

    pub fn run_log(
        &mut self,
        frames: &[f32],
        t_ns: &[i64],
        sort: bool,
    ) -> Result<EventBatch, SimError> {
        if t_ns.is_empty() {
            return Err(SimError::EmptyBatch);
        }
        let n = self.cfg.pixels();
        if frames.len() != n * t_ns.len() {
            return Err(SimError::ShapeMismatch {
                expected: n * t_ns.len(),
                got: frames.len(),
            });
        }
        self.check_run_monotonic(t_ns)?;
        let mut out = EventBatch::default();
        for (k, &t) in t_ns.iter().enumerate() {
            self.step_log(&frames[k * n..(k + 1) * n], t, &mut out)?;
        }
        if sort {
            out.sort_by_time();
        }
        Ok(out)
    }

    pub fn run_u8(
        &mut self,
        frames: &[u8],
        t_ns: &[i64],
        sort: bool,
    ) -> Result<EventBatch, SimError> {
        if t_ns.is_empty() {
            return Err(SimError::EmptyBatch);
        }
        let n = self.cfg.pixels();
        if frames.len() != n * t_ns.len() {
            return Err(SimError::ShapeMismatch {
                expected: n * t_ns.len(),
                got: frames.len(),
            });
        }
        self.check_run_monotonic(t_ns)?;
        let mut out = EventBatch::default();
        let mut log = vec![0f32; n];
        for (k, &t) in t_ns.iter().enumerate() {
            for (dst, &src) in log.iter_mut().zip(&frames[k * n..(k + 1) * n]) {
                *dst = self.lut[src as usize];
            }
            self.step_log(&log, t, &mut out)?;
        }
        if sort {
            out.sort_by_time();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_simulation::pixel::{step_pixel, PixelParams, PixelState, NO_EVENT};

    fn cfg(w: u32, h: u32) -> SimulatorConfig {
        SimulatorConfig {
            width: w,
            height: h,
            c_pos: 0.15,
            c_neg: 0.1,
            ..Default::default()
        }
    }

    /// Scalar reference: the same per-pixel loop, single thread, no batching.
    fn reference(cfg: &SimulatorConfig, frames: &[f32], t: &[i64]) -> EventBatch {
        let n = cfg.pixels();
        let (cp, cn) = threshold_maps(cfg);
        let mut st: Vec<PixelState> = frames[..n]
            .iter()
            .map(|&l| PixelState {
                l_ref: l,
                t_last: NO_EVENT,
            })
            .collect();
        let mut out = EventBatch::default();
        for k in 1..t.len() {
            let (f0, f1) = (&frames[(k - 1) * n..k * n], &frames[k * n..(k + 1) * n]);
            for i in 0..n {
                let p = PixelParams {
                    c_pos: cp[i],
                    c_neg: cn[i],
                    refractory_ns: cfg.refractory_ns,
                };
                let (x, y) = (
                    (i % cfg.width as usize) as u16,
                    (i / cfg.width as usize) as u16,
                );
                step_pixel(
                    &mut st[i],
                    &p,
                    f0[i],
                    t[k - 1],
                    f1[i],
                    t[k],
                    &mut |te, pol| {
                        out.x.push(x);
                        out.y.push(y);
                        out.t_ns.push(te);
                        out.p.push(pol);
                    },
                );
            }
        }
        out.sort_by_time();
        out
    }

    fn random_sequence(w: usize, h: usize, frames: usize, seed: u64) -> (Vec<f32>, Vec<i64>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut data = vec![0f32; w * h * frames];
        // Smooth-ish random walk per pixel so events are plentiful but bounded.
        for v in data.iter_mut().take(w * h) {
            *v = rng.gen_range(-3.0..0.0);
        }
        for k in 1..frames {
            for i in 0..w * h {
                data[k * w * h + i] = data[(k - 1) * w * h + i] + rng.gen_range(-0.5..0.5);
            }
        }
        let t: Vec<i64> = (0..frames as i64).map(|k| k * 1_000_000).collect();
        (data, t)
    }

    #[test]
    fn first_frame_emits_nothing_and_reset_restores() {
        let mut sim = EventSimulator::new(cfg(2, 2)).unwrap();
        let mut out = EventBatch::default();
        assert_eq!(sim.step_log(&[0.0; 4], 0, &mut out).unwrap(), 0);
        assert!(sim.is_initialised());
        assert_eq!(sim.step_log(&[1.0; 4], 1_000, &mut out).unwrap(), 4 * 6);
        sim.reset();
        assert!(!sim.is_initialised());
        out.clear();
        assert_eq!(sim.step_log(&[1.0; 4], 2_000, &mut out).unwrap(), 0);
    }

    #[test]
    fn shape_and_time_errors() {
        let mut sim = EventSimulator::new(cfg(2, 2)).unwrap();
        let mut out = EventBatch::default();
        assert!(matches!(
            sim.step_log(&[0.0; 3], 0, &mut out),
            Err(SimError::ShapeMismatch {
                expected: 4,
                got: 3
            })
        ));
        sim.step_log(&[0.0; 4], 10, &mut out).unwrap();
        assert!(matches!(
            sim.step_log(&[0.0; 4], 10, &mut out),
            Err(SimError::NonMonotonicTime { .. })
        ));
        assert!(matches!(
            sim.run_log(&[], &[], true),
            Err(SimError::EmptyBatch)
        ));
        assert!(matches!(
            sim.run_log(&[0.0; 8], &[20, 20], true),
            Err(SimError::NonMonotonicTime { index: 1 })
        ));
    }

    #[test]
    fn rows_parallel_matches_scalar_reference() {
        let c = SimulatorConfig {
            threshold_sigma: 0.2,
            seed: 7,
            refractory_ns: 300_000,
            ..cfg(64, 48)
        };
        let (frames, t) = random_sequence(64, 48, 16, 1);
        let mut sim = EventSimulator::new(c).unwrap();
        let got = sim.run_log(&frames, &t, true).unwrap();
        let want = reference(&c, &frames, &t);
        assert!(!want.is_empty());
        assert_eq!(got.len(), want.len());
        // Sort is by time only; compare as multisets of (t, x, y, p).
        let key = |b: &EventBatch| {
            let mut v: Vec<(i64, u16, u16, i8)> = (0..b.len())
                .map(|i| (b.t_ns[i], b.x[i], b.y[i], b.p[i]))
                .collect();
            v.sort();
            v
        };
        assert_eq!(key(&got), key(&want));
    }

    #[test]
    fn run_equals_repeated_step() {
        let c = cfg(16, 8);
        let (frames, t) = random_sequence(16, 8, 6, 3);
        let mut a = EventSimulator::new(c).unwrap();
        let run = a.run_log(&frames, &t, true).unwrap();
        let mut b = EventSimulator::new(c).unwrap();
        let mut stepped = EventBatch::default();
        for k in 0..t.len() {
            b.step_log(&frames[k * 128..(k + 1) * 128], t[k], &mut stepped)
                .unwrap();
        }
        stepped.sort_by_time();
        assert_eq!(run.t_ns, stepped.t_ns);
        assert_eq!(run.len(), stepped.len());
    }

    #[test]
    fn u8_path_matches_log_path_through_lut() {
        let c = cfg(8, 4);
        let mut sim_u8 = EventSimulator::new(c).unwrap();
        let lut = *sim_u8.log_lut();
        assert!((lut[0] - (1e-3f32).ln()).abs() < 1e-6);
        assert!((lut[255] - (1.0f32 + 1e-3).ln()).abs() < 1e-6);
        let raw: Vec<u8> = (0..32u8)
            .chain((0..32u8).map(|v| v.wrapping_mul(7)))
            .collect();
        let t = [0i64, 5_000];
        let out_u8 = sim_u8.run_u8(&raw, &t, true).unwrap();
        let logf: Vec<f32> = raw.iter().map(|&v| lut[v as usize]).collect();
        let mut sim_f = EventSimulator::new(c).unwrap();
        let out_f = sim_f.run_log(&logf, &t, true).unwrap();
        assert_eq!(out_u8, out_f);
        assert!(!out_u8.is_empty());
    }

    #[test]
    fn unsorted_run_is_still_a_permutation_of_sorted() {
        let c = cfg(8, 8);
        let (frames, t) = random_sequence(8, 8, 4, 9);
        let mut a = EventSimulator::new(c).unwrap();
        let mut unsorted = a.run_log(&frames, &t, false).unwrap();
        let mut b = EventSimulator::new(c).unwrap();
        let sorted = b.run_log(&frames, &t, true).unwrap();
        assert!(sorted.t_ns.windows(2).all(|w| w[0] <= w[1]));
        unsorted.sort_by_time();
        assert_eq!(unsorted.t_ns, sorted.t_ns);
    }
}
