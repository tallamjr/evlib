//! Rows-parallel event simulator over log-intensity or u8 frames.

use rayon::prelude::*;

use super::config::{threshold_maps, SimError, SimulatorConfig};
use super::pixel::{step_pixel, PixelParams, PixelState, NO_EVENT};

/// (timestamp, original index) pair used by the time sort.
type Keyed = (i64, u32);

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
    /// Parallel concatenation of batches into one struct of arrays.
    pub fn concat(parts: &[EventBatch]) -> EventBatch {
        let n: usize = parts.iter().map(EventBatch::len).sum();
        let mut out = EventBatch {
            x: (0..n).into_par_iter().map(|_| 0u16).collect(),
            y: (0..n).into_par_iter().map(|_| 0u16).collect(),
            t_ns: (0..n).into_par_iter().map(|_| 0i64).collect(),
            p: (0..n).into_par_iter().map(|_| 0i8).collect(),
        };
        fn fill<T: Copy + Send + Sync>(
            dst: &mut [T],
            parts: &[EventBatch],
            col: fn(&EventBatch) -> &[T],
        ) {
            let mut slots: Vec<&mut [T]> = Vec::with_capacity(parts.len());
            let mut rest = dst;
            for part in parts {
                let (head, tail) = rest.split_at_mut(part.len());
                slots.push(head);
                rest = tail;
            }
            slots
                .into_par_iter()
                .zip(parts.par_iter())
                .for_each(|(slot, part)| slot.copy_from_slice(col(part)));
        }
        fill(&mut out.x, parts, |b| &b.x);
        fill(&mut out.y, parts, |b| &b.y);
        fill(&mut out.t_ns, parts, |b| &b.t_ns);
        fill(&mut out.p, parts, |b| &b.p);
        out
    }
    #[inline]
    fn push(&mut self, x: u16, y: u16, t: i64, p: i8) {
        self.x.push(x);
        self.y.push(y);
        self.t_ns.push(t);
        self.p.push(p);
    }
    /// Sort all four columns by timestamp (unstable; equal timestamps keep no particular order).
    ///
    /// Parallel bucket sort on (t, index) pairs: a per-chunk counting sort into
    /// time buckets, then each bucket is sorted on its own. Comparison sorts
    /// with an indirect key were 3 to 10x slower on the pixel-grouped CUDA output.
    pub fn sort_by_time(&mut self) {
        let n = self.len();
        if n < 2 {
            return;
        }
        let t_min = *self.t_ns.par_iter().min().expect("non-empty");
        let t_max = *self.t_ns.par_iter().max().expect("non-empty");
        // About 2048 events per bucket keeps each local sort in cache.
        let n_buckets = (n / 2048).clamp(1, 1 << 16);
        let scale = n_buckets as f64 / ((t_max - t_min) as f64 + 1.0);
        let bucket_of = |t: i64| (((t - t_min) as f64 * scale) as usize).min(n_buckets - 1);
        let chunk_len = n.div_ceil(4 * rayon::current_num_threads()).max(1);
        // Per chunk: (t, index) pairs grouped by bucket, plus the bucket start offsets.
        let chunks: Vec<(Vec<Keyed>, Vec<u32>)> = self
            .t_ns
            .par_chunks(chunk_len)
            .enumerate()
            .map(|(ci, ts)| {
                let mut counts = vec![0u32; n_buckets + 1];
                for &t in ts {
                    counts[bucket_of(t) + 1] += 1;
                }
                for b in 0..n_buckets {
                    counts[b + 1] += counts[b];
                }
                let mut cursor = counts.clone();
                let mut local: Vec<Keyed> = vec![(0, 0); ts.len()];
                let base = (ci * chunk_len) as u32;
                for (j, &t) in ts.iter().enumerate() {
                    let b = bucket_of(t);
                    local[cursor[b] as usize] = (t, base + j as u32);
                    cursor[b] += 1;
                }
                (local, counts)
            })
            .collect();
        let buckets: Vec<Vec<Keyed>> = (0..n_buckets)
            .into_par_iter()
            .map(|b| {
                let size: usize = chunks
                    .iter()
                    .map(|(_, off)| (off[b + 1] - off[b]) as usize)
                    .sum();
                let mut merged = Vec::with_capacity(size);
                for (local, off) in &chunks {
                    merged.extend_from_slice(&local[off[b] as usize..off[b + 1] as usize]);
                }
                merged.sort_unstable();
                merged
            })
            .collect();
        drop(chunks);
        // Parallel concatenation into disjoint slices; rayon's flatten().collect()
        // took most of the sort time here.
        let mut keyed: Vec<Keyed> = (0..n).into_par_iter().map(|_| (0i64, 0u32)).collect();
        let mut slots: Vec<&mut [Keyed]> = Vec::with_capacity(n_buckets);
        let mut rest = keyed.as_mut_slice();
        for bucket in &buckets {
            let (head, tail) = rest.split_at_mut(bucket.len());
            slots.push(head);
            rest = tail;
        }
        slots
            .into_par_iter()
            .zip(buckets.par_iter())
            .for_each(|(dst, src)| dst.copy_from_slice(src));
        drop(buckets);
        self.x = keyed.par_iter().map(|&(_, i)| self.x[i as usize]).collect();
        self.y = keyed.par_iter().map(|&(_, i)| self.y[i as usize]).collect();
        self.p = keyed.par_iter().map(|&(_, i)| self.p[i as usize]).collect();
        self.t_ns = keyed.into_par_iter().map(|(t, _)| t).collect();
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

    /// Advance over frames `k_start..T` of one batch, rows-parallel across the
    /// whole batch. `at(k, i)` is the log intensity of pixel `i` in frame `k`.
    /// Each row chunk walks every frame with its own event buffer, so the
    /// parallel region spans the batch instead of one frame.
    fn walk_batch<F>(&mut self, t_ns: &[i64], k_start: usize, at: F) -> EventBatch
    where
        F: Fn(usize, usize) -> f32 + Sync,
    {
        let w = self.cfg.width as usize;
        let h = self.cfg.height as usize;
        let refractory_ns = self.cfg.refractory_ns;
        let t_first = self.prev_t;
        let (c_pos, c_neg) = (&self.c_pos, &self.c_neg);
        // About four chunks per thread keeps the load balanced when event
        // density varies across the image.
        let rows_per_chunk = h.div_ceil(4 * rayon::current_num_threads()).max(1);
        let chunk = rows_per_chunk * w;
        let parts: Vec<EventBatch> = self
            .state
            .par_chunks_mut(chunk)
            .zip(self.prev.par_chunks_mut(chunk))
            .enumerate()
            .map(|(ci, (states, prev))| {
                let mut local = EventBatch::default();
                let row0 = ci * rows_per_chunk;
                let mut t0 = t_first;
                for (k, &t1) in t_ns.iter().enumerate().skip(k_start) {
                    for (r, (row_states, row_prev)) in
                        states.chunks_mut(w).zip(prev.chunks_mut(w)).enumerate()
                    {
                        let row = row0 + r;
                        let base = row * w;
                        for (col, (st, l0)) in
                            row_states.iter_mut().zip(row_prev.iter_mut()).enumerate()
                        {
                            let i = base + col;
                            let params = PixelParams {
                                c_pos: c_pos[i],
                                c_neg: c_neg[i],
                                refractory_ns,
                            };
                            let l1 = at(k, i);
                            step_pixel(st, &params, *l0, t0, l1, t1, &mut |te, pol| {
                                local.push(col as u16, row as u16, te, pol);
                            });
                            *l0 = l1;
                        }
                    }
                    t0 = t1;
                }
                local
            })
            .collect();
        self.prev_t = *t_ns.last().expect("non-empty batch");
        EventBatch::concat(&parts)
    }

    /// Shared body of `run_log` and `run_u8`: validate, initialise from frame 0
    /// if needed, walk the batch, sort on request.
    fn run_batch<F>(&mut self, t_ns: &[i64], sort: bool, at: F) -> Result<EventBatch, SimError>
    where
        F: Fn(usize, usize) -> f32 + Sync,
    {
        if t_ns.is_empty() {
            return Err(SimError::EmptyBatch);
        }
        self.check_run_monotonic(t_ns)?;
        let n = self.cfg.pixels();
        let mut k_start = 0;
        if !self.initialised {
            self.state = (0..n)
                .map(|i| PixelState {
                    l_ref: at(0, i),
                    t_last: NO_EVENT,
                })
                .collect();
            self.prev = (0..n).map(|i| at(0, i)).collect();
            self.prev_t = t_ns[0];
            self.initialised = true;
            k_start = 1;
        }
        let mut out = self.walk_batch(t_ns, k_start, at);
        if sort {
            out.sort_by_time();
        }
        Ok(out)
    }

    fn check_batch_len(&self, frames_len: usize, t_len: usize) -> Result<(), SimError> {
        let n = self.cfg.pixels();
        if frames_len != n * t_len {
            return Err(SimError::ShapeMismatch {
                expected: n * t_len,
                got: frames_len,
            });
        }
        Ok(())
    }

    /// One frame; appends its events to `out` and returns their count.
    pub fn step_log(
        &mut self,
        log_frame: &[f32],
        t_ns: i64,
        out: &mut EventBatch,
    ) -> Result<usize, SimError> {
        self.check_frame(log_frame.len())?;
        let batch = self.run_batch(&[t_ns], false, |_, i| log_frame[i])?;
        out.extend(&batch);
        Ok(batch.len())
    }

    pub fn step_u8(
        &mut self,
        frame: &[u8],
        t_ns: i64,
        out: &mut EventBatch,
    ) -> Result<usize, SimError> {
        self.check_frame(frame.len())?;
        let lut = self.lut;
        let batch = self.run_batch(&[t_ns], false, |_, i| lut[frame[i] as usize])?;
        out.extend(&batch);
        Ok(batch.len())
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
        self.check_batch_len(frames.len(), t_ns.len())?;
        let n = self.cfg.pixels();
        self.run_batch(t_ns, sort, |k, i| frames[k * n + i])
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
        self.check_batch_len(frames.len(), t_ns.len())?;
        let n = self.cfg.pixels();
        let lut = self.lut;
        self.run_batch(t_ns, sort, |k, i| lut[frames[k * n + i] as usize])
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
    fn sort_by_time_orders_scrambled_events_and_keeps_rows_together() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
        // 300k events over a wide time span with many ties, in pixel-grouped order.
        let n = 300_000usize;
        let mut b = EventBatch::with_capacity(n);
        for i in 0..n {
            b.push(
                (i % 640) as u16,
                (i / 640 % 480) as u16,
                rng.gen_range(-5_000_000_000i64..5_000_000_000) / 1000 * 1000,
                if rng.gen_bool(0.5) { 1 } else { -1 },
            );
        }
        let before = b.clone();
        b.sort_by_time();
        assert_eq!(b.len(), n);
        assert!(b.t_ns.windows(2).all(|w| w[0] <= w[1]));
        let key = |q: &EventBatch| {
            let mut v: Vec<(i64, u16, u16, i8)> = (0..q.len())
                .map(|i| (q.t_ns[i], q.x[i], q.y[i], q.p[i]))
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(key(&b), key(&before));
        let mut tiny = EventBatch::default();
        tiny.push(1, 2, 5, 1);
        tiny.push(3, 4, 5, -1);
        tiny.sort_by_time();
        assert_eq!(tiny.t_ns, vec![5, 5]);
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
