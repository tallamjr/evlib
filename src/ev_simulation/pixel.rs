//! Pure per-pixel step of the ESIM threshold-crossing model.
//!
//! One function, no allocation, so the CPU rows-parallel path and the CUDA
//! kernel implement the same arithmetic.

/// Sentinel for "no event yet" in `PixelState::t_last`.
pub const NO_EVENT: i64 = i64::MIN;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelState {
    pub l_ref: f32,
    pub t_last: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelParams {
    pub c_pos: f32,
    pub c_neg: f32,
    pub refractory_ns: i64,
}

/// Interpolated crossing time of level `lc` on the segment (l0, t0) -> (l1, t1).
///
/// The fraction is computed in f64 and clamped to [0, 1]; the result is
/// rounded to the nearest nanosecond.
#[inline]
fn crossing_time(l0: f32, t0: i64, l1: f32, t1: i64, lc: f32) -> i64 {
    let dl = (l1 as f64) - (l0 as f64);
    let frac = if dl == 0.0 {
        1.0
    } else {
        (((lc as f64) - (l0 as f64)) / dl).clamp(0.0, 1.0)
    };
    t0 + ((t1 - t0) as f64 * frac).round() as i64
}

/// Advance one pixel across one frame interval, calling `emit(t_ns, polarity)`
/// for each event that survives the refractory period. Returns the emitted count.
#[inline]
pub fn step_pixel(
    state: &mut PixelState,
    params: &PixelParams,
    l0: f32,
    t0: i64,
    l1: f32,
    t1: i64,
    emit: &mut impl FnMut(i64, i8),
) -> u32 {
    let mut emitted = 0u32;
    if l1 > l0 {
        while l1 >= state.l_ref + params.c_pos {
            let lc = state.l_ref + params.c_pos;
            let t_ev = crossing_time(l0, t0, l1, t1, lc);
            state.l_ref = lc;
            if state.t_last == NO_EVENT || t_ev - state.t_last >= params.refractory_ns {
                emit(t_ev, 1);
                state.t_last = t_ev;
                emitted += 1;
            }
        }
    } else if l1 < l0 {
        while l1 <= state.l_ref - params.c_neg {
            let lc = state.l_ref - params.c_neg;
            let t_ev = crossing_time(l0, t0, l1, t1, lc);
            state.l_ref = lc;
            if state.t_last == NO_EVENT || t_ev - state.t_last >= params.refractory_ns {
                emit(t_ev, -1);
                state.t_last = t_ev;
                emitted += 1;
            }
        }
    }
    emitted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(
        state: &mut PixelState,
        p: &PixelParams,
        l0: f32,
        t0: i64,
        l1: f32,
        t1: i64,
    ) -> Vec<(i64, i8)> {
        let mut out = Vec::new();
        step_pixel(state, p, l0, t0, l1, t1, &mut |t, pol| out.push((t, pol)));
        out
    }

    fn params(c: f32, refractory_ns: i64) -> PixelParams {
        PixelParams {
            c_pos: c,
            c_neg: c,
            refractory_ns,
        }
    }

    #[test]
    fn linear_ramp_emits_evenly_spaced_positive_events() {
        // L rises 0 -> 1 over 1 s with c = 0.125 (exact in binary, so the f32
        // reference accumulates without drift): crossings at 0.125 s, ..., 1.0 s.
        let mut s = PixelState {
            l_ref: 0.0,
            t_last: NO_EVENT,
        };
        let ev = collect(&mut s, &params(0.125, 0), 0.0, 0, 1.0, 1_000_000_000);
        assert_eq!(ev.len(), 8);
        for (i, &(t, pol)) in ev.iter().enumerate() {
            let expected = (i as i64 + 1) * 125_000_000;
            assert!((t - expected).abs() <= 1, "event {i}: {t} vs {expected}");
            assert_eq!(pol, 1);
        }
        assert!((s.l_ref - 1.0).abs() < 1e-5);
        assert_eq!(s.t_last, ev.last().unwrap().0);
    }

    #[test]
    fn falling_then_rising_tracks_reference() {
        let mut s = PixelState {
            l_ref: 1.0,
            t_last: NO_EVENT,
        };
        let p = params(0.25, 0);
        let down = collect(&mut s, &p, 1.0, 0, 0.0, 1_000);
        assert_eq!(down.len(), 4);
        assert!(down.iter().all(|&(_, pol)| pol == -1));
        assert_eq!(down[0].0, 250);
        let up = collect(&mut s, &p, 0.0, 1_000, 0.6, 2_000);
        assert_eq!(up.len(), 2);
        assert!(up.iter().all(|&(_, pol)| pol == 1));
        // Residual 0.1 below the next crossing stays: reference is 0.5.
        assert!((s.l_ref - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sub_threshold_change_emits_nothing_and_keeps_reference() {
        let mut s = PixelState {
            l_ref: 0.0,
            t_last: NO_EVENT,
        };
        let ev = collect(&mut s, &params(0.5, 0), 0.0, 0, 0.49, 100);
        assert!(ev.is_empty());
        assert_eq!(s.l_ref, 0.0);
    }

    #[test]
    fn refractory_drops_events_but_moves_reference() {
        // 8 crossings 125 ns apart (c = 0.125); refractory 300 ns keeps 125, 500, 875.
        let mut s = PixelState {
            l_ref: 0.0,
            t_last: NO_EVENT,
        };
        let ev = collect(&mut s, &params(0.125, 300), 0.0, 0, 1.0, 1_000);
        let times: Vec<i64> = ev.iter().map(|e| e.0).collect();
        assert_eq!(times, vec![125, 500, 875]);
        assert!((s.l_ref - 1.0).abs() < 1e-5);
    }

    #[test]
    fn refractory_spans_steps() {
        let mut s = PixelState {
            l_ref: 0.0,
            t_last: 950,
        };
        let ev = collect(&mut s, &params(0.125, 100), 0.0, 1_000, 0.125, 1_020);
        assert!(ev.is_empty(), "1020 - 950 < 100 must be dropped");
        assert_eq!(s.l_ref, 0.125);
    }

    #[test]
    fn crossing_time_is_rounded_to_nearest_ns() {
        // c = 0.25, l 0 -> 0.75 in 10 ns: crossings at 3.33 -> 3, 6.67 -> 7, 10.
        let mut s = PixelState {
            l_ref: 0.0,
            t_last: NO_EVENT,
        };
        let ev = collect(&mut s, &params(0.25, 0), 0.0, 0, 0.75, 10);
        let times: Vec<i64> = ev.iter().map(|e| e.0).collect();
        assert_eq!(times, vec![3, 7, 10]);
    }

    #[test]
    fn reference_above_l0_after_dropped_events_still_clamps_fraction() {
        // l_ref may sit outside [l0, l1] by rounding; the fraction is clamped to [0, 1].
        let mut s = PixelState {
            l_ref: 0.05,
            t_last: NO_EVENT,
        };
        let ev = collect(&mut s, &params(0.1, 0), 0.0, 0, 0.16, 100);
        assert_eq!(ev.len(), 1);
        assert!(ev[0].0 >= 0 && ev[0].0 <= 100);
    }
}
