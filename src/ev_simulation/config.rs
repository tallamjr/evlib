//! Simulator configuration, error type, and per-pixel threshold maps.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::fmt;

/// Lowest per-pixel contrast threshold after mismatch sampling.
pub const THRESHOLD_FLOOR: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulatorConfig {
    pub width: u32,
    pub height: u32,
    pub c_pos: f32,
    pub c_neg: f32,
    /// Relative std-dev of the per-pixel threshold mismatch; 0 disables it.
    pub threshold_sigma: f32,
    pub refractory_ns: i64,
    /// Additive epsilon inside the log: L = ln(I + log_eps).
    pub log_eps: f32,
    pub seed: u64,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            c_pos: 0.2,
            c_neg: 0.2,
            threshold_sigma: 0.0,
            refractory_ns: 0,
            log_eps: 1e-3,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimError {
    InvalidConfig(String),
    ShapeMismatch {
        expected: usize,
        got: usize,
    },
    NonMonotonicTime {
        index: usize,
    },
    /// A float32 log frame holds NaN or an infinity at this flat (T*H*W) index.
    InvalidInput {
        index: usize,
    },
    EmptyBatch,
    NotInitialised,
    Backend(String),
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimError::InvalidConfig(m) => write!(f, "invalid simulator config: {m}"),
            SimError::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "frame length {got} does not match width*height*frames = {expected}"
                )
            }
            SimError::NonMonotonicTime { index } => {
                write!(
                    f,
                    "timestamps must be strictly increasing; violation at index {index}"
                )
            }
            SimError::InvalidInput { index } => {
                write!(f, "non-finite log intensity at flat index {index}")
            }
            SimError::EmptyBatch => write!(f, "no frames supplied"),
            SimError::NotInitialised => write!(f, "simulator has not seen a first frame"),
            SimError::Backend(m) => write!(f, "backend error: {m}"),
        }
    }
}

impl std::error::Error for SimError {}

impl SimulatorConfig {
    /// Negated comparisons are deliberate: they reject NaN thresholds, which
    /// a direct `<=`/`<` comparison would silently let through.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn validate(&self) -> Result<(), SimError> {
        if self.width == 0 || self.height == 0 {
            return Err(SimError::InvalidConfig(
                "width and height must be positive".into(),
            ));
        }
        // x/y are cast to i16 for the Python-facing event arrays; keep both in bound.
        if self.width > i16::MAX as u32 || self.height > i16::MAX as u32 {
            return Err(SimError::InvalidConfig(
                "width and height must be at most 32767".into(),
            ));
        }
        if !(self.c_pos > 0.0) || !(self.c_neg > 0.0) {
            return Err(SimError::InvalidConfig(
                "c_pos and c_neg must be positive".into(),
            ));
        }
        if self.refractory_ns < 0 {
            return Err(SimError::InvalidConfig(
                "refractory_ns must be non-negative".into(),
            ));
        }
        if !(self.threshold_sigma >= 0.0) {
            return Err(SimError::InvalidConfig(
                "threshold_sigma must be non-negative".into(),
            ));
        }
        if !(self.log_eps > 0.0) {
            return Err(SimError::InvalidConfig("log_eps must be positive".into()));
        }
        Ok(())
    }

    pub fn pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// Per-pixel (positive, negative) threshold maps, row major.
///
/// With `threshold_sigma == 0` both maps are the nominal values. Otherwise each
/// pixel draws N(c, sigma * c) from a ChaCha8 stream seeded by `seed`, clipped
/// at `THRESHOLD_FLOOR`. Positive map first, then negative, from one stream.
pub fn threshold_maps(cfg: &SimulatorConfig) -> (Vec<f32>, Vec<f32>) {
    let n = cfg.pixels();
    if cfg.threshold_sigma == 0.0 {
        return (vec![cfg.c_pos; n], vec![cfg.c_neg; n]);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let draw = |c: f32, rng: &mut ChaCha8Rng| -> Vec<f32> {
        let normal = Normal::new(c, cfg.threshold_sigma * c)
            .expect("validated sigma * c is finite and non-negative");
        (0..n)
            .map(|_| normal.sample(rng).max(THRESHOLD_FLOOR))
            .collect()
    };
    let pos = draw(cfg.c_pos, &mut rng);
    let neg = draw(cfg.c_neg, &mut rng);
    (pos, neg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SimulatorConfig {
        SimulatorConfig {
            width: 4,
            height: 3,
            ..SimulatorConfig::default()
        }
    }

    #[test]
    fn default_config_validates_with_shape() {
        assert!(cfg().validate().is_ok());
    }

    #[test]
    fn zero_shape_is_invalid() {
        let c = SimulatorConfig::default();
        assert!(matches!(c.validate(), Err(SimError::InvalidConfig(_))));
    }

    #[test]
    fn oversized_shape_is_invalid() {
        // width must fit in i16 (the Python-facing x array dtype); 40000 > 32767.
        let c = SimulatorConfig {
            width: 40000,
            ..cfg()
        };
        assert!(matches!(c.validate(), Err(SimError::InvalidConfig(_))));
    }

    #[test]
    fn non_positive_threshold_is_invalid() {
        let c = SimulatorConfig {
            c_pos: 0.0,
            ..cfg()
        };
        assert!(matches!(c.validate(), Err(SimError::InvalidConfig(_))));
        let c = SimulatorConfig {
            c_neg: -0.1,
            ..cfg()
        };
        assert!(matches!(c.validate(), Err(SimError::InvalidConfig(_))));
    }

    #[test]
    fn negative_refractory_sigma_or_eps_is_invalid() {
        assert!(SimulatorConfig {
            refractory_ns: -1,
            ..cfg()
        }
        .validate()
        .is_err());
        assert!(SimulatorConfig {
            threshold_sigma: -0.1,
            ..cfg()
        }
        .validate()
        .is_err());
        assert!(SimulatorConfig {
            log_eps: 0.0,
            ..cfg()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn sigma_zero_gives_nominal_maps() {
        let (pos, neg) = threshold_maps(&cfg());
        assert_eq!(pos.len(), 12);
        assert!(pos.iter().all(|&v| v == 0.2));
        assert!(neg.iter().all(|&v| v == 0.2));
    }

    #[test]
    fn seeded_maps_are_reproducible_and_clipped() {
        let c = SimulatorConfig {
            threshold_sigma: 5.0,
            seed: 42,
            ..cfg()
        };
        let (p1, n1) = threshold_maps(&c);
        let (p2, n2) = threshold_maps(&c);
        assert_eq!(p1, p2);
        assert_eq!(n1, n2);
        assert!(p1.iter().chain(n1.iter()).all(|&v| v >= 0.01));
        // A huge sigma with 12 pixels produces at least one clipped value.
        assert!(p1.iter().chain(n1.iter()).any(|&v| v == 0.01));
        let other = threshold_maps(&SimulatorConfig { seed: 43, ..c });
        assert_ne!(p1, other.0);
    }
}
