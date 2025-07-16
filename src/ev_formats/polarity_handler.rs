/// Polarity encoding handler for event camera data
///
/// Different event camera formats use different polarity encoding schemes:
/// - EVT2/Prophesee: 0=negative, 1=positive
/// - dv-processing: false=negative, true=positive
/// - evlib internal: -1=negative, 1=positive
/// - AEDAT: varies by version
///
/// This module provides conversion between different polarity encodings.
use std::collections::HashMap;

/// Supported polarity encoding schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolarityEncoding {
    /// 0=negative, 1=positive (most common in real data)
    #[default]
    ZeroOne,
    /// -1=negative, 1=positive (evlib internal representation)
    MinusOnePlusOne,
    /// false=negative, true=positive (dv-processing)
    TrueFalse,
    /// Auto-detect based on data analysis
    Auto,
}

/// Polarity conversion configuration
#[derive(Debug, Clone)]
pub struct PolarityConfig {
    /// Input encoding scheme
    pub input_encoding: PolarityEncoding,
    /// Output encoding scheme
    pub output_encoding: PolarityEncoding,
    /// Whether to validate polarity values
    pub validate_input: bool,
    /// Whether to reject invalid polarity values
    pub reject_invalid: bool,
}

impl Default for PolarityConfig {
    fn default() -> Self {
        PolarityConfig {
            input_encoding: PolarityEncoding::ZeroOne,
            output_encoding: PolarityEncoding::MinusOnePlusOne,
            validate_input: true,
            reject_invalid: false,
        }
    }
}

/// Polarity conversion errors
#[derive(Debug)]
pub enum PolarityError {
    InvalidInputValue(i32),
    UnsupportedConversion(PolarityEncoding, PolarityEncoding),
    AutoDetectionFailed,
}

impl std::fmt::Display for PolarityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolarityError::InvalidInputValue(value) => {
                write!(f, "Invalid polarity value: {}", value)
            }
            PolarityError::UnsupportedConversion(from, to) => {
                write!(f, "Unsupported conversion from {:?} to {:?}", from, to)
            }
            PolarityError::AutoDetectionFailed => {
                write!(f, "Failed to auto-detect polarity encoding")
            }
        }
    }
}

impl std::error::Error for PolarityError {}

/// Polarity handler for converting between different encoding schemes
pub struct PolarityHandler {
    config: PolarityConfig,
}

impl PolarityHandler {
    /// Create a new polarity handler with default configuration
    pub fn new() -> Self {
        PolarityHandler {
            config: PolarityConfig::default(),
        }
    }

    /// Create a new polarity handler with custom configuration
    pub fn with_config(config: PolarityConfig) -> Self {
        PolarityHandler { config }
    }

    /// Auto-detect polarity encoding from a sample of values
    pub fn auto_detect_encoding(values: &[i32]) -> Result<PolarityEncoding, PolarityError> {
        if values.is_empty() {
            return Err(PolarityError::AutoDetectionFailed);
        }

        let mut value_counts: HashMap<i32, usize> = HashMap::new();

        // Count occurrences of each value
        for &value in values {
            *value_counts.entry(value).or_insert(0) += 1;
        }

        // Check for different encoding patterns
        let has_zero = value_counts.contains_key(&0);
        let has_one = value_counts.contains_key(&1);
        let has_minus_one = value_counts.contains_key(&-1);
        let has_true = value_counts.contains_key(&1) && !has_zero && !has_minus_one;
        let has_false = value_counts.contains_key(&0) && !has_minus_one;

        // Determine most likely encoding
        if has_zero && has_one && !has_minus_one {
            // 0/1 encoding (most common)
            Ok(PolarityEncoding::ZeroOne)
        } else if has_minus_one && has_one && !has_zero {
            // -1/1 encoding (evlib internal)
            Ok(PolarityEncoding::MinusOnePlusOne)
        } else if has_false && has_true {
            // Boolean encoding (dv-processing)
            Ok(PolarityEncoding::TrueFalse)
        } else {
            // Ambiguous or invalid
            Err(PolarityError::AutoDetectionFailed)
        }
    }

    /// Convert a single polarity value
    pub fn convert_single(&self, value: i32) -> Result<i8, PolarityError> {
        self.convert_polarity(
            value,
            self.config.input_encoding,
            self.config.output_encoding,
        )
    }

    /// Convert polarity value between different encodings
    pub fn convert_polarity(
        &self,
        value: i32,
        from: PolarityEncoding,
        to: PolarityEncoding,
    ) -> Result<i8, PolarityError> {
        // If encodings are the same, just validate and return
        if from == to {
            return self.validate_and_convert_simple(value, from);
        }

        // Convert from input encoding to normalized form
        let normalized = match from {
            PolarityEncoding::ZeroOne => {
                match value {
                    0 => -1i8, // negative
                    1 => 1i8,  // positive
                    _ => {
                        if self.config.validate_input {
                            return Err(PolarityError::InvalidInputValue(value));
                        }
                        // Try to interpret as boolean
                        if value == 0 {
                            -1i8
                        } else {
                            1i8
                        }
                    }
                }
            }
            PolarityEncoding::MinusOnePlusOne => {
                match value {
                    -1 => -1i8, // negative
                    1 => 1i8,   // positive
                    _ => {
                        if self.config.validate_input {
                            return Err(PolarityError::InvalidInputValue(value));
                        }
                        // Try to interpret as boolean
                        if value == 0 {
                            -1i8
                        } else {
                            1i8
                        }
                    }
                }
            }
            PolarityEncoding::TrueFalse => {
                match value {
                    0 => -1i8, // false = negative
                    1 => 1i8,  // true = positive
                    _ => {
                        if self.config.validate_input {
                            return Err(PolarityError::InvalidInputValue(value));
                        }
                        // Try to interpret as boolean
                        if value == 0 {
                            -1i8
                        } else {
                            1i8
                        }
                    }
                }
            }
            PolarityEncoding::Auto => {
                return Err(PolarityError::UnsupportedConversion(from, to));
            }
        };

        // Convert from normalized form to output encoding
        let result = match to {
            PolarityEncoding::ZeroOne => {
                if normalized == -1 {
                    0
                } else {
                    1
                }
            }
            PolarityEncoding::MinusOnePlusOne => normalized,
            PolarityEncoding::TrueFalse => {
                if normalized == -1 {
                    0
                } else {
                    1
                }
            }
            PolarityEncoding::Auto => {
                return Err(PolarityError::UnsupportedConversion(from, to));
            }
        };

        Ok(result)
    }

    /// Validate and convert a value assuming same encoding
    fn validate_and_convert_simple(
        &self,
        value: i32,
        encoding: PolarityEncoding,
    ) -> Result<i8, PolarityError> {
        match encoding {
            PolarityEncoding::ZeroOne => match value {
                0 => Ok(0),
                1 => Ok(1),
                _ => {
                    if self.config.validate_input {
                        Err(PolarityError::InvalidInputValue(value))
                    } else {
                        Ok(if value == 0 { 0 } else { 1 })
                    }
                }
            },
            PolarityEncoding::MinusOnePlusOne => match value {
                -1 => Ok(-1),
                1 => Ok(1),
                _ => {
                    if self.config.validate_input {
                        Err(PolarityError::InvalidInputValue(value))
                    } else {
                        Ok(if value == 0 { -1 } else { 1 })
                    }
                }
            },
            PolarityEncoding::TrueFalse => match value {
                0 => Ok(0),
                1 => Ok(1),
                _ => {
                    if self.config.validate_input {
                        Err(PolarityError::InvalidInputValue(value))
                    } else {
                        Ok(if value == 0 { 0 } else { 1 })
                    }
                }
            },
            PolarityEncoding::Auto => Err(PolarityError::UnsupportedConversion(encoding, encoding)),
        }
    }

    /// Convert a batch of polarity values
    pub fn convert_batch(&self, values: &[i32]) -> Result<Vec<i8>, PolarityError> {
        let mut result = Vec::with_capacity(values.len());

        for &value in values {
            result.push(self.convert_single(value)?);
        }

        Ok(result)
    }

    /// Get statistics about polarity values in a dataset
    pub fn analyze_polarity_distribution(values: &[i32]) -> PolarityStats {
        let mut stats = PolarityStats::default();

        for &value in values {
            stats.total_count += 1;

            match value {
                0 => stats.zero_count += 1,
                1 => stats.one_count += 1,
                -1 => stats.minus_one_count += 1,
                _ => stats.other_count += 1,
            }
        }

        stats
    }

    /// Update configuration
    pub fn set_config(&mut self, config: PolarityConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &PolarityConfig {
        &self.config
    }
}

impl Default for PolarityHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about polarity value distribution
#[derive(Debug, Default)]
pub struct PolarityStats {
    pub total_count: usize,
    pub zero_count: usize,
    pub one_count: usize,
    pub minus_one_count: usize,
    pub other_count: usize,
}

impl PolarityStats {
    /// Get the ratio of zero values
    pub fn zero_ratio(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.zero_count as f64 / self.total_count as f64
        }
    }

    /// Get the ratio of one values
    pub fn one_ratio(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.one_count as f64 / self.total_count as f64
        }
    }

    /// Get the ratio of minus one values
    pub fn minus_one_ratio(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.minus_one_count as f64 / self.total_count as f64
        }
    }

    /// Suggest the most likely encoding based on statistics
    pub fn suggest_encoding(&self) -> PolarityEncoding {
        if self.zero_count > 0 && self.one_count > 0 && self.minus_one_count == 0 {
            PolarityEncoding::ZeroOne
        } else if self.minus_one_count > 0 && self.one_count > 0 && self.zero_count == 0 {
            PolarityEncoding::MinusOnePlusOne
        } else {
            PolarityEncoding::ZeroOne // Default fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_one_to_minus_one_plus_one() {
        let handler = PolarityHandler::new();

        // Test conversion from 0/1 to -1/1
        assert_eq!(handler.convert_single(0).unwrap(), -1);
        assert_eq!(handler.convert_single(1).unwrap(), 1);
    }

    #[test]
    fn test_minus_one_plus_one_to_zero_one() {
        let config = PolarityConfig {
            input_encoding: PolarityEncoding::MinusOnePlusOne,
            output_encoding: PolarityEncoding::ZeroOne,
            validate_input: true,
            reject_invalid: false,
        };
        let handler = PolarityHandler::with_config(config);

        // Test conversion from -1/1 to 0/1
        assert_eq!(handler.convert_single(-1).unwrap(), 0);
        assert_eq!(handler.convert_single(1).unwrap(), 1);
    }

    #[test]
    fn test_auto_detect_zero_one() {
        let values = vec![0, 1, 0, 1, 0, 1, 1, 0];
        let encoding = PolarityHandler::auto_detect_encoding(&values).unwrap();
        assert_eq!(encoding, PolarityEncoding::ZeroOne);
    }

    #[test]
    fn test_auto_detect_minus_one_plus_one() {
        let values = vec![-1, 1, -1, 1, -1, 1, 1, -1];
        let encoding = PolarityHandler::auto_detect_encoding(&values).unwrap();
        assert_eq!(encoding, PolarityEncoding::MinusOnePlusOne);
    }

    #[test]
    fn test_batch_conversion() {
        let handler = PolarityHandler::new();
        let input = vec![0, 1, 0, 1, 0];
        let output = handler.convert_batch(&input).unwrap();
        assert_eq!(output, vec![-1, 1, -1, 1, -1]);
    }

    #[test]
    fn test_polarity_stats() {
        let values = vec![0, 1, 0, 1, 0, 1, 1, 0];
        let stats = PolarityHandler::analyze_polarity_distribution(&values);

        assert_eq!(stats.total_count, 8);
        assert_eq!(stats.zero_count, 4);
        assert_eq!(stats.one_count, 4);
        assert_eq!(stats.minus_one_count, 0);
        assert_eq!(stats.zero_ratio(), 0.5);
        assert_eq!(stats.one_ratio(), 0.5);
        assert_eq!(stats.suggest_encoding(), PolarityEncoding::ZeroOne);
    }

    #[test]
    fn test_invalid_polarity_validation() {
        let config = PolarityConfig {
            input_encoding: PolarityEncoding::ZeroOne,
            output_encoding: PolarityEncoding::MinusOnePlusOne,
            validate_input: true,
            reject_invalid: true,
        };
        let handler = PolarityHandler::with_config(config);

        // Should fail with invalid value
        assert!(handler.convert_single(2).is_err());
        assert!(handler.convert_single(-5).is_err());
    }

    #[test]
    fn test_invalid_polarity_non_strict() {
        let config = PolarityConfig {
            input_encoding: PolarityEncoding::ZeroOne,
            output_encoding: PolarityEncoding::MinusOnePlusOne,
            validate_input: false,
            reject_invalid: false,
        };
        let handler = PolarityHandler::with_config(config);

        // Should work with invalid values (interpreted as boolean)
        assert_eq!(handler.convert_single(2).unwrap(), 1); // non-zero -> positive
        assert_eq!(handler.convert_single(-5).unwrap(), 1); // non-zero -> positive
    }
}
