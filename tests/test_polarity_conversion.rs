use evlib::ev_formats::polarity_handler::{PolarityConfig, PolarityEncoding, PolarityHandler};
use evlib::ev_formats::LoadConfig;

#[test]
fn test_polarity_conversion_zero_one_to_minus_one_plus_one() {
    // Test basic conversion from 0/1 to -1/1
    let handler = PolarityHandler::new(); // Uses default config: 0/1 -> -1/1

    // Test individual conversions
    assert_eq!(handler.convert_single(0).unwrap(), -1);
    assert_eq!(handler.convert_single(1).unwrap(), 1);

    // Test batch conversion
    let input = vec![0, 1, 0, 1, 0, 1, 1, 0];
    let output = handler.convert_batch(&input).unwrap();
    let expected = vec![-1, 1, -1, 1, -1, 1, 1, -1];
    assert_eq!(output, expected);

    println!("OK: Basic polarity conversion test passed");
}

#[test]
fn test_polarity_auto_detection() {
    // Test auto-detection of 0/1 encoding
    let zero_one_data = vec![0, 1, 0, 1, 0, 1, 1, 0];
    let detected_encoding = PolarityHandler::auto_detect_encoding(&zero_one_data).unwrap();
    assert_eq!(detected_encoding, PolarityEncoding::ZeroOne);

    // Test auto-detection of -1/1 encoding
    let minus_one_plus_one_data = vec![-1, 1, -1, 1, -1, 1, 1, -1];
    let detected_encoding =
        PolarityHandler::auto_detect_encoding(&minus_one_plus_one_data).unwrap();
    assert_eq!(detected_encoding, PolarityEncoding::MinusOnePlusOne);

    println!("OK: Polarity auto-detection test passed");
}

#[test]
fn test_polarity_stats() {
    // Test polarity statistics
    let data = vec![0, 1, 0, 1, 0, 1, 1, 0];
    let stats = PolarityHandler::analyze_polarity_distribution(&data);

    assert_eq!(stats.total_count, 8);
    assert_eq!(stats.zero_count, 4);
    assert_eq!(stats.one_count, 4);
    assert_eq!(stats.minus_one_count, 0);
    assert_eq!(stats.zero_ratio(), 0.5);
    assert_eq!(stats.one_ratio(), 0.5);
    assert_eq!(stats.suggest_encoding(), PolarityEncoding::ZeroOne);

    println!("OK: Polarity statistics test passed");
}

#[test]
fn test_load_config_with_polarity_encoding() {
    // Test LoadConfig with polarity encoding
    let config = LoadConfig::new().with_polarity_encoding(PolarityEncoding::ZeroOne);

    assert_eq!(config.polarity_encoding, Some(PolarityEncoding::ZeroOne));

    // Test chaining with other configurations
    let config = LoadConfig::new()
        .with_polarity_encoding(PolarityEncoding::MinusOnePlusOne)
        .with_sorting(true)
        .with_polarity(Some(true));

    assert_eq!(
        config.polarity_encoding,
        Some(PolarityEncoding::MinusOnePlusOne)
    );
    assert!(config.sort);
    assert_eq!(config.polarity, Some(true));

    println!("OK: LoadConfig polarity encoding test passed");
}

#[test]
fn test_polarity_config_variations() {
    // Test different polarity configurations
    let config = PolarityConfig {
        input_encoding: PolarityEncoding::ZeroOne,
        output_encoding: PolarityEncoding::MinusOnePlusOne,
        validate_input: true,
        reject_invalid: false,
    };

    let handler = PolarityHandler::with_config(config);

    // Test valid conversions
    assert_eq!(handler.convert_single(0).unwrap(), -1);
    assert_eq!(handler.convert_single(1).unwrap(), 1);

    // Test non-strict mode with invalid input
    let config_non_strict = PolarityConfig {
        input_encoding: PolarityEncoding::ZeroOne,
        output_encoding: PolarityEncoding::MinusOnePlusOne,
        validate_input: false,
        reject_invalid: false,
    };

    let handler_non_strict = PolarityHandler::with_config(config_non_strict);

    // Invalid values should be interpreted as boolean
    assert_eq!(handler_non_strict.convert_single(2).unwrap(), 1); // non-zero -> positive
    assert_eq!(handler_non_strict.convert_single(-5).unwrap(), 1); // non-zero -> positive

    println!("OK: Polarity config variations test passed");
}

#[test]
fn test_reverse_polarity_conversion() {
    // Test conversion from -1/1 to 0/1
    let config = PolarityConfig {
        input_encoding: PolarityEncoding::MinusOnePlusOne,
        output_encoding: PolarityEncoding::ZeroOne,
        validate_input: true,
        reject_invalid: false,
    };

    let handler = PolarityHandler::with_config(config);

    // Test individual conversions
    assert_eq!(handler.convert_single(-1).unwrap(), 0);
    assert_eq!(handler.convert_single(1).unwrap(), 1);

    // Test batch conversion
    let input = vec![-1, 1, -1, 1, -1, 1, 1, -1];
    let output = handler.convert_batch(&input).unwrap();
    let expected = vec![0, 1, 0, 1, 0, 1, 1, 0];
    assert_eq!(output, expected);

    println!("OK: Reverse polarity conversion test passed");
}

#[test]
fn test_polarity_error_handling() {
    // Test error handling with strict validation
    let config = PolarityConfig {
        input_encoding: PolarityEncoding::ZeroOne,
        output_encoding: PolarityEncoding::MinusOnePlusOne,
        validate_input: true,
        reject_invalid: true,
    };

    let handler = PolarityHandler::with_config(config);

    // Should fail with invalid values
    assert!(handler.convert_single(2).is_err());
    assert!(handler.convert_single(-5).is_err());

    // Should succeed with valid values
    assert!(handler.convert_single(0).is_ok());
    assert!(handler.convert_single(1).is_ok());

    println!("OK: Polarity error handling test passed");
}
