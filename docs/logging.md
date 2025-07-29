# Structured Logging with Tracing

evlib uses the [`tracing`](https://docs.rs/tracing/latest/tracing/) crate for structured, configurable logging. This provides better control over log verbosity, structured output, and integration with external logging systems.

## Quick Start

### Basic Setup

```rust
use evlib::tracing_config;

// Initialize with sensible defaults
tracing_config::init();

// Now load events - logging will be structured and configurable
let events = evlib::ev_formats::load_events_with_config("tests/data/eTram/h5/val_2/val_night_011_td.h5", &config)?;
```

### Environment Variable Control

Control logging via the `RUST_LOG` environment variable:

```bash
# Default: INFO level for evlib, WARN for everything else
cargo run

# Enable debug logging for evlib
RUST_LOG=evlib=debug cargo run

# Only show warnings and errors
RUST_LOG=evlib=warn cargo run

# Very detailed tracing for specific module
RUST_LOG=evlib::ev_formats::prophesee_ecf_codec=trace cargo run

# Multiple module control
RUST_LOG="evlib::ev_formats=debug,evlib::ev_processing=info" cargo run
```

## Log Levels

evlib uses the following log levels:

### `ERROR` - Critical Failures
- Model loading failures
- File parsing errors
- System-level problems
- Fallback mechanism failures

Example output:
```
2024-01-15T10:30:45.123456Z ERROR evlib::ev_formats::mod: Python fallback failed error="File not found"
```

### `WARN` - Warnings and Fallbacks
- Fallback behaviors activated
- Unsupported file formats
- Missing optional dependencies
- Performance degradation warnings

Example output:
```
2024-01-15T10:30:45.123456Z  WARN evlib::ev_formats::prophesee_ecf_codec: Unknown delta_bits value in ECF decoder delta_bits=7
```

### `INFO` - Status and Progress
- File loading progress
- Model downloading status
- Successful operations
- Stream status updates

Example output:
```
2024-01-15T10:30:45.123456Z  INFO evlib::ev_formats::mod: Native Rust ECF decoder loaded events events=1048576
```

### `DEBUG` - Development Information
- Detailed ECF decoding steps
- WebSocket connection details
- Model verification results
- Internal state information

Example output:
```
2024-01-15T10:30:45.123456Z DEBUG evlib::ev_formats::prophesee_ecf_codec: ECF base timestamp base_timestamp=1627849200000000
```

### `TRACE` - Very Detailed Debug
- Low-level operations
- Performance timing
- Memory allocation details

*Note: Currently no TRACE level logs in evlib, reserved for future use.*

## Configuration Options

### Default Configuration

```rust
use evlib::tracing_config;

// Standard configuration - good for most use cases
tracing_config::init();
```

### Debug Configuration

```rust
use evlib::tracing_config;

// Enables debug logging with file/line information
tracing_config::init_debug();
```

### Custom Configuration

```rust
use evlib::tracing_config;

// Fine-grained control over specific modules
tracing_config::init_with_filter("evlib::ev_formats=trace,evlib=info");
```

### Production Configuration

```rust
use evlib::tracing_config::examples;

// JSON output suitable for log aggregation
examples::init_production();
```

### Development Configuration

```rust
use evlib::tracing_config::examples;

// Pretty-printed output with colors and extra context
examples::init_development();
```

## Integration Examples

### With `env_logger` style

```rust
use tracing_subscriber::{EnvFilter, fmt};

// Similar to env_logger but with structured output
fmt()
    .with_env_filter(EnvFilter::from_default_env())
    .init();
```

### With `syslog`

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

tracing_subscriber::registry()
    .with(tracing_syslog::layer())
    .with(EnvFilter::from_default_env())
    .init();
```

### With File Output

```rust
use tracing_subscriber::{fmt, EnvFilter};
use std::fs::File;

let file = File::create("evlib.log")?;
fmt()
    .with_writer(file)
    .with_env_filter(EnvFilter::from_default_env())
    .init();
```

### With JSON for Kubernetes/Docker

```rust
use tracing_subscriber::{fmt, EnvFilter};

fmt()
    .json()
    .with_env_filter(EnvFilter::from_default_env())
    .init();
```

## Testing Integration

```rust
#[cfg(test)]
mod tests {
    use evlib::tracing_config;

    #[test]
    fn test_event_loading() {
        // Initialize tracing for tests
        tracing_config::init_test();

        // Test code here - logs will be captured in test output
        let events = evlib::ev_formats::load_events_from_text("tests/data/test.txt")?;
    }
}
```

## Python Integration

When using evlib from Python, you can still control Rust logging:

```python
import os
import evlib

# Set logging level before importing/using evlib
os.environ['RUST_LOG'] = 'evlib=debug'

# Initialize tracing from Python (if exposed)
# Note: This requires exposing tracing_config functions to Python
# evlib.tracing_config.init()

# Load events - Rust logs will be controlled by RUST_LOG
events = evlib.load_events('tests/data/eTram/h5/val_2/val_night_011_td.h5')
print(f"Loaded events with shape: {events.collect().shape}")
```

## Performance Considerations

### Runtime Overhead
- `trace!` and `debug!` macros compile to no-ops when disabled
- `info!`, `warn!`, `error!` have minimal overhead when disabled
- Structured logging is more efficient than string formatting
- Filtering happens at compile time when possible

### Memory Usage
- Structured logs use less memory than formatted strings
- Span context tracking has minimal overhead
- Field values are lazily formatted only when logged

## Common Patterns

### Module-Specific Debugging

```bash
# Debug only ECF decoding
RUST_LOG=evlib::ev_formats::prophesee_ecf_codec=debug cargo run

# Debug all format readers
RUST_LOG=evlib::ev_formats=debug cargo run

# Debug model loading
RUST_LOG=evlib::ev_processing::model_zoo=debug cargo run
```

### Production Logging

```bash
# Minimal logging for production
RUST_LOG=evlib=warn cargo run

# Structured JSON for log aggregation
RUST_LOG=evlib=info cargo run > logs.json
```

### Development Debugging

```bash
# Comprehensive debugging
RUST_LOG=evlib=debug cargo run

# Very detailed tracing
RUST_LOG=evlib=trace cargo run
```

## Migration from println!/eprintln!

Old code using print statements:
```rust
eprintln!("Loading {} events from file", events.len());
println!("ECF decoder failed: {}", error);
```

New structured logging:
```rust
use tracing::{info, error};

info!(events = events.len(), "Loading events from file");
error!(error = %error, "ECF decoder failed");
```

## Benefits

1. **Configurable Verbosity**: Control logging at runtime without recompilation
2. **Structured Output**: Key-value pairs for better parsing and filtering
3. **Performance**: Debug logs compile to no-ops when disabled
4. **Integration**: Works with existing Rust logging ecosystem
5. **Consistency**: Uniform logging patterns across the entire library
6. **Observability**: Better debugging and monitoring capabilities

## Troubleshooting

### No Logs Appearing

1. Check that tracing is initialized: `tracing_config::init()`
2. Verify `RUST_LOG` environment variable: `RUST_LOG=evlib=debug`
3. Ensure log level is appropriate for desired messages

### Too Many Logs

1. Increase filter level: `RUST_LOG=evlib=warn`
2. Use module-specific filtering: `RUST_LOG=evlib::ev_formats=info`
3. Switch to production configuration for cleaner output

### JSON Parsing Issues

1. Use `fmt().json()` for consistent JSON output
2. Ensure no mixed output formats in configuration
3. Consider using dedicated JSON logging libraries for complex parsing

For more information, see the [tracing documentation](https://docs.rs/tracing/latest/tracing/) and [tracing-subscriber documentation](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/).
