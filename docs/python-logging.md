# Python Integration with evlib Tracing

evlib provides structured logging through Rust's `tracing` crate, which can be controlled from Python for better debugging and observability.

## Quick Start

### Basic Logging Setup

```python
import evlib

# Initialize logging with default configuration
evlib.tracing_config.init()

# Now all evlib operations will have structured logging
events = evlib.load_events('data.h5')
```

### Environment Variable Control

The most convenient way to control logging is via environment variables:

```python
import os
import evlib

# Set before importing or initializing
os.environ['RUST_LOG'] = 'evlib=debug'

# Initialize tracing
evlib.tracing_config.init()

# All evlib operations will now log at debug level
events = evlib.load_events('large_file.h5')  # Will show progress messages
```

## Configuration Options

### Default Configuration
```python
import evlib

# Standard setup - INFO level for evlib, WARN for everything else
evlib.tracing_config.init()
```

### Debug Configuration
```python
import evlib

# Enables debug logging with file/line information
evlib.tracing_config.init_debug()
```

### Custom Filter Configuration
```python
import evlib

# Fine-grained control over specific modules
evlib.tracing_config.init_with_filter("evlib::ev_formats=trace,evlib=info")

# Only show ECF decoder messages
evlib.tracing_config.init_with_filter("evlib::ev_formats::prophesee_ecf_codec=debug")

# Production: only warnings and errors
evlib.tracing_config.init_with_filter("evlib=warn")
```

### Production Configuration
```python
import evlib

# JSON structured output suitable for log aggregation
evlib.tracing_config.init_production()
```

### Development Configuration
```python
import evlib

# Pretty-printed output with colors and extra context
evlib.tracing_config.init_development()
```

## Environment Variable Examples

### Basic Usage
```bash
# Run Python script with evlib debug logging
RUST_LOG=evlib=debug python my_script.py

# Only show warnings and errors
RUST_LOG=evlib=warn python my_script.py

# Very detailed ECF decoder logging
RUST_LOG=evlib::ev_formats::prophesee_ecf_codec=trace python my_script.py
```

### Module-Specific Control
```bash
# Debug format loading, info for everything else
RUST_LOG="evlib::ev_formats=debug,evlib=info" python my_script.py

# Debug model loading
RUST_LOG="evlib::ev_processing::model_zoo=debug" python my_script.py

# Trace-level logging for specific operations
RUST_LOG="evlib::ev_formats::prophesee_ecf_codec=trace" python my_script.py
```

## Integration with Python Logging

### Separate Rust and Python Logging

```python
import logging
import os
import evlib

# Set up Python logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up Rust logging separately
os.environ['RUST_LOG'] = 'evlib=info'
evlib.tracing_config.init()

# Use both logging systems
logger.info("Starting Python processing")  # Python log
events = evlib.load_events('data.h5')      # Rust logs from evlib
logger.info(f"Loaded {len(events.collect())} events")  # Python log
```

### Coordinated Logging Levels

```python
import logging
import os
import evlib

def setup_logging(debug=False):
    """Set up both Python and Rust logging with coordinated levels."""
    if debug:
        # Debug mode: verbose logging for both
        logging.basicConfig(level=logging.DEBUG)
        os.environ['RUST_LOG'] = 'evlib=debug'
        evlib.tracing_config.init_debug()
    else:
        # Production mode: minimal logging
        logging.basicConfig(level=logging.INFO)
        os.environ['RUST_LOG'] = 'evlib=warn'
        evlib.tracing_config.init_production()

# Usage
setup_logging(debug=True)  # Development
# setup_logging(debug=False)  # Production
```

## Common Use Cases

### Development and Debugging

```python
import os
import evlib

# Enable detailed debugging
os.environ['RUST_LOG'] = 'evlib=debug'
evlib.tracing_config.init_development()

# Load events - will show detailed progress and debug info
events = evlib.load_events('large_prophesee_file.h5')

# Create representations - will show internal processing details
import evlib.representations as evr
voxels = evr.create_voxel_grid(events.collect(), height=480, width=640, nbins=10)
```

### Production Monitoring

```python
import os
import evlib
import json

# Production logging with JSON output for log aggregation
os.environ['RUST_LOG'] = 'evlib=info'
evlib.tracing_config.init_production()

try:
    events = evlib.load_events('sensor_data.h5')
    # Processing will log structured JSON messages
    processed_data = process_events(events)
except Exception as e:
    # Error messages will be in structured JSON format
    raise
```

### Specific Module Debugging

```python
import os
import evlib

# Debug only ECF decoding issues
os.environ['RUST_LOG'] = 'evlib::ev_formats::prophesee_ecf_codec=trace'
evlib.tracing_config.init_debug()

# This will show very detailed ECF decoding steps
events = evlib.load_events('problematic_ecf_file.h5')
```

### Testing and CI/CD

```python
import os
import evlib
import pytest

@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    # Minimal logging during tests to reduce noise
    os.environ['RUST_LOG'] = 'evlib=warn'
    evlib.tracing_config.init()

def test_event_loading():
    # Only warnings and errors will be shown
    events = evlib.load_events('test_data.h5')
    assert len(events.collect()) > 0
```

## Log Message Format

### Default Format
```
2024-01-15T10:30:45.123456Z  INFO evlib::ev_formats::mod: Python fallback loaded events events=1048576
2024-01-15T10:30:45.123456Z  WARN evlib::ev_formats::prophesee_ecf_codec: Unknown delta_bits value delta_bits=7
```

### JSON Format (Production)
```json
{"timestamp":"2024-01-15T10:30:45.123456Z","level":"INFO","target":"evlib::ev_formats::mod","message":"Python fallback loaded events","events":1048576}
{"timestamp":"2024-01-15T10:30:45.123456Z","level":"WARN","target":"evlib::ev_formats::prophesee_ecf_codec","message":"Unknown delta_bits value","delta_bits":7}
```

### Pretty Format (Development)
```
2024-01-15T10:30:45.123456Z  INFO evlib::ev_formats::mod: Python fallback loaded events
    at src/ev_formats/mod.rs:336
    in evlib::ev_formats::load_events_from_hdf5
  events: 1048576
```

## Performance Considerations

### Initialization Cost
- Tracing initialization has minimal overhead
- Call `init()` once at application startup
- Avoid reinitializing multiple times

### Runtime Performance
- Debug/trace logs compile to no-ops when disabled
- Structured logging is more efficient than string formatting
- Environment variable filtering is very fast

### Memory Usage
```python
import os
import evlib

# Low memory usage - only warnings and errors
os.environ['RUST_LOG'] = 'evlib=warn'
evlib.tracing_config.init()

# Higher memory usage - all debug messages
os.environ['RUST_LOG'] = 'evlib=debug'
evlib.tracing_config.init_debug()
```

## Troubleshooting

### No Rust Logs Appearing

1. **Check initialization**:
   ```python
   import evlib
   evlib.tracing_config.init()  # Must call this!
   ```

2. **Check environment variable**:
   ```python
   import os
   os.environ['RUST_LOG'] = 'evlib=debug'  # Set before init()
   evlib.tracing_config.init()
   ```

3. **Verify log level**:
   ```python
   # Use debug level to see more messages
   evlib.tracing_config.init_debug()
   ```

### Too Many Log Messages

1. **Increase filter level**:
   ```python
   import os
   os.environ['RUST_LOG'] = 'evlib=warn'  # Only warnings and errors
   evlib.tracing_config.init()
   ```

2. **Use production config**:
   ```python
   evlib.tracing_config.init_production()  # Cleaner output
   ```

### Mixed Python/Rust Logging

1. **Separate concerns**:
   ```python
   import logging
   import evlib

   # Configure Python logging
   logging.basicConfig(level=logging.INFO)

   # Configure Rust logging separately
   evlib.tracing_config.init_with_filter("evlib=info")
   ```

## Advanced Usage

### Custom Output Formatting

While you can't directly control Rust log formatting from Python, you can:

1. **Capture and process output**:
```python
import subprocess
import os

# Run with specific logging and capture output
env = os.environ.copy()
env['RUST_LOG'] = 'evlib=info'

result = subprocess.run([
    'python', '-c',
    'import evlib; evlib.tracing_config.init(); evlib.load_events("data.h5")'
], env=env, capture_output=True, text=True)

# Process the captured log output
print(result.stderr)  # Tracing logs go to stderr by default
```

2. **File-based logging coordination**:
```python
import evlib
import logging

# Configure file output for both systems
logging.basicConfig(filename='app.log', level=logging.INFO)

# Rust logs will go to stderr, but can be redirected
evlib.tracing_config.init_with_filter("evlib=info")
```

For more advanced use cases, see the main [logging documentation](logging.md).
