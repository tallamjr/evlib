# Event Camera Visualization (WASM Edition)

A high-performance event camera simulator that runs entirely in your browser using WebAssembly. This implementation uses the ESIM algorithm from evlib to convert webcam video into event streams in real-time.

## Features

- **Real-time event generation** using the ESIM algorithm
- **GPU-accelerated visualization** with WebGL (with 2D canvas fallback)
- **Zero installation** - runs entirely in the browser
- **Camera access** via getUserMedia API
- **Adjustable parameters** for threshold and decay time
- **Performance metrics** display

## Quick Start

### Prerequisites

- Modern web browser with WebAssembly support (Chrome, Firefox, Safari, Edge)
- Webcam connected to your computer
- Python 3 (for the development server)

### Running the Application

1. Navigate to the wasm-evlib directory:
   ```bash
   cd wasm-evlib
   ```

2. Build the WASM module (if not already built):
   ```bash
   ./build.sh
   ```

3. Start the development server:
   ```bash
   ./serve.py
   ```

4. Open your browser to:
   ```
   http://localhost:8080/
   ```

5. Click "Start Camera" and allow camera permissions when prompted

## Controls

- **Start/Stop Camera**: Toggle webcam capture
- **Pause/Resume**: Pause event generation without stopping the camera
- **Threshold Slider**: Adjust the contrast threshold (lower = more events)
- **Decay Slider**: Control how long events remain visible

## Technical Details

### Architecture

```
Webcam → Canvas → WASM ESIM → Event Buffer → WebGL Renderer
```

1. **Camera Capture**: Uses getUserMedia to access webcam at 640x480
2. **ESIM Processing**: WASM module implements the ESIM algorithm to detect log intensity changes
3. **Event Generation**: Generates events when intensity changes exceed threshold
4. **GPU Rendering**: WebGL renders events with temporal decay effects

### Performance

- Processes video at 30-60 FPS depending on hardware
- Can handle 100,000+ events simultaneously
- WASM module is ~200KB (optimized for size)
- No network requests after initial load

## Development

### Building from Source

1. Install Rust and wasm-pack:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

2. Build the WASM module:
   ```bash
   wasm-pack build --target web --out-dir pkg
   ```

### Project Structure

```
wasm-evlib/
├── src/
│   └── lib.rs          # WASM ESIM implementation
├── pkg/                # Generated WASM module
├── index.html          # Web interface
├── build.sh            # Build script
├── serve.py            # Development server
└── Cargo.toml          # Rust dependencies
```

### Customization

To adjust ESIM parameters, modify the defaults in `src/lib.rs`:

```rust
pub struct EsimConfig {
    pub contrast_threshold_pos: f64,  // Default: 0.15
    pub contrast_threshold_neg: f64,  // Default: 0.15
    pub refractory_period_ms: f64,   // Default: 0.1
}
```

## Deployment

The application can be deployed to any static web hosting service:

1. Build the WASM module
2. Copy `index.html` and the `pkg/` directory to your web server
3. Ensure proper MIME types for `.wasm` files
4. Enable CORS headers if needed

### Single-file Deployment

For ultimate portability, you can inline the WASM module into the HTML file:

```bash
# Convert WASM to base64 and embed in HTML
base64 pkg/wasm_evlib_bg.wasm > wasm_base64.txt
# Then manually update index.html to load from base64
```

## Browser Compatibility

- **Chrome/Edge**: Full support (recommended)
- **Firefox**: Full support
- **Safari**: Full support (macOS 11.3+)
- **Mobile**: Limited support (no getUserMedia on iOS Safari)

## Troubleshooting

### Camera not working
- Ensure you're accessing via HTTP (not file://)
- Check browser permissions for camera access
- Try a different browser if issues persist

### Low performance
- Reduce threshold to generate fewer events
- Check if WebGL is enabled in your browser
- Close other GPU-intensive applications

### Build errors
- Ensure Rust toolchain is up to date: `rustup update`
- Update wasm-pack: `wasm-pack --version`
- Clear build cache: `rm -rf pkg target`

## License

This project uses the ESIM algorithm from evlib. See the main evlib repository for license information.
