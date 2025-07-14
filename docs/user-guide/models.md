# Neural Networks

Learn how to use evlib's neural network capabilities for event-based reconstruction, classification, and other deep learning tasks.

## Overview

evlib provides optimized neural network implementations for event camera data processing:

- **E2VID**: Event-to-Video reconstruction
- **Model Loading**: PyTorch and ONNX support
- **Preprocessing**: Event representations optimized for neural networks
- **Performance**: GPU acceleration where available

## E2VID: Event-to-Video Reconstruction

E2VID reconstructs intensity images from event streams using a U-Net architecture.

### Basic Usage

```python
import evlib
import numpy as np

# Load events
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

# Create voxel grid for neural network input
voxel_grid_data, voxel_grid_shape = evlib.representations.events_to_smooth_voxel_grid(xs, ys, ts, ps, 5, (640, 480))

# Load E2VID model
model = evlib.processing.load_e2vid_model("models/e2vid_unet.pth")

# Reconstruct image
reconstructed_image = evlib.processing.e2vid_reconstruct(model, voxel_grid)

print(f"Reconstructed image shape: {reconstructed_image.shape}")
```

### Model Download

```python
# Download pre-trained E2VID model
model_path = evlib.processing.download_e2vid_model(
    variant="unet",  # Available: "unet"
    save_dir="models/"
)

print(f"Model downloaded to: {model_path}")
```

### Batch Processing

```python
def process_event_sequence(event_file, window_duration=0.1):
    """Process events in temporal windows"""

    # Load all events
    xs, ys, ts, ps = evlib.formats.load_events(event_file)

    # Load model once
    model = evlib.processing.load_e2vid_model("models/e2vid_unet.pth")

    # Process in time windows
    t_start = ts.min()
    t_end = ts.max()
    current_time = t_start

    reconstructed_frames = []

    while current_time < t_end:
        # Get events in current window
        mask = (ts >= current_time) & (ts < current_time + window_duration)

        if mask.sum() > 0:
            # Create voxel grid
            voxel_data, voxel_shape = evlib.representations.events_to_smooth_voxel_grid(
                xs[mask], ys[mask], ts[mask], ps[mask],
                640, 480, 5
            )

            # Reconstruct
            frame = evlib.processing.e2vid_reconstruct(model, voxel)
            reconstructed_frames.append(frame)

        current_time += window_duration

    return reconstructed_frames

# Process entire sequence
frames = process_event_sequence("data/slider_depth/events.txt", window_duration=0.05)
print(f"Reconstructed {len(frames)} frames")
```

## Model Loading and Formats

### PyTorch Models

```python
# Load PyTorch model directly
import torch

# Method 1: Use evlib wrapper
model = evlib.processing.load_e2vid_model("models/e2vid_unet.pth")

# Method 2: Direct PyTorch loading (advanced)
model_dict = torch.load("models/e2vid_unet.pth", map_location='cpu')
model = evlib.processing.E2VIDUNet()
model.load_state_dict(model_dict)
model.eval()
```

### ONNX Models

```python
# Convert PyTorch to ONNX for deployment
def convert_to_onnx(pytorch_model, onnx_path):
    """Convert PyTorch model to ONNX format"""

    # Create dummy input
    dummy_input = torch.randn(1, 5, 480, 640)

    # Export to ONNX
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['voxel_grid'],
        output_names=['reconstructed_image']
    )

# Convert model
model = evlib.processing.load_e2vid_model("models/e2vid_unet.pth")
convert_to_onnx(model, "models/e2vid_unet.onnx")

# Load ONNX model
onnx_model = evlib.processing.load_onnx_model("models/e2vid_unet.onnx")
```

## Custom Model Integration

### Implementing Custom Models

```python
import torch
import torch.nn as nn

class CustomEventNet(nn.Module):
    """Custom neural network for event processing"""

    def __init__(self, input_channels=5, output_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Use custom model
def process_with_custom_model(xs, ys, ts, ps):
    # Create input representation
    voxel_data, voxel_shape = evlib.representations.events_to_smooth_voxel_grid(
        xs, ys, ts, ps, 640, 480, 5
    )

    # Convert to tensor
    voxel_tensor = torch.from_numpy(voxel).float().unsqueeze(0)

    # Load custom model
    model = CustomEventNet()
    model.load_state_dict(torch.load("models/custom_model.pth"))
    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(voxel_tensor)

    return output.squeeze().numpy()
```

## Preprocessing for Neural Networks

### Input Normalization

```python
def normalize_voxel_grid(voxel_grid):
    """Normalize voxel grid for neural network input"""

    # Z-score normalization
    mean = voxel_grid.mean()
    std = voxel_grid.std()
    normalized = (voxel_grid - mean) / (std + 1e-8)

    return normalized

# Apply normalization
voxel_data, voxel_shape = evlib.representations.events_to_smooth_voxel_grid(xs, ys, ts, ps, 5, (640, 480))
voxel_norm = normalize_voxel_grid(voxel)
```

### Data Augmentation

```python
def augment_voxel_grid(voxel_grid, augment_params):
    """Apply augmentation to voxel grid"""

    # Random spatial crop
    if augment_params.get('crop', False):
        h, w = voxel_grid.shape[-2:]
        crop_h, crop_w = augment_params['crop_size']

        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)

        voxel_grid = voxel_grid[..., start_h:start_h+crop_h, start_w:start_w+crop_w]

    # Random flip
    if augment_params.get('flip', False) and np.random.random() > 0.5:
        voxel_grid = np.flip(voxel_grid, axis=-1)

    # Random rotation
    if augment_params.get('rotation', False):
        angle = np.random.uniform(-augment_params['rotation'], augment_params['rotation'])
        # Apply rotation using appropriate method
        pass

    return voxel_grid

# Apply augmentation
augment_params = {
    'crop': True,
    'crop_size': (416, 416),
    'flip': True,
    'rotation': 5.0  # degrees
}

voxel_augmented = augment_voxel_grid(voxel_grid, augment_params)
```

## Performance Optimization

### GPU Acceleration

```python
import torch

def setup_gpu_processing():
    """Setup GPU processing if available"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    return device

# Setup GPU
device = setup_gpu_processing()

# Move model to GPU
model = evlib.processing.load_e2vid_model("models/e2vid_unet.pth")
model = model.to(device)

# Process on GPU
voxel_tensor = torch.from_numpy(voxel_grid).float().unsqueeze(0).to(device)
with torch.no_grad():
    output = model(voxel_tensor)
    result = output.cpu().numpy()
```

### Batch Processing

```python
def batch_process_events(event_files, batch_size=4):
    """Process multiple event files in batches"""

    model = evlib.processing.load_e2vid_model("models/e2vid_unet.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    results = []

    for i in range(0, len(event_files), batch_size):
        batch_files = event_files[i:i+batch_size]
        batch_voxels = []

        # Prepare batch
        for file in batch_files:
            xs, ys, ts, ps = evlib.formats.load_events(file)
            voxel_data, voxel_shape = evlib.representations.events_to_smooth_voxel_grid(
                xs, ys, ts, ps, 640, 480, 5
            )
            voxel_norm = normalize_voxel_grid(voxel)
            batch_voxels.append(voxel_norm)

        # Stack into batch tensor
        batch_tensor = torch.from_numpy(np.stack(batch_voxels)).float().to(device)

        # Process batch
        with torch.no_grad():
            batch_output = model(batch_tensor)
            batch_results = batch_output.cpu().numpy()

        results.extend(batch_results)

    return results
```

## Model Evaluation

### Reconstruction Quality Metrics

```python
def evaluate_reconstruction(reconstructed, ground_truth):
    """Evaluate reconstruction quality"""

    # Mean Squared Error
    mse = np.mean((reconstructed - ground_truth) ** 2)

    # Peak Signal-to-Noise Ratio
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))

    # Structural Similarity Index
    from skimage.metrics import structural_similarity as ssim
    ssim_score = ssim(reconstructed, ground_truth, data_range=1.0)

    # Perceptual metrics (if available)
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex')
        lpips_score = lpips_fn(
            torch.from_numpy(reconstructed).float().unsqueeze(0).unsqueeze(0),
            torch.from_numpy(ground_truth).float().unsqueeze(0).unsqueeze(0)
        ).item()
    except ImportError:
        lpips_score = None

    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_score,
        'lpips': lpips_score
    }

# Evaluate model
metrics = evaluate_reconstruction(reconstructed_image, ground_truth_image)
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.3f}")
```

### Benchmarking

```python
import time

def benchmark_model(model, voxel_grids, device, num_runs=10):
    """Benchmark model performance"""

    model = model.to(device)
    model.eval()

    # Warmup
    dummy_input = torch.randn(1, 5, 480, 640).to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)

    # Benchmark
    times = []

    for voxel in voxel_grids[:num_runs]:
        voxel_tensor = torch.from_numpy(voxel).float().unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            output = model(voxel_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time

    print(f"Average inference time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    print(f"FPS: {fps:.1f}")

    return avg_time, std_time
```

## Available Models

### E2VID Variants

| Model | Input Size | Parameters | PSNR | Notes |
|-------|------------|------------|------|-------|
| E2VID UNet | 640×480×5 | 8.4M | 24.3 dB | General purpose |

### Model Downloads

```python
# Available models
AVAILABLE_MODELS = {
    'e2vid_unet': {
        'url': 'https://download.ifi.uzh.ch/rpg/E2VID/models/E2VID_lightweight.pth.tar',
        'size': '33.8 MB',
        'description': 'E2VID lightweight U-Net for event-to-video reconstruction'
    }
}

# Download model
def download_model(model_name, save_dir="models/"):
    """Download pre-trained model"""

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    import os
    import urllib.request

    model_info = AVAILABLE_MODELS[model_name]
    model_path = os.path.join(save_dir, f"{model_name}.pth")

    if not os.path.exists(model_path):
        print(f"Downloading {model_name} ({model_info['size']})...")
        urllib.request.urlretrieve(model_info['url'], model_path)
        print(f"Model saved to: {model_path}")

    return model_path
```

## Best Practices

### 1. Input Preparation

```python
# Always use smooth voxel grids for neural networks
voxel_data, voxel_shape = evlib.representations.events_to_smooth_voxel_grid(
    xs, ys, ts, ps, 640, 480, 5
)

# Normalize input
voxel_norm = (voxel - voxel.mean()) / (voxel.std() + 1e-8)
```

### 2. Memory Management

```python
# Clear GPU memory periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Process in chunks for large datasets
def process_large_dataset(xs, ys, ts, ps, chunk_duration=1.0):
    current_time = ts.min()
    max_time = ts.max()

    while current_time < max_time:
        # Process chunk
        mask = (ts >= current_time) & (ts < current_time + chunk_duration)
        if mask.sum() > 0:
            # Process events in this chunk
            yield xs[mask], ys[mask], ts[mask], ps[mask]

        current_time += chunk_duration
```

### 3. Error Handling

```python
def safe_model_inference(model, voxel_grid):
    """Safe model inference with error handling"""

    try:
        # Validate input
        if voxel_grid.shape != (5, 480, 640):
            raise ValueError(f"Invalid input shape: {voxel_grid.shape}")

        # Convert to tensor
        voxel_tensor = torch.from_numpy(voxel_grid).float().unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = model(voxel_tensor)

        return output.squeeze().numpy()

    except Exception as e:
        print(f"Model inference failed: {e}")
        return None
```

## Next Steps

- [Visualization Guide](visualization.md): Display reconstruction results
- [API Reference](../api/processing.md): Detailed neural network functions
- [Examples](../examples/notebooks.md): Neural network examples
