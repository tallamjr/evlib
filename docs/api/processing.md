# Processing API Reference

Neural network processing functions for event camera data.

## E2VID Neural Network

### Model Loading

```python
def load_e2vid_model(model_path: str) -> torch.nn.Module:
    """Load E2VID model from file

    Args:
        model_path: Path to model file (.pth)

    Returns:
        Loaded PyTorch model ready for inference
    """
    pass

def download_e2vid_model(variant: str = "unet", save_dir: str = "models/") -> str:
    """Download pre-trained E2VID model

    Args:
        variant: Model variant ("unet")
        save_dir: Directory to save model

    Returns:
        Path to downloaded model file
    """
    pass
```

### Inference

```python
def e2vid_reconstruct(model: torch.nn.Module, voxel_grid: np.ndarray) -> np.ndarray:
    """Reconstruct image from voxel grid using E2VID

    Args:
        model: Loaded E2VID model
        voxel_grid: Input voxel grid (5, H, W)

    Returns:
        Reconstructed image (H, W) in range [0, 1]
    """
    pass
```

## ONNX Runtime

### Model Loading

```python
def load_onnx_model(model_path: str) -> Any:
    """Load ONNX model for inference

    Args:
        model_path: Path to ONNX model file

    Returns:
        ONNX runtime inference session
    """
    pass

def onnx_inference(session: Any, input_data: np.ndarray) -> np.ndarray:
    """Run ONNX model inference

    Args:
        session: ONNX inference session
        input_data: Input voxel grid

    Returns:
        Model output
    """
    pass
```

## Model Architectures

### E2VID UNet

```python
class E2VIDUNet(torch.nn.Module):
    """E2VID U-Net architecture for event-to-video reconstruction"""

    def __init__(self, input_channels: int = 5, output_channels: int = 1):
        """Initialize E2VID U-Net

        Args:
            input_channels: Number of input channels (temporal bins)
            output_channels: Number of output channels (1 for grayscale)
        """
        super().__init__()
        # Model implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Reconstructed image (B, 1, H, W)
        """
        pass
```

## Preprocessing

### Input Normalization

```python
def normalize_voxel_for_network(voxel_grid: np.ndarray) -> np.ndarray:
    """Normalize voxel grid for neural network input

    Args:
        voxel_grid: Input voxel grid (T, H, W)

    Returns:
        Normalized voxel grid
    """
    mean = voxel_grid.mean()
    std = voxel_grid.std()
    return (voxel_grid - mean) / (std + 1e-8)

def prepare_batch_input(voxel_grids: List[np.ndarray]) -> torch.Tensor:
    """Prepare batch of voxel grids for network input

    Args:
        voxel_grids: List of voxel grids

    Returns:
        Batch tensor (B, T, H, W)
    """
    batch = np.stack(voxel_grids)
    return torch.from_numpy(batch).float()
```

## Postprocessing

### Output Processing

```python
def postprocess_reconstruction(output: np.ndarray,
                             target_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Postprocess neural network output

    Args:
        output: Network output
        target_range: Target value range

    Returns:
        Processed output in target range
    """
    # Clamp to [0, 1] range
    output = np.clip(output, 0, 1)

    # Scale to target range
    if target_range != (0, 1):
        min_val, max_val = target_range
        output = output * (max_val - min_val) + min_val

    return output
```

## Performance Utilities

### GPU Setup

```python
def setup_gpu_device() -> torch.device:
    """Setup GPU device for processing

    Returns:
        PyTorch device (cuda or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device

def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to specified device

    Args:
        tensor: Input tensor
        device: Target device

    Returns:
        Tensor on target device
    """
    return tensor.to(device)
```

### Benchmarking

```python
def benchmark_model(model: torch.nn.Module,
                   input_shape: Tuple[int, ...],
                   device: torch.device,
                   num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model performance

    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        device: Device to run on
        num_runs: Number of benchmark runs

    Returns:
        Performance metrics
    """
    import time

    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'fps': 1.0 / np.mean(times)
    }
```

## Error Handling

### Validation

```python
def validate_model_input(voxel_grid: np.ndarray,
                        expected_shape: Tuple[int, ...]) -> None:
    """Validate model input format

    Args:
        voxel_grid: Input voxel grid
        expected_shape: Expected input shape

    Raises:
        ValueError: If input format is invalid
    """
    if voxel_grid.shape != expected_shape:
        raise ValueError(f"Invalid input shape: {voxel_grid.shape}, expected {expected_shape}")

    if not np.isfinite(voxel_grid).all():
        raise ValueError("Input contains non-finite values")
```
