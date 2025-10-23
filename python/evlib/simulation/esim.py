"""
ESIM (Event-based Simulator) Algorithm Implementation

This module implements the ESIM algorithm for converting intensity frames
to event camera data. It tracks log intensity changes and generates events
when changes exceed specified thresholds.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False
    torch = None

from .config import ESIMConfig


class ESIMSimulator:
    """
    ESIM (Event-based Simulator) for converting intensity frames to events.

    The simulator maintains internal state to track log intensity changes
    and generates events when changes exceed configured thresholds.

    Args:
        config: ESIMConfig instance with simulation parameters
    """

    def __init__(self, config: ESIMConfig):
        if not _torch_available:
            raise ImportError(
                "PyTorch is required for ESIMSimulator. Install with: pip install torch"
            )

        self.config = config
        self._device = self._setup_device()
        # Set dtype, but check device compatibility
        self._dtype = self._get_compatible_dtype()

        # Internal state tensors (initialized on first frame)
        self._log_last_intensity: Optional[torch.Tensor] = None
        self._intensity_buffer: Optional[torch.Tensor] = None
        self._last_event_time: Optional[torch.Tensor] = None
        self._initialized = False

        # Convert refractory period to seconds
        self._refractory_period_s = config.refractory_period_ms / 1000.0

    def _setup_device(self) -> torch.device:
        """Set up the computing device based on configuration."""
        if self.config.device == "auto":
            # Auto-select the best available device
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        # Validate device availability
        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        elif device.type == "mps":
            if not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                warnings.warn("MPS requested but not available, falling back to CPU")
                device = torch.device("cpu")

        return device

    def _get_compatible_dtype(self) -> torch.dtype:
        """Get dtype compatible with the selected device."""
        requested_dtype = (
            torch.float64 if self.config.dtype == "float64" else torch.float32
        )

        # MPS doesn't support float64
        if self._device.type == "mps" and requested_dtype == torch.float64:
            warnings.warn("MPS does not support float64, using float32 instead")
            return torch.float32

        return requested_dtype

    def reset(self) -> None:
        """Reset the simulator state."""
        self._log_last_intensity = None
        self._intensity_buffer = None
        self._last_event_time = None
        self._initialized = False

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single frame and generate events.

        Args:
            frame: Input frame as numpy array (H, W) for grayscale or (H, W, 3) for RGB
            timestamp: Timestamp for this frame in seconds

        Returns:
            Tuple of (x, y, t, polarity) arrays containing generated events
        """
        # Ensure frame is grayscale
        if len(frame.shape) == 3:
            # Convert RGB to grayscale
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

        height, width = frame.shape

        # Initialize state tensors on first frame
        if not self._initialized:
            self._initialize_state(height, width, frame, timestamp)
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Convert frame to tensor and normalize to [0, 1]
        intensity_tensor = (
            torch.from_numpy(frame).to(dtype=self._dtype, device=self._device) / 255.0
        )

        # Calculate log intensity change
        log_intensity = torch.log(
            torch.maximum(
                intensity_tensor,
                torch.tensor(self.config.log_floor, device=self._device),
            )
        )
        log_change = log_intensity - self._log_last_intensity
        self._intensity_buffer += log_change

        # Handle refractory period
        time_since_last_event = timestamp - self._last_event_time
        refractory_mask = time_since_last_event < self._refractory_period_s
        active_mask = ~refractory_mask

        # Generate events
        events = []

        # Positive events
        pos_mask = (
            self._intensity_buffer >= self.config.positive_threshold
        ) & active_mask
        if torch.any(pos_mask):
            pos_events = self._generate_events(
                pos_mask, self.config.positive_threshold, timestamp, polarity=1
            )
            events.append(pos_events)

            # Update buffer and event times for positive events
            num_events = (
                self._intensity_buffer[pos_mask] / self.config.positive_threshold
            ).to(torch.int32)
            self._intensity_buffer[pos_mask] -= (
                num_events * self.config.positive_threshold
            )
            self._last_event_time[pos_mask] = timestamp

        # Negative events
        neg_mask = (
            self._intensity_buffer <= -self.config.negative_threshold
        ) & active_mask
        if torch.any(neg_mask):
            neg_events = self._generate_events(
                neg_mask, self.config.negative_threshold, timestamp, polarity=-1
            )
            events.append(neg_events)

            # Update buffer and event times for negative events
            num_events = (
                self._intensity_buffer[neg_mask] / -self.config.negative_threshold
            ).to(torch.int32)
            self._intensity_buffer[neg_mask] += (
                num_events * self.config.negative_threshold
            )
            self._last_event_time[neg_mask] = timestamp

        # Update last intensity
        self._log_last_intensity = log_intensity

        # Combine and return events
        if events:
            all_events = torch.cat(events, dim=1)
            # Sort by timestamp
            sorted_indices = torch.argsort(all_events[2])
            all_events = all_events[:, sorted_indices]

            # Convert to numpy and return
            events_cpu = all_events.cpu().numpy()
            return (
                events_cpu[0].astype(np.int64),  # x
                events_cpu[1].astype(np.int64),  # y
                events_cpu[2].astype(np.float64),  # t
                events_cpu[3].astype(np.int64),  # polarity
            )
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def _initialize_state(
        self, height: int, width: int, frame: np.ndarray, timestamp: float
    ) -> None:
        """Initialize internal state tensors."""
        # Convert frame to tensor and normalize
        intensity_tensor = (
            torch.from_numpy(frame).to(dtype=self._dtype, device=self._device) / 255.0
        )

        # Initialize state tensors
        self._log_last_intensity = torch.log(
            torch.maximum(
                intensity_tensor,
                torch.tensor(self.config.log_floor, device=self._device),
            )
        )
        self._intensity_buffer = torch.zeros(
            (height, width), dtype=self._dtype, device=self._device
        )
        self._last_event_time = torch.zeros(
            (height, width), dtype=self._dtype, device=self._device
        )

        self._initialized = True

    def _generate_events(
        self, mask: torch.Tensor, threshold: float, timestamp: float, polarity: int
    ) -> torch.Tensor:
        """
        Generate events for pixels that exceed threshold.

        Args:
            mask: Boolean mask indicating which pixels have events
            threshold: Threshold value used for event generation
            timestamp: Current timestamp
            polarity: Event polarity (1 for positive, -1 for negative)

        Returns:
            Tensor of shape (4, N) containing [x, y, t, p] for N events
        """
        # Calculate number of events per pixel
        if polarity == 1:
            num_events = (self._intensity_buffer[mask] / threshold).to(torch.int32)
        else:
            num_events = (self._intensity_buffer[mask] / -threshold).to(torch.int32)

        # Get coordinates of active pixels
        y_coords, x_coords = torch.where(mask)

        # Repeat coordinates for multiple events per pixel
        x_rep = torch.repeat_interleave(x_coords, num_events)
        y_rep = torch.repeat_interleave(y_coords, num_events)

        # Create timestamp and polarity arrays
        total_events = x_rep.size(0)
        if total_events == 0:
            # Return empty tensor with correct shape
            return torch.zeros((4, 0), dtype=self._dtype, device=self._device)

        t_rep = torch.full(
            (total_events,), timestamp, dtype=self._dtype, device=self._device
        )
        p_rep = torch.full(
            (total_events,), polarity, dtype=self._dtype, device=self._device
        )

        return torch.stack([x_rep.to(self._dtype), y_rep.to(self._dtype), t_rep, p_rep])

    def get_state_info(self) -> dict:
        """Get information about the current simulator state."""
        if not self._initialized:
            return {"initialized": False}

        return {
            "initialized": True,
            "device": str(self._device),
            "dtype": str(self._dtype),
            "shape": self._log_last_intensity.shape,
            "buffer_stats": {
                "min": self._intensity_buffer.min().item(),
                "max": self._intensity_buffer.max().item(),
                "mean": self._intensity_buffer.mean().item(),
                "std": self._intensity_buffer.std().item(),
            },
        }

    @property
    def device(self) -> torch.device:
        """Get the computing device being used."""
        return self._device

    @property
    def is_initialized(self) -> bool:
        """Check if the simulator has been initialized."""
        return self._initialized
