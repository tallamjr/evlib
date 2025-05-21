#!/usr/bin/env python3
"""
ETAP Integration for evlib

This module provides integration between evlib and the ETAP (Event-based Tracking of Any Point) model.
It handles model loading, inference, and result conversion.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

# Add ETAP repository to Python path
ETAP_REPO_PATH = Path("/Users/tallam/github/tallamjr/clones/ETAP")
if ETAP_REPO_PATH.exists():
    sys.path.insert(0, str(ETAP_REPO_PATH / "src"))

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. ETAP functionality will be limited.")

try:
    # Import ETAP model and utilities (only if torch is available)
    if TORCH_AVAILABLE:
        from model.etap.model import Etap
        from representations.voxel_grid import VoxelGrid

        ETAP_AVAILABLE = True
    else:
        ETAP_AVAILABLE = False
except ImportError:
    ETAP_AVAILABLE = False
    if TORCH_AVAILABLE:
        warnings.warn("ETAP model not found. Please ensure ETAP repository is available.")

import evlib


class ETAPTracker:
    """
    High-level interface for ETAP point tracking using evlib.

    This class provides a convenient interface to:
    1. Load ETAP models
    2. Process event data
    3. Track arbitrary points
    4. Convert results to evlib format
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        window_len: int = 8,
        stride: int = 4,
        model_resolution: Tuple[int, int] = (512, 512),
        num_bins: int = 5,
    ):
        """
        Initialize ETAP tracker.

        Args:
            model_path: Path to ETAP model weights (.pth file)
            device: Device for inference ('cpu', 'cuda', 'mps', or 'auto')
            window_len: Temporal window length for tracking
            stride: Stride parameter for model
            model_resolution: Model input resolution (width, height)
            num_bins: Number of bins for voxel grid representation
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for ETAP tracking")

        if not ETAP_AVAILABLE:
            raise RuntimeError("ETAP model not available. Check repository path.")

        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Model parameters
        self.window_len = window_len
        self.stride = stride
        self.model_resolution = model_resolution
        self.num_bins = num_bins

        # Initialize model
        self.model = Etap(
            window_len=window_len,
            stride=stride,
            model_resolution=model_resolution,
            num_in_channels=num_bins,
        ).to(self.device)

        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Warning: Model path '{model_path}' not found. Using untrained model.")

        # Set to evaluation mode
        self.model.eval()

        # Initialize voxel grid converter
        self.voxel_grid = VoxelGrid(
            image_shape=(model_resolution[1], model_resolution[0]), num_bins=num_bins  # (height, width)
        )

        print(f"ETAP tracker initialized on device: {self.device}")

    def load_model(self, model_path: str) -> None:
        """Load ETAP model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded ETAP model from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model.")

    def events_to_tensor(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        ps: np.ndarray,
        resolution: Optional[Tuple[int, int]] = None,
    ):
        """
        Convert event data to tensor format for ETAP.

        Args:
            xs, ys: Event coordinates
            ts: Event timestamps
            ps: Event polarities
            resolution: Sensor resolution (width, height)

        Returns:
            Event tensor in format [B, T, C, H, W]
        """
        if resolution is None:
            resolution = self.model_resolution

        # Create events array in ETAP format: [y, x, t, p]
        events = np.column_stack([ys, xs, ts, ps])

        # Create voxel grid
        voxel_grid = self.voxel_grid(events)  # Shape: [num_bins, height, width]

        # Convert to torch tensor and add batch and time dimensions
        tensor = torch.from_numpy(voxel_grid).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [B=1, T=1, C, H, W]

        return tensor.to(self.device)

    def track_points(
        self,
        event_data: Union[Tuple[np.ndarray, ...], "torch.Tensor"],
        query_points: List[evlib.tracking.PyQueryPoint],
        resolution: Optional[Tuple[int, int]] = None,
        iters: int = 4,
    ) -> Dict[int, evlib.tracking.PyTrackResult]:
        """
        Track points through event data using ETAP.

        Args:
            event_data: Either (xs, ys, ts, ps) tuple or pre-processed tensor
            query_points: List of query points to track
            resolution: Sensor resolution (width, height)
            iters: Number of optimization iterations

        Returns:
            Dictionary mapping track IDs to tracking results
        """
        with torch.no_grad():
            # Prepare input tensor
            if isinstance(event_data, tuple):
                xs, ys, ts, ps = event_data
                video_tensor = self.events_to_tensor(xs, ys, ts, ps, resolution)
            else:
                video_tensor = event_data.to(self.device)

            # Prepare queries: [B, N, 3] where 3 = [frame_idx, x, y]
            queries = []
            for qp in query_points:
                # Scale coordinates to model resolution if needed
                x_scaled = (
                    qp.point.x
                    * self.model_resolution[0]
                    / (resolution[0] if resolution else self.model_resolution[0])
                )
                y_scaled = (
                    qp.point.y
                    * self.model_resolution[1]
                    / (resolution[1] if resolution else self.model_resolution[1])
                )
                queries.append([qp.frame_idx, x_scaled, y_scaled])

            queries = torch.tensor(queries, dtype=torch.float32).unsqueeze(0).to(self.device)  # [B=1, N, 3]

            # Run ETAP inference
            results = self.model(video_tensor, queries, iters=iters)

            # Extract predictions
            coords_predicted = results["coords_predicted"]  # [B, T, N, 2]
            vis_predicted = results["vis_predicted"]  # [B, T, N]

            # Convert back to evlib format
            track_results = {}
            B, T, N, _ = coords_predicted.shape

            for i in range(N):
                result = evlib.tracking.PyTrackResult()

                # Extract trajectory for this point
                coords = coords_predicted[0, :, i, :].cpu().numpy()  # [T, 2]
                visibility = vis_predicted[0, :, i].cpu().numpy()  # [T]

                # Scale coordinates back to original resolution
                if resolution:
                    coords[:, 0] *= resolution[0] / self.model_resolution[0]  # x
                    coords[:, 1] *= resolution[1] / self.model_resolution[1]  # y

                # Add each frame to result using the add_frame method
                for t in range(T):
                    point = evlib.tracking.PyPoint2D(coords[t, 0], coords[t, 1])
                    result.add_frame(t, point, visibility[t])

                track_results[i] = result

            return track_results


def create_etap_tracker(model_path: Optional[str] = None, device: str = "auto", **kwargs):
    """
    Convenience function to create an ETAP tracker.

    Args:
        model_path: Path to ETAP model weights
        device: Device for inference
        **kwargs: Additional arguments for ETAPTracker

    Returns:
        Initialized ETAP tracker or None if not available
    """
    if not TORCH_AVAILABLE or not ETAP_AVAILABLE:
        warnings.warn("ETAP tracker not available. Check PyTorch and ETAP installation.")
        return None

    return ETAPTracker(model_path=model_path, device=device, **kwargs)


def track_points_with_etap(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    query_points: List[evlib.tracking.PyQueryPoint],
    resolution: Tuple[int, int],
    model_path: Optional[str] = None,
    device: str = "auto",
    **kwargs,
):
    """
    One-shot function to track points using ETAP.

    Args:
        xs, ys, ts, ps: Event data
        query_points: Points to track
        resolution: Sensor resolution
        model_path: Path to ETAP model weights
        device: Device for inference
        **kwargs: Additional tracker arguments

    Returns:
        Tracking results or None if ETAP not available
    """
    tracker = create_etap_tracker(model_path=model_path, device=device, **kwargs)
    if tracker is None:
        return None

    return tracker.track_points((xs, ys, ts, ps), query_points, resolution)


def get_etap_status() -> Dict[str, bool]:
    """Get status of ETAP integration components."""
    return {
        "torch_available": TORCH_AVAILABLE,
        "etap_available": ETAP_AVAILABLE,
        "fully_functional": TORCH_AVAILABLE and ETAP_AVAILABLE,
    }


if __name__ == "__main__":
    # Test the integration
    print("Testing ETAP integration...")

    status = get_etap_status()
    print(f"PyTorch available: {status['torch_available']}")
    print(f"ETAP available: {status['etap_available']}")
    print(f"Fully functional: {status['fully_functional']}")

    if status["fully_functional"]:
        print("✅ ETAP integration fully available")
        try:
            tracker = create_etap_tracker()
            if tracker:
                print("✅ ETAP tracker created successfully")
            else:
                print("❌ Failed to create ETAP tracker")
        except Exception as e:
            print(f"❌ Error creating tracker: {e}")
    else:
        print("❌ ETAP integration not fully available")

    print("Integration test complete.")
