"""RVT: Recurrent Vision Transformers for Object Detection with Event Cameras.

This module implements the complete RVT model for event-based object detection,
combining the recurrent vision transformer backbone with YOLOX detection head.

Based on the CVPR 2023 paper "Recurrent Vision Transformers for Object Detection with Event Cameras"
by Mathias Gehrig and Davide Scaramuzza.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional, Dict, List, Any
import os
from pathlib import Path

from .base import BaseModel
from .config import ModelConfig
from .rvt_backbone import RVTBackbone, RVTConfig, RVTStateManager
from .yolox_fpn import PAFPN
from .yolox_head import YOLOXHead

# Note: Using evlib's native histogram functions instead of local implementations


class RVTModelConfig(ModelConfig):
    """Configuration for RVT model, extending base ModelConfig."""

    def __init__(self, **kwargs):
        # Extract RVT-specific parameters
        self.model_variant = kwargs.pop("model_variant", "tiny")
        self.temporal_bins = kwargs.pop("temporal_bins", 10)
        self.num_classes = kwargs.pop("num_classes", 2)

        # Detection parameters
        self.confidence_threshold = kwargs.pop("confidence_threshold", 0.1)
        self.nms_threshold = kwargs.pop("nms_threshold", 0.45)
        self.max_detections = kwargs.pop("max_detections", 300)

        # Input resolution
        self.input_height = kwargs.pop("input_height", 480)
        self.input_width = kwargs.pop("input_width", 640)

        # FPN parameters
        self.fpn_depth_multiplier = kwargs.pop("fpn_depth_multiplier", 0.33)

        # Training parameters
        self.use_l1_loss = kwargs.pop("use_l1_loss", False)
        self.focal_loss_alpha = kwargs.pop("focal_loss_alpha", 0.25)
        self.focal_loss_gamma = kwargs.pop("focal_loss_gamma", 1.5)

        # Initialize base class with remaining kwargs
        super().__init__(**kwargs)

    @classmethod
    def tiny(cls) -> "RVTModelConfig":
        """Configuration for RVT-Tiny model."""
        return cls(
            model_variant="tiny",
            base_channels=32,
            fpn_depth_multiplier=0.33,
        )

    @classmethod
    def small(cls) -> "RVTModelConfig":
        """Configuration for RVT-Small model."""
        return cls(
            model_variant="small",
            base_channels=64,
            fpn_depth_multiplier=0.50,
        )

    @classmethod
    def base(cls) -> "RVTModelConfig":
        """Configuration for RVT-Base model."""
        return cls(
            model_variant="base",
            base_channels=96,
            fpn_depth_multiplier=0.67,
        )


class RVT(BaseModel, nn.Module):
    """RVT: Recurrent Vision Transformers for Object Detection with Event Cameras.

    A PyTorch implementation of RVT for event-based object detection.
    Combines a recurrent vision transformer backbone with YOLOX detection head.

    Args:
        config: Model configuration
        pretrained: Whether to load pretrained weights
        variant: Model variant ("tiny", "small", "base")
        num_classes: Number of detection classes
    """

    def __init__(
        self,
        config: Optional[RVTModelConfig] = None,
        pretrained: bool = False,
        variant: str = "tiny",
        num_classes: int = 2,
    ):
        """Initialize RVT model."""
        # Initialize config
        if config is None:
            if variant == "tiny":
                config = RVTModelConfig.tiny()
            elif variant == "small":
                config = RVTModelConfig.small()
            elif variant == "base":
                config = RVTModelConfig.base()
            else:
                raise ValueError(f"Unknown variant: {variant}")

        config.num_classes = num_classes
        config.model_variant = variant

        # Initialize base classes
        BaseModel.__init__(self, config, pretrained=False)  # We'll load weights manually
        nn.Module.__init__(self)

        self.variant = variant
        self.num_classes = num_classes
        self.temporal_bins = config.temporal_bins

        # Device management
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model components
        self._build_model()

        # State management
        self.state_manager = RVTStateManager(self.backbone)
        self._current_worker_id = 0

        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights()

    def _build_model(self):
        """Build the RVT model components."""
        # Create RVT backbone configuration
        rvt_config = RVTConfig(
            input_channels=2 * self.temporal_bins,  # 20 for 10 bins * 2 polarities
            embed_dim=self.config.base_channels,
        )

        if self.variant == "tiny":
            rvt_config = RVTConfig.tiny()
        elif self.variant == "small":
            rvt_config = RVTConfig.small()
        elif self.variant == "base":
            rvt_config = RVTConfig.base()

        rvt_config.input_channels = 2 * self.temporal_bins

        # Build components
        self.backbone = RVTBackbone(rvt_config).to(self._device)

        # FPN input channels from backbone stages 2, 3, 4
        fpn_in_channels = self.backbone.get_stage_dims((2, 3, 4))
        self.fpn = PAFPN(
            in_channels=fpn_in_channels,
            depth_multiplier=self.config.fpn_depth_multiplier,
        ).to(self._device)

        # Detection head - use FPN output channels
        fpn_strides = self.backbone.get_strides((2, 3, 4))
        # For tiny model, FPN outputs are same as backbone dims: (64, 128, 256)
        # But YOLOX head expects uniform channels, so we'll use the FPN dims directly
        self.head = YOLOXHead(
            num_classes=self.num_classes,
            in_channels=fpn_in_channels,  # Use actual FPN output channels
            strides=fpn_strides,
        ).to(self._device)

    def _load_pretrained_weights(self):
        """Load pretrained weights from checkpoint."""
        # Look for weights in the models/weights directory
        weights_dir = Path(__file__).parent / "weights"

        # Look for RVT checkpoint files
        checkpoint_patterns = [
            f"rvt-{self.variant[0]}.ckpt",  # rvt-t.ckpt, rvt-s.ckpt, rvt-b.ckpt
            f"rvt_{self.variant}.ckpt",
            "rvt.ckpt",
        ]

        checkpoint_file = None
        for pattern in checkpoint_patterns:
            candidate = weights_dir / pattern
            if candidate.exists():
                checkpoint_file = candidate
                break

        if checkpoint_file is None:
            print(f"Warning: No pretrained weights found for RVT-{self.variant}")
            print(f"Looked in: {weights_dir}")
            print("Using randomly initialized weights.")
            return

        print(f"Loading pretrained weights from {checkpoint_file.name}")

        try:
            # Load PyTorch Lightning checkpoint (disable weights_only for compatibility)
            checkpoint = torch.load(checkpoint_file, map_location=self._device, weights_only=False)

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Convert PyTorch Lightning state dict to our format
            model_state_dict = {}

            # Create mapping from checkpoint naming to our naming
            converted_keys = 0

            for key, value in state_dict.items():
                # Convert checkpoint key to our model key
                new_key = self._convert_checkpoint_key(key)
                if new_key:
                    model_state_dict[new_key] = value
                    converted_keys += 1

            print(f"✓ Converted {converted_keys}/{len(state_dict)} checkpoint keys")

            # Try to load the converted state dict
            missing_keys, unexpected_keys = self.load_state_dict(model_state_dict, strict=False)

            loaded_params = len(model_state_dict) - len(missing_keys)
            total_params = len(self.state_dict())

            print(f"✓ Loaded {loaded_params}/{total_params} parameters from pretrained checkpoint")

            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")

            return  # Skip the old loading logic

            for old_key, value in state_dict.items():
                # Remove 'mdl.' prefix from PyTorch Lightning
                if key.startswith("mdl."):
                    new_key = key[4:]  # Remove 'mdl.'
                else:
                    new_key = key

                # Map backbone keys
                if new_key.startswith("backbone."):
                    new_key = new_key.replace("backbone.", "backbone.")
                # Map FPN keys
                elif new_key.startswith("fpn."):
                    new_key = new_key.replace("fpn.", "fpn.")
                # Map head keys
                elif new_key.startswith("yolox_head."):
                    new_key = new_key.replace("yolox_head.", "head.")

                model_state_dict[new_key] = value

            # Load weights with flexible matching
            missing_keys, unexpected_keys = self.load_state_dict(model_state_dict, strict=False)

            loaded_keys = len(model_state_dict) - len(unexpected_keys)
            total_keys = len(self.state_dict())

            print(f"✓ Loaded {loaded_keys}/{total_keys} parameters from pretrained checkpoint")

            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    for key in missing_keys[:10]:
                        print(f"  - {key}")

            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Using randomly initialized weights.")

    def _convert_checkpoint_key(self, checkpoint_key: str) -> Optional[str]:
        """Convert checkpoint key naming to our model naming convention.

        Args:
            checkpoint_key: Key from the checkpoint

        Returns:
            Converted key for our model, or None if not mappable
        """
        # Remove 'mdl.' prefix
        if checkpoint_key.startswith("mdl."):
            key = checkpoint_key[4:]
        else:
            key = checkpoint_key

        # Skip non-model parameters
        if not key.startswith(("backbone.", "head.", "fpn.", "yolox_head.")):
            return None

        # Apply naming convention conversions step by step
        converted_key = key

        # Step 1: Basic module name conversions
        converted_key = converted_key.replace("att_blocks", "attention_blocks")
        converted_key = converted_key.replace("att_grid", "grid_attn")
        converted_key = converted_key.replace("att_window", "window_attn")

        # Step 2: Handle attention parameters that need attn. prefix
        # Map attention parameters (qkv, proj) to be under attn module
        if ".grid_attn." in converted_key:
            # These parameters need to be moved under attn submodule
            attn_params = ["qkv.", "proj."]
            for param in attn_params:
                if f".{param}" in converted_key:
                    converted_key = converted_key.replace(f".grid_attn.{param}", f".grid_attn.attn.{param}")

        # Step 3: MLP structure conversion
        converted_key = converted_key.replace(".mlp.net.0.0.", ".mlp.fc1.")
        converted_key = converted_key.replace(".mlp.net.2.", ".mlp.fc2.")

        # Step 4: Handle layer scaling parameters - these might not exist in our model
        # For now, skip them as they're optional optimization components
        if ".ls1.gamma" in converted_key or ".ls2.gamma" in converted_key:
            return None  # Skip layer scaling parameters

        # Step 5: YOLOX head mapping
        if key.startswith("yolox_head."):
            converted_key = converted_key.replace("yolox_head.", "head.")

            # Handle head convolution structure differences
            # Checkpoint: head.cls_convs.0.0.weight -> Model: head.cls_convs.0.0.conv.weight
            conv_patterns = [
                "cls_convs.",
                "reg_convs.",
                "obj_convs.",
                "cls_preds.",
                "reg_preds.",
                "obj_preds.",
            ]
            for pattern in conv_patterns:
                if f".{pattern}" in converted_key:
                    # Add .conv. before weight/bias if missing
                    if ".weight" in converted_key or ".bias" in converted_key:
                        if ".conv.weight" not in converted_key and ".conv.bias" not in converted_key:
                            converted_key = converted_key.replace(".weight", ".conv.weight")
                            converted_key = converted_key.replace(".bias", ".conv.bias")

        # Step 6: FPN parameter mapping
        if key.startswith("fpn."):
            # Handle lateral convolution naming: fpn.lateral_convs.0. -> fpn.lateral_conv0.
            import re

            lateral_match = re.search(r"lateral_convs\.(\d+)\.", converted_key)
            if lateral_match:
                idx = lateral_match.group(1)
                converted_key = converted_key.replace(f"lateral_convs.{idx}.", f"lateral_conv{idx}.")

            # Handle fpn convolution naming: fpn.fpn_convs.0. -> fpn.fpn_conv0.
            fpn_match = re.search(r"fpn_convs\.(\d+)\.", converted_key)
            if fpn_match:
                idx = fpn_match.group(1)
                converted_key = converted_key.replace(f"fpn_convs.{idx}.", f"fpn_conv{idx}.")

        return converted_key

    def preprocess_events_to_histogram(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Preprocess events into stacked histogram representation using evlib's native functions.

        Args:
            events: Event data as tuple (xs, ys, ts, ps) or structured array
            height: Output height
            width: Output width

        Returns:
            Tuple of (histogram_tensor, height, width)
        """
        # Preprocess events using parent method
        xs, ys, ts, ps, height, width = self.preprocess_events(events, height, width)

        # Create stacked histogram using evlib's native functionality
        import polars as pl
        import evlib

        # Convert to polars DataFrame format expected by evlib
        timestamp_us = ((ts * 1e6).cpu().numpy() if isinstance(ts, torch.Tensor) else (ts * 1e6)).astype(int)

        events_df = pl.DataFrame(
            {
                "x": xs.cpu().numpy() if isinstance(xs, torch.Tensor) else xs,
                "y": ys.cpu().numpy() if isinstance(ys, torch.Tensor) else ys,
                "timestamp": timestamp_us,  # Int64 microseconds first
                "polarity": ps.cpu().numpy() if isinstance(ps, torch.Tensor) else ps,
            }
        )

        # Convert timestamp to Duration type as expected by evlib
        events_df = events_df.with_columns([pl.col("timestamp").cast(pl.Duration(time_unit="us"))])

        # Use evlib's native stacked histogram function
        hist_df = evlib.create_stacked_histogram(
            events_df,
            height,
            width,
            nbins=self.temporal_bins,
            window_duration_ms=50.0,  # Standard RVT window duration
        )

        # Convert histogram DataFrame to tensor format using native .to_torch()
        # Expected format: [2*nbins, height, width] where 2 is for pos/neg polarities
        channels = 2 * self.temporal_bins  # 20 for 10 bins * 2 polarities
        histogram = torch.zeros((channels, height, width), dtype=torch.float32, device=self._device)

        if len(hist_df) > 0:
            # Use evlib's efficient tensor conversion via polars .to_torch()
            # Convert relevant columns to torch tensors directly
            channel_tensor = hist_df.select(pl.col("channel")).to_torch().squeeze()
            time_bin_tensor = hist_df.select(pl.col("time_bin")).to_torch().squeeze()
            y_tensor = hist_df.select(pl.col("y")).to_torch().squeeze()
            x_tensor = hist_df.select(pl.col("x")).to_torch().squeeze()
            count_tensor = hist_df.select(pl.col("count")).to_torch().squeeze().float()

            # Calculate channel indices: channel + time_bin * 2
            # This interleaves pos/neg for each time bin: [t0_neg, t0_pos, t1_neg, t1_pos, ...]
            channel_indices = channel_tensor + time_bin_tensor * 2

            # Use torch advanced indexing for efficient tensor filling
            # Create masks for valid indices
            valid_mask = (
                (channel_indices >= 0)
                & (channel_indices < channels)
                & (y_tensor >= 0)
                & (y_tensor < height)
                & (x_tensor >= 0)
                & (x_tensor < width)
            )

            if valid_mask.any():
                valid_channels = channel_indices[valid_mask]
                valid_y = y_tensor[valid_mask]
                valid_x = x_tensor[valid_mask]
                valid_counts = count_tensor[valid_mask]

                # Use scatter_add for efficient accumulation
                histogram.index_put_((valid_channels, valid_y, valid_x), valid_counts, accumulate=True)

        return histogram, height, width

    def forward_backbone(
        self,
        histogram: torch.Tensor,
        previous_states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[int, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through backbone.

        Args:
            histogram: Stacked histogram tensor (B, 2*bins, H, W)
            previous_states: Previous LSTM states
            token_mask: Optional token mask for masking

        Returns:
            Tuple of (backbone_features, new_states)
        """
        return self.backbone(histogram, previous_states, token_mask)

    def forward_detect(
        self,
        backbone_features: Dict[int, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass through FPN and detection head.

        Args:
            backbone_features: Features from backbone stages
            targets: Ground truth targets for training

        Returns:
            Tuple of (predictions, losses)
        """
        # Extract features for FPN (stages 2, 3, 4)
        fpn_inputs = {stage: backbone_features[stage] for stage in [2, 3, 4] if stage in backbone_features}

        # FPN forward pass
        fpn_features = self.fpn(fpn_inputs)

        # Detection head forward pass
        if self.training and targets is not None:
            predictions, losses = self.head(fpn_features, targets)
        else:
            predictions, losses = self.head(fpn_features)

        return predictions, losses

    def forward(
        self,
        histogram: torch.Tensor,
        previous_states: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        targets: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        retrieve_detections: bool = True,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Complete forward pass through RVT model.

        Args:
            histogram: Input stacked histogram (B, 2*bins, H, W)
            previous_states: Previous LSTM states
            targets: Ground truth targets for training
            token_mask: Optional token mask
            retrieve_detections: Whether to run detection head

        Returns:
            Tuple of (predictions, losses, new_states)
        """
        # Backbone forward pass
        backbone_features, new_states = self.forward_backbone(histogram, previous_states, token_mask)

        predictions, losses = None, None

        if retrieve_detections:
            # Detection forward pass
            predictions, losses = self.forward_detect(backbone_features, targets)

        return predictions, losses, new_states

    def detect(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        reset_states: bool = False,
    ) -> List[Dict[str, Any]]:
        """Detect objects in event data.

        Args:
            events: Event data as tuple (xs, ys, ts, ps) or structured array
            height: Image height
            width: Image width
            confidence_threshold: Detection confidence threshold
            nms_threshold: NMS threshold
            reset_states: Whether to reset LSTM states

        Returns:
            List of detections, each containing:
            - 'bbox': [x1, y1, x2, y2] bounding box
            - 'score': Detection confidence
            - 'class': Class ID (0=pedestrian, 1=cyclist)
            - 'class_name': Class name string
        """
        self.eval()

        if confidence_threshold is None:
            confidence_threshold = self.config.confidence_threshold
        if nms_threshold is None:
            nms_threshold = self.config.nms_threshold

        # Preprocess events to histogram
        histogram, height, width = self.preprocess_events_to_histogram(events, height, width)
        histogram = histogram.unsqueeze(0)  # Add batch dimension

        # Get previous states
        if reset_states:
            self.state_manager.reset_worker_states(self._current_worker_id)

        previous_states = self.state_manager.get_states(self._current_worker_id)

        # Forward pass
        with torch.no_grad():
            predictions, _, new_states = self.forward(histogram, previous_states, retrieve_detections=True)

        # Save new states
        self.state_manager.save_states(self._current_worker_id, new_states)

        # Post-process predictions
        detections = self._postprocess_predictions(predictions, confidence_threshold, nms_threshold)

        return detections

    def _postprocess_predictions(
        self,
        predictions: torch.Tensor,
        confidence_threshold: float,
        nms_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Post-process model predictions into detection format.

        Args:
            predictions: Model predictions
            confidence_threshold: Confidence threshold
            nms_threshold: NMS threshold

        Returns:
            List of detection dictionaries
        """
        # Import YOLOX postprocessing (assumes it's available)
        try:
            from .yolox_head import postprocess
        except ImportError:
            # Fallback simple postprocessing
            return self._simple_postprocess(predictions, confidence_threshold)

        # Use YOLOX postprocessing
        processed = postprocess(
            predictions,
            num_classes=self.num_classes,
            conf_thre=confidence_threshold,
            nms_thre=nms_threshold,
        )

        detections = []
        # Dynamic class names based on num_classes
        if self.num_classes == 2:
            class_names = ["pedestrian", "cyclist"]
        elif self.num_classes == 3:
            class_names = ["pedestrian", "cyclist", "vehicle"]  # Common 3-class setup
        else:
            class_names = [f"class_{i}" for i in range(self.num_classes)]

        if processed[0] is not None:
            for detection in processed[0]:
                det_numpy = detection.cpu().numpy()

                # Handle different output formats
                if len(det_numpy) == 6:
                    x1, y1, x2, y2, score, class_id = det_numpy
                elif len(det_numpy) == 7:
                    # YOLOX format: [x1, y1, x2, y2, obj_score, class_score, class_id]
                    x1, y1, x2, y2, obj_score, class_score, class_id = det_numpy
                    score = obj_score * class_score  # Combined confidence
                else:
                    print(f"Warning: Unexpected detection format with {len(det_numpy)} values: {det_numpy}")
                    continue

                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "score": float(score),
                        "class": int(class_id),
                        "class_name": (
                            class_names[int(class_id)]
                            if int(class_id) < len(class_names)
                            else f"class_{int(class_id)}"
                        ),
                    }
                )

        return detections

    def _simple_postprocess(
        self,
        predictions: torch.Tensor,
        confidence_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Simple fallback postprocessing."""
        # This is a simplified version - in practice, you'd need proper NMS etc.
        detections = []
        # Add your simple postprocessing logic here
        return detections

    def reconstruct(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct detections from events (implements BaseModel interface).

        Note: RVT is an object detection model, not reconstruction.
        This method returns detection results in a visualization format.

        Args:
            events: Event data
            height: Image height
            width: Image width

        Returns:
            Detection visualization as numpy array
        """
        detections = self.detect(events, height, width)

        # Create visualization image
        if height is None or width is None:
            xs, ys, ts, ps, height, width = self.preprocess_events(events, height, width)

        # Create blank image
        vis_image = np.zeros((height, width), dtype=np.uint8)

        # Draw detection boxes
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            # Draw bounding box
            vis_image[y1 : y1 + 2, x1:x2] = 255  # Top edge
            vis_image[y2 - 2 : y2, x1:x2] = 255  # Bottom edge
            vis_image[y1:y2, x1 : x1 + 2] = 255  # Left edge
            vis_image[y1:y2, x2 - 2 : x2] = 255  # Right edge

        return vis_image

    def reset_states(self):
        """Reset LSTM states."""
        self.state_manager.reset_worker_states(self._current_worker_id)

    def set_worker_id(self, worker_id: int):
        """Set current worker ID for state management."""
        self._current_worker_id = worker_id

    def to(self, device):
        """Move model to device."""
        self._device = device
        return super().to(device)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RVT(\n"
            f"  variant='{self.variant}'\n"
            f"  num_classes={self.num_classes}\n"
            f"  temporal_bins={self.temporal_bins}\n"
            f"  config={self.config}\n"
            f"  device={self._device}\n"
            f")"
        )
