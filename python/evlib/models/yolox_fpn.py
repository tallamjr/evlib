"""YOLOX PANet Feature Pyramid Network (FPN) implementation.

This module implements the PANet-style Feature Pyramid Network used in YOLOX,
which combines top-down and bottom-up pathways for multi-scale feature fusion.

The FPN takes multi-scale backbone features and produces enhanced feature maps
for object detection at different scales.

Based on:
- YOLOX paper: "YOLOX: Exceeding YOLO Series in 2021"
- PANet paper: "Path Aggregation Network for Instance Segmentation"
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .yolox_blocks import BaseConv, CSPLayer, DWConv


class PAFPN(nn.Module):
    """PANet-style Feature Pyramid Network for YOLOX.

    This FPN implementation:
    1. Takes multi-scale backbone features (stages 2, 3, 4)
    2. Applies top-down pathway with lateral connections
    3. Applies bottom-up pathway for feature enhancement
    4. Outputs enhanced features for detection head

    The network follows the PANet design with CSP blocks for efficiency.
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (256, 512, 1024),
        depth_multiplier: float = 0.33,
        width_multiplier: float = 1.0,
        depthwise: bool = False,
        act: str = "silu",
    ):
        """Initialize PAFPN.

        Args:
            in_channels: Input channel dimensions for stages (2, 3, 4)
            depth_multiplier: Multiplier for number of blocks in CSP layers
            width_multiplier: Multiplier for channel dimensions (not used in tiny)
            depthwise: Whether to use depthwise separable convolutions
            act: Activation function name
        """
        super().__init__()

        # Store configuration
        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        self.depthwise = depthwise

        # Use depthwise conv if requested
        Conv = DWConv if depthwise else BaseConv

        # Calculate number of blocks in CSP layers
        csp_blocks = max(round(3 * depth_multiplier), 1)

        # Top-down pathway
        # Reduce channels from stage 4 (highest resolution input)
        self.lateral_conv0 = BaseConv(
            in_channels[2], in_channels[1], kernel_size=1, stride=1, act=act
        )

        # CSP layer after concatenation with stage 3
        self.C3_p4 = CSPLayer(
            in_channels=2 * in_channels[1],
            out_channels=in_channels[1],
            n=csp_blocks,
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

        # Reduce channels for stage 2 connection
        self.reduce_conv1 = BaseConv(
            in_channels[1], in_channels[0], kernel_size=1, stride=1, act=act
        )

        # CSP layer after concatenation with stage 2
        self.C3_p3 = CSPLayer(
            in_channels=2 * in_channels[0],
            out_channels=in_channels[0],
            n=csp_blocks,
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

        # Bottom-up pathway
        # Downsample from P3 to N3 level
        self.bu_conv2 = Conv(
            in_channels[0], in_channels[0], kernel_size=3, stride=2, act=act
        )

        # CSP layer for N3 (stage 3 level in bottom-up path)
        self.C3_n3 = CSPLayer(
            in_channels=2 * in_channels[0],
            out_channels=in_channels[1],
            n=csp_blocks,
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

        # Downsample from N3 to N4 level
        self.bu_conv1 = Conv(
            in_channels[1], in_channels[1], kernel_size=3, stride=2, act=act
        )

        # CSP layer for N4 (stage 4 level in bottom-up path)
        self.C3_n4 = CSPLayer(
            in_channels=2 * in_channels[1],
            out_channels=in_channels[2],
            n=csp_blocks,
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

        # Upsampling function
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, backbone_features: Dict[int, Tensor]) -> List[Tensor]:
        """Forward pass through PAFPN.

        Args:
            backbone_features: Dictionary mapping stage numbers to feature tensors
                              Expected keys: {2, 3, 4} for stages 2, 3, 4

        Returns:
            List of enhanced feature maps [P3_out, P4_out, P5_out] for detection
            - P3_out: Features at 1/8 resolution (from stage 2)
            - P4_out: Features at 1/16 resolution (from stage 3)
            - P5_out: Features at 1/32 resolution (from stage 4)
        """
        # Extract backbone features
        # Note: We expect 1-indexed stage numbers from RVT backbone
        [x2, x1, x0] = [backbone_features[i] for i in [2, 3, 4]]

        # Top-down pathway
        # Start from highest level (stage 4, smallest spatial resolution)
        fpn_out0 = self.lateral_conv0(x0)  # Reduce channels: 1024->512
        f_out0 = self.upsample(fpn_out0)  # Upsample: /32 -> /16

        # Match spatial dimensions for concatenation
        if f_out0.shape[2:] != x1.shape[2:]:
            f_out0 = torch.nn.functional.interpolate(
                f_out0, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        f_out0 = torch.cat([f_out0, x1], dim=1)  # Concat with stage 3: 512+512->1024
        f_out0 = self.C3_p4(f_out0)  # CSP layer: 1024->512

        # Continue top-down to stage 2 level
        fpn_out1 = self.reduce_conv1(f_out0)  # Reduce channels: 512->256
        f_out1 = self.upsample(fpn_out1)  # Upsample: /16 -> /8

        # Match spatial dimensions for concatenation
        if f_out1.shape[2:] != x2.shape[2:]:
            f_out1 = torch.nn.functional.interpolate(
                f_out1, size=x2.shape[2:], mode="bilinear", align_corners=False
            )

        f_out1 = torch.cat([f_out1, x2], dim=1)  # Concat with stage 2: 256+256->512
        pan_out2 = self.C3_p3(f_out1)  # CSP layer: 512->256

        # Bottom-up pathway
        # Build enhanced features by adding bottom-up information
        p_out1 = self.bu_conv2(pan_out2)  # Downsample: /8 -> /16, 256->256

        # Match spatial dimensions for concatenation
        if p_out1.shape[2:] != fpn_out1.shape[2:]:
            p_out1 = torch.nn.functional.interpolate(
                p_out1, size=fpn_out1.shape[2:], mode="bilinear", align_corners=False
            )

        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)  # Concat: 256+256->512
        pan_out1 = self.C3_n3(p_out1)  # CSP layer: 512->512

        # Continue bottom-up to highest level
        p_out0 = self.bu_conv1(pan_out1)  # Downsample: /16 -> /32, 512->512

        # Match spatial dimensions for concatenation
        if p_out0.shape[2:] != fpn_out0.shape[2:]:
            p_out0 = torch.nn.functional.interpolate(
                p_out0, size=fpn_out0.shape[2:], mode="bilinear", align_corners=False
            )

        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)  # Concat: 512+512->1024
        pan_out0 = self.C3_n4(p_out0)  # CSP layer: 1024->1024

        # Return enhanced features for detection head
        # Order: [P3, P4, P5] corresponding to [/8, /16, /32] resolutions
        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs

    def get_output_channels(self) -> Tuple[int, int, int]:
        """Get output channel dimensions for each scale.

        Returns:
            Tuple of (P3_channels, P4_channels, P5_channels)
        """
        return self.in_channels


class YoloXFPN(PAFPN):
    """Alias for PAFPN to match YOLOX naming convention."""

    pass


def create_yolox_fpn(
    backbone_channels: Tuple[int, int, int],
    model_size: str = "tiny",
    depthwise: bool = False,
    act: str = "silu",
) -> PAFPN:
    """Create YOLOX FPN with predefined configurations.

    Args:
        backbone_channels: Channel dimensions from backbone stages (2, 3, 4)
        model_size: Model size variant ('tiny', 'small', 'medium', 'large', 'xlarge')
        depthwise: Whether to use depthwise separable convolutions
        act: Activation function name

    Returns:
        Configured PAFPN instance
    """
    # Depth multipliers for different model sizes
    depth_configs = {
        "tiny": 0.33,
        "small": 0.33,
        "medium": 0.67,
        "large": 1.0,
        "xlarge": 1.33,
    }

    # Width multipliers (for future extension)
    width_configs = {
        "tiny": 1.0,
        "small": 1.0,
        "medium": 1.0,
        "large": 1.0,
        "xlarge": 1.0,
    }

    if model_size not in depth_configs:
        raise ValueError(f"Unsupported model size: {model_size}")

    return PAFPN(
        in_channels=backbone_channels,
        depth_multiplier=depth_configs[model_size],
        width_multiplier=width_configs[model_size],
        depthwise=depthwise,
        act=act,
    )


# For backwards compatibility and ease of import
__all__ = ["PAFPN", "YoloXFPN", "create_yolox_fpn"]
