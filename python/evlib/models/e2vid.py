"""E2VID model implementation using PyTorch.

Based on the official RPG E2VID implementation from:
https://github.com/uzh-rpg/rpg_e2vid
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
import os
import urllib.request
from pathlib import Path

from .base import BaseModel
from .config import ModelConfig


class ConvLayer(nn.Module):
    """Convolutional layer with optional normalization and activation."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation="relu", norm=None
    ):
        super().__init__()

        bias = False if norm == "BN" else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, "relu")
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    """Upsampling layer using bilinear interpolation + conv (avoids checkerboard artifacts)."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation="relu", norm=None
    ):
        super().__init__()

        bias = False if norm == "BN" else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, "relu")
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, in_channels, out_channels, norm=None):
        super().__init__()

        self.conv1 = ConvLayer(in_channels, out_channels, 3, padding=1, norm=norm)
        self.conv2 = ConvLayer(out_channels, out_channels, 3, padding=1, activation=None, norm=norm)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return F.relu(out)


class UNet(nn.Module):
    """E2VID UNet architecture based on official RPG implementation."""

    def __init__(
        self,
        num_input_channels=5,
        num_output_channels=1,
        skip_type="sum",
        activation="sigmoid",
        num_encoders=4,
        base_num_channels=32,
        num_residual_blocks=2,
        norm=None,
        use_upsample_conv=True,
    ):
        super().__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.activation = activation
        self.norm = norm
        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks

        # Choose upsampling method
        if use_upsample_conv:
            self.UpsampleLayer = UpsampleConvLayer
        else:
            self.UpsampleLayer = nn.ConvTranspose2d

        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        # Build network
        self._build_head()
        self._build_encoders()
        self._build_residual_blocks()
        self._build_decoders()
        self._build_prediction_layer()

        # Get activation function
        self.final_activation = getattr(torch, self.activation, "sigmoid")

    def _build_head(self):
        """Build the initial conv layer."""
        self.head = ConvLayer(
            self.num_input_channels,
            self.base_num_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            norm=self.norm,
        )

    def _build_encoders(self):
        """Build encoder layers."""
        self.encoders = nn.ModuleList()
        for i in range(self.num_encoders):
            input_size = self.base_num_channels * pow(2, i)
            output_size = self.base_num_channels * pow(2, i + 1)
            self.encoders.append(
                ConvLayer(input_size, output_size, kernel_size=5, stride=2, padding=2, norm=self.norm)
            )

    def _build_residual_blocks(self):
        """Build residual blocks for bottleneck."""
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def _build_decoders(self):
        """Build decoder layers."""
        self.decoders = nn.ModuleList()
        for i in range(self.num_encoders):
            input_size = self.base_num_channels * pow(2, self.num_encoders - i)
            output_size = input_size // 2

            # Account for skip connections
            if self.skip_type == "concat":
                input_size = input_size * 2

            self.decoders.append(
                self.UpsampleLayer(input_size, output_size, kernel_size=5, padding=2, norm=self.norm)
            )

    def _build_prediction_layer(self):
        """Build final prediction layer."""
        input_size = self.base_num_channels
        if self.skip_type == "concat":
            input_size = input_size * 2

        self.pred = ConvLayer(input_size, self.num_output_channels, 1, activation=None, norm=self.norm)

    def _apply_skip_connection(self, x1, x2):
        """Apply skip connection with size matching."""
        if self.skip_type == "sum":
            # Ensure tensors have the same spatial dimensions
            if x1.shape[-2:] != x2.shape[-2:]:
                # Resize x1 to match x2's spatial dimensions
                x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            return x1 + x2
        else:  # concat
            # Ensure tensors have the same spatial dimensions
            if x1.shape[-2:] != x2.shape[-2:]:
                # Resize x1 to match x2's spatial dimensions
                x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat([x1, x2], dim=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (N, num_input_channels, H, W)

        Returns:
            Output tensor of shape (N, num_output_channels, H, W)
        """
        # Head
        x = self.head(x)
        head = x

        # Encoder
        blocks = []
        for encoder in self.encoders:
            x = encoder(x)
            blocks.append(x)

        # Residual blocks (bottleneck)
        for resblock in self.resblocks:
            x = resblock(x)

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            skip_idx = self.num_encoders - i - 1
            x = decoder(self._apply_skip_connection(x, blocks[skip_idx]))

        # Final prediction with skip to head
        output = self.final_activation(self.pred(self._apply_skip_connection(x, head)))

        return output


class E2VID(BaseModel):
    """E2VID: Event to Video reconstruction model.

    A PyTorch implementation based on the official RPG E2VID model:
    "High Speed and High Dynamic Range Video with an Event Camera"
    by Rebecq et al., CVPR 2019.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        pretrained: bool = False,
        skip_type: str = "sum",
        num_encoders: int = 4,
        num_residual_blocks: int = 2,
        norm: Optional[str] = None,
    ):
        """Initialize E2VID model.

        Args:
            config: Model configuration
            pretrained: Whether to load pretrained weights
            skip_type: Skip connection type ('sum' or 'concat')
            num_encoders: Number of encoder layers
            num_residual_blocks: Number of residual blocks in bottleneck
            norm: Normalization type ('BN', 'IN', or None)
        """
        super().__init__(config, pretrained)
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_type = skip_type
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.norm = norm
        self._build_model()

    def _build_model(self):
        """Build the E2VID model."""
        self._model = UNet(
            num_input_channels=self.config.num_bins,
            num_output_channels=1,
            skip_type=self.skip_type,
            activation="sigmoid",
            num_encoders=self.num_encoders,
            base_num_channels=self.config.base_channels,
            num_residual_blocks=self.num_residual_blocks,
            norm=self.norm,
            use_upsample_conv=True,  # Better quality, avoids checkerboard artifacts
        ).to(self._device)

    def _load_pretrained_weights(self):
        """Load pretrained weights."""
        # Placeholder for pretrained weight loading
        print("Warning: Pretrained weight loading not yet implemented.")
        print("Using randomly initialized weights.")

    def reconstruct(
        self,
        events: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct frame from events.

        Args:
            events: Event data as tuple (xs, ys, ts, ps) or structured array
            height: Output height
            width: Output width

        Returns:
            Reconstructed frame as numpy array of shape (height, width)
        """
        # Preprocess events
        xs, ys, ts, ps, height, width = self.preprocess_events(events, height, width)

        # Convert events to voxel grid
        voxel_grid = self.events_to_voxel_grid(xs, ys, ts, ps, height, width)

        # Convert to PyTorch tensor
        input_tensor = torch.from_numpy(voxel_grid).float().unsqueeze(0).to(self._device)

        # Run inference
        self._model.eval()
        with torch.no_grad():
            output = self._model(input_tensor)

        # Convert back to numpy
        frame = output.squeeze().cpu().numpy()

        return frame

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"E2VID(config={self.config}, pretrained={self.pretrained}, "
            f"skip_type='{self.skip_type}', encoders={self.num_encoders}, "
            f"residual_blocks={self.num_residual_blocks}, norm='{self.norm}', "
            f"device={self._device})"
        )
