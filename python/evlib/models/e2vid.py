"""E2VID model implementation using PyTorch.

Based on the official RPG E2VID implementation from:
https://github.com/uzh-rpg/rpg_e2vid
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
from pathlib import Path

from .base import BaseModel
from .config import ModelConfig


class ConvLayer(nn.Module):
    """Convolutional layer with optional normalization and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        bias = False if norm == "BN" else True
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

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
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        bias = False if norm == "BN" else True
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

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
        x_upsampled = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        out = self.conv2d(x_upsampled)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    """Transposed convolution layer (matches pretrained model architecture)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        bias = False if norm == "BN" else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=bias,
        )

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
        out = self.transposed_conv2d(x)

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
        self.conv2 = ConvLayer(
            out_channels, out_channels, 3, padding=1, activation=None, norm=norm
        )

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
            self.UpsampleLayer = TransposedConvLayer

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
                ConvLayer(
                    input_size,
                    output_size,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    norm=self.norm,
                )
            )

    def _build_residual_blocks(self):
        """Build residual blocks for bottleneck."""
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(
                ResidualBlock(
                    self.max_num_channels, self.max_num_channels, norm=self.norm
                )
            )

    def _build_decoders(self):
        """Build decoder layers."""
        decoder_input_sizes = list(
            reversed(
                [
                    self.base_num_channels * pow(2, i + 1)
                    for i in range(self.num_encoders)
                ]
            )
        )

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            # Apply skip connection logic to input size (matching reference implementation)
            decoder_input_size = (
                input_size if self.skip_type == "sum" else 2 * input_size
            )
            output_size = input_size // 2

            self.decoders.append(
                self.UpsampleLayer(
                    decoder_input_size,
                    output_size,
                    kernel_size=5,
                    padding=2,
                    norm=self.norm,
                )
            )

    def _build_prediction_layer(self):
        """Build final prediction layer."""
        # Apply skip connection logic to prediction layer input size (matching reference)
        input_size = (
            self.base_num_channels
            if self.skip_type == "sum"
            else 2 * self.base_num_channels
        )
        self.pred = ConvLayer(
            input_size, self.num_output_channels, 1, activation=None, norm=self.norm
        )

    def _apply_skip_connection(self, x1, x2):
        """Apply skip connection with size matching."""
        if self.skip_type == "sum":
            # Ensure tensors have the same spatial dimensions
            if x1.shape[-2:] != x2.shape[-2:]:
                # Resize x1 to match x2's spatial dimensions
                x1 = F.interpolate(
                    x1, size=x2.shape[-2:], mode="bilinear", align_corners=False
                )
            return x1 + x2
        else:  # concat
            # Ensure tensors have the same spatial dimensions
            if x1.shape[-2:] != x2.shape[-2:]:
                # Resize x1 to match x2's spatial dimensions
                x1 = F.interpolate(
                    x1, size=x2.shape[-2:], mode="bilinear", align_corners=False
                )
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

        # Decoder with skip connections (matching reference implementation)
        for i, decoder in enumerate(self.decoders):
            skip_idx = self.num_encoders - i - 1
            # Apply skip connection BEFORE decoder (as in reference)
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
        num_encoders: int = 4,  # Original RPG E2VID default
        num_residual_blocks: int = 2,  # Original RPG E2VID default
        norm: Optional[str] = None,  # Original RPG E2VID default (None, not BN)
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
        # Initialize config first
        self.config = config or ModelConfig()
        self.pretrained = pretrained

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_type = skip_type
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.norm = norm
        self._model = None

        # Build model, then load pretrained weights if requested
        self._build_model()
        if pretrained:
            self._load_pretrained_weights()

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

        # Look for weights in the models/weights directory
        weights_dir = Path(__file__).parent / "weights"
        weight_files = list(weights_dir.glob("*.pth*"))

        if not weight_files:
            print("Warning: No pretrained weights found in models/weights/")
            print("Using randomly initialized weights.")
            return

        # Use the first weight file found
        weight_file = weight_files[0]
        print(f"Loading pretrained weights from {weight_file.name}")

        try:
            checkpoint = torch.load(weight_file, map_location=self._device)

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Detect architecture from pretrained weights
            head_weight_key = None
            for key in state_dict.keys():
                if "head.conv2d.weight" in key or "head.weight" in key:
                    head_weight_key = key
                    break

            # Count number of encoders from state dict
            encoder_count = len(
                [
                    k
                    for k in state_dict.keys()
                    if "encoders." in k and "conv.conv2d.weight" in k
                ]
            )

            # Detect base channels and architecture
            architecture_changed = False
            if head_weight_key:
                pretrained_base_channels = state_dict[head_weight_key].shape[0]
                if pretrained_base_channels != self.config.base_channels:
                    print(
                        f"Adjusting base_channels: {self.config.base_channels} → {pretrained_base_channels}"
                    )
                    self.config.base_channels = pretrained_base_channels
                    architecture_changed = True

            if encoder_count > 0 and encoder_count != self.num_encoders:
                print(
                    f"Adjusting num_encoders: {self.num_encoders} → {encoder_count} (pretrained model is 'lite' variant)"
                )
                self.num_encoders = encoder_count
                architecture_changed = True

            # Check for normalization layers
            has_norm = any("norm_layer" in key for key in state_dict.keys())
            if has_norm and self.norm is None:
                print(
                    "Detected normalization layers, enabling batch normalization (pretrained model uses BN)"
                )
                self.norm = "BN"
                architecture_changed = True

            # Check if pretrained model uses transposed convolutions
            has_transposed_conv = any(
                "transposed_conv2d" in key for key in state_dict.keys()
            )
            if has_transposed_conv:
                print(
                    "Detected TransposedConv layers, adjusting architecture for pretrained compatibility"
                )
                # Rebuild with TransposedConvLayer for exact weight compatibility
                architecture_changed = True

            # Rebuild model if architecture changed
            if architecture_changed:
                # For pretrained models, use TransposedConvLayer to match weight structure exactly
                if has_transposed_conv:
                    self._model = UNet(
                        num_input_channels=self.config.num_bins,
                        num_output_channels=1,
                        skip_type=self.skip_type,
                        activation="sigmoid",
                        num_encoders=self.num_encoders,
                        base_num_channels=self.config.base_channels,
                        num_residual_blocks=self.num_residual_blocks,
                        norm=self.norm,
                        use_upsample_conv=False,  # Use TransposedConvLayer for pretrained compatibility
                    ).to(self._device)
                else:
                    self._build_model()

            # The pretrained model uses 'unetrecurrent.' prefix, but our model doesn't
            # We need to map the keys appropriately
            model_state_dict = {}

            for key, value in state_dict.items():
                # Remove 'unetrecurrent.' prefix if present
                new_key = key.replace("unetrecurrent.", "")

                # Map encoder keys (our model structure may differ)
                if new_key.startswith("encoders.") and ".recurrent_block." in new_key:
                    # Skip recurrent block weights for now as our UNet doesn't have them
                    continue
                elif new_key.startswith("encoders.") and ".conv." in new_key:
                    # Map encoder conv weights: encoders.0.conv.conv2d.weight -> encoders.0.conv2d.weight
                    new_key = new_key.replace(".conv.", ".")
                elif (
                    new_key.startswith("decoders.") and ".transposed_conv2d." in new_key
                ):
                    # Map decoder weights - our model uses UpsampleConvLayer with .conv2d
                    new_key = new_key.replace(".transposed_conv2d.", ".conv2d.")

                # Only include keys that match our model structure
                if any(
                    pattern in new_key
                    for pattern in [
                        "head.",
                        "encoders.",
                        "decoders.",
                        "resblocks.",
                        "pred.",
                    ]
                ):
                    model_state_dict[new_key] = value

            # Try to load compatible weights
            missing_keys, unexpected_keys = self._model.load_state_dict(
                model_state_dict, strict=False
            )

            loaded_keys = len(model_state_dict) - len(unexpected_keys)
            total_model_keys = len(self._model.state_dict())

            if loaded_keys > total_model_keys * 0.5:  # At least 50% compatibility
                print(
                    f"✓ Pretrained weights loaded successfully ({loaded_keys}/{total_model_keys} parameters)"
                )
            else:
                print(
                    f"⚠ Partial weight loading ({loaded_keys}/{total_model_keys} parameters)"
                )

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Using randomly initialized weights.")

    def events_to_voxel_grid(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        ps: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Convert events to voxel grid representation for E2VID.

        This uses a simple temporal binning approach that's appropriate for E2VID,
        rather than the more complex bilinear interpolation in evlib.representations.

        Args:
            xs: X coordinates of events
            ys: Y coordinates of events
            ts: Timestamps of events
            ps: Polarities of events
            height: Grid height
            width: Grid width

        Returns:
            Voxel grid as numpy array of shape (num_bins, height, width)
        """
        if len(ts) == 0:
            return np.zeros((self.config.num_bins, height, width), dtype=np.float32)

        # Normalise timestamps to [0, 1]
        t_min, t_max = ts.min(), ts.max()
        if t_max == t_min:
            t_normalised = np.zeros_like(ts)
        else:
            t_normalised = (ts - t_min) / (t_max - t_min)

        # Create voxel grid
        voxel_grid = np.zeros((self.config.num_bins, height, width), dtype=np.float32)

        # Compute temporal bin indices (simple binning, no interpolation)
        t_idx = (t_normalised * (self.config.num_bins - 1)).astype(np.int32)
        t_idx = np.clip(t_idx, 0, self.config.num_bins - 1)

        # Clip spatial coordinates
        xs_clipped = np.clip(xs.astype(np.int32), 0, width - 1)
        ys_clipped = np.clip(ys.astype(np.int32), 0, height - 1)

        # Accumulate polarities (simple accumulation, no interpolation)
        for i in range(len(xs)):
            voxel_grid[t_idx[i], ys_clipped[i], xs_clipped[i]] += ps[i]

        return voxel_grid

    def reconstruct(
        self,
        events: Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
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
        input_tensor = (
            torch.from_numpy(voxel_grid).float().unsqueeze(0).to(self._device)
        )

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
