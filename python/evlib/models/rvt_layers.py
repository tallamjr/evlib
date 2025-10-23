"""RVT (Recurrent Vision Transformer) specific layers and components.

This module implements the core building blocks of the RVT architecture:
- MaxViT attention layers (window and grid attention)
- Depthwise separable Conv LSTM
- Downsampling and upsampling layers
- Utility functions for spatial partitioning

Based on the CVPR 2023 paper "Recurrent Vision Transformers for Object Detection with Event Cameras"
by Mathias Gehrig and Davide Scaramuzza.
"""

from typing import Optional, Tuple, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PartitionType(Enum):
    """Types of spatial partitioning for MaxViT attention."""

    WINDOW = "window"
    GRID = "grid"


def nhwc_to_nchw(x: Tensor) -> Tensor:
    """Convert tensor from NHWC to NCHW format."""
    return x.permute(0, 3, 1, 2).contiguous()


def nchw_to_nhwc(x: Tensor) -> Tensor:
    """Convert tensor from NCHW to NHWC format."""
    return x.permute(0, 2, 3, 1).contiguous()


def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, int, int]:
    """Partition input tensor into non-overlapping windows.

    Args:
        x: Input tensor of shape (B, H, W, C)
        window_size: Size of the window

    Returns:
        Tuple of (windowed tensor, num_windows_h, num_windows_w)
    """
    B, H, W, C = x.shape

    # Pad to make divisible by window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w

    num_windows_h = H // window_size
    num_windows_w = W // window_size

    # Partition into windows
    x = x.view(B, num_windows_h, window_size, num_windows_w, window_size, C)
    x = x.permute(
        0, 1, 3, 2, 4, 5
    ).contiguous()  # (B, num_windows_h, num_windows_w, window_size, window_size, C)
    x = x.view(-1, window_size * window_size, C)  # (B * num_windows, window_size^2, C)

    return x, num_windows_h, num_windows_w


def window_reverse(
    x: Tensor,
    window_size: int,
    num_windows_h: int,
    num_windows_w: int,
    original_h: int,
    original_w: int,
) -> Tensor:
    """Reverse window partitioning.

    Args:
        x: Windowed tensor of shape (B * num_windows, window_size^2, C)
        window_size: Size of the window
        num_windows_h: Number of windows in height dimension
        num_windows_w: Number of windows in width dimension
        original_h: Original height before padding
        original_w: Original width before padding

    Returns:
        Tensor of shape (B, original_h, original_w, C)
    """
    B = x.shape[0] // (num_windows_h * num_windows_w)
    C = x.shape[-1]

    # Reshape back to windows
    x = x.view(B, num_windows_h, num_windows_w, window_size, window_size, C)
    x = x.permute(
        0, 1, 3, 2, 4, 5
    ).contiguous()  # (B, num_windows_h, window_size, num_windows_w, window_size, C)
    x = x.view(B, num_windows_h * window_size, num_windows_w * window_size, C)

    # Remove padding if any
    x = x[:, :original_h, :original_w, :].contiguous()

    return x


def grid_partition(x: Tensor, grid_size: int) -> Tensor:
    """Partition input tensor into grid for grid attention.

    Args:
        x: Input tensor of shape (B, H, W, C)
        grid_size: Size of the grid

    Returns:
        Grid-partitioned tensor
    """
    B, H, W, C = x.shape

    # Pad to make divisible by grid_size
    pad_h = (grid_size - H % grid_size) % grid_size
    pad_w = (grid_size - W % grid_size) % grid_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w

    # Partition into grid
    x = x.view(B, grid_size, H // grid_size, grid_size, W // grid_size, C)
    x = x.permute(
        0, 2, 4, 1, 3, 5
    ).contiguous()  # (B, H//grid_size, W//grid_size, grid_size, grid_size, C)
    x = x.view(
        -1, grid_size * grid_size, C
    )  # (B * (H//grid_size) * (W//grid_size), grid_size^2, C)

    return x


def grid_reverse(
    x: Tensor, grid_size: int, H: int, W: int, original_h: int, original_w: int
) -> Tensor:
    """Reverse grid partitioning.

    Args:
        x: Grid-partitioned tensor
        grid_size: Size of the grid
        H: Padded height
        W: Padded width
        original_h: Original height before padding
        original_w: Original width before padding

    Returns:
        Tensor of shape (B, original_h, original_w, C)
    """
    num_grids_h = H // grid_size
    num_grids_w = W // grid_size
    B = x.shape[0] // (num_grids_h * num_grids_w)
    C = x.shape[-1]

    # Ensure the tensor can be reshaped correctly
    expected_elements = B * num_grids_h * num_grids_w * grid_size * grid_size * C
    actual_elements = x.numel()

    if expected_elements != actual_elements:
        raise RuntimeError(
            f"Cannot reshape tensor with {actual_elements} elements to "
            f"({B}, {num_grids_h}, {num_grids_w}, {grid_size}, {grid_size}, {C}) "
            f"which requires {expected_elements} elements"
        )

    # Reshape back from grid
    x = x.view(B, num_grids_h, num_grids_w, grid_size, grid_size, C)
    x = x.permute(
        0, 1, 3, 2, 4, 5
    ).contiguous()  # (B, num_grids_h, grid_size, num_grids_w, grid_size, C)
    x = x.view(B, H, W, C)

    # Remove padding if any
    x = x[:, :original_h, :original_w, :].contiguous()

    return x


class Attention(nn.Module):
    """Multi-head attention module for MaxViT."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP module used in MaxViT blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PartitionAttention(nn.Module):
    """Partition attention module (window or grid) for MaxViT."""

    def __init__(
        self,
        dim: int,
        partition_type: PartitionType,
        partition_size: int = 7,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        skip_first_norm: bool = False,
        init_values: Optional[float] = 1e-5,  # LayerScale init value
    ):
        super().__init__()
        self.partition_type = partition_type
        self.partition_size = partition_size
        self.skip_first_norm = skip_first_norm

        if not skip_first_norm:
            self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # LayerScale parameters (ls1 for attention, ls2 for MLP)
        if init_values is not None:
            self.ls1 = nn.Parameter(init_values * torch.ones(dim))
            self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.ls1 = None
            self.ls2 = None

        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            Output tensor of shape (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Store for residual
        shortcut = x

        # Pre-norm
        if not self.skip_first_norm:
            x = self.norm1(x)

        # Partition
        if self.partition_type == PartitionType.WINDOW:
            x_partitioned, num_windows_h, num_windows_w = window_partition(
                x, self.partition_size
            )
            # Apply attention
            x_partitioned = self.attn(x_partitioned)
            # Reverse partition
            x = window_reverse(
                x_partitioned, self.partition_size, num_windows_h, num_windows_w, H, W
            )
        else:  # GRID
            # For grid attention, we use a fixed grid size
            grid_size = min(self.partition_size, min(H, W))
            # Pad to ensure dimensions are divisible by grid_size
            pad_h = (grid_size - H % grid_size) % grid_size
            pad_w = (grid_size - W % grid_size) % grid_size

            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
                padded_H, padded_W = H + pad_h, W + pad_w
            else:
                padded_H, padded_W = H, W

            x_partitioned = grid_partition(x, grid_size)
            # Apply attention
            x_partitioned = self.attn(x_partitioned)
            # Reverse partition
            x = grid_reverse(x_partitioned, grid_size, padded_H, padded_W, H, W)

        # Residual connection with LayerScale
        if self.ls1 is not None:
            x = shortcut + self.drop_path(self.ls1 * x)
        else:
            x = shortcut + self.drop_path(x)

        # MLP block with LayerScale
        mlp_out = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            x = x + self.drop_path(self.ls2 * mlp_out)
        else:
            x = x + self.drop_path(mlp_out)

        return x


class MaxViTBlock(nn.Module):
    """MaxViT block with window and grid attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        grid_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        skip_first_norm: bool = False,
        init_values: Optional[float] = 1e-5,  # LayerScale init value
    ):
        super().__init__()

        self.window_attn = PartitionAttention(
            dim=dim,
            partition_type=PartitionType.WINDOW,
            partition_size=window_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            skip_first_norm=skip_first_norm,
            init_values=init_values,
        )

        self.grid_attn = PartitionAttention(
            dim=dim,
            partition_type=PartitionType.GRID,
            partition_size=grid_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            skip_first_norm=False,  # Always apply norm for grid attention
            init_values=init_values,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: window attention followed by grid attention."""
        x = self.window_attn(x)
        x = self.grid_attn(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer for converting from NCHW to NHWC with downsampling."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 4,
        stride: Optional[int] = None,
        padding: int = 0,
        norm_layer: Optional[nn.Module] = None,
        overlap: bool = True,  # Use overlapping patches like reference
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size

        # Use reference implementation's overlapping patch approach
        if overlap:
            kernel_size = (patch_size - 1) * 2 + 1  # Reference formula
            padding = kernel_size // 2
        else:
            kernel_size = patch_size
            padding = padding

        self.conv = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            bias=False,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: NCHW -> NHWC with embedding."""
        x = self.conv(x)  # (B, embed_dim, H', W')
        x = nchw_to_nhwc(x)  # (B, H', W', embed_dim)

        if self.norm:
            x = self.norm(x)

        return x

    def output_is_normed(self) -> bool:
        """Return whether the output is normalized."""
        return self.norm is not None


class ConvDownsample(nn.Module):
    """Convolutional downsampling layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=norm_layer is None,
        )
        self.norm = norm_layer(out_channels) if norm_layer else None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: NCHW -> NHWC with downsampling."""
        x = self.conv(x)  # (B, out_channels, H', W')
        x = nchw_to_nhwc(x)  # (B, H', W', out_channels)

        if self.norm:
            x = self.norm(x)

        return x

    def output_is_normed(self) -> bool:
        """Return whether the output is normalized."""
        return self.norm is not None


def get_downsample_layer(
    in_channels: int,
    out_channels: int,
    downsample_factor: int,
    downsample_type: str = "patch",
    use_norm: bool = True,
) -> Union[PatchEmbed, ConvDownsample]:
    """Get appropriate downsampling layer."""
    norm_layer = nn.LayerNorm if use_norm else None

    if downsample_type == "patch":
        return PatchEmbed(
            in_channels,
            out_channels,
            patch_size=downsample_factor,
            norm_layer=norm_layer,
        )
    elif downsample_type == "conv":
        return ConvDownsample(
            in_channels, out_channels, stride=downsample_factor, norm_layer=norm_layer
        )
    else:
        raise ValueError(f"Unknown downsample type: {downsample_type}")


class DWSConvLSTM2d(nn.Module):
    """Depthwise Separable Convolutional LSTM matching reference RVT implementation."""

    def __init__(
        self,
        dim: int,
        dws_conv: bool = True,
        dws_conv_only_hidden: bool = True,
        dws_conv_kernel_size: int = 3,
        cell_update_dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(dws_conv, bool)
        assert isinstance(dws_conv_only_hidden, bool)
        self.dim = dim

        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim

        # Depthwise separable conv for spatial mixing (matches reference)
        self.conv3x3_dws = (
            nn.Conv2d(
                in_channels=conv3x3_dws_dim,
                out_channels=conv3x3_dws_dim,
                kernel_size=dws_conv_kernel_size,
                padding=dws_conv_kernel_size // 2,
                groups=conv3x3_dws_dim,
            )
            if dws_conv
            else nn.Identity()
        )

        # 1x1 conv for gate computation (matches reference checkpoint structure)
        self.conv1x1 = nn.Conv2d(
            in_channels=xh_dim, out_channels=gates_dim, kernel_size=1
        )

        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(
        self,
        x: torch.Tensor,
        h_and_c_previous: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass matching reference RVT implementation.

        Args:
            x: Input tensor (N, C, H, W)
            h_and_c_previous: Previous (hidden, cell) state tuple ((N, C, H, W), (N, C, H, W))

        Returns:
            Tuple of (hidden, cell) tensors ((N, C, H, W), (N, C, H, W))
        """
        if h_and_c_previous is None:
            # Generate zero states
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x)
            h_and_c_previous = (hidden, cell)
        h_tm1, c_tm1 = h_and_c_previous

        if self.conv_only_hidden:
            h_tm1 = self.conv3x3_dws(h_tm1)

        # Concatenate input and hidden (reference approach)
        xh = torch.cat((x, h_tm1), dim=1)

        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)

        # Single conv1x1 for all gates (matches checkpoint structure)
        mix = self.conv1x1(xh)

        # Split into gates and cell input
        gates, cell_input = torch.tensor_split(mix, [self.dim * 3], dim=1)
        assert gates.shape[1] == cell_input.shape[1] * 3

        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)
        assert forget_gate.shape == input_gate.shape == output_gate.shape

        cell_input = self.cell_update_dropout(torch.tanh(cell_input))

        c_t = forget_gate * c_tm1 + input_gate * cell_input
        h_t = output_gate * torch.tanh(c_t)

        return h_t, c_t
