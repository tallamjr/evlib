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


def _as_partition_tuple(partition_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Coerce a scalar or 2-tuple partition size to a ``(h, w)`` tuple."""
    if isinstance(partition_size, int):
        return (partition_size, partition_size)
    partition_size = tuple(partition_size)
    assert len(partition_size) == 2, (
        f"partition size must be 2-tuple, got {partition_size}"
    )
    return (int(partition_size[0]), int(partition_size[1]))


def window_partition(x: Tensor, window_size: Union[int, Tuple[int, int]]) -> Tensor:
    """Partition input into non-overlapping windows (reference-identical).

    The tuple is the WINDOW SIZE: each window is ``window_size[0] x
    window_size[1]`` and there are ``H // window_size[0]`` by ``W //
    window_size[1]`` windows. Mirrors
    ``ssms_event_cameras/RVT/.../maxvit.window_partition``: divisibility is
    asserted (no internal padding); the harness pre-pads the input so every
    stage divides evenly.

    Args:
        x: Input tensor of shape (B, H, W, C)
        window_size: Window size, an int (square) or ``(h, w)`` tuple

    Returns:
        Windowed tensor of shape ``(B * num_windows, win_h * win_w, C)``
    """
    win_h, win_w = _as_partition_tuple(window_size)
    B, H, W, C = x.shape
    assert H % win_h == 0, f"height ({H}) must be divisible by window ({win_h})"
    assert W % win_w == 0, f"width ({W}) must be divisible by window ({win_w})"

    x = x.view(B, H // win_h, win_h, W // win_w, win_w, C)
    # (B, num_windows_h, num_windows_w, win_h, win_w, C) -> flatten spatial dims
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(-1, win_h * win_w, C)
    return x


def window_reverse(
    x: Tensor,
    window_size: Union[int, Tuple[int, int]],
    img_size: Tuple[int, int],
) -> Tensor:
    """Reverse window partitioning (reference-identical).

    Args:
        x: Windowed tensor of shape ``(B * num_windows, win_h * win_w, C)``
        window_size: Window size used in :func:`window_partition`
        img_size: The ``(H, W)`` of the tensor that was partitioned

    Returns:
        Tensor of shape (B, H, W, C)
    """
    win_h, win_w = _as_partition_tuple(window_size)
    H, W = img_size
    C = x.shape[-1]

    x = x.view(-1, H // win_h, W // win_w, win_h, win_w, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(-1, H, W, C)
    return x


def grid_partition(x: Tensor, grid_size: Union[int, Tuple[int, int]]) -> Tensor:
    """Partition input into a grid for grid attention (reference-identical).

    Mirrors ``ssms_event_cameras/RVT/.../maxvit.grid_partition``: the tuple is
    the GRID-WINDOW size, so each attention window is ``grid_size[0] x
    grid_size[1]`` tokens sampled on a stride-``(H // grid_h, W // grid_w)``
    lattice, and there are ``(H // grid_h) * (W // grid_w)`` such windows.
    Divisibility is asserted (no internal padding).

    Args:
        x: Input tensor of shape (B, H, W, C)
        grid_size: Grid size, an int (square) or ``(h, w)`` tuple

    Returns:
        Grid-partitioned tensor of shape
        ``(B * (H // grid_h) * (W // grid_w), grid_h * grid_w, C)``
    """
    grid_h, grid_w = _as_partition_tuple(grid_size)
    B, H, W, C = x.shape
    assert H % grid_h == 0, f"height ({H}) must be divisible by grid ({grid_h})"
    assert W % grid_w == 0, f"width ({W}) must be divisible by grid ({grid_w})"

    x = x.view(B, grid_h, H // grid_h, grid_w, W // grid_w, C)
    # -> (B, H//grid_h, W//grid_w, grid_h, grid_w, C); the trailing (grid_h,
    # grid_w) are the attention-window tokens, the leading spatial dims index
    # the windows. This matches the reference's (-1, grid_h, grid_w, C).
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(-1, grid_h * grid_w, C)
    return x


def grid_reverse(
    x: Tensor,
    grid_size: Union[int, Tuple[int, int]],
    img_size: Tuple[int, int],
) -> Tensor:
    """Reverse grid partitioning (reference-identical).

    Args:
        x: Grid-partitioned tensor from :func:`grid_partition`
        grid_size: Grid size used in :func:`grid_partition`
        img_size: The ``(H, W)`` of the tensor that was partitioned

    Returns:
        Tensor of shape (B, H, W, C)
    """
    grid_h, grid_w = _as_partition_tuple(grid_size)
    H, W = img_size
    C = x.shape[-1]

    x = x.view(-1, H // grid_h, W // grid_w, grid_h, grid_w, C)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(-1, H, W, C)
    return x


class Attention(nn.Module):
    """Multi-head attention module for MaxViT.

    The reference RVT (``SelfAttentionCl``) fixes ``dim_head`` and derives
    ``num_heads = dim // dim_head`` per stage (so a 32-wide stage has 1 head, a
    256-wide stage has 8). A fixed ``num_heads`` across stages would change the
    head grouping for every stage except the widest and diverge from the trained
    checkpoint, so this module is parametrised by ``dim_head`` to match.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int = 32,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % dim_head == 0, (
            f"dim ({dim}) must be a multiple of dim_head ({dim_head})"
        )
        self.num_heads = dim // dim_head
        head_dim = dim_head
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        head_dim = C // self.num_heads
        # Reference ``SelfAttentionCl`` layout: the qkv projection is read as
        # (num_heads, dim_head * 3) and only then split into q/k/v. This differs
        # from a (3, num_heads, dim_head) split whenever num_heads > 1, so it must
        # match the reference to reproduce the trained weights.
        q, k, v = (
            self.qkv(x)
            .view(B, N, self.num_heads, head_dim * 3)
            .transpose(1, 2)
            .chunk(3, dim=3)
        )  # each (B, num_heads, N, head_dim)

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
        partition_size: Union[int, Tuple[int, int]] = 7,
        dim_head: int = 32,
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
        self.partition_size = _as_partition_tuple(partition_size)
        self.skip_first_norm = skip_first_norm

        if not skip_first_norm:
            self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            dim_head=dim_head,
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
        img_size = (H, W)

        # Store for residual
        shortcut = x

        # Pre-norm
        if not self.skip_first_norm:
            x = self.norm1(x)

        # Partition (reference-identical: input is pre-sized so the partition
        # divides every stage evenly; no internal padding).
        if self.partition_type == PartitionType.WINDOW:
            x_partitioned = window_partition(x, self.partition_size)
            x_partitioned = self.attn(x_partitioned)
            x = window_reverse(x_partitioned, self.partition_size, img_size)
        else:  # GRID
            x_partitioned = grid_partition(x, self.partition_size)
            x_partitioned = self.attn(x_partitioned)
            x = grid_reverse(x_partitioned, self.partition_size, img_size)

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
        dim_head: int = 32,
        partition_size: Union[int, Tuple[int, int]] = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        skip_first_norm: bool = False,
        init_values: Optional[float] = 1e-5,  # LayerScale init value
    ):
        super().__init__()

        # The reference uses a single resolution-derived partition size for BOTH
        # the window block (as window size) and the grid block (as grid size).
        self.window_attn = PartitionAttention(
            dim=dim,
            partition_type=PartitionType.WINDOW,
            partition_size=partition_size,
            dim_head=dim_head,
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
            partition_size=partition_size,
            dim_head=dim_head,
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
