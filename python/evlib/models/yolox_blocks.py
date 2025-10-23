"""YOLOX building blocks for object detection.

This module implements the core building blocks used in YOLOX architecture:
- BaseConv: Basic convolution with batch normalization and activation
- DWConv: Depthwise separable convolution
- CSPLayer: Cross Stage Partial layer for efficient feature extraction
- Focus: Efficient downsampling layer
- SPPBottleneck: Spatial Pyramid Pooling bottleneck

Based on the YOLOX paper: "YOLOX: Exceeding YOLO Series in 2021"
"""

import torch
import torch.nn as nn


def get_activation(name: str = "silu", inplace: bool = True) -> nn.Module:
    """Get activation function by name.

    Args:
        name: Activation function name ('silu', 'relu', 'lrelu', 'swish')
        inplace: Whether to use inplace operations

    Returns:
        Activation module
    """
    if name == "silu" or name == "swish":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError(f"Unsupported activation type: {name}")
    return module


class BaseConv(nn.Module):
    """Basic convolution block: Conv2d -> BatchNorm2d -> Activation.

    This is the fundamental building block used throughout YOLOX architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        act: str = "silu",
    ):
        """Initialize BaseConv.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            groups: Number of groups for grouped convolution
            bias: Whether to use bias in convolution
            act: Activation function name
        """
        super().__init__()

        # Calculate padding for 'same' padding
        pad = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for fused conv+bn (used in deployment)."""
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise separable convolution: DWConv + PWConv.

    Reduces parameters and computation while maintaining similar performance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        act: str = "silu",
    ):
        """Initialize DWConv.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Depthwise convolution kernel size
            stride: Convolution stride
            act: Activation function name
        """
        super().__init__()

        # Depthwise convolution
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            act=act,
        )

        # Pointwise convolution
        self.pconv = BaseConv(
            in_channels, out_channels, kernel_size=1, stride=1, groups=1, act=act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck block with optional shortcut connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ):
        """Initialize Bottleneck.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            shortcut: Whether to use residual connection
            expansion: Channel expansion ratio for hidden layer
            depthwise: Whether to use depthwise convolution
            act: Activation function name
        """
        super().__init__()

        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.

    CSP architecture reduces computation by splitting feature maps and
    processing only half through bottleneck blocks, then concatenating.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ):
        """Initialize CSPLayer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n: Number of bottleneck blocks
            shortcut: Whether to use shortcuts in bottlenecks
            expansion: Channel expansion ratio
            depthwise: Whether to use depthwise convolution
            act: Activation function name
        """
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        # Split input into two paths
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        # Bottleneck blocks for first path
        self.m = nn.Sequential(
            *[
                Bottleneck(
                    hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
                )
                for _ in range(n)
            ]
        )

        # Final convolution to combine paths
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x_1 = self.conv1(x)  # First path through bottlenecks
        x_2 = self.conv2(x)  # Second path (identity)

        x_1 = self.m(x_1)  # Apply bottleneck blocks
        x = torch.cat((x_1, x_2), dim=1)  # Concatenate paths

        return self.conv3(x)


class Focus(nn.Module):
    """Focus layer for efficient downsampling.

    Rearranges input tensor by slicing and concatenating to reduce spatial
    dimensions while increasing channel dimensions. More efficient than
    regular convolution for downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        act: str = "silu",
    ):
        """Initialize Focus layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for final convolution
            stride: Stride for final convolution
            act: Activation function name
        """
        super().__init__()

        self.conv = BaseConv(
            in_channels * 4, out_channels, kernel_size, stride, act=act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H//2, W//2)
        """
        # Slice tensor into 4 parts and concatenate
        # x[..., ::2, ::2] -> top-left pixels
        # x[..., 1::2, ::2] -> top-right pixels
        # x[..., ::2, 1::2] -> bottom-left pixels
        # x[..., 1::2, 1::2] -> bottom-right pixels

        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]

        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1
        )

        return self.conv(x)


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling bottleneck.

    Applies multiple pooling operations with different kernel sizes
    to capture multi-scale features, then concatenates results.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (5, 9, 13),
        act: str = "silu",
    ):
        """Initialize SPPBottleneck.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_sizes: Tuple of pooling kernel sizes
            act: Activation function name
        """
        super().__init__()

        hidden_channels = in_channels // 2

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        # Multiple pooling layers
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )

        # Final convolution (original + 3 pooled features)
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)

        # Apply different pooling operations
        pooled_features = [x] + [m(x) for m in self.m]

        # Concatenate all features
        x = torch.cat(pooled_features, dim=1)

        return self.conv2(x)
