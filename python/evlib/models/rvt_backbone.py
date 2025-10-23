"""RVT (Recurrent Vision Transformer) backbone implementation.

This module implements the RVT backbone which combines MaxViT attention blocks
with convolutional LSTM for recurrent processing across temporal sequences.

Based on the CVPR 2023 paper "Recurrent Vision Transformers for Object Detection with Event Cameras"
by Mathias Gehrig and Davide Scaramuzza.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .rvt_layers import MaxViTBlock, DWSConvLSTM2d, get_downsample_layer, nhwc_to_nchw


@dataclass
class RVTConfig:
    """Configuration for RVT backbone."""

    # Input/output configuration
    input_channels: int = 20  # 10 temporal bins * 2 polarities
    embed_dim: int = 32  # Base embedding dimension (tiny model)

    # Architecture configuration
    dim_multiplier: Tuple[int, ...] = (1, 2, 4, 8)  # Channel multipliers per stage
    num_blocks: Tuple[int, ...] = (1, 1, 1, 1)  # Number of MaxViT blocks per stage
    num_stages: int = 4

    # Stem configuration
    stem_patch_size: int = 4
    stem_stride: Optional[int] = None

    # Attention configuration
    num_heads: int = 8
    window_size: int = 7
    grid_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    # LayerScale configuration
    init_values: Optional[float] = 1e-5  # LayerScale init value, None to disable

    # LSTM configuration
    lstm_kernel_size: int = 3
    dws_conv: bool = False
    dws_conv_only_hidden: bool = True
    dws_conv_kernel_size: int = 3

    # Downsampling configuration
    downsample_type: str = "patch"  # "patch" or "conv"
    use_norm: bool = True

    # Masking (for self-supervised learning)
    enable_masking: bool = False

    @classmethod
    def tiny(cls) -> "RVTConfig":
        """Create configuration for RVT-Tiny model."""
        return cls(embed_dim=32)

    @classmethod
    def small(cls) -> "RVTConfig":
        """Create configuration for RVT-Small model."""
        return cls(embed_dim=64)

    @classmethod
    def base(cls) -> "RVTConfig":
        """Create configuration for RVT-Base model."""
        return cls(embed_dim=96)


class RVTStage(nn.Module):
    """Single stage of the RVT backbone.

    Each stage consists of:
    1. Downsampling (patch embedding or conv)
    2. MaxViT attention blocks
    3. Convolutional LSTM
    4. Optional mask token handling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_factor: int,
        num_blocks: int,
        config: RVTConfig,
        enable_masking: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_factor = downsample_factor
        self.num_blocks = num_blocks
        self.enable_masking = enable_masking

        # Downsampling layer (NCHW -> NHWC) - named to match checkpoint
        self.downsample_cf2cl = get_downsample_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            downsample_factor=downsample_factor,
            downsample_type=config.downsample_type,
            use_norm=config.use_norm,
        )

        # MaxViT attention blocks
        self.attention_blocks = nn.ModuleList()
        for i in range(num_blocks):
            skip_first_norm = (i == 0) and self.downsample_cf2cl.output_is_normed()

            block = MaxViTBlock(
                dim=out_channels,
                num_heads=config.num_heads,
                window_size=config.window_size,
                grid_size=config.grid_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=config.drop_path_rate,
                skip_first_norm=skip_first_norm,
                init_values=config.init_values,
            )
            self.attention_blocks.append(block)

        # Convolutional LSTM (updated to match reference architecture)
        self.lstm = DWSConvLSTM2d(
            dim=out_channels,
            dws_conv=config.dws_conv,
            dws_conv_only_hidden=config.dws_conv_only_hidden,
            dws_conv_kernel_size=config.dws_conv_kernel_size,
        )

        # Mask token for self-supervised learning
        if enable_masking:
            self.mask_token = nn.Parameter(
                torch.zeros(1, 1, 1, out_channels), requires_grad=True
            )
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.mask_token = None

    def forward(
        self,
        x: Tensor,
        hidden_state: Optional[Tuple[Tensor, Tensor]] = None,
        token_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through the stage.

        Args:
            x: Input tensor of shape (B, C, H, W)
            hidden_state: Previous LSTM hidden state (h, c)
            token_mask: Optional mask for tokens (B, H, W) for masking

        Returns:
            Tuple of (output, new_hidden_state)
        """
        # Downsample: NCHW -> NHWC
        x = self.downsample_cf2cl(x)  # (B, H', W', C')

        # Apply token masking if provided
        if token_mask is not None and self.mask_token is not None:
            # Resize mask to match downsampled resolution
            _, H, W, _ = x.shape
            if token_mask.shape[-2:] != (H, W):
                token_mask = (
                    torch.nn.functional.interpolate(
                        token_mask.float().unsqueeze(1), size=(H, W), mode="nearest"
                    )
                    .squeeze(1)
                    .bool()
                )

            # Apply mask
            x[token_mask] = self.mask_token

        # Apply attention blocks
        for block in self.attention_blocks:
            x = block(x)

        # Convert back to NCHW for LSTM
        x = nhwc_to_nchw(x)  # (B, C', H', W')

        # Apply LSTM (new architecture returns (h, c) directly)
        h, c = self.lstm(x, hidden_state)
        output = h  # Output is the hidden state
        new_hidden_state = (h, c)

        return output, new_hidden_state


class RVTBackbone(nn.Module):
    """RVT backbone with multiple stages and recurrent processing."""

    def __init__(self, config: RVTConfig):
        super().__init__()

        self.config = config
        self.num_stages = config.num_stages

        # Validate configuration
        assert len(config.dim_multiplier) == config.num_stages
        assert len(config.num_blocks) == config.num_stages

        # Build stages
        self.stages = nn.ModuleList()
        self.stage_dims = []
        self.strides = []

        current_channels = config.input_channels
        current_stride = 1

        for stage_idx in range(config.num_stages):
            # Calculate stage dimensions
            stage_dim = config.embed_dim * config.dim_multiplier[stage_idx]
            self.stage_dims.append(stage_dim)

            # Calculate downsampling factor
            if stage_idx == 0:
                downsample_factor = config.stem_patch_size
            else:
                downsample_factor = 2

            current_stride *= downsample_factor
            self.strides.append(current_stride)

            # Enable masking only for first stage
            enable_masking = config.enable_masking and stage_idx == 0

            # Create stage
            stage = RVTStage(
                in_channels=current_channels,
                out_channels=stage_dim,
                downsample_factor=downsample_factor,
                num_blocks=config.num_blocks[stage_idx],
                config=config,
                enable_masking=enable_masking,
            )

            self.stages.append(stage)
            current_channels = stage_dim

    def get_stage_dims(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get dimensions for specified stages.

        Args:
            stages: Tuple of stage numbers (1-indexed)

        Returns:
            Tuple of stage dimensions
        """
        stage_indices = [s - 1 for s in stages]
        assert all(0 <= idx < self.num_stages for idx in stage_indices)
        return tuple(self.stage_dims[idx] for idx in stage_indices)

    def get_strides(self, stages: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get strides for specified stages.

        Args:
            stages: Tuple of stage numbers (1-indexed)

        Returns:
            Tuple of stage strides
        """
        stage_indices = [s - 1 for s in stages]
        assert all(0 <= idx < self.num_stages for idx in stage_indices)
        return tuple(self.strides[idx] for idx in stage_indices)

    def forward(
        self,
        x: Tensor,
        previous_states: Optional[List[Optional[Tuple[Tensor, Tensor]]]] = None,
        token_mask: Optional[Tensor] = None,
    ) -> Tuple[Dict[int, Tensor], List[Tuple[Tensor, Tensor]]]:
        """Forward pass through all stages.

        Args:
            x: Input tensor of shape (B, C, H, W)
            previous_states: List of previous LSTM states for each stage
            token_mask: Optional token mask for first stage

        Returns:
            Tuple of (stage_outputs, new_states)
            - stage_outputs: Dict mapping stage number (1-indexed) to output tensor
            - new_states: List of new LSTM states for each stage
        """
        if previous_states is None:
            previous_states = [None] * self.num_stages

        assert len(previous_states) == self.num_stages

        stage_outputs = {}
        new_states = []

        current_input = x
        for stage_idx, stage in enumerate(self.stages):
            # Apply token mask only to first stage
            mask = token_mask if stage_idx == 0 else None

            # Forward through stage
            output, new_state = stage(current_input, previous_states[stage_idx], mask)

            # Store outputs and states
            stage_number = stage_idx + 1  # 1-indexed
            stage_outputs[stage_number] = output
            new_states.append(new_state)

            # Update input for next stage
            current_input = output

        return stage_outputs, new_states

    def reset_states(self, batch_size: int, device: torch.device) -> List[None]:
        """Reset LSTM states (returns list of None for initialization)."""
        return [None] * self.num_stages

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RVTBackbone(\n"
            f"  config={self.config}\n"
            f"  stage_dims={self.stage_dims}\n"
            f"  strides={self.strides}\n"
            f")"
        )


# Utility functions for state management
class RVTStateManager:
    """Manages LSTM states for RVT backbone across batches and sequences."""

    def __init__(self, backbone: RVTBackbone):
        self.backbone = backbone
        self.states: Dict[int, List[Optional[Tuple[Tensor, Tensor]]]] = {}

    def reset_worker_states(self, worker_id: int):
        """Reset states for a specific worker."""
        self.states[worker_id] = self.backbone.reset_states(1, torch.device("cpu"))

    def get_states(self, worker_id: int) -> List[Optional[Tuple[Tensor, Tensor]]]:
        """Get states for a worker."""
        if worker_id not in self.states:
            self.reset_worker_states(worker_id)
        return self.states[worker_id]

    def save_states(
        self, worker_id: int, states: List[Tuple[Tensor, Tensor]], detach: bool = True
    ):
        """Save states for a worker."""
        if detach:
            # Detach states to prevent gradient accumulation
            states = [(h.detach(), c.detach()) for h, c in states]
        self.states[worker_id] = states

    def reset_batch_states(self, worker_id: int, reset_mask: Tensor):
        """Reset states for specific samples in a batch.

        Args:
            worker_id: Worker ID
            reset_mask: Boolean tensor of shape (B,) indicating which samples to reset
        """
        if worker_id not in self.states:
            return

        current_states = self.states[worker_id]
        if current_states[0] is None:
            return

        # Reset states for masked samples
        for stage_idx in range(len(current_states)):
            if current_states[stage_idx] is not None:
                h, c = current_states[stage_idx]
                h[reset_mask] = 0
                c[reset_mask] = 0
