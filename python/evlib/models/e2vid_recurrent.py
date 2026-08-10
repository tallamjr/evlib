"""Recurrent E2VID (E2VID_lightweight) matching the bundled `e2vid-lite.pth` checkpoint.

Architecture and preprocessing follow the official RPG implementation:
https://github.com/uzh-rpg/rpg_e2vid

Every encoder is followed by a ConvLSTM, so `reconstruct` takes and returns the
recurrent state. Use it to reconstruct a video sequence chunk by chunk.
"""

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel
from .config import ModelConfig
from .e2vid import ConvLayer, TransposedConvLayer

ConvLSTMState = Tuple[torch.Tensor, torch.Tensor]

DEFAULT_WEIGHTS_FILENAME = "e2vid-lite.pth"
CHECKPOINT_PREFIX = "unetrecurrent."


class ConvLSTM(nn.Module):
    """ConvLSTM cell whose single gate convolution is named `Gates`.

    The name must stay `Gates` to match the checkpoint key `recurrent_block.Gates.weight`.
    """

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(
            input_size + hidden_size,
            4 * hidden_size,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self, input_tensor: torch.Tensor, previous_state: Optional[ConvLSTMState] = None
    ) -> ConvLSTMState:
        if previous_state is None:
            state_shape = (
                input_tensor.shape[0],
                self.hidden_size,
            ) + tuple(input_tensor.shape[2:])
            zeros = torch.zeros(
                state_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )
            previous_state = (zeros, zeros.clone())

        previous_hidden, previous_cell = previous_state

        gates = self.Gates(torch.cat((input_tensor, previous_hidden), dim=1))
        input_gate, remember_gate, output_gate, cell_gate = gates.chunk(4, dim=1)

        input_gate = torch.sigmoid(input_gate)
        remember_gate = torch.sigmoid(remember_gate)
        output_gate = torch.sigmoid(output_gate)
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * previous_cell) + (input_gate * cell_gate)
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell


class RecurrentConvLayer(nn.Module):
    """Strided convolution followed by a ConvLSTM.

    Child names `conv` and `recurrent_block` match the checkpoint keys.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        activation: Optional[str] = "relu",
        norm: Optional[str] = None,
    ):
        super().__init__()

        self.conv = ConvLayer(
            in_channels, out_channels, kernel_size, stride, padding, activation, norm
        )
        self.recurrent_block = ConvLSTM(
            input_size=out_channels, hidden_size=out_channels, kernel_size=3
        )

    def forward(
        self, x: torch.Tensor, previous_state: Optional[ConvLSTMState]
    ) -> Tuple[torch.Tensor, ConvLSTMState]:
        x = self.conv(x)
        state = self.recurrent_block(x, previous_state)
        return state[0], state


class ResidualBlock(nn.Module):
    """Residual block with flat `conv1`/`bn1`/`conv2`/`bn2` children.

    The `ResidualBlock` in `e2vid.py` nests its convolutions inside `ConvLayer`, which
    produces `conv1.conv2d.weight`. The checkpoint stores `conv1.weight`, so this block
    keeps the convolutions and batch norms as direct children.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: Optional[str] = None):
        super().__init__()

        bias = norm != "BN"
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.norm = norm
        if norm == "BN":
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        if self.norm in ("BN", "IN"):
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ("BN", "IN"):
            out = self.bn2(out)
        out = out + residual
        return self.relu(out)


class UNetRecurrent(nn.Module):
    """Recurrent UNet: every encoder is followed by a ConvLSTM, skips are summed."""

    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int = 1,
        num_encoders: int = 3,
        base_num_channels: int = 32,
        num_residual_blocks: int = 2,
        norm: Optional[str] = "BN",
    ):
        super().__init__()

        self.num_input_channels = num_input_channels
        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.norm = norm
        max_num_channels = base_num_channels * pow(2, num_encoders)

        # The head carries no normalisation in the reference, hence its conv bias.
        self.head = ConvLayer(
            num_input_channels, base_num_channels, kernel_size=5, stride=1, padding=2
        )

        self.encoders = nn.ModuleList()
        for encoder_index in range(num_encoders):
            self.encoders.append(
                RecurrentConvLayer(
                    base_num_channels * pow(2, encoder_index),
                    base_num_channels * pow(2, encoder_index + 1),
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    norm=norm,
                )
            )

        self.resblocks = nn.ModuleList(
            ResidualBlock(max_num_channels, max_num_channels, norm=norm)
            for _ in range(num_residual_blocks)
        )

        decoder_input_sizes = reversed(
            [base_num_channels * pow(2, i + 1) for i in range(num_encoders)]
        )
        self.decoders = nn.ModuleList(
            TransposedConvLayer(
                input_size, input_size // 2, kernel_size=5, padding=2, norm=norm
            )
            for input_size in decoder_input_sizes
        )

        self.pred = ConvLayer(
            base_num_channels, num_output_channels, 1, activation=None, norm=norm
        )

    def forward(
        self, x: torch.Tensor, previous_states: Optional[List[ConvLSTMState]] = None
    ) -> Tuple[torch.Tensor, List[ConvLSTMState]]:
        """Reconstruct one frame.

        Args:
            x: Voxel grid of shape (N, num_input_channels, H, W). H and W must be
                divisible by 2 ** num_encoders.
            previous_states: ConvLSTM state per encoder, or None for a fresh sequence.

        Returns:
            Tuple of (image in [0, 1] with shape (N, 1, H, W), new state per encoder).
        """
        if previous_states is None:
            previous_states = [None] * self.num_encoders

        x = self.head(x)
        head_output = x

        encoder_outputs = []
        states: List[ConvLSTMState] = []
        for encoder, previous_state in zip(self.encoders, previous_states):
            x, state = encoder(x, previous_state)
            encoder_outputs.append(x)
            states.append(state)

        for resblock in self.resblocks:
            x = resblock(x)

        for decoder_index, decoder in enumerate(self.decoders):
            skip = encoder_outputs[self.num_encoders - decoder_index - 1]
            x = decoder(x + skip)

        return torch.sigmoid(self.pred(x + head_output)), states


def events_to_voxel_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    num_bins: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Build a voxel grid with bilinear interpolation along time.

    Each event splits its polarity between the two nearest temporal bins. Reference:
    `evlib/representation/voxel_grid.py` in the event-vision-library package, itself
    a copy of https://github.com/uzh-rpg/rpg_e2vid.

    Args:
        xs, ys: Integer pixel coordinates.
        ts: Timestamps in seconds.
        ps: Polarities; zeros are treated as -1.
        num_bins: Number of temporal bins.
        height, width: Sensor resolution.

    Returns:
        Voxel grid of shape (num_bins, height, width).
    """
    voxel_grid = np.zeros((num_bins, height, width), dtype=np.float32).ravel()
    if len(ts) == 0:
        return voxel_grid.reshape(num_bins, height, width)

    xs = np.clip(xs.astype(np.int64), 0, width - 1)
    ys = np.clip(ys.astype(np.int64), 0, height - 1)
    polarities = np.where(ps == 0, -1.0, ps).astype(np.float32)

    t_min, t_max = float(np.amin(ts)), float(np.amax(ts))
    delta_t = t_max - t_min
    if delta_t == 0:
        delta_t = 1.0

    scaled_ts = (num_bins - 1) * (ts - t_min) / delta_t
    bin_indices = scaled_ts.astype(np.int64)
    bin_fractions = scaled_ts - bin_indices
    values_left = polarities * (1.0 - bin_fractions)
    values_right = polarities * bin_fractions

    valid = bin_indices < num_bins
    np.add.at(
        voxel_grid,
        xs[valid] + ys[valid] * width + bin_indices[valid] * width * height,
        values_left[valid],
    )

    valid = (bin_indices + 1) < num_bins
    np.add.at(
        voxel_grid,
        xs[valid] + ys[valid] * width + (bin_indices[valid] + 1) * width * height,
        values_right[valid],
    )

    return voxel_grid.reshape(num_bins, height, width)


def normalise_voxel_grid(voxel_grid: torch.Tensor) -> torch.Tensor:
    """Rescale the nonzero voxels to zero mean and unit standard deviation.

    Zero voxels stay zero. Reference: `EventPreprocessor.__call__` in
    `evlib/processing/reconstruction/e2vid_module/utils/inference_utils.py` of the
    event-vision-library package, itself a copy of https://github.com/uzh-rpg/rpg_e2vid.
    """
    nonzero = voxel_grid != 0
    num_nonzeros = nonzero.sum()
    if num_nonzeros == 0:
        return voxel_grid

    mean = voxel_grid.sum() / num_nonzeros
    stddev = torch.sqrt((voxel_grid**2).sum() / num_nonzeros - mean**2)
    return nonzero.float() * (voxel_grid - mean) / stddev


class _CropParameters:
    """Reflection padding to a size the encoders can subsample, plus the inverse crop.

    Reference: `CropParameters` in
    `evlib/processing/reconstruction/e2vid_module/utils/inference_utils.py`.
    """

    def __init__(self, width: int, height: int, num_encoders: int):
        subsample_factor = pow(2, num_encoders)
        padded_width = subsample_factor * math.ceil(width / subsample_factor)
        padded_height = subsample_factor * math.ceil(height / subsample_factor)

        padding_top = math.ceil(0.5 * (padded_height - height))
        padding_bottom = math.floor(0.5 * (padded_height - height))
        padding_left = math.ceil(0.5 * (padded_width - width))
        padding_right = math.floor(0.5 * (padded_width - width))
        self.pad = nn.ReflectionPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )

        centre_x = math.floor(padded_width / 2)
        centre_y = math.floor(padded_height / 2)
        self.ix0 = centre_x - math.floor(width / 2)
        self.ix1 = centre_x + math.ceil(width / 2)
        self.iy0 = centre_y - math.floor(height / 2)
        self.iy1 = centre_y + math.ceil(height / 2)


class E2VIDRecurrent(BaseModel):
    """Recurrent E2VID reconstruction with a persistent ConvLSTM state.

    The pretrained path builds the architecture from the shapes in
    `weights/e2vid-lite.pth` and loads every tensor with `strict=True`.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        pretrained: bool = False,
        num_encoders: int = 3,
        num_residual_blocks: int = 2,
        norm: Optional[str] = "BN",
    ):
        """Initialise the model.

        Args:
            config: Model configuration. Ignored for the fields the checkpoint dictates
                when `pretrained` is True.
            pretrained: Load the bundled `e2vid-lite.pth` weights.
            num_encoders: Encoder count when not loading pretrained weights.
            num_residual_blocks: Bottleneck block count when not loading pretrained weights.
            norm: 'BN', 'IN' or None when not loading pretrained weights.
        """
        self.config = config or ModelConfig()
        self.pretrained = pretrained
        self.num_bins = self.config.num_bins
        self.base_num_channels = self.config.base_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.norm = norm
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[UNetRecurrent] = None

        if pretrained:
            self._load_pretrained_weights()
        else:
            self._build_model()

    @staticmethod
    def weights_path() -> Path:
        """Absolute path of the bundled recurrent checkpoint."""
        return Path(__file__).parent / "weights" / DEFAULT_WEIGHTS_FILENAME

    def _build_model(self):
        """Build the recurrent UNet on the selected device."""
        self._model = UNetRecurrent(
            num_input_channels=self.num_bins,
            num_output_channels=1,
            num_encoders=self.num_encoders,
            base_num_channels=self.base_num_channels,
            num_residual_blocks=self.num_residual_blocks,
            norm=self.norm,
        ).to(self._device)
        self._model.eval()

    def _load_pretrained_weights(self):
        """Build the architecture from the checkpoint shapes and load it strictly."""
        weights_path = self.weights_path()
        if not weights_path.exists():
            raise FileNotFoundError(f"Bundled weights not found at {weights_path}")

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        raw_state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.removeprefix(CHECKPOINT_PREFIX): value
            for key, value in raw_state_dict.items()
        }

        self._apply_checkpoint_architecture(state_dict)
        self._build_model()
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

    def _apply_checkpoint_architecture(self, state_dict: dict):
        """Read the architecture the checkpoint dictates from its tensor shapes."""
        head_weight = state_dict["head.conv2d.weight"]
        self.num_bins = int(head_weight.shape[1])
        self.base_num_channels = int(head_weight.shape[0])
        self.config.num_bins = self.num_bins
        self.config.base_channels = self.base_num_channels
        self.num_encoders = sum(
            1 for key in state_dict if key.endswith(".conv.conv2d.weight")
        )
        self.num_residual_blocks = sum(
            1
            for key in state_dict
            if key.startswith("resblocks.") and key.endswith(".conv1.weight")
        )
        # Only BatchNorm2d gives `norm_layer` an affine weight. The InstanceNorm2d that
        # ConvLayer builds has affine=False, so it stores no `norm_layer.weight`.
        self.norm = (
            "BN"
            if any(key.endswith("norm_layer.weight") for key in state_dict)
            else None
        )

    def reconstruct(
        self,
        events: Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any
        ],
        height: Optional[int] = None,
        width: Optional[int] = None,
        state: Optional[List[ConvLSTMState]] = None,
    ) -> Tuple[np.ndarray, List[ConvLSTMState]]:
        """Reconstruct one frame from a chunk of events.

        Feed the returned state back in for the next chunk of the same sequence.

        Args:
            events: Structured array, tuple of (xs, ys, ts, ps), or Polars frame.
            height: Output height. Inferred from the events when None.
            width: Output width. Inferred from the events when None.
            state: ConvLSTM state from the previous call, or None to start a sequence.

        Returns:
            Tuple of (frame as float32 in [0, 1] with shape (height, width), new state).
        """
        xs, ys, ts, ps, height, width = self.preprocess_events(events, height, width)

        voxel_grid = events_to_voxel_grid(xs, ys, ts, ps, self.num_bins, height, width)
        input_tensor = torch.from_numpy(voxel_grid).unsqueeze(0).to(self._device)
        input_tensor = normalise_voxel_grid(input_tensor)

        crop = _CropParameters(width, height, self.num_encoders)
        input_tensor = crop.pad(input_tensor)

        with torch.no_grad():
            padded_frame, new_state = self._model(input_tensor, state)

        frame = padded_frame[0, 0, crop.iy0 : crop.iy1, crop.ix0 : crop.ix1]
        return frame.cpu().numpy().astype(np.float32), new_state

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"E2VIDRecurrent(pretrained={self.pretrained}, num_bins={self.num_bins}, "
            f"base_num_channels={self.base_num_channels}, encoders={self.num_encoders}, "
            f"residual_blocks={self.num_residual_blocks}, norm='{self.norm}', "
            f"device={self._device})"
        )
