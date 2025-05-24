#!/usr/bin/env python3
"""
Enhanced PyTorch to ONNX converter for event-to-video reconstruction models.

This script loads actual PyTorch checkpoints and converts them to optimized ONNX format.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell implementation for E2VID."""

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2

        # Gates: input, forget, cell, output
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=self.padding)

    def forward(self, x, hidden_state):
        h, c = hidden_state
        combined = torch.cat([x, h], dim=1)

        gates = self.gates(combined)
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, (h, c)


class ConvLayer(nn.Module):
    """Convolutional layer with optional normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, norm="BN"):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels)
        else:
            self.norm_layer = None

    def forward(self, x):
        x = self.conv2d(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return F.relu(x)


class ResidualBlock(nn.Module):
    """Residual block for E2VID."""

    def __init__(self, channels, norm="BN"):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, 3, norm=norm)
        self.conv2 = ConvLayer(channels, channels, 3, norm=norm)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class E2VIDRecurrent(nn.Module):
    """E2VID UNet with ConvLSTM architecture matching the actual checkpoint."""

    def __init__(
        self,
        num_bins=5,
        base_channels=32,
        num_encoders=3,
        num_residual_blocks=2,
        norm="BN",
        use_upsample_conv=True,
    ):
        super().__init__()

        # Head
        self.head = ConvLayer(num_bins, base_channels, 5, padding=2, norm=None)

        # Encoders with ConvLSTM
        self.encoders = nn.ModuleList()
        in_ch = base_channels
        # Channel progression: 32 -> 64 -> 128 -> 256
        encoder_channels = [base_channels * (2 ** (i + 1)) for i in range(num_encoders)]

        for i, out_ch in enumerate(encoder_channels):
            encoder = nn.ModuleDict(
                {
                    "conv": ConvLayer(in_ch, out_ch, 5, stride=2, padding=2, norm=norm),
                    "recurrent_block": ConvLSTMCell(out_ch, out_ch, 3),
                }
            )
            self.encoders.append(encoder)
            in_ch = out_ch

        # Residual blocks
        self.resblocks = nn.ModuleList([ResidualBlock(in_ch, norm=norm) for _ in range(num_residual_blocks)])

        # Decoders
        self.decoders = nn.ModuleList()
        decoder_channels = list(reversed(encoder_channels[:-1])) + [base_channels]

        for i, out_ch in enumerate(decoder_channels):
            if use_upsample_conv:
                decoder = nn.ModuleDict(
                    {
                        "upsample": nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                        "conv": ConvLayer(out_ch * 2, out_ch, 5, padding=2, norm=norm),
                    }
                )
            else:
                decoder = nn.ModuleDict({"conv": ConvLayer(in_ch * 2, out_ch, 5, padding=2, norm=norm)})
            self.decoders.append(decoder)
            in_ch = out_ch

        # Prediction
        self.prediction = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x, states=None):
        # Head
        x = self.head(x)

        # Encode
        skip_connections = []
        new_states = []

        for i, encoder in enumerate(self.encoders):
            x = encoder["conv"](x)
            skip_connections.append(x)

            # Initialize state if not provided
            if states is None or i >= len(states):
                b, c, h, w = x.shape
                h_state = torch.zeros(b, c, h, w, device=x.device)
                c_state = torch.zeros(b, c, h, w, device=x.device)
                state = (h_state, c_state)
            else:
                state = states[i]

            # ConvLSTM
            h, new_state = encoder["recurrent_block"](x, state)
            x = h
            new_states.append(new_state)

        # Residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # Decode
        for i, decoder in enumerate(self.decoders):
            if "upsample" in decoder:
                x = decoder["upsample"](x)
            else:
                # Upsample using interpolation
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

            # Skip connection - resize x to match skip dimensions
            skip = skip_connections[-(i + 1)]
            # Always resize to ensure dimensions match (ONNX-friendly)
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = decoder["conv"](x)

        # Prediction
        x = self.prediction(x)
        x = torch.sigmoid(x)

        return x, new_states


def load_e2vid_from_checkpoint(checkpoint_path):
    """Load E2VID model from PyTorch checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract model configuration
    if "model" in checkpoint:
        config = checkpoint["model"]
        model = E2VIDRecurrent(
            num_bins=config.get("num_bins", 5),
            base_channels=config.get("base_num_channels", 32),
            num_encoders=config.get("num_encoders", 3),
            num_residual_blocks=config.get("num_residual_blocks", 2),
            norm=config.get("norm", "BN"),
            use_upsample_conv=config.get("use_upsample_conv", True),
        )
    else:
        # Default configuration
        model = E2VIDRecurrent()

    # Load weights
    if "state_dict" in checkpoint:
        # Map checkpoint keys to model keys
        state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            # Remove 'unetrecurrent.' prefix if present
            if key.startswith("unetrecurrent."):
                key = key[14:]  # Remove 'unetrecurrent.'
            state_dict[key] = value

        model.load_state_dict(state_dict, strict=False)
        print("Loaded E2VID weights from checkpoint")

    return model


def optimize_onnx_model(onnx_path):
    """Apply optimization passes to ONNX model."""
    try:
        import onnx
        from onnx import optimizer

        # Load model
        model = onnx.load(onnx_path)

        # Available optimization passes
        passes = [
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_pad",
            "eliminate_unused_initializer",
            "fuse_bn_into_conv",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_pad_into_conv",
        ]

        # Optimize
        optimized_model = optimizer.optimize(model, passes)

        # Save optimized model
        opt_path = onnx_path.replace(".onnx", "_optimized.onnx")
        onnx.save(optimized_model, opt_path)

        print(f"Saved optimized model to {opt_path}")
        return opt_path
    except ImportError:
        print("ONNX optimizer not available. Install with: pip install onnx")
        return onnx_path


def verify_onnx_model(pytorch_model, onnx_path, input_shape):
    """Verify ONNX model produces same output as PyTorch model."""
    try:
        import onnxruntime as ort

        # Create test input
        test_input = torch.randn(*input_shape)

        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output, _ = pytorch_model(test_input)

        # ONNX inference
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        onnx_output = ort_session.run(None, {input_name: test_input.numpy()})[0]

        # Compare outputs
        diff = np.abs(pytorch_output.numpy() - onnx_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print("Model verification:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-5:
            print("  ✓ Model outputs match!")
            return True
        else:
            print("  ✗ Model outputs differ significantly")
            return False

    except ImportError:
        print("ONNX Runtime not available. Install with: pip install onnxruntime")
        return None


def convert_model_to_onnx(model, model_name, output_dir, input_shape=(1, 5, 256, 256)):
    """Convert PyTorch model to ONNX with optimization and verification."""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    print(f"Converting {model_name} to ONNX...")
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output", "states"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=False,
    )

    print(f"Saved to {onnx_path}")

    # Optimize
    opt_path = optimize_onnx_model(onnx_path)

    # Verify
    verify_onnx_model(model, opt_path, input_shape)

    # Save model info
    info = {
        "model_name": model_name,
        "input_shape": list(input_shape),
        "onnx_path": os.path.basename(onnx_path),
        "optimized_path": os.path.basename(opt_path),
        "file_size": os.path.getsize(opt_path),
        "opset_version": 11,
    }

    info_path = os.path.join(output_dir, f"{model_name}_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return opt_path


def main():
    parser = argparse.ArgumentParser(description="Enhanced PyTorch to ONNX converter for event-based models")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint (.pth or .pth.tar)"
    )
    parser.add_argument("--model-name", type=str, default="e2vid", help="Name for the output model")
    parser.add_argument(
        "--output-dir", type=str, default="models/onnx", help="Output directory for ONNX models"
    )
    parser.add_argument("--height", type=int, default=256, help="Input height")
    parser.add_argument("--width", type=int, default=256, help="Input width")
    parser.add_argument("--num-bins", type=int, default=5, help="Number of time bins in voxel grid")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if "e2vid" in args.model_name.lower():
        model = load_e2vid_from_checkpoint(args.checkpoint)
    else:
        print(f"Unknown model type: {args.model_name}")
        return

    # Convert to ONNX
    input_shape = (1, args.num_bins, args.height, args.width)
    convert_model_to_onnx(model, args.model_name, str(output_dir), input_shape)


if __name__ == "__main__":
    main()
