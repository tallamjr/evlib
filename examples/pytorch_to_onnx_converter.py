#!/usr/bin/env python3
"""Convert PyTorch E2VID models to ONNX format for use with evlib

This script provides utilities to convert pre-trained PyTorch models
to ONNX format, enabling efficient inference with ONNX Runtime.

Usage:
    python pytorch_to_onnx_converter.py --model e2vid --input model.pth --output model.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class E2VIDNet(nn.Module):
    """E2VID UNet architecture for PyTorch"""

    def __init__(self, num_bins=5, base_channels=32):
        super().__init__()
        # Encoder
        self.enc1 = self._encoder_block(num_bins, base_channels)
        self.enc2 = self._encoder_block(base_channels, base_channels * 2)
        self.enc3 = self._encoder_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._encoder_block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec4 = self._decoder_block(base_channels * 16, base_channels * 4)
        self.dec3 = self._decoder_block(base_channels * 8, base_channels * 2)
        self.dec2 = self._decoder_block(base_channels * 4, base_channels)
        self.dec1 = self._decoder_block(base_channels * 2, base_channels)

        # Output
        self.output_conv = nn.Conv2d(base_channels, 1, 1)

    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([nn.functional.interpolate(b, scale_factor=2), e4], dim=1))
        d3 = self.dec3(torch.cat([nn.functional.interpolate(d4, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([nn.functional.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([nn.functional.interpolate(d2, scale_factor=2), e1], dim=1))

        # Output
        return torch.sigmoid(self.output_conv(d1))


class FireNet(nn.Module):
    """FireNet lightweight architecture for PyTorch"""

    def __init__(self, num_bins=5):
        super().__init__()
        self.stem = nn.Conv2d(num_bins, 32, 3, padding=1)

        self.fire1 = self._fire_module(32, 16, 32)
        self.fire2 = self._fire_module(64, 16, 32)
        self.fire3 = self._fire_module(64, 32, 64)
        self.fire4 = self._fire_module(128, 32, 64)

        self.output_conv = nn.Conv2d(128, 1, 1)

    def _fire_module(self, in_channels, squeeze_channels, expand_channels):
        return FireModule(in_channels, squeeze_channels, expand_channels)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        return torch.sigmoid(self.output_conv(x))


class FireModule(nn.Module):
    """Fire module: squeeze + expand"""

    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.expand_1x1 = nn.Conv2d(squeeze_channels, expand_channels, 1)
        self.expand_3x3 = nn.Conv2d(squeeze_channels, expand_channels, 3, padding=1)

    def forward(self, x):
        squeezed = torch.relu(self.squeeze(x))
        return torch.cat(
            [torch.relu(self.expand_1x1(squeezed)), torch.relu(self.expand_3x3(squeezed))],
            dim=1,
        )


def load_pytorch_model(model_path, model_type="e2vid", num_bins=5):
    """Load PyTorch model from checkpoint"""
    if model_type == "e2vid":
        model = E2VIDNet(num_bins=num_bins)
    elif model_type == "firenet":
        model = FireNet(num_bins=num_bins)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def convert_to_onnx(model, output_path, input_shape=(1, 5, 256, 256), opset_version=11, verbose=True):
    """Convert PyTorch model to ONNX format"""
    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    if verbose:
        print(f"Converting model with input shape: {input_shape}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["voxel_grid"],
        output_names=["reconstructed_frame"],
        dynamic_axes={
            "voxel_grid": {0: "batch_size", 2: "height", 3: "width"},
            "reconstructed_frame": {0: "batch_size", 2: "height", 3: "width"},
        },
        verbose=verbose,
    )

    if verbose:
        print(f"Model saved to: {output_path}")


def verify_onnx_model(model_path):
    """Verify ONNX model is valid"""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnx or onnxruntime not installed, skipping verification")
        return True

    # Check model
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure is valid")

        # Test inference
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        # Handle dynamic axes
        concrete_shape = []
        for dim in input_shape:
            if isinstance(dim, str):
                concrete_shape.append(1 if "batch" in dim else 256)
            else:
                concrete_shape.append(dim if dim is not None else 256)

        # Test inference
        dummy_input = np.random.randn(*concrete_shape).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        print(f"✓ ONNX inference successful, output shape: {outputs[0].shape}")

        return True
    except Exception as e:
        print(f"✗ ONNX verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch E2VID models to ONNX format")
    parser.add_argument(
        "--model",
        type=str,
        choices=["e2vid", "firenet"],
        default="e2vid",
        help="Model architecture type",
    )
    parser.add_argument("--input", type=str, required=True, help="Input PyTorch model path (.pth)")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX model path (.onnx)")
    parser.add_argument("--num-bins", type=int, default=5, help="Number of voxel grid bins")
    parser.add_argument("--height", type=int, default=256, help="Input height (default: 256)")
    parser.add_argument("--width", type=int, default=256, help="Input width (default: 256)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11)")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX model verification")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Load PyTorch model
    print(f"Loading {args.model} model from: {input_path}")
    try:
        model = load_pytorch_model(input_path, args.model, args.num_bins)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Convert to ONNX
    input_shape = (args.batch_size, args.num_bins, args.height, args.width)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        convert_to_onnx(model, str(output_path), input_shape, args.opset, verbose=not args.quiet)
    except Exception as e:
        print(f"Error converting model: {e}")
        sys.exit(1)

    # Verify if requested
    if not args.no_verify:
        if not verify_onnx_model(output_path):
            sys.exit(1)

    print(f"\n✓ Successfully converted {args.model} to ONNX format!")
    print(f"  Input shape: {input_shape}")
    print(f"  Output path: {output_path}")
    print(
        f"\nUse with evlib:\n"
        f"  frame = evlib.processing.events_to_video_advanced(\n"
        f"      xs, ys, ts, ps, height, width,\n"
        f'      model_type="onnx",\n'
        f'      model_path="{output_path}"\n'
        f"  )"
    )


if __name__ == "__main__":
    main()
