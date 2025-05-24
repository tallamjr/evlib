#!/usr/bin/env python3
"""
Convert PyTorch event-to-video reconstruction models to ONNX format.

This script downloads PyTorch models from their original sources and converts
them to ONNX format for use with evlib.
"""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn


class E2VidUNet(nn.Module):
    """Minimal E2VID UNet architecture for ONNX export."""

    def __init__(self, in_channels=5, base_channels=64):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        bottleneck = self.bottleneck(enc2)
        dec1 = self.decoder1(torch.cat([bottleneck, enc2], dim=1))
        dec2 = self.decoder2(torch.cat([dec1, enc1], dim=1))
        return torch.sigmoid(dec2)


class FireNet(nn.Module):
    """Minimal FireNet architecture for ONNX export."""

    def __init__(self, in_channels=5, base_channels=16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            self._res_block(base_channels, base_channels),
            self._res_block(base_channels, base_channels),
            self._res_block(base_channels, base_channels),
        )
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def _res_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.head(x)
        x = x + self.res_blocks(x)
        x = self.tail(x)
        return torch.sigmoid(x)


def download_file(url, dest_path):
    """Download a file from URL to destination path."""
    print(f"Downloading from {url}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False


def calculate_sha256(file_path):
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def convert_pytorch_to_onnx(model, dummy_input, output_path, model_name):
    """Convert PyTorch model to ONNX format."""
    print(f"Converting {model_name} to ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Successfully converted to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to convert: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX")
    parser.add_argument(
        "--model",
        choices=["e2vid", "firenet", "all"],
        default="all",
        help="Model to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--download-pytorch",
        action="store_true",
        help="Download original PyTorch models",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    models_to_convert = []
    if args.model == "all":
        models_to_convert = ["e2vid", "firenet"]
    else:
        models_to_convert = [args.model]

    # Model URLs and info
    model_info = {
        "e2vid": {
            "url": "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar",
            "class": E2VidUNet,
            "kwargs": {"in_channels": 5, "base_channels": 64},
        },
        "firenet": {
            "url": None,  # FireNet checkpoint needs manual download
            "class": FireNet,
            "kwargs": {"in_channels": 5, "base_channels": 16},
        },
    }

    for model_name in models_to_convert:
        print(f"\n=== Processing {model_name} ===")
        info = model_info[model_name]

        # Create model instance
        model = info["class"](**info["kwargs"])
        model.eval()

        # Load weights if downloading PyTorch models
        if args.download_pytorch and info["url"]:
            with tempfile.NamedTemporaryFile(suffix=".pth.tar") as tmp:
                if download_file(info["url"], tmp.name):
                    try:
                        checkpoint = torch.load(tmp.name, map_location="cpu")
                        if "state_dict" in checkpoint:
                            model.load_state_dict(checkpoint["state_dict"])
                        else:
                            model.load_state_dict(checkpoint)
                        print(f"Loaded weights for {model_name}")
                    except Exception as e:
                        print(f"Warning: Could not load weights: {e}")
                        print("Proceeding with random weights...")

        # Create dummy input
        dummy_input = torch.randn(1, 5, 256, 256)

        # Convert to ONNX
        onnx_path = output_dir / f"{model_name}.onnx"
        if convert_pytorch_to_onnx(model, dummy_input, str(onnx_path), model_name):
            # Calculate checksum
            checksum = calculate_sha256(onnx_path)
            file_size = os.path.getsize(onnx_path)
            print(f"Model: {model_name}")
            print(f"Path: {onnx_path}")
            print(f"Size: {file_size:,} bytes")
            print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
