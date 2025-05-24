#!/usr/bin/env python3
"""
Test PyTorch weight loading functionality.

This script helps verify that we can load PyTorch checkpoints
and convert them to a format suitable for Candle.
"""

import torch
from pathlib import Path
import json


def analyze_checkpoint(checkpoint_path):
    """Analyze a PyTorch checkpoint and extract key information."""
    print(f"Analyzing checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Check if it's a state dict or a full checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Checkpoint contains a state_dict")
        state_dict = checkpoint["state_dict"]

        # Print other keys in checkpoint
        other_keys = [k for k in checkpoint.keys() if k != "state_dict"]
        if other_keys:
            print(f"Other keys in checkpoint: {other_keys}")
    else:
        print("Checkpoint is a direct state_dict")
        state_dict = checkpoint

    # Analyze state dict
    print(f"\nState dict contains {len(state_dict)} tensors")

    # Group keys by prefix
    prefixes = {}
    for key in state_dict.keys():
        prefix = key.split(".")[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)

    print("\nKey prefixes:")
    for prefix, keys in prefixes.items():
        print(f"  {prefix}: {len(keys)} keys")

    # Show sample keys
    print("\nSample keys (first 10):")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        tensor = state_dict[key]
        print(f"  {key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    return state_dict


def create_key_mapping(state_dict, model_type):
    """Create a mapping from PyTorch keys to Candle keys."""
    mappings = {}

    if model_type == "e2vid_unet":
        # E2VID UNet specific mappings
        for key in state_dict.keys():
            if key.startswith("encoders.0."):
                # Map encoder layers
                candle_key = key.replace("encoders.0.", "encoder1.")
                mappings[key] = candle_key
            elif key.startswith("encoders.1."):
                candle_key = key.replace("encoders.1.", "encoder2.")
                mappings[key] = candle_key
            elif key.startswith("decoders."):
                # Map decoder layers
                candle_key = key.replace("decoders.", "decoder")
                mappings[key] = candle_key
            else:
                # Keep other keys as-is for now
                mappings[key] = key

    elif model_type == "firenet":
        # FireNet specific mappings
        for key in state_dict.keys():
            if key.startswith("head."):
                mappings[key] = key
            elif key.startswith("res_blocks."):
                mappings[key] = key
            elif key.startswith("tail."):
                mappings[key] = key
            else:
                mappings[key] = key

    return mappings


def save_weight_info(checkpoint_path, output_path):
    """Save weight information for Rust to use."""
    state_dict = analyze_checkpoint(checkpoint_path)

    # Extract model type from filename
    filename = Path(checkpoint_path).stem.lower()
    if "e2vid" in filename and "lightweight" in filename:
        model_type = "e2vid_unet"
    elif "firenet" in filename:
        model_type = "firenet"
    else:
        model_type = "unknown"

    print(f"\nDetected model type: {model_type}")

    # Create key mapping
    mappings = create_key_mapping(state_dict, model_type)

    # Save mapping info
    info = {
        "model_type": model_type,
        "num_parameters": len(state_dict),
        "key_mappings": mappings,
        "tensor_info": {},
    }

    # Add tensor information
    for key, tensor in state_dict.items():
        info["tensor_info"][key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
        }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved weight info to: {output_path}")


def test_tensor_conversion():
    """Test converting PyTorch tensors to numpy format."""
    # Create test tensors
    test_tensors = {
        "conv.weight": torch.randn(64, 5, 3, 3),
        "conv.bias": torch.randn(64),
        "bn.weight": torch.ones(64),
        "bn.bias": torch.zeros(64),
        "bn.running_mean": torch.zeros(64),
        "bn.running_var": torch.ones(64),
    }

    print("Testing tensor conversion:")
    for name, tensor in test_tensors.items():
        # Convert to numpy
        np_array = tensor.detach().cpu().numpy()

        # Get info
        print(f"  {name}:")
        print(f"    PyTorch shape: {list(tensor.shape)}")
        print(f"    NumPy shape: {np_array.shape}")
        print(f"    dtype: {np_array.dtype}")
        print(f"    bytes: {np_array.tobytes()[:20]}...")  # First 20 bytes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PyTorch weight loading")
    parser.add_argument("--checkpoint", type=str, help="Path to PyTorch checkpoint")
    parser.add_argument(
        "--output", type=str, default="weight_info.json", help="Output path for weight info JSON"
    )
    parser.add_argument("--test-conversion", action="store_true", help="Test tensor conversion")

    args = parser.parse_args()

    if args.test_conversion:
        test_tensor_conversion()
    elif args.checkpoint:
        save_weight_info(args.checkpoint, args.output)
    else:
        # If no checkpoint provided, test with the downloaded E2VID model
        e2vid_path = Path("models/E2VID_lightweight.pth.tar")
        if e2vid_path.exists():
            save_weight_info(e2vid_path, "e2vid_weight_info.json")
        else:
            print("No checkpoint found. Please provide --checkpoint or download E2VID model first.")
