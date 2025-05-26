#!/usr/bin/env python3
"""
Analyze the E2VID PyTorch model structure to understand what we need to implement in Candle
"""
import torch


def analyze_e2vid_structure():
    """Analyze the exact structure of the E2VID model"""

    model_path = "models/E2VID_lightweight.pth.tar"

    print("🔍 Analyzing E2VID PyTorch Model Structure")
    print("=" * 50)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    print(f"📊 Total parameters: {len(state_dict)}")
    print(f"🏗️ Architecture: {checkpoint.get('arch', 'Unknown')}")
    print(f"📝 Model info: {checkpoint.get('model', {})}")

    # Group parameters by component
    components = {}
    for key in state_dict.keys():
        parts = key.split(".")
        component = parts[1] if len(parts) > 1 else "root"  # Skip 'unetrecurrent' prefix

        if component not in components:
            components[component] = []
        components[component].append(key)

    print("\n🏗️ Model Architecture Analysis:")
    print("-" * 30)

    for component, params in components.items():
        print(f"\n📁 {component.upper()} ({len(params)} parameters):")

        # Group by layer index if applicable
        if any(any(char.isdigit() for char in param) for param in params if "." in param):
            # Has numbered layers
            layers = {}
            for param in params:
                parts = param.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    layer_idx = parts[2]
                    if layer_idx not in layers:
                        layers[layer_idx] = []
                    layers[layer_idx].append(param)
                else:
                    if "other" not in layers:
                        layers["other"] = []
                    layers["other"].append(param)

            for layer_idx in sorted(layers.keys()):
                layer_params = layers[layer_idx]
                print(f"  📋 Layer {layer_idx}: {len(layer_params)} params")
                for param in layer_params[:3]:  # Show first 3
                    tensor = state_dict[param]
                    print(f"    - {param}: {list(tensor.shape)}")
                if len(layer_params) > 3:
                    print(f"    ... and {len(layer_params) - 3} more")
        else:
            # No numbered layers
            for param in params[:5]:  # Show first 5
                tensor = state_dict[param]
                print(f"  - {param}: {list(tensor.shape)}")
            if len(params) > 5:
                print(f"  ... and {len(params) - 5} more")

    # Analyze specific components in detail
    print("\n🔬 Detailed Component Analysis:")
    print("-" * 30)

    # Head
    head_params = [k for k in state_dict.keys() if "head" in k]
    if head_params:
        print("\n🎯 HEAD (Input processing):")
        for param in head_params:
            tensor = state_dict[param]
            print(f"  {param}: {list(tensor.shape)}")

    # Encoders
    encoder_params = [k for k in state_dict.keys() if "encoders" in k]
    if encoder_params:
        print("\n⬇️  ENCODERS (Downsampling):")
        encoder_layers = {}
        for param in encoder_params:
            layer_num = param.split(".")[2]
            if layer_num not in encoder_layers:
                encoder_layers[layer_num] = []
            encoder_layers[layer_num].append(param)

        for layer_num in sorted(encoder_layers.keys()):
            layer_params = encoder_layers[layer_num]
            print(f"  Layer {layer_num}:")
            for param in layer_params:
                tensor = state_dict[param]
                print(f"    {param}: {list(tensor.shape)}")

    # Decoders
    decoder_params = [k for k in state_dict.keys() if "decoders" in k]
    if decoder_params:
        print("\n⬆️  DECODERS (Upsampling):")
        decoder_layers = {}
        for param in decoder_params:
            layer_num = param.split(".")[2]
            if layer_num not in decoder_layers:
                decoder_layers[layer_num] = []
            decoder_layers[layer_num].append(param)

        for layer_num in sorted(decoder_layers.keys()):
            layer_params = decoder_layers[layer_num]
            print(f"  Layer {layer_num}:")
            for param in layer_params:
                tensor = state_dict[param]
                print(f"    {param}: {list(tensor.shape)}")

    # ResBlocks
    resblock_params = [k for k in state_dict.keys() if "resblocks" in k]
    if resblock_params:
        print("\n🔄 RESBLOCKS (Middle processing):")
        resblock_layers = {}
        for param in resblock_params:
            layer_num = param.split(".")[2]
            if layer_num not in resblock_layers:
                resblock_layers[layer_num] = []
            resblock_layers[layer_num].append(param)

        for layer_num in sorted(resblock_layers.keys()):
            layer_params = resblock_layers[layer_num]
            print(f"  ResBlock {layer_num}:")
            for param in layer_params:
                tensor = state_dict[param]
                print(f"    {param}: {list(tensor.shape)}")

    # Prediction/Output
    pred_params = [k for k in state_dict.keys() if "pred" in k]
    if pred_params:
        print("\n🎯 PREDICTION (Output):")
        for param in pred_params:
            tensor = state_dict[param]
            print(f"  {param}: {list(tensor.shape)}")

    # Model hyperparameters
    model_info = checkpoint.get("model", {})
    if model_info:
        print("\n⚙️  MODEL HYPERPARAMETERS:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    analyze_e2vid_structure()
