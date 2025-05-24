#!/bin/bash
# Download pre-trained event-to-video reconstruction models

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

echo "=== Downloading Pre-trained Models ==="

# E2VID Lightweight (Official from UZH-RPG)
echo -e "\n1. Downloading E2VID Lightweight..."
wget -nc "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" \
     -O "$MODELS_DIR/E2VID_lightweight.pth.tar"

# Calculate checksum
if [ -f "$MODELS_DIR/E2VID_lightweight.pth.tar" ]; then
    echo "SHA256: $(sha256sum "$MODELS_DIR/E2VID_lightweight.pth.tar" | cut -d' ' -f1)"
fi

# SPADE-E2VID models would need to be downloaded from the repository
echo -e "\n2. SPADE-E2VID models:"
echo "Please clone https://github.com/RodrigoGantier/SPADE_E2VID"
echo "Models available:"
echo "  - E2VID.pth.tar"
echo "  - E2VID_lightweight.pth.tar"
echo "  - SPADE_E2VID.pth"
echo "  - SPADE_E2VID_2.pth"
echo "  - SPADE_E2VID_ABS.pth"

# FireNet
echo -e "\n3. FireNet model:"
echo "Please clone https://github.com/cedric-scheerlinck/rpg_e2vid (cedric/firenet branch)"
echo "Model checkpoint: firenet_1000.pth.tar"

# SSL-E2VID
echo -e "\n4. SSL-E2VID models:"
echo "Please check https://github.com/tudelft/ssl_e2vid"
echo "for available pre-trained models"

# ET-Net
echo -e "\n5. ET-Net models:"
echo "Repository: https://github.com/WarranWeng/ET-Net"
echo "Models to be released according to the paper"

echo -e "\n=== Model Conversion ==="
echo "To convert PyTorch models to ONNX format, run:"
echo "python scripts/convert_pytorch_to_onnx.py --download-pytorch"
