#!/bin/bash
# ==============================================================================
# MedSAM-2 Checkpoint Download Script
# ==============================================================================
# Downloads required model weights for MedSAM-2 inference
#
# Usage:
#   bash download_ckpts.sh
#
# Checkpoints downloaded:
#   1. SAM-2 base model (sam2.1_hiera_large.pt) - Meta's foundation model
#   2. MedSAM-2 fine-tuned weights - Medical imaging adaptation
# ==============================================================================

set -e

# Configuration
CHECKPOINTS_DIR="./checkpoints"
SAM2_MODEL_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_MODEL_NAME="sam2.1_hiera_large.pt"

# MedSAM-2 weights from Hugging Face
MEDSAM2_REPO="wanglab/MedSAM-2"
MEDSAM2_WEIGHTS="medsam2_checkpoint.pt"

echo "=============================================="
echo "MedSAM-2 Checkpoint Download Script"
echo "=============================================="

# Create checkpoints directory
mkdir -p "$CHECKPOINTS_DIR"

# Function to download with progress
download_file() {
    local url="$1"
    local output="$2"
    echo "Downloading: $(basename "$output")"
    if command -v wget &> /dev/null; then
        wget -c --show-progress -O "$output" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output" "$url"
    else
        echo "Error: Neither wget nor curl found. Please install one of them."
        exit 1
    fi
}

# Download SAM-2 base model
echo ""
echo "[1/2] Downloading SAM-2.1 Base Model (Hiera Large)..."
echo "      This is Meta's Segment Anything 2.1 foundation model"
echo "      Size: ~900MB"
echo ""

SAM2_PATH="$CHECKPOINTS_DIR/$SAM2_MODEL_NAME"
if [ -f "$SAM2_PATH" ]; then
    echo "      SAM-2 checkpoint already exists, skipping..."
else
    download_file "$SAM2_MODEL_URL" "$SAM2_PATH"
fi

# Download MedSAM-2 weights via huggingface-hub
echo ""
echo "[2/2] Downloading MedSAM-2 Fine-tuned Weights..."
echo "      Medical imaging adaptation layer"
echo ""

# Check if huggingface-hub is available
if python -c "import huggingface_hub" 2>/dev/null; then
    python -c "
from huggingface_hub import hf_hub_download
import os

checkpoint_dir = '$CHECKPOINTS_DIR'
medsam2_path = os.path.join(checkpoint_dir, '$MEDSAM2_WEIGHTS')

if os.path.exists(medsam2_path):
    print('      MedSAM-2 checkpoint already exists, skipping...')
else:
    try:
        # Try to download from Hugging Face
        downloaded_path = hf_hub_download(
            repo_id='$MEDSAM2_REPO',
            filename='$MEDSAM2_WEIGHTS',
            local_dir=checkpoint_dir,
            local_dir_use_symlinks=False
        )
        print(f'      Downloaded to: {downloaded_path}')
    except Exception as e:
        print(f'      Note: MedSAM-2 weights not available from HF ({e})')
        print('      Using SAM-2 base model - fine-tuning weights can be added later')
"
else
    echo "      Note: huggingface-hub not installed, skipping MedSAM-2 specific weights"
    echo "      The SAM-2 base model will be used for inference"
fi

echo ""
echo "=============================================="
echo "Checkpoint Download Complete!"
echo "=============================================="
echo ""
echo "Checkpoints location: $CHECKPOINTS_DIR/"
ls -lh "$CHECKPOINTS_DIR/" 2>/dev/null || echo "(directory created, awaiting downloads)"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate medsam2"
echo "  2. Run inference test: python -m tests.test_vision_engine"
echo ""
