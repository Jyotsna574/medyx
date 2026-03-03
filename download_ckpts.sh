#!/bin/bash
# ==============================================================================
# MedSAM Checkpoint Download Script
# ==============================================================================
# Downloads required model weights for MedSAM inference (bowang-lab/MedSAM)
#
# Usage:
#   bash download_ckpts.sh
#
# Checkpoints downloaded:
#   medsam_vit_b.pth - MedSAM ViT-B fine-tuned on medical images (~380MB)
#
# Reference:
#   Ma et al., "Segment Anything in Medical Images", Nature Communications 2024
#   https://github.com/bowang-lab/MedSAM
# ==============================================================================

set -e

# Configuration
CHECKPOINTS_DIR="${MEDSAM_CHECKPOINT_DIR:-${MEDSAM2_CHECKPOINT_DIR:-./checkpoints}}"
MEDSAM_CHECKPOINT_URL="https://huggingface.co/wanglab/medsam-vit-b/resolve/main/medsam_vit_b.pth"
MEDSAM_CHECKPOINT_NAME="medsam_vit_b.pth"

echo "=============================================="
echo "MedSAM Checkpoint Download Script"
echo "=============================================="
echo "Model: MedSAM (bowang-lab/MedSAM)"
echo "Architecture: ViT-B"
echo "Target: $CHECKPOINTS_DIR"
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

# Download MedSAM checkpoint
echo ""
echo "[1/1] Downloading MedSAM ViT-B Checkpoint..."
echo "      Fine-tuned on 1.5M+ medical image-mask pairs"
echo "      Size: ~380MB"
echo ""

MEDSAM_PATH="$CHECKPOINTS_DIR/$MEDSAM_CHECKPOINT_NAME"
if [ -f "$MEDSAM_PATH" ]; then
    echo "      MedSAM checkpoint already exists, skipping download..."
    echo "      Location: $MEDSAM_PATH"
else
    download_file "$MEDSAM_CHECKPOINT_URL" "$MEDSAM_PATH"
    echo "      Download complete: $MEDSAM_PATH"
fi

echo ""
echo "=============================================="
echo "Checkpoint Download Complete!"
echo "=============================================="
echo ""
echo "Checkpoints location: $CHECKPOINTS_DIR/"
ls -lh "$CHECKPOINTS_DIR/" 2>/dev/null || echo "(directory created)"
echo ""
echo "Installation (if not done):"
echo "  git clone https://github.com/bowang-lab/MedSAM"
echo "  cd MedSAM && pip install -e ."
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate medsam"
echo "  2. Run inference: python run_mas_diagnosis.py"
echo ""
