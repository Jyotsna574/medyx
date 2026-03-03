#!/bin/bash
#SBATCH --job-name=medyx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#
# MedYX - Medical Diagnosis with MAS
# Usage: sbatch run_job.example.sh

set -e

# ============================================================
# PATHS - Edit these to match your cluster setup
# ============================================================

# Project code location
export PROJECT_DIR="/home/ddp/medyx"
cd "${PROJECT_DIR}"

# Load environment variables from .env if present
[ -f .env ] && set -a && source .env && set +a

# ============================================================
# MedSAM Configuration (Vision Model)
# ============================================================

# Path to MedSAM repo (contains segment_anything module)
export MEDSAM_ROOT="/home/ddp/medyx/MedSAM"

# Full path to SAM checkpoint file (no auto-download on cluster)
export MEDSAM_CHECKPOINT_PATH="/scratch/ed21b031/models/medsam_checkpoints/sam_vit_b_01ec64.pth"

# ============================================================
# Med42-8B Configuration (LLM)
# ============================================================

export ACTIVE_PROVIDER=local
export LOCAL_MODEL_PATH="/home/ddp/medyx/models/med42_8b"

# HuggingFace cache (for any additional downloads)
export HF_HOME="/scratch/ed21b031/models/huggingface"

# ============================================================
# Run
# ============================================================

echo "============================================================"
echo "MedYX Diagnosis Job Starting"
echo "============================================================"
echo "Project: ${PROJECT_DIR}"
echo "MedSAM Root: ${MEDSAM_ROOT}"
echo "SAM Checkpoint: ${MEDSAM_CHECKPOINT_PATH}"
echo "LLM Model: ${LOCAL_MODEL_PATH}"
echo "============================================================"

python run_mas_diagnosis.py
