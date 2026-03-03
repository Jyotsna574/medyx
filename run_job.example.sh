#!/bin/bash
#SBATCH --job-name=medyx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#
# Usage: cp run_job.example.sh run_job.sh; edit paths; sbatch run_job.sh

set -e
PROJECT_DIR="${PROJECT_DIR:-$HOME/ddp/medyx}"
cd "${PROJECT_DIR}"

[ -f .env ] && set -a && source .env && set +a

# Model cache base (persists across job runs)
MODELS_BASE="${MODELS_BASE:-/scratch/ed21b031/models}"

export ACTIVE_PROVIDER=local
export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-$MODELS_BASE/med42_8b}"

# MedSAM: checkpoint dir; download once, reuse on subsequent runs
export MEDSAM_CHECKPOINT_DIR="${MEDSAM_CHECKPOINT_DIR:-$MODELS_BASE/medsam_checkpoints}"

# HuggingFace (Med42-8B, etc.): cache dir; download once, reuse on subsequent runs
export HF_HOME="${HF_HOME:-$MODELS_BASE/huggingface}"

python run_mas_diagnosis.py
