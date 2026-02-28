#!/bin/bash
#SBATCH --job-name=medyx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#
# Param Shakti - Run MAS Diagnostic Pipeline with local Med42 model
# Usage: sbatch run_job.sh

set -e

# Project directory (adjust if needed)
PROJECT_DIR="${HOME}/ddp/medyx"
cd "${PROJECT_DIR}"

# Load env vars from .env if present (Neo4j creds, etc.)
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
fi

# Local model settings
export ACTIVE_PROVIDER=local
export LOCAL_MODEL_PATH="${HOME}/ddp/medyx/models/med42_8b"

# Neo4j: MUST be set in .env for cluster runs
# Jobs run on GPU nodes; Neo4j runs on login node.
# Set NEO4J_URI=bolt://login01:7687 (or your login hostname) in .env
export NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"
if [ -z "${NEO4J_URI}" ]; then
    echo "ERROR: NEO4J_URI not set. Create .env with NEO4J_URI=bolt://<login-node>:7687"
    exit 1
fi

echo "=============================================="
echo "MAD MAS Diagnosis - Param Shakti"
echo "=============================================="
echo "ACTIVE_PROVIDER=${ACTIVE_PROVIDER}"
echo "LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH}"
echo "NEO4J_URI=${NEO4J_URI}"
echo "=============================================="

python run_mas_diagnosis.py
