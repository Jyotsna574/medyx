#!/bin/bash
# Run on LOGIN NODE (has internet). Do NOT run via sbatch.
# Usage: bash setup_login.sh

set -e
cd ~/ddp/medyx

pip install -r requirements_cluster.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "Done. Now: sbatch run_job.sh"
