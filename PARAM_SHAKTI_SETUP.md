# Param Shakti (HPC Cluster) Setup Guide

This guide covers running the MedicalAgentDiagnosis MAS pipeline on Param Shakti with a local HuggingFace model (e.g. Med42-8B).

## Prerequisites

- GPU node access (2x GPU recommended for 70B models; 1x GPU for 7B/8B)
- Pre-downloaded model in your home directory

## 1. Clone / Sync Code

```bash
cd ~/ddp/medyx   # or your project path
# If git available:
git pull origin master
# Otherwise: copy files from your local machine via scp
```

## 2. Install Dependencies

```bash
# Create/activate conda env (if using)
conda create -n medyx python=3.10 -y
conda activate medyx

# Install requirements (use cluster file - no streamlit, no protobuf conflicts)
pip install -r requirements_cluster.txt

# Install PyTorch with CUDA (match your cluster's CUDA version)
# For CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.x:
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 3. Neo4j (Required)

Neo4j is required. The pipeline does not use fallback guidelines.

- Install: Ensure `neo4j>=5.0.0` is in `requirements_cluster.txt` (uncomment if needed).
- Configure: Set `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` in your environment or `.env`.
- On cluster: Jobs run on GPU nodes; Neo4j usually runs on the login node. Use `NEO4J_URI=bolt://login01:7687` (or your login node hostname) so the job can reach Neo4j.

## 4. Download Model (One-Time)

```bash
# Using HuggingFace CLI
huggingface-cli download m42-health/med42-v2-8B --local-dir ~/models/med42_8b

# Or download manually and extract to ~/models/med42_8b
```

## 5. Download MedSAM-2 Checkpoints

Place checkpoints in `./checkpoints/` (relative to where you run the script):

- `medsam2_tiny.pt` or `medsam2_huge.pt` - download from the MedSAM-2 release.

## 6. Run

```bash
# Request GPU node (example)
sbatch run_job.sh

# Or interactive:
srun --partition=gpu --gres=gpu:1 --mem=32G --pty bash

# Set env vars and run
export ACTIVE_PROVIDER=local
export LOCAL_MODEL_PATH=/home/$USER/ddp/medyx/models/med42_8b

python run_mas_diagnosis.py
```

## 7. Example Slurm Script (`run_job.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=medyx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

cd /home/$USER/ddp/medyx

export ACTIVE_PROVIDER=local
export LOCAL_MODEL_PATH=/home/$USER/ddp/medyx/models/med42_8b

python run_mas_diagnosis.py
```

## Package Summary (What Gets Used)

| Package       | Purpose                          | Required |
|---------------|----------------------------------|----------|
| pydantic      | Config, schemas                  | Yes      |
| PyYAML        | Config files                     | Yes      |
| camel-ai      | Multi-agent framework            | Yes      |
| transformers  | HuggingFace LLM                  | Yes      |
| bitsandbytes  | 4-bit quantization               | Yes      |
| numpy         | Vision (MedSAM-2)                | Yes      |
| Pillow        | Image loading                    | Yes      |
| scikit-image  | Vision processing                | Yes      |
| neo4j         | Knowledge graph                 | Yes      |
| streamlit     | Web UI (not used on cluster)     | No       |

## Troubleshooting

- **`ModuleNotFoundError: neo4j`** or **`Neo4jConnectionError`**: Install neo4j (`pip install neo4j`) and ensure Neo4j is running. Set `NEO4J_URI` to the correct host (e.g. `bolt://login01:7687` for cluster).
- **`ModuleNotFoundError: numpy`**: Run `pip install -r requirements_cluster.txt` again.
- **CUDA out of memory**: Use 4-bit quantization (default) or a smaller model (med42_8b instead of 70B).
- **Test image missing**: Ensure `test_chest_xray.png` exists in the working directory or pass a valid image path.
