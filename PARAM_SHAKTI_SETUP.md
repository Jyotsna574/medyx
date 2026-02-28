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

## 3. Neo4j (Required – Cluster Setup)

Neo4j is required. The pipeline does not use fallback guidelines.

**Architecture**: Jobs run on GPU nodes; Neo4j runs on the login node. The GPU job must connect to Neo4j on the login node.

### 3a. Run Neo4j on the Login Node

On Param Shakti (or your cluster), Neo4j must run on a node that GPU nodes can reach—typically the **login node**.

```bash
# SSH to cluster, stay on login node
ssh user@login01    # use your cluster's login hostname

# Option A: Neo4j Community via module/package (if available)
module load neo4j    # if your cluster provides it
neo4j start

# Option B: Standalone Neo4j tarball
# Download from neo4j.com, extract, run:
# ./bin/neo4j start
```

Ensure Neo4j listens on the network (not just localhost): in `conf/neo4j.conf`, set:
```properties
server.default_listen_address=0.0.0.0
```

### 3b. Create .env on the Cluster

In your project directory **on the cluster**, create `.env`:

```bash
cd ~/ddp/medyx   # or your project path

cat > .env << 'EOF'
# Replace login01 with your cluster's login node hostname
NEO4J_URI=bolt://login01:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
EOF
```

**Important**: Use the login node **hostname** (e.g. `login01`, `login`, `frontend01`), not `127.0.0.1`. On a GPU node, `127.0.0.1` is the GPU node itself, which does not run Neo4j.

### 3c. Load the Required Schema

The retriever expects `Entity` nodes (`type`: disease, drug, gene/protein, effect/phenotype) and `RELATED` relationships. In Neo4j Browser (`http://<login-node>:7474`) or `cypher-shell`, run:

```cypher
CREATE (g:Entity {name: "Glaucoma", type: "disease", source: "ICD-10"});
CREATE (p:Entity {name: "Pneumonia", type: "disease", source: "ICD-10"});
CREATE (t1:Entity {name: "Timolol", type: "drug"});
CREATE (a:Entity {name: "Amoxicillin", type: "drug"});
MATCH (g:Entity {name: "Glaucoma"}), (t1:Entity {name: "Timolol"}) CREATE (g)-[:RELATED]->(t1);
MATCH (p:Entity {name: "Pneumonia"}), (a:Entity {name: "Amoxicillin"}) CREATE (p)-[:RELATED]->(a);
```

(Add more entities as needed for your use case.)

### 3d. Verify

```bash
# From login node
python -c "
from infrastructure.rag.neo4j_retriever import Neo4jKnowledgeRetriever
import os
os.environ.setdefault('NEO4J_URI', 'bolt://localhost:7687')  # if Neo4j on same node
r = Neo4jKnowledgeRetriever()
r.connect()
print('Neo4j OK')
r.close()
"
```

When submitting via `sbatch`, the job loads `.env` automatically (see `run_job.sh`), so `NEO4J_URI` will point to the login node.

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
