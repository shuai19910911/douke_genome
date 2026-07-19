#!/bin/sh
#SBATCH -J lgfm_minhash
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 1-00:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
DATA_ROOT=${DATA_ROOT:?DATA_ROOT is required}
REGISTRY=${REGISTRY:?REGISTRY is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
STORE_ROOT=${STORE_ROOT:?STORE_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/audit_genome_sketch.py" \
  --data-root "$DATA_ROOT" --registry "$REGISTRY" --index "$TASK_ID" \
  --output-dir "$OUTPUT_ROOT" --store-root "$STORE_ROOT" --ksize 31 --scaled 10000
