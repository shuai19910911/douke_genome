#!/bin/sh
#SBATCH -J lgfm_rc_exact
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 08:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
REGISTRY=${REGISTRY:?REGISTRY is required}
STORE_ROOT=${STORE_ROOT:?STORE_ROOT is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
mkdir -p "$OUTPUT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src"
exec "$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/audit_orientation_identity.py" --registry "$REGISTRY" --index "$TASK_ID" --store-root "$STORE_ROOT" --output-dir "$OUTPUT_DIR"
