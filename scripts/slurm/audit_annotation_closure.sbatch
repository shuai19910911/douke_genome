#!/bin/sh
#SBATCH -J soy_gff_closure
#SBATCH -p q02,q03,q04,q05
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 04:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
CANDIDATES=${CANDIDATES:?CANDIDATES is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
WORKERS=${SLURM_CPUS_PER_TASK:-16}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/audit_annotation_closure.py" \
  --project-root "$PROJECT_ROOT" \
  --candidates "$CANDIDATES" \
  --output "$OUTPUT" \
  --workers "$WORKERS"
