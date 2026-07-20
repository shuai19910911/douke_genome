#!/bin/sh
#SBATCH -J soy_data_refine
#SBATCH -p q02,q03,q04,q05
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
CONFIG=${CONFIG:?CONFIG is required}
OUTPUT=${OUTPUT:?OUTPUT is required}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/audit_data_refinement.py" \
  --project-root "$PROJECT_ROOT" \
  --config "$CONFIG" \
  --output "$OUTPUT"
