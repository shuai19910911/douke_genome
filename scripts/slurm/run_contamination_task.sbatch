#!/bin/sh
#SBATCH -J soy_contam_qc
#SBATCH -p q02,q03,q04,q05
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
TASKS=${TASKS:?TASKS is required}
CONFIG=${CONFIG:?CONFIG is required}
TIARA_IMAGE=${TIARA_IMAGE:?TIARA_IMAGE is required}
UNIVEC_DB=${UNIVEC_DB:?UNIVEC_DB is required}
QC_ENV=${QC_ENV:?QC_ENV is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
TASK_INDEX=${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:?TASK_INDEX or SLURM_ARRAY_TASK_ID is required}}
CPUS=${SLURM_CPUS_PER_TASK:-8}

cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -B scripts/run_contamination_task.py \
  --project-root "$PROJECT_ROOT" \
  --tasks "$TASKS" \
  --task-index "$TASK_INDEX" \
  --config "$CONFIG" \
  --tiara-image "$TIARA_IMAGE" \
  --univec-db "$UNIVEC_DB" \
  --qc-env "$QC_ENV" \
  --output-dir "$OUTPUT_DIR" \
  --cpus "$CPUS"
