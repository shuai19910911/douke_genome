#!/bin/sh
#SBATCH -J soy_qc_controller
#SBATCH -p q02,q03,q04,q05
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 3-00:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
TASKS=${TASKS:?TASKS is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
POLL_SECONDS=${POLL_SECONDS:-30}
MAXIMUM_ATTEMPTS=${MAXIMUM_ATTEMPTS:-3}

cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -u -B scripts/submit_split_qc_batches.py \
  --project-root "$PROJECT_ROOT" \
  --tasks "$TASKS" \
  --poll-seconds "$POLL_SECONDS" \
  --maximum-attempts "$MAXIMUM_ATTEMPTS"
