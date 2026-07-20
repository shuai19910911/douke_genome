#!/bin/sh
#SBATCH -J soy_busco_qc
#SBATCH -p q02,q03,q04,q05
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 48:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
TASKS=${TASKS:?TASKS is required}
LINEAGE=${LINEAGE:?LINEAGE is required}
QC_ENV=${QC_ENV:?QC_ENV is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
TASK_INDEX=${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:?TASK_INDEX or SLURM_ARRAY_TASK_ID is required}}
BUSCO_MODES=${BUSCO_MODES:-both}
case "$BUSCO_MODES" in
  both) MODES=proteins,genome ;;
  proteins|genome) MODES=$BUSCO_MODES ;;
  *) printf 'invalid BUSCO_MODES: %s\n' "$BUSCO_MODES" >&2; exit 2 ;;
esac
CPUS=${SLURM_CPUS_PER_TASK:-4}

cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$QC_ENV/bin/python" -B scripts/run_busco_task.py \
  --project-root "$PROJECT_ROOT" \
  --tasks "$TASKS" \
  --task-index "$TASK_INDEX" \
  --lineage "$LINEAGE" \
  --qc-env "$QC_ENV" \
  --output-dir "$OUTPUT_DIR" \
  --modes "$MODES" \
  --cpus "$CPUS"
