#!/bin/sh
#SBATCH -J lgfm_zip_genome
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 04:00:00

set -eu

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
DATA_ROOT=${DATA_ROOT:?DATA_ROOT is required}
REGISTRY=${REGISTRY:?REGISTRY is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
JOB_ID=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
case "$TASK_ID:$JOB_ID" in
  *[!0-9:]*) printf '%s\n' "invalid SLURM array identity" >&2; exit 2 ;;
esac

TEMP_BASE=${TMPDIR:-/tmp}
TEMP_WORK=${TEMP_BASE}/lgfm_archive_genome_${JOB_ID}_${TASK_ID}
cleanup() {
  rm -rf -- "$TEMP_WORK"
}
trap cleanup EXIT HUP INT TERM
mkdir -p "$TEMP_WORK" "$OUTPUT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/audit_archive_genome.py" \
  --data-root "$DATA_ROOT" \
  --registry "$REGISTRY" \
  --index "$TASK_ID" \
  --output-dir "$OUTPUT_ROOT" \
  --temporary-dir "$TEMP_WORK"
