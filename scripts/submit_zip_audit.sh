#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/raw}
INVENTORY=${INVENTORY:-${PROJECT_ROOT}/data_manifests/raw_inventory.tsv}
RESULT_ROOT=${RESULT_ROOT:-${PROJECT_ROOT}/data_manifests/archive_qc_results}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/data_manifests}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3)}
PARTITIONS=q02,q03,q04,q05
mkdir -p "$PROJECT_ROOT/logs/slurm" "$RESULT_ROOT" "$OUTPUT_ROOT"

JOB_ID=$(sbatch --parsable \
  --partition="$PARTITIONS" \
  --chdir="$PROJECT_ROOT" \
  --output="$PROJECT_ROOT/logs/slurm/zip_audit-%j.out" \
  --error="$PROJECT_ROOT/logs/slurm/zip_audit-%j.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},INVENTORY=${INVENTORY},RESULT_ROOT=${RESULT_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},PYTHON_BIN=${PYTHON_BIN}" \
  "$PROJECT_ROOT/scripts/slurm/audit_zip_archives.sbatch")
printf '%s\n' "$JOB_ID"
