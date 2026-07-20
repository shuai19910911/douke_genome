#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/raw}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/data_manifests}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3)}
PARTITIONS=q02,q03,q04,q05
LOG_DIR=${PROJECT_ROOT}/logs/slurm

[ -d "${DATA_ROOT}" ]
[ -x "${PYTHON_BIN}" ]
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

JOB_ID=$(sbatch --parsable \
    --partition="${PARTITIONS}" \
    --chdir="${PROJECT_ROOT}" \
    --output="${LOG_DIR}/raw_inventory-%j.out" \
    --error="${LOG_DIR}/raw_inventory-%j.err" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},PYTHON_BIN=${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/slurm/inventory_raw_data.sbatch")
printf '%s\n' "${JOB_ID}"
