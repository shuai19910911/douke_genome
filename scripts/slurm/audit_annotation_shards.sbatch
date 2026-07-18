#!/bin/sh
#SBATCH -J lgfm_annotation_qc
#SBATCH -p q04
#SBATCH --array=0-3%4
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 1-00:00:00

set -eu

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
DATA_ROOT=${DATA_ROOT:?DATA_ROOT is required}
SHARD_ROOT=${SHARD_ROOT:?SHARD_ROOT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}

case ${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} in
    0|1|2|3) ;;
    *) echo "invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

SHARD_ID=$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")
SHARD_PATH=${SHARD_ROOT}/shard_${SHARD_ID}.tsv
[ -f "${SHARD_PATH}" ] || { echo "missing shard: ${SHARD_PATH}" >&2; exit 2; }
[ -d "${DATA_ROOT}" ] || { echo "missing data root: ${DATA_ROOT}" >&2; exit 2; }

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${PROJECT_ROOT}/src

exec "${PYTHON_BIN}" -B "${PROJECT_ROOT}/scripts/audit_annotation_shard.py" \
    --data-root "${DATA_ROOT}" \
    --shard "${SHARD_PATH}" \
    --output-dir "${OUTPUT_ROOT}"
