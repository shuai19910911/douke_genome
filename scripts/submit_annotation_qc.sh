#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/raw}
SHARD_ROOT=${SHARD_ROOT:-${PROJECT_ROOT}/data_manifests/annotation_qc_shards}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/data_manifests/annotation_qc_results}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3)}
PARTITIONS=q02,q03,q04,q05

for shard in 00 01 02 03; do
    [ -f "${SHARD_ROOT}/shard_${shard}.tsv" ] || {
        echo "missing shard: ${SHARD_ROOT}/shard_${shard}.tsv" >&2
        exit 2
    }
done
mkdir -p "${PROJECT_ROOT}/logs/slurm" "${OUTPUT_ROOT}"

exec sbatch --parsable --partition="${PARTITIONS}" --chdir="${PROJECT_ROOT}" \
    --output="${PROJECT_ROOT}/logs/slurm/annotation_qc-%A_%a.out" \
    --error="${PROJECT_ROOT}/logs/slurm/annotation_qc-%A_%a.err" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},SHARD_ROOT=${SHARD_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},PYTHON_BIN=${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/slurm/audit_annotation_shards.sbatch"
