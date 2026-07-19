#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REGISTRY=${REGISTRY:-${PROJECT_ROOT}/data_manifests/genome_sketch_candidates.tsv}
STORE_ROOT=${STORE_ROOT:-${PROJECT_ROOT}/data/processed/sequence_store}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/data_manifests/orientation_identity_results}
PYTHON_BIN=${PYTHON_BIN:-/home/user/zhangzhishuai/.local/share/mamba/envs/douke_genomemodel/bin/python}
PARTITIONS=${PARTITIONS:-q02,q03,q04,q05}
THROTTLE=${THROTTLE:-48}
LOG_DIR=${PROJECT_ROOT}/logs/slurm
COUNT=$(($(wc -l < "$REGISTRY") - 1))
[ "$COUNT" -gt 0 ] || { echo "empty registry" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
exec sbatch --parsable --partition="$PARTITIONS" --array="0-$((COUNT - 1))%${THROTTLE}" --output="${LOG_DIR}/orientation_identity-%A_%a.out" --error="${LOG_DIR}/orientation_identity-%A_%a.err" --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},REGISTRY=${REGISTRY},STORE_ROOT=${STORE_ROOT},OUTPUT_DIR=${OUTPUT_DIR},PYTHON_BIN=${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/slurm/audit_orientation_identity.sbatch"
