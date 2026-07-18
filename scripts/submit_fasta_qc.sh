#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/raw}
SHARD_ROOT=${SHARD_ROOT:-${PROJECT_ROOT}/data_manifests/assembly_qc_shards}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/data_manifests/fasta_qc_results}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3)}
PARTITION=${PARTITION:-q03}

[ -d "$DATA_ROOT" ] || { printf 'missing data root: %s\n' "$DATA_ROOT" >&2; exit 2; }
[ -x "$PYTHON_BIN" ] || { printf 'invalid Python: %s\n' "$PYTHON_BIN" >&2; exit 2; }
for shard in 00 01 02 03 04 05; do
  [ -r "${SHARD_ROOT}/shard_${shard}.tsv" ] || { printf 'missing shard: %s\n' "${SHARD_ROOT}/shard_${shard}.tsv" >&2; exit 2; }
done
mkdir -p "${PROJECT_ROOT}/logs/slurm" "$OUTPUT_ROOT"

job_id=$(sbatch --parsable \
  --partition="$PARTITION" \
  --chdir="$PROJECT_ROOT" \
  --output="${PROJECT_ROOT}/logs/slurm/fasta_qc-%A_%a.out" \
  --error="${PROJECT_ROOT}/logs/slurm/fasta_qc-%A_%a.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},SHARD_ROOT=${SHARD_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},PYTHON_BIN=${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/slurm/audit_fasta_shards.sbatch")
printf '%s\n' "$job_id"
