#!/bin/sh
#SBATCH -J lgfm_fasta_qc
#SBATCH -p q03
#SBATCH --array=0-5%6
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 2-00:00:00

set -eu

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
DATA_ROOT=${DATA_ROOT:?DATA_ROOT is required}
SHARD_ROOT=${SHARD_ROOT:?SHARD_ROOT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

shard_id=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
shard_path=${SHARD_ROOT}/shard_${shard_id}.tsv
[ -r "$shard_path" ] || { printf 'missing shard: %s\n' "$shard_path" >&2; exit 2; }
[ -d "$DATA_ROOT" ] || { printf 'missing data root: %s\n' "$DATA_ROOT" >&2; exit 2; }
[ -x "$PYTHON_BIN" ] || { printf 'invalid Python: %s\n' "$PYTHON_BIN" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
exec "$PYTHON_BIN" -B "${PROJECT_ROOT}/scripts/audit_fasta_shard.py" \
  --data-root "$DATA_ROOT" \
  --shard "$shard_path" \
  --output-dir "$OUTPUT_ROOT"
