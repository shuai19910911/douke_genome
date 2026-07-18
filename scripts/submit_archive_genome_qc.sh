#!/bin/sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/raw}
REGISTRY=${REGISTRY:-${PROJECT_ROOT}/data_manifests/archive_genome_candidates.tsv}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/data_manifests/archive_genome_qc_results}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3)}
PARTITIONS=${PARTITIONS:-q02,q03,q04,q05}
THROTTLE=${THROTTLE:-34}

COUNT=$(wc -l < "$REGISTRY")
COUNT=$((COUNT - 1))
[ "$COUNT" -gt 0 ] || { printf '%s\n' "empty archive genome registry" >&2; exit 2; }
case "$THROTTLE" in ''|*[!0-9]*) printf '%s\n' "invalid THROTTLE" >&2; exit 2 ;; esac
[ "$THROTTLE" -gt 0 ] || { printf '%s\n' "THROTTLE must be positive" >&2; exit 2; }
[ "$THROTTLE" -le "$COUNT" ] || THROTTLE=$COUNT
LAST=$((COUNT - 1))

mkdir -p "$PROJECT_ROOT/logs/slurm" "$OUTPUT_ROOT"
JOB_ID=$(sbatch --parsable \
  --partition="$PARTITIONS" \
  --array="0-${LAST}%${THROTTLE}" \
  --chdir="$PROJECT_ROOT" \
  --output="$PROJECT_ROOT/logs/slurm/archive_genome_qc-%A_%a.out" \
  --error="$PROJECT_ROOT/logs/slurm/archive_genome_qc-%A_%a.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATA_ROOT=${DATA_ROOT},REGISTRY=${REGISTRY},OUTPUT_ROOT=${OUTPUT_ROOT},PYTHON_BIN=${PYTHON_BIN}" \
  "$PROJECT_ROOT/scripts/slurm/audit_archive_genomes.sbatch")
printf '%s\n' "$JOB_ID"
