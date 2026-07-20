#!/bin/sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/raw}
REGISTRY=${REGISTRY:-${PROJECT_ROOT}/data_manifests/genome_sketch_candidates.tsv}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/data_manifests/genome_sketch_results}
STORE_ROOT=${STORE_ROOT:-${PROJECT_ROOT}/data/processed/sequence_store}
PYTHON_BIN=${PYTHON_BIN:-python3}
PARTITIONS=q02,q03,q04,q05
THROTTLE=${THROTTLE:-48}
COUNT=$(($(wc -l < "$REGISTRY") - 1))
if [ "$COUNT" -le 0 ]; then echo "empty genome sketch registry" >&2; exit 2; fi
LAST=$((COUNT - 1))
if [ "$THROTTLE" -gt "$COUNT" ]; then THROTTLE=$COUNT; fi
mkdir -p "$OUTPUT_ROOT" "$STORE_ROOT" "$PROJECT_ROOT/logs/slurm"
sbatch --parsable --partition="$PARTITIONS" --array="0-${LAST}%${THROTTLE}" \
  --output="$PROJECT_ROOT/logs/slurm/genome-sketch-%A_%a.out" \
  --error="$PROJECT_ROOT/logs/slurm/genome-sketch-%A_%a.err" \
  --export="PROJECT_ROOT=$PROJECT_ROOT,DATA_ROOT=$DATA_ROOT,REGISTRY=$REGISTRY,OUTPUT_ROOT=$OUTPUT_ROOT,STORE_ROOT=$STORE_ROOT,PYTHON_BIN=$PYTHON_BIN" \
  "$PROJECT_ROOT/scripts/slurm/audit_genome_sketches.sbatch"
