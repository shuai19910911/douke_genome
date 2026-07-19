#!/bin/sh
#SBATCH -J lgfm_minhash_agg
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 12:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
REGISTRY=${REGISTRY:?REGISTRY is required}
RESULT_DIR=${RESULT_DIR:?RESULT_DIR is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src"
exec "$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/aggregate_genome_similarity.py" --registry "$REGISTRY" --result-dir "$RESULT_DIR" --output-dir "$OUTPUT_DIR" --related-threshold 0.80 --near-threshold 0.95
