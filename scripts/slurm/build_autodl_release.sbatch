#!/bin/sh
#SBATCH -J lgfm_autodl_build
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 08:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
DATASET_MANIFEST=${DATASET_MANIFEST:?DATASET_MANIFEST is required}
ENVIRONMENT_ARCHIVE=${ENVIRONMENT_ARCHIVE:?ENVIRONMENT_ARCHIVE is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/build_autodl_release.py" --project-root "$PROJECT_ROOT" --dataset-manifest "$DATASET_MANIFEST" --environment-archive "$ENVIRONMENT_ARCHIVE" --output "$OUTPUT"
