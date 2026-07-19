#!/bin/sh
#SBATCH -J lgfm_autodl_verify
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 08:00:00

set -eu
PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
RELEASE_ROOT=${RELEASE_ROOT:?RELEASE_ROOT is required}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/verify_autodl_release.py" --release-root "$RELEASE_ROOT"
exec "$PYTHON_BIN" -B "$PROJECT_ROOT/scripts/verify_autodl_release.py" --release-root "$RELEASE_ROOT" --deep
